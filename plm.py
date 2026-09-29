# %% [markdown]
# # PLM v2 — Perceptive Language Model for neuromorphic drone control
#
# Vision + LiDAR/4D-radar + IMU + language  ->  BEV grounding  ->  LIF spiking attention + closed-form LTC core
# ->  CNN action-chunk head, trained with a 4-stage curriculum:
#
#   Stage 1  Modality pre-alignment in a shared Bird's-Eye-View grid (cell-level contrastive + occupancy + Doppler)
#   Stage 2  Self-supervised continuous-time world model (multi-horizon latent JEPA + spike-rate regularisation, TBPTT)
#   Stage 3  Language-conditioned behaviour cloning of H-step action chunks (discounted trajectory loss + jerk penalty)
#   Stage 4  Speculative co-training: edge SNN drafter distilled against the full target (KL + safety hinge),
#            hard-rejection replay from closed-loop rollouts, optional GRPO fine-tuning of the target.
#
# Runtime: asynchronous speculative control (drafter on the critical path, batched target verification off it),
# GPU-resident MuscleMemoryCache of verified action chunks, and a control-barrier-function (CBF) safety filter.
#
# Run everything:   python plm.py --mode full      (GPU; ~1 h on a T4 / ~20 min on an A100)
# Smoke test:       python plm.py --mode quick     (CPU, a few minutes)

# %%
from __future__ import annotations

import argparse
import contextlib
import copy
import json
import math
import os
import random
import shutil
import time
from collections import deque
from dataclasses import asdict, dataclass, fields

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

try:
    import timm
except ImportError:  # the model still runs with a small built-in patch encoder
    timm = None


# %% [markdown]
# ## 1. Configuration

# %%
@dataclass
class Config:
    out_dir: str = "runs/plm"
    seed: int = 0
    device: str = ("cuda" if torch.cuda.is_available() else
                   "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
    num_workers: int = 2

    # ---- sensors / simulator ----
    img_size: int = 224
    vit_name: str = "vit_tiny_patch16_224"
    pretrained: bool = True
    lora_rank: int = 4               # LoRA adapters on the frozen ViT (0 = fully frozen)
    lora_blocks: int = 2             # adapt the last N transformer blocks
    lidar_zyx: tuple = (8, 32, 32)   # voxel grid (height, lateral, forward)
    lidar_extent: tuple = ((-4.0, 4.0), (-12.0, 12.0), (-6.0, 18.0))  # metres, ego frame (z, y, x)
    bev_grid: int = 8                # G x G BEV cells
    n_obstacles: int = 6
    dt_range: tuple = (0.03, 0.08)   # non-uniform control intervals (s)
    vmax: float = 3.0                # m/s, action normalisation
    rmax: float = 1.0                # rad/s, yaw-rate normalisation
    dart_noise: float = 0.15         # DART-style execution noise for BC data coverage
    warmup_max: int = 120            # expert steps before recording: covers the whole flight incl. near-goal states
    dagger_rounds: int = 2           # Stage 3b: on-policy states labelled by the expert (0 = off)
    dagger_episodes: int = 40
    dagger_beta: float = 0.3         # probability of executing the expert action during collection
    dagger_steps: int = 2000
    proprio_dropout: float = 0.0     # Stage 3: optionally zero velocity/accel inputs per sample
    policy_proprio: str = "no_motion"  # "no_motion": policy/drafter do not see ego velocity/accel/yaw-rate (anti-copycat)
                                       # "full": all 9 IMU channels (ablation: exhibits copycat failure)

    # ---- model ----
    d_bev: int = 64
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 3
    ltc_hidden: int = 256
    n_pool_queries: int = 4
    spike_v: bool = True             # spike V as well as Q, K
    lif_tau_init: float = 0.1        # s
    ltc_tau_init: float = 0.5        # s
    surrogate_k: float = 5.0         # fast-sigmoid surrogate slope
    spectral_max: float = 0.99       # eigen/spectral clamp on LTC recurrent matrix
    text_backbone: str = "simple"    # "simple" | "clip" (frozen HF CLIP text tower)
    spiking: bool = True             # ablation: False = real-valued attention (no LIF)
    spike_inputs: bool = False       # spike-driven projections: LIF on the inputs of W_qkv and the LTC's W_x,
                                     # so those matmuls become accumulates (the dominant energy term of the core)
    core_ff: str = "ltc"             # ablation: "ltc" | "mlp" (no continuous-time recurrence)
    skip_stage2: bool = False        # ablation: no self-supervised dynamics pre-training
    perception_ckpt: str = ""        # reuse a trained Stage-1 perception (skips Stage 1)
    cache_dir: str = ""              # where feature caches live (default: out_dir)
    stop_after_stage3: bool = False  # ablation runs: skip Stage 4 / final evaluation
    max_cmd_len: int = 8
    dropout: float = 0.1

    # ---- horizons ----
    T: int = 8                       # policy context window
    H: int = 10                      # action-chunk horizon
    jepa_horizons: tuple = (1, 2, 4)
    T_long: int = 32                 # world-model sequence length
    tbptt: int = 16                  # truncated BPTT chunk

    # ---- drafter ----
    d_draft: int = 96

    # ---- budgets ----
    s1_steps: int = 3000
    s1_batch: int = 32
    s1_lr: float = 3e-4
    n_cache_episodes: int = 1000
    n_val_episodes: int = 100
    s2_steps: int = 3000
    s2_batch: int = 32
    s2_lr: float = 3e-4
    s3_steps: int = 6000
    s3_batch: int = 64
    s3_lr: float = 3e-4
    s4_distill_steps: int = 2000
    s4_batch: int = 64
    s4_lr: float = 3e-4
    s4_rounds: int = 3
    s4_rollouts_per_round: int = 8
    s4_replay_steps: int = 300
    grpo_iters: int = 20
    grpo_group: int = 8
    grpo_lr: float = 1e-5
    episode_ticks: int = 150
    eval_episodes: int = 20
    log_every: int = 100

    # ---- loss weights ----
    tau_contra: float = 0.07
    lam_occ: float = 1.0
    lam_vel: float = 1.0
    lam_sem: float = 1.0
    lam_psem: float = 1.0
    cam_grid: int = 7                # camera tokens pooled to at most cam_grid x cam_grid for the policy
    lam_spike: float = 1.0
    r_target: float = 0.08           # target firing rate (5-10 %)
    gamma: float = 0.9               # trajectory-horizon discount
    lam_theta: float = 1.0           # yaw-rate weight
    lam_v: float = 1.0               # velocity weight
    lam_jerk: float = 1.0
    lam_jepa_aux: float = 0.2
    lam_nll: float = 0.1
    beta_safe: float = 1.0
    rho_safe: float = 0.15
    grpo_kl: float = 0.05
    grad_clip: float = 1.0

    # ---- speculative runtime ----
    accept_tol: float = 0.10         # |draft - target| <= tol + z * sigma_target, every action dim
    accept_z: float = 0.5
    verify_every: int = 4            # max verification batch (adaptive, halves to 1 after a rejection)
    cache_capacity: int = 2048
    cache_sim: float = 0.985
    cache_max_sigma: float = 0.15
    use_cbf: bool = True

    @classmethod
    def quick(cls, **kw):
        c = cls(img_size=64, pretrained=False, lidar_zyx=(8, 16, 16), bev_grid=4, d_bev=32, d_model=64,
                n_heads=2, n_layers=2, ltc_hidden=64, d_draft=32, T=6, H=6, jepa_horizons=(1, 2), T_long=12,
                tbptt=6, s1_steps=60, s1_batch=8, n_cache_episodes=48, n_val_episodes=12, s2_steps=60,
                s2_batch=8, s3_steps=120, s3_batch=16, s4_distill_steps=60, s4_batch=16, s4_rounds=2,
                s4_rollouts_per_round=2, s4_replay_steps=20, grpo_iters=2, grpo_group=4, episode_ticks=90,
                dagger_rounds=1, dagger_episodes=2, dagger_steps=20,
                eval_episodes=3, log_every=20, num_workers=0, out_dir="runs/plm_quick")
        for k, v in kw.items():
            setattr(c, k, v)
        return c

    @classmethod
    def small(cls, **kw):
        """CPU-scale research configuration (used for the numbers in the paper draft)."""
        c = cls(img_size=96, lidar_zyx=(8, 24, 24), bev_grid=6, d_bev=48, d_model=128, n_heads=4, n_layers=2,
                ltc_hidden=128, d_draft=48, T=8, H=8, jepa_horizons=(1, 2, 4), T_long=16, tbptt=8,
                s1_steps=1500, s1_batch=16, n_cache_episodes=400, n_val_episodes=60, s2_steps=800, s2_batch=16,
                s3_steps=2500, s3_batch=32, s4_distill_steps=600, s4_batch=32, s4_rounds=3,
                s4_rollouts_per_round=6, s4_replay_steps=150, grpo_iters=0, episode_ticks=150, eval_episodes=30,
                warmup_max=30, dagger_rounds=0,
                log_every=100, num_workers=2, out_dir="runs/plm_small")
        for k, v in kw.items():
            setattr(c, k, v)
        return c

    @property
    def n_cells(self):
        return self.bev_grid ** 2

    @property
    def n_scene(self):
        return self.n_cells + min(self.img_size // 16, self.cam_grid) ** 2

    @property
    def episode_len(self):
        return max(self.T_long + max(self.jepa_horizons), self.T + max(self.H - 1, max(self.jepa_horizons))) + 1


SENSOR_DIM = 9     # body vel (3), yaw rate, body accel (3), altitude, dt
ACTION_DIM = 4     # forward vel, left vel, up vel, yaw rate  (normalised to [-1, 1])
ACTION_NAMES = ["v_forward", "v_left", "v_up", "yaw_rate"]


def proprio_mask(cfg):
    """Channels of the 9-D IMU the *policy* may see. Ego velocity/acceleration/yaw-rate explain ~97% of the variance
    of expert velocity commands (the drone tracks its last command), so a policy that sees them learns to copy its
    own motion and never leaves a hover. The world model's predictor still receives the full IMU as ω_t."""
    m = torch.ones(SENSOR_DIM)
    if cfg.policy_proprio == "no_motion":
        m[:7] = 0.0
    return m


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


def amp_dtype(device):
    if str(device).startswith("cuda") and torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability()
        return torch.bfloat16 if major >= 8 else torch.float16
    return None


def autocast(device):
    """BF16 (Ampere+) or FP16 autocast for projections / ViT. ODE + membrane updates opt out (see fp32_region)."""
    dt = amp_dtype(device)
    return torch.autocast("cuda", dtype=dt) if dt is not None else contextlib.nullcontext()


def fp32_region(x):
    return torch.autocast(device_type=x.device.type, enabled=False) if x.is_cuda else contextlib.nullcontext()


# %% [markdown]
# ## 2. Language commands

# %%
INTENT_PHRASES = [
    ["fly to the red beacon", "go to the red beacon", "navigate to the red beacon"],
    ["fly to the blue beacon", "go to the blue beacon", "navigate to the blue beacon"],
    ["hover in place", "hold position", "stop and hover"],
    ["ascend slowly", "climb slowly", "go up"],
    ["descend slowly", "go down", "lower altitude"],
    ["turn left", "rotate left", "yaw left"],
    ["turn right", "rotate right", "yaw right"],
]
GOAL_INTENTS = (0, 1)
VOCAB = ["<pad>", "<unk>"] + sorted({w for ps in INTENT_PHRASES for p in ps for w in p.split()})
VOCAB_IDX = {w: i for i, w in enumerate(VOCAB)}


def tokenize(text, max_len):
    ids = [VOCAB_IDX.get(w, 1) for w in text.lower().split()][:max_len]
    return np.array(ids + [0] * (max_len - len(ids)), dtype=np.int64)


def detokenize(ids):
    return " ".join(VOCAB[i] for i in ids if i > 0)


# %% [markdown]
# ## 3. Drone simulator
# Egocentric frame: x forward, y left, z up. Sensors: FPV camera, LiDAR/4D-radar voxels (occupancy + Doppler),
# IMU/proprioception, non-uniform dt. A potential-field expert with tangential escape provides action labels.

# %%
def _rot_world_to_ego(yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


_GRID_CACHE = {}


def _grids(cfg):
    key = (cfg.img_size, cfg.lidar_zyx, cfg.lidar_extent, cfg.bev_grid)
    if key in _GRID_CACHE:
        return _GRID_CACHE[key]
    (z0, z1), (y0, y1), (x0, x1) = cfg.lidar_extent
    Z, Y, X = cfg.lidar_zyx
    zc = z0 + (np.arange(Z) + 0.5) * (z1 - z0) / Z
    yc = y0 + (np.arange(Y) + 0.5) * (y1 - y0) / Y
    xc = x0 + (np.arange(X) + 0.5) * (x1 - x0) / X
    vox = np.stack(np.meshgrid(xc, yc, zc, indexing="ij"), -1).transpose(2, 1, 0, 3)  # [Z,Y,X,(x,y,z)]
    G = cfg.bev_grid
    by = y0 + (np.arange(G) + 0.5) * (y1 - y0) / G
    bx = x0 + (np.arange(G) + 0.5) * (x1 - x0) / G
    bev = np.stack(np.meshgrid(bx, by, indexing="xy"), -1)  # [G(y), G(x), (x,y)]
    cell = max((y1 - y0) / G, (x1 - x0) / G)
    frustum = (bev[..., 0] > 0.5) & (np.abs(bev[..., 1]) < bev[..., 0])  # 90 deg camera FOV
    S = cfg.img_size
    out = dict(vox=vox.astype(np.float32), bev=bev.astype(np.float32), cell=cell,
               frustum=frustum.reshape(-1), S=S, cam_grid=min(S // 16, cfg.cam_grid))
    _GRID_CACHE[key] = out
    return out


class DroneSim:
    def __init__(self, cfg: Config, seed: int, intent: int | None = None):
        self.cfg = cfg
        r = self.rng = np.random.default_rng(seed)
        self.intent = int(r.integers(len(INTENT_PHRASES))) if intent is None else intent
        phr = INTENT_PHRASES[self.intent]
        self.command = phr[int(r.integers(len(phr)))]
        self.p = np.array([0.0, 0.0, r.uniform(2.5, 4.0)])
        self.v = np.zeros(3)
        self.acc = np.zeros(3)
        self.yaw = r.uniform(-0.4, 0.4)
        self.yaw_rate = 0.0
        self.a_prev = np.zeros(ACTION_DIM)
        while True:
            self.beacons = np.array([[r.uniform(9, 15), r.uniform(-6, 6), r.uniform(1.5, 5.0)] for _ in range(2)])
            if np.linalg.norm(self.beacons[0] - self.beacons[1]) > 4.0:
                break
        n = cfg.n_obstacles
        self.obs_p = np.zeros((n, 3))
        self.obs_r = r.uniform(0.6, 1.4, n)
        for i in range(n):
            for _ in range(50):
                c = np.array([r.uniform(3, 13), r.uniform(-6, 6), r.uniform(1.0, 6.0)])
                far_start = np.linalg.norm(c - self.p) > self.obs_r[i] + 2.0
                far_beacons = np.all(np.linalg.norm(self.beacons - c, axis=1) > self.obs_r[i] + 1.5)
                if far_start and far_beacons:
                    break
            self.obs_p[i] = c
        self.obs_v = np.zeros((n, 3))
        moving = r.random(n) < 0.4
        self.obs_v[moving, 1] = r.uniform(-1.0, 1.0, moving.sum())
        shade = r.uniform(0.35, 0.7, n)
        self.obs_col = np.stack([shade * 0.9, shade, shade * 0.8], 1)
        self.beacon_col = np.array([[1.0, 0.1, 0.1], [0.1, 0.25, 1.0]])
        self.t = 0.0
        self.last_dt = self.sample_dt()
        self.collided = False
        self.min_goal_dist = self.goal_dist()

    # ---------- dynamics ----------
    def sample_dt(self):
        return float(self.rng.uniform(*self.cfg.dt_range))

    def goal_dist(self):
        if self.intent in GOAL_INTENTS:
            return float(np.linalg.norm(self.beacons[self.intent] - self.p))
        return float("nan")

    def expert_raw(self):
        c = self.cfg
        v_att = np.zeros(3)
        r = 0.0
        if self.intent in GOAL_INTENTS:
            d = self.beacons[self.intent] - self.p
            dist = np.linalg.norm(d) + 1e-6
            v_att = d / dist * c.vmax * min(1.0, dist / 3.0)
            r = float(np.clip(1.5 * _wrap(math.atan2(d[1], d[0]) - self.yaw), -c.rmax, c.rmax))
        elif self.intent == 3:
            v_att = np.array([0.0, 0.0, 0.6])
        elif self.intent == 4:
            v_att = np.array([0.0, 0.0, -0.6 if self.p[2] > 1.2 else 0.0])
        elif self.intent == 5:
            r = 0.6
        elif self.intent == 6:
            r = -0.6
        v_rep = np.zeros(3)
        for o, rad in zip(self.obs_p, self.obs_r):
            diff = self.p - o
            dist = np.linalg.norm(diff) + 1e-6
            s = dist - rad - 0.3
            if s < 3.0:
                n = diff / dist
                mag = min(4.0, 1.5 * (1.0 / max(s, 0.1) - 1.0 / 3.0))
                v_rep += mag * n
                t = np.array([-n[1], n[0], 0.0])
                side = np.sign(t @ v_att) or 1.0
                v_rep += 0.5 * mag * side * t
        if self.p[2] < 1.0:
            v_rep[2] += 2.0 * (1.0 - self.p[2])
        u = v_att + v_rep
        sp = np.linalg.norm(u)
        if sp > c.vmax:
            u = u / sp * c.vmax
        body = _rot_world_to_ego(self.yaw) @ u
        return np.clip(np.array([*(body / c.vmax), r / c.rmax]), -1, 1)

    def expert_action(self, dt):
        """First-order-smoothed expert (actuator-feasible labels)."""
        alpha = 1 - math.exp(-dt / 0.1)
        self.a_prev = self.a_prev + alpha * (self.expert_raw() - self.a_prev)
        return self.a_prev.copy()

    def step(self, a, dt):
        c = self.cfg
        a = np.clip(a, -1, 1)
        u_world = _rot_world_to_ego(self.yaw).T @ (a[:3] * c.vmax)
        alpha = 1 - math.exp(-dt / 0.15)
        v_new = self.v + alpha * (u_world - self.v)
        self.acc = (v_new - self.v) / dt
        self.p = self.p + v_new * dt
        self.v = v_new
        self.yaw_rate += alpha * (a[3] * c.rmax - self.yaw_rate)
        self.yaw = _wrap(self.yaw + self.yaw_rate * dt)
        self.obs_p = self.obs_p + self.obs_v * dt
        bounce = np.abs(self.obs_p[:, 1]) > 8.0
        self.obs_v[bounce, 1] *= -1
        self.t += dt
        self.last_dt = dt
        d = np.linalg.norm(self.obs_p - self.p, axis=1) - self.obs_r
        if np.any(d < 0.3) or self.p[2] < 0.2:
            self.collided = True
        if self.intent in GOAL_INTENTS:
            self.min_goal_dist = min(self.min_goal_dist, self.goal_dist())

    def step_expert(self, noise=0.0):
        dt = self.sample_dt()
        a = self.expert_action(dt)
        self.step(a + noise * self.rng.standard_normal(ACTION_DIM), dt)
        return a, dt

    def success(self):
        return (self.intent in GOAL_INTENTS) and self.min_goal_dist < 1.2 and not self.collided

    # ---------- sensors ----------
    def _objects(self):
        """(centres, radii, velocities, colours) of every physical object: obstacles + beacons."""
        P = np.concatenate([self.obs_p, self.beacons])
        R = np.concatenate([self.obs_r, np.full(2, 0.8)])
        V = np.concatenate([self.obs_v, np.zeros((2, 3))])
        C = np.concatenate([self.obs_col, self.beacon_col])
        return P, R, V, C

    def sensors(self):
        c = self.cfg
        Rw = _rot_world_to_ego(self.yaw)
        vb = Rw @ self.v / c.vmax
        ab = Rw @ self.acc / 5.0
        s = np.array([*vb, self.yaw_rate / c.rmax, *ab, self.p[2] / 5.0, self.last_dt / 0.1], dtype=np.float32)
        s[:8] += 0.01 * self.rng.standard_normal(8).astype(np.float32)
        return s

    def render(self):
        g = _grids(self.cfg)
        S = g["S"]
        f = S / 2.0
        img = np.empty((3, S, S), np.float32)
        h = S // 2
        sky = np.linspace(0.95, 0.75, h, dtype=np.float32)
        img[0, :h] = (0.55 * sky)[:, None]
        img[1, :h] = (0.72 * sky)[:, None]
        img[2, :h] = sky[:, None]
        ground = np.linspace(0.35, 0.5, S - h, dtype=np.float32) * (1.0 - 0.03 * min(self.p[2], 8))
        img[0, h:] = (ground * 1.0)[:, None]
        img[1, h:] = (ground * 0.8)[:, None]
        img[2, h:] = (ground * 0.55)[:, None]
        P, R, _, C = self._objects()
        ego = (P - self.p) @ _rot_world_to_ego(self.yaw).T
        order = np.argsort(-ego[:, 0])
        yy, xx = np.mgrid[0:S, 0:S].astype(np.float32)
        cls = np.zeros((S, S), np.int64)
        n_obs = len(self.obs_p)
        for i in order:
            x, y, z = ego[i]
            if x < 0.3:
                continue
            u = S / 2 - f * y / x
            v = S / 2 - f * z / x
            rad = f * R[i] / x
            u0, u1 = int(max(0, u - rad - 1)), int(min(S, u + rad + 2))
            v0, v1 = int(max(0, v - rad - 1)), int(min(S, v + rad + 2))
            if u0 >= u1 or v0 >= v1:
                continue
            m = (xx[v0:v1, u0:u1] - u) ** 2 + (yy[v0:v1, u0:u1] - v) ** 2 <= rad ** 2
            shade = float(np.clip(1.2 - x / 25.0, 0.35, 1.0))
            for ch in range(3):
                img[ch, v0:v1, u0:u1][m] = C[i, ch] * shade
            cls[v0:v1, u0:u1][m] = 1 if i < n_obs else 2 + (i - n_obs)
        img += 0.02 * self.rng.standard_normal(img.shape).astype(np.float32)
        gc = g["cam_grid"]
        p = S // gc
        c = cls[: gc * p, : gc * p].reshape(gc, p, gc, p).transpose(0, 2, 1, 3).reshape(gc * gc, -1)
        psem = np.where((c == 3).any(1), 3, np.where((c == 2).any(1), 2, np.where((c == 1).mean(1) > 0.25, 1, 0)))
        return np.clip(img, 0, 1), psem.astype(np.int64)

    def lidar_and_bev(self):
        """LiDAR/4D-radar voxels [2,Z,Y,X] (occupancy, Doppler) and BEV targets (occupancy, radial velocity)."""
        c = self.cfg
        g = _grids(c)
        P, R, V, _ = self._objects()
        Rw = _rot_world_to_ego(self.yaw)
        ego = (P - self.p) @ Rw.T
        v_rel = (V - self.v) @ Rw.T
        los = ego / (np.linalg.norm(ego, axis=1, keepdims=True) + 1e-6)
        v_rad = np.sum(v_rel * los, 1) / c.vmax
        vox = g["vox"]
        Z, Y, X = c.lidar_zyx
        occ = np.zeros((Z, Y, X), np.float32)
        dop = np.zeros((Z, Y, X), np.float32)
        best = np.full((Z, Y, X), np.inf, np.float32)
        for i in range(len(P)):
            d = np.linalg.norm(vox - ego[i], axis=-1)
            m = (d < R[i]) & (d < best)
            occ[m] = 1.0
            dop[m] = v_rad[i]
            best[m] = d[m]
        drop = self.rng.random(occ.shape) < 0.05
        occ[drop] = 0.0
        dop[drop] = 0.0
        spur = self.rng.random(occ.shape) < 0.002
        occ[spur] = 1.0
        lidar = np.stack([occ, dop])
        (z0, z1), _, _ = c.lidar_extent
        bev = g["bev"].reshape(-1, 2)
        bocc = np.zeros(len(bev), np.float32)
        bvel = np.zeros(len(bev), np.float32)
        bsem = np.zeros(len(bev), np.int64)  # 0 empty, 1 obstacle, 2 red beacon, 3 blue beacon
        n_obs = len(self.obs_p)
        bbest = np.full(len(bev), np.inf, np.float32)
        for i in range(len(P)):
            if not (z0 - R[i] < ego[i, 2] < z1 + R[i]):
                continue
            d = np.linalg.norm(bev - ego[i, :2], axis=1)
            m = (d < R[i] + 0.35 * g["cell"]) & (d < bbest)
            bocc[m] = 1.0
            bvel[m] = v_rad[i]
            bsem[m] = 1 if i < n_obs else 2 + (i - n_obs)
            bbest[m] = d[m]
        return lidar, bocc, bvel, bsem

    def observe(self, render=True):
        lidar, occ, vel, sem = self.lidar_and_bev()
        ob = dict(lidar=lidar, occ=occ, vel=vel, sem=sem, sensors=self.sensors(),
                  dt=np.float32(self.last_dt))
        if render:
            ob["frame"], ob["psem"] = self.render()
        return ob


def rollout_episode(cfg, seed, L, render=True):
    """Expert episode with DART execution noise. Labels are the clean *raw* expert commands.
    (Smoothed labels are ~equal to the drone's current velocity, which the policy observes through the IMU: the
    policy then learns to copy its own velocity — 'copycat' causal confusion — and never leaves a hover.)"""
    sim = DroneSim(cfg, seed)
    rng = sim.rng
    noise = cfg.dart_noise if rng.random() < 0.5 else 0.0
    for _ in range(int(rng.integers(0, cfg.warmup_max + 1))):
        sim.step_expert(noise)
        if sim.collided or (sim.intent in GOAL_INTENTS and sim.goal_dist() < 1.5):
            break
    keys = ["lidar", "occ", "vel", "sensors", "dt"] + (["frame"] if render else [])
    buf = {k: [] for k in keys}
    acts = []
    for _ in range(L):
        ob = sim.observe(render)
        for k in keys:
            buf[k].append(ob[k])
        dt = sim.sample_dt()
        a = sim.expert_raw()
        acts.append(a.astype(np.float32))
        sim.step(a + noise * rng.standard_normal(ACTION_DIM), dt)
    ep = {k: np.stack(v) for k, v in buf.items()}
    ep["actions"] = np.stack(acts)
    ep["cmd"] = tokenize(sim.command, cfg.max_cmd_len)
    ep["intent"] = np.int64(sim.intent)
    return ep


class FrameDataset(Dataset):
    """Single random frames for Stage 1 (cheap: simulate without rendering, render only the sampled step)."""

    def __init__(self, cfg, n, seed_base):
        self.cfg, self.n, self.seed_base = cfg, n, seed_base

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        sim = DroneSim(self.cfg, self.seed_base + i)
        for _ in range(int(sim.rng.integers(0, 60))):
            sim.step_expert(self.cfg.dart_noise)
            if sim.collided:
                break
        ob = sim.observe(render=True)
        return {k: torch.from_numpy(np.asarray(ob[k])) for k in ("frame", "lidar", "occ", "vel", "sem", "psem")}


class EpisodeDataset(Dataset):
    def __init__(self, cfg, n, seed_base, L, render=True):
        self.cfg, self.n, self.seed_base, self.L, self.render = cfg, n, seed_base, L, render

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        ep = rollout_episode(self.cfg, self.seed_base + i, self.L, self.render)
        return {k: torch.from_numpy(np.asarray(v)) for k, v in ep.items()}


# %% [markdown]
# ## 4. Perception: frozen/LoRA ViT, LiDAR/radar pillar encoder, camera→BEV lifter

# %%
class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r=4, alpha=8.0):
        super().__init__()
        self.base = base
        self.scale = alpha / r
        self.A = nn.Parameter(torch.randn(r, base.in_features) * 0.01)
        self.B = nn.Parameter(torch.zeros(base.out_features, r))

    def forward(self, x):
        return self.base(x) + (x @ self.A.t() @ self.B.t()) * self.scale


class _TinyPatchEncoder(nn.Module):
    """Fallback when timm is unavailable."""

    def __init__(self, img, patch=16, dim=192):
        super().__init__()
        self.proj = nn.Conv2d(3, dim, patch, patch)
        self.embed_dim = dim
        self.num_prefix_tokens = 0
        self.blocks = nn.ModuleList()

    def forward_features(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


def build_vit(cfg):
    """Returns (vit, is_pretrained). Without pretrained weights the ViT is trained from scratch in Stage 1
    (a frozen random ViT would only be a random projection)."""
    if timm is None:
        print("[warn] timm not installed: using a small patch encoder instead of a ViT")
        return _TinyPatchEncoder(cfg.img_size), False
    if cfg.pretrained:
        try:
            return timm.create_model(cfg.vit_name, pretrained=True, num_classes=0, img_size=cfg.img_size), True
        except Exception as e:  # offline / hub blocked
            print(f"[warn] pretrained ViT unavailable ({type(e).__name__}); training the ViT from scratch")
    return timm.create_model(cfg.vit_name, pretrained=False, num_classes=0, img_size=cfg.img_size), False


class CameraEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.vit, self.pretrained = build_vit(cfg)
        for p in self.vit.parameters():
            p.requires_grad = not self.pretrained
        if self.pretrained and cfg.lora_rank > 0 and len(self.vit.blocks):
            for blk in list(self.vit.blocks)[-cfg.lora_blocks:]:
                blk.attn.qkv = LoRALinear(blk.attn.qkv, cfg.lora_rank)
        self.n_prefix = getattr(self.vit, "num_prefix_tokens", 1)
        self.proj = nn.Linear(self.vit.embed_dim, cfg.d_bev)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        with torch.no_grad():
            self.n_patches = self.vit.forward_features(torch.zeros(1, 3, cfg.img_size, cfg.img_size)).shape[1] - self.n_prefix

    def forward(self, x):
        tok = self.vit.forward_features((x - self.mean) / self.std)[:, self.n_prefix:]
        return self.proj(tok)


class _PillarTrunk(nn.Module):
    def __init__(self, G, widths, d_out):
        super().__init__()
        self.G = G

        def blk(i, o, s):
            return nn.Sequential(nn.Conv3d(i, o, 3, s, 1), nn.GroupNorm(4, o), nn.GELU())
        a, b, c = widths
        self.net = nn.Sequential(blk(2, a, 1), blk(a, b, 2), blk(b, c, 2))
        self.out = nn.Conv2d(2 * c, d_out, 1)

    def forward(self, v):
        h = self.net(v).amax(dim=2)  # collapse height -> pillars
        h = torch.cat([F.adaptive_avg_pool2d(h, self.G), F.adaptive_max_pool2d(h, self.G)], 1)
        return self.out(h).flatten(2).transpose(1, 2)  # [N, G*G, d]


class LiDAREncoder(nn.Module):
    """3D conv over (occupancy, Doppler) voxels, height-collapsed to BEV pillars (radar-PointPillars analogue).
    Two trunks: a geometry trunk (aligned with the camera by the contrastive loss) and a small motion trunk that
    only the Doppler loss shapes. A single shared trunk loses Doppler: the camera sees one frame and carries no
    velocity, so cross-modal alignment pushes motion information out of the shared features."""

    def __init__(self, cfg):
        super().__init__()
        self.geo = _PillarTrunk(cfg.bev_grid, (16, 32, 64), cfg.d_bev)
        self.dyn = _PillarTrunk(cfg.bev_grid, (8, 16, 32), cfg.d_bev // 2)

    def forward(self, v):
        return self.geo(v), self.dyn(v)


class CrossBlock(nn.Module):
    def __init__(self, d, heads, dropout=0.0):
        super().__init__()
        self.lq, self.lkv, self.l2 = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, q, kv):
        k = self.lkv(kv)
        q = q + self.attn(self.lq(q), k, k, need_weights=False)[0]
        return q + self.mlp(self.l2(q))


class BEVLifter(nn.Module):
    """Learned BEV cell queries cross-attend to camera patch tokens (perspective → metric BEV lift)."""

    def __init__(self, cfg, n_patches):
        super().__init__()
        d = cfg.d_bev
        self.queries = nn.Parameter(torch.randn(cfg.n_cells, d) * 0.02)
        self.cam_pos = nn.Parameter(torch.randn(n_patches, d) * 0.02)
        self.blocks = nn.ModuleList([CrossBlock(d, 4) for _ in range(2)])

    def forward(self, cam_tok):
        q = self.queries.expand(cam_tok.shape[0], -1, -1)
        kv = cam_tok + self.cam_pos
        for b in self.blocks:
            q = b(q, kv)
        return q


class Perception(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_bev
        self.cam = CameraEncoder(cfg)
        self.lidar = LiDAREncoder(cfg)
        self.lifter = BEVLifter(cfg, self.cam.n_patches)
        self.bev_pos = nn.Parameter(torch.randn(cfg.n_cells, d) * 0.02)
        self.fuse_mlp = nn.Sequential(nn.Linear(2 * d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
        self.fuse_ln = nn.LayerNorm(d)
        self.occ_head = nn.Linear(d, 1)
        self.sem_head = nn.Linear(d, 4)  # camera-only semantics: empty / obstacle / red / blue beacon
        self.psem_head = nn.Linear(d, 4)  # per camera token (perspective) semantics
        self.dyn_proj = nn.Linear(d // 2, d)
        self.vel_head = nn.Sequential(nn.Linear(d // 2 + d, d), nn.GELU(), nn.Linear(d, 1))
        self.proj_rad = nn.Linear(d, 64)
        self.proj_vis = nn.Linear(d, 64)
        self.register_buffer("frustum", torch.from_numpy(_grids(cfg)["frustum"]))

    def lidar_parts(self, lidar):
        geo, dyn = self.lidar(lidar)
        f_geo = geo + self.bev_pos
        return f_geo, dyn, f_geo + self.dyn_proj(dyn)

    def lidar_bev(self, lidar):
        """Radar/LiDAR BEV features used downstream (geometry + motion)."""
        return self.lidar_parts(lidar)[2]

    def camera_tokens(self, frames):
        return self.cam(frames)

    def camera_bev(self, frames, cam=None):
        return self.lifter(self.cam(frames) if cam is None else cam)

    def pool_cam(self, cam):
        """Camera patch tokens pooled to <= cam_grid^2 perspective tokens (kept for bearing / colour cues)."""
        N, P, d = cam.shape
        g = int(round(P ** 0.5))
        gc = _grids(self.cfg)["cam_grid"]
        x = cam.transpose(1, 2).reshape(N, d, g, g)
        if g != gc:
            x = F.adaptive_avg_pool2d(x, gc)
        return x.flatten(2).transpose(1, 2)

    def scene(self, fused, cam_pooled):
        """Scene tokens for the policy: G^2 metric BEV cells followed by perspective camera tokens."""
        return torch.cat([fused, cam_pooled], 1)

    def fuse(self, f_rad, f_vis):
        return self.fuse_ln(f_rad + f_vis + self.fuse_mlp(torch.cat([f_rad, f_vis], -1)))

    def forward(self, frames, lidar):
        f_geo, f_dyn, f_rad = self.lidar_parts(lidar)
        cam = self.camera_tokens(frames)
        f_vis = self.camera_bev(frames, cam)
        fused = self.fuse(f_rad, f_vis)
        cam_p = self.pool_cam(cam)
        return dict(f_geo=f_geo, f_dyn=f_dyn, f_rad=f_rad, f_vis=f_vis, fused=fused, cam_p=cam_p,
                    scene=self.scene(fused, cam_p))

    def losses(self, out, occ, vel, sem, psem):
        """L_contra (cell-level radar↔camera InfoNCE) + occupancy (metric depth) + L_vel (Doppler)
        + L_sem (camera BEV semantics from ground-truth boxes: which cell holds which beacon)."""
        f_rad, f_vis, fused = out["f_geo"].float(), out["f_vis"].float(), out["fused"].float()
        N, C, _ = f_rad.shape
        fr = self.frustum[None].expand(N, C)
        occ_logit = self.occ_head(f_vis).squeeze(-1)
        pos_w = torch.tensor(8.0, device=occ.device)
        l_occ = F.binary_cross_entropy_with_logits(occ_logit[fr], occ[fr], pos_weight=pos_w)
        occm = occ > 0.5
        v_hat = self.vel_head(torch.cat([out["f_dyn"].float(), fused], -1)).squeeze(-1)
        l_vel = (v_hat - vel).abs()[occm].mean() if occm.any() else v_hat.sum() * 0
        sel = (occm & fr).flatten().nonzero().squeeze(-1)
        if sel.numel() > 512:
            sel = sel[torch.randperm(sel.numel(), device=sel.device)[:512]]
        if sel.numel() >= 2:
            zr = F.normalize(self.proj_rad(f_rad.reshape(N * C, -1)[sel]), dim=-1)
            zv = F.normalize(self.proj_vis(f_vis.reshape(N * C, -1)[sel]), dim=-1)
            logits = zr @ zv.t() / self.cfg.tau_contra
            lab = torch.arange(sel.numel(), device=logits.device)
            l_con = 0.5 * (F.cross_entropy(logits, lab) + F.cross_entropy(logits.t(), lab))
            acc = (logits.argmax(1) == lab).float().mean()
        else:
            l_con = f_rad.sum() * 0
            acc = torch.tensor(float("nan"))
        with torch.no_grad():
            pred = (occ_logit > 0) & fr
            tgt = occm & fr
            iou = (pred & tgt).sum() / ((pred | tgt).sum() + 1e-6)
            vel_zero = vel.abs()[occm].mean() if occm.any() else torch.tensor(0.0)
        sem_logit = self.sem_head(f_vis)
        w = torch.tensor([0.2, 1.0, 4.0, 4.0], device=sem.device)
        l_sem = F.cross_entropy(sem_logit[fr], sem[fr], weight=w)
        with torch.no_grad():
            bm = (sem >= 2) & fr
            beacon_acc = (sem_logit.argmax(-1)[bm] == sem[bm]).float().mean() if bm.any() else torch.tensor(float("nan"))
        p_logit = self.psem_head(out["cam_p"].float())
        l_psem = F.cross_entropy(p_logit.reshape(-1, 4), psem.reshape(-1), weight=w)
        with torch.no_grad():
            pm = psem >= 2
            patch_beacon_acc = (p_logit.argmax(-1)[pm] == psem[pm]).float().mean() if pm.any() else torch.tensor(float("nan"))
        loss = l_con + self.cfg.lam_occ * l_occ + self.cfg.lam_vel * l_vel + self.cfg.lam_sem * l_sem \
            + self.cfg.lam_psem * l_psem
        stats = dict(l_contra=l_con.item(), l_occ=l_occ.item(), l_vel=l_vel.item(), l_sem=l_sem.item(),
                     beacon_cell_acc=beacon_acc.item(), l_psem=l_psem.item(), patch_beacon_acc=patch_beacon_acc.item(),
                     contra_top1=acc.item(),
                     cam_occ_iou=iou.item(), vel_mae=l_vel.item(), vel_mae_zero_baseline=vel_zero.item(),
                     n_pairs=int(sel.numel()))
        return loss, stats


# %% [markdown]
# ## 5. Language encoder and state tokenizer (text-conditioned cross-attention over fused BEV)

# %%
class TextEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg.d_model
        self.mode = cfg.text_backbone
        if self.mode == "clip":
            try:
                from transformers import CLIPTextModel, CLIPTokenizer
                name = "openai/clip-vit-base-patch32"
                self.tok = CLIPTokenizer.from_pretrained(name)
                self.clip = CLIPTextModel.from_pretrained(name).eval()
                for p in self.clip.parameters():
                    p.requires_grad = False
                self.proj = nn.Linear(self.clip.config.hidden_size, d)
                self._memo = {}
            except Exception as e:
                print(f"[warn] CLIP text tower unavailable ({type(e).__name__}); falling back to simple encoder")
                self.mode = "simple"
        if self.mode == "simple":
            self.emb = nn.Embedding(len(VOCAB), d, padding_idx=0)
            self.pos = nn.Parameter(torch.randn(cfg.max_cmd_len, d) * 0.02)
            self.enc = nn.TransformerEncoderLayer(d, cfg.n_heads, 2 * d, dropout=0.0, batch_first=True, norm_first=True)
        self.ln = nn.LayerNorm(d)

    def forward(self, tokens):  # [B, Lc] int
        if self.mode == "clip":
            outs = []
            for row in tokens.tolist():
                s = detokenize(row)
                if s not in self._memo:
                    with torch.no_grad():
                        t = self.tok([s], return_tensors="pt", padding=True).to(tokens.device)
                        self._memo[s] = self.clip.to(tokens.device)(**t).pooler_output[0]
                outs.append(self._memo[s])
            return self.ln(self.proj(torch.stack(outs)))
        mask = tokens > 0
        x = self.enc(self.emb(tokens) + self.pos[: tokens.shape[1]], src_key_padding_mask=~mask)
        x = (x * mask[..., None]).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        return self.ln(x)


class StateTokenizer(nn.Module):
    """s_t = LN(CrossAttn(q = learned queries + command, kv = fused BEV cells) + proprio(sensors_t) + command)."""

    def __init__(self, cfg):
        super().__init__()
        d = cfg.d_model
        self.in_proj = nn.Linear(cfg.d_bev, d)
        self.scene_pos = nn.Parameter(torch.randn(cfg.n_scene, d) * 0.02)  # BEV cells + camera tokens
        self.ln_kv = nn.LayerNorm(d)
        self.queries = nn.Parameter(torch.randn(cfg.n_pool_queries, d) * 0.02)
        self.attn = nn.MultiheadAttention(d, cfg.n_heads, batch_first=True)
        self.out = nn.Linear(cfg.n_pool_queries * d, d)
        self.proprio = nn.Sequential(nn.Linear(SENSOR_DIM, d), nn.GELU(), nn.Linear(d, d))
        self.ln = nn.LayerNorm(d)
        self.register_buffer("pmask", proprio_mask(cfg))

    def forward(self, bev, sensors, cmd):  # bev [B,L,C,d_bev], sensors [B,L,9], cmd [B,d]
        sensors = sensors * self.pmask
        B, L = bev.shape[:2]
        kv = self.ln_kv(self.in_proj(bev.flatten(0, 1)) + self.scene_pos)
        q = self.queries[None] + cmd.repeat_interleave(L, 0)[:, None]
        pooled = self.attn(q, kv, kv, need_weights=False)[0].flatten(1)
        s = self.out(pooled).view(B, L, -1)
        return self.ln(s + self.proprio(sensors) + cmd[:, None])


# %% [markdown]
# ## 6. Neuromorphic core: LIF spiking attention (fast-sigmoid surrogate) + closed-form LTC
#
# LIF membrane: u_t = β u_{t-1} + (1-β) x_t,  β = exp(-Δt_t / τ),   spike = H(u - θ), soft reset.
# LTC (closed-form, non-uniform Δt):
#   f = softplus(W_x x_t + W_h h_{t-1} + b),   g = 1/τ + f
#   h_t = h_{t-1} ⊙ exp(-Δt g) + (A ⊙ f / g) ⊙ (1 - exp(-Δt g))
# Both run in FP32 regardless of autocast.

# %%
class FastSigmoidSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, k):
        ctx.save_for_backward(x)
        ctx.k = k
        return (x > 0).to(x.dtype)

    @staticmethod
    def backward(ctx, g):
        (x,) = ctx.saved_tensors
        return g / (1.0 + ctx.k * x.abs()) ** 2, None


class LIF(nn.Module):
    def __init__(self, n, tau_init, thresh=1.0, k=5.0):
        super().__init__()
        self.log_tau = nn.Parameter(torch.full((n,), math.log(tau_init)))
        self.thresh, self.k = thresh, k

    def forward(self, x, dt, u=None):  # x [B,T,n], dt [B,T]
        with fp32_region(x):
            x, dt = x.float(), dt.float()
            tau = self.log_tau.exp().clamp(min=1e-3)
            u = torch.zeros_like(x[:, 0]) if u is None else u.float()
            out = []
            for t in range(x.shape[1]):
                beta = torch.exp(-dt[:, t, None] / tau)
                u = beta * u + (1 - beta) * x[:, t]
                s = FastSigmoidSpike.apply(u - self.thresh, self.k)
                u = u - s * self.thresh
                out.append(s)
            return torch.stack(out, 1), u


class SpikingSelfAttention(nn.Module):
    def __init__(self, d, heads, cfg):
        super().__init__()
        self.h, self.dh = heads, d // heads
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)
        self.spike_v = cfg.spike_v
        self.spiking = cfg.spiking
        mk = lambda: LIF(d, cfg.lif_tau_init, k=cfg.surrogate_k)
        self.lif_in = mk() if (cfg.spike_inputs and cfg.spiking) else None
        self.lif_q, self.lif_k = (mk(), mk()) if cfg.spiking else (None, None)
        self.lif_v = mk() if (cfg.spike_v and cfg.spiking) else None

    def forward(self, x, dt, st):
        B, T, D = x.shape
        in_rate, u_in = [], None
        if self.lif_in is not None:  # spike-driven projection: binary input -> W_qkv is accumulate-only
            x, u_in = self.lif_in(x, dt, st.get("in"))
            in_rate = [x.mean()]
            x = x.to(self.qkv.weight.dtype)
        q, k, v = self.qkv(x).chunk(3, -1)
        if not self.spiking:
            q, k, v = (z.view(B, T, self.h, self.dh).transpose(1, 2) for z in (q, k, v))
            o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            return self.out(o.transpose(1, 2).reshape(B, T, D)), {}, []
        q, uq = self.lif_q(q, dt, st.get("q"))
        k, uk = self.lif_k(k, dt, st.get("k"))
        new = {"q": uq, "k": uk}
        if u_in is not None:
            new["in"] = u_in
        rates = in_rate + [q.mean(), k.mean()]
        if self.lif_v is not None:
            v, uv = self.lif_v(v, dt, st.get("v"))
            new["v"] = uv
            rates.append(v.mean())
        q, k, v = (z.view(B, T, self.h, self.dh).transpose(1, 2) for z in (q, k, v.to(q.dtype)))
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.out(o.transpose(1, 2).reshape(B, T, D).to(x.dtype)), new, rates


class LTCCell(nn.Module):
    def __init__(self, n_in, n, tau_init):
        super().__init__()
        self.W_x = nn.Linear(n_in, n)
        self.W_h = nn.Linear(n, n, bias=False)
        nn.init.orthogonal_(self.W_h.weight, gain=0.5)
        self.log_tau = nn.Parameter(torch.full((n,), math.log(tau_init)))
        self.A = nn.Parameter(torch.randn(n) * 0.5)

    def forward(self, x, dt, h=None):
        pre = self.W_x(x)
        with fp32_region(x):
            pre, dt = pre.float(), dt.float()
            h = torch.zeros_like(pre[:, 0]) if h is None else h.float()
            inv_tau = 1.0 / self.log_tau.exp().clamp(min=1e-3)
            Wh = self.W_h.weight.float()
            out = []
            for t in range(pre.shape[1]):
                f = F.softplus(pre[:, t] + h @ Wh.t())
                g = inv_tau + f
                e = torch.exp(-dt[:, t, None] * g)
                h = h * e + (self.A * f / g) * (1 - e)
                out.append(h)
            return torch.stack(out, 1), h


class CoreLayer(nn.Module):
    def __init__(self, d, heads, ltc_hidden, cfg):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = SpikingSelfAttention(d, heads, cfg)
        self.ff_mode = cfg.core_ff
        self.lif_ltc_in = LIF(d, cfg.lif_tau_init, k=cfg.surrogate_k) if (cfg.spike_inputs and cfg.spiking) else None
        if cfg.core_ff == "ltc":
            self.ltc = LTCCell(d, ltc_hidden, cfg.ltc_tau_init)
        else:
            self.ltc = nn.Sequential(nn.Linear(d, ltc_hidden), nn.GELU())
        self.ltc_out = nn.Linear(ltc_hidden, d)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x, dt, st):
        a, s_attn, rates = self.attn(self.ln1(x), dt, st.get("attn", {}))
        x = x + self.drop(a)
        if self.ff_mode != "ltc":
            x = x + self.drop(self.ltc_out(self.ltc(self.ln2(x))))
            return x, {"attn": s_attn}, rates
        z = self.ln2(x)
        new = {"attn": s_attn}
        if self.lif_ltc_in is not None:  # spike-driven W_x
            z, new["ltc_in"] = self.lif_ltc_in(z, dt, st.get("ltc_in"))
            rates = rates + [z.mean()]
            z = z.to(x.dtype)
        hs, h = self.ltc(z, dt, st.get("ltc"))
        x = x + self.drop(self.ltc_out(hs.to(x.dtype)))
        new["ltc"] = h
        return x, new, rates


class Core(nn.Module):
    def __init__(self, d, heads, n_layers, ltc_hidden, cfg):
        super().__init__()
        self.layers = nn.ModuleList([CoreLayer(d, heads, ltc_hidden, cfg) for _ in range(n_layers)])
        self.ln = nn.LayerNorm(d)

    def forward(self, x, dt, state=None):
        state = state or [{} for _ in self.layers]
        new, rates = [], []
        for layer, st in zip(self.layers, state):
            x, ns, r = layer(x, dt, st)
            new.append(ns)
            rates += r
        return self.ln(x), new, rates


def detach_state(s):
    if isinstance(s, torch.Tensor):
        return s.detach()
    if isinstance(s, dict):
        return {k: detach_state(v) for k, v in s.items()}
    if isinstance(s, list):
        return [detach_state(v) for v in s]
    return s


@torch.no_grad()
def clamp_spectral(module, max_sv):
    """Eigenvalue control for BPTT stability: rescale W_h so that ||W_h||_2 <= max_sv (hence |λ| <= max_sv)."""
    for m in module.modules():
        if isinstance(m, LTCCell):
            s = torch.linalg.matrix_norm(m.W_h.weight.detach().float().cpu(), ord=2).to(m.W_h.weight.device)
            if s > max_sv:
                m.W_h.weight.mul_(max_sv / s)


def _mean(x):
    return float(np.mean(x)) if len(x) else 0.0


def spike_loss(rates, cfg):
    if not rates:
        return torch.zeros(())
    return cfg.lam_spike * sum((r - cfg.r_target) ** 2 for r in rates) / len(rates)


# %% [markdown]
# ## 7. Heads: CNN action-chunk head (Gaussian), multi-horizon JEPA predictor

# %%
class ChunkHead(nn.Module):
    """h_t -> H-step action chunk (mean, log-std) via a 1-D temporal conv decoder."""

    def __init__(self, d, H, A, c=64):
        super().__init__()
        self.c, self.H = c, H
        self.fc = nn.Linear(d, c * H)
        self.conv = nn.Sequential(nn.Conv1d(c, c, 3, padding=1), nn.GELU(), nn.Conv1d(c, c, 3, padding=1), nn.GELU())
        self.mu = nn.Conv1d(c, A, 1)
        self.ls = nn.Conv1d(c, A, 1)
        nn.init.constant_(self.ls.bias, -1.5)

    def forward(self, h):
        lead = h.shape[:-1]
        z = self.fc(h.reshape(-1, h.shape[-1])).view(-1, self.c, self.H)
        z = z + self.conv(z)
        mu = torch.tanh(self.mu(z)).transpose(1, 2)
        ls = self.ls(z).transpose(1, 2).clamp(-5, 1)
        return mu.reshape(*lead, self.H, -1).float(), ls.reshape(*lead, self.H, -1).float()


class JEPAPredictor(nn.Module):
    """g_ψ(h_t, ω_t, horizon) -> future fused-BEV latent per cell, trained against stop-grad targets.
    Residual form: ŝ_{t+h} = sg(s_t) + g_ψ(...), so the core must explain *change* (motion of the scene and ego),
    not re-encode the present — and the predictor starts exactly at the persistence baseline (zero-init)."""

    def __init__(self, cfg):
        super().__init__()
        d = cfg.d_model
        self.nh, self.C, self.db = len(cfg.jepa_horizons), cfg.n_cells, cfg.d_bev
        self.h_emb = nn.Embedding(self.nh, d)
        self.imu = nn.Linear(SENSOR_DIM, d)
        self.net = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, self.C * self.db))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, h, sensors, current):  # current = sg(target_t) [B,T,C,d_bev] -> [B,T,nh,C,d_bev]
        z = h[:, :, None] + self.imu(sensors)[:, :, None] + self.h_emb.weight[None, None]
        B, T = h.shape[:2]
        delta = self.net(z).view(B, T, self.nh, self.C, self.db).float()
        return current.detach()[:, :, None].float() + delta


def jepa_targets(scene, n_cells):
    """World-model targets: the G^2 metric BEV cells of the scene tokens (camera tokens excluded)."""
    bev = scene[..., :n_cells, :].float()
    return F.layer_norm(bev, bev.shape[-1:])


def jepa_loss(pred, targets, t_idx, horizons):
    """pred [B,T,nh,C,d] for timesteps t_idx (1-D LongTensor into targets' time axis)."""
    losses, persist = [], []
    L = targets.shape[1]
    for j, hz in enumerate(horizons):
        tt = t_idx + hz
        ok = tt < L
        if not ok.any():
            continue
        tgt = targets[:, tt[ok]]
        losses.append(F.mse_loss(pred[:, ok, j], tgt))
        persist.append(F.mse_loss(targets[:, t_idx[ok]], tgt).detach())
    if not losses:
        z = pred.sum() * 0
        return z, z.detach()
    return torch.stack(losses).mean(), torch.stack(persist).mean()


# %% [markdown]
# ## 8. Full target model (PLM) and edge drafter

# %%
class PLM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.perception = Perception(cfg)
        self.text = TextEncoder(cfg)
        self.tok = StateTokenizer(cfg)
        self.core = Core(cfg.d_model, cfg.n_heads, cfg.n_layers, cfg.ltc_hidden, cfg)
        self.head = ChunkHead(cfg.d_model, cfg.H, ACTION_DIM)
        self.jepa = JEPAPredictor(cfg)

    def tokens(self, fused, sensors, cmd_tokens):
        return self.tok(fused, sensors, self.text(cmd_tokens))

    def policy_from_tokens(self, s, dt, state=None, last_only=False):
        """last_only: decode the action chunk for the final step only (all the controller needs at run time)."""
        h, state, rates = self.core(s, dt, state)
        mu, ls = self.head(h[:, -1:] if last_only else h)
        return dict(h=h, mu=mu, ls=ls, rates=rates, state=state)

    def policy(self, fused, sensors, dt, cmd_tokens, state=None):
        out = self.policy_from_tokens(self.tokens(fused, sensors, cmd_tokens), dt, state)
        return out


class EdgeDrafter(nn.Module):
    """Cheap drafter: no camera/ViT — LiDAR/radar BEV + proprio + command, 1-layer LIF/LTC core."""

    def __init__(self, cfg):
        super().__init__()
        d = cfg.d_draft
        self.cfg = cfg
        self.bev_in = nn.Linear(cfg.d_bev, d)
        self.pool = nn.Linear(2 * d, d)
        self.proprio = nn.Sequential(nn.Linear(SENSOR_DIM, d), nn.GELU(), nn.Linear(d, d))
        self.cmd = nn.EmbeddingBag(len(VOCAB), d, mode="mean", padding_idx=0)
        self.ln = nn.LayerNorm(d)
        self.core = Core(d, 2, 1, d, cfg)
        self.head = ChunkHead(d, cfg.H, ACTION_DIM, c=32)
        self.register_buffer("pmask", proprio_mask(cfg))

    def latent(self, rad, sensors, cmd_tokens):  # rad [B,L,C,d_bev]
        sensors = sensors * self.pmask
        z = F.gelu(self.bev_in(rad))
        z = self.pool(torch.cat([z.mean(2), z.amax(2)], -1))
        return self.ln(z + self.proprio(sensors) + self.cmd(cmd_tokens)[:, None])

    def policy_from_latent(self, z, dt, state=None, last_only=False):
        h, state, rates = self.core(z, dt, state)
        mu, ls = self.head(h[:, -1:] if last_only else h)
        return dict(mu=mu, ls=ls, rates=rates, state=state)

    def forward(self, rad, sensors, dt, cmd_tokens):
        return self.policy_from_latent(self.latent(rad, sensors, cmd_tokens), dt)


def count_params(m, trainable=False):
    return sum(p.numel() for p in m.parameters() if (p.requires_grad or not trainable))


# %% [markdown]
# ## 9. Losses for Stages 3-4

# %%
def traj_loss(mu, target, cfg):
    """Σ_k γ^k (λθ ||θ̂ - θ*||² + λv ||v̂ - v*||²)  over the H-step chunk (normalised by Σ γ^k)."""
    H = mu.shape[-2]
    w = cfg.gamma ** torch.arange(H, device=mu.device, dtype=torch.float32)
    lam = torch.tensor([cfg.lam_v] * 3 + [cfg.lam_theta], device=mu.device)
    per = ((mu - target) ** 2 * lam).sum(-1)  # [..., H]
    return (per * w).sum(-1).mean() / w.sum()


def jerk(u):
    return ((u[..., 2:, :] - 2 * u[..., 1:-1, :] + u[..., :-2, :]) ** 2).sum(-1).mean()


def gaussian_nll(x, mu, ls):
    return (0.5 * ((x - mu) / ls.exp()) ** 2 + ls).mean()


def gaussian_kl(mu_p, ls_p, mu_q, ls_q):
    """KL(p || q) for diagonal Gaussians, summed over action dims, averaged elsewhere."""
    vp, vq = (2 * ls_p).exp(), (2 * ls_q).exp()
    return (ls_q - ls_p + (vp + (mu_p - mu_q) ** 2) / (2 * vq) - 0.5).sum(-1).mean()


def gaussian_logp(x, mu, ls):
    return (-0.5 * ((x - mu) / ls.exp()) ** 2 - ls - 0.5 * math.log(2 * math.pi)).sum((-1, -2))


# %% [markdown]
# ## 10. Training utilities: optimiser with AMP / clipping / cosine schedule, logger

# %%
class Optim:
    def __init__(self, params, lr, steps, cfg, wd=0.01):
        self.params = [p for p in params if p.requires_grad]
        self.opt = torch.optim.AdamW(self.params, lr=lr, weight_decay=wd)
        warm = max(1, int(0.05 * steps))
        self.sched = torch.optim.lr_scheduler.LambdaLR(
            self.opt, lambda s: max(0.05, min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / max(1, steps))))))
        self.scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype(cfg.device) == torch.float16)
        self.clip = cfg.grad_clip

    def backward(self, loss):
        self.scaler.scale(loss).backward()

    def step(self):
        self.scaler.unscale_(self.opt)
        gn = torch.nn.utils.clip_grad_norm_(self.params, self.clip)
        self.scaler.step(self.opt)
        self.scaler.update()
        self.opt.zero_grad(set_to_none=True)
        self.sched.step()
        return float(gn)


class Logger:
    def __init__(self, out_dir):
        self.out_dir = out_dir
        self.hist = {}
        os.makedirs(out_dir, exist_ok=True)

    def log(self, stage, step, **kv):
        self.hist.setdefault(stage, []).append(dict(step=step, **kv))

    def print(self, stage, step, **kv):
        self.log(stage, step, **kv)
        msg = " ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in kv.items())
        print(f"[{stage}] step {step}: {msg}", flush=True)

    def save(self):
        with open(os.path.join(self.out_dir, "history.json"), "w") as f:
            json.dump(self.hist, f, indent=1)
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            stages = [s for s in self.hist if len(self.hist[s]) > 1]
            if not stages:
                return
            fig, axes = plt.subplots(1, len(stages), figsize=(4.2 * len(stages), 3.2), squeeze=False)
            for ax, s in zip(axes[0], stages):
                rows = self.hist[s]
                steps = [r["step"] for r in rows]
                for k in rows[0]:
                    if k == "step" or not isinstance(rows[0][k], (int, float)):
                        continue
                    if k.startswith(("loss", "l_", "val_")):
                        ax.plot(steps, [r.get(k, np.nan) for r in rows], label=k)
                ax.set_title(s)
                ax.set_yscale("log")
                ax.legend(fontsize=6)
            fig.tight_layout()
            fig.savefig(os.path.join(self.out_dir, "training_curves.png"), dpi=120)
            plt.close(fig)
        except Exception as e:
            print(f"[warn] plotting skipped: {e}")


def to_dev(batch, device):
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


# %% [markdown]
# ## 11. Stage 1 — Modality pre-alignment & BEV grounding

# %%
def stage1(cfg, plm, log):
    dev = cfg.device
    P = plm.perception
    dl = DataLoader(FrameDataset(cfg, cfg.s1_steps * cfg.s1_batch, 1_000_000), batch_size=cfg.s1_batch,
                    num_workers=cfg.num_workers, drop_last=True, persistent_workers=cfg.num_workers > 0)
    opt = Optim(P.parameters(), cfg.s1_lr, cfg.s1_steps, cfg)
    P.train()
    t0 = time.time()
    for step, b in enumerate(dl):
        b = to_dev(b, dev)
        with autocast(dev):
            out = P(b["frame"], b["lidar"])
        loss, st = P.losses(out, b["occ"], b["vel"], b["sem"], b["psem"])
        opt.backward(loss)
        opt.step()
        if step % cfg.log_every == 0 or step == cfg.s1_steps - 1:
            log.print("stage1", step, loss=loss.item(), **{k: v for k, v in st.items() if k != "n_pairs"},
                      sec=time.time() - t0)
    return evaluate_stage1(cfg, plm, log)


@torch.no_grad()
def evaluate_stage1(cfg, plm, log, n=256):
    P = plm.perception.eval()
    dl = DataLoader(FrameDataset(cfg, n, 9_000_000), batch_size=32, num_workers=cfg.num_workers)
    agg = {}
    for b in dl:
        b = to_dev(b, cfg.device)
        with autocast(cfg.device):
            out = P(b["frame"], b["lidar"])
        _, st = P.losses(out, b["occ"], b["vel"], b["sem"], b["psem"])
        for k, v in st.items():
            agg.setdefault(k, []).append(v)
    res = {k: float(np.nanmean(v)) for k, v in agg.items()}
    log.print("stage1_val", 0, **res)
    return res


# %% [markdown]
# ## 12. Feature cache (frozen perception → fused / radar BEV, FP16, host memory)

# %%
@torch.no_grad()
def build_cache(cfg, plm, n_episodes, seed_base, path=None):
    if path and os.path.exists(path):
        print(f"loading cache {path}")
        return torch.load(path)
    dev = cfg.device
    P = plm.perception.eval()
    L = cfg.episode_len
    dl = DataLoader(EpisodeDataset(cfg, n_episodes, seed_base, L), batch_size=2, num_workers=cfg.num_workers)
    keys = ["fused", "rad", "sensors", "dt", "actions", "cmd", "intent"]
    store = {k: [] for k in keys}
    t0 = time.time()
    for b in dl:
        B = b["frame"].shape[0]
        fr = b["frame"].flatten(0, 1).to(dev)
        li = b["lidar"].flatten(0, 1).to(dev)
        fused, rad = [], []
        for i in range(0, fr.shape[0], 64):
            with autocast(dev):
                o = P(fr[i:i + 64], li[i:i + 64])
            fused.append(o["scene"].half().cpu())
            rad.append(o["f_rad"].half().cpu())
        store["fused"].append(torch.cat(fused).view(B, L, cfg.n_scene, -1))
        store["rad"].append(torch.cat(rad).view(B, L, cfg.n_cells, -1))
        for k in ["sensors", "dt", "actions", "cmd", "intent"]:
            store[k].append(b[k])
    cache = {k: torch.cat(v) for k, v in store.items()}
    mb = sum(v.numel() * v.element_size() for v in cache.values()) / 2 ** 20
    print(f"cache: {n_episodes} episodes x {L} steps, {mb:.0f} MB, {time.time() - t0:.0f}s")
    if path:
        torch.save(cache, path)
    return cache


def sample_window(cache, B, L, keys, device, max_start=None):
    N, Lep = cache["dt"].shape
    max_start = Lep - L if max_start is None else max_start
    idx = torch.randint(N, (B,))
    t0 = torch.randint(0, max_start + 1, (B,))
    tt = t0[:, None] + torch.arange(L)[None]
    out = {}
    for k in keys:
        v = cache[k]
        out[k] = (v[idx] if v.dim() <= 2 and k in ("cmd", "intent") else v[idx[:, None], tt]).to(device).float() \
            if k not in ("cmd", "intent") else v[idx].to(device)
    return out


# %% [markdown]
# ## 13. Stage 2 — Self-supervised continuous-time dynamics (latent JEPA + spike-rate regularisation, TBPTT)

# %%
def stage2(cfg, plm, cache, val_cache, log):
    dev = cfg.device
    for p in plm.perception.parameters():
        p.requires_grad = False
    mods = [plm.text, plm.tok, plm.core, plm.jepa]
    opt = Optim([p for m in mods for p in m.parameters()], cfg.s2_lr, cfg.s2_steps, cfg)
    maxh = max(cfg.jepa_horizons)
    L = cfg.T_long + maxh
    for m in mods:
        m.train()
    t0 = time.time()
    for step in range(cfg.s2_steps):
        b = sample_window(cache, cfg.s2_batch, L, ["fused", "sensors", "dt", "cmd"], dev)
        targets = jepa_targets(b["fused"], cfg.n_cells)
        state, tot, tot_p, rates_all = None, 0.0, 0.0, []
        n_chunks = math.ceil(cfg.T_long / cfg.tbptt)
        for c0 in range(0, cfg.T_long, cfg.tbptt):
            c1 = min(c0 + cfg.tbptt, cfg.T_long)
            with autocast(dev):
                s = plm.tokens(b["fused"][:, c0:c1], b["sensors"][:, c0:c1], b["cmd"])
                h, state, rates = plm.core(s, b["dt"][:, c0:c1], state)
                pred = plm.jepa(h, b["sensors"][:, c0:c1], targets[:, c0:c1])
            l_dyn, l_pers = jepa_loss(pred, targets, torch.arange(c0, c1, device=dev), cfg.jepa_horizons)
            l_sp = spike_loss(rates, cfg)
            opt.backward((l_dyn + l_sp) / n_chunks)
            state = detach_state(state)
            tot += l_dyn.item() / n_chunks
            tot_p += l_pers.item() / n_chunks
            rates_all += [r.item() for r in rates]
        gn = opt.step()
        clamp_spectral(plm.core, cfg.spectral_max)
        if step % cfg.log_every == 0 or step == cfg.s2_steps - 1:
            log.print("stage2", step, loss_dyn=tot, persistence_baseline=tot_p, spike_rate=_mean(rates_all),
                      grad_norm=gn, sec=time.time() - t0)
    return evaluate_stage2(cfg, plm, val_cache, log)


@torch.no_grad()
def evaluate_stage2(cfg, plm, cache, log, batches=8):
    for m in (plm.text, plm.tok, plm.core, plm.jepa):
        m.eval()
    maxh = max(cfg.jepa_horizons)
    L = cfg.T_long + maxh
    dyn, pers, rates = [], [], []
    g = torch.Generator().manual_seed(0)
    torch.manual_seed(123)
    for _ in range(batches):
        b = sample_window(cache, cfg.s2_batch, L, ["fused", "sensors", "dt", "cmd"], cfg.device)
        targets = jepa_targets(b["fused"], cfg.n_cells)
        with autocast(cfg.device):
            s = plm.tokens(b["fused"][:, :cfg.T_long], b["sensors"][:, :cfg.T_long], b["cmd"])
            h, _, r = plm.core(s, b["dt"][:, :cfg.T_long])
            pred = plm.jepa(h, b["sensors"][:, :cfg.T_long], targets[:, :cfg.T_long])
        ld, lp = jepa_loss(pred, targets, torch.arange(cfg.T_long, device=cfg.device), cfg.jepa_horizons)
        dyn.append(ld.item())
        pers.append(lp.item())
        rates += [x.item() for x in r]
    res = dict(val_jepa=float(np.mean(dyn)), val_persistence=float(np.mean(pers)),
               jepa_vs_persistence=float(np.mean(dyn) / max(1e-8, np.mean(pers))), spike_rate=_mean(rates))
    log.print("stage2_val", 0, **res)
    return res


# %% [markdown]
# ## 14. Stage 3 — Language-conditioned trajectory imitation (action chunks + jerk penalty)

# %%
def chunk_labels(actions, T, H):
    """actions [B, T+H-1, A] -> [B, T, H, A]"""
    return actions.unfold(1, H, 1).permute(0, 1, 3, 2)[:, :T]


def stage3(cfg, plm, cache, val_cache, log):
    dev = cfg.device
    mods = [plm.text, plm.tok, plm.core, plm.head, plm.jepa]
    opt = Optim([p for m in mods for p in m.parameters()], cfg.s3_lr, cfg.s3_steps, cfg)
    maxh = max(cfg.jepa_horizons)
    L = cfg.T + max(cfg.H - 1, maxh)
    for m in mods:
        m.train()
    t0 = time.time()
    for step in range(cfg.s3_steps):
        b = sample_window(cache, cfg.s3_batch, L, ["fused", "sensors", "dt", "actions", "cmd"], dev)
        tgt = chunk_labels(b["actions"][:, :cfg.T + cfg.H - 1], cfg.T, cfg.H)
        if cfg.proprio_dropout > 0:  # velocity + acceleration channels
            drop = (torch.rand(b["sensors"].shape[0], 1, 1, device=dev) < cfg.proprio_dropout).float()
            mask = torch.ones(SENSOR_DIM, device=dev)
            mask[[0, 1, 2, 4, 5, 6]] = 0.0
            b["sensors"] = b["sensors"] * (1 - drop * (1 - mask))
        with autocast(dev):
            out = plm.policy(b["fused"][:, :cfg.T], b["sensors"][:, :cfg.T], b["dt"][:, :cfg.T], b["cmd"])
            pred = plm.jepa(out["h"], b["sensors"][:, :cfg.T], jepa_targets(b["fused"][:, :cfg.T], cfg.n_cells))
        l_traj = traj_loss(out["mu"], tgt, cfg)
        l_jerk = jerk(out["mu"])
        l_nll = gaussian_nll(tgt, out["mu"].detach(), out["ls"])
        l_jepa, _ = jepa_loss(pred, jepa_targets(b["fused"], cfg.n_cells), torch.arange(cfg.T, device=dev), cfg.jepa_horizons)
        l_sp = spike_loss(out["rates"], cfg)
        loss = l_traj + cfg.lam_jerk * l_jerk + cfg.lam_nll * l_nll + cfg.lam_jepa_aux * l_jepa + l_sp
        opt.backward(loss)
        gn = opt.step()
        clamp_spectral(plm.core, cfg.spectral_max)
        if step % cfg.log_every == 0 or step == cfg.s3_steps - 1:
            log.print("stage3", step, loss=loss.item(), l_traj=l_traj.item(), l_jerk=l_jerk.item(),
                      l_nll=l_nll.item(), l_jepa=l_jepa.item(),
                      spike_rate=_mean([r.item() for r in out["rates"]]), grad_norm=gn, sec=time.time() - t0)
    return evaluate_stage3(cfg, plm, val_cache, log)


@torch.no_grad()
def evaluate_stage3(cfg, plm, cache, log, batches=8):
    plm.eval()
    L = cfg.T + cfg.H - 1
    torch.manual_seed(321)
    m = {"val_first_mse": [], "val_chunk_mse": [], "val_mean_baseline_mse": [], "pred_jerk": [], "expert_jerk": [],
         "sigma": []}
    for _ in range(batches):
        b = sample_window(cache, cfg.s3_batch, L, ["fused", "sensors", "dt", "actions", "cmd"], cfg.device)
        tgt = chunk_labels(b["actions"], cfg.T, cfg.H)
        with autocast(cfg.device):
            out = plm.policy(b["fused"][:, :cfg.T], b["sensors"][:, :cfg.T], b["dt"][:, :cfg.T], b["cmd"])
        mu = out["mu"]
        m["val_first_mse"].append(F.mse_loss(mu[:, -1, 0], tgt[:, -1, 0]).item())
        m["val_chunk_mse"].append(F.mse_loss(mu, tgt).item())
        m["val_mean_baseline_mse"].append(F.mse_loss(tgt.mean((0, 1, 2), keepdim=True).expand_as(tgt), tgt).item())
        m["pred_jerk"].append(jerk(mu).item())
        m["expert_jerk"].append(jerk(tgt).item())
        m["sigma"].append(out["ls"].exp().mean().item())
    res = {k: float(np.mean(v)) for k, v in m.items()}
    log.print("stage3_val", 0, **res)
    return res


# %% [markdown]
# ## 15. Runtime: Muscle-memory cache, CBF safety filter, speculative controller, closed-loop evaluation

# %%
class MuscleMemoryCache:
    """Device-resident cosine-NN cache of target-verified action chunks, keyed by the drafter latent + command.
    Replaces the notebook's CPU/NumPy linear scan with one batched similarity, LRU eviction and confidence gating."""

    def __init__(self, dim, H, A, capacity=2048, sim_threshold=0.985, device="cpu"):
        self.keys = torch.zeros(capacity, dim, device=device)
        self.vals = torch.zeros(capacity, H, A, device=device)
        self.cmd = torch.full((capacity,), -1, dtype=torch.long, device=device)
        self.last = torch.zeros(capacity, dtype=torch.long, device=device)
        self.valid = torch.zeros(capacity, dtype=torch.bool, device=device)
        self.thr, self.clock = sim_threshold, 0
        self.hits = self.misses = self.inserts = self.evictions = 0

    @staticmethod
    def cmd_id(tokens):
        return int(hash(tuple(int(t) for t in tokens)) % (2 ** 31))

    def lookup(self, key, cmd_id):
        self.clock += 1
        if not self.valid.any():
            self.misses += 1
            return None
        sims = F.normalize(self.keys, dim=-1) @ F.normalize(key, dim=-1)
        sims[~self.valid | (self.cmd != cmd_id)] = -2.0
        s, i = sims.max(0)
        if s >= self.thr:
            self.hits += 1
            self.last[i] = self.clock
            return self.vals[i]
        self.misses += 1
        return None

    def insert(self, key, chunk, cmd_id):
        free = (~self.valid).nonzero()
        if len(free):
            i = free[0, 0]
        else:
            i = self.last.argmin()
            self.evictions += 1
        self.keys[i], self.vals[i], self.cmd[i] = key, chunk, cmd_id
        self.last[i], self.valid[i] = self.clock, True
        self.inserts += 1

    def stats(self):
        n = self.hits + self.misses
        return dict(hits=self.hits, misses=self.misses, hit_rate=self.hits / max(1, n), size=int(self.valid.sum()),
                    inserts=self.inserts, evictions=self.evictions)


def cbf_filter(sim, a, cfg, alpha=2.0, margin=0.45):
    """Control-barrier-function safety filter on the commanded velocity (single-integrator model).
    h_i = ||p - o_i||² - (r_i + m)²,  enforce  ḣ_i + α h_i >= 0  by sequential half-space projection.
    Uses obstacle states from the simulator (stands in for a LiDAR tracker)."""
    u = _rot_world_to_ego(sim.yaw).T @ (np.clip(a[:3], -1, 1) * cfg.vmax)
    u0 = u.copy()
    for _ in range(2):
        for o, r, vo in zip(sim.obs_p, sim.obs_r, sim.obs_v):
            n = sim.p - o
            h = n @ n - (r + margin) ** 2
            c = 2 * n
            b = -alpha * h + c @ vo
            if c @ u < b:
                u = u + (b - c @ u) / (c @ c + 1e-9) * c
        hz = sim.p[2] - 0.6
        if u[2] < -alpha * hz:
            u[2] = -alpha * hz
    sp = np.linalg.norm(u)
    if sp > cfg.vmax:
        u = u / sp * cfg.vmax
    out = a.copy()
    out[:3] = np.clip(_rot_world_to_ego(sim.yaw) @ u / cfg.vmax, -1, 1)
    return out, bool(np.linalg.norm(u - u0) > 1e-3)


class HardRejectionBuffer:
    def __init__(self, capacity=4096):
        self.items = deque(maxlen=capacity)

    def add(self, **kw):
        self.items.append({k: v.detach().cpu() if torch.is_tensor(v) else v for k, v in kw.items()})

    def __len__(self):
        return len(self.items)

    def sample(self, B, device):
        idx = np.random.randint(len(self.items), size=B)
        rows = [self.items[i] for i in idx]
        return {k: torch.stack([r[k] for r in rows]).to(device) for k in rows[0]}


class PLMRuntime:
    """Closed-loop controller.
    modes: 'expert' | 'target' (full model every tick) | 'target_chunk' (execute k actions of each chunk) |
           'drafter' (edge model only) | 'speculative' (drafter on the critical path, target verifies the last
           k drafted actions in one batched pass, overrides on rejection, adaptive k) """

    def __init__(self, cfg, plm, drafter=None, cache=None, mode="speculative", hard_buffer=None, use_cbf=None):
        self.cfg, self.plm, self.drafter, self.cache, self.mode = cfg, plm, drafter, cache, mode
        self.hard = hard_buffer
        self.use_cbf = cfg.use_cbf if use_cbf is None else use_cbf

    @torch.no_grad()
    def reset(self, sim):
        c = self.cfg
        self.sim = sim
        self.win = deque(maxlen=c.T)
        self.cmd = torch.from_numpy(tokenize(sim.command, c.max_cmd_len))[None].to(c.device)
        self.cmd_id = MuscleMemoryCache.cmd_id(self.cmd[0].tolist())
        self.cmd_emb = self.plm.text(self.cmd)
        self.plan, self.pending = [], []
        self.k = c.verify_every
        self.n = dict(ticks=0, target_calls=0, target_windows=0, drafts=0, verified=0, accepted=0, cache_hits=0,
                      cbf=0, critical_ms=0.0, offpath_ms=0.0)

    def _sync(self):
        if str(self.cfg.device).startswith("cuda"):
            torch.cuda.synchronize()
        elif str(self.cfg.device) == "mps":
            torch.mps.synchronize()

    @torch.no_grad()
    def _perceive(self, ob):
        c, P = self.cfg, self.plm.perception
        li = torch.from_numpy(ob["lidar"])[None].to(c.device)
        sens = torch.from_numpy(ob["sensors"])[None, None].to(c.device)
        dt = torch.tensor([[float(ob["dt"])]], device=c.device)
        self._sync()
        t0 = time.perf_counter()
        with autocast(c.device):
            rad = P.lidar_bev(li)
            zd = self.drafter.latent(rad[:, None], sens, self.cmd)[0, 0] if self.drafter is not None else None
        self._sync()
        t1 = time.perf_counter()
        fr = torch.from_numpy(ob["frame"])[None].to(c.device)
        with autocast(c.device):
            cam = P.camera_tokens(fr)
            scene = P.scene(P.fuse(rad, P.camera_bev(fr, cam)), P.pool_cam(cam))
            s = self.plm.tok(scene[:, None], sens, self.cmd_emb)[0, 0]
        self._sync()
        t2 = time.perf_counter()
        self.win.append(dict(rad=rad[0], sens=sens[0, 0], dt=dt[0, 0], s=s, zd=zd, scene=scene[0]))
        return (t1 - t0) * 1e3, (t2 - t1) * 1e3

    def _stack(self, key, wins=None):
        return torch.stack([w[key] for w in (wins or self.win)])[None]

    @torch.no_grad()
    def _target(self, windows):
        """Batched target over several windows (all same length) -> first action mean/sigma and full chunk of last step."""
        mu_all, sig_all = [None] * len(windows), [None] * len(windows)
        groups = {}
        for i, win in enumerate(windows):  # windows are shorter than T only during the first ticks
            groups.setdefault(len(win), []).append(i)
        for idx in groups.values():
            s = torch.stack([torch.stack([w["s"] for w in windows[i]]) for i in idx])
            dt = torch.stack([torch.stack([w["dt"] for w in windows[i]]) for i in idx])
            with autocast(self.cfg.device):
                out = self.plm.policy_from_tokens(s, dt, last_only=True)
            for j, i in enumerate(idx):
                mu_all[i], sig_all[i] = out["mu"][j, -1], out["ls"][j, -1].exp()
        self.n["target_calls"] += 1
        self.n["target_windows"] += len(windows)
        return torch.stack(mu_all), torch.stack(sig_all)

    @torch.no_grad()
    def _draft(self):
        z = self._stack("zd")
        dt = self._stack("dt")
        with autocast(self.cfg.device):
            out = self.drafter.policy_from_latent(z, dt, last_only=True)
        self.n["drafts"] += 1
        return out["mu"][0, -1], out["ls"][0, -1].exp()

    @torch.no_grad()
    def act(self, ob):
        c = self.cfg
        crit, off = self._perceive(ob)
        self.n["ticks"] += 1
        self._sync()
        t0 = time.perf_counter()
        if self.mode == "expert":
            a = self.sim.expert_raw()
            crit += (time.perf_counter() - t0) * 1e3
        elif self.mode == "target":
            mu, _ = self._target([list(self.win)])
            a = mu[0, 0].float().cpu().numpy()
            self._sync()
            crit += off + (time.perf_counter() - t0) * 1e3
            off = 0.0
        elif self.mode == "target_chunk":
            if not self.plan:
                mu, _ = self._target([list(self.win)])
                self.plan = list(mu[0, : c.verify_every].float().cpu().numpy())
            a = self.plan.pop(0)
            self._sync()
            crit += off + (time.perf_counter() - t0) * 1e3
            off = 0.0
        elif self.mode == "drafter":
            mu, _ = self._draft()
            a = mu[0].float().cpu().numpy()
            self._sync()
            crit += (time.perf_counter() - t0) * 1e3
        else:  # speculative
            key = self.win[-1]["zd"].float()
            hit = self.cache.lookup(key, self.cmd_id) if self.cache is not None else None
            if hit is not None:
                prop = hit[0]
                self.n["cache_hits"] += 1
            else:
                prop = self._draft()[0][0]
            a = prop.float().cpu().numpy()
            self._sync()
            crit += (time.perf_counter() - t0) * 1e3
            self.pending.append((list(self.win), prop, key))
            if len(self.pending) >= self.k:  # off the critical path in a real deployment
                t1 = time.perf_counter()
                mu, sig = self._target([w for w, _, _ in self.pending])
                props = torch.stack([p for _, p, _ in self.pending]).to(mu.dtype)
                ok = ((props - mu[:, 0]).abs() <= c.accept_tol + c.accept_z * sig[:, 0]).all(-1)
                self.n["verified"] += len(ok)
                self.n["accepted"] += int(ok.sum())
                for (win, p, k), good, m_i, s_i in zip(self.pending, ok.tolist(), mu, sig):
                    if good and self.cache is not None and float(s_i.mean()) < c.cache_max_sigma:
                        self.cache.insert(k, m_i, self.cmd_id)
                    if not good and self.hard is not None and len(win) == c.T:
                        self.hard.add(rad=self._stack("rad", win)[0], sens=self._stack("sens", win)[0],
                                      dt=self._stack("dt", win)[0], cmd=self.cmd[0], mu=m_i, sig=s_i)
                if not bool(ok[-1]):
                    a = mu[-1, 0].float().cpu().numpy()  # override the current tick with the verified action
                self.k = 1 if not bool(ok.all()) else min(c.verify_every, self.k * 2)
                self.pending = []
                self._sync()
                off += (time.perf_counter() - t1) * 1e3
        if self.use_cbf and self.mode != "expert":
            a, hit = cbf_filter(self.sim, np.asarray(a, dtype=np.float64), c)
            self.n["cbf"] += int(hit)
        self.n["critical_ms"] += crit
        self.n["offpath_ms"] += off
        return np.asarray(a, dtype=np.float64)


@torch.no_grad()
def run_episode(cfg, runtime, seed, intent=None, ticks=None):
    sim = DroneSim(cfg, seed, intent)
    runtime.reset(sim)
    acts = []
    for _ in range(ticks or cfg.episode_ticks):
        ob = sim.observe(render=True)
        a = runtime.act(ob)
        acts.append(a)
        sim.step(a, sim.sample_dt())
        if sim.collided or (sim.intent in GOAL_INTENTS and sim.goal_dist() < 1.2):
            break
    A = np.array(acts)
    j = float(np.mean(np.sum((A[2:] - 2 * A[1:-1] + A[:-2]) ** 2, -1))) if len(A) > 2 else 0.0
    return dict(success=sim.success(), collided=sim.collided, goal=sim.intent in GOAL_INTENTS,
                final_dist=sim.goal_dist(), jerk=j, **runtime.n)


def evaluate_closed_loop(cfg, plm, drafter, modes, log, n=None, seed_base=5_000_000, cache=None, tag="eval"):
    plm.eval()
    if drafter is not None:
        drafter.eval()
    n = n or cfg.eval_episodes
    res = {}
    for mode in modes:
        rt = PLMRuntime(cfg, plm, drafter, cache if mode == "speculative" else None, mode)
        eps = [run_episode(cfg, rt, seed_base + i, intent=GOAL_INTENTS[i % 2] if i % 3 else None) for i in range(n)]
        goal = [e for e in eps if e["goal"]]
        ticks = sum(e["ticks"] for e in eps)
        r = dict(success_rate=float(np.mean([e["success"] for e in goal])) if goal else float("nan"),
                 collision_rate=float(np.mean([e["collided"] for e in eps])),
                 final_goal_dist=float(np.nanmean([e["final_dist"] for e in goal])) if goal else float("nan"),
                 action_jerk=float(np.mean([e["jerk"] for e in eps])),
                 target_passes_per_tick=sum(e["target_calls"] for e in eps) / ticks,
                 target_windows_per_tick=sum(e["target_windows"] for e in eps) / ticks,
                 acceptance_rate=(sum(e["accepted"] for e in eps) / max(1, sum(e["verified"] for e in eps)))
                 if mode == "speculative" else float("nan"),
                 cache_hit_rate=sum(e["cache_hits"] for e in eps) / ticks,
                 cbf_intervention_rate=sum(e["cbf"] for e in eps) / ticks,
                 critical_path_ms=sum(e["critical_ms"] for e in eps) / ticks,
                 offpath_ms=sum(e["offpath_ms"] for e in eps) / ticks)
        res[mode] = r
        log.print(f"{tag}_{mode}", 0, **r)
    return res


# %% [markdown]
# ## 15b. Stage 3b — DAgger: expert labels on the states the learned policy actually visits

# %%
@torch.no_grad()
def dagger_collect(cfg, plm, n_episodes, seed_base):
    """Fly the current policy (expert executed with prob. β), label every visited state with the raw expert command,
    and cut the flights into cache-format windows. Perception outputs come from the runtime (no re-encoding)."""
    plm.eval()
    L = cfg.episode_len
    rt = PLMRuntime(cfg, plm, None, None, "target")
    rows = {k: [] for k in ["fused", "rad", "sensors", "dt", "actions", "cmd", "intent"]}
    rng = np.random.default_rng(seed_base)
    for i in range(n_episodes):
        sim = DroneSim(cfg, seed_base + i)
        rt.reset(sim)
        ep = {k: [] for k in ["fused", "rad", "sensors", "dt", "actions"]}
        for _ in range(cfg.episode_ticks):
            ob = sim.observe(render=True)
            a_pol = rt.act(ob)
            a_exp = sim.expert_raw()
            w = rt.win[-1]
            ep["fused"].append(w["scene"].half().cpu())
            ep["rad"].append(w["rad"].half().cpu())
            ep["sensors"].append(torch.from_numpy(ob["sensors"]))
            ep["dt"].append(torch.tensor(float(ob["dt"])))
            ep["actions"].append(torch.from_numpy(a_exp.astype(np.float32)))
            sim.step(a_exp if rng.random() < cfg.dagger_beta else a_pol, sim.sample_dt())
            if sim.collided or (sim.intent in GOAL_INTENTS and sim.goal_dist() < 1.2):
                break
        n = len(ep["dt"])
        for t0 in range(0, n - L + 1, max(1, L // 2)):
            for k in ep:
                rows[k].append(torch.stack(ep[k][t0:t0 + L]))
            rows["cmd"].append(torch.from_numpy(tokenize(sim.command, cfg.max_cmd_len)))
            rows["intent"].append(torch.tensor(sim.intent))
    return {k: torch.stack(v) for k, v in rows.items()} if rows["dt"] else None


def dagger(cfg, plm, cache, val_cache, log):
    hist = []
    for rnd in range(cfg.dagger_rounds):
        new = dagger_collect(cfg, plm, cfg.dagger_episodes, 6_000_000 + 1000 * rnd)
        if new is not None:
            for k in cache:
                cache[k] = torch.cat([cache[k], new[k].to(cache[k].dtype)])
        n_new = 0 if new is None else len(new["dt"])
        steps = cfg.s3_steps
        cfg.s3_steps = cfg.dagger_steps
        res = stage3(cfg, plm, cache, val_cache, log)
        cfg.s3_steps = steps
        log.print("dagger", rnd, new_windows=n_new, cache_windows=len(cache["dt"]), val_chunk_mse=res["val_chunk_mse"])
        hist.append(dict(round=rnd, new_windows=n_new, **res))
    return hist


# %% [markdown]
# ## 16. Stage 4 — Speculative co-training (distillation + hard-rejection replay) and GRPO fine-tuning

# %%
def distill_step(cfg, plm, drafter, opt, b, tgt=None):
    """L_spec_align = KL(p_θ || q_φ) + β * hinge(||û_draft - u*_target|| - ρ_safe)   (hinge = differentiable
    surrogate of the indicator; the indicator rate is reported as `unsafe_rate`)."""
    dev = cfg.device
    if tgt is None:
        with torch.no_grad(), autocast(dev):
            t = plm.policy(b["fused"], b["sensors"], b["dt"], b["cmd"])
        mu_p, ls_p = t["mu"], t["ls"]
    else:
        mu_p, ls_p = tgt
    with autocast(dev):
        d = drafter(b["rad"], b["sensors"], b["dt"], b["cmd"])
    mu_q, ls_q = d["mu"], d["ls"]
    if mu_p.dim() == 3:  # hard-buffer rows: only the last step is supervised
        mu_q, ls_q = mu_q[:, -1], ls_q[:, -1]
    kl = gaussian_kl(mu_p, ls_p, mu_q, ls_q)
    err = (mu_q - mu_p).norm(dim=-1)
    hinge = F.relu(err - cfg.rho_safe).mean()
    loss = kl + cfg.beta_safe * hinge + spike_loss(d["rates"], cfg)
    opt.backward(loss)
    opt.step()
    clamp_spectral(drafter.core, cfg.spectral_max)
    return dict(loss=loss.item(), kl=kl.item(), unsafe_rate=(err > cfg.rho_safe).float().mean().item())


@torch.no_grad()
def offline_acceptance(cfg, plm, drafter, cache, batches=8):
    plm.eval()
    drafter.eval()
    acc = []
    torch.manual_seed(7)
    for _ in range(batches):
        b = sample_window(cache, cfg.s4_batch, cfg.T, ["fused", "rad", "sensors", "dt", "cmd"], cfg.device)
        with autocast(cfg.device):
            t = plm.policy(b["fused"], b["sensors"], b["dt"], b["cmd"])
            d = drafter(b["rad"], b["sensors"], b["dt"], b["cmd"])
        ok = ((d["mu"][:, -1, 0] - t["mu"][:, -1, 0]).abs() <= cfg.accept_tol + cfg.accept_z * t["ls"][:, -1, 0].exp()).all(-1)
        acc.append(ok.float().mean().item())
    return float(np.mean(acc))


def stage4(cfg, plm, drafter, cache, val_cache, log):
    dev = cfg.device
    plm.eval()
    for p in plm.parameters():
        p.requires_grad = False
    drafter.train()
    total = cfg.s4_distill_steps + cfg.s4_rounds * cfg.s4_replay_steps
    opt = Optim(drafter.parameters(), cfg.s4_lr, total, cfg)
    keys = ["fused", "rad", "sensors", "dt", "cmd"]
    t0 = time.time()
    for step in range(cfg.s4_distill_steps):
        b = sample_window(cache, cfg.s4_batch, cfg.T, keys, dev)
        st = distill_step(cfg, plm, drafter, opt, b)
        if step % cfg.log_every == 0 or step == cfg.s4_distill_steps - 1:
            log.print("stage4_distill", step, **st, sec=time.time() - t0)
    res = dict(offline_acceptance_after_distill=offline_acceptance(cfg, plm, drafter, val_cache))
    log.print("stage4_val", 0, **res)
    hard = HardRejectionBuffer()
    mem = MuscleMemoryCache(cfg.d_draft, cfg.H, ACTION_DIM, cfg.cache_capacity, cfg.cache_sim, dev)
    rounds = []
    for rnd in range(cfg.s4_rounds):
        drafter.eval()
        rt = PLMRuntime(cfg, plm, drafter, mem, "speculative", hard_buffer=hard)
        eps = [run_episode(cfg, rt, 7_000_000 + rnd * 1000 + i) for i in range(cfg.s4_rollouts_per_round)]
        alpha = sum(e["accepted"] for e in eps) / max(1, sum(e["verified"] for e in eps))
        drafter.train()
        for step in range(cfg.s4_replay_steps):
            half = cfg.s4_batch // 2
            b = sample_window(cache, cfg.s4_batch - (half if len(hard) >= half else 0), cfg.T, keys, dev)
            distill_step(cfg, plm, drafter, opt, b)
            if len(hard) >= half:
                h = hard.sample(half, dev)
                hb = dict(rad=h["rad"], sensors=h["sens"], dt=h["dt"], cmd=h["cmd"])
                st = distill_step(cfg, plm, drafter, opt, hb, tgt=(h["mu"], h["sig"].clamp(min=1e-3).log()))
        rounds.append(dict(round=rnd, rollout_acceptance=alpha, hard_buffer=len(hard)))
        log.print("stage4_round", rnd, rollout_acceptance=alpha, hard_buffer=len(hard),
                  offline_acceptance=offline_acceptance(cfg, plm, drafter, val_cache))
    res["rounds"] = rounds
    res["offline_acceptance_final"] = offline_acceptance(cfg, plm, drafter, val_cache)
    return res, mem


def grpo_finetune(cfg, plm, log):
    """Group-relative policy optimisation on action chunks, scored by open-loop simulator rollouts from real
    closed-loop states: r = goal progress - 5·collision - 0.5·jerk. KL-regularised to the BC reference."""
    if cfg.grpo_iters <= 0:
        return {}
    dev = cfg.device
    ref = copy.deepcopy(plm).eval()
    for p in ref.parameters():
        p.requires_grad = False
    train_mods = [plm.core, plm.head]
    for m in train_mods:
        for p in m.parameters():
            p.requires_grad = True
    opt = Optim([p for m in train_mods for p in m.parameters()], cfg.grpo_lr, cfg.grpo_iters, cfg)
    hist = []
    for it in range(cfg.grpo_iters):
        plm.eval()
        rt = PLMRuntime(cfg, plm, None, None, "target_chunk", use_cbf=False)
        sim = DroneSim(cfg, 8_000_000 + it, intent=GOAL_INTENTS[it % 2])
        rt.reset(sim)
        samples = []
        for tick in range(cfg.episode_ticks):
            ob = sim.observe(render=True)
            a = rt.act(ob)
            if tick % 5 == 4 and len(rt.win) == cfg.T:
                samples.append((copy.deepcopy(sim), torch.stack([w["s"] for w in rt.win]).float(),
                                torch.stack([w["dt"] for w in rt.win]).float()))
            sim.step(a, sim.sample_dt())
            if sim.collided or sim.goal_dist() < 1.2:
                break
        if not samples:
            continue
        s = torch.stack([x[1] for x in samples]).to(dev)
        dt = torch.stack([x[2] for x in samples]).to(dev)
        with torch.no_grad():
            o = plm.policy_from_tokens(s, dt)
            mu, ls = o["mu"][:, -1], o["ls"][:, -1]
            G = cfg.grpo_group
            acts = (mu[:, None] + ls.exp()[:, None] * torch.randn(len(samples), G, *mu.shape[1:], device=dev)).clamp(-1, 1)
        rewards = torch.zeros(len(samples), G)
        for i, (snap, _, _) in enumerate(samples):
            for g in range(G):
                sm = copy.deepcopy(snap)
                d0 = sm.goal_dist()
                A = acts[i, g].cpu().numpy()
                for a in A:
                    sm.step(a, sm.sample_dt())
                    if sm.collided:
                        break
                j = float(np.sum((A[2:] - 2 * A[1:-1] + A[:-2]) ** 2))
                rewards[i, g] = (d0 - sm.goal_dist()) - 5.0 * sm.collided - 0.5 * j
        adv = ((rewards - rewards.mean(1, keepdim=True)) / (rewards.std(1, keepdim=True) + 1e-6)).to(dev)
        plm.train()
        o = plm.policy_from_tokens(s, dt)
        mu_n, ls_n = o["mu"][:, -1], o["ls"][:, -1]
        logp = gaussian_logp(acts, mu_n[:, None], ls_n[:, None])
        with torch.no_grad():
            r_o = ref.policy_from_tokens(s, dt)
        kl = gaussian_kl(mu_n, ls_n, r_o["mu"][:, -1], r_o["ls"][:, -1])
        loss = -(adv * logp).mean() / (cfg.H * ACTION_DIM) + cfg.grpo_kl * kl
        opt.backward(loss)
        opt.step()
        clamp_spectral(plm.core, cfg.spectral_max)
        hist.append(float(rewards.mean()))
        log.print("grpo", it, mean_reward=float(rewards.mean()), kl=kl.item(), n_states=len(samples))
    plm.eval()
    for p in plm.parameters():
        p.requires_grad = False
    return dict(reward_first=hist[0] if hist else None, reward_last=hist[-1] if hist else None)


# %% [markdown]
# ## 17. Checkpointing & Hugging Face endpoint export

# %%
HANDLER_SRC = r'''"""Hugging Face Inference Endpoint handler for PLM v2 (generated by plm.py)."""
import json, os
import numpy as np
import torch
from plm import Config, PLM, EdgeDrafter, tokenize, ACTION_NAMES


class EndpointHandler:
    def __init__(self, path=""):
        with open(os.path.join(path, "plm_config.json")) as f:
            cfg = json.load(f)
        cfg["device"] = "cuda" if torch.cuda.is_available() else "cpu"
        cfg["pretrained"] = False
        tup = lambda v: tuple(tup(x) for x in v) if isinstance(v, list) else v
        self.cfg = Config(**{k: tup(v) for k, v in cfg.items()})
        self.plm = PLM(self.cfg).to(self.cfg.device).eval()
        self.plm.load_state_dict(torch.load(os.path.join(path, "plm_target.pt"), map_location=self.cfg.device))

    @torch.no_grad()
    def __call__(self, data):
        """data = {"inputs": {"frames": [T,3,S,S] floats in [0,1], "lidar": [T,2,Z,Y,X], "sensors": [T,9],
                              "dt": [T], "command": "fly to the red beacon"}}"""
        x = data.get("inputs", data)
        c, dev = self.cfg, self.cfg.device
        fr = torch.tensor(np.asarray(x["frames"]), dtype=torch.float32, device=dev)
        li = torch.tensor(np.asarray(x["lidar"]), dtype=torch.float32, device=dev)
        se = torch.tensor(np.asarray(x["sensors"]), dtype=torch.float32, device=dev)[None]
        dt = torch.tensor(np.asarray(x["dt"]), dtype=torch.float32, device=dev)[None]
        cmd = torch.from_numpy(tokenize(x["command"], c.max_cmd_len))[None].to(dev)
        fused = self.plm.perception(fr, li)["scene"][None]
        out = self.plm.policy(fused, se, dt, cmd)
        mu, sig = out["mu"][0, -1].cpu().numpy(), out["ls"][0, -1].exp().cpu().numpy()
        scale = np.array([c.vmax, c.vmax, c.vmax, c.rmax])
        first = {n: float(v) for n, v in zip(ACTION_NAMES, mu[0] * scale)}
        return {"action_chunk_normalised": mu.tolist(), "sigma": sig.tolist(), "first_action_si": first,
                "units": {"v_forward": "m/s", "v_left": "m/s", "v_up": "m/s", "yaw_rate": "rad/s"}}
'''


def export(cfg, plm, drafter, out_dir):
    d = os.path.join(out_dir, "export")
    os.makedirs(d, exist_ok=True)
    torch.save(plm.state_dict(), os.path.join(d, "plm_target.pt"))
    if drafter is not None:
        torch.save(drafter.state_dict(), os.path.join(d, "plm_drafter.pt"))
    with open(os.path.join(d, "plm_config.json"), "w") as f:
        json.dump(asdict(cfg), f, indent=1)
    try:
        shutil.copy(os.path.abspath(__file__), os.path.join(d, "plm.py"))
    except NameError:  # running inside a notebook
        print("[info] copy plm.py next to handler.py manually when exporting from a notebook")
    with open(os.path.join(d, "handler.py"), "w") as f:
        f.write(HANDLER_SRC)
    with open(os.path.join(d, "requirements.txt"), "w") as f:
        f.write("torch\ntimm\nnumpy\n")
    print(f"exported to {d}")
    return d


# %% [markdown]
# ## 18. Orchestration

# %%
def run_all(cfg: Config):
    set_seed(cfg.seed)
    os.makedirs(cfg.out_dir, exist_ok=True)
    log = Logger(cfg.out_dir)
    dev = cfg.device
    print(f"device={dev} amp={amp_dtype(dev)}")
    plm = PLM(cfg).to(dev)
    drafter = EdgeDrafter(cfg).to(dev)
    report = dict(config=asdict(cfg),
                  params=dict(target_total=count_params(plm), target_trainable=count_params(plm, True),
                              vit=count_params(plm.perception.cam.vit), drafter=count_params(drafter)))
    print(json.dumps(report["params"]))

    if cfg.perception_ckpt:
        print(f"\n=== Stage 1: loading perception from {cfg.perception_ckpt} ===")
        plm.perception.load_state_dict(torch.load(cfg.perception_ckpt, map_location=dev))
        report["stage1"] = evaluate_stage1(cfg, plm, log)
    else:
        print("\n=== Stage 1: modality pre-alignment & BEV grounding ===")
        report["stage1"] = stage1(cfg, plm, log)
        torch.save(plm.perception.state_dict(), os.path.join(cfg.out_dir, "perception.pt"))

    print("\n=== Building frozen-perception feature cache ===")
    cdir = cfg.cache_dir or cfg.out_dir
    cache = build_cache(cfg, plm, cfg.n_cache_episodes, 2_000_000, os.path.join(cdir, "cache_train.pt"))
    val_cache = build_cache(cfg, plm, cfg.n_val_episodes, 3_000_000, os.path.join(cdir, "cache_val.pt"))

    print("\n=== Stage 2: continuous-time world model (JEPA + spike regularisation, TBPTT) ===")
    if cfg.skip_stage2:
        for p in plm.perception.parameters():
            p.requires_grad = False
        report["stage2"] = evaluate_stage2(cfg, plm, val_cache, log)
    else:
        report["stage2"] = stage2(cfg, plm, cache, val_cache, log)

    print("\n=== Stage 3: language-conditioned action-chunk imitation ===")
    report["stage3"] = stage3(cfg, plm, cache, val_cache, log)
    if cfg.dagger_rounds > 0:
        print("\n=== Stage 3b: DAgger (on-policy states, expert labels) ===")
        report["dagger"] = dagger(cfg, plm, cache, val_cache, log)
    report["closed_loop_bc"] = evaluate_closed_loop(cfg, plm, None, ["expert", "target", "target_chunk"], log,
                                                    tag="eval_bc")

    if cfg.stop_after_stage3:
        torch.save(plm.state_dict(), os.path.join(cfg.out_dir, "plm_stage3.pt"))
        with open(os.path.join(cfg.out_dir, "plm_config.json"), "w") as f:
            json.dump(asdict(cfg), f, indent=1)
        log.save()
        with open(os.path.join(cfg.out_dir, "report.json"), "w") as f:
            json.dump(report, f, indent=1, default=str)
        return plm, drafter, report

    print("\n=== Stage 4: speculative co-training ===")
    report["stage4"], mem = stage4(cfg, plm, drafter, cache, val_cache, log)

    if cfg.grpo_iters > 0:
        print("\n=== Stage 4b: GRPO fine-tuning of the target ===")
        report["grpo"] = grpo_finetune(cfg, plm, log)

    print("\n=== Final closed-loop evaluation ===")
    mem_eval = MuscleMemoryCache(cfg.d_draft, cfg.H, ACTION_DIM, cfg.cache_capacity, cfg.cache_sim, dev)
    report["closed_loop_final"] = evaluate_closed_loop(
        cfg, plm, drafter, ["expert", "target", "target_chunk", "drafter", "speculative"], log, cache=mem_eval,
        tag="eval_final")
    report["cache_stats"] = mem_eval.stats()
    export(cfg, plm, drafter, cfg.out_dir)
    log.save()
    with open(os.path.join(cfg.out_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=1, default=str)
    print_summary(report)
    return plm, drafter, report


def print_summary(report):
    print("\n================ SUMMARY ================")
    s1 = report["stage1"]
    print(f"Stage 1  contrastive top-1 {s1['contra_top1']:.3f} | camera→BEV occupancy IoU {s1['cam_occ_iou']:.3f} | "
          f"Doppler MAE {s1['vel_mae']:.3f} (zero baseline {s1['vel_mae_zero_baseline']:.3f})")
    s2 = report["stage2"]
    print(f"Stage 2  JEPA loss {s2['val_jepa']:.4f} vs persistence {s2['val_persistence']:.4f} "
          f"(ratio {s2['jepa_vs_persistence']:.3f}) | spike rate {s2['spike_rate']:.3f}")
    s3 = report["stage3"]
    print(f"Stage 3  chunk MSE {s3['val_chunk_mse']:.4f} (mean baseline {s3['val_mean_baseline_mse']:.4f}) | "
          f"jerk pred {s3['pred_jerk']:.5f} vs expert {s3['expert_jerk']:.5f}")
    s4 = report["stage4"]
    print(f"Stage 4  offline acceptance {s4['offline_acceptance_after_distill']:.3f} -> "
          f"{s4['offline_acceptance_final']:.3f}")
    print(f"\n{'mode':14s} {'success':>8s} {'collide':>8s} {'accept':>7s} {'tgt/tick':>9s} {'crit ms':>8s} {'off ms':>7s}")
    for m, r in report["closed_loop_final"].items():
        print(f"{m:14s} {r['success_rate']:8.2f} {r['collision_rate']:8.2f} {r['acceptance_rate']:7.2f} "
              f"{r['target_passes_per_tick']:9.2f} {r['critical_path_ms']:8.1f} {r['offpath_ms']:7.1f}")


def _in_notebook():
    try:
        return get_ipython().__class__.__name__ == "ZMQInteractiveShell"  # noqa: F821
    except NameError:
        return False


if __name__ == "__main__" and not _in_notebook():
    ap = argparse.ArgumentParser(description="PLM v2: 4-stage training, speculative runtime, export")
    ap.add_argument("--mode", choices=["quick", "small", "full"], default="quick")
    ap.add_argument("--out", default=None)
    ap.add_argument("--set", nargs="*", default=[], help="override config fields, e.g. --set s3_steps=10000 T=12")
    args = ap.parse_args()
    cfg = {"quick": Config.quick, "small": Config.small, "full": Config}[args.mode]()
    types = {f.name: f.type for f in fields(Config)}
    for kv in args.set:
        k, v = kv.split("=", 1)
        cur = getattr(cfg, k)
        if isinstance(cur, str):
            setattr(cfg, k, v)
        else:
            val = json.loads(v)
            setattr(cfg, k, tuple(val) if isinstance(cur, tuple) else type(cur)(val))
    if args.out:
        cfg.out_dir = args.out
    run_all(cfg)
