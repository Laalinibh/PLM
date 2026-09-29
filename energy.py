"""Operation counts and energy estimate for PLM (per control tick).

Counts multiply-accumulates (MACs) of every matmul / conv with torch's FlopCounterMode (elementwise ops such as the
LIF/LTC updates, softmax and norms are excluded, as is standard). The spike-driven part of the core — Q·K^T and A·V with
binary spike operands — is re-expressed as accumulates (ACs) at the *measured* firing rates.
Energy uses the 45 nm CMOS figures of Horowitz (ISSCC 2014) used throughout the spiking-transformer literature:
E_MAC = 4.6 pJ (32-bit FP multiply + add), E_AC = 0.9 pJ (32-bit FP add).

Usage: python energy.py [run_dir]    -> writes <run_dir>/energy.json
"""
import json
import os
import sys

import numpy as np
import torch
from torch.utils.flop_counter import FlopCounterMode

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plm import PLM, Config, DroneSim, EdgeDrafter, sample_window  # noqa: E402

E_MAC, E_AC = 4.6e-12, 0.9e-12
run = sys.argv[1] if len(sys.argv) > 1 else "runs/plm_small"
exp = os.path.join(run, "export")
has_export = os.path.exists(os.path.join(exp, "plm_target.pt"))
raw = json.load(open(os.path.join(exp if has_export else run, "plm_config.json")))
tup = lambda v: tuple(tup(x) for x in v) if isinstance(v, list) else v
cfg = Config(**{k: tup(v) for k, v in raw.items() if k in Config.__dataclass_fields__})
cfg.device, cfg.pretrained = "cpu", False
torch.set_num_threads(2)
plm = PLM(cfg).eval()
plm.load_state_dict(torch.load(os.path.join(exp, "plm_target.pt") if has_export else os.path.join(run, "plm_stage3.pt"),
                               map_location="cpu"))
drafter = EdgeDrafter(cfg).eval()
if has_export:
    drafter.load_state_dict(torch.load(os.path.join(exp, "plm_drafter.pt"), map_location="cpu"))
for p in list(plm.parameters()) + list(drafter.parameters()):
    p.requires_grad_(False)


def macs(fn):
    with FlopCounterMode(display=False) as fc:
        with torch.no_grad():
            fn()
    return fc.get_total_flops() / 2.0


# ---------------- one observation ----------------
sim = DroneSim(cfg, 123, intent=0)
ob = sim.observe(render=True)
fr = torch.from_numpy(ob["frame"])[None]
li = torch.from_numpy(ob["lidar"])[None]
sens = torch.from_numpy(ob["sensors"])[None, None]
from plm import tokenize  # noqa: E402
cmd = torch.from_numpy(tokenize(sim.command, cfg.max_cmd_len))[None]
P = plm.perception
with torch.no_grad():
    rad = P.lidar_bev(li)
    cam = P.camera_tokens(fr)
    scene = P.scene(P.fuse(rad, P.camera_bev(fr, cam)), P.pool_cam(cam))
    cmd_emb = plm.text(cmd)
T = cfg.T
s_win = torch.randn(1, T, cfg.d_model)
dt_win = torch.full((1, T), 0.055)
z_win = torch.randn(1, T, cfg.d_draft)

comp = {
    "lidar_encoder": macs(lambda: P.lidar_bev(li)),
    "camera_vit": macs(lambda: P.camera_tokens(fr)),
    "bev_lifter_fusion": macs(lambda: P.fuse(rad, P.camera_bev(fr, cam))) - 0.0,
    "state_tokenizer": macs(lambda: plm.tok(scene[:, None], sens, cmd_emb)),
    "target_core(window T)": macs(lambda: plm.core(s_win, dt_win)),
    "target_head(last step)": macs(lambda: plm.head(s_win[:, -1:])),
    "drafter_latent": macs(lambda: drafter.latent(rad[:, None], sens, cmd)),
    "drafter_core_head(window T)": macs(lambda: drafter.policy_from_latent(z_win, dt_win, last_only=True)),
}
# camera_bev(fr, cam) reuses cam tokens, so lifter+fusion MACs exclude the ViT.

# ---------------- measured firing rates (validation windows) ----------------
cache = torch.load(os.path.join(cfg.cache_dir or run, "cache_val.pt"))
torch.manual_seed(0)
rates = {}
with torch.no_grad():
    for _ in range(8):
        b = sample_window(cache, 32, T, ["fused", "sensors", "dt", "cmd"], "cpu")
        out = plm.policy(b["fused"], b["sensors"], b["dt"], b["cmd"])
        for i, r in enumerate(out["rates"]):
            rates.setdefault(i, []).append(r.item())
names = []
for layer in range(cfg.n_layers):
    names += ([f"L{layer}.in"] if cfg.spike_inputs else []) + [f"L{layer}.Q", f"L{layer}.K"] \
        + ([f"L{layer}.V"] if cfg.spike_v else []) + ([f"L{layer}.ltc_in"] if cfg.spike_inputs else [])
rate = {names[i]: float(np.mean(v)) for i, v in rates.items()}

# ---------------- spike-driven attention matmuls ----------------
d = cfg.d_model
qk_ann = av_ann = T * T * d          # per layer, all heads, one window
ann_attn, snn_attn_ac = 0.0, 0.0
for layer in range(cfg.n_layers):
    rq, rv = rate[f"L{layer}.Q"], rate.get(f"L{layer}.V", 1.0)
    ann_attn += qk_ann + av_ann
    snn_attn_ac += qk_ann * rq + av_ann * rv   # binary operand -> accumulate only where a spike is present
# spike-driven projections (spike_inputs): W_qkv and the LTC's W_x see binary inputs
proj_ann, proj_ac = 0.0, 0.0
if cfg.spike_inputs:
    for layer in range(cfg.n_layers):
        qkv, wx = T * d * 3 * d, T * d * cfg.ltc_hidden
        proj_ann += qkv + wx
        proj_ac += qkv * rate[f"L{layer}.in"] + wx * rate[f"L{layer}.ltc_in"]
core = comp["target_core(window T)"]          # the liquid-spiking layers only
core_dense = core - ann_attn - proj_ann
mean_rate = float(np.mean(list(rate.values())))

E = lambda m: m * E_MAC
res = dict(
    config=dict(T=T, d_model=d, n_layers=cfg.n_layers, img_size=cfg.img_size, bev_grid=cfg.bev_grid),
    energy_constants_pJ=dict(MAC=4.6, AC=0.9, source="Horowitz, ISSCC 2014, 45 nm, 32-bit FP"),
    macs_per_tick=comp,
    firing_rates=rate, mean_firing_rate=mean_rate,
    core=dict(total_macs_ann=core, attention_matmul_macs=ann_attn, attention_fraction=ann_attn / core,
              spike_driven_attention_acs=snn_attn_ac,
              energy_ann_uJ=E(core) * 1e6,
              spike_driven_fraction_of_core=(ann_attn + proj_ann) / core,
              energy_spiking_as_built_uJ=(E(core_dense) + (snn_attn_ac + proj_ac) * E_AC) * 1e6,
              energy_fully_spike_driven_bound_uJ=(core * mean_rate * E_AC) * 1e6),
)
c = res["core"]
c["saving_as_built"] = 1 - c["energy_spiking_as_built_uJ"] / c["energy_ann_uJ"]
c["saving_fully_spike_driven_bound"] = 1 - c["energy_fully_spike_driven_bound_uJ"] / c["energy_ann_uJ"]

# ---------------- per-tick energy by controller (ANN-equivalent MAC energy, µJ) ----------------
perc_full = comp["lidar_encoder"] + comp["camera_vit"] + comp["bev_lifter_fusion"] + comp["state_tokenizer"]
head = comp["target_head(last step)"]
target_tick = perc_full + core + head
draft_crit = comp["lidar_encoder"] + comp["drafter_latent"] + comp["drafter_core_head(window T)"]
spec_total = draft_crit + comp["camera_vit"] + comp["bev_lifter_fusion"] + comp["state_tokenizer"] + core + head
res["per_tick_uJ"] = {
    "target_every_tick": E(target_tick) * 1e6,
    "drafter_only": E(draft_crit) * 1e6,
    "verify_behind_critical_path": E(draft_crit) * 1e6,
    "verify_behind_total": E(spec_total) * 1e6,
    "camera_vit_share_of_target": comp["camera_vit"] / target_tick,
}
json.dump(res, open(os.path.join(run, "energy.json"), "w"), indent=1)
print(json.dumps(res, indent=1))
