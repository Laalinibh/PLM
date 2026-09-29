# PLM v2: a Perceptive Language Model for neuromorphic drone control

PLM maps **camera + LiDAR/4D-radar + IMU + a natural-language command** to a **short, smooth chunk of flight commands**. It uses a neuromorphic core: leaky integrate-and-fire (LIF) spiking attention and liquid time-constant (LTC) continuous-time recurrence. The model is trained with a **four-stage curriculum**. At run time a cheap "edge drafter" flies the drone, and the full model verifies the drafter's actions in batches, off the critical path.

This repository is a rebuild of the original `plmfinalwithmem.ipynb` notebook. It follows the training and systems design worked out in the design conversation (`plm.pdf`: *"now get into its training"* and the follow-ups on GPU systems, speculative decoding and robotics safety). Everything lives in a single file, `plm.py`, split into notebook cells with `# %%` markers. `PLM_v2.ipynb` is the same code as a Colab notebook.

```
python plm.py --mode quick          # CPU smoke test, ~3 min
python plm.py --mode small          # CPU-scale research run (numbers in the paper draft), ~1-2 h on 2 cores
python plm.py --mode full           # GPU run: pretrained ViT, 224 px, full-flight coverage + DAgger (CUDA or Apple MPS)
python plm.py --mode small --set spiking=false core_ff=mlp   # ablations: any Config field can be overridden
```

---

## 1. What changed from the original notebook

| Area | Original notebook | PLM v2 |
|---|---|---|
| Data | `torch.randn` frames/LiDAR/sensors; actions = `tanh(sensors)`, so nothing visual was learnable | Procedural drone simulator. Rendered FPV camera, LiDAR voxels with a Doppler channel, IMU, moving obstacles, two beacons, 7 language intents with paraphrases, a potential-field expert, non-uniform Δt, DART-style state perturbation |
| Stage 1 contrastive | `InfoNCE(embeddings, embeddings)`. The positive is the query itself, so the loss is trivial | Cell-level radar↔camera InfoNCE in a shared BEV grid, plus camera→occupancy (metric depth) and a Doppler-velocity loss |
| Fusion | Concatenate three pooled vectors | Camera patch tokens lifted into a metric BEV grid by cross-attention, fused with LiDAR pillars, then pooled by **command-conditioned** cross-attention |
| Spiking neuron | One LIF step per forward pass, so the membrane never integrates over time; triangular surrogate | LIF unrolled over time with **Δt-dependent leak** β = exp(−Δt/τ), soft reset, fast-sigmoid surrogate 1/(1+k\|z\|)², spike-rate regularisation |
| LTC | `state + (cand − state)/tau` applied to a *stack of layers* rather than across time; no Δt | **Closed-form LTC across time** with real non-uniform Δt, per-unit reversal potentials, spectral clamp on the recurrent matrix |
| Action head | Linear, one action per step | CNN decoder emitting an **H-step Gaussian action chunk** (mean, log-σ) |
| Losses | MSE | Discounted multi-horizon trajectory loss, jerk penalty, Gaussian NLL for σ, auxiliary JEPA, spike-rate regulariser |
| Speculative decoding | Repeats one draft action k times, and "verifies" by re-running the target on a duplicated last frame | Asynchronous speculative control. The drafter acts every tick; the target verifies the last k drafts in one batched pass, overrides on rejection, adapts k, and feeds rejections to a hard-rejection buffer |
| Drafter | Same model with 1 layer (still runs the ViT) | Separate **edge drafter with no camera path** (LiDAR BEV + IMU + command → 1-layer LIF/LTC), distilled with KL plus a safety hinge |
| Muscle memory | CPU/NumPy linear scan over raw embeddings with L2 threshold; returns cached actions for any input | Device-resident cosine-NN cache keyed by drafter latent **and command**, confidence-gated, LRU eviction, filled only with **target-verified** chunks |
| Safety | none | Control-barrier-function (CBF) velocity filter on every executed command |
| RL | none | Optional GRPO on action chunks with simulator-scored rewards, KL-anchored to the BC policy |
| Stability | none | Global-norm clipping 1.0, spectral clamp, TBPTT with detached carry-over, FP32 ODE/membrane under BF16/FP16 autocast, frozen-feature cache |
| Evaluation | Printed losses | Per-stage metrics against trivial baselines, closed-loop success/collision rates, acceptance rate, target passes per tick, critical-path vs off-path latency |
| Deployment | `handler.py` string | `export/` with weights, config, `plm.py` and a Hugging Face `EndpointHandler` |

---

## 2. Architecture

```
                    ┌──────────────────────── PERCEPTION (Stage 1, then frozen) ─────────────────────────┐
FPV camera  ──► ViT-Tiny (pretrained; LoRA on last 2 blocks) ──► patch tokens ──► BEV lifter ──► f_vis ─┐ │
                                                          (learned G×G cell queries cross-attend)         │ │
LiDAR / 4D radar voxels ─┬► geometry 3D-CNN trunk ─► height-collapse ─► BEV pillars ─► f_geo ─┐ (contrastive ↔ f_vis)
 [occupancy, Doppler]    └► motion 3D-CNN trunk   ─► height-collapse ─► BEV pillars ─► f_dyn ─┤
                                                             f_rad = f_geo + W·f_dyn ◄──────────┘   │
                                                                                                 ▼ │
                                                                         fused = LN(f_rad + f_vis + MLP[f_rad‖f_vis])
                                                                         heads: occupancy(f_vis), semantics(f_vis), Doppler(f_dyn‖fused)
                    camera patch tokens ──► pooled to ≤7×7 perspective tokens (head: per-token semantics)
                    scene tokens = [ G² fused BEV cells ‖ pooled camera tokens ]   (cached after Stage 1)
                    └───────────────────────────────────────────────────────────────────────────────────┘
command "fly to the red beacon" ─► text encoder ─► c
                                                   │
scene tokens ─► StateTokenizer: queries + c cross-attend over BEV cells + camera tokens, + proprio(altitude, Δt) + c ─► s_t
                                                   │
                   ┌──── CORE (× n_layers, pre-norm) ───────────────────────────────────┐
  s_1..s_T, Δt ──► │ LIF spiking causal self-attention (Q,K,V spike trains over time)  │
                   │ closed-form LTC recurrence across time (uses real Δt)              │──► h_t
                   └────────────────────────────────────────────────────────────────────┘
                                                   │
             ┌─────────────────────────────────────┼──────────────────────────────────┐
             ▼                                     ▼                                  ▼
  CNN chunk head: μ, σ for u_{t..t+H-1}   JEPA predictor ŝ_{t+h}=sg(s_t)+g(h_t,ω_t,h)   spike-rate monitor

EDGE DRAFTER (no camera):  f_rad ─► mean/max pool ─┐
                           IMU ─► MLP ─────────────┼─► z_t ─► 1-layer LIF/LTC core ─► CNN chunk head (μ, σ)
                           command ─► EmbeddingBag ┘
```

**Sizes.** Full config: ~5.5 M-param ViT-Tiny (frozen, with LoRA), ~2–3 M trainable PLM parameters, ~0.1 M-param drafter. The core is deliberately small: the design targets an edge drone with a ~20 ms control deadline, not a data-centre model.

### 2.1 Sensors and simulator (`DroneSim`)

Everything is in an egocentric frame: x forward, y left, z up.

* **Camera:** pinhole FPV render with a 90° field of view, sky/ground background, depth-shaded obstacles and red/blue beacons, and pixel noise.
* **LiDAR / 4D radar:** voxels (default 8×32×32 over z ∈ [−4, 4], y ∈ [−12, 12], x ∈ [−6, 18] m). There are two channels: **occupancy** and **radial Doppler velocity** of the occupying object. 5 % point dropout and 0.2 % spurious returns are added. The Doppler channel is the "4D radar" of the design doc. It makes the velocity-consistency loss physically meaningful, since no single-frame occupancy grid can reveal velocity.
* **IMU / proprioception (9-d):** body velocity, yaw rate, body acceleration, altitude and Δt.
* **Non-uniform Δt:** each control interval is drawn from U[30, 80] ms. This is the Δt_k = t_k − t_{k−1} that the LIF leak and the LTC ODE integrate over.
* **Beacons:** red and blue spheres (0.8 m radius), visible to both camera and LiDAR. Only the camera can tell them apart.
* **Language:** 7 intents (fly to red / blue beacon, hover, ascend, descend, turn left / right), each with three paraphrases.
* **Expert:** attractive field to the goal, repulsive field from obstacles with a tangential escape term (to avoid local minima), and ground avoidance. Labels are the **raw** expert commands. The quadrotor's first-order velocity tracking smooths execution, and the jerk penalty smooths the learned chunks.
* **DART:** half the episodes execute expert + N(0, 0.15) while recording the clean expert label, so behaviour cloning sees recoveries.

### 2.2 Why the policy does not see its own velocity (anti-copycat)

The IMU's ego-velocity, acceleration and yaw-rate channels explain **97 %** of the variance of the expert's forward-velocity command (R² of a linear fit), because the drone is tracking its previous command. Our first full run gave the policy all nine IMU channels. Offline it looked excellent: action-chunk MSE was 19× below the mean-action baseline. In closed loop it scored **0 % success and never left a hover**. Its final distance to the goal equalled its starting distance, because it had learned to copy its own velocity, and from rest that means staying at rest. This is the "copycat" form of causal confusion in behaviour cloning.

Randomly dropping these channels in training does not help, because at test time they are always present. The fix (`policy_proprio="no_motion"`, the default) removes the motion channels from the **policy and drafter inputs**. They keep altitude and Δt, and they still perceive relative motion through radar Doppler. The **world model's predictor keeps the full IMU** as ω_t. The copycat run is kept in `runs/plm_small_copycat/` as evidence. The lesson: *never* select a policy on offline action error alone; always report closed-loop metrics.

### 2.3 Why BEV (bird's-eye view)

The design doc's Stage 1 says to *"map sparse radar Doppler returns and 2D perspective CLIP tokens into a unified, spatially calibrated BEV coordinate space before training any control policies."* A shared metric grid gives the two modalities a common index, *cell (i, j)*. That makes cross-modal alignment a well-posed per-cell contrastive problem, instead of the ill-posed "embedding vs itself" in the notebook. It also gives the world model a spatial target to predict.

### 2.4 Why spiking attention and LTC

* **LIF attention.** Q, K and V become binary spike trains, so QKᵀ is an accumulate over sparse events rather than dense multiply-accumulates. This is the energy argument for neuromorphic hardware. The leak uses the *actual* elapsed time, β = exp(−Δt/τ), so irregular sensor timing is handled natively. The spike-rate regulariser holds firing near 5–10 % (r_target = 0.08). That avoids **quiescence** (no spikes, so zero gradient) and **hyperactivity** (saturated energy budget).
* **Closed-form LTC.** h_t = h_{t−1}·e^{−Δt(1/τ+f)} + (A·f/(1/τ+f))·(1 − e^{−Δt(1/τ+f)}), with conductance f = softplus(W_x x + W_h h + b) ≥ 0. The effective time constant depends on the input ("liquid"). The update is exact for piecewise-constant inputs, so no ODE solver is needed, and it is stable for any Δt. State stays bounded between 0 and the reversal potentials A.
* **Fast-sigmoid surrogate** (σ′(z) = 1/(1+k|z|)²), as the design doc specifies. Its heavier tails than the original triangular surrogate keep gradients alive for neurons far from threshold.

### 2.5 Why action chunks, and why a CNN head

The design doc's Stage 3: *"Rather than predicting a single instantaneous action, the CNN head outputs an action trajectory chunk across horizon H = 10."* Chunks give temporal consistency. A 1-D conv decoder over the chunk's time axis makes neighbouring actions share features, which, together with the jerk penalty, gives smooth commands. The head also outputs log-σ. σ is what makes the drafter-to-target KL well defined, and it sets the speculative acceptance tolerance.

### 2.6 Why the edge drafter has no camera

In speculative decoding, the draft model's whole purpose is to be cheap. The ViT and BEV lifter dominate per-tick cost, so the drafter uses only the LiDAR pillar features (computed anyway), IMU and the command. The camera path and the full core run **off the critical path**, to verify.

---

## 3. Training curriculum (the four stages)

The design doc warns: *"Attempting to backpropagate from raw motor control down into an untrained spiking network, frozen CLIP, and raw radar tensors all at once will result in immediate gradient collapse or catastrophic forgetting."* Each stage therefore trains one thing against a signal that exists at that point.

### Stage 1: Modality pre-alignment and BEV grounding (`stage1`)
Trains: 3D CNN, BEV lifter, fusion, LoRA adapters (or the whole ViT if no pretrained weights are available).

* **L_contra:** symmetric InfoNCE between radar cell features f_rad⁽ⁱ⁾ and lifted camera features f_vis⁽ⁱ⁾ *of the same metric cell*. Negatives are all other occupied cells in the batch. It is restricted to occupied cells **inside the camera frustum**, because cells the camera cannot see would be noise positives.
* **L_occ:** camera-only occupancy prediction against LiDAR occupancy. This is the "metric depth alignment" of the doc: the camera must learn where things are in metres.
* **L_vel:** L1 radial-velocity regression on occupied cells, which is the doc's radial velocity consistency loss.
* **L_psem:** per-camera-token semantics (none / obstacle / red / blue) on the pooled perspective tokens. This is an easy 2-D task that makes the camera tokens carry colour and bearing directly.
* **L_sem:** camera-only BEV semantics (empty / obstacle / red beacon / blue beacon) from ground-truth object boxes, the doc's "ground-truth 3D bounding boxes" supervision. It is class-weighted towards beacons.

> **Design note: geometry-only alignment makes the policy colour-blind.** Without L_sem, all Stage-1 objectives were geometric: alignment with LiDAR, occupancy and Doppler. LiDAR cannot see colour, so the frozen BEV features never had to encode which beacon was red. The policy then learned to fly straight at full speed. That is a good offline fit, because forward velocity dominates the label variance, but it **never steered to the named beacon**. Close to the goal the expert commanded a yaw rate of 0.45; the policy commanded ≈0. When perception is frozen before the policy is trained, perception must be supervised on **every attribute the language can refer to**.
>
> Adding BEV semantics alone was not enough at 96 px. Putting a 2 px beacon into the right metric cell requires estimating its depth, and beacon-cell accuracy stayed at 0.3–0.5. The policy does not need the beacon's depth, only its **bearing**. So the tokenizer attends over *scene tokens*: the BEV cells plus the pooled perspective camera tokens, trained with the per-token semantic loss above (≈ 0.89 beacon accuracy by step 300). BEV provides metric obstacle geometry; perspective tokens provide what-and-which-direction.

Reported against baselines: contrastive top-1, camera→BEV occupancy IoU, beacon-cell classification accuracy, and Doppler MAE vs predicting zero.

> **Design note found during development: cross-modal alignment erases motion.** With one shared radar trunk, Doppler error stayed at the predict-zero baseline for 1,000+ steps, even though Doppler is an *input* channel. That held whether the head read the fused features or the radar features. The same trunk trained on the Doppler loss alone reached 0.074 MAE against a 0.21 baseline in 300 steps on fresh data. The cause: cell-level contrastive alignment with a *single camera frame*, which contains no velocity information, pushes motion features out of the shared representation. The fix is a separate small **motion trunk** (f_dyn) that the contrastive loss never touches, added into f_rad downstream. With it, Doppler MAE falls below half of the baseline within 300 steps. The general lesson: when aligning modalities, keep a private channel for information only one of them can observe.

After Stage 1, perception is frozen and every episode is encoded once into an **FP16 feature cache** on the host. This is the design doc's *"pre-computed BEV spatial indices cached in memory to eliminate GPU starvation"*. Stages 2–4 then train in seconds per thousand steps and never touch the ViT.

### Stage 2: Self-supervised continuous-time dynamics (`stage2`)
Trains: text encoder, state tokenizer, core, JEPA predictor. No action labels are used.

* **L_dynamics (latent JEPA, multi-horizon):** predict the future fused-BEV latent of every cell at horizons h ∈ {1, 2, 4}, from h_t and IMU ω_t. The target is the stop-gradient output of the frozen Stage-1 encoder, so there is no representation collapse and no EMA teacher is needed.
* **Residual form:** ŝ_{t+h} = sg(s_t) + g_ψ(h_t, ω_t, h), with g_ψ zero-initialised. With a plain predictor, the first run's loss was 12× *worse* than "nothing changes", because the core was forced to re-encode the whole present. The residual form starts exactly at the persistence baseline, so the core only has to model **change**: ego-motion and moving obstacles. The report prints `jepa_vs_persistence`; < 1 means real dynamics were learned.
* **L_spike:** λ Σ_l (r̄_l − r_target)².
* Sequences of T_long steps trained with **truncated BPTT** (chunk `tbptt`), carrying the detached hidden and membrane state across chunks.

### Stage 3: Language-conditioned trajectory imitation (`stage3`)
Trains: text, tokenizer, core, chunk head, JEPA (as an auxiliary).

* **L_traj** = Σ_k γᵏ (λ_θ‖θ̂−θ*‖² + λ_v‖v̂−v*‖²) over the H-step chunk (γ = 0.9, normalised by Σγᵏ).
* **L_jerk** = Σ‖û_{k+2} − 2û_{k+1} + û_k‖²: second temporal difference, to protect actuators.
* **L_NLL:** Gaussian NLL of the expert chunk under (μ.detach(), σ). This trains calibrated σ without letting the variance term distort the mean.
* **Auxiliary JEPA** (λ = 0.2) keeps the dynamics knowledge from Stage 2.
* After Stage 3, the target is evaluated **closed-loop** in three modes: every tick, chunk execution, and against the expert.

### Stage 3b: DAgger (`dagger`, on by default in `full`)
Behaviour cloning only sees states the expert visits. The policy flies the drone (executing the expert's action with probability β = 0.3), and the expert labels every state it actually reaches. These windows are added to the cache, and Stage 3 is fine-tuned on the enlarged set (`dagger_rounds` rounds of `dagger_episodes` flights). Together with `warmup_max`, which starts recorded windows anywhere in a flight rather than only in its first seconds, this targets the coverage gap diagnosed in the CPU run: near-goal states were almost absent from the training data.

### Stage 4: Speculative co-training and RL fine-tuning (`stage4`, `grpo_finetune`)
Trains: the drafter (then optionally the target core and head with GRPO).

* **Distillation:** L_spec_align = KL(p_θ(·|x) ‖ q_φ(·|x)) + β·hinge(‖û_draft − u*_target‖ − ρ_safe). The design doc's indicator 𝟙(‖·‖ > ρ_safe) has zero gradient almost everywhere, so a hinge is used as its differentiable surrogate. The indicator itself is logged as `unsafe_rate`.
* **Hard-rejection buffer.** Closed-loop speculative rollouts are run. Every state whose draft the target rejected is stored with the target's (μ, σ). The drafter is then retrained on a 50/50 mix of buffer states and ordinary data, so it adapts specifically to its failure cases. Acceptance is logged per round.
* **GRPO (optional, `grpo_iters`):** at states visited in closed loop, G chunks are sampled from the target's Gaussian and each is rolled out open-loop in a *copy* of the simulator. Reward = goal progress − 5·collision − 0.5·jerk. The group-relative advantage weights log-probabilities, and a KL term anchors the policy to the frozen BC reference. This is the "PPO/GRPO" in the design doc's Stage 4, applied to chunks (no value network is needed).

---

## 4. Runtime (`PLMRuntime`)

| mode | what runs per tick | purpose |
|---|---|---|
| `expert` | privileged potential-field controller | upper bound |
| `target` | full model every tick | quality reference |
| `target_chunk` | full model every k ticks, executes k actions of the chunk open-loop | the no-drafter way to save compute |
| `drafter` | edge drafter only | lower bound on the cheap path |
| `speculative` | cache lookup, else drafter, on the critical path; target verifies the last k drafts in **one batched pass** off the critical path | the proposed system |

**Speculative control and its guarantee.** In LLM speculative decoding, the target verifies k drafted tokens *before* they are emitted. A drone cannot wait for that, because the observation needed to verify action t+i only exists after actions t…t+i−1 have been executed. PLM therefore uses **asynchronous verification**:
1. The drafter's action executes immediately after the CBF filter.
2. Every k ticks, the target scores the k stored windows in one batch.
3. A draft is accepted if every action dimension satisfies |u_draft − μ_target| ≤ tol + z·σ_target.
4. If the latest draft is rejected, the target's action overrides the current tick. k then drops to 1 until the drafter is trusted again, when it doubles back up to `verify_every`.
5. Rejected states go to the hard-rejection buffer. Accepted, confident (σ below the cache threshold) target chunks go to the muscle-memory cache.

Guarantee: every executed action is **either target-verified, or within one verification interval (≤ k ticks) of being checked, and always CBF-filtered**. Latency is reported split into `critical_path_ms` and `offpath_ms`, because that split is the actual claim.

**MuscleMemoryCache.** Device-resident keys and values, one batched cosine similarity per lookup, per-command partitioning (a cached "go left" chunk can never answer "go right"), LRU eviction, and insertion only of *verified, low-σ* target chunks. It replaces the notebook's version, which scanned a Python list with an L2 threshold and returned cached actions regardless of the command.

**CBF safety filter.** h_i = ‖p − o_i‖² − (r_i + m)². The commanded velocity u is projected so that ḣ_i + α h_i ≥ 0 for every obstacle and the ground. This is the doc's "wrap untrusted neural outputs in formal control barrier functions." Obstacle states currently come from the simulator, standing in for a LiDAR tracker (see Limitations).

---

## 5. Numerical-stability rules (from the design doc's table)

| Problem | Implementation |
|---|---|
| BPTT exploding gradients in LTC | `clip_grad_norm_(…, 1.0)` + `clamp_spectral`: after every step, rescale W_h so ‖W_h‖₂ ≤ 0.99 (which bounds \|λ\| < 1) |
| Memory over long horizons | TBPTT chunks (`tbptt`) with `detach_state` carry-over |
| Mixed-precision instability | BF16 autocast on Ampere+ (FP16 with GradScaler otherwise) for projections and ViT; **LIF membranes and LTC integration forced to FP32** (`fp32_region`) |
| Data-loader starvation | Frozen-perception FP16 feature cache; Stage-1 frames generated in worker processes |

---

## 6. What from the design conversation was **not** implemented, and why

`plm.pdf` also covers frontier-scale systems: 4D parallelism (FSDP/ZeRO-3, Megatron TP/SP, 1F1B pipelines), MFU accounting, Triton fused RMSNorm/SwiGLU/FlashAttention kernels, FP8/MX formats, ring/tree all-reduce, silent-data-corruption canaries, straggler mitigation, multi-tier checkpointing, radix-tree KV caches, MLA, grammar-constrained decoding and MoE routing.

These solve problems of **70B–1T-parameter models on thousands of GPUs**. PLM has a few million trainable parameters and fits on one GPU with room to spare, so FSDP, TP, PP, FP8, MoE and SDC detection would add complexity with no benefit. A fused RMSNorm kernel would not change wall-clock time at this size: the recurrent time loop, not memory bandwidth, is the bottleneck.

The ideas that *do* transfer are implemented:
* keep ODE state in FP32 under mixed precision
* cache precomputed features
* speculative drafting in feature space rather than raw-input space: the drafter consumes perception *features*, similar in spirit to EAGLE drafting from hidden states
* CBF safety filtering

**Tree / EAGLE multi-candidate speculation** assumes the target can verify a draft *before* it is executed. Under asynchronous verification (above), only one action per tick can be executed, so a tree of candidates has no branch to commit to. It is left as future work for a synchronous, planning-style variant.

**Flow matching / mixture density heads** (the doc's remedy for mode averaging) are a natural next step: the Gaussian chunk head can average "pass left" and "pass right". For now, the tangential expert term plus the CBF mitigates this.

---

## 7. Outputs

### Running on your machine
* **NVIDIA GPU:** `python plm.py --mode full`. BF16 on Ampere and newer, otherwise FP16 with a GradScaler. Pretrained ViT-Tiny weights download automatically through `timm`.
* **Apple Silicon MacBook:** the same command. The device is picked automatically (CUDA → MPS → CPU). Autocast is off on MPS; everything runs in FP32. For a first try, use `--mode small`.
* **Colab:** open `PLM_v2.ipynb` and pick a GPU runtime.
* **Everything the paper still needs:** `./run_gpu.sh` runs 3 seeds of `--mode full`, the 4 ablations, regenerates every table (`paper/make_results.py --main … --seeds …`) and recompiles the PDF.
* **Only ablations:** `MODE=full BASE=runs/full_s0 PREFIX=runs/full_abl_ ./run_ablations.sh`.
* **Not yet verified on a GPU:** the `full` configuration has been dry-run end to end on CPU at tiny budgets (224 px, 7×7 camera-token pooling, DAgger, GRPO, export). The CUDA mixed-precision path (BF16/FP16 autocast, GradScaler) and MPS have not run on real hardware yet. If something breaks, it will most likely be there; `--set` lets you work around it without editing code.
* After a GPU run, `paper/results_text.tex` (the hand-written interpretation of the CPU numbers) must be rewritten to match the new tables.

`runs/<name>/`
* `report.json`: config, parameter counts, every stage's validation metrics, closed-loop tables and cache stats
* `history.json`, `training_curves.png`: per-stage training curves
* `perception.pt`, `cache_train.pt`, `cache_val.pt`: reusable for ablations (`--set perception_ckpt=… cache_dir=…`)
* `export/`: `plm_target.pt`, `plm_drafter.pt`, `plm_config.json`, `plm.py`, `handler.py` (Hugging Face `EndpointHandler`), `requirements.txt`

Endpoint input: `{"inputs": {"frames": [T,3,S,S], "lidar": [T,2,Z,Y,X], "sensors": [T,9], "dt": [T], "command": "fly to the red beacon"}}`. The output is the normalised action chunk, σ, and the first action in SI units.

---

## 8. Results

Full tables are in [RESULTS.md](RESULTS.md), generated by `paper/make_results.py` from the run reports. The paper draft is `paper/plm_paper.pdf`, and what separates it from a NeurIPS acceptance is in [PUBLICATION_ROADMAP.md](PUBLICATION_ROADMAP.md).

CPU-scale run (`--mode small`, one seed, 30 evaluation episodes):

| | PLM | baseline |
|---|---|---|
| Stage 1 radar↔camera cell top-1 | 0.41 | 0.007 (chance) |
| Stage 1 beacon accuracy (camera tokens) | 0.93 | 0.25 |
| Stage 1 Doppler MAE | 0.048 | 0.209 (predict 0) |
| Stage 2 latent prediction MSE | 0.0343 | 0.0376 (persistence) |
| Stage 3 action-chunk MSE | 0.0062 | 0.0803 (mean action) |

| Closed loop | Success | Collision | Critical-path ms |
|---|---|---|---|
| Expert (privileged) | 1.00 | 0.00 | – |
| Target every tick | 0.12 | 0.00 | 24.2 |
| Target open-loop chunk | 0.17 | 0.00 | 18.6 |
| Drafter only | 0.08 | 0.03 | 7.5 |
| **Verify-behind** | 0.12 | 0.00 | **4.1** |
| Target every tick, + 2 DAgger rounds | 0.12 | 0.00 | – (final goal distance 11.7 → 6.4 m; chunk-mode success 0.21; 17× jerkier) |

What this shows: every stage beats its trivial baseline, and verify-behind keeps the target's success and zero collisions at about 6× lower critical-path latency. Absolute success is low. The diagnosis is data coverage: DAgger halves the final distance to the goal but does not yet convert it into success at this scale (§3, Stage 3b). The repository also keeps the two failed runs whose offline error looked *better* (copycat: chunk MSE 0.0042 with 0 % success; colour-blind: 4 %). Those numbers come from a **from-scratch ViT at 96 px on 2 CPU cores** (the pretrained-weight download was blocked in that environment), so they validate the pipeline and the relative comparisons, not absolute performance. For publishable numbers, run `--mode full` on a GPU (pretrained ViT-Tiny, 224 px), repeat for ≥ 3 seeds, and add the external benchmarks listed in the paper's limitations.

---

## 9. Limitations (honest list)

* **Simulator, not a flight stack.** Kinematic first-order dynamics, sphere obstacles and a rendered FPV image with no textures or lighting. This validates the method, but it is not evidence of real-world transfer. Next step: AirSim / Colosseum, Flightmare, or Isaac Sim / Pegasus with PX4 software-in-the-loop, then hardware.
* **The CBF uses ground-truth obstacle states.** A real system needs a LiDAR/radar tracker feeding it.
* **The energy claim is not measured.** Spike sparsity is reported, but joules per inference need neuromorphic hardware (Loihi 2, SpiNNaker 2, Speck) or at least a synaptic-operation-count model.
* **Latency is measured on a general-purpose CPU/GPU** in a single process. "Off-path" verification runs sequentially here; a real deployment would put it on a separate stream or core.
* **The expert is privileged** (it knows every obstacle position), so BC inherits its local-minimum behaviour. A learned or MPC expert would be stronger.
* **The Gaussian chunk head is unimodal** (see §6).
