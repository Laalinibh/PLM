# From this draft to a NeurIPS-grade submission

The paper draft (`paper/plm_paper.pdf`) is written to NeurIPS structure and states its claims honestly. As it stands, it would very likely be **rejected at the NeurIPS main track**. Below are the reasons, ranked by how strongly a reviewer would weigh them, with the fix for each.

## A. Shortcomings that block acceptance

| # | Shortcoming | Why reviewers will object | Fix |
|---|---|---|---|
| 1 | **Toy simulator only** | Kinematic first-order dynamics, sphere obstacles, untextured rendering. No established benchmark, so there is nothing to compare against. | Re-run in a physics + photorealistic simulator (§C) and on at least one public benchmark task. |
| 2 | **No external baselines** | Only internal ablations. Reviewers expect comparisons with a VLA (OpenVLA / Octo / π0-style head), a CfC/NCP liquid policy, a Spikformer-style SNN policy, and plain action chunking (ACT). | Add 3–4 baselines trained on the same data. |
| 3 | **CPU-scale, one seed, 30 eval episodes** | Confidence intervals are wide (the Wilson intervals in Table 2 show this). NeurIPS expects ≥ 3–5 seeds with mean ± std or CIs. | Run `--mode full` on GPU with seeds 0–4. On a GPU, all stages take about 20–60 min per seed. |
| 4 | **From-scratch ViT at 96 px** | The pretrained-weight download was blocked where the numbers were produced. | The GPU config loads pretrained ViT-Tiny at 224 px automatically. |
| 5 | **The energy / neuromorphic claim is unmeasured** | Spike rate is not energy. "Neuromorphic" in the title invites the question "how many joules?" | At minimum, report synaptic-operation (SynOp) counts against an ANN's MAC counts using standard 45 nm energy-per-op figures. Ideally, deploy the core on Loihi 2 (Intel INRC) or SynSense Speck/Xylo. |
| 6 | **Speculative speed-up is latency, not compute** | Verify-behind still runs the target on every window; it only moves that work off the critical path. Reviewers will ask for a wall-clock benefit on real edge hardware. | Measure end-to-end loop latency on a Jetson Orin Nano / NX, with the target on a separate CUDA stream or DLA and the drafter on CPU. Also add a *verification-subsampling* variant that really does reduce target FLOPs, and plot the safety trade-off. |
| 7 | **Privileged CBF and expert** | Both use ground-truth obstacle states. | Feed the CBF from a tracker built on the LiDAR BEV occupancy. Replace the potential-field expert with an MPC or an RL expert (the privileged-teacher → student recipe of Loquercio et al. / Kaufmann et al.). |
| 8 | **Language is template-level** | 7 intents × 3 paraphrases isn't "language-conditioned" in the sense reviewers mean. | Use CLIP/T5 text (already supported: `text_backbone="clip"`), held-out paraphrases, compositional commands ("go around the moving obstacle to the red beacon"), and report zero-shot paraphrase generalisation. |
| 9 | **Unimodal Gaussian head** | Mode averaging around obstacles (left vs right). | Add a flow-matching or mixture head and show fewer collisions near symmetric obstacles. |
| 10 | **Scope** | Five components in one paper (BEV + spiking + LTC + curriculum + speculative runtime) reads as a system report. NeurIPS reviewers reward one clear, well-tested idea. | Lead with **verify-behind speculative control** as *the* contribution: the theory (Proposition 1) plus the latency/safety trade-off. Present PLM as the testbed, and the three findings (alignment erases motion, residual JEPA, copycat) as secondary insights. |

**Venue fit.** With fixes 1–4 and 10 done, NeurIPS main track is plausible. Otherwise better-matched venues are **CoRL** or **ICRA/IROS**, where robotics reviewers expect real hardware (see §C), or **RA-L** as a journal. Workshop routes for the current version: NeurIPS workshops on robot learning or on neuromorphic / efficient ML. **TMLR** is a strong option for a careful version without hardware.

## B. What is already publishable-quality

* Three **negative/diagnostic findings**, each backed by a controlled comparison:
  1. Cell-level cross-modal contrastive alignment erases single-modality motion (Doppler) unless it has a private channel.
  2. A non-residual latent world model does worse than persistence; the residual form beats it.
  3. The copycat failure: 19× better offline, 0 % closed-loop.
* The **verify-behind** formulation and its bounded-lag / CBF-safety proposition.
* A clean, single-file implementation with per-stage baselines. It is easy to reproduce, which reviewers value.

## C. Simulated drone or real drone?

**Both, in this order. Never go to hardware first.**

**Step 1: a better simulator (required for any ML venue).** Pick one:
* **Aerial Gym** (NVIDIA Isaac Gym/Lab based, NTNU): massively parallel GPU quadrotor simulation with depth and camera sensors. Best for RL and GRPO at scale.
* **Pegasus Simulator** (NVIDIA Isaac Sim + PX4): photorealistic, runs the real PX4 autopilot in software-in-the-loop. Best bridge to hardware.
* **Flightmare** (UZH RPG): fast Unity rendering with separate physics; used in the agile-flight literature.
* **gym-pybullet-drones** (U. Toronto): lightweight, Crazyflie dynamics, easy to start.
* **Colosseum**: the community continuation of Microsoft AirSim (the original AirSim repository was archived), Unreal-based.

Swap `DroneSim` for a wrapper that exposes the same `observe()` / `step()` / `expert_raw()` interface. Everything else in `plm.py` stays unchanged.

**Step 2: software-in-the-loop, then hardware-in-the-loop.** Run the trained policy against PX4 SITL (via Pegasus or Gazebo), sending velocity setpoints through MAVSDK or ROS 2 / MAVROS. Then run the same stack on the actual companion computer, with the simulator still providing the sensors.

**Step 3: a real drone (strongly recommended for CoRL/ICRA; a big plus at NeurIPS).**
* *Small and cheap, indoors:* **Crazyflie 2.1 + AI-deck** (camera, no LiDAR). Good for language-conditioned goal reaching with a motion-capture safety net. Run the policy off-board.
* *Full sensor suite:* a **Holybro X500 v2 or ModalAI Starling 2 (VOXL 2)** frame with PX4, a **Jetson Orin Nano/NX** companion computer, a camera, and a **Livox Mid-360** LiDAR or a **TI mmWave radar** (true Doppler, matching the model's 4D-radar channel).
* *Safety protocol:* tethered or netted flight area, a geofence, a human safety pilot with a kill switch, the CBF running onboard, and velocity limits well below the training `vmax` for the first flights.

**What to report from hardware:** success and collision rates over ≥ 20 flights per condition, loop latency (critical path vs off-path), drafter acceptance rate, and CBF intervention rate. Also report a sim-vs-real gap table, where the same metrics are measured in simulation.

## D. Minimal plan (about 4–6 weeks)

1. **Week 1:** `./run_gpu.sh` (3 seeds of `--mode full` + 4 ablations + table regeneration); extend to 5 seeds; add a SynOp energy estimate; rewrite `paper/results_text.tex` for the GPU numbers.
2. **Week 2:** port to Aerial Gym or Pegasus; retrain; add baselines (ACT chunking, CfC policy, Spikformer policy, and an OpenVLA-style fine-tune if compute allows).
3. **Week 3:** held-out paraphrases, compositional commands, CLIP text; a flow-matching head.
4. **Weeks 4–5:** PX4 SITL → HITL on a Jetson; latency measurements; 20+ real flights (Crazyflie indoors, or X500 in a netted area).
5. **Week 6:** rewrite around verify-behind as the main contribution; fill in the NeurIPS checklist; release the code.
