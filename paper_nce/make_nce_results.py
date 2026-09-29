"""Generate every number and table of the NCE paper from run reports, energy.json files and logs."""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from plm import Config, DroneSim, GOAL_INTENTS  # noqa: E402

R = os.path.join(HERE, "..", "runs")
J = lambda *p: json.load(open(os.path.join(R, *p)))
ex = lambda *p: os.path.exists(os.path.join(R, *p))
main, en = J("plm_small", "report.json"), J("plm_small", "energy.json")
cfg = Config.small()
out = {}


def f(x, d=3):
    return "--" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{d}f}"


def pct(x, d=1):
    return f"{100 * x:.{d}f}"


def wilson(p, n, z=1.96):
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0, c - h), min(1, c + h)


def write(name, s):
    open(os.path.join(HERE, name), "w").write(s)


ng = sum(1 for i in range(cfg.eval_episodes)
         if (GOAL_INTENTS[i % 2] if i % 3 else DroneSim(cfg, 5_000_000 + i).intent) in GOAL_INTENTS)
s1, s2, s3, s4 = main["stage1"], main["stage2"], main["stage3"], main["stage4"]
out.update(NObs=cfg.n_obstacles, NCache=cfg.n_cache_episodes, EvalEps=cfg.eval_episodes,
           ParamsTarget=f"{main['params']['target_total'] / 1e6:.2f}\\,M", ParamsVit=f"{main['params']['vit'] / 1e6:.2f}\\,M",
           ParamsDrafter=f"{main['params']['drafter'] / 1e3:.0f}\\,k", JepaRatio=f(s2["jepa_vs_persistence"], 2),
           BeaconPatch=f(s1["patch_beacon_acc"], 2))

# ---------------- Table: per-stage diagnostics ----------------
write("nce_table_stages.tex", f"""\\begin{{table}}[t]\\centering\\small
\\caption{{Per-stage diagnostics on held-out data, each against a trivial baseline.}}\\label{{tab:stages}}
\\begin{{tabular}}{{llcc}}\\toprule
Stage & Metric & PLM & Baseline \\\\\\midrule
1 & cell-level radar$\\leftrightarrow$camera top-1 & {f(s1['contra_top1'])} & {f(1 / s1['n_pairs'], 4)} (chance) \\\\
1 & camera$\\to$BEV occupancy IoU & {f(s1['cam_occ_iou'])} & -- \\\\
1 & beacon accuracy, camera tokens / BEV cells & {f(s1['patch_beacon_acc'], 2)} / {f(s1['beacon_cell_acc'], 2)} & 0.25 (chance) \\\\
1 & Doppler MAE & {f(s1['vel_mae'])} & {f(s1['vel_mae_zero_baseline'])} (predict 0) \\\\
2 & latent prediction MSE & {f(s2['val_jepa'], 4)} & {f(s2['val_persistence'], 4)} (persistence) \\\\
2 & mean firing rate & {f(s2['spike_rate'])} & target 0.08 \\\\
3 & action-chunk MSE & {f(s3['val_chunk_mse'], 4)} & {f(s3['val_mean_baseline_mse'], 4)} (mean action) \\\\
3 & chunk jerk & {f(s3['pred_jerk'], 4)} & {f(s3['expert_jerk'], 4)} (expert) \\\\
4 & offline draft acceptance & {f(s4['offline_acceptance_final'])} & {f(s4['offline_acceptance_after_distill'])} (distillation only) \\\\
\\bottomrule\\end{{tabular}}\\end{{table}}
""")

# ---------------- Energy (system level) ----------------
m = en["macs_per_tick"]
uJ = lambda mac: mac * 4.6e-6  # MACs -> µJ at 4.6 pJ
core_as_built_uJ = en["core"]["energy_spiking_as_built_uJ"]
cl = main["closed_loop_final"]
spec = cl["speculative"]
perc = m["lidar_encoder"] + m["camera_vit"] + m["bev_lifter_fusion"] + m["state_tokenizer"]
core = m["target_core(window T)"] + m["target_head(last step)"]
crit_draft = m["lidar_encoder"] + m["drafter_latent"] + (1 - spec["cache_hit_rate"]) * m["drafter_core_head(window T)"]
modes_uJ = {
    "target": (uJ(perc + core), uJ(perc + core)),
    "target_chunk": (uJ(perc + core * cl["target_chunk"]["target_passes_per_tick"]),) * 2,
    "drafter": (uJ(m["lidar_encoder"] + m["drafter_latent"] + m["drafter_core_head(window T)"]),) * 2,
    "speculative": (uJ(crit_draft), uJ(crit_draft + m["camera_vit"] + m["bev_lifter_fusion"] + m["state_tokenizer"]
                                       + core * spec["target_windows_per_tick"])),
}
out["VitSharePct"] = pct(m["camera_vit"] / (perc + core), 0)
out["CritRatio"] = f"{modes_uJ['target'][0] / modes_uJ['speculative'][0]:.0f}"
rows = [("LiDAR/radar pillar encoder", m["lidar_encoder"]), ("camera ViT-Tiny", m["camera_vit"]),
        ("BEV lifter + fusion", m["bev_lifter_fusion"]), ("state tokenizer (1 step)", m["state_tokenizer"]),
        ("target liquid--spiking core ($T$ steps)", m["target_core(window T)"]),
        ("target action head (last step)", m["target_head(last step)"]),
        ("drafter latent (1 step)", m["drafter_latent"]),
        ("drafter core ($T$ steps) + head (last step)", m["drafter_core_head(window T)"])]
body = "\n".join(f"{n} & {v / 1e6:.2f} & {uJ(v):.1f} \\\\" for n, v in rows)
write("nce_table_energy.tex", f"""\\begin{{table}}[t]\\centering\\small
\\caption{{Dense operation count and energy per control tick, by component (MACs at 4.6\\,pJ, before any spike-driven
reduction). The camera encoder dominates; the liquid--spiking core is {pct(m["target_core(window T)"] / (perc + core))}\\% of a full-model tick.}}\\label{{tab:energy}}
\\begin{{tabular}}{{lrr}}\\toprule Component & MMACs & $\\mu$J \\\\\\midrule
{body}
\\bottomrule\\end{{tabular}}\\end{{table}}
""")

# ---------------- Neuromorphic variants ----------------
out["AttnFracPct"] = pct(en["core"]["attention_fraction"])
out["SavingAsBuiltPct"] = pct(en["core"]["saving_as_built"])
out["MeanRate"] = f(en["mean_firing_rate"], 3)
variants = [("LIF spiking attention (PLM)", main, en, True)]
for name, d in [("Dense attention (no LIF)", "abl_nospike"), ("No LTC (feed-forward MLP)", "abl_noltc"),
                ("Spike-driven projection inputs", "abl_spikein")]:
    if ex(d, "report.json"):
        e = J(d, "energy.json") if ex(d, "energy.json") else None
        variants.append((name, J(d, "report.json"), e, False))
rows, si = [], None
for name, rep, e, is_main in variants:
    bc = rep["closed_loop_bc"]
    if name.startswith("Dense"):
        frac, ecore, save, rate = 0.0, en["core"]["energy_ann_uJ"], 0.0, None
    else:
        frac = e["core"].get("spike_driven_fraction_of_core", e["core"]["attention_fraction"])
        ecore, save, rate = e["core"]["energy_spiking_as_built_uJ"], e["core"]["saving_as_built"], e["mean_firing_rate"]
    if name.startswith("Spike-driven"):
        si = dict(frac=frac, save=save, rep=rep, rate=rate)
    rows.append(f"{name} & {pct(frac)}\\% & {ecore:.2f} & {pct(save)}\\% & {f(rate, 3)} & "
                f"{f(rep['stage3']['val_chunk_mse'], 4)} & {f(bc['target']['success_rate'], 2)} & "
                f"{f(bc['target_chunk']['success_rate'], 2)} & {f(bc['target']['collision_rate'], 2)} \\\\")
write("nce_table_neuro.tex", f"""\\begin{{table}}[t]\\centering\\small\\setlength{{\\tabcolsep}}{{3.5pt}}
\\caption{{Core variants (Stages 2--3 retrained from the same frozen perception and feature cache; closed loop on the same
{cfg.eval_episodes} seeds, {ng} goal-reaching). ``Spike-driven share'' is the fraction of the core's dense MACs whose inputs
are spikes; energy is per control tick (window $T$) with accumulates at the measured firing rate.}}\\label{{tab:neuro}}
\\resizebox{{\\linewidth}}{{!}}{{\\begin{{tabular}}{{lcccccccc}}\\toprule
Core & Spike-driven & Core $\\mu$J & Saving & Rate & Chunk MSE & Success & Success & Collision \\\\
 & share & /tick & vs dense & & & (every tick) & (chunk) & \\\\\\midrule
{chr(10).join(rows)}
\\bottomrule\\end{{tabular}}}}\\end{{table}}
""")
out["SiFracPct"] = pct(si["frac"]) if si else "--"
out["SiSavingPct"] = pct(si["save"]) if si else "--"
out["SiChunk"] = f(si["rep"]["stage3"]["val_chunk_mse"], 4) if si else "--"
out["SiSucc"] = f(si["rep"]["closed_loop_bc"]["target"]["success_rate"], 2) if si else "--"
out["SiRate"] = f(si["rate"], 3) if si else "--"

# ---------------- Closed loop ----------------
names = {"expert": "Expert (privileged)", "target": "Target, every tick", "target_chunk": "Target, chunk ($k{=}4$)",
         "drafter": "Drafter only", "speculative": "\\textbf{Verify-behind}"}
rows = []
for mode, r in cl.items():
    lo, hi = wilson(r["success_rate"], ng)
    e_c, e_t = modes_uJ.get(mode, (None, None))
    rows.append(f"{names[mode]} & {f(r['success_rate'], 2)} {{\\scriptsize[{f(lo, 2)},{f(hi, 2)}]}} & "
                f"{f(r['collision_rate'], 2)} & {f(r['action_jerk'], 4)} & {f(r['acceptance_rate'], 2)} & "
                f"{f(r['critical_path_ms'], 1)} & {f(e_c, 0)} & {f(e_t, 0)} \\\\")
write("nce_table_closed.tex", f"""\\begin{{table}}[t]\\centering\\small\\setlength{{\\tabcolsep}}{{3.5pt}}
\\caption{{Closed-loop control ({cfg.eval_episodes} episodes, {ng} goal-reaching; success with 95\\% Wilson interval).
Critical-path latency on a 2-core CPU; energy per tick from operation counts (critical path / total including off-path
verification).}}\\label{{tab:closed}}
\\resizebox{{\\linewidth}}{{!}}{{\\begin{{tabular}}{{lccccccc}}\\toprule
Controller & Success & Collision & Jerk & Accept. & Crit.\\ ms & Crit.\\ $\\mu$J & Total $\\mu$J \\\\\\midrule
{chr(10).join(rows)}
\\bottomrule\\end{{tabular}}}}\\end{{table}}
""")
out["SpecCritUJ"] = f(modes_uJ["speculative"][0], 0)
out["TargetUJ"] = f(modes_uJ["target"][0], 0)


# ---------------- Seeds ----------------
seed_reps = [main] + [J(d, "report.json") for d in ("seed1", "seed2") if ex(d, "report.json")]
out["NSeeds"] = str(len(seed_reps))
if len(seed_reps) > 1:
    su = [r["closed_loop_bc"]["target"]["success_rate"] for r in seed_reps]
    sc = [r["closed_loop_bc"]["target_chunk"]["success_rate"] for r in seed_reps]
    mse = [r["stage3"]["val_chunk_mse"] for r in seed_reps]
    out["SeedSucc"] = f"{np.mean(su):.2f} $\\pm$ {np.std(su):.2f}"
    out["SeedChunkSucc"] = f"{np.mean(sc):.2f} $\\pm$ {np.std(sc):.2f}"
    out["SeedMse"] = f"{np.mean(mse):.4f} $\\pm$ {np.std(mse):.4f}"
    out["SeedList"] = ", ".join(f(x, 2) for x in su)
else:
    out.update(SeedSucc="--", SeedChunkSucc="--", SeedMse="--", SeedList="--")


# ---------------- Diagnostic failures ----------------
def parse_log(path):
    d = {}
    for line in open(path):
        if line.startswith("[stage3_val]") or line.startswith("[eval_bc_target]"):
            for kv in line.split(":", 1)[1].split():
                if "=" in kv:
                    k, v = kv.split("=")
                    d[k] = float(v)
    return d


diag = [("Full IMU to policy (copycat)", parse_log(os.path.join(R, "plm_small_copycat", "small_run.log"))),
        ("Geometry-only perception (colour-blind)", parse_log(os.path.join(R, "plm_small_colourblind", "small_run.log"))),
        ("PLM", dict(val_chunk_mse=s3["val_chunk_mse"], val_mean_baseline_mse=s3["val_mean_baseline_mse"],
                     success_rate=main["closed_loop_bc"]["target"]["success_rate"],
                     final_goal_dist=main["closed_loop_bc"]["target"]["final_goal_dist"]))]
if ex("plm_small_dagger", "report.json"):
    dg = J("plm_small_dagger", "report.json")
    diag.append(("PLM + DAgger (2 rounds)", dict(val_chunk_mse=dg["stage3"]["val_chunk_mse"],
                                                  val_mean_baseline_mse=dg["stage3"]["val_mean_baseline_mse"],
                                                  success_rate=dg["closed_loop_bc"]["target"]["success_rate"],
                                                  final_goal_dist=dg["closed_loop_bc"]["target"]["final_goal_dist"])))
rows = [f"{n} & {f(d['val_chunk_mse'], 4)} & {d['val_mean_baseline_mse'] / d['val_chunk_mse']:.1f}$\\times$ & "
        f"{f(d['success_rate'], 2)} & {f(d['final_goal_dist'], 1)} \\\\" for n, d in diag]
write("nce_table_diag.tex", f"""\\begin{{table}}[t]\\centering\\small
\\caption{{Offline imitation error versus closed-loop outcome (target policy, every tick). Lower offline error did not mean
better control.}}\\label{{tab:diag}}
\\resizebox{{\\linewidth}}{{!}}{{\\begin{{tabular}}{{lcccc}}\\toprule
Run & Chunk MSE & vs.\\ mean action & Success & Final goal dist.\\ (m) \\\\\\midrule
{chr(10).join(rows)}
\\bottomrule\\end{{tabular}}}}\\end{{table}}
""")

with open(os.path.join(HERE, "nce_macros.tex"), "w") as fh:
    for k, v in out.items():
        fh.write(f"\\newcommand{{\\nce{k}}}{{{v}}}\n")
print(json.dumps(out, indent=1))
print("goal episodes", ng, "| variants", [v[0] for v in variants], "| modes µJ", {k: [round(x, 1) for x in v] for k, v in modes_uJ.items()})
