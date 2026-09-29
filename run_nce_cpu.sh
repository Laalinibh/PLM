#!/usr/bin/env bash
# CPU-scale runs for the NCE submission: reuse the main run's perception + cache, retrain Stages 2-3, closed-loop eval.
cd /home/claude/plm
COMMON="perception_ckpt=runs/plm_small/perception.pt cache_dir=runs/plm_small stop_after_stage3=true"
python3 plm.py --mode small --out runs/abl_nospike --set $COMMON spiking=false > runs_abl_nospike.log 2>&1
python3 plm.py --mode small --out runs/abl_noltc   --set $COMMON core_ff=mlp   > runs_abl_noltc.log 2>&1
python3 plm.py --mode small --out runs/seed1       --set $COMMON seed=1        > runs_seed1.log 2>&1
python3 plm.py --mode small --out runs/seed2       --set $COMMON seed=2        > runs_seed2.log 2>&1
echo ALLDONE > nce_runs.done
