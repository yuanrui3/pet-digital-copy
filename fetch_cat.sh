#!/bin/bash
# Copies a cat run's scores, pictures, turntable and log from the cluster into work/pets/out/cat/LOG/ (git-ignored).
# usage: work/pets/fetch_cat.sh LOG JOBID     e.g. fetch_cat.sh fg-skel-heldout-r2 12380000
LOG=$1; JOB=$2
HOST=USER@CLUSTER_HOST
D=/scratch/CLUSTER/USER/claude-pets
KEY=(-i "$HOME/.ssh/id_ed25519_cluster" -o BatchMode=yes -o IdentitiesOnly=yes -o ConnectTimeout=20)
OUT="$(cd "$(dirname "$0")" && pwd)/out/cat/$LOG"
mkdir -p "$OUT"
for f in heldout.json identity.json sheet_heldout.jpg lookalikes.jpg turntable.mp4 opts.log; do
  scp -q "${KEY[@]}" "$HOST:$D/experiments_cat/$LOG/$f" "$OUT/" 2>/dev/null || echo "$LOG: no $f"
done
# the job log without progress bars
ssh "${KEY[@]}" "$HOST" "tr '\r' '\n' < $D/logs/cat-$JOB.out 2>/dev/null | grep -v -E 'it/s\]|s/it\]'" 2>/dev/null \
  | grep -v -E '^\*+$|BoilerKey|^$' > "$OUT/run.log"
echo "fetched into $OUT"; ls "$OUT"
