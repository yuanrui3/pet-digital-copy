#!/bin/bash
# Copies each finished run's scores, pictures, turntable and log from the cluster into work/pets/out/ (git-ignored).
# usage: work/pets/fetch_results.sh [ARM]   (default B)
ARM=${1:-B}
HOST=USER@CLUSTER_HOST
D=/scratch/CLUSTER/USER/claude-pets
KEY=(-i "$HOME/.ssh/id_ed25519_cluster" -o BatchMode=yes -o IdentitiesOnly=yes -o ConnectTimeout=20)
OUT="$(cd "$(dirname "$0")" && pwd)/out/$ARM"
mkdir -p "$OUT"
scp -q "${KEY[@]}" "$HOST:$D/logs/full_jobs.txt" "$OUT/" || exit 1
read -r _ j1 _ j2 _ j3 < "$OUT/full_jobs.txt"
for pair in "1030_23106_17099:$j1" "1050_43243_38827:$j2" "1022_15133_10069:$j3"; do
  seq=${pair%%:*}; job=${pair##*:}
  mkdir -p "$OUT/$seq"
  scp -q "${KEY[@]}" "$HOST:$D/experiments/$ARM/$seq/eval/*" "$OUT/$seq/" 2>/dev/null && echo "$seq: eval fetched" || echo "$seq: no eval yet"
  scp -q "${KEY[@]}" "$HOST:$D/experiments/$ARM/$seq/metric.txt" "$OUT/$seq/" 2>/dev/null
  # the training log without progress bars, so "Early stop" and errors can be checked here
  ssh "${KEY[@]}" "$HOST" "tr '\r' '\n' < $D/logs/run-$job.out 2>/dev/null | grep -v -E 'it/s\]|s/it\]'" 2>/dev/null \
    | grep -v -E '^\*+$|BoilerKey|^$' > "$OUT/$seq/run.log"
done
