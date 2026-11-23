#!/bin/zsh
# Rebuild the local preview (the scratch folder is cleared now and then): page plus local-only hooks, a copy of the
# page data, the server and screenshot helper where .claude/launch.json expects them.
# usage: make_preview.sh SCRATCH
S=$1; P=${0:A:h}; V=${P:h}/viewer
mkdir -p $S/pet_viewer/data
cp $P/serve_local.py $P/shotq.sh $S/
rsync -a $V/data/ $S/pet_viewer/data/
/usr/bin/python3 - "$V/index.html" "$P/hooks.js" "$S/pet_viewer/index.html" <<'PY'
import sys
src, hooks, out = sys.argv[1:4]
s = open(src).read()
anchor = '    if (qs.has("t")) { setHomeTime(Math.min(home.tracks[home.k].frames - 1, +qs.get("t"))); home.snap = true; }\n'
assert anchor in s
open(out, "w").write(s.replace(anchor, anchor + open(hooks).read()))
PY
echo "preview ready in $S/pet_viewer"
