#!/bin/zsh
# usage: shotq.sh OUT QUERY   (QUERY without the leading ?, e.g. "video=19&t=150&facecam=-90&fdist=0.5")
OUT=$1; Q=$2
PROF=$(mktemp -d "${TMPDIR:-/tmp}/petshot.XXXX")
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --headless=new --use-angle=swiftshader --enable-unsafe-swiftshader --ignore-gpu-blocklist --hide-scrollbars --window-size=1100,1000 --virtual-time-budget=90000 --user-data-dir=$PROF --screenshot=$OUT "http://localhost:8770/index.html?$Q#home" >/dev/null 2>&1
rm -rf "$PROF"
