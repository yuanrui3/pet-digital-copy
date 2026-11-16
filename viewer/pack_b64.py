"""Turns data/NAME.bin into base64 text chunks (data/NAME.b64.K.txt) that artifacts can serve, and records the
number of chunks in data/NAME.json. Standard library only.  usage: python3 pack_b64.py NAME [NAME ...]"""
import base64
import glob
import json
import os
import sys

CHUNK = 12_000_000  # characters per file, a multiple of 4, well under the 16 MB limit per text file
here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
for name in sys.argv[1:]:
    text = base64.b64encode(open(os.path.join(here, name + ".bin"), "rb").read()).decode("ascii")
    for old in glob.glob(os.path.join(here, name + ".b64.*.txt")):
        os.remove(old)
    n = 0
    for n, start in enumerate(range(0, len(text), CHUNK), 1):
        with open(os.path.join(here, "%s.b64.%d.txt" % (name, n - 1)), "w") as fh:
            fh.write(text[start:start + CHUNK])
    meta_path = os.path.join(here, name + ".json")
    meta = json.load(open(meta_path))
    meta["chunks"] = n
    json.dump(meta, open(meta_path, "w"))
    print(name, len(text), "characters in", n, "chunks")
