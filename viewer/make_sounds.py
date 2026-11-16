"""Cuts Keep's sound set from the user's curated pet videos (pet-video-references-2026-11-15 on the cluster; audio
pulled to SRC as mono 44.1 kHz FLAC) and writes data/sounds/*.mp3 plus data/sounds/sounds.json with credits.
Standard library + ffmpeg (with rubberband and libmp3lame).  usage: python3 make_sounds.py SRC_DIR

Segments were chosen by reading spectrograms (no audio classifier was available): each is a stretch where the
cat's sound is clear and no human speech shows. Pikachu's own meow (meow-1..6, made earlier) stays first.
"""
import json
import os
import re
import subprocess
import sys

SRC = sys.argv[1]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "sounds")

CLIPS = {  # source file -> credit line (from the dataset's manifest.csv)
    "cat_meowing_green_eyed": ("\"Green eyed cat\" by NairobiPapel (Kamaa), CC BY-SA 4.0, Wikimedia Commons",
                               "https://commons.wikimedia.org/wiki/File:Green_eyed_cat.webm"),
    "cat_chirping_holly": ("\"Cat making noises at the birds outside\" (Holly) by Ali T, CC BY 2.0, Wikimedia Commons",
                           "https://commons.wikimedia.org/wiki/File:Cat_making_noises_at_the_birds_outside.webm"),
    "cat_purring_kneading_amelia": ("\"Amelia - cat purrs and kneads\" by Paulo O, CC BY 2.0, Wikimedia Commons",
                                    "https://commons.wikimedia.org/wiki/File:Amelia_-cat_purrs_and_kneads.webm"),
    "cat_grooming_purring": ("\"Orange tabby cat grooming and purring close-up\" by Brion VIBBER, CC BY-SA 3.0, Wikimedia Commons",
                             "https://commons.wikimedia.org/wiki/File:Orange_tabby_cat_grooming_and_purring_close-up.webm"),
    "cat_purring_chattering_lola": ("\"Lola Purring and Chattering\" by Nina Paley, CC BY 3.0, Internet Archive",
                                    "https://archive.org/details/youtube-t89bGOrWmew"),
    "tomcat-calls": ("\"Tomcat calling\" by Drummyfish, CC0 1.0, Wikimedia Commons",
                     "https://commons.wikimedia.org/wiki/File:Tomcat_calling.webm"),
    "cat-growls-hisses": ("\"Black and white cat growl and hiss 2\" by Abujoy, CC BY 4.0, Wikimedia Commons",
                          "https://commons.wikimedia.org/wiki/File:Black_and_white_cat_growl_and_hiss_2.webm"),
    "momo-growls-hisses": ("\"Momo growls and hisses at newcomer\" by Nina Paley, CC BY 3.0, Internet Archive",
                           "https://archive.org/details/youtube-7T1as6FQOJY"),
}

# name, group, source, start s, end s, filter chain, peak dBFS, pitch ratio (1 = unchanged), loop
SEGMENTS = [
    ("meow-g1", "meow", "cat_meowing_green_eyed", 0.20, 1.00, "highpass=f=300,afftdn=nr=12:nf=-50", -3, 0.84, False),
    # (4.05-4.65 s of the same clip was too faint: mostly noise once raised; left out)
    ("meow-g3", "meow", "cat_meowing_green_eyed", 5.35, 6.15, "highpass=f=300,afftdn=nr=12:nf=-50", -3, 0.84, False),
    ("meow-g4", "meow", "cat_meowing_green_eyed", 8.15, 9.00, "highpass=f=300,afftdn=nr=12:nf=-50", -3, 0.84, False),
    # Holly's recording is noisy; the first chirp (0.52-1.10 s) stayed mostly noise and is left out
    ("trill-2", "trill", "cat_chirping_holly", 1.70, 2.45,
     "highpass=f=750,lowpass=f=6000,afftdn=nr=30:nf=-40:tn=1", -6, 1, False),
    ("purr-1", "purr", "cat_purring_kneading_amelia", 0.90, 12.90, "highpass=f=30,lowpass=f=2500,afftdn=nr=6:nf=-55", -6, 1, True),
    ("purr-2", "purr", "cat_grooming_purring", 16.0, 38.0, "highpass=f=30,lowpass=f=2500,afftdn=nr=8:nf=-55", -6, 1, True),
    ("chatter-1", "chatter", "cat_purring_chattering_lola", 50.0, 58.5,
     "highpass=f=700,lowpass=f=3500,afftdn=nr=30:nf=-75:tn=1", -8, 1, False),
    ("call-1", "call", "tomcat-calls", 3.65, 6.30, "highpass=f=250,afftdn=nr=12:nf=-50", -3, 1, False),
    ("call-2", "call", "tomcat-calls", 7.95, 11.15, "highpass=f=250,afftdn=nr=12:nf=-50", -3, 1, False),
    ("hiss-1", "hiss", "cat-growls-hisses", 8.10, 9.95, "highpass=f=150,afftdn=nr=6:nf=-50", -4, 1, False),
    ("hiss-2", "hiss", "momo-growls-hisses", 12.95, 14.75, "highpass=f=80,afftdn=nr=8:nf=-55", -4, 1, False),
]
XF = 0.5  # seconds of crossfade that close a loop


def run(args):
    return subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"] + args, check=True,
                          capture_output=True, text=True)


def peak_db(path):
    r = subprocess.run(["ffmpeg", "-hide_banner", "-i", path, "-af", "volumedetect", "-f", "null", "-"],
                       capture_output=True, text=True)
    return float(re.search(r"max_volume: (-?[\d.]+) dB", r.stderr).group(1))


def main():
    tmp = os.path.join(SRC, "work")
    os.makedirs(tmp, exist_ok=True)
    made = []
    for name, group, src, a, b, chain, peak, pitch, loop in SEGMENTS:
        inp = os.path.join(SRC, src + ".flac")
        raw = os.path.join(tmp, name + ".wav")
        extra = XF if loop else 0.0
        fx = "atrim=%.3f:%.3f,asetpts=PTS-STARTPTS,%s" % (a, b + extra, chain)
        if pitch != 1:
            fx += ",rubberband=pitch=%.3f:formant=preserved" % pitch
        run(["-i", inp, "-af", fx, "-ar", "44100", "-ac", "1", raw])
        shaped = os.path.join(tmp, name + "_s.wav")
        if loop:  # seamless loop: the tail fades out while the head fades in, laid over the end
            T = b - a
            fc = ("[0:a]atrim=%.3f:%.3f,asetpts=PTS-STARTPTS[body];"
                  "[0:a]atrim=%.3f:%.3f,asetpts=PTS-STARTPTS,afade=t=out:d=%.3f[tail];"
                  "[0:a]atrim=0:%.3f,asetpts=PTS-STARTPTS,afade=t=in:d=%.3f[head];"
                  "[tail][head]amix=inputs=2:normalize=0[x];[body][x]concat=n=2:v=0:a=1[out]") % (
                XF, T, T, T + XF, XF, XF, XF)
            run(["-i", raw, "-filter_complex", fc, "-map", "[out]", shaped])
        else:
            d = b - a
            run(["-i", raw, "-af", "afade=t=in:d=0.02,afade=t=out:st=%.3f:d=0.08" % max(0, d - 0.08), shaped])
        gain = peak - peak_db(shaped)
        mp3 = os.path.join(OUT, name + ".mp3")
        run(["-i", shaped, "-af", "volume=%.2fdB" % gain, "-c:a", "libmp3lame", "-q:a", "3", mp3])
        made.append({"file": name + ".mp3", "group": group, "loop": loop, "source": src,
                     "from_s": a, "to_s": b, "pitch": pitch})
        print("%-10s %-8s %5.2f s  gain %+5.1f dB  %s" % (name, group, b - a, gain, src))

    groups = {"meow": [{"file": "meow-%d.mp3" % k, "group": "meow", "loop": False, "source": "pikachu"} for k in range(1, 7)]}
    for m in made:
        groups.setdefault(m["group"], []).append(m)
    used = []
    for m in made:
        if m["source"] not in used:
            used.append(m["source"])
    manifest = {
        "groups": groups,
        "credit": ("Meow: Pikachu's own, from its owner's video cat-pikachiu_13 in the BANMo data (CC0), cleaned; five of its "
                   "six meows are pitch- and length-shifted copies. The other sounds are other cats', from openly licensed "
                   "videos, cut, cleaned and level-matched; the borrowed meows are lowered about 3 semitones toward "
                   "Pikachu's pitch:"),
        "sources": [{"text": CLIPS[s][0], "url": CLIPS[s][1]} for s in used],
    }
    with open(os.path.join(OUT, "sounds.json"), "w") as fh:
        json.dump(manifest, fh, indent=1)
    print("wrote", len(made), "sounds and sounds.json")


if __name__ == "__main__":
    main()
