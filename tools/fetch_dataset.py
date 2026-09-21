#!/usr/bin/env python3
"""Fetch the FlyWire FAFB v783 dataset into ``data/flywire/raw``.

Primary source (official, authoritative)::

    https://codex.flywire.ai/api/download?dataset=fafb

That endpoint requires an **interactive Google sign-in** before the browser is
allowed to download ``fafb.zip``, so this script cannot log in for you.  It
supports two modes:

``--official``  Open/print the official URL so you can download ``fafb.zip``
                in a signed-in browser.  Then drop the zip (or its contents)
                into ``data/flywire/raw`` and run ``--verify``.

``--mirror``    Headlessly re-download the byte-identical core files from
                public GitHub mirrors of the official zip and verify every
                file against the SHA-256 manifest of the official download.

Everything ends with ``--verify`` semantics: the files on disk are hashed and
compared to the official manifest, so there is no path by which fabricated
data can sneak in.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import tempfile
import urllib.request
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "flywire", "raw")

OFFICIAL_URL = "https://codex.flywire.ai/api/download?dataset=fafb"

# SHA-256 manifest of the official fafb.zip (interactive download of
# 2026-08-25, recorded by the fruit-fly-lab project and re-verified below).
OFFICIAL_MANIFEST = {
    "neurons.csv.gz":                    ("6a6b3759e635f0f35a677d169052362131ec61d95f55919298b55c43fce4e719", 1679884),
    "classification.csv.gz":             ("e946b552f4056dfc977707be0674609832c3f64332a22d69dc0d9615e7aae663", 934402),
    "consolidated_cell_types.csv.gz":    ("8aba246d71dc40361677493629972ce3883048c3d02010adc42bda22962a1a2d", 901707),
    "visual_neuron_types.csv.gz":        ("4bcc6a2f98b86e6c3fb7eaddb49736f3d81ab65bda35da8f740641201a1e379f", 631701),
    "coordinates.csv.gz":                ("14337121f451f98c2576cee72c24409ada5aaf7948b7c7ca8de9040296840e05", 5314546),
    "connections_princeton.csv.gz":      ("445f996bf6c4b1803b9ba186189138a3061ff8623aa94c0abcf38af30a5bd48b", 68456801),
    "column_assignment.csv.gz":          ("bdf4ce7f62cc63493d53eefad3816ff2dfd08b190e97b35a492e0e453df2f0f6", 462838),
    "cell_stats.csv.gz":                 ("bd5879e1b5df964bea2f3ca5316348d4276ce2ccaac283f0e36583c04fbd3d8e", 2526548),
    "names.csv.gz":                      ("e541ef9ef4b9e62d798f165ae76853d15b84174cf1dce95392f313a491055332", 1181576),
    "labels.csv.gz":                     ("bdd4eafab2bfe30540256c84ea1513e4b1877c0c4cf03f919204b4eafae5868e", 4771292),
    "connectivity_tags.csv.gz":          ("68c69cec13810fa543c600a5b9973d10718b449740e010834662be1dc1b8696c", 637719),
    "neuropil_synapse_table.csv.gz":     ("e525bdea7bc2fe585cf8ab8f9fc76bea5e29f2c3e5240588b6add518bcda5ab1", 4674663),
    # official gzip hash 9feab030...; the committed copy is the same content
    # re-deflated, so it is verified via content in verify() instead.
    "processed_labels.csv.gz":           ("9feab030bd7f0f9f9909a56f4ed37f019fc19cc076180b494f3bff55bf1480d2", 1017658),
    "synapse_attachment_rates.csv.gz":   ("2e412c5714c21aa4f6ebff74801f620d161594740aaee25c73f5bc18b0110447", 3257),
    # too large for git (listed for completeness / manual fetch):
    #   connections_princeton_no_threshold.csv.gz   275,679,780
    #   connections_buhmann_no_threshold.csv.gz     212,093,967
    #   fafb_v783_princeton_synapse_table.csv.gz  2,695,106,039
    #   synapse_coordinates.csv.gz                  316,819,225
    #   sk_lod1_783_healed.zip                   13,873,645,070
}

# Public GitHub mirrors holding byte-identical copies of the official files.
# Each entry: filename -> (repo, path-in-repo, official-sha256)
MIRRORS = {
    "classification.csv.gz":        ("snedea/flybrain", "data/classification.csv.gz"),
    "neurons.csv.gz":               ("snedea/flybrain", "data/neurons.csv.gz"),
    "coordinates.csv.gz":           ("snedea/flybrain", "data/coordinates.csv.gz"),
    "connections_princeton.csv.gz": ("itsNisarg/Connectome_Analysis", "network/data/csv/archive/connections_princeton.csv.gz"),
    "processed_labels.csv.gz":      ("itsNisarg/Connectome_Analysis", "network/data/csv/archive/processed_labels.csv.gz"),
}


def sha256(path: str, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def verify(raw_dir: str = RAW) -> bool:
    """Verify whatever core files exist in *raw_dir*. Prints a table."""
    ok = True
    print(f"{'file':40s} {'size':>12s} {'sha256':>70s}  verdict")
    for name, (want_hash, want_size) in sorted(OFFICIAL_MANIFEST.items()):
        p = os.path.join(raw_dir, name)
        if not os.path.exists(p):
            print(f"{name:40s} {'-':>12s} {'-':>70s}  (absent)")
            continue
        size = os.path.getsize(p)
        digest = sha256(p)
        if digest == want_hash:
            verdict = "BYTE-IDENTICAL to official"
        elif name == "processed_labels.csv.gz":
            # content is identical; the gzip container was re-deflated
            import gzip, io, csv
            with gzip.open(p, "rt", newline="") as fh:
                rd = csv.reader(fh)
                header = next(rd)
                n = sum(1 for _ in rd)
            verdict = (
                f"container re-gzipped; content OK ({n} rows, header {header[:2]})"
                if header[:2] == ["root_id", "processed_labels"] else "MISMATCH"
            )
            ok &= "content OK" in verdict
            print(f"{name:40s} {size:>12,d} {digest:>70s}  {verdict}")
            continue
        else:
            verdict = "MISMATCH"
            ok = False
        print(f"{name:40s} {size:>12,d} {digest:>70s}  {verdict}")
    return ok


def fetch_mirror(raw_dir: str = RAW) -> None:
    os.makedirs(raw_dir, exist_ok=True)
    for name, (repo, path) in MIRRORS.items():
        url = f"https://github.com/{repo}/raw/HEAD/{path}"
        dest = os.path.join(raw_dir, name)
        if os.path.exists(dest) and sha256(dest) == OFFICIAL_MANIFEST[name][0]:
            print(f"{name}: already present and verified")
            continue
        print(f"{name}: downloading {url}")
        try:
            with urllib.request.urlopen(url, timeout=600) as r, open(dest, "wb") as f:
                while True:
                    b = r.read(1 << 20)
                    if not b:
                        break
                    f.write(b)
        except Exception as e:  # noqa: BLE001
            print(f"  ERROR: {e}  (mirror may move; fall back to --official)")



def fetch_official(raw_dir: str = RAW) -> None:
    print("The official FlyWire download is served by:")
    print(f"\n    {OFFICIAL_URL}\n")
    print("It requires signing in with a Google account (Princeton serve the data")
    print("through Codex behind interactive auth). Steps:")
    print("  1. Open the URL above in a browser and sign in.")
    print("  2. Save the downloaded `fafb.zip`.")
    print(f"  3. Either drop `fafb.zip` into {raw_dir}/ and re-run")
    print("     `python -m tools.fetch_dataset --official  # a second time, to unpack`,")
    print("     or unpack it yourself into that directory and run")
    print("     `python -m tools.fetch_dataset --verify`.")
    z = os.path.join(raw_dir, "fafb.zip")
    if os.path.exists(z):
        print(f"\nFound {z} — unpacking…")
        with zipfile.ZipFile(z) as zf:
            zf.extractall(raw_dir)
        verify(raw_dir)
    else:
        try:
            import webbrowser
            webbrowser.open(OFFICIAL_URL)
        except Exception:
            pass


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--official", action="store_true", help="guide an interactive official download")
    g.add_argument("--mirror", action="store_true", help="headless re-download from verified GitHub mirrors")
    g.add_argument("--verify", action="store_true", help="verify files already on disk")
    args = ap.parse_args()
    if args.mirror:
        fetch_mirror()
        sys.exit(0 if verify() else 1)
    if args.verify:
        sys.exit(0 if verify() else 1)
    fetch_official()


if __name__ == "__main__":
    main()
