#!/usr/bin/env python3
"""Check that tests/data holds only public item names and plain numbers.

Every item name found in the test data (first column of the TSV files, JSON file names and
their "pair"/"item" fields) must be one of the 24 public clips (ODAQ, vocadito, OpenGameArt)
or a synthetic signal (`syn-*`). Also flags absolute paths and e-mail addresses.

With `--no-speech` (the public export, D-0276) a speech item (`SPEECH`) is also a finding,
and so is any file or line naming one.

usage: python3 scripts/check-test-data.py [--no-speech] [crate_dir]   (exit 1 on any finding)
"""
import json
import pathlib
import re
import sys

PUBLIC = {
    "acoustic-guitar", "attack-castanets", "attack-clapping", "attack-claves", "brass-trumpet",
    "classical-violin", "electronic-retrowave", "electronic-slipstream",
    "electronic-synth-dialogue", "jazz-ensemble", "piano-regret", "piano-vampire",
    "preecho-glockenspiel", "rock-fight-them", "rock-metal-better-future",
    "rock-metal-boss-battle", "rock-theme-song", "speech-applause", "strings-violin-solo",
    "vocal-choral", "vocal-solo-english", "vocal-solo-tagalog-a", "vocal-solo-tagalog-b",
    "vocal-solo-vocalise",
}

# Items whose content is speech: kept out of the public export, its tests and its CI.
SPEECH = {"speech-applause"}


def item_of(name: str) -> str:
    """`acoustic-guitar.sori64` -> `acoustic-guitar`; synthetic names stay whole."""
    return name.split(".", 1)[0]


def allowed(item: str) -> bool:
    return item in PUBLIC or item.startswith("syn-")


def main() -> int:
    args = sys.argv[1:]
    no_speech = "--no-speech" in args
    args = [a for a in args if a != "--no-speech"]
    root = pathlib.Path(args[0] if args else pathlib.Path(__file__).parent.parent)
    data = root / "tests" / "data"
    findings, items, files, size = [], set(), 0, 0
    for path in sorted(data.rglob("*")):
        if not path.is_file():
            continue
        files += 1
        size += path.stat().st_size
        text = path.read_text(encoding="utf-8")
        rel = path.relative_to(root)
        if re.search(r"(^|[\s\"'])/(home|Users|mnt|tmp)/", text):
            findings.append(f"{rel}: absolute path")
        if re.search(r"[\w.+-]+@[\w-]+\.[\w.]+", text):
            findings.append(f"{rel}: e-mail address")
        if no_speech and any(s in text or s in path.name for s in SPEECH):
            findings.append(f"{rel}: names a speech item")
        if path.suffix == ".tsv" or path.name.endswith("md5.txt"):
            all_lines = [l for l in text.splitlines() if l]
            col = 0
            head = all_lines[0].lstrip("# ").split("\t") if all_lines else []
            if all_lines and (all_lines[0].startswith("#") or head[0] in {"pair", "item"}):
                # Header row: take the column named item/pair.
                names = [h.split()[0] if h.split() else h for h in head]
                col = next((i for i, h in enumerate(names) if h in {"item", "pair"}), 0)
                all_lines = all_lines[1:]
            for line in all_lines:
                if line.startswith("#"):
                    continue
                cell = line.split()[-1] if path.name.endswith("md5.txt") else line.split("\t")[col]
                items.add(item_of(pathlib.Path(cell).name))
        elif path.suffix == ".json":
            items.add(item_of(path.stem))
            obj = json.loads(text)
            for key in ("pair", "item"):
                if isinstance(obj, dict) and key in obj:
                    items.add(item_of(str(obj[key])))
    bad = sorted(i for i in items if not allowed(i))
    findings += [f"non-public item name: {i}" for i in bad]
    if no_speech:
        findings += [f"speech item: {i}" for i in sorted(items & SPEECH)]
    public = sorted(i for i in items if i in PUBLIC)
    synthetic = sorted(i for i in items if i.startswith("syn-"))
    print(f"files {files}, {size / 1024:.0f} KiB; items: {len(public)} public "
          f"({len(items & SPEECH)} speech), {len(synthetic)} synthetic, {len(bad)} other")
    for f in findings:
        print("FINDING:", f)
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
