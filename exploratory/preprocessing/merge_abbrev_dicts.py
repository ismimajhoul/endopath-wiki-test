import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DICT_DIR = BASE_DIR / "Data" / "DATA_PROCESSED" / "Correction_mots"

LOCAL_SURE = DICT_DIR / "abbrev_sure.json"
LOCAL_AMB  = DICT_DIR / "abbrev_ambigue.json"
WIKI_SURE  = DICT_DIR / "abbrev_sure_wikipedia.json"
WIKI_AMB   = DICT_DIR / "abbrev_ambigue_wikipedia.json"

OUT_SURE = DICT_DIR / "abbrev_sure_merged.json"
OUT_AMB  = DICT_DIR / "abbrev_ambigue_merged.json"

BLACKLIST_PATH = Path("tools") / "abbrev_sure_blacklist.json"

def load_blacklist_sure() -> set[str]:
    try:
        if BLACKLIST_PATH.exists():
            d = json.loads(BLACKLIST_PATH.read_text(encoding="utf-8"))
            return {normalize_key(k) for k in d.keys()}
    except Exception as e:
        print(f"[WARN] Impossible de lire la blacklist {BLACKLIST_PATH}: {e!r}")
    # fallback
    return {"SP", "SU", "RCP"}

BLACKLIST_SURE = load_blacklist_sure()




def load_json(p: Path) -> dict:
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"JSON invalide: {p}\n"
            f"  ligne {e.lineno}, colonne {e.colno}\n"
            f"  message: {e.msg}"
        ) from e



def norm_key(k: str) -> str:
    return "".join(k.split()).upper()


def normalize_dict(d: dict) -> dict:
    out = {}
    for k, v in d.items():
        nk = norm_key(str(k))
        if not nk:
            continue
        out[nk] = str(v).strip()
    return out

def normalize_sure(d: dict) -> dict:
    out = {}
    for k, v in d.items():
        nk = norm_key(str(k))
        if not nk:
            continue
        out[nk] = str(v).strip()
    return out

def normalize_amb(d: dict) -> dict:
    out = {}
    for k, v in d.items():
        nk = norm_key(str(k))
        if not nk:
            continue
        if isinstance(v, list):
            out[nk] = [str(x).strip() for x in v if str(x).strip()]
        else:
            # tolère format "a | b | c"
            s = str(v).strip()
            out[nk] = [x.strip() for x in s.split("|") if x.strip()]
    return out



def main():
    local_sure = normalize_sure(load_json(LOCAL_SURE))
    local_amb  = normalize_amb(load_json(LOCAL_AMB))
    wiki_sure  = normalize_sure(load_json(WIKI_SURE))
    wiki_amb   = normalize_amb(load_json(WIKI_AMB))


    # Fusion SÛR : local prioritaire
    merged_sure = dict(wiki_sure)
    merged_sure.update(local_sure)

    # Fusion AMBIGU : local prioritaire
    merged_amb = dict(wiki_amb)
    merged_amb.update(local_amb)

    # Blacklist définitive des abréviations sûres
    for k in list(merged_sure.keys()):
        if k in BLACKLIST_SURE:
            merged_sure.pop(k, None)
    
    # Si une clé est sûre, elle ne doit pas rester ambiguë
    for k in list(merged_amb.keys()):
        if k in merged_sure:
            merged_amb.pop(k, None)


    OUT_SURE.write_text(json.dumps(merged_sure, ensure_ascii=False, indent=2), encoding="utf-8")
    OUT_AMB.write_text(json.dumps(merged_amb, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[OK] {OUT_SURE} -> {len(merged_sure)} entrées")
    print(f"[OK] {OUT_AMB}  -> {len(merged_amb)} entrées")


if __name__ == "__main__":
    main()
