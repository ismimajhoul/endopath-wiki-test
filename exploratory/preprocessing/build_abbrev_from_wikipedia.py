# build_abbrev_from_wikipedia.py
# Build abbrev_sure_wikipedia.json (Dict[str,str]) and abbrev_ambigue_wikipedia.json (Dict[str,List[str]])
# Robust: User-Agent + fallback API + fallback HTML scrape (last resort)

import json
import re
import html as _html
from pathlib import Path
from typing import Any, Dict, List

import requests


# ---------------------------------------------------------------------
# Constants / Regex
# ---------------------------------------------------------------------
WIKI_API = "https://fr.wikipedia.org/w/api.php"
PAGE_TITLE = "Liste_d%27abr%C3%A9viations_en_sant%C3%A9"
PAGE_HUMAN_URL = "https://fr.wikipedia.org/wiki/Liste_d%27abr%C3%A9viations_en_sant%C3%A9"

OUT_SURE = Path("abbrev_sure_wikipedia.json")
OUT_AMB = Path("abbrev_ambigue_wikipedia.json")

# [5], [ 12 ] (après unescape: &#91;5&#93; -> [5])
_WIKI_CIT_RE = re.compile(r"\[\s*\d+\s*\]")
# <ref>...</ref> en wikitext/html (par prudence)
_REF_TAG_RE = re.compile(r"<ref[^>]*>.*?</ref>", re.I | re.S)
# espaces multiples
_MULTI_SPACE_RE = re.compile(r"\s+")

# split expansions: virgules, ;, /, |, ou/or (avec espaces)
_SPLIT_RE = re.compile(r"\s*(?:,|;|\||/|\bor\b|\bou\b)\s*", flags=re.IGNORECASE)

# ligne wikitext typique: "* ECBU : examen cyto..."
_LINE_RE = re.compile(r"^\*\s*([^:]+?)\s*:\s*(.+?)\s*$", re.MULTILINE)

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



# ---------------------------------------------------------------------
# HTTP Session
# ---------------------------------------------------------------------
SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "ENDOPATH-AbbrevBot/1.0 (local-script; purpose: research; python requests)",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "fr-FR,fr;q=0.9,en;q=0.8",
})


# ---------------------------------------------------------------------
# Cleaning helpers
# ---------------------------------------------------------------------
def normalize_key(k: str) -> str:
    """Normalize abbreviation key (strip spaces, uppercase)."""
    k = _html.unescape(str(k or "")).strip()
    k = re.sub(r"\s+", "", k)
    return k.upper()


def clean_wiki_expansion(s: Any) -> str:
    """
    Nettoie une expansion issue de Wikipedia/export HTML :
    - decode entités HTML (&#91;5&#93; => [5])
    - supprime <ref>...</ref> éventuels
    - supprime les marqueurs de références [n]
    - supprime artefacts de wikitext simples ([[...]], {{...}})
    - normalise espaces et ponctuation de fin
    """
    if s is None:
        return ""

    s = _html.unescape(str(s))

    # enlever ref tags
    s = _REF_TAG_RE.sub("", s)

    # enlever wikilinks basiques [[...]]
    s = re.sub(r"\[\[|\]\]", "", s)

    # enlever templates {{...}} (non greedy)
    s = re.sub(r"\{\{.*?\}\}", "", s)

    # enlever citations [5]
    s = _WIKI_CIT_RE.sub("", s)

    # normaliser espaces
    s = _MULTI_SPACE_RE.sub(" ", s).strip()

    # retirer ponctuation parasite de fin
    s = s.rstrip(" ,;.")
    return s


def split_expansions(raw: Any) -> List[str]:
    """
    Normalise une expansion en liste de choix UI:
    - si list -> nettoie chaque item
    - si string -> split sur , ; / | ou/or
    - dédoublonne en conservant l'ordre
    """
    if raw is None:
        return []

    if isinstance(raw, list):
        items = raw
    else:
        s = clean_wiki_expansion(raw)
        if not s:
            return []
        items = _SPLIT_RE.split(s)

    out: List[str] = []
    seen = set()

    for x in items:
        x2 = clean_wiki_expansion(x)
        if not x2:
            continue
        k = x2.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(x2)

    return out


# ---------------------------------------------------------------------
# Wikipedia fetch (wikitext)
# ---------------------------------------------------------------------
def fetch_wikitext_via_parse() -> str:
    params = {
        "action": "parse",
        "page": PAGE_TITLE,
        "prop": "wikitext",
        "format": "json",
    }
    r = SESSION.get(WIKI_API, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    return data["parse"]["wikitext"]["*"]


def fetch_wikitext_via_query() -> str:
    params = {
        "action": "query",
        "format": "json",
        "prop": "revisions",
        "titles": PAGE_TITLE,
        "rvprop": "content",
        "rvslots": "main",
        "formatversion": "2",
    }
    r = SESSION.get(WIKI_API, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    pages = data.get("query", {}).get("pages", [])
    if not pages:
        raise RuntimeError("Aucune page retournée par l'API query.")
    revs = pages[0].get("revisions", [])
    if not revs:
        raise RuntimeError("Aucune révision retournée par l'API query.")
    slot = revs[0].get("slots", {}).get("main", {})
    content = slot.get("content")
    if not content:
        raise RuntimeError("Contenu wikitext introuvable dans query/revisions.")
    return content


def fetch_wikitext() -> str:
    try:
        return fetch_wikitext_via_parse()
    except Exception as e1:
        print(f"[WARN] API parse refusée/erreur: {repr(e1)}")

    try:
        return fetch_wikitext_via_query()
    except Exception as e2:
        print(f"[WARN] API query refusée/erreur: {repr(e2)}")

    # dernier recours: HTML
    print("[WARN] Fallback HTML (moins fiable).")
    r = SESSION.get(PAGE_HUMAN_URL, timeout=30)
    r.raise_for_status()
    page_html = r.text

    li_texts = re.findall(r"<li>(.*?)</li>", page_html, flags=re.DOTALL | re.IGNORECASE)
    cleaned = []
    for li in li_texts:
        t = re.sub(r"<.*?>", "", li)
        t = _html.unescape(t)
        t = _MULTI_SPACE_RE.sub(" ", t).strip()
        if ":" in t:
            cleaned.append("* " + t)
    return "\n".join(cleaned)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main() -> int:
    wt = fetch_wikitext()

    sure: Dict[str, str] = {}
    amb: Dict[str, List[str]] = {}

    for m in _LINE_RE.finditer(wt):
        raw_key = m.group(1)
        raw_val = m.group(2)

        key = normalize_key(raw_key)
        if not key or len(key) > 20:
            continue

        # expansions list (nettoyée + split)
        exps = split_expansions(raw_val)

        if not exps:
            continue

        # Clés explicitement interdites en "SÛRE"
        if key in BLACKLIST_SURE:
            amb[key] = exps
            continue

        if len(exps) == 1:
            sure[key] = exps[0]
        else:
            amb[key] = exps


    OUT_SURE.write_text(json.dumps(sure, ensure_ascii=False, indent=2), encoding="utf-8")
    OUT_AMB.write_text(json.dumps(amb, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[OK] {OUT_SURE} -> {len(sure)} entrées")
    print(f"[OK] {OUT_AMB}  -> {len(amb)} entrées (listes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
