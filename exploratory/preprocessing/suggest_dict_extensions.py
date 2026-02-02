#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
suggest_dict_extensions.py

But:
- Lire tokens_a_corriger.csv + tokens_valides.csv
- Lire le dictionnaire de corrections (dictionnaire_correction.json)
- Proposer des suggestions classées par "familles" (CSV) consommées par app.py

Objectifs de cette version (merge "sans régression") :
1) Conserver les sorties existantes :
   - suggestions_manual_dict.csv
   - suggestions_auto_diacritics.csv
   - suggestions_auto_typos.csv
   - suggestions_auto_abbrev.csv
   - suggestions_auto_rejected.csv
2) Ajouter/renforcer :
   - DIACRITICS_STRICT vs DIACRITICS_MULTI (2 CSV dédiés)
   - ABBREV_CANDIDATE vs ABBREV_AMBIGU (2 CSV dédiés)
   - Prise en compte d'un dictionnaire d'abréviations ambiguës (abbrev_ambigue.json)
     et d'un dictionnaire d'abréviations sûres (abbrev_sure.json) (optionnels)

Notes importantes :
- Les colonnes minimales attendues côté app.py : token_source, match, count, score, edit_dist, category.
  Si certaines manquent, app.py normalise.
- Les règles de PRIORITÉ de classification sont essentielles pour éviter les régressions :
  DICT -> DIACRITICS -> TYPO -> ABBREV -> ENRICH -> REJECTED
"""

from __future__ import annotations

import argparse
import json
import os
import html
print("[DEBUG] Running:", os.path.abspath(__file__))
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


import pandas as pd
from rapidfuzz.distance import Levenshtein

# Optionnel (typo heuristics)
try:
    from wordfreq import zipf_frequency
except Exception:
    zipf_frequency = None


import re
from typing import Any, Dict, List

try:
    # si spacy est dispo dans ce script
    from spacy.lang.fr.stop_words import STOP_WORDS as FR_STOPWORDS
except Exception:
    FR_STOPWORDS = set()

# split expansions: virgule, point-virgule, "ou"/"or", slash, pipe…
_SPLIT_RE = re.compile(r"\s*(?:,|;|\bor\b|\bou\b|/|\||\u2013|\u2014)\s*", flags=re.IGNORECASE)

# forme "sigle" plausible (MAJ) : 2..6 caractères alphanum
_SIGLE_RE = re.compile(r"^[A-Z0-9]{2,6}$")

# whitelist de minuscules à accepter comme abréviations ambiguës
ABBREV_LOWER_WHITELIST = {"sf", "tt", "cp", "qq", "j"}

# blacklist minimale (en plus des stopwords spacy)
AMBIGU_BLACKLIST = {
    "pas", "est", "des", "ce", "se", "si", "la", "les", "du", "de", "en", "a",
}


_WORD_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ]+", re.UNICODE)

VOWELS = set("aeiouyàâäæéèêëîïôöœùûüÿ")

# ---------------------------------------------------------------------
# Abbreviation heuristics
# ---------------------------------------------------------------------

# We only want ABBREV ? to capture *rare* short tokens that look like sigles,
# not common French function words (pas, sur, donc, ...).
ZIPF_MAX_ABBREV_CANDIDATE = 3.5  # lower = rarer; 3.0 is already quite permissive
ABBREV_MAX_LEN = 6

# Minimal French stopwords list to prevent false positives in ABBREV ?
STOPWORDS_FR = {
    "a","à","au","aux","avec","ce","ces","cette","chez","dans","de","des","du","elle","en","et","est","il","ils",
    "je","la","le","les","leur","lui","ma","mais","me","meme","même","mes","mon","ne","ni","non","nos","notre",
    "nous","on","ou","par","pas","plus","pour","qu","que","qui","sa","se","ses","son","sur","ta","te","tes","toi",
    "ton","tu","un","une","vos","votre","vous","y","car","donc","très","tres","mal",
}

_UPPER_TOKEN_RE = re.compile(r"^[A-Z0-9][A-Z0-9+\-_/]*$")  # ECBU, HPV, RCP, JJ, etc.

from spacy.lang.fr.stop_words import STOP_WORDS as FR_STOPWORDS

def is_common_function_word(tok: str) -> bool:
    """
    Mots-outils / tokens trop courts => on ne propose jamais de correction d'accent.
    """
    t = (tok or "").strip().lower()
    if not t:
        return True
    if len(t) <= 2:
        return True
    if t in FR_STOPWORDS:
        return True
    return False

def is_upper_token(tok: str) -> bool:
    tok = tok.strip()
    return bool(_UPPER_TOKEN_RE.match(tok))

def merge_abbrev_dicts(base: dict, wiki: dict) -> dict:
    out = dict(base)
    for k, v in wiki.items():
        k = k.strip().upper()
        if k not in out:
            out[k] = v
    return out

def merge_abbrev_sure(base: Dict[str, Any], wiki: Dict[str, Any]) -> Dict[str, str]:
    """
    Fusion pour ABBREV_SURE:
    - base prioritaire (n'écrase jamais)
    - wiki complète les clés manquantes
    - valeur finale: str nettoyée
    """
    out: Dict[str, str] = {}

    # base prioritaire
    for k, v in (base or {}).items():
        kk = str(k).strip().upper()
        vv = str(v).strip()
        if kk and vv:
            out[kk] = vv

    # wiki en complément
    for k, v in (wiki or {}).items():
        kk = str(k).strip().upper()
        vv = str(v).strip()
        if kk and vv and kk not in out:
            out[kk] = vv

    return out


def merge_abbrev_ambigue(
    base: Dict[str, Any],
    wiki: Dict[str, Any],
) -> Dict[str, List[str]]:
    """
    Fusion pour ABBREV_AMBIGUE:
    - union des expansions (base + wiki)
    - déduplication
    - normalisation: List[str] nettoyée
    - filtrage des clés non-sigles (stopwords, tokens langue)
    - split des expansions Wikipedia "a, b ou c" -> ["a","b","c"]
    """

    def _split_expansions(raw: Any) -> List[str]:
        """
        Normalise une valeur d'expansion en liste exploitable UI.
        - décode HTML (&#91;5&#93; -> [5])
        - supprime [n]
        - split sur virgule / 'ou' / 'or' / slash / pipe
        - dédoublonne en conservant l'ordre
        """
        if raw is None:
            return []

        # 1) flatten: on accepte string ou list (ou list de strings)
        if isinstance(raw, list):
            candidates = raw
        else:
            s = str(raw).strip()
            if not s:
                return []
            candidates = [s]

        items: List[str] = []
        for c in candidates:
            s = html.unescape(str(c or "")).strip()
            if not s:
                continue

            # retire refs [5] après unescape
            s = re.sub(r"\[\s*\d+\s*\]", "", s).strip()

            # split "a, b ou c / d | e"
            parts = _SPLIT_RE.split(s)
            for p in parts:
                p2 = str(p).strip()
                p2 = html.unescape(p2).strip()
                p2 = re.sub(r"\[\s*\d+\s*\]", "", p2).strip()
                if p2:
                    items.append(p2)

        # 2) dédoublonnage en conservant l'ordre
        uniq: List[str] = []
        seen = set()
        for x in items:
            k = x.lower()
            if k not in seen:
                uniq.append(x)
                seen.add(k)

        return uniq

    def _is_blacklisted_key(k: str) -> bool:
        t = (k or "").strip().lower()
        if not t:
            return True
        if t in AMBIGU_BLACKLIST:
            return True
        if t in FR_STOPWORDS:
            return True
        return False

    def _accept_key(raw_key: str) -> str:
        """
        Retourne une clé normalisée si acceptée, sinon "".
        - MAJ sigle par défaut
        - minuscules acceptées uniquement via whitelist
        """
        rk = (raw_key or "").strip()
        if not rk:
            return ""

        # stopwords / blacklist (on teste sur la forme lower brute)
        if _is_blacklisted_key(rk):
            return ""

        # minuscules : uniquement whitelist
        if rk.islower():
            if rk.lower() not in ABBREV_LOWER_WHITELIST:
                return ""
            # on garde en lower ici (l'écriture CSV pourra upper si tu veux)
            # mais pour la clé de merge, on conserve "sf" (pas "SF")
            return rk.lower()

        # sinon : on force MAJ
        kk = rk.upper()

        # filtrage forme sigle plausible
        if not _SIGLE_RE.match(kk):
            return ""

        # blacklist/stopwords après upper (ex: "A" -> à éviter)
        if _is_blacklisted_key(kk):
            return ""

        return kk

    out: Dict[str, List[str]] = {}

    # base
    for k, v in (base or {}).items():
        kk = _accept_key(str(k))
        if not kk:
            continue
        lst = _split_expansions(v)
        out.setdefault(kk, [])
        if lst:
            out[kk] = _merge_lists(out[kk], lst)

    # wiki (fusion)
    for k, v in (wiki or {}).items():
        kk = _accept_key(str(k))
        if not kk:
            continue
        lst = _split_expansions(v)
        out.setdefault(kk, [])
        if lst:
            out[kk] = _merge_lists(out[kk], lst)

    return out


def _merge_lists(a: List[str], b: List[str]) -> List[str]:
    """Union a+b avec dédoublonnage conservant l'ordre."""
    out = []
    seen = set()
    for x in (a or []) + (b or []):
        s = str(x).strip()
        if not s:
            continue
        k = s.lower()
        if k in seen:
            continue
        out.append(s)
        seen.add(k)
    return out

def _norm(s: str) -> str:
    return str(s).strip().lower()


def _strip_accents(s: str) -> str:
    # on évite unicodedata pour rester simple : normalisation NFD
    import unicodedata
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def _is_word_like(token: str) -> bool:
    return bool(_WORD_RE.fullmatch(token))


def _consonant_ratio(token: str) -> float:
    letters = [c for c in token.lower() if c.isalpha()]
    if not letters:
        return 0.0
    consonants = [c for c in letters if c not in VOWELS]
    return len(consonants) / max(1, len(letters))


def read_csv_tokens(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "token_source" not in df.columns:
        if "token" in df.columns:
            df = df.rename(columns={"token": "token_source"})
        else:
            raise ValueError(f"CSV invalide: colonne token_source absente: {path}")
    if "count" not in df.columns:
        df["count"] = 0

    if "word_like" not in df.columns:
        df["word_like"] = df["token_source"].astype(str).apply(lambda x: 1 if _is_word_like(_norm(x)) else 0)

    if "zipf" not in df.columns:
        # zipf_frequency = optionnel
        if zipf_frequency is not None:
            df["zipf"] = df["token_source"].astype(str).apply(lambda x: float(zipf_frequency(_norm(x), "fr")))
        else:
            df["zipf"] = 0.0

    return df


def read_json(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    # normalisation des clés en lower (convention actuelle du projet)
    out = {}
    for k, v in obj.items():
        out[_norm(k)] = str(v).strip()
    return out


def read_json_list_map(path: Path) -> Dict[str, List[str]]:
    """
    Pour abbrev_ambigue.json : {"TV": ["toucher vaginal", "télévision"], ...}
    On retourne keys normalisées en UPPER (pour matcher des tokens comme "TV", "tv", "Tv")
    """
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    out: Dict[str, List[str]] = {}
    for k, v in obj.items():
        key = str(k).strip().upper()
        if isinstance(v, list):
            out[key] = [str(x).strip() for x in v if str(x).strip()]
        else:
            out[key] = [str(v).strip()]
    return out


def write_json_if_missing(path: Path, content: dict) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(content, f, ensure_ascii=False, indent=2)


def build_valid_list(df_valid: pd.DataFrame) -> List[str]:
    vals = []
    for x in df_valid["token_source"].astype(str).tolist():
        t = _norm(x)
        if t:
            vals.append(t)
    return sorted(set(vals))


@dataclass
class Suggestion:
    token_source: str
    match: str = ""
    count: int = 0
    score: float = 0.0
    edit_dist: int = 0
    category: str = ""
    expansions: str = ""   # <-- NEW (pour ABBREV_AMBIGU)

    def to_row(self) -> dict:
        return {
            "token_source": self.token_source,
            "match": self.match,
            "count": self.count,
            "score": self.score,
            "edit_dist": self.edit_dist,
            "category": self.category,
            "expansions": self.expansions,  # <-- NEW
        }

# -----------------------------------------------------------------
# Guard-fous DIACRITICS (anti régression sur mots-outils)
# -----------------------------------------------------------------
def _diacritics_allowed(tok: str) -> bool:
    t = (tok or "").strip().lower()
    if not t:
        return False
    # On évite les mots très courts et les mots-outils (de/du/la/au/et/…)
    if len(t) <= 3:
        return False
    if t in FR_STOPWORDS:
        return False
    return True


# -----------------------------------------------------------------
# INTENSITY / GRADATION clinique : mot++ / mot+++ => mot (++) / mot (+++)
# -----------------------------------------------------------------
_INTENSITY_RE = re.compile(
    r"^(?P<base>[a-zàâçéèêëîïôûùüÿñæœ\-]+)(?P<pluses>\+{2,})$",
    re.IGNORECASE,
)


def suggest_diacritics_strict(df_fix: pd.DataFrame, valid_list: List[str], covered: set) -> List[Suggestion]:
    # strict = 1 mot, même chaîne sans accents
    by_stripped: Dict[str, str] = {}
    for v in valid_list:
        vv = _norm(v)
        if not vv:
            continue
        # garde-fou: on ne veut pas que des stopwords alimentent la table de correction
        if is_common_function_word(vv):
            continue
        by_stripped.setdefault(_strip_accents(vv), vv)

    out: List[Suggestion] = []
    for _, r in df_fix.iterrows():
        src = str(r.get("token_source", ""))
        tok = _norm(src)

        if not tok or tok in covered:
            continue
        # Guard-fous anti régression accents (de/du/la/…)
        if not _diacritics_allowed(tok):
            continue
        if int(r.get("word_like", 0) or 0) != 1:
            continue
        if " " in tok:
            continue

        # garde-fou: jamais de diacritiques sur mots-outils / tokens courts
        if is_common_function_word(tok):
            continue

        key = _strip_accents(tok)
        v = by_stripped.get(key)
        if not v or v == tok:
            continue

        # garde-fou: on ne propose une correction que si on AJOUTE des accents
        # (évite des corrections absurdes sur des mots déjà accentués / variations)
        if (_strip_accents(tok) == tok) and (_strip_accents(v) != v):
            out.append(
                Suggestion(
                    token_source=src,
                    match=v,
                    count=int(r.get("count", 0) or 0),
                    score=100.0,  # score symbolique
                    edit_dist=0,
                    category="DIACRITICS_STRICT",
                )
            )

    return out



def suggest_diacritics_multi(
    df_fix: pd.DataFrame,
    valid_list: List[str],
    covered: set
) -> List[Suggestion]:
    """
    Multi-mots : ne s'active que si token_source contient des espaces.
    On corrige mot par mot UNIQUEMENT quand :
      - le mot est "word-like"
      - ce n'est pas un mot-outil
      - on AJOUTE des accents (jamais l'inverse)
    """

    # Index : forme désaccentuée -> forme valide accentuée
    by_stripped: Dict[str, str] = {}
    for v in valid_list:
        vv = _norm(v)
        if not vv:
            continue
        if is_common_function_word(vv):
            continue
        by_stripped.setdefault(_strip_accents(vv), vv)

    out: List[Suggestion] = []

    for _, r in df_fix.iterrows():
        src = str(r.get("token_source", ""))
        tok = _norm(src)

        if not tok or tok in covered:
            continue
        if int(r.get("word_like", 0) or 0) != 1:
            continue
        if " " not in tok:
            continue

        parts = tok.split()
        mapped = []
        changed = False

        for p in parts:
            pp = _norm(p)

            # Garde-fou absolu : pas de correction sur mots-outils / très courts
            if is_common_function_word(pp):
                mapped.append(p)
                continue

            cand = by_stripped.get(_strip_accents(pp), pp)

            # On corrige UNIQUEMENT si :
            # - le token courant n'a PAS d'accent
            # - le candidat en A
            if (
                _strip_accents(pp) == pp and
                _strip_accents(cand) != cand and
                cand != pp
            ):
                mapped.append(cand)
                changed = True
            else:
                mapped.append(p)

        if changed:
            out.append(
                Suggestion(
                    token_source=src,
                    match=" ".join(mapped),
                    count=int(r.get("count", 0) or 0),
                    score=100.0,
                    edit_dist=0,
                    category="DIACRITICS_MULTI",
                )
            )

    return out


def suggest_intensity_markers(df_fix: pd.DataFrame, covered: set) -> List[Suggestion]:
    """
    Détecte les tokens cliniques de type:
      - "douleur++" / "souple+++" / "invalidantes++++"
    et propose:
      - "douleur (++)" / "souple (+++)" / ...
    """
    out: List[Suggestion] = []

    for _, r in df_fix.iterrows():
        src = str(r.get("token_source", ""))
        tok = _norm(src)
        if not tok or tok in covered:
            continue

        m = _INTENSITY_RE.match(tok)
        if not m:
            continue

        base = m.group("base")
        pluses = m.group("pluses")
        match = f"{base} ({pluses})"

        out.append(Suggestion(
            token_source=src,
            match=match,
            count=int(r.get("count", 0)),
            score=100.0,
            edit_dist=0,
            category="INTENSITY",
        ))

    return out


def suggest_typos(
    df_fix: pd.DataFrame,
    valid_list: List[str],
    covered: set,
    max_dist: int,
    min_zipf: float,
) -> List[Suggestion]:
    """
    Version "TYPO stricte" (faibles faux positifs) :
    - tokens >= 5
    - uniquement lettres (pas chiffres, pas -, pas ', pas .)
    - pas FULL UPPER (souvent abrév.)
    - correction candidate à distance 1 (ou <= max_dist si tu veux)
    - la correction est nettement plus fréquente (Zipf delta)
    """
    out: List[Suggestion] = []

    for _, r in df_fix.iterrows():
        src = str(r["token_source"])
        raw = src.strip()
        tok = _norm(raw)

        if not tok or tok in covered:
            continue
        if int(r.get("word_like", 0)) != 1:
            continue

        # -------------------------
        # Filtres "stricts"
        # -------------------------
        if len(tok) < 5:
            continue
        if any(ch.isdigit() for ch in raw):
            continue
        if "-" in raw or "." in raw or "'" in raw or "’" in raw:
            continue
        # éviter les abréviations / sigles
        letters = [c for c in raw if c.isalpha()]
        if letters and all(c.isupper() for c in letters):
            continue

        zipf_tok = float(r.get("zipf", 0.0))

        # si déjà "fréquent", on ne corrige pas
        if zipf_tok > 0 and zipf_tok >= min_zipf:
            continue

        # -------------------------
        # Recherche meilleur candidat
        # -------------------------
        best = None
        best_dist = 999
        best_zipf = 0.0

        for v in valid_list:
            if tok == v:
                continue
            d = Levenshtein.distance(tok, v)
            if d <= max_dist and d < best_dist:
                best = v
                best_dist = d
                best_zipf = _zipf_fr(v)
                if best_dist == 1:
                    break

        if not best:
            continue

        # on ne garde que les corrections "quasi certaines"
        if best_dist != 1:
            continue

        # la correction doit être significativement plus fréquente
        # (ajuste 1.0 -> 1.5 si tu veux encore moins de bruit)
        if best_zipf > 0 and zipf_tok > 0:
            if (best_zipf - zipf_tok) < 1.0:
                continue

        out.append(
            Suggestion(
                token_source=src,
                match=best,
                count=int(r.get("count", 0)),
                score=float(100 - best_dist * 10),
                edit_dist=int(best_dist),
                category="TYPO",
            )
        )

    return out

def _zipf_fr(token: str) -> float:
    if zipf_frequency is None:
        return 0.0
    try:
        return float(zipf_frequency(token, "fr"))
    except Exception:
        return 0.0

def is_abbrev_candidate(tok: str) -> bool:
    t = tok.strip()
    if not t:
        return False
    # Heuristique simple : court, souvent MAJ, ou mixte (sigles), ou contient des points
    if len(t) > ABBREV_MAX_LEN:
        return False
    if "." in t:
        return True
    # sigles : au moins 2 lettres, majoritairement majuscules
    letters = [c for c in t if c.isalpha()]
    if len(letters) < 2:
        return False
    upp = sum(1 for c in letters if c.isupper())
    low = sum(1 for c in letters if c.islower())
    # “AW”, “RCP”, “HPV”, “DIU” etc.
    if upp >= 2 and low == 0:
        return True
    # “tono” / “dig” : ça peut être une abréviation “forme mot”
    # mais on ne la garde candidate que si elle n’est pas un mot fréquent FR
    return False

def _ensure_token_source(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "token_source" not in df.columns:
        if "token" in df.columns:
            df["token_source"] = df["token"].astype(str)
        else:
            raise ValueError("Input dataframe must contain 'token_source' or 'token' column.")
    if "count" not in df.columns:
        df["count"] = 0
    if "word_like" not in df.columns:
        df["word_like"] = 1
    if "zipf" not in df.columns:
        df["zipf"] = df["token_source"].astype(str).apply(lambda t: _zipf_fr(t))
    return df[["token_source", "count", "word_like", "zipf"]]

from typing import Dict, List, Tuple

def _ensure_token_source_for_abbrev(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise df_fix / df_valid pour l'étape abréviations.
    Colonnes minimales attendues en sortie :
      - token_source (str)
      - count (int)
      - word_like (int 0/1)
      - zipf (float)  [si wordfreq dispo]
    """
    df = df.copy()

    if "token_source" not in df.columns:
        if "token" in df.columns:
            df["token_source"] = df["token"].astype(str)
        else:
            raise ValueError("df must contain 'token_source' or 'token'")

    if "count" not in df.columns:
        df["count"] = 0

    if "word_like" not in df.columns:
        # par défaut on considère 'word_like' (sinon trop restrictif)
        df["word_like"] = 1

    if "zipf" not in df.columns:
        if zipf_frequency is None:
            df["zipf"] = 0.0
        else:
            df["zipf"] = df["token_source"].astype(str).apply(lambda t: float(zipf_frequency(t, "fr")))

    return df[["token_source", "count", "word_like", "zipf"]]


def suggest_abbrev(
    df_fix: pd.DataFrame,
    df_valid: pd.DataFrame,
    covered: set,
    abbrev_sure: Dict[str, str],
    abbrev_ambigue: Dict[str, List[str]],
) -> Tuple[List[Suggestion], List[Suggestion], List[Suggestion]]:
    """
    Retourne (sure, ambigu, candidate).

    Règles:
      - SÛRE : UNIQUEMENT si token.upper() est dans abbrev_sure.json.
      - AMBIGUË : si token.upper() est dans abbrev_ambigue.json (jamais auto).
      - CANDIDATE : ressemble à une abréviation, mais pas dans les dictionnaires,
                    et pas un mot courant FR, et pas déjà couvert (DICT/DIACRITICS/TYPO/...).
    """
    out_sure: List[Suggestion] = []
    out_amb: List[Suggestion] = []
    out_cand: List[Suggestion] = []

    df_abbrev = pd.concat(
        [
            _ensure_token_source_for_abbrev(df_fix),
            _ensure_token_source_for_abbrev(df_valid),
        ],
        ignore_index=True,
    ).drop_duplicates(subset=["token_source"], keep="first")

    for _, r in df_abbrev.iterrows():
        src = str(r["token_source"])
        tok = src.strip()
        tok_norm = _norm(tok)
        if not tok_norm:
            continue

        # (0) Si déjà traité ailleurs (DICT/PHRASE/DIACRITICS/TYPO/...) => jamais abréviation
        if tok_norm in covered:
            continue

        key_upper = tok.upper()

        # (1) AMBIGUË (priorité) : toujours choix utilisateur
        if key_upper in abbrev_ambigue:
            out_amb.append(
                Suggestion(
                    token_source=src,
                    match="",  # UI : "(à décider)"
                    count=int(r.get("count", 0)),
                    score=0.0,
                    edit_dist=0,
                    category="ABBREV_AMBIGU",
                )
            )
            continue

        # (2) SÛRE : UNIQUEMENT via abbrev_sure.json + token en MAJUSCULES dans le texte
        if key_upper in abbrev_sure and is_upper_token(tok):
            out_sure.append(
                Suggestion(
                    token_source=src,  # src tel quel (case du texte)
                    match=abbrev_sure[key_upper],
                    count=int(r.get("count", 0)),
                    score=100.0,
                    edit_dist=0,
                    category="ABBREV_SURE",
                )
            )
            continue


        # (3) CANDIDATS : heuristique stricte
        if int(r.get("word_like", 0)) != 1:
            continue

        t_low = tok.lower()
        if t_low in STOPWORDS_FR:
            continue

        # éviter de proposer des mots courants comme abréviations
        zipf = float(r.get("zipf", 0.0))
        if zipf_frequency is not None and zipf > ZIPF_MAX_ABBREV_CANDIDATE:
            continue

        if len(tok) > ABBREV_MAX_LEN:
            continue

        if is_abbrev_candidate(tok):
            out_cand.append(
                Suggestion(
                    token_source=src,
                    match="",  # "(à décider)"
                    count=int(r.get("count", 0)),
                    score=0.0,
                    edit_dist=0,
                    category="ABBREV_CANDIDATE",
                )
            )

    return out_sure, out_amb, out_cand


def suggest_enrich_metier(df_fix: pd.DataFrame, covered: set) -> List[Suggestion]:
    """
    “À enrichir métier” : on capte des tokens non couverts, word-like,
    avec zipf très faible (mot rare) ou forme atypique.
    """
    out: List[Suggestion] = []
    for _, r in df_fix.iterrows():
        src = str(r["token_source"])
        tok = _norm(src)
        if not tok or tok in covered:
            continue
        if int(r.get("word_like", 0)) != 1:
            continue

        zipf = float(r.get("zipf", 0.0))
        # seuil volontairement bas
        if zipf <= 1.5:
            out.append(
                Suggestion(
                    token_source=src,
                    match="",
                    count=int(r.get("count", 0)),
                    score=0.0,
                    edit_dist=0,
                    category="ENRICH_METIER",
                )
            )
    return out

def write_abbrev_sure_csv(out_path: Path, abbrev_sure: Dict[str, str]) -> None:
    rows = []
    for k_upper, expansion in sorted(abbrev_sure.items()):
        k_upper = str(k_upper).strip().upper()
        if not k_upper:
            continue
        rows.append({
            "token_source": k_upper,          # <-- MAJUSCULES (pas lower)
            "match": str(expansion).strip(),
            "count": 0,
            "score": 100.0,
            "edit_dist": 0,
            "category": "ABBREV_SURE",
            "expansions": "",
        })

    df = pd.DataFrame(
        rows,
        columns=["token_source", "match", "count", "score", "edit_dist", "category", "expansions"]
    )
    df.to_csv(out_path, index=False, encoding="utf-8")

def write_abbrev_amb_csv(out_path: Path, abbrev_ambigue: Dict[str, List[str]]) -> None:
    rows = []
    for k_upper, exps in sorted(abbrev_ambigue.items()):
        k_upper = str(k_upper).strip().upper()
        if not k_upper:
            continue
        if not isinstance(exps, list):
            exps = [str(exps)]
        exps_clean = [str(x).strip() for x in exps if str(x).strip()]
        rows.append({
            "token_source": k_upper,   # <-- MAJUSCULES
            "match": "",
            "count": 0,
            "score": 0.0,
            "edit_dist": 0,
            "category": "ABBREV_AMBIGU",
            "expansions": " | ".join(exps_clean),
        })

    df = pd.DataFrame(rows, columns=["token_source", "match", "count", "score", "edit_dist", "category", "expansions"])
    df.to_csv(out_path, index=False, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens-fix", default="Data/DATA_PROCESSED/tokens_a_corriger.csv")
    parser.add_argument("--tokens-valid", default="Data/DATA_PROCESSED/tokens_valides.csv")
    parser.add_argument("--dict-json", default="Data/DATA_PROCESSED/Correction_mots/dictionnaire_correction.json")
    parser.add_argument("--out-dir", default="Data/DATA_PROCESSED/Correction_mots")

    # typo config
    parser.add_argument("--typo-max-dist", type=int, default=2)
    parser.add_argument("--typo-min-zipf", type=float, default=2.5)

    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    path_fix = (base_dir / args.tokens_fix).resolve()
    path_valid = (base_dir / args.tokens_valid).resolve()
    path_dict = (base_dir / args.dict_json).resolve()
    out_dir = (base_dir / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Sorties
    out_manual = out_dir / "suggestions_manual_dict.csv"

    out_diac = out_dir / "suggestions_auto_diacritics.csv"
    out_diac_strict = out_dir / "suggestions_auto_diacritics_strict.csv"
    out_diac_multi = out_dir / "suggestions_auto_diacritics_multi.csv"

    out_typos = out_dir / "suggestions_auto_typos.csv"
    out_intensity = out_dir / "suggestions_auto_intensity.csv"

    out_abbr_sure = out_dir / "suggestions_auto_abbrev.csv"
    out_abbr_amb = out_dir / "suggestions_auto_abbrev_ambigu.csv"
    out_abbr_cand = out_dir / "suggestions_auto_abbrev_candidate.csv"

    out_enrich = out_dir / "suggestions_domain_enrich.csv"
    out_rej = out_dir / "suggestions_auto_rejected.csv"


    # -------------------------------------------------------------
    # Dictionnaires d'abréviations (base + wikipedia)
    # -------------------------------------------------------------
    path_abbrev_sure = out_dir / "abbrev_sure.json"
    path_abbrev_amb = out_dir / "abbrev_ambigue.json"
    path_abbrev_sure_wiki = out_dir / "abbrev_sure_wikipedia.json"
    path_abbrev_amb_wiki = out_dir / "abbrev_ambigue_wikipedia.json"

    # Templates si absents
    write_json_if_missing(
        path_abbrev_sure,
        {
            "IRM": "imagerie par résonance magnétique",
            "CI": "contre-indication",
            "POP": "pilule progestative",
            "RAS": "rien à signaler",
            "DIU": "dispositif intra-utérin",
            "HPV": "papillomavirus humain",
            "CNGOF": "Collège National des Gynécologues et Obstétriciens Français",
        },
    )
    write_json_if_missing(
        path_abbrev_amb,
        {
            "TV": ["toucher vaginal", "télévision"],
            "AW": ["arrêt de travail"],
            "CR": ["compte rendu", "créatinine"],
            "DIG": ["digestif", "digital", "digestion"],
            "SD": ["sans douleur", "syndrome (à préciser)"],
            "SU": ["suivi (à préciser)", "sans urgence (à préciser)"],
            "EC": ["échographie", "examen clinique"],
            "TONO": ["tonus", "tonométrie (à préciser)"],
            "LUS": ["ligament utéro-sacré", "autre (à préciser)"],
        },
    )

    # -------------------------------------------------------------
    # Lecture des entrées
    # -------------------------------------------------------------
    print(f"[INFO] Lecture tokens à corriger : {path_fix}")
    df_fix = read_csv_tokens(path_fix)

    print(f"[INFO] Lecture tokens valides : {path_valid}")
    df_valid = read_csv_tokens(path_valid)

    print(f"[INFO] Lecture dictionnaire : {path_dict}")
    dict_all = read_json(path_dict)

    valid_list = build_valid_list(df_valid)

    # Split dict entries :
    # - single word => DICT
    # - multi word  => PHRASE_DICT (géré runtime dans app.py)
    dict_single = {k: v for k, v in dict_all.items() if " " not in k}

    # -------------------------------------------------------------
    # Merge abréviations base + wiki (wiki optionnel)
    # -------------------------------------------------------------
    abbrev_sure_base = read_json(path_abbrev_sure)
    abbrev_sure_wiki = read_json(path_abbrev_sure_wiki) if path_abbrev_sure_wiki.exists() else {}
    abbrev_sure = merge_abbrev_sure(abbrev_sure_base, abbrev_sure_wiki)
    # -> dict normalisé: { "ECBU": "....", ... }

    abbrev_amb_base = read_json_list_map(path_abbrev_amb)
    abbrev_amb_wiki = read_json_list_map(path_abbrev_amb_wiki) if path_abbrev_amb_wiki.exists() else {}
    abbrev_ambigue = merge_abbrev_ambigue(abbrev_amb_base, abbrev_amb_wiki)
    # -> dict normalisé: { "TV": ["...", "..."], ... }

    print(
        "[INFO] Abbrev chargées: "
        f"sure={len(abbrev_sure)} (base+wiki), "
        f"ambigue={len(abbrev_ambigue)} (base+wiki)"
    )

    # IMPORTANT: un seul covered, alimenté progressivement, jamais réinitialisé
    covered: set[str] = set()

    # -----------------------------------------------------------------
    # DICT (manuel validé)
    # -----------------------------------------------------------------
    rows_manual = []
    for _, r in df_fix.iterrows():
        src = str(r.get("token_source", ""))
        tok = _norm(src)
        if not tok:
            continue
        if tok in dict_single:
            rows_manual.append(
                Suggestion(
                    token_source=src,
                    match=dict_single[tok],
                    count=int(r.get("count", 0)),
                    score=100.0,
                    edit_dist=0,
                    category="MANUAL_DICT",
                ).to_row()
            )
            covered.add(tok)

    df_manual = pd.DataFrame(rows_manual)
    df_manual.to_csv(out_manual, index=False, encoding="utf-8")
    print(f"[OK] Manual dict          : {len(df_manual)} -> {out_manual}")

    # -----------------------------------------------------------------
    # DIACRITICS (strict + multi)
    # -----------------------------------------------------------------
    diac_strict = suggest_diacritics_strict(df_fix, valid_list, covered)
    df_diac_strict = pd.DataFrame([s.to_row() for s in diac_strict])
    df_diac_strict.to_csv(out_diac_strict, index=False, encoding="utf-8")
    print(f"[OK] DIACRITICS_STRICT    : {len(df_diac_strict)} -> {out_diac_strict}")
    if not df_diac_strict.empty:
        covered |= {_norm(x) for x in df_diac_strict["token_source"].astype(str).tolist() if _norm(x)}

    diac_multi = suggest_diacritics_multi(df_fix, valid_list, covered)
    df_diac_multi = pd.DataFrame([s.to_row() for s in diac_multi])
    df_diac_multi.to_csv(out_diac_multi, index=False, encoding="utf-8")
    print(f"[OK] DIACRITICS_MULTI     : {len(df_diac_multi)} -> {out_diac_multi}")
    if not df_diac_multi.empty:
        covered |= {_norm(x) for x in df_diac_multi["token_source"].astype(str).tolist() if _norm(x)}

    # compat : union strict+multi
    if not df_diac_strict.empty or not df_diac_multi.empty:
        df_diac = pd.concat([df_diac_strict, df_diac_multi], ignore_index=True)
    else:
        df_diac = pd.DataFrame(columns=["token_source", "match", "count", "score", "edit_dist", "category"])
    df_diac.to_csv(out_diac, index=False, encoding="utf-8")
    print(f"[OK] DIACRITICS (union)   : {len(df_diac)} -> {out_diac}")

    # -----------------------------------------------------------------
    # ABBREVIATIONS (AVANT TYPO)
    # -----------------------------------------------------------------
    write_abbrev_sure_csv(out_abbr_sure, abbrev_sure)
    print(f"[OK] Abbrev SURE (json)    : {len(abbrev_sure)} -> {out_abbr_sure}")

    write_abbrev_amb_csv(out_abbr_amb, abbrev_ambigue)
    print(f"[OK] Abbrev AMBIGU (json)  : {len(abbrev_ambigue)} -> {out_abbr_amb}")

    # On marque sure/ambigue comme "couverts" pour éviter TYPO/enrich/rejected
    covered |= {_norm(k) for k in abbrev_sure.keys() if _norm(k)}
    covered |= {_norm(k) for k in abbrev_ambigue.keys() if _norm(k)}

    _ab_sure, _ab_amb, ab_cand = suggest_abbrev(
        df_fix=df_fix,
        df_valid=df_valid,
        covered=covered,
        abbrev_sure=abbrev_sure,
        abbrev_ambigue=abbrev_ambigue,
    )

    df_ab_cand = (
        pd.DataFrame([s.to_row() for s in ab_cand])
        if ab_cand
        else pd.DataFrame(columns=["token_source", "match", "count", "score", "edit_dist", "category"])
    )
    if "expansions" not in df_ab_cand.columns:
        df_ab_cand["expansions"] = ""
    df_ab_cand.to_csv(out_abbr_cand, index=False, encoding="utf-8")
    print(f"[OK] Abbrev candidates     : {len(df_ab_cand)} -> {out_abbr_cand}")

    if not df_ab_cand.empty:
        covered |= {_norm(x) for x in df_ab_cand["token_source"].astype(str).tolist() if _norm(x)}

    # -----------------------------------------------------------------
    # INTENSITY / GRADATION clinique (mot++ / mot+++)
    # -----------------------------------------------------------------
    intensity = suggest_intensity_markers(df_fix, covered)
    df_intensity = pd.DataFrame([s.to_row() for s in intensity])
    df_intensity.to_csv(out_intensity, index=False, encoding="utf-8")
    print(f"[OK] INTENSITY            : {len(df_intensity)} -> {out_intensity}")
    if not df_intensity.empty:
        covered |= {_norm(x) for x in df_intensity["token_source"].astype(str).tolist() if _norm(x)}


    # -----------------------------------------------------------------
    # TYPO (APRES ABBREV)
    # -----------------------------------------------------------------
    typos = suggest_typos(
        df_fix,
        valid_list,
        covered,
        max_dist=args.typo_max_dist,
        min_zipf=args.typo_min_zipf,
    )
    df_typos = pd.DataFrame([s.to_row() for s in typos])
    df_typos.to_csv(out_typos, index=False, encoding="utf-8")
    print(f"[OK] TYPOS               : {len(df_typos)} -> {out_typos}")
    if not df_typos.empty:
        covered |= {_norm(x) for x in df_typos["token_source"].astype(str).tolist() if _norm(x)}

    # -----------------------------------------------------------------
    # ENRICH_METIER
    # -----------------------------------------------------------------
    enrich = suggest_enrich_metier(df_fix, covered)
    df_enrich = pd.DataFrame([s.to_row() for s in enrich])
    df_enrich.to_csv(out_enrich, index=False, encoding="utf-8")
    print(f"[OK] Enrich métier         : {len(df_enrich)} -> {out_enrich}")
    if not df_enrich.empty:
        covered |= {_norm(x) for x in df_enrich["token_source"].astype(str).tolist() if _norm(x)}

    # -----------------------------------------------------------------
    # Rejected / noise
    # -----------------------------------------------------------------
    rej_rows = []
    for _, r in df_fix.iterrows():
        src = str(r.get("token_source", ""))
        tok = _norm(src)
        if not tok or tok in covered:
            continue
        rej_rows.append(
            Suggestion(
                token_source=src,
                match="",
                count=int(r.get("count", 0)),
                score=0.0,
                edit_dist=0,
                category="REJECTED",
            ).to_row()
        )

    df_rej = pd.DataFrame(rej_rows)
    df_rej.to_csv(out_rej, index=False, encoding="utf-8")
    print(f"[OK] Rejected/noise        : {len(df_rej)} -> {out_rej}")

    print("[FIN] Génération des suggestions terminée.")


if __name__ == "__main__":
    main()
