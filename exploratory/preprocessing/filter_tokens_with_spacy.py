from pathlib import Path
import re
import unicodedata
from typing import Dict, List, Tuple

import pandas as pd
import spacy
from spacy.lang.fr.stop_words import STOP_WORDS as FR_STOPWORDS
from wordfreq import zipf_frequency

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "Data" / "DATA_PROCESSED" / "vocab_dossier_gyneco_from_xlsx.csv"

OUTPUT_VALID = BASE_DIR / "Data" / "DATA_PROCESSED" / "tokens_valides.csv"
OUTPUT_INVALID = BASE_DIR / "Data" / "DATA_PROCESSED" / "tokens_invalides.csv"
OUTPUT_TO_FIX = BASE_DIR / "Data" / "DATA_PROCESSED" / "tokens_a_corriger.csv"

# Seuils wordfreq (ZIPF)
ZIPF_MIN = 2.5  # < 2.5 = mot très rare / suspect
ZIPF_VERY_COMMON = 4.0  # >= 4.0 = mot très fréquent (souvent mot-outil / commun)

# Token "word-like"
_WORD_RE = re.compile(r"^[A-Za-zÀ-ÖØ-öø-ÿ][A-Za-zÀ-ÖØ-öø-ÿ'\-]*$")

# ---------------------------------------------------------------------
# Chargement modèle SpaCy FR
# ---------------------------------------------------------------------
print("[INFO] Chargement du modèle SpaCy fr_core_news_md ...")
nlp = spacy.load("fr_core_news_md")
print("[INFO] Modèle SpaCy chargé.")

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def strip_accents(s: str) -> str:
    """Supprime les diacritiques (é -> e, ç -> c, etc.)"""
    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    return unicodedata.normalize("NFC", s)

def has_diacritics(s: str) -> bool:
    return s != strip_accents(s)

def is_word_like(token: str) -> bool:
    return bool(_WORD_RE.match(token))

def is_common_function_word(tok: str) -> bool:
    """
    Mot-outil / token trop court : on ne doit jamais le pousser en correction.
    Exemples: de, du, la, et, ou, ... (stopwords FR SpaCy)
    """
    t = (tok or "").strip().lower()
    if not t:
        return True
    if len(t) <= 2:
        return True
    if t in FR_STOPWORDS:
        return True
    return False

def compute_zipf(tok: str) -> float:
    t = (tok or "").strip().lower()
    if not t:
        return 0.0
    try:
        return float(zipf_frequency(t, "fr"))
    except Exception:
        return 0.0

def is_valid_token_conservative(token: str, zipf: float, spacy_is_oov: bool) -> bool:
    """
    Heuristique lexicale conservative :
    - token word-like
    - et (reconnu par SpaCy) OU (Zipf >= seuil)
    """
    t = (token or "").strip()
    if not t:
        return False
    if not is_word_like(t):
        return False

    return (not spacy_is_oov) or (zipf >= ZIPF_MIN)

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Fichier introuvable: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)

    # Normalisation colonnes
    if "token" in df.columns and "token_source" not in df.columns:
        df = df.rename(columns={"token": "token_source"})
    if "token_source" not in df.columns:
        raise ValueError("Colonne token_source manquante dans le vocabulaire d'entrée.")
    if "count" not in df.columns:
        df["count"] = 1

    # Nettoyage léger
    df["token_source"] = df["token_source"].astype(str).map(lambda x: x.strip())
    df = df[df["token_source"] != ""].copy()

    # ---------------------------------------------------------------------
    # Index: forme "désaccentuée" => existe-t-il une variante AVEC diacritiques ?
    # Exemple: ulcère & ulcere -> key "ulcere" => diacritics_present=True
    # ---------------------------------------------------------------------
    diacritics_present_by_key: Dict[str, bool] = {}
    for tok in df["token_source"].astype(str).tolist():
        t = tok.strip()
        if not t:
            continue
        key = strip_accents(t.lower())
        if not key:
            continue
        if has_diacritics(t):
            diacritics_present_by_key[key] = True
        else:
            diacritics_present_by_key.setdefault(key, False)

    # ---------------------------------------------------------------------
    # Pré-calcul SpaCy OOV en batch (uniquement pour tokens "candidats")
    # - IMPORTANT: on évite SpaCy sur stopwords / courts
    # ---------------------------------------------------------------------
    tokens: List[str] = df["token_source"].astype(str).tolist()

    # Pré-calcul zipf + word_like
    zipf_map: Dict[str, float] = {}
    word_like_map: Dict[str, int] = {}
    common_map: Dict[str, bool] = {}

    for tok in tokens:
        tl = tok.lower().strip()
        common = is_common_function_word(tl)
        common_map[tok] = common
        z = compute_zipf(tl)
        zipf_map[tok] = z
        word_like_map[tok] = 1 if is_word_like(tok) else 0

    # Tokens sur lesquels on veut SpaCy (candidats)
    # - word_like
    # - pas common
    # - zipf pas "très commun" (sinon ça sert rarement)
    spacy_candidates: List[str] = [
        tok for tok in tokens
        if word_like_map.get(tok, 0) == 1
        and not common_map.get(tok, False)
        and zipf_map.get(tok, 0.0) < ZIPF_VERY_COMMON
    ]

    spacy_oov_map: Dict[str, bool] = {}
    if spacy_candidates:
        for doc in nlp.pipe(spacy_candidates, batch_size=256):
            text = doc.text
            if len(doc) == 0:
                spacy_oov_map[text] = True
            else:
                spacy_oov_map[text] = bool(getattr(doc[0], "is_oov", True))

    # Par défaut: si pas passé par spacy => OOV=True (conservatif)
    def get_spacy_oov(tok: str) -> bool:
        return spacy_oov_map.get(tok, True)

    # ---------------------------------------------------------------------
    # Classification
    # ---------------------------------------------------------------------
    valid_rows: List[dict] = []
    invalid_rows: List[dict] = []

    for _, row in df.iterrows():
        tok = str(row["token_source"]).strip()
        cnt = int(row.get("count") or 0)
        tok_lower = tok.lower().strip()

        z = zipf_map.get(tok, 0.0)
        wl = word_like_map.get(tok, 0)
        common = common_map.get(tok, False)

        # Règle anti-régression accents :
        # si une variante accentuée existe dans le vocab ET que ce token n'a pas de diacritiques
        # => on force en "à corriger"
        #
        # MAIS : on n'applique jamais ça aux mots-outils / très courts (sinon "de/du/la" partent en accent…)
        key = strip_accents(tok_lower)
        force_to_fix = False
        if (not common) and key:
            force_to_fix = bool(diacritics_present_by_key.get(key, False) and (not has_diacritics(tok)))

        # Si mot-outil / court : TOUJOURS valide
        if common:
            valid_rows.append(
                {"token_source": tok, "count": cnt, "zipf": z, "word_like": wl}
            )
            continue

        # Si pas word-like : invalide / à corriger (selon ta logique actuelle)
        if wl == 0:
            invalid_rows.append(
                {"token_source": tok, "count": cnt, "zipf": z, "word_like": wl}
            )
            continue

        spacy_oov = get_spacy_oov(tok)

        if (not force_to_fix) and is_valid_token_conservative(tok, z, spacy_oov):
            valid_rows.append(
                {"token_source": tok, "count": cnt, "zipf": z, "word_like": wl}
            )
        else:
            invalid_rows.append(
                {"token_source": tok, "count": cnt, "zipf": z, "word_like": wl}
            )

    df_valid = pd.DataFrame(valid_rows)
    df_invalid = pd.DataFrame(invalid_rows)

    if not df_valid.empty:
        df_valid.sort_values(by="count", ascending=False, inplace=True)
    if not df_invalid.empty:
        df_invalid.sort_values(by="count", ascending=False, inplace=True)

    OUTPUT_VALID.parent.mkdir(parents=True, exist_ok=True)

    df_valid.to_csv(OUTPUT_VALID, index=False, encoding="utf-8")
    df_invalid.to_csv(OUTPUT_INVALID, index=False, encoding="utf-8")

    # Ton pipeline consomme tokens_a_corriger.csv => on le garde aligné sur invalides
    df_invalid.to_csv(OUTPUT_TO_FIX, index=False, encoding="utf-8")

    print(f"[OK] Tokens valides    : {len(df_valid)} -> {OUTPUT_VALID}")
    print(f"[OK] Tokens invalides  : {len(df_invalid)} -> {OUTPUT_INVALID}")
    print(f"[OK] Tokens à corriger  : {len(df_invalid)} -> {OUTPUT_TO_FIX}")
    print("[INFO] (Stopwords/courts sont forcés en 'valides' pour éviter les faux ACCENT.)")

if __name__ == "__main__":
    main()
