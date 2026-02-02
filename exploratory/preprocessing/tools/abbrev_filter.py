from wordfreq import zipf_frequency
import spacy
import re

# Charger SpaCy FR
_nlp = spacy.load("fr_core_news_md")

# Préfixes / suffixes médicaux robustes
MED_PREFIXES = (
    "gyn", "abd", "hyst", "echo", "eco", "diag", "chir",
    "anesth", "ov", "ut", "pelv", "endo", "recto", "colo"
)

MED_SUFFIXES = (
    "scop", "tom", "metr", "graph", "plast", "ectom",
    "path", "alg", "ite", "ose"
)

# Abréviations explicitement autorisées
ABBREV_WHITELIST = {
    "abdo", "bilat", "gyn", "diag", "hospit",
    "anesth", "ov", "ut", "tens", "coag", "stim",
    "ecbu", "ttt", "bhcg", "fcv", "rsg"
}

def is_noise_token(tok: str) -> bool:
    """Rejette bruit linguistique évident"""
    if tok.isdigit():
        return True
    if len(tok) <= 2:
        return True
    if zipf_frequency(tok, "fr") >= 4.5:  # mot courant
        return True
    doc = _nlp(tok)
    return doc[0].is_stop


def looks_medical_abbrev(tok: str) -> bool:
    """Heuristique stricte abréviation médicale"""
    t = tok.lower()

    if t in ABBREV_WHITELIST:
        return True

    if t.startswith(MED_PREFIXES):
        return True

    if t.endswith(MED_SUFFIXES):
        return True

    # pattern type "rsg", "fcv", "bhcg"
    if re.fullmatch(r"[a-z]{3,6}", t):
        return False  # trop ambigu sans signal médical

    return False


def is_valid_abbreviation(tok: str) -> bool:
    if is_noise_token(tok):
        return False

    if not (3 <= len(tok) <= 8):
        return False

    return looks_medical_abbrev(tok)
