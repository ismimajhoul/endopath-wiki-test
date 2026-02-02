# extract_text_and_vocab_from_dossier_gyneco.py

"""
Objectif :
---------
1) Lire le fichier Excel protégé :
       Data/DATA_RAW/dossier-gyneco-23-03-2022.xlsx
   en utilisant msoffcrypto (comme le pipeline officiel).

2) Générer :
   - Data/DATA_PROCESSED/dossier_gyneco_texte_par_patiente.csv
       -> 1 ligne par patiente, concaténation de toutes les colonnes textuelles

   - Data/DATA_PROCESSED/vocab_dossier_gyneco_from_xlsx.csv
       -> vocabulaire + fréquence d’apparition des tokens
"""

import io
import json
import re
from collections import Counter
from pathlib import Path

import msoffcrypto
import pandas as pd


# ---------------------------------------------------------------------------
# Paramètres de chemins
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

# Dossier gynéco : on privilégie le fichier converti "vrai .xlsx" si présent
CANDIDATES_GYNECO = [
    BASE_DIR / "Data" / "DATA_RAW" / "dossier-gyneco-23-03-2022_converti.xlsx",
    BASE_DIR / "Data" / "DATA_RAW" / "dossier-gyneco-23-03-2022.xlsx",
]
PATH_XLSX = next((p for p in CANDIDATES_GYNECO if p.exists()), CANDIDATES_GYNECO[0])
PATH_PASSWORD = BASE_DIR / "Data" / "PASSWORD_1.txt"  # optionnel (uniquement si le fichier est chiffré)

OUT_DIR = BASE_DIR / "Data" / "DATA_PROCESSED"
OUT_TEXT_PAR_PATIENTE = OUT_DIR / "dossier_gyneco_texte_par_patiente.csv"
OUT_VOCAB = OUT_DIR / "vocab_dossier_gyneco_from_xlsx.csv"


# ---------------------------------------------------------------------------
# Lecture du fichier Excel protégé via msoffcrypto
# ---------------------------------------------------------------------------

def read_protected_excel(path_excel, path_password):
    """
    Lecture d'un fichier Excel :
    - Si le fichier est chiffré (Excel protégé), on tente un déchiffrement via msoffcrypto.
    - Si le fichier n'est PAS chiffré (cas actuel après conversion LibreOffice/Excel),
      on lit directement avec pandas/openpyxl.

    Le fichier de mot de passe est optionnel : il n'est requis que si le document est chiffré.
    """
    # Tentative lecture "standard" (cas le plus fréquent)
    try:
        print(f"[INFO] Lecture Excel standard : {path_excel}")
        return pd.read_excel(path_excel, engine="openpyxl", dtype=str)
    except Exception as e_std:
        print(f"[WARN] Lecture standard échouée ({type(e_std).__name__}: {e_std}). Tentative déchiffrement...")

    # Tentative déchiffrement (uniquement si un mot de passe est fourni)
    if not Path(path_password).exists():
        raise RuntimeError(
            f"Le fichier Excel semble chiffré ou non lisible en lecture standard, "
            f"mais le fichier password est introuvable: {path_password}"
        )

    print(f"[INFO] Lecture password : {path_password}")
    password = Path(path_password).read_text(encoding="utf-8").strip()

    print(f"[INFO] Déchiffrement Excel protégé : {path_excel}")
    decrypted = io.BytesIO()
    with open(path_excel, "rb") as f:
        office_file = msoffcrypto.OfficeFile(f)
        office_file.load_key(password=password)
        try:
            office_file.decrypt(decrypted)
        except Exception as e_dec:
            # Si le fichier n'est pas chiffré, msoffcrypto lève typiquement "Unencrypted document"
            msg = str(e_dec).lower()
            if "unencrypted" in msg or "not encrypted" in msg:
                print("[INFO] Document non chiffré détecté par msoffcrypto. Relecture standard.")
                return pd.read_excel(path_excel, engine="openpyxl", dtype=str)
            raise

    decrypted.seek(0)
    print("[INFO] Lecture du contenu déchiffré dans pandas...")
    return pd.read_excel(decrypted, engine="openpyxl", dtype=str)



# ---------------------------------------------------------------------------
# Normalisation légère du texte
# ---------------------------------------------------------------------------

def normalize_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str):
    if not text:
        return []
    return text.split(" ")


# ---------------------------------------------------------------------------
# Script principal
# ---------------------------------------------------------------------------

def main():

    # -------------------------------
    # 1) Lecture du fichier Excel
    # -------------------------------
    print(f"[INFO] Tentative lecture Excel protégé : {PATH_XLSX}")
    df = read_protected_excel(PATH_XLSX, PATH_PASSWORD)
    print(f"[INFO] Fichier chargé : {df.shape[0]} lignes, {df.shape[1]} colonnes")

    # Vérifier présence colonne IPP
    col_ipp_candidates = [c for c in df.columns if "#IPP" in c or "IPP" in c.upper()]
    if not col_ipp_candidates:
        raise ValueError("Impossible de trouver une colonne IPP dans le fichier Excel.")
    COL_IPP = col_ipp_candidates[0]
    print(f"[INFO] Colonne identifiant patiente détectée : {COL_IPP}")

    # -------------------------------
    # 2) Détection des colonnes textuelles
    # -------------------------------
    colonnes_textuelles = []
    for col in df.columns:
        if col == COL_IPP:
            continue
        if df[col].dtype == object:
            # tester si la colonne contient au moins un vrai texte
            series_non_na = df[col].dropna()
            has_text = any(isinstance(x, str) and x.strip() for x in series_non_na.head(50))
            if has_text:
                colonnes_textuelles.append(col)

    print(f"[INFO] Colonnes textuelles détectées ({len(colonnes_textuelles)}) :")
    for c in colonnes_textuelles:
        print("   -", c)

    # -------------------------------
    # 3) Construire TexteConcatene par patiente
    # -------------------------------
    lignes = []
    for idx, row in df.iterrows():
        ipp = row[COL_IPP]
        fragments = []

        for col in colonnes_textuelles:
            val = row[col]
            if isinstance(val, str) and val.strip():
                fragments.append(val)

        texte_concat = " ".join(fragments)
        lignes.append({"IPP": ipp, "TexteConcatene": texte_concat})

    df_text = pd.DataFrame(lignes)

    # Si plusieurs lignes pour un même IPP -> concatène
    df_text = df_text.groupby("IPP", as_index=False)["TexteConcatene"].apply(
        lambda s: " ".join(str(x) for x in s if x)
    )
    df_text.columns = ["IPP", "TexteConcatene"]

    # Create output directory
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df_text.to_csv(OUT_TEXT_PAR_PATIENTE, index=False, encoding="utf-8")
    print(f"[OK] Texte par patiente écrit dans : {OUT_TEXT_PAR_PATIENTE}")
    print(f"     → {df_text.shape[0]} patientes")

    # -------------------------------
    # 4) Construire vocabulaire
    # -------------------------------
    counter = Counter()

    for texte in df_text["TexteConcatene"]:
        tex_norm = normalize_text(texte)
        tokens = tokenize(tex_norm)
        counter.update(tokens)

    # enlever token vide
    counter.pop("", None)

    vocab_df = pd.DataFrame(
        [{"token": tok, "count": freq} for tok, freq in counter.most_common()]
    )
    vocab_df.to_csv(OUT_VOCAB, index=False, encoding="utf-8")

    print(f"[OK] Vocabulaire écrit dans : {OUT_VOCAB}")
    print(f"     → {len(vocab_df)} tokens uniques")

    print("[FIN] Script terminé avec succès.")


if __name__ == "__main__":
    main()