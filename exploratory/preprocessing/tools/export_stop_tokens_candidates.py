# export_stop_tokens_candidates.py
#
# Exporte la liste des tokens présents dans suggestions_auto.csv
# pour aider à construire la stop-list.

from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
SUGG_PATH = BASE_DIR / "Data" / "DATA_PROCESSED" / "Correction_mots" / "suggestions_auto.csv"
OUT_PATH = BASE_DIR / "Data" / "DATA_PROCESSED" / "Correction_mots" / "stop_tokens_candidates.csv"

def main():
    if not SUGG_PATH.exists():
        raise FileNotFoundError(f"Fichier suggestions_auto introuvable : {SUGG_PATH}")

    df = pd.read_csv(SUGG_PATH)

    # On ne garde que quelques colonnes utiles pour l'analyse
    cols = ["token_source", "count", "match_1", "score_1"]
    cols_presentes = [c for c in cols if c in df.columns]
    df_out = df[cols_presentes].copy()

    # On se débarrasse des doublons, au cas où
    df_out = df_out.drop_duplicates(subset=["token_source"])

    # Tri par fréquence décroissante
    df_out = df_out.sort_values(by="count", ascending=False)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(OUT_PATH, index=False, encoding="utf-8")

    print(f"[OK] Fichier de travail écrit dans : {OUT_PATH}")
    print(f"     ({len(df_out)} tokens uniques)")

if __name__ == "__main__":
    main()
