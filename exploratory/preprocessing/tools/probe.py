from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DATA_RAW = BASE_DIR / "Data" / "DATA_RAW"

FILES = {
    "INCLUSION": DATA_RAW / "INCLUSION RECHERCHE CLINIQUE.xlsx",
    "GYNECO":    DATA_RAW / "dossier-gyneco-23-03-2022_converti.xlsx",
    "RECUEIL":   DATA_RAW / "Recueil_MMJ.xlsx",
    "PMSI":      DATA_RAW / "2022 - Donnees PMSI - Protocole ENDOPATHS - GHN..ALTRAN_converti.xlsx",
}


def detect_excel_kind(path: Path) -> str:
    """
    Retourne:
      - 'xlsx' si fichier ZIP (signature PK)
      - 'xls'  si OLE2 (signature D0 CF 11 E0 ...)
      - 'unknown' sinon
    """
    with path.open("rb") as f:
        sig = f.read(8)
    if sig[:2] == b"PK":
        return "xlsx"
    if sig[:4] == b"\xD0\xCF\x11\xE0":
        return "xls"
    return "unknown"


def list_sheets(path: Path, kind: str) -> list[str]:
    if kind == "xlsx":
        xls = pd.ExcelFile(path, engine="openpyxl")
        return list(xls.sheet_names)
    if kind == "xls":
        # xlrd requis
        xls = pd.ExcelFile(path, engine="xlrd")
        return list(xls.sheet_names)
    raise ValueError(f"Format non supporté: {kind} ({path.name})")


def read_sheet(path: Path, kind: str, sheet_name=None) -> pd.DataFrame:
    """
    IMPORTANT: sheet_name=None => pandas renvoie un dict {sheet: df}
    donc on force un feuillet unique (0 par défaut) pour toujours renvoyer un DataFrame.
    """
    engine = "openpyxl" if kind == "xlsx" else "xlrd"
    if sheet_name is None:
        sheet_name = 0
    return pd.read_excel(path, sheet_name=sheet_name, dtype=str, engine=engine)


def main() -> int:
    print("=== PROBE ENDOPATH EXCELS ===")
    print(f"BASE_DIR : {BASE_DIR}")
    print(f"DATA_RAW : {DATA_RAW}\n")

    for key, path in FILES.items():
        print("=" * 90)
        print(f"[{key}] {path}")

        if not path.exists():
            print("  -> ERREUR: fichier introuvable.")
            continue

        kind = detect_excel_kind(path)
        print(f"  - Type détecté : {kind}")

        if kind == "xls":
            print("  - Note: ce fichier est un XLS (OLE2). Il faut 'xlrd' pour le lire.")
        elif kind == "xlsx":
            print("  - Note: ce fichier est un vrai XLSX (ZIP). 'openpyxl' suffit.")

        # Feuillets
        try:
            sheets = list_sheets(path, kind)
            print(f"  - Feuillets ({len(sheets)}): {sheets}")
        except ImportError as e:
            print("  -> IMPORT ERROR (dépendance manquante)")
            print(f"     {e}")
            if kind == "xls":
                print("     Action: pip install xlrd")
            elif kind == "xlsx":
                print("     Action: pip install openpyxl")
            continue
        except Exception as e:
            print("  -> ERREUR listing feuillets:", repr(e))
            continue

        # Lecture preview
        try:
            df = read_sheet(path, kind, sheet_name=0)
            print(f"  - Chargé: {df.shape[0]} lignes, {df.shape[1]} colonnes")
            print("  - Colonnes:")
            for c in df.columns.tolist():
                print(f"     - {c}")
            print("\n  - Aperçu (5 premières lignes):")
            print(df.head(5).to_string(index=False))
        except Exception as e:
            print("  -> ERREUR lecture feuillet:", repr(e))

        print()

    print("=== FIN PROBE ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
