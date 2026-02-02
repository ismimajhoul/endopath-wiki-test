#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
probe_excel_files.py
Objectif : diagnostiquer "ce que l'on peut lire" dans vos fichiers Excel, avec traces.
- Détecte si le fichier est un vrai XLSX (zip) ou un XLS (OLE2).
- Liste les feuilles disponibles.
- Pour chaque feuille, affiche un aperçu des colonnes et quelques lignes.
- Met en évidence les colonnes attendues (ex: "*IPP*", "NUM_INCLUSION", "Gynécologie > Consultation>Consultation", etc.)

Exécution :
    python probe_excel_files.py

Pré-requis :
    pip install pandas openpyxl
    Pour lire les .xls (binaire OLE2) : pip install xlrd==2.0.1
"""
from __future__ import annotations

from pathlib import Path
import sys
import traceback

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_RAW = BASE_DIR / "Data" / "DATA_RAW"

FILES = [
    DATA_RAW / "INCLUSION RECHERCHE CLINIQUE.xlsx",
    DATA_RAW / "dossier-gyneco-23-03-2022_converti.xlsx",
    DATA_RAW / "Recueil_MMJ.xlsx",
    DATA_RAW / "2022 - Donnees PMSI - Protocole ENDOPATHS - GHN..ALTRAN_converti.xlsx",
]

HIGHLIGHT_COLS = [
    "NUM_INCLUSION",
    "*IPP*",
    "IPP",
    "Gynécologie > Consultation>Date consultation",
    "Gynécologie > Consultation>Consultation",
    "diag endo profonde",
]


def detect_excel_kind(path: Path) -> str:
    """
    Retourne 'xlsx' si fichier ZIP/OOXML, 'xls' si OLE2, sinon 'unknown'
    """
    with path.open("rb") as f:
        sig = f.read(8)
    if sig.startswith(b"PK\x03\x04"):
        return "xlsx"
    if sig.startswith(b"\xD0\xCF\x11\xE0\xA1\xB1\x1A\xE1"):
        return "xls"  # OLE2
    return "unknown"


def pick_engine(kind: str) -> str | None:
    if kind == "xlsx":
        return "openpyxl"
    if kind == "xls":
        return "xlrd"
    return None


def safe_read_excel(path: Path, sheet_name, engine: str, nrows: int | None = None) -> pd.DataFrame:
    return pd.read_excel(path, sheet_name=sheet_name, dtype=str, engine=engine, nrows=nrows)


def main() -> int:
    out_lines: list[str] = []
    out_lines.append("[INFO] BASE_DIR = " + str(BASE_DIR))
    out_lines.append("[INFO] DATA_RAW = " + str(DATA_RAW))
    out_lines.append("")

    for fp in FILES:
        out_lines.append("=" * 80)
        out_lines.append(f"[FILE] {fp}")
        if not fp.exists():
            out_lines.append("  -> NOT FOUND")
            continue

        kind = detect_excel_kind(fp)
        engine = pick_engine(kind)
        out_lines.append(f"  - Type détecté : {kind}")
        out_lines.append(f"  - Engine pandas : {engine}")

        if engine == "xlrd":
            out_lines.append("  - Note: ce fichier est un XLS (OLE2). Il faut 'xlrd' pour le lire.")
            out_lines.append("    Si ImportError : pip install xlrd==2.0.1")
        if engine is None:
            out_lines.append("  -> Format non reconnu comme Excel standard (XLSX/XLS).")
            continue

        # Liste des feuilles
        try:
            xls = pd.ExcelFile(fp, engine=engine)
            out_lines.append(f"  - Feuilles ({len(xls.sheet_names)}): {xls.sheet_names}")
        except Exception as e:
            out_lines.append("  -> ERREUR lecture ExcelFile(): " + repr(e))
            out_lines.append("  -> TRACEBACK:\n" + traceback.format_exc())
            continue

        # Pour chaque feuille : colonnes + aperçu
        for s in xls.sheet_names:
            out_lines.append("")
            out_lines.append(f"  [SHEET] {s}")

            try:
                df_head = safe_read_excel(fp, sheet_name=s, engine=engine, nrows=5)
            except Exception as e:
                out_lines.append("    -> ERREUR read_excel(): " + repr(e))
                out_lines.append("    -> TRACEBACK:\n" + traceback.format_exc())
                continue

            cols = list(df_head.columns)
            out_lines.append(f"    - Colonnes ({len(cols)}):")
            # Marquer les colonnes importantes
            for c in cols[:200]:
                mark = " *" if c in HIGHLIGHT_COLS else ""
                out_lines.append(f"      - {c}{mark}")
            if len(cols) > 200:
                out_lines.append(f"      ... ({len(cols) - 200} colonnes supplémentaires non affichées)")

            # Aperçu lignes (5)
            out_lines.append("    - Aperçu 5 lignes (colonnes max 12):")
            preview_cols = cols[:12]
            out_lines.append("      " + " | ".join(preview_cols))
            for i in range(min(5, len(df_head))):
                row = [("" if pd.isna(df_head.at[i, c]) else str(df_head.at[i, c])) for c in preview_cols]
                out_lines.append("      " + " | ".join(row))

    # Ecriture fichier output
    out_path = BASE_DIR / "output_probe_v2.txt"
    out_path.write_text("\n".join(out_lines), encoding="utf-8")
    print(f"[OK] Rapport écrit : {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
