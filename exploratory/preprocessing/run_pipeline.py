#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run_pipeline.py
Orchestrateur du pipeline Endopath (preprocessing).

Objectif :
- Exécuter les scripts dans le bon ordre
- Vérifier la présence des fichiers attendus
- Stopper immédiatement si une étape échoue
- Produire un log clair

Hypothèses (à ajuster si besoin) :
- On exécute ce script depuis le dossier preprocessing (là où se trouvent app.py, create_endopath_diag_db.py, etc.)
- Chaque script est exécutable via "python <script>.py" sans arguments, et utilise des chemins relatifs (Data/DATA_RAW, Data/DATA_PROCESSED, etc.)

Si tes scripts attendent des arguments CLI (--db, --input, ...), dis-moi et on adapte.
"""

from __future__ import annotations

import os
import sys
import time
import subprocess
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

DATA_RAW = BASE_DIR / "Data" / "DATA_RAW"
DATA_PROCESSED = BASE_DIR / "Data" / "DATA_PROCESSED"
CORRECTION_DIR = DATA_PROCESSED / "Correction_mots"

DB_PATH = BASE_DIR / "endopath_diag.db"

# Principaux scripts (ordre recommandé)
SCRIPTS_ORDER = [
    "create_endopath_diag_db.py",
    "extract_text_and_vocab_from_dossier_gyneco.py",
    "build_tokens_pipeline.py",
    "filter_tokens_with_spacy.py",
    "suggest_dict_extensions.py",
    # Optionnels (si tu veux les inclure automatiquement)
    # "build_abbrev_from_wikipedia.py",
    # "merge_abbrev_dicts.py",
]


@dataclass
class Step:
    name: str
    script: Path
    args: List[str]
    must_exist_before: List[Path]
    expected_after: List[Path]
    optional: bool = False


def _run_cmd(cmd: List[str], cwd: Path) -> None:
    print(f"\n[RUN] {' '.join(cmd)}")
    p = subprocess.run(cmd, cwd=str(cwd))
    if p.returncode != 0:
        raise RuntimeError(f"Commande en échec (code={p.returncode}): {' '.join(cmd)}")


def _check_exists(label: str, paths: List[Path], optional: bool = False) -> None:
    missing = [p for p in paths if not p.exists()]
    if missing:
        msg = "\n".join([f" - {p}" for p in missing])
        if optional:
            print(f"[WARN] {label}: fichiers manquants (optionnel) :\n{msg}")
            return
        raise FileNotFoundError(f"{label}: fichiers manquants :\n{msg}")


def _banner(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def build_steps() -> List[Step]:
    """
    Déclare ici les entrées/sorties attendues.
    Tu peux ajuster les 'expected_after' selon ton pipeline réel.
    """
    steps: List[Step] = []

    # 1) DB (XLSX -> SQLite)
    steps.append(
        Step(
            name="Build SQLite DB (XLSX -> endopath_diag.db)",
            script=BASE_DIR / "create_endopath_diag_db.py",
            args=[],
            must_exist_before=[
                DATA_RAW / "INCLUSION RECHERCHE CLINIQUE.xlsx",
                DATA_RAW / "Recueil_MMJ.xlsx",
                DATA_RAW / "2022 - Donnees PMSI - Protocole ENDOPATHS - GHN..ALTRAN_converti.xlsx",
                DATA_RAW / "dossier-gyneco-23-03-2022_converti.xlsx",
            ],
            expected_after=[DB_PATH],
        )
    )

    # 2) Extract texte + vocab depuis dossier gyneco XLSX
    steps.append(
        Step(
            name="Extract texte patient + vocab (XLSX dossier-gyneco -> CSV)",
            script=BASE_DIR / "extract_text_and_vocab_from_dossier_gyneco.py",
            args=[],
            must_exist_before=[
                DATA_RAW / "dossier-gyneco-23-03-2022_converti.xlsx",
           ],
            expected_after=[
                DATA_PROCESSED / "dossier_gyneco_texte_par_patiente.csv",
                DATA_PROCESSED / "vocab_dossier_gyneco_from_xlsx.csv",
            ],
        )
    )

    # 3) Build tokens pipeline (DB/CSV -> tokens bruts / agrégats)
    steps.append(
        Step(
            name="Build tokens pipeline",
            script=BASE_DIR / "build_tokens_pipeline.py",
            args=[],
            must_exist_before=[
                DB_PATH,
                DATA_PROCESSED / "dossier_gyneco_texte_par_patiente.csv",
            ],
            expected_after=[
                # selon ton implémentation exacte, ajuste si besoin
                DATA_PROCESSED / "all_words.csv",
            ],
            optional=True,  # car selon ton pipeline, ça peut ne pas produire ce fichier
        )
    )

    # 4) Filtrage tokens SpaCy
    steps.append(
        Step(
            name="Filter tokens with SpaCy (valid/invalid/to_fix)",
            script=BASE_DIR / "filter_tokens_with_spacy.py",
            args=[],
            must_exist_before=[
                DATA_PROCESSED / "vocab_dossier_gyneco_from_xlsx.csv",
            ],
            expected_after=[
                DATA_PROCESSED / "tokens_valides.csv",
                DATA_PROCESSED / "tokens_invalides.csv",
                DATA_PROCESSED / "tokens_a_corriger.csv",
            ],
        )
    )

    # 5) Suggestions + dictionnaires + CSV (Correction_mots)
    steps.append(
        Step(
            name="Suggest dictionary extensions + generate suggestions CSV",
            script=BASE_DIR / "suggest_dict_extensions.py",
            args=[],
            must_exist_before=[
                DATA_PROCESSED / "tokens_a_corriger.csv",
                CORRECTION_DIR / "dictionnaire_correction.json",
                CORRECTION_DIR / "abbrev_sure_merged.json",
                CORRECTION_DIR / "abbrev_ambigue_merged.json",
            ],
            expected_after=[
                CORRECTION_DIR / "suggestions_manual_dict.csv",
                CORRECTION_DIR / "suggestions_auto_diacritics_strict.csv",
                CORRECTION_DIR / "suggestions_auto_diacritics_multi.csv",
                CORRECTION_DIR / "suggestions_auto_diacritics.csv",
                CORRECTION_DIR / "suggestions_auto_typos.csv",
                CORRECTION_DIR / "suggestions_auto_abbrev.csv",
                CORRECTION_DIR / "suggestions_auto_abbrev_ambigu.csv",
                CORRECTION_DIR / "suggestions_auto_abbrev_candidate.csv",
                CORRECTION_DIR / "suggestions_domain_enrich.csv",
                CORRECTION_DIR / "suggestions_auto_rejected.csv",
            ],
        )
    )

    return steps


def main(argv: List[str]) -> int:
    _banner("ENDOPATH — RUN PIPELINE")

    print(f"[INFO] BASE_DIR       = {BASE_DIR}")
    print(f"[INFO] DATA_RAW       = {DATA_RAW}")
    print(f"[INFO] DATA_PROCESSED = {DATA_PROCESSED}")
    print(f"[INFO] DB_PATH        = {DB_PATH}")

    steps = build_steps()

    # Vérif scripts
    for s in steps:
        if not s.script.exists():
            raise FileNotFoundError(f"Script introuvable: {s.script}")

    started = time.time()

    for i, step in enumerate(steps, start=1):
        _banner(f"STEP {i}/{len(steps)} — {step.name}")

        _check_exists("Prerequisites", step.must_exist_before, optional=step.optional)

        cmd = [sys.executable, str(step.script)] + step.args
        try:
            _run_cmd(cmd, cwd=BASE_DIR)
        except Exception as e:
            print(f"\n[ERROR] Step failed: {step.name}")
            print(f"[ERROR] {e}")
            return 1

        _check_exists("Expected outputs", step.expected_after, optional=step.optional)
        print("[OK] Step completed.")

    elapsed = time.time() - started
    _banner(f"PIPELINE DONE — {elapsed:.1f}s")

    print("\nNext:")
    print("  1) Lancer l'UI Flask : python app.py")
    print("  2) Ouvrir : http://127.0.0.1:5000")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted.")
        raise SystemExit(130)
