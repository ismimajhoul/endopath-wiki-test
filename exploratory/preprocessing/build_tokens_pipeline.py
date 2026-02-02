#!/usr/bin/env python3
"""
Pipeline AMONT de génération des tokens :

1) Extraction texte + vocabulaire depuis l'Excel
2) Filtrage linguistique SpaCy → tokens_valides / invalides / a_corriger

À exécuter AVANT suggest_dict_extensions.py
"""

from pathlib import Path
import subprocess
import sys


BASE_DIR = Path(__file__).resolve().parent

# Scripts existants
SCRIPT_EXTRACT = BASE_DIR / "extract_text_and_vocab_from_dossier_gyneco.py"
SCRIPT_FILTER  = BASE_DIR / "filter_tokens_with_spacy.py"

# Sorties attendues
OUT_PROCESSED = BASE_DIR / "Data" / "DATA_PROCESSED"

EXPECTED_FILES = [
    OUT_PROCESSED / "vocab_dossier_gyneco_from_xlsx.csv",
    OUT_PROCESSED / "dossier_gyneco_texte_par_patiente.csv",
    OUT_PROCESSED / "tokens_valides.csv",
    OUT_PROCESSED / "tokens_invalides.csv",
    OUT_PROCESSED / "tokens_a_corriger.csv",
]


def run_step(label: str, script: Path) -> None:
    print(f"\n[STEP] {label}")
    if not script.exists():
        print(f"[ERROR] Script introuvable : {script}")
        sys.exit(1)

    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=BASE_DIR,
    )

    if result.returncode != 0:
        print(f"[ERROR] Échec de l'étape : {label}")
        sys.exit(1)


def check_outputs() -> None:
    print("\n[CHECK] Vérification des fichiers générés")
    missing = [p for p in EXPECTED_FILES if not p.exists()]

    if missing:
        print("[ERROR] Fichiers manquants :")
        for p in missing:
            print(f"  - {p}")
        sys.exit(1)

    for p in EXPECTED_FILES:
        print(f"[OK] {p.relative_to(BASE_DIR)}")


def main() -> None:
    print("[PIPELINE] Génération des tokens – démarrage")
    print(f"[INFO] Base dir : {BASE_DIR}")

    run_step("Extraction texte + vocabulaire", SCRIPT_EXTRACT)
    run_step("Filtrage linguistique SpaCy", SCRIPT_FILTER)

    check_outputs()

    print("\n[PIPELINE] Tokens prêts.")
    print("→ Vous pouvez maintenant lancer : suggest_dict_extensions.py")


if __name__ == "__main__":
    main()
