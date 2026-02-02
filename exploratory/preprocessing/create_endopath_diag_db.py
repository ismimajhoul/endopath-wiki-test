# -*- coding: utf-8 -*-
"""
create_endopath_diag_db.py

Crée la base SQLite endopath_diag.db compatible avec l'appli :
- charge INCLUSION_RECHERCHE_CLINIQUE (NUM_INCLUSION <-> IPP)
- charge Recueil_MMJ (diag endo profonde)
- charge dossier gyneco (texte Consultation) => gyneco_diag_raw
- charge PMSI => pmsi_diag
- crée diag_comparaison
- crée diag_validations (vide) pour validation dans l'appli

Exécution:
  python .\create_endopath_diag_db.py
"""

from __future__ import annotations

from pathlib import Path
import sqlite3
import sys
import traceback
import pandas as pd


# ---------------------------------------------------------------------
# Paths (adaptés à ton arborescence)
# ---------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_RAW = BASE_DIR / "Data" / "DATA_RAW"
DB_PATH = BASE_DIR / "endopath_diag.db"

# IMPORTANT: utiliser les fichiers convertis en "vrai xlsx" (signature PK)
XLSX_INCLUSION = DATA_RAW / "INCLUSION RECHERCHE CLINIQUE.xlsx"
XLSX_RECUEIL   = DATA_RAW / "Recueil_MMJ.xlsx"
XLSX_GYNECO    = DATA_RAW / "dossier-gyneco-23-03-2022_converti.xlsx"
XLSX_PMSI      = DATA_RAW / "2022 - Donnees PMSI - Protocole ENDOPATHS - GHN..ALTRAN_converti.xlsx"

# Feuilles "figées" suite à tes probes / retours
SHEET_INCLUSION = "INCLUSION RECHERCHE CLINIQUE"
SHEET_RECUEIL   = "Données Cap gemini"
SHEET_GYNECO    = "ExtractionDonnees_23 mars 2022"
SHEET_PMSI_DIAGS = "TAB_HOSPIT_DIAGS"


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def info(msg: str) -> None:
    print(msg, flush=True)

def fatal(msg: str, code: int = 2) -> None:
    print(msg, file=sys.stderr, flush=True)
    raise SystemExit(code)

def file_signature(path: Path, n: int = 16) -> bytes:
    with path.open("rb") as f:
        return f.read(n)

def ensure_real_xlsx(path: Path) -> None:
    """
    Un vrai .xlsx commence par 'PK\\x03\\x04' (zip).
    Les faux xlsx OLE2 commencent par D0 CF 11 E0...
    """
    if not path.exists():
        fatal(f"[FATAL] Fichier introuvable: {path}")
    sig = file_signature(path, 4)
    if sig != b"PK\x03\x04":
        # OLE2 ou autre
        sig16 = file_signature(path, 16)
        fatal(
            "[FATAL] Fichier Excel non reconnu comme 'vrai xlsx' (ZIP).\n"
            f"  - Fichier: {path}\n"
            f"  - Signature (16 bytes): {sig16.hex(' ')}\n"
            "=> Ouvre le fichier dans LibreOffice/Excel et 'Enregistrer sous' en .xlsx.\n"
            "   Vérifie que le fichier commence par 50 4B 03 04."
        )

def read_xlsx(path: Path, sheet: str, dtype=str, header=0) -> pd.DataFrame:
    ensure_real_xlsx(path)
    info(f"[READ] {path.name} | sheet='{sheet}' | header={header}")
    df = pd.read_excel(path, sheet_name=sheet, dtype=dtype, engine="openpyxl", header=header)
    info(f"[READ] -> shape={df.shape} cols={len(df.columns)}")
    return df

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # normalisation douce: trim + supprimer espaces de fin/début
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df

def to_sql_replace(conn: sqlite3.Connection, df: pd.DataFrame, table: str) -> None:
    info(f"[SQL] write table '{table}' (replace) rows={len(df)} cols={len(df.columns)}")
    df.to_sql(table, conn, if_exists="replace", index=False)
    info(f"[OK] table '{table}' écrite.")

def sql_exec(conn: sqlite3.Connection, sql: str) -> None:
    conn.executescript(sql)

def safe_str(s) -> str:
    if s is None:
        return ""
    return str(s).strip()

# ---------------------------------------------------------------------
# Build steps
# ---------------------------------------------------------------------
def build_inclusion(conn: sqlite3.Connection) -> None:
    df = normalize_columns(read_xlsx(XLSX_INCLUSION, SHEET_INCLUSION, dtype=str))
    # mapping attendu (selon tes traces):
    # - "Numéro inclusion" -> NUM_INCLUSION
    # - "*IPP*" -> *IPP*
    required_src = ["Numéro inclusion", "*IPP*"]
    for c in required_src:
        if c not in df.columns:
            fatal(f"[FATAL] Colonne manquante dans inclusion: '{c}'. Colonnes trouvées={list(df.columns)[:30]}...")

    out = pd.DataFrame({
        "NUM_INCLUSION": df["Numéro inclusion"].map(safe_str),
        "*IPP*": df["*IPP*"].map(safe_str),
    })
    # nettoyage lignes vides
    out = out[(out["NUM_INCLUSION"] != "") & (out["*IPP*"] != "")]
    to_sql_replace(conn, out, "INCLUSION_RECHERCHE_CLINIQUE")

    # index utile
    sql_exec(conn, """
    CREATE INDEX IF NOT EXISTS idx_inclusion_num_inclusion ON INCLUSION_RECHERCHE_CLINIQUE("NUM_INCLUSION");
    CREATE INDEX IF NOT EXISTS idx_inclusion_ipp ON INCLUSION_RECHERCHE_CLINIQUE("*IPP*");
    """)

def build_recueil(conn: sqlite3.Connection) -> None:
    df = normalize_columns(read_xlsx(XLSX_RECUEIL, SHEET_RECUEIL, dtype=str))
    # attendu: "Numéro anonymat" + "diag endo profonde *" (tu as vu ça dans le probe)
    # NB: on conserve aussi toutes les colonnes en brut (93 cols) pour compat / usages futurs.
    if "Numéro anonymat" not in df.columns:
        fatal("[FATAL] Colonne manquante dans recueil: 'Numéro anonymat'.")

    # le libellé exact peut varier ("diag endo profonde" / "diag endo profonde *")
    diag_candidates = [c for c in df.columns if c.lower().strip().startswith("diag endo profonde")]
    if not diag_candidates:
        fatal("[FATAL] Colonne diag endo profonde introuvable dans recueil.")
    diag_col = diag_candidates[0]

    info(f"[INFO] Recueil: diag colonne détectée = '{diag_col}'")

    # On garde tout, mais on renomme au moins l'identifiant attendu par nos requêtes
    df = df.copy()
    df["NUM_INCLUSION"] = df["Numéro anonymat"].map(safe_str)
    # normalise une colonne simple utilisée plus tard
    df["diag endo profonde"] = df[diag_col].map(safe_str)
    to_sql_replace(conn, df, "Recueil_MMJ")

    sql_exec(conn, """
    CREATE INDEX IF NOT EXISTS idx_recueil_num_inclusion ON Recueil_MMJ("NUM_INCLUSION");
    """)

def build_gyneco_diag_raw(conn: sqlite3.Connection) -> None:
    df = normalize_columns(read_xlsx(XLSX_GYNECO, SHEET_GYNECO, dtype=str))

    # Colonnes confirmées par toi:
    ipp_col = "#IPP"
    date_col = "Gynécologie > Consultation>Date consultation"
    consult_col = "Gynécologie > Consultation>Consultation"

    for c in [ipp_col, date_col, consult_col]:
        if c not in df.columns:
            fatal(f"[FATAL] Colonne manquante dans gyneco: '{c}'.")

    out = pd.DataFrame({
        "IPP": df[ipp_col].map(safe_str),
        "date_consultation": df[date_col].map(safe_str),
        "diag_gyneco": df[consult_col].map(safe_str),
    })

    # filtre lignes non vides
    out = out[(out["IPP"] != "") & (out["diag_gyneco"] != "")]
    to_sql_replace(conn, out, "gyneco_diag_raw")

    sql_exec(conn, """
    CREATE INDEX IF NOT EXISTS idx_gyneco_raw_ipp ON gyneco_diag_raw("IPP");
    """)

def build_pmsi_diag(conn: sqlite3.Connection) -> None:
    # 1) Lecture standard
    df = normalize_columns(read_xlsx(XLSX_PMSI, SHEET_PMSI_DIAGS, dtype=str, header=0))

    # 2) Si NUM_INCLUSION absent, relire en utilisant la 3e ligne comme header (header=2)
    if "NUM_INCLUSION" not in df.columns:
        info("[WARN] PMSI: 'NUM_INCLUSION' absent avec header=0. "
             "On relit TAB_HOSPIT_DIAGS avec header=2 (3e ligne).")
        df = normalize_columns(read_xlsx(XLSX_PMSI, SHEET_PMSI_DIAGS, dtype=str, header=2))

    # 3) Vérifs colonnes attendues (selon ton fichier)
    required = ["NUM_INCLUSION", "CIMCD", "CIM_LIBELLE_COMPLET"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        fatal(
            "[FATAL] PMSI/TAB_HOSPIT_DIAGS: colonnes manquantes après relecture.\n"
            f"  - Manquantes: {missing}\n"
            f"  - Colonnes trouvées: {list(df.columns)}"
        )

    # 4) Construire un champ diag lisible
    def make_diag_row(row) -> str:
        code = safe_str(row.get("CIMCD", ""))
        lib = safe_str(row.get("CIM_LIBELLE_COMPLET", ""))
        if code and lib:
            return f"{code} - {lib}"
        return code or lib

    tmp = df.copy()
    tmp["NUM_INCLUSION"] = tmp["NUM_INCLUSION"].map(safe_str)
    tmp["diag_item"] = tmp.apply(make_diag_row, axis=1).map(safe_str)
    tmp = tmp[(tmp["NUM_INCLUSION"] != "") & (tmp["diag_item"] != "")]

    # 5) Table brute + agrégation distincte
    to_sql_replace(conn, tmp[["NUM_INCLUSION", "diag_item"]], "pmsi_diag_items")

    sql_exec(conn, """
    DROP TABLE IF EXISTS pmsi_diag;
    CREATE TABLE pmsi_diag AS
    WITH dedup AS (
      SELECT DISTINCT NUM_INCLUSION, diag_item
      FROM pmsi_diag_items
      WHERE trim(diag_item) <> ''
    )
    SELECT
      NUM_INCLUSION,
      group_concat(diag_item, ' | ') AS diag_pmsi
    FROM dedup
    GROUP BY NUM_INCLUSION;
    CREATE INDEX IF NOT EXISTS idx_pmsi_diag_num_inclusion ON pmsi_diag("NUM_INCLUSION");
    """)

    info("[OK] pmsi_diag construit (à partir de pmsi_diag_items).")

def ensure_diag_corrections(conn: sqlite3.Connection) -> None:
    # Table UI: persistance du "texte après" et statut de correction/validation
    sql_exec(conn, """
    CREATE TABLE IF NOT EXISTS diag_corrections (
        id_patiente        TEXT PRIMARY KEY,
        texte_apres        TEXT,
        status_texte       TEXT NOT NULL DEFAULT 'attente_correction',
        applied_keys_json  TEXT,
        updated_at         TEXT,
        updated_by         TEXT,
        validated_at       TEXT,
        validated_by       TEXT
    );

    CREATE INDEX IF NOT EXISTS idx_diag_corrections_status ON diag_corrections(status_texte);
    """)


def ensure_diag_validations(conn: sqlite3.Connection) -> None:
    # Table utilisée par l'appli pour stocker les validations humaines
    sql_exec(conn, """
    CREATE TABLE IF NOT EXISTS diag_validations (
        id_patiente            TEXT PRIMARY KEY,
        diagnostic_medical     TEXT,
        validation_statut      TEXT,
        raison_incoherence     TEXT,
        utilisateur_validation TEXT,
        date_validation        TEXT
    );
    """)

def build_diag_comparaison(conn: sqlite3.Connection) -> None:
    # Table pivot: recueil vs gyneco vs pmsi + validations
    sql_exec(conn, """
    DROP TABLE IF EXISTS diag_comparaison;

    CREATE TABLE diag_comparaison AS
    WITH
    map_inclusion AS (
        SELECT
            CAST("NUM_INCLUSION" AS TEXT) AS id_patiente,
            CAST("*IPP*" AS TEXT)        AS IPP
        FROM INCLUSION_RECHERCHE_CLINIQUE
    ),

    -- gyneco: agrégation distincte par inclusion
    gyneco_distinct AS (
        SELECT DISTINCT
            m.id_patiente,
            trim(g.diag_gyneco) AS diag_gyneco
        FROM gyneco_diag_raw g
        JOIN map_inclusion m
          ON m.IPP = CAST(g.IPP AS TEXT)
        WHERE g.diag_gyneco IS NOT NULL
          AND trim(g.diag_gyneco) <> ''
    ),
    gyneco_by_inclusion AS (
        SELECT
            id_patiente,
            group_concat(diag_gyneco, ' | ') AS diag_gyneco
        FROM gyneco_distinct
        GROUP BY id_patiente
    ),

    -- pmsi: déjà agrégé
    pmsi_by_inclusion AS (
        SELECT
            CAST(NUM_INCLUSION AS TEXT) AS id_patiente,
            diag_pmsi
        FROM pmsi_diag
    ),

    -- recueil: diag endo profonde normalisé
    recueil_by_inclusion AS (
        SELECT
            CAST(NUM_INCLUSION AS TEXT) AS id_patiente,
            CASE
                WHEN lower(trim("diag endo profonde")) IN ('vrai','oui','true','1') THEN 'VRAI'
                WHEN lower(trim("diag endo profonde")) IN ('faux','non','false','0') THEN 'FAUX'
                ELSE trim("diag endo profonde")
            END AS diag_recueil
        FROM Recueil_MMJ
    ),

    validations AS (
        SELECT
            CAST(id_patiente AS TEXT) AS id_patiente,
            diagnostic_medical,
            validation_statut,
            raison_incoherence,
            utilisateur_validation,
            date_validation
        FROM diag_validations
    )

    SELECT
        CAST(i.NUM_INCLUSION AS TEXT) AS id_patiente,
        r.diag_recueil,
        g.diag_gyneco,
        p.diag_pmsi,
        v.diagnostic_medical,
        v.validation_statut,
        v.raison_incoherence,
        v.utilisateur_validation,
        v.date_validation
    FROM INCLUSION_RECHERCHE_CLINIQUE i
    LEFT JOIN recueil_by_inclusion r ON r.id_patiente = CAST(i.NUM_INCLUSION AS TEXT)
    LEFT JOIN gyneco_by_inclusion  g ON g.id_patiente = CAST(i.NUM_INCLUSION AS TEXT)
    LEFT JOIN pmsi_by_inclusion    p ON p.id_patiente = CAST(i.NUM_INCLUSION AS TEXT)
    LEFT JOIN validations          v ON v.id_patiente = CAST(i.NUM_INCLUSION AS TEXT);

    CREATE INDEX IF NOT EXISTS idx_diag_comparaison_id_patiente ON diag_comparaison(id_patiente);
    """)

    info("[OK] diag_comparaison construite.")

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main() -> int:
    info(f"[INFO] BASE_DIR={BASE_DIR}")
    info(f"[INFO] DATA_RAW={DATA_RAW}")
    info(f"[INFO] DB_PATH={DB_PATH}")

    # check files
    for p in [XLSX_INCLUSION, XLSX_RECUEIL, XLSX_GYNECO, XLSX_PMSI]:
        info(f"[CHECK] {p.name}")
        ensure_real_xlsx(p)
        info("        -> OK (vrai xlsx)")

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    # SQLite
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute("PRAGMA foreign_keys = OFF;")
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA synchronous = NORMAL;")

        info("\n=== STEP 1: INCLUSION ===")
        build_inclusion(conn)

        info("\n=== STEP 2: RECUEIL ===")
        build_recueil(conn)

        info("\n=== STEP 3: GYNECO (raw) ===")
        build_gyneco_diag_raw(conn)

        info("\n=== STEP 4: PMSI (diag) ===")
        build_pmsi_diag(conn)

        info("\n=== STEP 5: diag_validations (empty if absent) ===")
        ensure_diag_validations(conn)

        info("\n=== STEP 5b: diag_corrections (empty if absent) ===")
        ensure_diag_corrections(conn)

        info("\n=== STEP 6: diag_comparaison ===")
        build_diag_comparaison(conn)

        conn.commit()
        info("\n[OK] Terminé. Base prête pour l'appli.")
        return 0

    except SystemExit:
        # fatal() déclenche SystemExit
        raise
    except Exception as e:
        info("[ERROR] Exception non gérée.")
        info(repr(e))
        info("[TRACEBACK]\n" + traceback.format_exc())
        try:
            conn.rollback()
        except Exception:
            pass
        return 1
    finally:
        conn.close()

if __name__ == "__main__":
    raise SystemExit(main())
