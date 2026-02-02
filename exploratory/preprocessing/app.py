from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from os import path
from pathlib import Path
import json
from unittest import result
import zipfile
import re
import sqlite3
from typing import Dict, List, Optional, Tuple

import pandas as pd
from extract_text_and_vocab_from_dossier_gyneco import OUT_DIR
from flask import Flask, render_template, jsonify,request, redirect, url_for, session
from flask import render_template_string


from flask import render_template_string, url_for
import json

HTML_EXPORT_RESULT = """
<!doctype html>
<html lang="fr">
<head>
  <meta charset="utf-8">
  <title>Export terminé</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 24px; }
    .box { max-width: 900px; padding: 16px; border: 1px solid #ddd; border-radius: 8px; }
    pre { background: #f7f7f7; padding: 12px; border-radius: 6px; overflow:auto; }
    a.btn {
      display:inline-block;
      padding: 8px 12px;
      border: 1px solid #ddd;
      border-radius: 6px;
      background:#fafafa;
      text-decoration:none;
      color:#000;
    }
    a.btn:hover { background:#f0f0f0; }
  </style>
</head>
<body>
  <div class="box">
    <h2>Export terminé</h2>
    <pre>{{ payload }}</pre>
    <p>
      <a class="btn" href="{{ back_url }}">Retour à la liste des patientes</a>
    </p>
  </div>
</body>
</html>
"""


# -----------------------------------------------------------------------------
# Paths / config (DB-first)
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "endopath_diag.db"

DATA_DIR = BASE_DIR / "Data" / "DATA_PROCESSED"
CORR_DIR = DATA_DIR / "Correction_mots"

DICT_JSON_PATH = CORR_DIR / "dictionnaire_correction.json"

# CSV outputs produced by suggest_dict_extensions.py
# (We support both the "old" names and the new split names.)
SUGG_FILES = {
    "TYPO": [
        CORR_DIR / "suggestions_auto_typos.csv",
    ],
    "DIACRITICS_STRICT": [
        CORR_DIR / "suggestions_auto_diacritics_strict.csv",
        CORR_DIR / "suggestions_auto_diacritics.csv",
    ],
    "DIACRITICS_MULTI": [
        CORR_DIR / "suggestions_auto_diacritics_multi.csv",
        CORR_DIR / "suggestions_auto_diacritics.csv",
    ],

    # ✅ ABBREV SÛRES : uniquement ce fichier
    "ABBREV_SURE": [
        CORR_DIR / "suggestions_auto_abbrev.csv",
    ],

    # ✅ ABBREV AMBIGUËS : uniquement ce fichier
    "ABBREV_AMBIGU": [
        CORR_DIR / "suggestions_auto_abbrev_ambigu.csv",
    ],

    # ✅ ABBREV ? (candidates) : uniquement ce fichier
    "ABBREV_CANDIDATE": [
        CORR_DIR / "suggestions_auto_abbrev_candidate.csv",
    ],

    "MANUAL_DICT": [
        CORR_DIR / "suggestions_manual_dict.csv",
    ],
    "NOISE": [
        CORR_DIR / "suggestions_auto_rejected.csv",
    ],
    "INTENSITY": [CORR_DIR / "suggestions_auto_intensity.csv"],
}


# -----------------------------------------------------------------------------
# Families / UI mapping (distinct per family as requested)
# -----------------------------------------------------------------------------
FAMILY_META = {
    "DICT": dict(label="DICT", ui_badge_css="fam-dict", is_auto=True),
    "PHRASE_DICT": dict(label="PHRASE", ui_badge_css="fam-phrase", is_auto=True),
    "TYPO": dict(label="TYPO", ui_badge_css="fam-typo", is_auto=True),
    "DIACRITICS_STRICT": dict(label="ACCENT", ui_badge_css="fam-accent", is_auto=True),
    "DIACRITICS_MULTI": dict(label="ACCENT+", ui_badge_css="fam-accentplus", is_auto=True),
    "ABBREV_SURE": dict(label="ABRÉV SÛRE", ui_badge_css="fam-abbrev-sure", is_auto=True),
    "ABBREV_CANDIDATE": dict(label="ABRÉV ?", ui_badge_css="fam-abbrev-cand", is_auto=True),
    "EXPERT_NORMALIZATION": dict(label="EXPERT", ui_badge_css="fam-expert", is_auto=True),

    "ABBREV_AMBIGU": dict(label="ABRÉV AMBIGUË", ui_badge_css="fam-choice", is_auto=False),
    "MORPHO_AGREEMENT": dict(label="ACCORD", ui_badge_css="fam-choice", is_auto=False),
    "ENRICH_METIER": dict(label="MÉTIER", ui_badge_css="fam-choice", is_auto=False),

    "ALERTE_SEMANTIQUE": dict(label="ALERTE", ui_badge_css="fam-alert", is_auto=False),
    "NOISE": dict(label="BRUIT", ui_badge_css="fam-noise", is_auto=False),
    "INTENSITY": dict(label="INTENSITÉ", ui_badge_css="fam-typo", is_auto=True),
}


DEFAULT_FAMILIES = [
    "PHRASE_DICT",
    "DICT",
    "INTENSITY",
    "DIACRITICS_STRICT",
    "DIACRITICS_MULTI",
    #"TYPO",
    "ABBREV_SURE",
    "ABBREV_CANDIDATE",
    "ABBREV_AMBIGU",
    "MORPHO_AGREEMENT",
    "ENRICH_METIER",
]

ABBREV_AMBIGU_ALLOW_LOWER = {"pro"}

NOISE_TOKENS = {"nan", "none", "null"}

ABBREV_LOWER_WHITELIST = {"sf", "tt", "cp", "qq", "j","ex"}


# Choice-only abbreviations list (blue)
ABBREV_AMBIGU_SET = {
    "sp", "aw", "ec", "tv", "tono", "sd", "su", "cr", "lus", "dig",
}

PUBLIC_ENDPOINTS = {"login", "static"}  # static = fichiers css/js, etc.

_WORDISH_RE = re.compile(r"^[0-9A-Za-zÀ-ÖØ-öø-ÿ_-]+$")

# INTENSITY clinique: mot++ / mot+++ / mot++++  => mot (++)
_INTENSITY_RE = re.compile(r"(?<!\w)(?![A-Za-z]\+\+)([A-Za-zÀ-ÖØ-öø-ÿ]{2,})(\+{2,4})(?!\w)")

# ---------------------------------------------------------------------
# Nettoyage UI: suppression des séparateurs "=====" (pollution)
# ---------------------------------------------------------------------
_RE_EQ_LINE = re.compile(r"(?m)^\s*={3,}\s*$")   # lignes composées uniquement de '='
_RE_EQ_RUN  = re.compile(r"={3,}")              # séquences longues dans une ligne

def strip_eq_separators(text: str) -> str:
    """
    Supprime les séparateurs de type '=====' (souvent utilisés comme barres visuelles).
    - retire les lignes entières faites de '='
    - remplace les longues séquences '=' restantes par un espace
    Ne touche pas aux '=' isolés (ex: 'Na=140' ou 'A=B' si ça arrive).
    """
    if not text:
        return text
    text = _RE_EQ_LINE.sub("", text)
    text = _RE_EQ_RUN.sub(" ", text)
    # évite les doubles espaces créés par le remplacement
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text


def _highlight_literal(text: str, token: str, css_class: str, case_sensitive: bool = False) -> str:
    """
    Surligne token dans text en échappant la regex.
    Utilise des bornes de mots UNIQUEMENT si token est “word-like”.
    """
    if not token:
        return text

    flags = 0 if case_sensitive else re.IGNORECASE
    tok = token.strip()

    # échappement regex pour +, (, ), ?, etc.
    pat = re.escape(tok)

    # Bornes de mots seulement si token est “word-like”
    if _WORDISH_RE.match(tok):
        pat = r"\b" + pat + r"\b"

    return re.sub(
        pat,
        lambda m: f'<span class="{css_class}">{m.group(0)}</span>',
        text,
        flags=flags,
    )

def _count_occ_word_case_sensitive(text: str, token: str) -> int:
    if not token:
        return 0
    pattern = r"\b" + re.escape(token) + r"\b"
    return len(re.findall(pattern, text))   # <-- pas de lower, pas IGNORECASE

import re

def _strip_abbrev_annotations(text: str, tokens_cs: list[str]) -> str:
    """
    Enlève des formes: TOKEN [quelque chose] -> TOKEN
    pour une liste de tokens (casse respectée).
    """
    out = text
    for token_cs in tokens_cs:
        token_cs = (token_cs or "").strip()
        if not token_cs:
            continue
        # ex: SP [santé publique]  -> SP
        pat = r"\b" + re.escape(token_cs) + r"\b\s*\[\s*[^]\r\n]*?\s*\]"
        out = re.sub(pat, token_cs, out)
    return out


def _apply_abbrev_annotations(text: str, abbrev_choices: dict[str, str]) -> str:
    """
    Version idempotente:
    - enlève d'abord toute annotation existante TOKEN [..]
    - puis applique l'annotation courante TOKEN [expansion]
    """
    out = text

    tokens = []
    for token_cs, expansion in (abbrev_choices or {}).items():
        token_cs = (token_cs or "").strip()
        expansion = (expansion or "").strip()
        if token_cs:
            tokens.append(token_cs)

    # 1) strip ancien choix (même si on va remettre autre chose)
    if tokens:
        out = _strip_abbrev_annotations(out, tokens)

    # 2) appliquer choix courant
    for token_cs, expansion in (abbrev_choices or {}).items():
        token_cs = (token_cs or "").strip()
        expansion = (expansion or "").strip()
        if not token_cs or not expansion:
            continue

        pattern = r"\b" + re.escape(token_cs) + r"\b"
        repl = token_cs + " [" + expansion + "]"
        out = re.sub(pattern, repl, out)

    return out



EXPORT_DIR = Path(__file__).resolve().parent / "export_patientes"


def _ensure_export_dir() -> Path:
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    return EXPORT_DIR


def _now_iso() -> str:
    # ISO 8601 avec timezone
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _get_after_text_for_export(conn, num_inclusion: str) -> str:
    """
    Règle métier export:
      - si diag_corrections.texte_apres existe et non vide -> priorité
      - sinon diag_comparaison.diag_gyneco
    """
    num_inclusion = (num_inclusion or "").strip()
    if not num_inclusion:
        return ""

    # 1) diag_corrections.texte_apres
    if _table_exists(conn, "diag_corrections") and _column_exists(conn, "diag_corrections", "texte_apres"):
        key_col = "id_patiente" if _column_exists(conn, "diag_corrections", "id_patiente") else "num_inclusion"
        row = conn.execute(
            f"SELECT texte_apres FROM diag_corrections WHERE {key_col} = ?",
            (num_inclusion,),
        ).fetchone()
        if row:
            try:
                txt = row["texte_apres"]
            except Exception:
                txt = row[0]
            if (txt or "").strip():
                return str(txt)

    # 2) diag_comparaison.diag_gyneco
    if _table_exists(conn, "diag_comparaison") and _column_exists(conn, "diag_comparaison", "diag_gyneco"):
        # Votre DB semble utiliser d.id_patiente ; gardons un fallback
        if _column_exists(conn, "diag_comparaison", "id_patiente"):
            row = conn.execute(
                "SELECT diag_gyneco FROM diag_comparaison WHERE id_patiente = ?",
                (num_inclusion,),
            ).fetchone()
        else:
            # fallback si autre schéma
            row = conn.execute(
                "SELECT diag_gyneco FROM diag_comparaison WHERE num_inclusion = ?",
                (num_inclusion,),
            ).fetchone()

        if row:
            try:
                txt = row["diag_gyneco"]
            except Exception:
                txt = row[0]
            return "" if txt is None else str(txt)

    return ""


def _export_patient_json(conn, num_inclusion: str) -> dict:
    """
    Produit le JSON minimal de livraison.
    """
    after_text = _get_after_text_for_export(conn, num_inclusion)
    return {
        "schema_version": "1.0",
        "exported_at": _now_iso(),
        "patient_id": num_inclusion,
        "text_after": after_text,
    }


# -----------------------------------------------------------------------------
# Data structures
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class Suggestion:
    family: str
    token_source: str
    match: str  # empty for "choice-only" suggestions
    patient_occ: int
    total_count: Optional[int] = None
    score: Optional[float] = None
    note: Optional[str] = None

    @property
    def is_auto(self) -> bool:
        return bool(FAMILY_META.get(self.family, {}).get("is_auto", False)) and bool(self.match)

    @property
    def family_label(self) -> str:
        return FAMILY_META.get(self.family, {}).get("label", self.family)

    @property
    def badge_css(self) -> str:
        return FAMILY_META.get(self.family, {}).get("ui_badge_css", "fam-unk")

    def key(self) -> str:
        return f"{self.family}::{self.token_source}::{self.match}"

# -----------------------------------------------------------------------------
# Helpers: DB, dictionary, CSV
# -----------------------------------------------------------------------------
def _table_exists(conn, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (name,),
    ).fetchone()
    return row is not None


def _column_exists(conn, table: str, col: str) -> bool:
    try:
        rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    except Exception:
        return False
    cols = {r[1] for r in rows}  # r[1] = name
    return col in cols


def get_patient_correction(conn: sqlite3.Connection, num_inclusion: str) -> sqlite3.Row | None:
    q = """
    SELECT id_patiente, texte_apres, status_texte, updated_at, updated_by, validated_at, validated_by
    FROM diag_corrections
    WHERE id_patiente = ?
    """
    return conn.execute(q, (num_inclusion,)).fetchone()


def upsert_patient_correction(
    conn: sqlite3.Connection,
    num_inclusion: str,
    texte_apres: str,
    status_texte: str,
    username: str,
    applied_keys_json: str | None = None,
) -> None:
    # Règle: toute sauvegarde "Appliquer la sélection" remet la validation à zéro
    q = """
    INSERT INTO diag_corrections(
        id_patiente, texte_apres, status_texte, applied_keys_json,
        updated_at, updated_by, validated_at, validated_by
    )
    VALUES (?, ?, ?, ?, datetime('now'), ?, NULL, NULL)
    ON CONFLICT(id_patiente) DO UPDATE SET
        texte_apres       = excluded.texte_apres,
        status_texte      = excluded.status_texte,
        applied_keys_json = excluded.applied_keys_json,
        updated_at        = excluded.updated_at,
        updated_by        = excluded.updated_by,
        validated_at      = NULL,
        validated_by      = NULL
    """
    conn.execute(q, (num_inclusion, texte_apres, status_texte, applied_keys_json, username))


def validate_patient_correction(conn: sqlite3.Connection, num_inclusion: str, username: str) -> None:
    # Valide sans modifier le texte (on garde la dernière sauvegarde)
    q = """
    INSERT INTO diag_corrections(
        id_patiente, status_texte, updated_at, updated_by, validated_at, validated_by
    )
    VALUES (?, 'corrections_validees', datetime('now'), ?, datetime('now'), ?)
    ON CONFLICT(id_patiente) DO UPDATE SET
        status_texte = 'corrections_validees',
        updated_at   = datetime('now'),
        updated_by   = excluded.updated_by,
        validated_at = datetime('now'),
        validated_by = excluded.validated_by
    """
    conn.execute(q, (num_inclusion, username, username))



def get_db_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def get_all_patients(conn):
    """
    Retourne une liste de patientes (id_patiente / NUM_INCLUSION) + statut.

    Supporte:
    - legacy: table `patients`
    - actuel: diag_comparaison + diag_corrections (+ diag_validations éventuelle)
    - fallback: INCLUSION_RECHERCHE_CLINIQUE.NUM_INCLUSION
    """

    def _pick_join_col(table: str, preferred: str, fallback: str):
        if _column_exists(conn, table, preferred):
            return preferred
        if _column_exists(conn, table, fallback):
            return fallback
        return None

    def _pick_validation_date_col():
        # DB possibles: validated_at (rare) ou date_validation (chez vous)
        if _column_exists(conn, "diag_validations", "validated_at"):
            return "validated_at"
        if _column_exists(conn, "diag_validations", "date_validation"):
            return "date_validation"
        return None

    # --- 1) Legacy: table patients
    if _table_exists(conn, "patients"):
        p_key = _pick_join_col("patients", "id_patiente", "NUM_INCLUSION") or "id_patiente"

        if _table_exists(conn, "diag_corrections"):
            c_key = _pick_join_col("diag_corrections", "id_patiente", "num_inclusion")
        else:
            c_key = None

        if c_key:
            q = f"""
            SELECT
                p.{p_key} AS id_patiente,
                COALESCE(c.status_texte, p.status_texte, 'attente_correction') AS status_texte
            FROM patients p
            LEFT JOIN diag_corrections c
                   ON c.{c_key} = p.{p_key}
            ORDER BY p.{p_key}
            """
        else:
            q = f"""
            SELECT
                p.{p_key} AS id_patiente,
                COALESCE(p.status_texte, 'attente_correction') AS status_texte
            FROM patients p
            ORDER BY p.{p_key}
            """
        return conn.execute(q).fetchall()

    # --- 2) DB actuelle: diag_comparaison
    if _table_exists(conn, "diag_comparaison"):
        d_key = _pick_join_col("diag_comparaison", "id_patiente", "NUM_INCLUSION") or "id_patiente"

        has_corr = _table_exists(conn, "diag_corrections")
        c_key = _pick_join_col("diag_corrections", "id_patiente", "num_inclusion") if has_corr else None
        has_c_validated_at = has_corr and _column_exists(conn, "diag_corrections", "validated_at")

        has_valid = _table_exists(conn, "diag_validations")
        v_key = _pick_join_col("diag_validations", "id_patiente", "num_inclusion") if has_valid else None
        v_date_col = _pick_validation_date_col() if has_valid else None

        join_corr = f"LEFT JOIN diag_corrections c ON c.{c_key} = d.{d_key}" if (has_corr and c_key) else ""
        join_valid = f"LEFT JOIN diag_validations v ON v.{v_key} = d.{d_key}" if (has_valid and v_key) else ""

        cond_corr_valid = "0"
        if has_c_validated_at:
            cond_corr_valid = "c.validated_at IS NOT NULL AND trim(c.validated_at) <> ''"

        cond_v_valid = "0"
        if has_valid and v_date_col:
            cond_v_valid = f"v.{v_date_col} IS NOT NULL AND trim(v.{v_date_col}) <> ''"

        q = f"""
        SELECT
            d.{d_key} AS id_patiente,
            CASE
                WHEN ({cond_corr_valid}) THEN 'corrections_validees'
                WHEN ({cond_v_valid}) THEN 'corrections_validees'
                WHEN c.status_texte IS NOT NULL AND trim(c.status_texte) <> '' THEN c.status_texte
                WHEN d.validation_statut IS NOT NULL AND trim(d.validation_statut) <> '' THEN d.validation_statut
                ELSE 'attente_correction'
            END AS status_texte
        FROM diag_comparaison d
        {join_corr}
        {join_valid}
        ORDER BY d.{d_key}
        """
        return conn.execute(q).fetchall()

    # --- 3) Fallback: INCLUSION_RECHERCHE_CLINIQUE
    if _table_exists(conn, "INCLUSION_RECHERCHE_CLINIQUE") and _column_exists(conn, "INCLUSION_RECHERCHE_CLINIQUE", "NUM_INCLUSION"):
        has_corr = _table_exists(conn, "diag_corrections")
        c_key = _pick_join_col("diag_corrections", "id_patiente", "num_inclusion") if has_corr else None

        if has_corr and c_key:
            q = f"""
            SELECT
                i.NUM_INCLUSION AS id_patiente,
                COALESCE(c.status_texte, 'attente_correction') AS status_texte
            FROM INCLUSION_RECHERCHE_CLINIQUE i
            LEFT JOIN diag_corrections c
                   ON c.{c_key} = i.NUM_INCLUSION
            ORDER BY i.NUM_INCLUSION
            """
        else:
            q = """
            SELECT
                i.NUM_INCLUSION AS id_patiente,
                'attente_correction' AS status_texte
            FROM INCLUSION_RECHERCHE_CLINIQUE i
            ORDER BY i.NUM_INCLUSION
            """
        return conn.execute(q).fetchall()

    return []


    # ------------------------------------------------------------------
    # 3) Fallback minimal : INCLUSION_RECHERCHE_CLINIQUE.NUM_INCLUSION
    # ------------------------------------------------------------------
    if (
        _table_exists(conn, "INCLUSION_RECHERCHE_CLINIQUE")
        and _column_exists(conn, "INCLUSION_RECHERCHE_CLINIQUE", "NUM_INCLUSION")
    ):
        has_corr = _table_exists(conn, "diag_corrections")
        c_key = _pick_join_col("diag_corrections", "id_patiente", "num_inclusion") if has_corr else None

        if has_corr and c_key:
            q = f"""
            SELECT
                i.NUM_INCLUSION AS id_patiente,
                COALESCE(c.status_texte, 'attente_correction') AS status_texte
            FROM INCLUSION_RECHERCHE_CLINIQUE i
            LEFT JOIN diag_corrections c
                   ON c.{c_key} = i.NUM_INCLUSION
            ORDER BY i.NUM_INCLUSION
            """
        else:
            q = """
            SELECT
                i.NUM_INCLUSION AS id_patiente,
                'attente_correction' AS status_texte
            FROM INCLUSION_RECHERCHE_CLINIQUE i
            ORDER BY i.NUM_INCLUSION
            """

        return conn.execute(q).fetchall()

    return []


def get_patient_text(conn, num_inclusion: str) -> str:
    """
    Retourne le texte clinique brut "avant" pour une patiente.

    Priorité (pour éviter la régression) :
    1) diag_comparaison.diag_gyneco (agrégé, c'est le texte à corriger)
    2) fallback : NUM_INCLUSION -> INCLUSION_RECHERCHE_CLINIQUE."*IPP*" (ou IPP) -> gyneco_diag_raw.diag_gyneco
    """
    num_inclusion = (num_inclusion or "").strip()
    if not num_inclusion:
        return ""

    # 1) PRIORITÉ : diag_comparaison.diag_gyneco
    if _table_exists(conn, "diag_comparaison") and _column_exists(conn, "diag_comparaison", "diag_gyneco"):
        row = conn.execute(
            """
            SELECT diag_gyneco
            FROM diag_comparaison
            WHERE id_patiente = ?
            LIMIT 1
            """,
            (num_inclusion,),
        ).fetchone()
        if row and row[0]:
            txt = str(row[0]).strip()
            if txt:
                return txt

    # 2) fallback : map NUM_INCLUSION -> IPP via INCLUSION_RECHERCHE_CLINIQUE
    ipp = None
    if _table_exists(conn, "INCLUSION_RECHERCHE_CLINIQUE"):
        ipp_col_sql = None
        if _column_exists(conn, "INCLUSION_RECHERCHE_CLINIQUE", "*IPP*"):
            ipp_col_sql = '"*IPP*"'  # colonne avec astérisques => guillemets obligatoires
        elif _column_exists(conn, "INCLUSION_RECHERCHE_CLINIQUE", "IPP"):
            ipp_col_sql = "IPP"

        if ipp_col_sql and _column_exists(conn, "INCLUSION_RECHERCHE_CLINIQUE", "NUM_INCLUSION"):
            row = conn.execute(
                f"SELECT {ipp_col_sql} FROM INCLUSION_RECHERCHE_CLINIQUE WHERE NUM_INCLUSION = ?",
                (num_inclusion,),
            ).fetchone()
            if row:
                ipp = (row[0] or "").strip()

    # 3) lire gyneco_diag_raw
    if ipp and _table_exists(conn, "gyneco_diag_raw") and _column_exists(conn, "gyneco_diag_raw", "diag_gyneco"):
        has_date = _column_exists(conn, "gyneco_diag_raw", "date_consultation")

        if has_date:
            row = conn.execute(
                """
                SELECT diag_gyneco
                FROM gyneco_diag_raw
                WHERE IPP = ?
                ORDER BY date_consultation DESC
                LIMIT 1
                """,
                (ipp,),
            ).fetchone()
        else:
            row = conn.execute(
                """
                SELECT diag_gyneco
                FROM gyneco_diag_raw
                WHERE IPP = ?
                LIMIT 1
                """,
                (ipp,),
            ).fetchone()

        if row and row[0]:
            return str(row[0])

    return ""

def _strip_leading_apres_label(text: str) -> str:
    """
    Évite le bug où le contenu stocké commence par 'Après' (libellé UI injecté).
    """
    if not text:
        return ""
    t = str(text)

    # Supprime une première ligne exactement "Après" (avec espaces) + sauts de ligne
    t = re.sub(r"^\s*Après\s*\r?\n(\r?\n)?", "", t, flags=re.IGNORECASE)
    return t

def get_before_text_from_comparaison(conn, num_inclusion: str) -> str:
    num_inclusion = (num_inclusion or "").strip()
    if not num_inclusion:
        return ""

    if _table_exists(conn, "diag_comparaison") and _column_exists(conn, "diag_comparaison", "diag_gyneco"):
        row = conn.execute(
            """
            SELECT diag_gyneco
            FROM diag_comparaison
            WHERE id_patiente = ?
            LIMIT 1
            """,
            (num_inclusion,),
        ).fetchone()
        if row and row[0]:
            return str(row[0])
    return ""

def get_after_text_from_corrections(conn, num_inclusion: str) -> str:
    num_inclusion = (num_inclusion or "").strip()
    if not num_inclusion:
        return ""

    if _table_exists(conn, "diag_corrections") and _column_exists(conn, "diag_corrections", "texte_apres"):
        # Attention au nom de clé: id_patiente chez vous
        key_col = "id_patiente" if _column_exists(conn, "diag_corrections", "id_patiente") else "num_inclusion"
        row = conn.execute(
            f"""
            SELECT texte_apres
            FROM diag_corrections
            WHERE {key_col} = ?
            LIMIT 1
            """,
            (num_inclusion,),
        ).fetchone()

        if row and row[0]:
            return _strip_leading_apres_label(str(row[0]))

    return ""

def get_before_after_texts(conn, num_inclusion: str) -> tuple[str, str]:
    """
    Règle métier:
    - AVANT : diag_comparaison.diag_gyneco (texte source)
    - APRÈS : diag_corrections.texte_apres si existe et non vide, sinon AVANT
    """
    before_text = get_before_text_from_comparaison(conn, num_inclusion)
    after_text = get_after_text_from_corrections(conn, num_inclusion)

    if not (after_text or "").strip():
        after_text = before_text
    # --- Nettoyage UI (présentation) : retire les séparateurs "====="
    before_text = strip_eq_separators(before_text)
    after_text  = strip_eq_separators(after_text)

    return before_text, after_text


def load_json_dictionary(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    out: Dict[str, str] = {}
    for k, v in data.items():
        if not isinstance(k, str) or not isinstance(v, str):
            continue
        out[k.strip().lower()] = v.strip()
    return out


def split_dictionary(d: Dict[str, str]) -> Tuple[Dict[str, str], Dict[str, str]]:
    single = {}
    phrase = {}
    for k, v in d.items():
        if not k or not v:
            continue
        if re.search(r"\s+", k):
            phrase[k] = v
        else:
            single[k] = v
    return single, phrase

def _read_sugg_csv_any(path_candidates: List[Path], family: str) -> pd.DataFrame:
    std_cols = ["token_source", "match", "count", "score", "edit_dist", "note", "expansions"]

    for p in path_candidates:
        if not p.exists():
            continue

        try:
            # IMPORTANT: p (pas "path") + désactiver NA parsing (NA -> "nan")
            df = pd.read_csv(
                p,
                encoding="utf-8",
                dtype=str,
                keep_default_na=False,
                na_filter=False,
            )
            print("[DBG] read_csv:", p, "rows=", len(df))
            print(df.head(3))

        except pd.errors.EmptyDataError:
            return pd.DataFrame(columns=std_cols)
        except Exception:
            continue

        if df is None or (df.empty and len(df.columns) == 0):
            return pd.DataFrame(columns=std_cols)

        cols = {c.lower().strip(): c for c in df.columns}

        token_col = cols.get("token_source") or cols.get("token") or cols.get("source")
        match_col = cols.get("match") or cols.get("correction") or cols.get("target")
        count_col = cols.get("count") or cols.get("freq") or cols.get("occ")
        score_col = cols.get("score") or cols.get("similarity")
        edit_col = cols.get("edit_dist") or cols.get("edit") or cols.get("distance")
        note_col = cols.get("note") or cols.get("reason")
        expansions_col = cols.get("expansions") or cols.get("choices") or cols.get("options")

        out = pd.DataFrame(columns=std_cols)
        out["token_source"] = df[token_col].astype(str) if token_col else ""
        out["match"] = df[match_col].astype(str) if match_col else ""
        out["count"] = df[count_col] if count_col else None
        out["score"] = df[score_col] if score_col else None
        out["edit_dist"] = df[edit_col] if edit_col else None
        out["note"] = df[note_col].astype(str) if note_col else ""
        out["expansions"] = df[expansions_col].astype(str) if expansions_col else ""

        # Filtre par famille : family ou category
        if "family" in cols:
            fam_col = cols["family"]
            out = out[df[fam_col].astype(str).str.upper().str.strip() == family.upper()].copy()
        elif "category" in cols:
            cat_col = cols["category"]
            out = out[df[cat_col].astype(str).str.upper().str.strip() == family.upper()].copy()

        # Nettoyage
        out["token_source"] = out["token_source"].astype(str).str.strip()
        out["match"] = out["match"].astype(str).str.strip()
        out["expansions"] = out["expansions"].astype(str).str.strip()
        out["note"] = out["note"].astype(str).str.strip()

        out = out[out["token_source"] != ""]
        return out.reset_index(drop=True)

    return pd.DataFrame(columns=std_cols)

def _count_occ_word(text_lower: str, token_lower: str) -> int:
    if not token_lower:
        return 0
    pattern = r"\b" + re.escape(token_lower) + r"\b"
    return len(re.findall(pattern, text_lower))


def _count_occ_phrase(text_lower: str, phrase: str) -> int:
    if not text_lower or not phrase:
        return 0
    # match literal (safe for +, (), ?, etc.)
    pat = re.escape(phrase.lower().strip())
    return len(re.findall(pat, text_lower))


def _looks_like_abbrev(token_lower: str) -> bool:
    return token_lower.isalpha() and 2 <= len(token_lower) <= 5


# -----------------------------------------------------------------------------
# Build suggestions for a given patient text
# -----------------------------------------------------------------------------
def build_patient_suggestions(
    text: str,
    dict_single: Dict[str, str],
    dict_phrase: Dict[str, str],
    enabled_families: List[str],
) -> List[Suggestion]:
    text_lower = text.lower()
    suggestions: List[Suggestion] = []

    # ------------------------------------------------------------
    # PHRASE_DICT depuis dictionnaire (multi-mots)
    # ------------------------------------------------------------
    if "PHRASE_DICT" in enabled_families:
        for src, dst in dict_phrase.items():
            occ = _count_occ_phrase(text_lower, src)
            if occ > 0:
                suggestions.append(Suggestion(
                    family="PHRASE_DICT",
                    token_source=src,
                    match=dst,
                    patient_occ=occ,
                ))

    # ------------------------------------------------------------
    # DICT depuis dictionnaire (mono-mot)
    # ------------------------------------------------------------
    if "DICT" in enabled_families:
        for src, dst in dict_single.items():
            if re.search(r"[^a-z0-9àâçéèêëîïôûùüÿñæœ\-]", src):
                occ = _count_occ_phrase(text_lower, src)   # au lieu de _count_occ_word
            else:
                occ = _count_occ_word_no_apostrophe(text, src)


            if occ > 0:
                suggestions.append(Suggestion(
                    family="DICT",
                    token_source=src,
                    match=dst,
                    patient_occ=occ,
                ))

        # ------------------------------------------------------------
    # INTENSITY (mot++ / mot+++ / mot++++) — indépendant des CSV
    # ------------------------------------------------------------
    if "INTENSITY" in enabled_families:
        # On détecte sur le texte original (case-sensitive), mais on normalise peu
        # pour garder exactement "souple+++" affiché.
        hits = _INTENSITY_RE.findall(text)
        # findall retourne (mot, pluses)
        if hits:
            # Compter correctement par token exact (ex: "souple+++")
            counts: Dict[str, int] = {}
            for w, pluses in hits:
                tok = f"{w}{pluses}"
                counts[tok] = counts.get(tok, 0) + 1

            for tok_cs, occ in sorted(counts.items(), key=lambda x: (-x[1], x[0].lower())):
                # match "mot (+++)"
                m = _INTENSITY_RE.match(tok_cs)
                if not m:
                    continue
                word = m.group(1)
                pluses = m.group(2)

                suggestions.append(Suggestion(
                    family="INTENSITY",
                    token_source=tok_cs,
                    match=f"{word} ({pluses})",
                    patient_occ=occ,
                ))


    # ------------------------------------------------------------
    # Suggestions depuis les CSV
    # ------------------------------------------------------------
    for fam, paths in SUGG_FILES.items():
        if fam not in enabled_families:
            continue

        df = _read_sugg_csv_any(paths, fam)
        if df.empty:
            continue

        for _, r in df.iterrows():
            src_raw = str(r.get("token_source", "")).strip()
            if not src_raw:
                continue

            # Ignore les tokens bruit (nan/none/null) pour toutes les familles CSV
            if src_raw.strip().lower() in NOISE_TOKENS:
                continue

            dst = str(r.get("match", "")).strip()
            out_fam = fam

            # Cas spécial: MANUAL_DICT re-route vers DICT / PHRASE_DICT
            if fam == "MANUAL_DICT":
                src_tmp = src_raw.lower().strip()
                out_fam = "PHRASE_DICT" if re.search(r"\s+", src_tmp) else "DICT"
                if out_fam not in enabled_families:
                    continue

            # ------------------------------------------------------------
            # ABBREV_SURE : matching STRICT sur la casse (texte original)
            # ------------------------------------------------------------
            # Stopwords courts qui ne doivent JAMAIS être traités en abréviation en minuscules
            ABBREV_SURE_STOPWORDS = {"DE", "DU", "DES", "PAR"}

            if out_fam == "ABBREV_SURE":
                src_cs = src_raw.strip()  # ex: "TV", "IRM", "DE"
                if not src_cs:
                    continue

                # Garde-fou : ces tokens ne valent "abrév" que s'ils apparaissent en MAJ
                if src_cs.upper() in ABBREV_SURE_STOPWORDS:
                    # Si dans le CSV tu as "DE" mais que le texte contient "de", le case-sensitive ne matchera pas.
                    # On laisse donc le test normal faire foi, mais on évite tout fallback.
                    pass

                occ = _count_occ_word_case_sensitive(text, src_cs)
                if occ <= 0:
                    continue

                suggestions.append(Suggestion(
                    family="ABBREV_SURE",
                    token_source=src_cs,
                    match=dst,
                    patient_occ=occ,
                    total_count=int(r["count"]) if pd.notna(r.get("count")) else None,
                    score=float(r["score"]) if pd.notna(r.get("score")) else None,
                    note=str(r.get("note")) if pd.notna(r.get("note")) else None,
                ))
                continue


            # ------------------------------------------------------------
            # ABBREV_AMBIGU : matching STRICT sur la casse (texte original)
            # ------------------------------------------------------------
            # ------------------------------------------------------------
            # ABBREV_AMBIGU : matching contrôlé (casse marquée + exception minuscules courtes)
            # ------------------------------------------------------------
            if out_fam == "ABBREV_AMBIGU":
                sigle = src_raw.strip()           # ex: "SF" dans le CSV
                sigle_title = sigle.title()       # "Sf"
                sigle_lower = sigle.lower()       # "sf"

                occ = _count_occ_word_case_sensitive(text, sigle)

                # TitleCase court (Ec, Tv…) : fréquent en clinique
                if occ <= 0 and len(sigle) <= 4:
                    occ = _count_occ_word_case_sensitive(text, sigle_title)

                # Exception contrôlée : certains sigles sont fréquemment tapés en minuscules ("sf", "tt", "cp"...)
                # On ne l'autorise que pour les sigles très courts afin de limiter les faux positifs.
                if occ <= 0 and sigle_lower in ABBREV_LOWER_WHITELIST:
                    occ = _count_occ_word_case_sensitive(text, sigle_lower)


                if occ > 0:
                    expansions_raw = str(r.get("expansions") or "").strip()
                    note_raw = str(r.get("note") or "").strip()

                    if expansions_raw:
                        note = "Possibles : " + expansions_raw.replace("|", " / ")
                    elif note_raw:
                        note = note_raw
                    else:
                        note = "Possibles : (expansions absentes du CSV)"

                    # Affichage : on reprend la forme réellement rencontrée dans le texte (priorité MAJ, puis TitleCase, puis lower)
                    token_display = (
                        sigle if _count_occ_word_case_sensitive(text, sigle) > 0
                        else sigle_title if _count_occ_word_case_sensitive(text, sigle_title) > 0
                        else sigle_lower
                    )

                    suggestions.append(Suggestion(
                        family="ABBREV_AMBIGU",
                        token_source=token_display,
                        match="",
                        patient_occ=occ,
                        note=note,
                    ))
                continue


            # ------------------------------------------------------------
            # Autres familles : matching normal en lower
            # ------------------------------------------------------------
            src = src_raw.lower().strip()

            if re.search(r"\s+", src):
                occ = _count_occ_phrase(text_lower, src)
            else:
                occ = _count_occ_word_no_apostrophe(text, src)


            if occ <= 0:
                continue

            suggestions.append(Suggestion(
                family=out_fam,
                token_source=src,
                match=dst,
                patient_occ=occ,
                total_count=int(r["count"]) if pd.notna(r.get("count")) else None,
                score=float(r["score"]) if pd.notna(r.get("score")) else None,
                note=str(r.get("note")) if pd.notna(r.get("note")) else None,
            ))

            
    # ------------------------------------------------------------
    # BRUIT / NOISE (artefacts) — indépendant des CSV
    # ------------------------------------------------------------
    if "NOISE" in enabled_families:
        for t in sorted(NOISE_TOKENS):
            occ = _count_occ_word(text_lower, t)
            if occ > 0:
                suggestions.append(Suggestion(
                    family="NOISE",
                    token_source=t,
                    match="",
                    patient_occ=occ,
                    note="Artefact ETL (valeur manquante) – à ignorer",
                ))

    # ------------------------------------------------------------
    # Dédoublonnage
    # ------------------------------------------------------------
    uniq = {}
    for s in suggestions:
        uniq[s.key()] = s

    return sorted(
        uniq.values(),
        key=lambda s: (not s.is_auto, s.family, -s.patient_occ, s.token_source),
    )

def _apply_time_unit_normalization(text: str) -> str:
    """
    Normalisation d'unités cliniques courantes :
      - 7j      -> 7 jours
      - 4 j     -> 4 jours
      - 10/j    -> 10 / jour
      - 10 / j  -> 10 / jour
    """
    out = text or ""

        # 10/j -> 10 / jour  (gère "10/j", "10 /j", "10/ j", "10 / j")
    out = re.sub(
        r"\b(\d+)\s*/\s*j\b",
        r"\1 / jour",
        out,
        flags=re.IGNORECASE
    )

    # qq j / qques j / qqs j -> quelques jours
    out = re.sub(
        r"\b(q{1,2}u?e?l?q?u?e?s?|qqs?|qques)\s*j\b",
        "quelques jours",
        out,
        flags=re.IGNORECASE
    )

    # 7j -> 7 jours (gère "7j" et "7 j")
    out = re.sub(
        r"\b(\d+)\s*j\b",
        r"\1 jours",
        out,
        flags=re.IGNORECASE
    )
    return out

def _apply_replacements_html(text: str, applied_sugs: List[Suggestion]) -> str:
    """
    Applique les remplacements et entoure UNIQUEMENT les remplacements effectués
    avec <span class='hl-after-applied'>...</span>.
    """
    out = text

    for s in sorted(applied_sugs, key=lambda x: -len((x.token_source or ""))):
        src = (s.token_source or "").strip()
        dst = (s.match or "").strip()
        if not src or not dst:
            continue

        # Multi-mots
        if re.search(r"\s+", src):
            parts = [re.escape(p) for p in src.split()]
            core = r"\s+".join(parts)
            pattern = r"\b" + core + r"\b(?![’'])"

            def repl(_m, _dst=dst):
                return f"<span class='hl-after-applied'>{_escape_html(_dst)}</span>"

            out = re.sub(pattern, repl, out, flags=re.IGNORECASE)
            continue

        # Mono-token
        core = re.escape(src)
        apostrophe_guard = r"(?![’'])"

        if _WORDISH_RE.match(src):
            pattern = r"\b" + core + r"\b" + apostrophe_guard
        else:
            pattern = core + apostrophe_guard

        def repl(_m, _dst=dst):
            return f"<span class='hl-after-applied'>{_escape_html(_dst)}</span>"

        out = re.sub(pattern, repl, out, flags=re.IGNORECASE)

    return out


# -----------------------------------------------------------------------------
# Apply corrections + highlighting
# -----------------------------------------------------------------------------
def _apply_replacements(text: str, applied_sugs: List[Suggestion]) -> str:
    out = text

    # On remplace d'abord les tokens les plus longs
    for s in sorted(applied_sugs, key=lambda x: -len((x.token_source or ""))):
        src = (s.token_source or "").strip()
        dst = (s.match or "").strip()
        if not src or not dst:
            continue

        # ------------------------------------------------------------------
        # RÈGLE CLÉ :
        # - Les abréviations (sûres / ambiguës / candidates) sont CASE-SENSITIVE
        # - Le reste (DICT, PHRASE, TYPO, etc.) reste IGNORECASE
        # ------------------------------------------------------------------
        flags = re.IGNORECASE
        #if s.family in {"ABBREV_SURE", "ABBREV_AMBIGU", "ABBREV_CANDIDATE"}:
        #    flags = 0

        # --- Cas multi-mots ---
        if re.search(r"\s+", src):
            parts = [re.escape(p) for p in src.split()]
            core = r"\s+".join(parts)
            pattern = r"\b" + core + r"\b(?![’'])"
            out = re.sub(pattern, dst, out, flags=flags)
            continue

        # --- Cas mono-token ---
        core = re.escape(src)

        # Garde-fou apostrophe :
        # un token suivi de ' ou ’ ne doit PAS être remplacé
        apostrophe_guard = r"(?![’'])"

        if _WORDISH_RE.match(src):
            pattern = r"\b" + core + r"\b" + apostrophe_guard
        else:
            pattern = core + apostrophe_guard

        out = re.sub(pattern, dst, out, flags=flags)

    # Normalisation unités (après toutes les corrections)
    out = _apply_time_unit_normalization(out)
    return out


def _count_occ_word_no_apostrophe(text: str, token: str) -> int:
    token = (token or "").strip()
    if not token:
        return 0
    core = re.escape(token.lower())
    pattern = r"\b" + core + r"\b(?![’'])"
    return len(re.findall(pattern, text.lower(), flags=0))


def _highlight_text_before(text: str, selected_sugs: List[Suggestion]) -> str:
    out = text

    for s in sorted(selected_sugs, key=lambda x: -len((x.token_source or ""))):
        src = (s.token_source or "").strip()
        if not src:
            continue

        css = "hl-before-auto" if s.is_auto else "hl-before-choice"

        # ------------------------------------------------------------------
        # RÈGLE CLÉ :
        # - Les abréviations sont surlignées EN RESPECTANT LA CASSE
        # - Le reste reste en IGNORECASE
        # ------------------------------------------------------------------
        flags = re.IGNORECASE
        if s.family in {"ABBREV_SURE", "ABBREV_AMBIGU", "ABBREV_CANDIDATE"}:
            # case-sensitive seulement si c'est une vraie abréviation typique (HSG, TV, 5-FU, etc.)
            if src.upper() == src and any(c.isalpha() for c in src):
                flags = 0


        # --- Cas multi-mots ---
        if re.search(r"\s+", src):
            parts = [re.escape(p) for p in src.split()]
            core = r"\s+".join(parts)
            pattern = r"\b" + core + r"\b(?![’'])"
        else:
            core = re.escape(src)
            if _WORDISH_RE.match(src):
                pattern = r"\b" + core + r"\b(?![’'])"
            else:
                pattern = core

        out = re.sub(
            pattern,
            lambda m: f"<span class='{css}'>{m.group(0)}</span>",
            out,
            flags=flags,
        )

    return out


def _highlight_text_after(
    text_after: str,
    applied_sugs: List[Suggestion],
    abbrev_apply_map: dict[str, str] | None = None,
) -> str:
    out = text_after

    # 1) Vert sur les remplacements auto (match) — avec bornes si token "wordish"
    for s in sorted(applied_sugs, key=lambda x: -len((x.match or ""))):
        dst = (s.match or "").strip()
        if not dst:
            continue

        # garde-fou apostrophe identique au before
        apostrophe_guard = r"(?![’'])"

        if re.search(r"\s+", dst):
            # multi-mots
            parts = [re.escape(p) for p in dst.split()]
            core = r"\s+".join(parts)
            pattern = r"\b" + core + r"\b" + apostrophe_guard
        else:
            core = re.escape(dst)
            if _WORDISH_RE.match(dst):
                pattern = r"\b" + core + r"\b" + apostrophe_guard
            else:
                pattern = core

        out = re.sub(
            pattern,
            lambda m: f"<span class='hl-after-applied'>{m.group(0)}</span>",
            out,
            flags=re.IGNORECASE,
        )

    # 2) ABBREV_AMBIGU : "TV [toucher vaginal]" -> surlignage bleu (choix utilisateur)
    if abbrev_apply_map:
        for token_cs, expansion in abbrev_apply_map.items():
            token_cs = (token_cs or "").strip()
            expansion = (expansion or "").strip()
            if not token_cs or not expansion:
                continue

            pattern = (
                r"\b" + re.escape(token_cs) + r"\b\s*\[\s*"
                + re.escape(expansion) + r"\s*\]"
            )
            out = re.sub(
                pattern,
                lambda m: f"<span class='hl-after-choice'>{m.group(0)}</span>",
                out,
            )

    return out




def split_sentences(text: str) -> List[str]:
    if not text.strip():
        return []
    tmp = text.replace("|", ". ")
    parts = re.split(r"(?<=[\.\!\?])\s+|\n+", tmp)
    return [p.strip() for p in parts if p.strip()]

def join_sentences(sentences: List[str]) -> str:
    """
    Reconstruction simple.
    On joint par ". " si la phrase n'a pas déjà une ponctuation finale.
    (Heuristique volontairement simple et robuste.)
    """
    out = []
    for s in sentences:
        s = (s or "").strip()
        if not s:
            continue
        if re.search(r"[\.!\?]$", s):
            out.append(s)
        else:
            out.append(s + ".")
    return " ".join(out).strip()



# -----------------------------------------------------------------------------
# Flask app
# -----------------------------------------------------------------------------
app = Flask(__name__)
app.secret_key = "endopath-preprocess-demo"

DICT_ALL = load_json_dictionary(DICT_JSON_PATH)
DICT_SINGLE, DICT_PHRASE = split_dictionary(DICT_ALL)

@app.route("/", methods=["GET", "POST"])
def login():
    # Si déjà loggé, aller directement à /patients
    if request.method == "GET" and (session.get("username") or "").strip():
        return redirect(url_for("patients"))

    error = ""
    if request.method == "POST":
        username = (request.form.get("username") or "").strip()
        if not username:
            error = "Veuillez saisir un nom d'utilisateur."
        else:
            session["username"] = username
            return redirect(url_for("patients"))

    return render_template("login.html", error=error)

@app.route("/logout", methods=["GET"])
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/patients")
def patients():
    # Garde-fou : username non vide
    if not (session.get("username") or "").strip():
        return redirect(url_for("login"))

    username = session.get("username", "")
    with get_db_conn() as conn:
        rows = get_all_patients(conn)

    return render_template("patients.html", patients=rows, username=username)

def _parse_families_from_request() -> List[str]:
    fams = request.args.getlist("families")

    # Si le user a soumis le form "familles", on accepte le cas "0 famille"
    if request.args.get("families_present") == "1":
        return fams  # peut être []

    # Sinon (premier affichage), on met les defaults
    if not fams:
        fams = DEFAULT_FAMILIES.copy()

    known = set(FAMILY_META.keys()) | set(SUGG_FILES.keys())
    cleaned = []
    for f in fams:
        f = str(f).strip()
        if f and f in known:
            cleaned.append(f)

    ordered = [f for f in DEFAULT_FAMILIES if f in cleaned]
    for f in cleaned:
        if f not in ordered:
            ordered.append(f)
    return ordered


@app.route("/patient/<num_inclusion>", methods=["GET", "POST"])
def patient(num_inclusion: str):
    username = (session.get("username") or "").strip()
    if not username:
        return redirect(url_for("login"))

    phrase_mode = request.args.get("phrase_mode", "0") == "1"
    enabled_families = _parse_families_from_request()

    # -----------------------------
    # Etat POST (sélections)
    # -----------------------------
    selected_keys: set[str] = set()
    action = ""
    selected_abbrev_choices: dict[str, str] = {}

    if request.method == "POST":
        action = (request.form.get("action") or "").strip()
        selected_keys = set(request.form.getlist("selected_keys"))

        # radios ABBREV_AMBIGU : name="choice_<s.key()>"
        for k in request.form.keys():
            if k.startswith("choice_"):
                skey = k[len("choice_"):]
                val = (request.form.get(k) or "").strip()
                if val:
                    selected_abbrev_choices[skey] = val


    # -----------------------------
    # Lecture texte + suggestions
    # -----------------------------
    with get_db_conn() as conn:
        original_text, _ = get_before_after_texts(conn, num_inclusion)
        saved_row = get_patient_correction(conn, num_inclusion)  # diag_corrections
        corrected_text_db = (saved_row["texte_apres"] if saved_row and "texte_apres" in saved_row.keys() else "") or ""



        all_sugs = build_patient_suggestions(
            text=original_text,
            dict_single=DICT_SINGLE,
            dict_phrase=DICT_PHRASE,
            enabled_families=enabled_families,
        )
        sug_by_key = {s.key(): s for s in all_sugs}

        # Sélection courante
        if request.method == "GET":
            selected = [s for s in all_sugs if s.is_auto]
        else:
            selected = [s for s in all_sugs if s.key() in selected_keys]

        selected_auto = [s for s in selected if s.is_auto]
        selected_choice = [s for s in selected if not s.is_auto]

        # Application auto (remplacements)
        corrected_text = _apply_replacements(original_text, selected_auto)

        # ABBREV_AMBIGU: annoter si (sigle coché) ET (expansion choisie)
        abbrev_apply_map: dict[str, str] = {}
        for skey, expansion in selected_abbrev_choices.items():
            if skey not in selected_keys:
                continue
            s = sug_by_key.get(skey)
            if not s or s.family != "ABBREV_AMBIGU":
                continue
            token_cs = s.token_source  # ex: "TV", "LUS" (case sensitive)
            abbrev_apply_map[token_cs] = expansion

        if abbrev_apply_map:
            corrected_text = _apply_abbrev_annotations(corrected_text, abbrev_apply_map)

        # -----------------------------
        # Persistance / statut
        # -----------------------------
        saved_row = get_patient_correction(conn, num_inclusion)
        
        # (corrected_text_db a déjà appliqué la règle métier)
        if request.method == "GET":
            corrected_text = corrected_text_db


        if request.method == "POST":
            if action == "apply":
                upsert_patient_correction(
                    conn=conn,
                    num_inclusion=num_inclusion,
                    texte_apres=corrected_text,
                    status_texte="corrections_en_cours",
                    username=username,
                    applied_keys_json=None,
                )
                conn.commit()
                saved_row = get_patient_correction(conn, num_inclusion)

            elif action == "validate":
                validate_patient_correction(conn, num_inclusion, username=username)
                conn.commit()
                saved_row = get_patient_correction(conn, num_inclusion)

    # -----------------------------
    # Rendu UI
    # -----------------------------
    highlighted_before = _highlight_text_before(original_text, selected_auto + selected_choice)
    highlighted_after = _highlight_text_after(corrected_text, selected_auto)

    original_sentences = split_sentences(original_text)
    corrected_sentences = split_sentences(corrected_text) if phrase_mode else []

    if len(corrected_sentences) < len(original_sentences):
        corrected_sentences = corrected_sentences + original_sentences[len(corrected_sentences):]
    elif len(corrected_sentences) > len(original_sentences):
        corrected_sentences = corrected_sentences[:len(original_sentences)]


    fam_order = [
        "PHRASE_DICT", "DICT", "ABBREV_SURE",
        "DIACRITICS_STRICT", "DIACRITICS_MULTI","INTENSITY",
        "TYPO", "ABBREV_CANDIDATE", "EXPERT_NORMALIZATION",
        "ABBREV_AMBIGU", "MORPHO_AGREEMENT", "ENRICH_METIER",
        "ALERTE_SEMANTIQUE", "NOISE",
    ]

    def fam_meta(f: str):
        m = FAMILY_META.get(f, {})
        return {
            "family": f,
            "label": m.get("label", f),
            "badge_css": m.get("ui_badge_css", "fam-unk"),
            "is_auto": bool(m.get("is_auto", False)),
        }

    families_ui = [fam_meta(f) for f in fam_order if f in FAMILY_META or f in SUGG_FILES]

    suggestions_by_family: dict[str, list] = {f["family"]: [] for f in families_ui}
    for s in all_sugs:
        suggestions_by_family.setdefault(s.family, []).append(s)

    return render_template(
        "patient.html",
        num_inclusion=num_inclusion,
        username=username,
        now=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        enabled_families=enabled_families,
        families_ui=families_ui,
        suggestions_by_family=suggestions_by_family,
        selected_key_set={s.key() for s in selected},
        selected_abbrev_choices=selected_abbrev_choices,  # IMPORTANT pour radios
        highlighted_before=highlighted_before,
        highlighted_after=highlighted_after,
        phrase_mode=phrase_mode,
        original_sentences=original_sentences,
        corrected_sentences=corrected_sentences,
        original_text_empty=(original_text.strip() == ""),
        status_texte=(saved_row["status_texte"] if saved_row else "attente_correction"),
        saved_updated_at=(saved_row["updated_at"] if saved_row else None),
        saved_updated_by=(saved_row["updated_by"] if saved_row else None),
    )

from flask import jsonify, request
@app.route("/patient/<num_inclusion>/sentence_save", methods=["POST"])
def patient_sentence_save(num_inclusion: str):
    username = (session.get("username") or "").strip()
    if not username:
        return jsonify({"ok": False, "error": "Not logged"}), 401

    data = request.get_json(force=True) or {}

    raw_idx = data.get("sent_idx", -1)   # NE PAS utiliser "or -1" ici
    try:
        sent_idx = int(raw_idx)
    except (TypeError, ValueError):
        sent_idx = -1

    new_sentence = (data.get("new_sentence") or "").strip()


    # Pour reconstruire un "après" cohérent si aucun texte n'est encore sauvegardé,
    # on réutilise le même calcul que preview : selected_keys + abbrev_choices + enabled_families.
    selected_keys = set(data.get("selected_keys") or [])
    abbrev_choices = data.get("abbrev_choices") or {}
    enabled_families = data.get("enabled_families") or []

    if sent_idx < 0:
        return jsonify({"ok": False, "error": "sent_idx invalide"}), 400

    with get_db_conn() as conn:
        original_text = get_patient_text(conn, num_inclusion)
        saved_row = get_patient_correction(conn, num_inclusion)

        # 1) base "after" : priorité au texte déjà sauvegardé
        if saved_row and (saved_row["texte_apres"] or "").strip():
            base_after = saved_row["texte_apres"]
        else:
            # sinon, on reconstruit depuis la sélection comme dans preview
            all_sugs = build_patient_suggestions(
                text=original_text,
                dict_single=DICT_SINGLE,
                dict_phrase=DICT_PHRASE,
                enabled_families=enabled_families,
            )
            sug_by_key = {s.key(): s for s in all_sugs}

            selected = [s for s in all_sugs if s.key() in selected_keys]
            selected_auto = [s for s in selected if s.is_auto]

            base_after = _apply_replacements(original_text, selected_auto)

            # ABBREV_AMBIGU: annotation si choisi
            abbrev_apply_map = {}
            for skey, expansion in (abbrev_choices or {}).items():
                s = sug_by_key.get(skey)
                if not s or s.family != "ABBREV_AMBIGU":
                    continue
                exp = (expansion or "").strip()
                if not exp:
                    continue
                abbrev_apply_map[s.token_source] = exp


            if abbrev_apply_map:
                base_after = _apply_abbrev_annotations(base_after, abbrev_apply_map)

        # 2) split + remplacement phrase i
        after_sents = split_sentences(base_after)

        # PAD pour éviter le 400 "hors plage" si Après plus court que Avant
        before_sents = split_sentences(original_text)
        if len(after_sents) < len(before_sents):
            after_sents.extend(before_sents[len(after_sents):])

        if sent_idx >= len(after_sents):
            return jsonify({
                            "ok": False,
                            "error": "sent_idx hors plage",
                            "sent_idx": sent_idx,
                            "len_after": len(after_sents),
                            "len_before": len(before_sents),
                        }), 400


        after_sents[sent_idx] = new_sentence
        new_after_text = join_sentences(after_sents)

        # 3) sauvegarde DB comme un "apply"
        upsert_patient_correction(
            conn=conn,
            num_inclusion=num_inclusion,
            texte_apres=new_after_text,
            status_texte="corrections_en_cours",
            username=username,
            applied_keys_json=None,
        )
        conn.commit()

    return jsonify({"ok": True})

@app.route("/patient/<num_inclusion>/sentence_reset", methods=["POST"])
def patient_sentence_reset(num_inclusion: str):
    username = (session.get("username") or "").strip()
    if not username:
        return jsonify({"ok": False, "error": "Not logged"}), 401

    data = request.get_json(force=True) or {}
    raw_idx = data.get("sent_idx", None)
    try:
        sent_idx = int(raw_idx)
    except Exception:
        return jsonify({"ok": False, "error": f"sent_idx invalide | idx={raw_idx!r}"}), 400

    selected_keys = set(data.get("selected_keys") or [])
    abbrev_choices = data.get("abbrev_choices") or {}
    enabled_families = data.get("enabled_families") or []

    if sent_idx < 0:
        return jsonify({"ok": False, "error": f"sent_idx invalide | idx={sent_idx}"}), 400

    with get_db_conn() as conn:
        original_text = get_patient_text(conn, num_inclusion)
        saved_row = get_patient_correction(conn, num_inclusion)

        # -----------------------------
        # 1) Recalcule l'état "engine" (comme preview) selon la sélection courante
        # -----------------------------
        all_sugs = build_patient_suggestions(
            text=original_text,
            dict_single=DICT_SINGLE,
            dict_phrase=DICT_PHRASE,
            enabled_families=enabled_families,
        )
        sug_by_key = {s.key(): s for s in all_sugs}
        selected = [s for s in all_sugs if s.key() in selected_keys]
        selected_auto = [s for s in selected if s.is_auto]

        engine_after = _apply_replacements(original_text, selected_auto)

        # ABBREV_AMBIGU: annotation si choix fait
        abbrev_apply_map = {}
        for skey, expansion in (abbrev_choices or {}).items():
            # on ne force pas skey in selected_keys ici, on suit l'état radio envoyé
            s = sug_by_key.get(skey)
            if not s or s.family != "ABBREV_AMBIGU":
                continue
            exp = (expansion or "").strip()
            if exp:
                abbrev_apply_map[s.token_source] = exp

        if abbrev_apply_map:
            engine_after = _apply_abbrev_annotations(engine_after, abbrev_apply_map)

        engine_sents = split_sentences(engine_after)
        if sent_idx >= len(engine_sents):
            return jsonify({"ok": False, "error": "sent_idx hors plage"}), 400

        # -----------------------------
        # 2) Base persistée = texte déjà sauvegardé (si existe), sinon engine_after
        # -----------------------------
        # 2) Base persistée = texte déjà sauvegardé (si existe), sinon engine_after
        saved_after = ""
        if saved_row:
            try:
                saved_after = (saved_row["texte_apres"] or "")
            except Exception:
                # fallback tuple
                try:
                    saved_after = (saved_row[0] or "")
                except Exception:
                    saved_after = ""

        if saved_after.strip():
            base_after = saved_after
        else:
            base_after = engine_after


        base_sents = split_sentences(base_after)
        if sent_idx >= len(base_sents):
            # sécurité : aligne si segmentation différente
            # (on préfère ne pas crasher)
            return jsonify({"ok": False, "error": "sent_idx hors plage (base_after)"}), 400

        # -----------------------------
        # 3) Reset phrase i = phrase engine (comme si jamais modifiée manuellement)
        # -----------------------------
        base_sents[sent_idx] = engine_sents[sent_idx]
        new_after_text = join_sentences(base_sents)

        # Option: si plus aucune différence avec engine_after, on peut supprimer la correction
        # pour revenir totalement à l'état "non modifié manuellement".
        if new_after_text.strip() == engine_after.strip():
            # supprime la ligne diag_corrections (si présente)
            if _table_exists(conn, "diag_corrections"):
                if _column_exists(conn, "diag_corrections", "id_patiente"):
                    conn.execute("DELETE FROM diag_corrections WHERE id_patiente = ?", (num_inclusion,))
                elif _column_exists(conn, "diag_corrections", "num_inclusion"):
                    conn.execute("DELETE FROM diag_corrections WHERE num_inclusion = ?", (num_inclusion,))
            conn.commit()
            return jsonify({"ok": True, "deleted": True})

        # Sinon: on sauvegarde le nouvel "Après"
        upsert_patient_correction(
            conn=conn,
            num_inclusion=num_inclusion,
            texte_apres=new_after_text,
            status_texte="corrections_en_cours",
            username=username,
            applied_keys_json=None,
        )
        conn.commit()

    return jsonify({"ok": True, "deleted": False})


@app.route("/export/patient/<num_inclusion>", methods=["GET"])
def export_patient(num_inclusion: str):
    username = (session.get("username") or "").strip()
    if not username:
        return jsonify({"ok": False, "error": "Not logged"}), 401

    num_inclusion = (num_inclusion or "").strip()
    if not num_inclusion:
        return jsonify({"ok": False, "error": "patient_id vide"}), 400

    _ensure_export_dir()

    with get_db_conn() as conn:
        payload = _export_patient_json(conn, num_inclusion)

    out_path = EXPORT_DIR / f"{num_inclusion}.json"
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    result = {
        "ok": True,
        "patient_id": num_inclusion,
        "file": str(out_path),
    }

    return render_template_string(
        HTML_EXPORT_RESULT,
        payload=json.dumps(result, ensure_ascii=False, indent=2),
        back_url=url_for("patients")
)


@app.route("/export/all", methods=["GET"])
def export_all_patients():
    username = (session.get("username") or "").strip()
    if not username:
        return jsonify({"ok": False, "error": "Not logged"}), 401

    _ensure_export_dir()

    # Paramètre optionnel: ?zip=1 pour créer un ZIP des JSON individuels
    make_zip = request.args.get("zip", "0") in ("1", "true", "yes")

    with get_db_conn() as conn:
        # Source de la liste : votre page /patients s'appuie sur get_all_patients(conn)
        rows = get_all_patients(conn)

        # rows contient (id_patiente, status_texte) selon votre fonction
        patient_ids = []
        for r in rows:
            try:
                pid = r["id_patiente"]
            except Exception:
                pid = r[0]
            pid = (pid or "").strip()
            if pid:
                patient_ids.append(pid)

        # 1) JSONL global
        jsonl_path = EXPORT_DIR / "endopath_after.jsonl"
        with jsonl_path.open("w", encoding="utf-8") as f:
            for pid in patient_ids:
                payload = _export_patient_json(conn, pid)
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")

        # 2) JSON individuels + ZIP (si demandé)
        zip_path = None
        if make_zip:
            zip_path = EXPORT_DIR / "endopath_after_patientes.zip"
            with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                for pid in patient_ids:
                    payload = _export_patient_json(conn, pid)
                    tmp_json = json.dumps(payload, ensure_ascii=False, indent=2)
                    # Chemin dans le zip
                    zf.writestr(f"export_patientes/{pid}.json", tmp_json)

    result = {
        "ok": True,
        "count": len(patient_ids),
        "jsonl": str(jsonl_path),
        "zip": str(zip_path) if zip_path else None,
        "dir": str(EXPORT_DIR),
        }

    return render_template_string(
        HTML_EXPORT_RESULT,
        payload=json.dumps(result, ensure_ascii=False, indent=2),
        back_url=url_for("patients")  # nom de la route liste
    )

@app.route("/patient/<num_inclusion>/preview", methods=["POST"])
def patient_preview(num_inclusion: str):
    data = request.get_json(force=True) or {}

    selected_keys = set(data.get("selected_keys") or [])
    abbrev_choices = data.get("abbrev_choices") or {}
    phrase_mode = bool(data.get("phrase_mode"))
    enabled_families = data.get("enabled_families") or []

    with get_db_conn() as conn:
        original_text, corrected_text_db = get_before_after_texts(conn, num_inclusion)

    all_sugs = build_patient_suggestions(
        text=original_text,
        dict_single=DICT_SINGLE,
        dict_phrase=DICT_PHRASE,
        enabled_families=enabled_families,
    )
    sug_by_key = {s.key(): s for s in all_sugs}

    selected = [s for s in all_sugs if s.key() in selected_keys]
    selected_auto = [s for s in selected if s.is_auto]
    selected_choice = [s for s in selected if not s.is_auto]

    # 1) base "engine"
    engine_after = _apply_replacements(original_text, selected_auto)

    # 2) base affichée: DB si existe, sinon engine
    if selected_keys:
        corrected_text = engine_after
    else:
        corrected_text = corrected_text_db if (corrected_text_db or "").strip() else engine_after

    # 3) ABBREV_AMBIGU: construire map des choix ACTIFS
    abbrev_apply_map = {}
    for skey, expansion in (abbrev_choices or {}).items():
        if skey not in selected_keys:
            continue
        s = sug_by_key.get(skey)
        if not s or s.family != "ABBREV_AMBIGU":
            continue
        token_cs = (s.token_source or "").strip()
        exp = (expansion or "").strip()
        if token_cs and exp:
            abbrev_apply_map[token_cs] = exp

    # 4) IMPORTANT: enlever les anciennes annotations pour toutes les ABBREV_AMBIGU détectées
    #    (ça gère le cas "on décoche SP et l'ancien SP [..] reste")
    amb_tokens = []
    for s in all_sugs:
        if s.family == "ABBREV_AMBIGU":
            tok = (s.token_source or "").strip()
            if tok:
                amb_tokens.append(tok)

    if amb_tokens:
        corrected_text = _strip_abbrev_annotations(corrected_text, amb_tokens)

    # 5) puis appliquer les choix courants
    if abbrev_apply_map:
        corrected_text = _apply_abbrev_annotations(corrected_text, abbrev_apply_map)

    # ------------------------------------------------------------------
    # Mode phrase par phrase
    # ------------------------------------------------------------------
    if phrase_mode:
        original_sentences = split_sentences(original_text)
        corrected_sentences = split_sentences(corrected_text)

        mask = data.get("sentence_color_mask")
        if not isinstance(mask, list):
            mask = [True] * len(original_sentences)

        if len(mask) < len(original_sentences):
            mask = mask + [True] * (len(original_sentences) - len(mask))
        if len(mask) > len(original_sentences):
            mask = mask[: len(original_sentences)]

        before_html = []
        after_html = []

        for i, s in enumerate(original_sentences):
            if mask[i]:
                before_html.append(_highlight_text_before(s, selected_auto + selected_choice))
            else:
                before_html.append(_escape_html(s))

        for i, s in enumerate(corrected_sentences):
            if i < len(mask) and mask[i]:
                # IMPORTANT: passer aussi abbrev_apply_map au highlight_after
                after_html.append(_highlight_text_after(s, selected_auto, abbrev_apply_map))
            else:
                after_html.append(_escape_html(s))

        return jsonify({
            "mode": "sentences",
            "before_sentences_html": before_html,
            "after_sentences_html": after_html,
        })

    # ------------------------------------------------------------------
    # Mode normal
    # ------------------------------------------------------------------
    highlighted_before = _highlight_text_before(original_text, selected_auto + selected_choice)
    highlighted_after = _highlight_text_after(corrected_text, selected_auto, abbrev_apply_map)

    return jsonify({
        "mode": "full",
        "highlighted_before": highlighted_before,
        "highlighted_after": highlighted_after,
    })


def _escape_html(s: str) -> str:
    # petit helper si tu n'en as pas déjà un
    import html
    return html.escape(s or "")


@app.route("/")
def index():
    return redirect(url_for("login"))

@app.before_request
def require_login():
    # Laisser passer login + static
    if request.endpoint in PUBLIC_ENDPOINTS:
        return

    # Si pas loggé → login
    if not session.get("username"):
        return redirect(url_for("login"))

if __name__ == "__main__":
    app.run(debug=True)
