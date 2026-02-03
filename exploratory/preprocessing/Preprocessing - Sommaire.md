# Préprocessing — Sommaire

## Pages associées
- [[Preprocessing - Extraction de texte]]
- [[Preprocessing - Filtrage linguistique]]
- [[Preprocessing - Suggestions]]
- [[Preprocessing - UI Flask]]


````markdown
# ENDOPATH — Pipeline de préparation de données & UI Flask

Ce dépôt contient :

- un **pipeline de préparation et d’enrichissement de données cliniques**
  (XLSX → SQLite → CSV → suggestions linguistiques),
- une **UI web Flask** (HTML / CSS / JavaScript / AJAX) permettant
  la visualisation, la correction et la normalisation de textes médicaux
  patiente par patiente.

---

## 1) Arborescence principale

Racine : `exploratory/preprocessing/`

### Scripts principaux
- `create_endopath_diag_db.py`
- `extract_text_and_vocab_from_dossier_gyneco.py`
- `build_tokens_pipeline.py`
- `filter_tokens_with_spacy.py`
- `suggest_dict_extensions.py`
- `run_pipeline.py`
- `app.py` (UI Flask)

### Données
- `Data/DATA_RAW/` : sources Excel + fichiers de mots de passe
- `Data/DATA_PROCESSED/` : exports CSV et artefacts générés
- `Data/DATA_PROCESSED/Correction_mots/` :
  dictionnaires métier + suggestions

### Base de données
- `endopath_diag.db` (SQLite)

---

## 2) Pré-requis

### 2.1 Python
Python **3.10+ recommandé**  
(Compatible 3.12 / 3.13 selon versions de dépendances)

### 2.2 Dépendances principales (indicatif)
- pandas
- numpy
- openpyxl
- msoffcrypto-tool
- flask
- spacy
- wordfreq
- rapidfuzz / python-Levenshtein
- unidecode
- tqdm

### 2.3 Modèle SpaCy (français)
```bash
python -m spacy download fr_core_news_md
````

---

## 3) Vue d’ensemble du pipeline

```text
Data/DATA_RAW/*.xlsx (+ PASSWORD_*.txt)
      |
      v
create_endopath_diag_db.py
  -> endopath_diag.db
      |
      v
extract_text_and_vocab_from_dossier_gyneco.py
  -> Data/DATA_PROCESSED/dossier_gyneco_texte_par_patiente.csv
  -> Data/DATA_PROCESSED/vocab_dossier_gyneco_from_xlsx.csv
      |
      v
build_tokens_pipeline.py (optionnel selon version)
  -> Data/DATA_PROCESSED/all_words.csv
      |
      v
filter_tokens_with_spacy.py
  -> tokens_valides.csv
  -> tokens_invalides.csv
  -> tokens_a_corriger.csv
      |
      v
suggest_dict_extensions.py
  -> Data/DATA_PROCESSED/Correction_mots/suggestions_*.csv
      |
      v
app.py (UI Flask)
```

---

## 4) Documentation

* Création de la base de données (XLSX → SQLite)
* Pipeline NLP (tokens, filtrage, suggestions)
* Dictionnaires et gouvernance métier
* Guide utilisateur de l’interface Flask

---

## 5) Exécution rapide (recommandée)

### 5.1 Lancer le pipeline complet

```bash
python run_pipeline.py
```

Ce script :

* vérifie les fichiers requis dans `Data/DATA_RAW/`
* génère la base `endopath_diag.db`
* produit les CSV dans `Data/DATA_PROCESSED/`
* génère les suggestions linguistiques

### 5.2 Lancer l’UI Flask

```bash
python app.py
```

Ouvrir ensuite :

```
http://127.0.0.1:5000
```

---

## 6) Dictionnaires (gestion Git)

* `Data/DATA_PROCESSED/Correction_mots/dictionnaire_correction.json`
* `Data/DATA_PROCESSED/Correction_mots/abbrev_sure_merged.json`
* `Data/DATA_PROCESSED/Correction_mots/abbrev_ambigue_merged.json`

### À ignorer (générés)

* `Data/DATA_PROCESSED/*.csv`
* `Data/DATA_PROCESSED/Correction_mots/suggestions_*.csv`

---

## 7) Dépannage

### 7.1 Fichier manquant

Vérifier la présence des fichiers XLSX et PASSWORD dans `Data/DATA_RAW/`.

### 7.2 Modèle SpaCy manquant

```bash
python -m spacy download fr_core_news_md
```

### 7.3 Pas de suggestions dans l’UI

Vérifier que `suggest_dict_extensions.py` a bien généré les fichiers :

```
Data/DATA_PROCESSED/Correction_mots/suggestions_*.csv
```

````

