← [Accueil du Wiki](home.md) / [Préprocessing — Sommaire](preprocessing-sommaire.md)

---

# Installation — ENDOPATH

Cette page décrit **l’installation complète de l’environnement ENDOPATH**,  
depuis une machine vierge jusqu’à un pipeline prêt à être exécuté.

---

## Objectif

Obtenir un environnement fonctionnel permettant de :

- exécuter le pipeline de préprocessing ENDOPATH (`run_pipeline.py`)
- générer et exploiter une base de données SQLite
- lancer l’interface web Flask (`app.py`)

---

## 1) Pré-requis système

### 1.1 Système d’exploitation

- Windows 10 / 11
- Linux (Ubuntu 20.04+)
- macOS (non testé mais compatible en principe)

---

## 2) Installation de Python

### 2.1 Version recommandée

- **Python ≥ 3.10**
- Validé : 3.10, 3.11, 3.12
- ⚠️ Éviter Python 3.7 / 3.8 (scripts anciens, conflits de dépendances)

### 2.2 Vérification

Vérifier la version installée avec la commande :  
`python --version`

---

## 3) Base de données SQLite

### 3.1 Principe

ENDOPATH repose sur une **base de données SQLite locale** qui contient :

- les données médicales prétraitées
- les diagnostics et annotations
- les liens entre textes, tokens et suggestions

La base est :

- créée automatiquement par le pipeline
- stockée sous forme d’un fichier unique
- utilisée en lecture par l’interface web Flask

---

### 3.2 Prérequis SQLite

Aucune installation spécifique de serveur n’est requise.

- SQLite est **embarqué nativement avec Python**
- le module `sqlite3` fait partie de la bibliothèque standard Python

👉 **Aucune dépendance Python supplémentaire n’est nécessaire** pour SQLite.

---

### 3.3 Fichier de base de données

À l’issue de l’exécution du pipeline, le fichier suivant est créé ou mis à jour :

- `endopath_diag.db`

Ce fichier est :

- généré dans le répertoire de preprocessing
- indispensable au fonctionnement de l’interface Flask
- à conserver entre les exécutions de l’UI

⚠️ Supprimer ce fichier implique de **relancer le pipeline complet**.

---

## 4) Dépendances Python

### 4.1 Dépendances principales

Installer les librairies nécessaires :

- `pip install pandas numpy openpyxl flask tqdm`
- `pip install msoffcrypto-tool`
- `pip install spacy wordfreq rapidfuzz unidecode`

---

### 4.2 Modèle SpaCy (français)

Obligatoire pour l’étape de filtrage linguistique :

- `python -m spacy download fr_core_news_md`

---

## 5) Arborescence attendue

Avant exécution, vérifier la présence de l’arborescence suivante :

- exploratory/preprocessing/
  - Data/
    - DATA_RAW/
      - fichiers `*.xlsx`
      - fichiers `PASSWORD_*.txt`
    - DATA_PROCESSED/
  - endopath_diag.db (après exécution du pipeline)
  - run_pipeline.py
  - app.py

---

## 6) Données sources (obligatoire)

### 6.1 Fichiers XLSX requis (exemples)

Dans le dossier `Data/DATA_RAW/` :

- `INCLUSION RECHERCHE CLINIQUE.xlsx`
- `Recueil_MMJ.xlsx`
- `dossier-gyneco-23-03-2022_converti.xlsx`
- `2022 - Donnees PMSI - Protocole ENDOPATHS - GHN..ALTRAN_converti.xlsx`

---

### 6.2 Fichiers de mots de passe

Si certains fichiers XLSX sont chiffrés :

- `PASSWORD_1.txt`
- `PASSWORD_2.txt`

---

## 7) Lancement de l’installation

Lancer le pipeline de préprocessing :  
`python run_pipeline.py`

Cette étape :

- crée la base SQLite
- génère les CSV intermédiaires
- prépare les données pour l’interface UI

---

## 8) Vérification de l’installation

Vérifier que le pipeline est opérationnel :  
`python run_pipeline.py --help`

Si l’aide s’affiche et que le fichier `endopath_diag.db` est présent,  
alors **l’environnement est correctement installé** ✅
