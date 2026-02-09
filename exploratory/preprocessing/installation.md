← [Accueil du Wiki](home.md) / [Préprocessing — Sommaire](preprocessing-sommaire.md)

---

# Installation — ENDOPATH

Cette page décrit **l’installation complète de l’environnement ENDOPATH**,  
depuis une machine vierge jusqu’à un pipeline prêt à être exécuté.

---

## Objectif

Obtenir un environnement fonctionnel permettant de :

- exécuter le pipeline de préprocessing ENDOPATH (`run_pipeline.py`)
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

## 3) Dépendances Python

### 3.1 Dépendances principales

Installer les librairies nécessaires :

- `pip install pandas numpy openpyxl flask tqdm`
- `pip install msoffcrypto-tool`
- `pip install spacy wordfreq rapidfuzz unidecode`

---

### 3.2 Modèle SpaCy (français)

Obligatoire pour l’étape de filtrage linguistique :

- `python -m spacy download fr_core_news_md`

---

## 4) Arborescence attendue

Avant exécution, vérifier la présence de l’arborescence suivante :

- exploratory/preprocessing/
  - Data/
    - DATA_RAW/
      - fichiers `*.xlsx`
      - fichiers `PASSWORD_*.txt`
    - DATA_PROCESSED/
  - run_pipeline.py
  - app.py

---

## 5) Données sources (obligatoire)

### 5.1 Fichiers XLSX requis (exemples)

Dans le dossier `Data/DATA_RAW/` :

- `INCLUSION RECHERCHE CLINIQUE.xlsx`
- `Recueil_MMJ.xlsx`
- `dossier-gyneco-23-03-2022_converti.xlsx`
- `2022 - Donnees PMSI - Protocole ENDOPATHS - GHN..ALTRAN_converti.xlsx`

---

### 5.2 Fichiers de mots de passe

Si certains fichiers XLSX sont chiffrés :

- `PASSWORD_1.txt`
- `PASSWORD_2.txt`

---

## 6) Lancement de l’installation

Lancer le pipeline avec la commande :  
`python run_pipeline.py`

---

## 7) Vérification de l’installation

Afficher l’aide du pipeline :  
`python run_pipeline.py --help`

Si l’aide s’affiche, alors **l’environnement est correctement installé** ✅
