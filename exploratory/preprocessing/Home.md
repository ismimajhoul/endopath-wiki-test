# ENDOPATH — Analyse et correction de textes médicaux

## Présentation
ENDOPATH est une application d’analyse, de normalisation et de correction
de comptes rendus médicaux (gynécologie), combinant :

- un pipeline de traitement de données (Python),
- des dictionnaires métier médicaux,
- une interface web de relecture et correction assistée.

L’objectif est d’améliorer la qualité, la cohérence et la lisibilité
des textes cliniques tout en conservant la maîtrise humaine.

---

## Architecture (vue synthétique)

- **Pipeline Python** : ingestion XLSX → base SQLite → CSV → suggestions
- **Moteur métier** : règles linguistiques, dictionnaires, abréviations
- **UI Web (Flask)** : correction interactive Avant / Après

---

## Démarrage rapide

### Prérequis
- Python 3.10+
- pip
- Modèle spaCy français

```bash
pip install -r requirements.txt
python -m spacy download fr_core_news_md

## Exécution complète 
python run_pipeline.py
python app.py


Accès UI :
http://127.0.0.1:5000