← [Accueil du Wiki](home.md) / [Préprocessing — Sommaire](preprocessing-sommaire.md)

---

# Lancement — ENDOPATH

Cette page décrit **les deux phases distinctes d’exécution du projet ENDOPATH** :

1. **Exécution du pipeline de préprocessing** (phase lourde, batch, ponctuelle)
2. **Lancement de l’interface web Flask** (phase interactive, répétable)

⚠️ Ces deux phases ont des objectifs et des contraintes **fondamentalement différentes**.

---

## 1) Phase 1 — Pipeline de préprocessing (batch)

### 1.1 Principe

Le pipeline :

- lit les fichiers Excel sources
- construit la base SQLite
- génère les CSV intermédiaires
- calcule les suggestions de correction

👉 **Cette phase est dite “fatale / one-shot”** :

- elle est coûteuse
- elle modifie l’état des données
- elle **ne se relance pas à chaque usage**

---

### 1.2 Lancer le pipeline

Depuis le dossier `exploratory/preprocessing/`, exécuter la commande :

`python run_pipeline.py`

---

### 1.3 Résultat attendu

À l’issue de l’exécution :

- la base SQLite `endopath_diag.db` est créée ou mise à jour
- les fichiers CSV sont générés dans `Data/DATA_PROCESSED/`
- les suggestions sont générées dans `Data/DATA_PROCESSED/Correction_mots/`

✅ Le pipeline est alors prêt pour l’exploitation par l’interface UI.

---

### 1.4 Quand relancer le pipeline ?

Relancer uniquement si :

- les fichiers XLSX sources ont changé
- les dictionnaires métier ont été massivement modifiés
- une évolution structurelle du pipeline est introduite

❌ Ne pas relancer pour un simple usage de l’UI.

---

## 2) Phase 2 — Lancement de l’interface web Flask

### 2.1 Principe

L’interface Flask :

- lit les données déjà préparées (base SQLite et CSV)
- permet la sélection des corrections
- affiche un rendu dynamique Avant / Après
- n’écrit pas de nouvelles données lourdes

👉 Cette phase est :

- légère
- répétable
- destinée à l’usage quotidien

---

### 2.2 Lancer l’interface

Depuis le dossier `exploratory/preprocessing/`, exécuter :

`python app.py`

---

### 2.3 Accès à l’interface

Ouvrir un navigateur et accéder à l’adresse :

`http://127.0.0.1:5000`

(Si HTTPS est configuré ultérieurement, l’URL sera adaptée.)

---

## 3) Récapitulatif des phases

- **Phase 1 — Préprocessing**
  - Script : `python run_pipeline.py`
  - Fréquence : rare
  - Rôle : préparer les données

- **Phase 2 — UI interactive**
  - Script : `python app.py`
  - Fréquence : quotidienne
  - Rôle : explorer et corriger les données

---

## 4) Bonnes pratiques

✔️ Installer l’environnement une seule fois  
✔️ Lancer le pipeline une seule fois  
✔️ Utiliser l’interface UI autant que nécessaire  

❌ Ne pas relancer le pipeline sans raison
