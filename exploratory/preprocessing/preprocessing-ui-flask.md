← [Retour au sommaire Préprocessing](preprocessing-sommaire.md)

---

# README — UI Flask (ENDOPATH)

## 1. Objectif de l’UI

L’UI ENDOPATH est une **application web de relecture et de correction assistée de textes médicaux**.

Elle permet à un utilisateur métier (médecin, data-manager, expert clinique) de :

* visualiser le **texte clinique brut** (“Avant”),
* visualiser le **texte corrigé** (“Après”),
* activer / désactiver des **suggestions de correction**,
* gérer des **abréviations ambiguës** avec choix explicite,
* travailler en **mode global** ou **phrase par phrase**,
* contrôler précisément **ce qui est modifié et pourquoi**.

👉 L’UI **ne contient aucune logique métier lourde** :
elle délègue le traitement au backend Python et se concentre sur l’ergonomie et la traçabilité.

---

## 2. Architecture générale

```
Navigateur (HTML / CSS / JS)
        |
        |  AJAX / JSON
        v
Flask (app.py)
        |
        |  Python (NLP / règles métier)
        v
CSV / DB SQLite
```

### Principe clé

* **Client léger** : affichage + interactions utilisateur
* **Serveur riche** : analyse linguistique, règles métier, décisions
* **Échanges JSON** : état UI → calcul serveur → HTML prêt à afficher

---

## 3. Pages principales

### 3.1 Page de login

**Fichier** : `templates/login.html`

* Accès simple à l’application
* Pas de logique métier (peut être remplacée par SSO ultérieurement)

---

### 3.2 Liste des patientes

**Fichier** : `templates/patients.html`
**Route** : `/patients`

Fonctionnalités :

* affichage de la liste des patientes disponibles,
* navigation vers la fiche détaillée d’une patiente,
* séparation claire entre **navigation** et **analyse**.

---

### 3.3 Fiche patiente (cœur de l’UI)

**Fichier** : `templates/patient.html`
**Route** : `/patient/<num_inclusion>`

Contenu :

* Texte **Avant** (original)
* Texte **Après** (corrigé)
* Panneau de suggestions, organisées par **familles**
* Contrôles UI :

  * coche/décoche des suggestions,
  * sélection d’expansion pour abréviations ambiguës,
  * reset,
  * mode phrase par phrase.

---

## 4. Interaction UI ↔ Backend

### 4.1 Chargement initial

Au chargement de la page patiente :

* le backend fournit :

  * le texte brut,
  * l’état initial des suggestions,
  * les dictionnaires applicables,
  * le texte corrigé initial (si existant).

---

### 4.2 Preview dynamique (temps réel)

Lors de chaque action utilisateur (checkbox, choix d’abréviation, mode phrase) :

* l’UI envoie un **JSON de contexte** au backend :

  ```json
  {
    "selected_keys": [...],
    "abbrev_choices": {...},
    "phrase_mode": true,
    "enabled_families": [...]
  }
  ```

* le backend :

  * applique les règles linguistiques,
  * effectue les remplacements,
  * génère le HTML final (Avant / Après),
  * renvoie un rendu prêt à afficher.

👉 L’UI **n’interprète pas le texte**, elle l’affiche.

---

## 5. Convention d’affichage (lisibilité médicale)

### Avant (texte source)

* Mots concernés par des suggestions :

  * surlignés en **rouge** (auto),
  * ou en **couleur distincte** (choix utilisateur).

### Après (texte corrigé)

* Corrections effectivement appliquées :

  * surlignées en **vert**,
* Abréviations ambiguës annotées :

  * format : `TV [toucher vaginal]`,
  * surlignage dédié.

👉 Principe fondamental :
**ce qui est coloré correspond exactement à ce qui est modifié**.

---

## 6. Cas complexes gérés

L’UI permet de gérer correctement :

* abréviations ambiguës (choix explicite requis),
* coexistence de corrections automatiques et manuelles,
* changements successifs (cocher / décocher),
* reset propre (sans résidus d’annotations),
* travail phrase par phrase (masquage ciblé).

---

## 7. Technologies utilisées

* **Backend** : Python, Flask
* **Frontend** : HTML, CSS, JavaScript
* **Transport** : AJAX / JSON
* **Données** :

  * CSV générés par le pipeline NLP,
  * SQLite (source clinique structurée).

---

## 8. Philosophie de conception

* Séparation stricte UI / métier
* Aucune logique linguistique côté navigateur
* Comportement déterministe et reproductible
* Adapté à un contexte médical (traçabilité, relecture humaine)

---

## 9. Démarrage rapide

> ⚠️ **ATTENTION — exécution du pipeline (action sensible et irréversible)**  
>  
> Le script `run_pipeline.py` sert à **générer / reconstruire** les données utilisées par l’application.  
>  
> - **À exécuter uniquement lors de la première utilisation** ou lors d’une **reconstruction volontaire**.  
> - Si une base de données existe déjà, relancer `run_pipeline.py` **écrase l’état courant** et  
>   **entraîne la perte des corrections de texte déjà appliquées**.  
> - Cette action est **irréversible** en l’absence de sauvegarde préalable.  
>  
> ❗ **Anti-boulette** : ne lance jamais `run_pipeline.py` “pour tester” ou “par réflexe”.  
> Si l’UI fonctionne, **ne touche pas au pipeline**.

```bash
# CAS A — Première utilisation (initialisation UNIQUEMENT)
# Générer les données (À FAIRE UNE SEULE FOIS)
python run_pipeline.py

# Lancer l’UI
python app.py

# CAS B — Utilisation courante (base déjà initialisée)
# ⚠️ Ne PAS relancer le pipeline
python app.py

Puis ouvrir :

```
http://127.0.0.1:5000
```

