← [Préprocessing — Sommaire](preprocessing-sommaire.md)

---

# Préprocessing — Historique

...

````markdown
# Historique — utilisation de `dictionnaire_correction.json` (ancienne approche “Nicolai”)

## Objectif de cette page
Comprendre **comment le script historique** applique les corrections à partir du fichier
`dictionnaire_correction.json`, et comment **reproduire** le résultat sur un exemple gynéco.

Cette page sert de **référence “avant l’UI”** pour justifier ensuite les évolutions 
apportées par l’application Flask (UI interactive, familles, surlignage Avant/Après, 
abréviations ambiguës, mode phrase, etc.).

---

## 1) Fichiers concernés
- `preprocess_NLP.py`  ✅ **contient la logique d’application du dictionnaire**
- `preprocess_dossier_gyneco.py`  (pipeline historique “dossier gynéco”, pas forcément le même que l’UI Flask)
- `dictionnaire_correction.json`  ✅ **source of truth des corrections “mot -> mot”**

---

## 2) Où Nicolai utilise `dictionnaire_correction.json` (localiser le code)
Dans `preprocess_NLP.py`, on trouve :

### 2.1 Chargement du dictionnaire
Fonction :
- `load_dict_correction()`  
  - ouvre un JSON et fait `json.load(...)`
  - ⚠️ **dans la version historique**, le chemin est un **chemin absolu Linux** (ex: `/home/ubuntu/.../dictionnaire_correction.json`)  
    → en environnement Windows / autre repo, il faut **corriger le chemin**.

### 2.2 Application du dictionnaire
Fonctions :
- `replace_word(sentence: str, target_word: str, replacement_word: str)`
  - applique un `re.sub()` avec motif : `r"\b{target_word}\b"`
  - remplace **uniquement si match exact au niveau “mot”** (frontières `\b`)
- `appliquer_correction(liste_token: list, dict_correction: dict)`
  - itère sur les tokens et remplace si `token in dict_correction`
- `correction_series(serie: pd.Series)`
  - convertit le texte en tokens
  - applique les corrections mot à mot
  - reconstruit la chaîne avec des espaces

---

## 3) Principe de fonctionnement (comment ça marche)
### 3.1 Entrée
Une `Series` pandas contenant une colonne texte (ex: Résumé / Consultation / etc.)

### 3.2 Tokenisation “historique”
Le texte est “splitté” par :
```python
re.split('\n|[ -,:.\'/;()_]', text)
````

Conséquences :

* la ponctuation est utilisée comme séparateur
* on obtient une liste de tokens parfois vides (`''`)
* les apostrophes et certains caractères sont “cassants” (ex: `l'examen` devient `l` + `examen`)

### 3.3 Correction par dictionnaire

Pour chaque token `t` :

* si `t` est une clé du dictionnaire, on remplace par `dict[t]`
* sinon, on garde `t`

### 3.4 Reconstruction

On reconstruit une phrase par concaténation avec espaces (`contract_string()`).

Conséquences :

* la phrase finale peut **perdre une partie de la ponctuation**
* les espaces peuvent être “bizarres” (multiples / avant ponctuation)

---

## 4) Reproduire l’exécution (checklist “reproduire le code”)

### 4.1 Pré-requis

* Python + pandas
* le fichier `dictionnaire_correction.json` accessible localement

### 4.2 Correctif indispensable (chemin du JSON)

Dans `preprocess_NLP.py`, remplacer le chemin absolu par un chemin relatif au repo.

Exemple robuste :

```python
from pathlib import Path
import json

BASE_DIR = Path(__file__).resolve().parent
DICT_PATH = BASE_DIR / "Data" / "DATA_PROCESSED" / "Correction_mots" / "dictionnaire_correction.json"

def load_dict_correction(path: Path = DICT_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
```

### 4.3 Exemple minimal (sur un texte gynéco)

```python
import pandas as pd
from preprocess_NLP import correction_series

s = pd.Series([
    "L'examen retrouve un kyste ovarien. resection prevue. endometriose superf."
])

out = correction_series(s)
print(out.iloc[0])
```

✅ Si `dictionnaire_correction.json` contient par ex :

* `"resection": "résection"`
* `"endometriose": "endométriose"`
  alors ces tokens seront remplacés.

---

## 5) Générer un exemple concret sur données gynéco (checklist “exemple concret”)

### 5.1 À partir d’un CSV “texte patient”

Si tu as un export avec une colonne texte (ex: `Consultation` ou `diag_gyneco`) :

```python
import pandas as pd
from preprocess_NLP import correction_series

df = pd.read_csv("Data/DATA_PROCESSED/dossier_gyneco_texte_par_patiente.csv")
# Exemple : adapter le nom de colonne au fichier réel
df["texte_corrige"] = correction_series(df["texte"])
df[["id_patiente", "texte", "texte_corrige"]].head(3)
```

Résultat : tu obtiens un “avant/après” **offline**, sans UI.

---

## 6) Limites de l’approche historique (important pour “justifier les évolutions”)

1. **Perte / dégradation de la ponctuation**

   * la tokenisation par split détruit l’information de structure
2. **Gestion faible des cas “multi-mots”**

   * le dictionnaire est “token -> token” (pas phrase -> phrase)
3. **Sensibilité au format du texte**

   * si le token contient apostrophe, accents, etc., ça peut empêcher la correspondance
4. **Chemins non portables**

   * chemin absolu Linux dans `load_dict_correction`
5. **Pas de logique UI / sélection**

   * pas de notion de “familles”, pas de choix utilisateur, pas de preview dynamique

---

## 7) Réponses directes à la checklist Planner

### ✅ Localiser le code et l’usage du dictionnaire

* `preprocess_NLP.py` : `load_dict_correction()` + `correction_series()` + `appliquer_correction()`

### ✅ Comprendre le principe de fonctionnement

* split texte -> tokens -> remplacement si clé dans JSON -> reconstruction

### ✅ Reproduire l’exécution du code

* corriger le chemin du JSON en relatif
* appeler `correction_series(pd.Series([...]))`

### ✅ Générer un exemple concret sur données gynéco

* appliquer `correction_series()` à une colonne texte issue d’un CSV patient

### ✅ Paragraphe de synthèse (réunion)

> Avant l’UI Flask, les corrections reposaient sur une approche “batch” : un dictionnaire JSON de remplacements simples (token->token) appliqué via une tokenisation rudimentaire (split sur ponctuation). Cette approche permettait de corriger rapidement des fautes fréquentes mais montrait des limites (ponctuation, multi-mots, cas ambigus, absence de sélection UI). L’application Flask a ensuite été développée pour rendre le processus interactif, traçable et piloté par l’utilisateur (sélections, preview Avant/Après, abréviations ambiguës, mode phrase, reset).

---
