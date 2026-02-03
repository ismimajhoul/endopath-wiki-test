← [Retour au sommaire Préprocessing](preprocessing-sommaire.md)

---

## Étape 4 — Suggestions & dictionnaires

### Fichier

`suggest_dict_extensions.py`

### Objectif

À partir de `tokens_a_corriger.csv` + dictionnaires existants, **produire des suggestions** regroupées par “familles” :

* **diacritiques** (mots sans accents → mot accentué),
* **typos** (corrections orthographiques proches),
* **abréviations sûres** (ABBREV_SURE),
* **abréviations ambiguës** (ABBREV_AMBIGU avec expansions),
* **enrichissement domaine**,
* **rejets auto** (ce que l’algo refuse).

Ces suggestions sont consommées par l’UI Flask (`app.py`) pour proposer des corrections à l’utilisateur.

### Entrées (inputs)

Dans ton repo, ça colle à ce que tu as listé :

1. Tokens candidats

* `Data/DATA_PROCESSED/tokens_a_corriger.csv`
* (souvent aussi) `Data/DATA_PROCESSED/tokens_valides.csv` (pour contexte / blacklist / éviter faux positifs)

2. Dictionnaires “source of truth”

* `Data/DATA_PROCESSED/Correction_mots/dictionnaire_correction.json`
* `Data/DATA_PROCESSED/Correction_mots/abbrev_sure_merged.json`
* `Data/DATA_PROCESSED/Correction_mots/abbrev_ambigue_merged.json`

3. (optionnel selon version)

* `Data/DATA_PROCESSED/Correction_mots/all_words.txt` (lexique global)
* blacklists / stop tokens (tu as des outils dans `tools/`)

### Sorties (outputs)

Dans `Data/DATA_PROCESSED/Correction_mots/` :

* `suggestions_manual_dict.csv`
  (souvent : suggestions directement issues du dictionnaire existant ou “prioritaires”)
* `suggestions_auto_diacritics_strict.csv`
* `suggestions_auto_diacritics_multi.csv`
* `suggestions_auto_diacritics.csv` (union/agrégat selon implémentation)
* `suggestions_auto_typos.csv`
* `suggestions_auto_abbrev.csv` (ABBREV_SURE)
* `suggestions_auto_abbrev_ambigu.csv` (ABBREV_AMBIGU + expansions)
* `suggestions_auto_abbrev_candidate.csv` (si ton algo en génère)
* `suggestions_domain_enrich.csv`
* `suggestions_auto_rejected.csv`

👉 Même logique : ce sont des **artefacts générés**, généralement **non versionnés** (sauf si tu veux “figer” un snapshot de référence).

### Modules Python utilisés (et pourquoi)

Typiquement :

* `pandas` : manip CSV, joins, scores, export.
* `json` : charger dictionnaires (corrections, abréviations).
* `re` : heuristiques sur tokens.
* `difflib` ou distance d’édition (selon code) : propositions typos.
* `pathlib.Path` : chemins.
* parfois `unicodedata` : gestion accents / normalisation.
* éventuellement `wordfreq` / fréquence : prioriser suggestions.

### Déroulé interne (logique “familles”)

1. **Chargement des inputs**

   * tokens à corriger,
   * dictionnaire de correction,
   * abréviations sûres + ambiguës.

2. **Génération des suggestions DICT / MANUAL**

   * Si un token est connu dans `dictionnaire_correction.json`, on le classe en “manuel/dict”.

3. **Suggestions “diacritiques”**

   * cas typique : `endometriose` → `endométriose`
   * la variante “strict” = match exact sans ambiguïté,
   * la variante “multi” = plusieurs candidats possibles.

4. **Suggestions “typos”**

   * propose des corrections proches (distance d’édition + score).

5. **Suggestions “abrév sure”**

   * mapping direct abréviation → expansion (ou normalisation).

6. **Suggestions “abrév ambigu”**

   * plusieurs expansions possibles, stockées dans une colonne `expansions` (souvent une string “A | B | C”).

7. **Suggestions domaine / enrichissement**

   * règles spécifiques projet (lexique médical, termes fréquents, etc.).

8. **Rejets auto**

   * conserve les tokens écartés avec raison (utile pour debug et tuning).

9. **Écriture des CSV suggestions**
   Chaque fichier correspond à une famille / catégorie, consommable ensuite par `app.py`.

### Comment exécuter

Depuis `exploratory/preprocessing/` :

```bash
python suggest_dict_extensions.py
```

### Contrôles rapides

* vérifier que les fichiers `suggestions_*.csv` sont (re)générés.
* vérifier la présence de colonnes clés (souvent) :

  * `token_source`, `match`, `category`, `score`, `edit_dist`, `expansions` (selon famille)


