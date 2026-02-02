## Étape 3 — Filtrage linguistique (SpaCy)

### Fichier

`filter_tokens_with_spacy.py`

### Objectif

À partir d’un vocabulaire brut (issu de l’extraction dossier gynéco), **classifier les tokens** en :

* **valides** (à conserver),
* **invalides** (bruit / artefacts / tokens non pertinents),
* **à corriger** (candidats pour corrections / enrichissement dictionnaires).

Cette étape sert à **réduire le bruit** et à créer un “backlog” propre pour l’étape de suggestions (Étape E).

### Entrées (inputs)

Sources attendues dans ton arborescence :

* `Data/DATA_PROCESSED/vocab_dossier_gyneco_from_xlsx.csv`
  (produit par `extract_text_and_vocab_from_dossier_gyneco.py`)

Selon ta version, il peut aussi utiliser :

* `Data/DATA_PROCESSED/all_words.csv` (si tu passes par une étape de build tokens/vocab intermédiaire)

### Sorties (outputs)

Dans `Data/DATA_PROCESSED/` :

* `tokens_valides.csv`
* `tokens_invalides.csv`
* `tokens_a_corriger.csv`

👉 Ces 3 fichiers sont des **artefacts de pipeline**, généralement **non versionnés**.

### Modules Python utilisés (et pourquoi)

Typiquement (à confirmer sur ton code exact si tu ré-uploade le fichier) :

* `pandas` : lecture/écriture CSV, tri, regroupements, colonnes de score/flags.
* `spacy` : analyse linguistique FR (lemmatisation, POS, stopwords, etc.).
* `wordfreq` (ZIPF) : repérer les tokens très rares → suspects.
* `re` : règles de filtrage regex (ponctuation, tokens mixtes, patterns parasites).
* `pathlib.Path` : chemins robustes.
* éventuellement `unicodedata` : normalisation (accents / caractères spéciaux).

### Déroulé interne (logique “pipeline”)

1. **Chargement du vocab brut**
   Le script lit un CSV contenant au minimum un champ “token” (souvent `token_source` ou `token`) et potentiellement des infos d’occurrence/fréquence.

2. **Normalisation**
   Exemples typiques :

   * trim, lowercase (selon stratégie),
   * suppression de tokens vides,
   * homogénéisation apostrophes / tirets,
   * filtrage des tokens trop courts / trop longs.

3. **Analyse SpaCy**
   Passage des tokens dans `fr_core_news_md` (ou similaire) pour :

   * détecter ponctuation pure,
   * stopwords,
   * tokens non alpha (ou alpha+mix),
   * POS / shape (utile pour heuristiques).

4. **Score de rareté (ZIPF)**
   Avec `wordfreq.zipf_frequency(token, "fr")` :

   * tokens très rares → plus de chances d’être erreurs de saisie, OCR-like, concaténations, etc.
   * ces tokens sont orientés vers `tokens_a_corriger.csv`.

5. **Classification**
   Règles usuelles :

   * **valides** : alpha, longueur raisonnable, pas stopword, pas trop rare, pas pattern parasite.
   * **invalides** : ponctuation, suites de symboles, tokens de “formatage”, etc.
   * **à corriger** : tokens plausibles mais rares / suspects / variants sans accents / abréviations non reconnues.

6. **Écriture des 3 CSV**
   Les fichiers servent ensuite à :

   * alimenter les dictionnaires / suggestions,
   * éventuellement vérifier l’impact après enrichissement dico (re-run pipeline).

### Comment exécuter

Depuis `exploratory/preprocessing/` :

```bash
python filter_tokens_with_spacy.py
```

### Prérequis

* SpaCy FR :

```bash
python -m spacy download fr_core_news_md
```

### Contrôles rapides

* `tokens_a_corriger.csv` doit être non vide (sauf si le corpus est déjà très propre).
* Vérifier quelques lignes : tokens “bizarres”, sans accents, abréviations, etc.

---

