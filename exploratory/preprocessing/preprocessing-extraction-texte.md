← [Accueil du Wiki](home.md) / [Préprocessing — Sommaire](preprocessing-sommaire.md)

---


---

# Étape 2 — Extraction texte + vocab “dossier gynéco”

**Script : `extract_text_and_vocab_from_dossier_gyneco.py`**

---

## Objectif

Extraire, à partir du **fichier Excel “dossier gynécologique”**, deux artefacts fondamentaux pour la suite du pipeline :

1. **Un CSV “texte par patiente”**, utilisé directement par l’UI Flask
2. **Un CSV de vocabulaire brut**, utilisé par le pipeline NLP (SpaCy, suggestions, dictionnaires)

Cette étape constitue le **pont entre les données cliniques brutes (XLSX)** et :

* le **NLP** (tokens / corrections),
* l’**affichage UI patient**.

---

## Entrées (inputs)

Le script travaille principalement à partir de `Data/DATA_RAW/`.

### Fichiers requis

* **Dossier gynécologique converti**

  ```
  Data/DATA_RAW/dossier-gyneco-23-03-2022_converti.xlsx
  ```

* **Mot de passe (si le fichier est chiffré)**

  ```
  Data/DATA_RAW/PASSWORD_1.txt
  ```

> ⚠️ Le mot de passe est lu dynamiquement ; s’il est absent et que le fichier est chiffré, le script échoue explicitement.

---

## Sorties (outputs)

Les fichiers sont générés dans `Data/DATA_PROCESSED/` :

1. **Texte clinique par patiente**

   ```
   Data/DATA_PROCESSED/dossier_gyneco_texte_par_patiente.csv
   ```

   Contenu typique :

   * `id_patiente`
   * `texte` (texte clinique concaténé / nettoyé)

2. **Vocabulaire brut extrait**

   ```
   Data/DATA_PROCESSED/vocab_dossier_gyneco_from_xlsx.csv
   ```

   Contenu typique :

   * `token`
   * `count` (fréquence d’apparition globale)

Ces deux fichiers ont **des usages distincts** :

* le premier alimente l’UI,
* le second alimente le NLP.

---

## Modules Python utilisés (et pourquoi)

* **`pathlib.Path`**
  Gestion robuste des chemins (`BASE_DIR / "Data" / ...`), portable Windows/Linux.

* **`pandas`**

  * `read_excel()` pour charger l’Excel
  * nettoyage des DataFrame
  * `to_csv()` pour les exports structurés

* **`msoffcrypto`** (si présent)

  * Déchiffrement des fichiers Excel protégés par mot de passe

* **`openpyxl`**

  * Backend Excel pour pandas
  * Accès fiable aux feuilles et cellules

* **`re` (regex)**

  * Nettoyage du texte clinique
  * Normalisation minimale (espaces, ponctuation, séparateurs)

* **`collections.Counter`**

  * Comptage des tokens pour le vocabulaire brut

---

## Déroulé interne du script (pas à pas)

### 1) Initialisation des chemins

En tête de script :

* Définition de `BASE_DIR`
* Définition de :

  * `DATA_RAW`
  * `DATA_PROCESSED`
* Définition du chemin du fichier Excel gynéco
* Définition du chemin du fichier mot de passe

---

### 2) Ouverture du fichier Excel (avec ou sans chiffrement)

Logique typique :

* Si le fichier est chiffré :

  * lecture du mot de passe depuis `PASSWORD_1.txt`
  * déchiffrement via `msoffcrypto`
* Sinon :

  * lecture directe avec `pandas.read_excel()`

👉 Le script est **tolérant** : il tente d’abord une lecture standard, puis bascule vers le déchiffrement si nécessaire.

---

### 3) Sélection des colonnes utiles

Dans le fichier “dossier gynéco”, le script :

* Identifie la colonne contenant le **texte clinique libre**
  (souvent quelque chose comme *Consultation*, *Texte*, *Observation gynécologique*, selon la version)

* Identifie la colonne **id_patiente / numéro d’inclusion**

* Ignore toutes les autres colonnes

👉 Cette étape est volontairement **conservative** : on ne garde que ce qui sert au NLP et à l’UI.

---

### 4) Construction du CSV “texte par patiente”

Pour chaque patiente :

* récupération du texte brut
* nettoyage minimal :

  * suppression des `NaN`
  * normalisation des espaces
  * concaténation si le texte est réparti sur plusieurs lignes

Export final :

```
dossier_gyneco_texte_par_patiente.csv
```

Structure :

| id_patiente | texte                |
| ----------- | -------------------- |
| AE-060      | “… texte clinique …” |
| AM-164      | “… texte clinique …” |

👉 **C’est ce fichier qui est relu par `app.py` pour l’affichage patient.**

---

### 5) Extraction du vocabulaire brut

À partir de l’ensemble des textes :

* découpage naïf en tokens (split / regex simple)
* passage en minuscules
* comptage global via `Counter`

Aucune décision linguistique ici :

* pas de SpaCy
* pas de stopwords
* pas de correction

Export final :

```
vocab_dossier_gyneco_from_xlsx.csv
```

Structure :

| token        | count |
| ------------ | ----- |
| endometriose | 446   |
| echo         | 500   |
| andré        | 555   |

👉 Ce fichier est **l’entrée directe de `filter_tokens_with_spacy.py`**.

---

## Comment exécuter l’étape seule

Depuis `exploratory/preprocessing/` :

```bash
python extract_text_and_vocab_from_dossier_gyneco.py
```

Résultat attendu :

* `Data/DATA_PROCESSED/dossier_gyneco_texte_par_patiente.csv`
* `Data/DATA_PROCESSED/vocab_dossier_gyneco_from_xlsx.csv`

---

## Contrôles rapides après exécution

```bash
ls Data/DATA_PROCESSED
```

Puis, en Python ou Excel :

* Le CSV texte contient autant de lignes que de patientes attendues
* Le vocabulaire contient plusieurs milliers de tokens
* Aucun champ `texte` n’est vide

Si `dossier_gyneco_texte_par_patiente.csv` est vide :

* problème de nom de colonne
* problème de feuille Excel
* fichier non déchiffré correctement

---

## Rôle de cette étape dans le pipeline global

Cette étape :

* **alimente directement l’UI Flask**
* **alimente tout le pipeline NLP**
* est **indépendante de la base SQLite** (contrairement à l’étape A)

Elle peut donc être :

* relancée seule,
* modifiée sans impacter la DB,
* testée indépendamment.

---

## Point clé pour la suite

* `dossier_gyneco_texte_par_patiente.csv`
  → UI (`app.py`, `patient.html`)

* `vocab_dossier_gyneco_from_xlsx.csv`
  → `filter_tokens_with_spacy.py`
  → `suggest_dict_extensions.py`

---


