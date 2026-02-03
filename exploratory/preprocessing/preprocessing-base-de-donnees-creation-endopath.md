← [Retour au sommaire Préprocessing](preprocessing-sommaire.md)

---

# Création de la base de données ENDOPATH (XLSX → SQLite)

**Objectif :** documenter le processus de construction de la base SQLite `endopath_diag.db` à partir des fichiers Excel sources du projet ENDOPATH.

Cette page constitue la **première étape du pipeline de traitement des données** utilisé par l’application.

## Contexte

Le projet ENDOPATH manipule des données cliniques issues de plusieurs sources Excel
(inclusion, recueil clinique, dossiers gynécologiques, PMSI).

Afin de garantir :
- la cohérence des données,
- la reproductibilité des traitements,
- et la séparation entre données et logique métier,

l’ensemble des sources est consolidé dans une **base SQLite unique** : `endopath_diag.db`.


Étape 1 — Construire la base SQLite (XLSX → SQLite)
Fichier

create_endopath_diag_db.py

Objectif

Consolider les sources Excel “brutes” (inclusion, recueil, gynéco, PMSI) dans une base SQLite unique : endopath_diag.db, afin d’alimenter ensuite :

les scripts d’extraction CSV (tokens/vocabs/suggestions),

puis l’UI Flask (et éventuellement d’autres frontends).

Entrées (inputs)

Le script travaille à partir de BASE_DIR/Data/DATA_RAW/ et de 4 fichiers XLSX :

Data/DATA_RAW/INCLUSION RECHERCHE CLINIQUE.xlsx

Data/DATA_RAW/Recueil_MMJ.xlsx

Data/DATA_RAW/dossier-gyneco-23-03-2022_converti.xlsx

Data/DATA_RAW/2022 - Donnees PMSI - Protocole ENDOPATHS - GHN..ALTRAN_converti.xlsx

⚠️ Le script attend des fichiers réellement XLSX (zip Office). Il vérifie le header PK via ensure_real_xlsx().

Sortie (output)

Fichier DB : endopath_diag.db (créé à la racine du projet, à côté du script)

Tables SQLite créées/écrasées :

INCLUSION_RECHERCHE_CLINIQUE

Recueil_MMJ

gyneco_diag_raw

pmsi_diag

diag_comparaison

Modules Python utilisés (et pourquoi)

pathlib.Path : gestion robuste des chemins (Windows/Linux), construction BASE_DIR / "Data" / ...

pandas :

pd.read_excel(...) pour charger les feuilles Excel,

nettoyage simple et standardisation des colonnes,

to_sql(...) pour insérer rapidement dans SQLite.

sqlite3 : connexion, création de table, index, requêtes SQL, contrôle des schémas.

traceback : logs détaillés en cas d’erreur.

Déroulé interne (ce que fait le script, dans l’ordre)
1) Initialisation des chemins

Constantes en haut du fichier :

BASE_DIR

DATA_RAW

chemins des 4 XLSX

DB_PATH = BASE_DIR / "endopath_diag.db"

➡️ 
2) Validation “vrai XLSX”

Fonction : ensure_real_xlsx(path: Path)

Lit les 2 premiers octets du fichier

Vérifier que c’est bien un zip office (b"PK")


➡️ Cela évite les faux .xlsx (HTML renommé, CSV déguisé, fichier corrompu, etc.).

3) Création / reset de la base

Fonction : ensure_table(conn, table_name, columns_sql)

Supprime la table si elle existe

Recrée la table avec un schéma SQL explicite (types TEXT/REAL/INTEGER)

Sert de “socle” pour insérer proprement ensuite

4) Table INCLUSION_RECHERCHE_CLINIQUE (Inclusion ↔ IPP)

Fonction : build_inclusion(conn)

Charge l’Excel inclusion via pd.read_excel

Normalise les noms de colonnes (trim)

Sélectionne / renomme pour obtenir (au minimum) :

id_patiente (ex: “AE-060”)

ipp (identifiant patient)

Écrit dans SQLite (to_sql(..., if_exists="replace"))

Crée un index sur id_patiente si prévu

➡️ C’est la table “référence” qui permet de relier le reste au numéro d’inclusion.

5) Table Recueil_MMJ

Fonction : build_recueil_mmj(conn)

Charge l’Excel recueil

Conserve les colonnes utiles (variables cliniques / structure)

Écrit dans SQLite

➡️ Sert de source structurée complémentaire (selon ce que tu exploiteras ensuite).

6) Table gyneco_diag_raw (texte clinique principal)

Fonction : build_gyneco_diag_raw(conn)

Charge le fichier “dossier gynéco … converti”

Construit une table brute contenant le texte source par patiente

Le script vise typiquement une colonne “consultation/texte” (selon ton Excel)

Écrit dans SQLite

➡️ C’est LA source principale pour tout ce qui est NLP/tokens/suggestions et affichage UI.

7) Table pmsi_diag (PMSI)

Fonction : build_pmsi_diag(conn)

Charge l’Excel PMSI

Extrait les champs utiles (codes, libellés, etc.)

Écrit dans SQLite

➡️ Source structurée, utile si tu veux enrichir le contexte ou comparer clinique vs codage.

8) Table diag_comparaison (table pivot / vue “hub”)

Fonction : build_diag_comparaison(conn)

Crée une table pivot qui regroupe, par id_patiente :

données inclusion

éventuellement champs texte gynéco

éventuellement champs PMSI

éventuellement champs recueil MMJ

Le but est d’avoir une table “centrale” à requêter facilement.

➡️ Très pratique pour la suite du pipeline (exports CSV, UI, contrôles qualité).

Comment exécuter

Depuis la racine du projet (là où est create_endopath_diag_db.py) :

python create_endopath_diag_db.py


Résultat attendu :

endopath_diag.db est (re)créée

les tables listées ci-dessus existent et sont remplies.

Contrôles rapides après exécution

Dans SQLiteStudio ou en Python :

SELECT COUNT(*) FROM INCLUSION_RECHERCHE_CLINIQUE;
SELECT COUNT(*) FROM gyneco_diag_raw;
SELECT COUNT(*) FROM pmsi_diag;
SELECT COUNT(*) FROM diag_comparaison;


Si gyneco_diag_raw est vide → l’Excel “dossier gynéco” n’a pas été lu comme attendu (souvent problème de nom de feuille/colonne).

Point important pour la suite (pipeline complet)

Cette DB endopath_diag.db est ensuite la source stable pour :

l’extraction CSV (tokens/vocab/suggestions),

puis l’UI Flask qui lit les CSV (ou lit la DB si tu la fais évoluer).


## Place dans le pipeline global

Cette étape correspond à **l’étape 1 du pipeline ENDOPATH**.

Elle est un prérequis obligatoire pour :
- l’extraction des textes patients,
- la génération des vocabulaires et tokens,
- le calcul des suggestions de correction,
- l’affichage dans l’UI Flask.

Les étapes suivantes sont décrites dans les pages Wiki dédiées.

## Pages associées

- Étape 2 — Extraction texte & vocabulaire (à venir)
- Étape 3 — Filtrage linguistique (SpaCy) (à venir)
- Étape 4 — Suggestions & dictionnaires (à venir)
- Guide UI Flask (à venir)

