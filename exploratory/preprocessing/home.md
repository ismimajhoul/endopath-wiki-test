# ENDOPATH — Wiki

Bienvenue sur la documentation ENDOPATH.

Ce wiki regroupe la documentation **technique** et **procédurale** du projet :
- pipeline de préparation de données (XLSX → SQLite → CSV → suggestions)
- règles de normalisation et de correction linguistique
- interface web Flask de revue et validation
- organisation des scripts et enchaînement d’exécution

---

## Démarrage rapide

1. [Installation](installation.md)
2. [Lancement](lancement.md)
3. [Préprocessing — Sommaire](preprocessing-sommaire.md)
4. [Préprocessing — Historique (ancienne approche Nicolai)](preprocessing-historique.md)
5. [Préprocessing — Base de données (Création ENDOPATH)](preprocessing-base-de-donnees-creation-endopath.md)
6. [Préprocessing — Extraction de texte](preprocessing-extraction-texte.md)
7. [Préprocessing — Filtrage linguistique](preprocessing-filtrage-linguistique.md)
8. [Préprocessing — Suggestions](preprocessing-suggestions.md)
9. [Préprocessing — UI Flask](preprocessing-ui-flask.md)
10. [Synthèse & Benchmark](synthese-benchmark.md)

---

## Annexes

- [Gouvernance des dictionnaires](annexe-gouvernance-dictionnaires.md)
- [Choix techniques & contraintes (RGPD, données cliniques)](annexe-choix-techniques-contraintes.md)
- [Références & liens utiles](annexe-references-liens.md)

---

## Conventions de documentation

- 1 page = 1 sujet
- Les pages décrivent un **processus**
- Le préfixe `Preprocessing —` structure la navigation
- Les liens sont des **liens Markdown relatifs**
- Les noms de fichiers utilisent des **slugs ASCII** (compatibles GitHub / GitLab)

---

## Historique

Les évolutions de documentation sont versionnées via les commits de ce dépôt.
