# ENDOPATH — Wiki

Bienvenue sur la documentation ENDOPATH.

Ce wiki regroupe la documentation **technique** et **procédurale** du projet :
- pipeline de préparation de données (XLSX → SQLite → CSV → suggestions)
- règles de normalisation / correction de vocabulaire
- UI Flask de revue et application des corrections
- organisation des scripts et exécution pas à pas

---

## Démarrage rapide (où commencer)

1. Lire la vue d’ensemble : [[Preprocessing - Sommaire]]
2. Créer la base :
   - [[Preprocessing - Base de données (Création ENDOPATH)]]
3. Exécuter le pipeline :
   - [[Preprocessing - Extraction de texte]]
   - [[Preprocessing - Filtrage linguistique]]
   - [[Preprocessing - Suggestions]]
4. Revoir / appliquer les corrections :
   - [[Preprocessing - UI Flask]]

---

## Préprocessing

- [[Preprocessing - Sommaire]]
- [[Preprocessing - Base de données (Création ENDOPATH)]]
- [[Preprocessing - Extraction de texte]]
- [[Preprocessing - Filtrage linguistique]]
- [[Preprocessing - Suggestions]]
- [[Preprocessing - UI Flask]]

---

## Base de données

- [[Base de données - Création ENDOPATH]]  
  Création/initialisation de la base (XLSX → SQLite), structure et tables/vues attendues.

---

## Conventions

### Nom des pages
- Les pages wiki doivent décrire **un sujet** (pas un dossier).
- Un fichier = une page.
- Le préfixe indique le domaine : `Preprocessing - ...`, `Base de données - ...`, etc.

### Liens internes
Utiliser le format GitHub Wiki :
- `[[Nom de la page]]`  
Sans extension `.md`.

---

## Glossaire minimal

- **Patiente** : entité patient dans le corpus (données anonymisées si nécessaire).
- **Dictionnaire** : ensemble de corrections/normalisations (abréviations, variantes orthographiques…).
- **Suggestions** : propositions automatiques de nouvelles entrées dans le dictionnaire (à valider via UI).

---

## Changelog doc
- Les évolutions de documentation sont versionnées par commits dans ce repo “wiki”.
