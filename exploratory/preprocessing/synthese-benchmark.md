← [Accueil du Wiki](home.md) / [Préprocessing — Sommaire](preprocessing-sommaire.md)

---
# Synthèse & Benchmark — ENDOPATH

## 1. Contexte et objectifs

Le projet ENDOPATH vise à améliorer la **qualité, la lisibilité et la cohérence**
des textes cliniques gynécologiques, en combinant :

- un pipeline de prétraitement NLP,
- des dictionnaires métier contrôlés,
- une interface utilisateur de validation médicale.

Cette page propose :
- une **synthèse des approches existantes**,
- un **benchmark raisonné**,
- une **justification des choix retenus**.

---

## 2. Approches étudiées

### 2.1 Approche historique — Batch par dictionnaire (Nicolai)

**Principe :**
- dictionnaire JSON (mot → mot),
- tokenisation simple,
- application hors ligne (batch).

**Avantages :**
- simplicité,
- rapidité d’exécution,
- facile à maintenir sur petits volumes.

**Limites :**
- perte de ponctuation,
- pas de multi-mots,
- pas de gestion des ambiguïtés,
- aucune interaction utilisateur,
- résultats non traçables.

➡️ Voir : *Préprocessing — Historique*

---

### 2.2 Approches NLP classiques (état de l’art)

| Approche | Avantages | Limites |
|--------|---------|--------|
| Règles regex | Rapide, explicable | Fragile, difficile à maintenir |
| Correcteurs automatiques | Bonne couverture | Erreurs métier |
| Modèles ML / LLM | Puissance sémantique | RGPD, coût, explicabilité |

➡️ **Non retenues en l’état** pour un contexte clinique sensible.

---

### 2.3 Approche retenue — Pipeline + UI interactive

**Principe :**
- détection automatique (suggestions),
- classification par familles,
- validation humaine via UI Flask,
- rendu Avant / Après contrôlé.

**Bénéfices clés :**
- explicabilité totale,
- contrôle utilisateur,
- traçabilité des corrections,
- évolutivité métier.

---

## 3. Benchmark comparatif

| Critère | Batch historique | NLP auto | ENDOPATH (actuel) |
|------|-----------------|----------|------------------|
| Explicabilité | ✔️ | ❌ | ✔️ |
| Gestion ambiguïtés | ❌ | ⚠️ | ✔️ |
| Multi-mots | ❌ | ✔️ | ✔️ |
| Interaction utilisateur | ❌ | ❌ | ✔️ |
| RGPD | ✔️ | ❌ | ✔️ |
| Maintenabilité | ⚠️ | ❌ | ✔️ |

---

## 4. Recommandation finale

L’approche **pipeline + UI Flask** est la plus adaptée au contexte ENDOPATH car elle :

- respecte les contraintes cliniques et RGPD,
- conserve une explicabilité totale,
- permet une amélioration continue des dictionnaires,
- implique le médecin dans la validation finale.

Cette architecture constitue une **base robuste** pour de futures évolutions
(Django / Angular, workflows, historisation).

---

## 5. Lien avec le reste du Wiki

- Approche historique : *Préprocessing — Historique*
- Pipeline détaillé : *Préprocessing — Sommaire*
- Interface : *Préprocessing — UI Flask*
