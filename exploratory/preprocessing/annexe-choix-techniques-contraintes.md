← [Accueil du Wiki](home.md)

# Annexe — Choix techniques & contraintes (RGPD, données cliniques)

## Objectif

Documenter les **contraintes non fonctionnelles** ayant guidé les choix
d’architecture du projet ENDOPATH.

---

## 1) Contraintes réglementaires

### Données cliniques sensibles
- données patientes
- textes médicaux libres
- contexte hospitalier

➡️ **RGPD prioritaire**

---

## 2) Conséquences techniques

| Sujet | Décision |
|-----|---------|
| Cloud externe | ❌ refusé |
| LLM publics | ❌ refusé |
| Pipeline local | ✔️ |
| Validation humaine | ✔️ obligatoire |

---

## 3) Justification du choix Flask

- léger
- explicable
- séparation claire front / back
- compatible audit et traçabilité

---

## 4) Non-choix assumés

- Pas de correction automatique silencieuse
- Pas de modèle opaque
- Pas de dépendance cloud

➡️ **La qualité clinique prime sur l’automatisation maximale.**

---

