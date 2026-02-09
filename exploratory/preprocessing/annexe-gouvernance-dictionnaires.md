← [Accueil du Wiki](home.md)

# Annexe — Gouvernance des dictionnaires

## Objectif

Définir les règles de **gestion, validation et évolution** des dictionnaires
utilisés dans le pipeline ENDOPATH.

Ces dictionnaires constituent la **source de vérité métier** des corrections.

---

## 1) Dictionnaires concernés

- `dictionnaire_correction.json`
- `abbrev_sure_merged.json`
- `abbrev_ambigue_merged.json`

Ils sont stockés dans :
Data/DATA_PROCESSED/Correction_mots/

---

## 2) Rôles et responsabilités

| Rôle | Responsabilité |
|-----|----------------|
| Développeur | Implémentation, règles d’application |
| Référent métier | Validation du contenu linguistique |
| Projet | Arbitrage des évolutions |

---

## 3) Cycle de vie d’une correction

1. Détection automatique (tokens / suggestions)
2. Validation via UI Flask (choix utilisateur)
3. Consolidation dans un dictionnaire
4. Versionnement Git
5. Redéploiement pipeline

➡️ **Aucune correction n’est appliquée en production sans validation humaine.**

---

## 4) Règles de versionnement

- Les dictionnaires sont **versionnés**
- Les fichiers générés (`suggestions_*.csv`) **ne le sont pas**
- Toute modification est tracée par commit

---

## 5) Justification

Cette gouvernance garantit :
- explicabilité totale
- maîtrise métier
- auditabilité clinique
- conformité RGPD
