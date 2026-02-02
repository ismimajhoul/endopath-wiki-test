Ce projet utilise GitLab CI pour automatiser :

- L’exécution des tests unitaires avec **pytest**.
- La génération d’un rapport de couverture de code avec **pytest-cov**.
- L’analyse statique du code avec **pylint**.

Le pipeline est défini dans le fichier `.gitlab-ci.yml` à la racine du dépôt.

## Fonctionnement du pipeline CI

- À chaque **push** sur la branche principale (`main` ou `master`), GitLab lance automatiquement les jobs définis dans `.gitlab-ci.yml`.
- **Tests** : installation des dépendances, exécution des tests avec collecte de la couverture.
- **Linting** : analyse du code source avec pylint pour détecter les erreurs et améliorer la qualité.
- Les rapports de couverture sont générés au format Cobertura (`coverage.xml`) et affichés dans l’interface GitLab.
- Les résultats de pylint sont affichés dans les logs du job.

## Utilisation

1. Poussez vos modifications sur la branche principale.
2. GitLab déclenche automatiquement le pipeline.
3. Consultez les résultats dans l’onglet **CI/CD > Pipelines**.
4. La couverture de code est visible dans les merge requests.
5. Les erreurs pylint apparaissent dans les logs du job `lint`.

## Bonnes pratiques

- Configurez `pylint` via un fichier `.pylintrc` pour adapter les règles à votre projet.
- Utilisez les variables CI/CD GitLab pour gérer les secrets et configurations sensibles.
- Vérifier le coverage avant chaque `commit`. **Aucun code non testé ne doit apparaitre dans le dossiers `src`.**