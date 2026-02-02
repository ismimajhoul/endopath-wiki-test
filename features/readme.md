# Features Gherkin pour le projet Endopath

Chaque **Feature** correspond à une fonctionnalité ou un module du projet, et chaque **Scenario** détaille un cas d’usage précis, avec des étapes claires **Given / When / Then**. Les scénarios sont organisés par sous-domaines, ils sont analogues au éxigences attendu pour le projet.

## Feature: Prétraitement des données textuelles des dossiers gynécologiques

**Objectif** : Assurer la qualité, la cohérence et la standardisation des textes médicaux avant toute analyse NLP.

### Scenario: Correction orthographique des textes bruts

```
Given un dossier gynécologique contenant des textes avec fautes d’orthographe et abréviations non standardisées
When le script de correction orthographique est appliqué avec le dictionnaire de corrections
Then le texte est corrigé sans fautes majeures
And les abréviations sont remplacées par leur forme standardisée
```


### Scenario: Gestion des mots inconnus et abréviations spécifiques

```
Given un texte médical contenant des mots inconnus et des abréviations non référencées
When une vérification lexicale est effectuée avec la liste all_words.txt
Then tous les mots inconnus sont signalés pour validation manuelle
And un dictionnaire de corrections est mis à jour
```


### Scenario: Extraction manuelle vs automatique des informations textuelles

```
Given un ensemble de données extraites manuellement des dossiers gynécologiques
And un ensemble de données extraites automatiquement par scripts NLP
When les deux ensembles sont comparés pour une variable donnée
Then le taux de correspondance est calculé
And ce taux est supérieur ou égal à 80%
```


## Feature: Gestion et nettoyage des données structurées (Recueil)

**Objectif** : Garantir la qualité des données structurées avant modélisation.

### Scenario: Filtrage des patientes selon le taux de données manquantes

```
Given un jeu de données avec des variables contenant jusqu’à 27% de valeurs manquantes
When on filtre les patientes pour ne conserver que celles avec moins de 20% de données manquantes
Then le jeu de données résultant contient uniquement des patientes avec un taux de données manquantes < 20%
```


### Scenario: Suppression des colonnes avec trop de valeurs manquantes

```
Given un jeu de données avec des colonnes ayant plus de 90% de valeurs manquantes
When on supprime ces colonnes du jeu de données
Then ces colonnes ne sont plus présentes dans le jeu de données final
```


### Scenario: Uniformisation des formats et valeurs

```
Given un jeu de données avec des variables binaires codées de manière hétérogène (ex: "Oui"/"Non", "1"/"0")
When on applique une normalisation des valeurs binaires en 0 et 1
Then toutes les variables binaires sont uniformisées en 0 (absence) ou 1 (présence)
```


## Feature: Sélection et importance des variables cliniques

**Objectif** : Identifier les variables les plus pertinentes pour la prédiction de l’endométriose.

### Scenario: Identification des variables significatives par test chi2

```
Given un jeu de données structuré complet
When on applique un test statistique chi2 sur chaque variable par rapport au diagnostic d’endométriose
Then une liste de variables avec p-value < 0.05 est retournée
And cette liste contient au moins 8 variables significatives
```


### Scenario: Réduction du nombre de variables pour améliorer la performance

```
Given une liste initiale de variables cliniques
When on sélectionne uniquement les 9 variables les plus significatives plus âge, IMC et parité
Then les modèles entraînés avec ce sous-ensemble ont une meilleure performance (sensibilité et spécificité)
```


## Feature: Modélisation et prédiction de l’endométriose

**Objectif** : Évaluer la performance des modèles ML sur différentes sous-populations.

### Scenario: Entraînement du modèle XGBClassifier sur données complètes

```
Given un jeu de données structuré complet avec 181 patientes
When le modèle XGBClassifier est entraîné et évalué
Then la sensibilité est au moins 0.56
And la spécificité est au moins 0.81
```


### Scenario: Entraînement du modèle XGBClassifier sur cohortes filtrées (<20% données manquantes)

```
Given un jeu de données filtré avec moins de 20% de données manquantes (130 patientes)
When le modèle XGBClassifier est entraîné et évalué
Then la sensibilité est au moins 0.67
And la spécificité est au moins 0.83
```


### Scenario: Impact du taux de données manquantes sur la performance

```
Given plusieurs sous-jeux de données avec différents seuils de données manquantes (<20%, <25%, etc.)
When on entraîne un modèle XGBClassifier sur chacun
Then la performance (sensibilité et spécificité) diminue avec l’augmentation du taux de données manquantes
```


## Feature: Extraction automatique des symptômes via NLP

**Objectif** : Transformer les textes médicaux en variables cliniques exploitables.

### Scenario: Extraction des symptômes avec modèle Decision Tree

```
Given des textes médicaux prétraités
When un modèle Decision Tree est entraîné pour prédire la présence d’un symptôme
Then la précision est inférieure à 0.6 (performance faible)
```


### Scenario: Extraction des symptômes avec modèle LSTM

```
Given des textes médicaux prétraités
When un modèle LSTM est entraîné pour prédire la présence d’un symptôme
Then la précision est supérieure à celle du Decision Tree mais inférieure à 0.75
```


### Scenario: Extraction des symptômes avec modèle Transformer CamemBERT

```
Given des textes médicaux prétraités
When un modèle CamemBERT est appliqué pour prédire la présence d’un symptôme
Then la précision est supérieure à 0.7 pour certains symptômes ciblés
```


### Scenario: Extraction d’entités médicales par NER CamemBERT-bio-gliner

```
Given des textes médicaux prétraités
When le modèle NER CamemBERT-bio-gliner est appliqué
Then les entités biomédicales (médicaments, symptômes, maladies) sont extraites
And les termes spécifiques à l’endométriose sont identifiés mais peu discriminants
```


## Feature: Comparaison et synthèse des données recueil vs dossiers gynécologiques

**Objectif** : Valider la cohérence et prioriser les sources en cas de conflit.

### Scenario: Calcul du taux de correspondance pour une variable donnée

```
Given les données recueil et gynéco pour la variable "Antécédent endométriose"
When on calcule le taux de correspondance entre les deux sources
Then ce taux est supérieur ou égal à 0.8
```


### Scenario: Synthèse des données en priorisant la source gynéco

```
Given des données conflictuelles entre recueil et gynéco pour une variable
When la synthèse des données est effectuée avec priorité à la source gynéco
Then la valeur finale correspond à celle de la source gynéco
```


### Scenario: Synthèse des données en priorisant la source recueil

```
Given des données conflictuelles entre recueil et gynéco pour une variable
When la synthèse des données est effectuée avec priorité à la source recueil
Then la valeur finale correspond à celle de la source recueil
```


## Feature: Validation globale du pipeline Endopath

**Objectif** : Assurer la cohérence et la performance du pipeline complet.

### Scenario: Pipeline complet avec données nettoyées et variables sélectionnées

```
Given un jeu de données structuré nettoyé et filtré (<20% données manquantes)
And les textes médicaux prétraités et corrigés
When le pipeline complet (extraction, sélection, modélisation) est exécuté
Then la prédiction finale de l’endométriose atteint une sensibilité ≥ 0.65 et une spécificité ≥ 0.8
```


### Scenario: Robustesse du pipeline face à des données incohérentes

```
Given des données recueil et gynéco présentant des conflits sur certaines variables
When le pipeline applique la synthèse priorisant la source gynéco
Then la cohérence des données est maintenue
And la performance du modèle reste stable
```