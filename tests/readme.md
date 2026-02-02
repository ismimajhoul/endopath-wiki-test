# 1. Exemple : Prétraitement des textes médicaux - Correction orthographique

### Fichier `features/preprocessing_correction.feature`

```gherkin
Feature: Prétraitement des textes médicaux

  Scenario: Correction orthographique et normalisation des dossiers gynécologiques
    Given un texte médical brut contenant des fautes d’orthographe et des abréviations non standardisées
    When le script de correction orthographique est appliqué avec le dictionnaire de corrections
    Then le texte corrigé ne contient plus de fautes majeures
    And les abréviations sont uniformisées
```


### Fichier `tests/test_preprocessing_correction.py`

```python
import pytest
from pytest_bdd import scenario, given, when, then

# Exemple simplifié de dictionnaire de corrections
DICTIONNAIRE_CORRECTION = {
    "endométriosee": "endométriose",
    "douleurss": "douleurs",
    "abdomenn": "abdomen",
    "échograhie": "échographie",
    "IRMm": "IRM",
    "douleur pelvienne": "douleur pelvienne"
}

def correction_orthographique(texte, dictionnaire):
    mots = texte.split()
    mots_corriges = [dictionnaire.get(mot, mot) for mot in mots]
    return " ".join(mots_corriges)

@scenario('features/preprocessing_correction.feature', 'Correction orthographique et normalisation des dossiers gynécologiques')
def test_correction_orthographique():
    pass

@pytest.fixture
def texte_brut():
    return "La patiente présente une endométriosee avec douleurss abdominales et IRMm."

@pytest.fixture
def dictionnaire():
    return DICTIONNAIRE_CORRECTION

@given("un texte médical brut contenant des fautes d’orthographe et des abréviations non standardisées")
def given_texte_brut(texte_brut):
    return texte_brut

@when("le script de correction orthographique est appliqué avec le dictionnaire de corrections")
def when_correction_appliquee(given_texte_brut, dictionnaire):
    return correction_orthographique(given_texte_brut, dictionnaire)

@then("le texte corrigé ne contient plus de fautes majeures")
def then_texte_corrige(when_correction_appliquee):
    # Vérifie qu'aucun mot mal orthographié connu ne subsiste
    fautes_connues = ["endométriosee", "douleurss", "abdomenn", "échograhie", "IRMm"]
    for faute in fautes_connues:
        assert faute not in when_correction_appliquee

@then("les abréviations sont uniformisées")
def then_abreviations_uniformisees(when_correction_appliquee):
    # Exemple simple : vérifier que 'IRMm' est devenu 'IRM'
    assert "IRM" in when_correction_appliquee
```


# 2. Exemple : Gestion des données manquantes - Filtrage patientes

### Fichier `features/missing_data_filter.feature`

```gherkin
Feature: Gestion des données manquantes

  Scenario: Filtrage des patientes selon le taux de données manquantes
    Given un jeu de données avec des patientes ayant des taux variables de données manquantes
    When on filtre les patientes pour ne conserver que celles avec moins de 20% de données manquantes
    Then le jeu de données résultant ne contient que des patientes avec un taux de données manquantes inférieur à 20%
```


### Fichier `tests/test_missing_data_filter.py`

```python
import pytest
from pytest_bdd import scenario, given, when, then
import pandas as pd

@scenario('features/missing_data_filter.feature', 'Filtrage des patientes selon le taux de données manquantes')
def test_filtrage_patientes():
    pass

@pytest.fixture
def jeu_donnees():
    # Exemple simplifié : DataFrame avec patientes et taux de données manquantes
    data = {
        'patiente_id': [1, 2, 3, 4],
        'var1': [1, None, 0, 1],
        'var2': [None, None, 1, 1],
        'var3': [1, 1, None, 1]
    }
    df = pd.DataFrame(data)
    # Calcul taux manquantes par patiente
    df['taux_manquantes'] = df.isnull().mean(axis=1)
    return df

@given("un jeu de données avec des patientes ayant des taux variables de données manquantes")
def given_jeu_donnees(jeu_donnees):
    return jeu_donnees

@when("on filtre les patientes pour ne conserver que celles avec moins de 20% de données manquantes")
def when_filtrage_applique(given_jeu_donnees):
    df_filtre = given_jeu_donnees[given_jeu_donnees['taux_manquantes'] < 0.2]
    return df_filtre

@then("le jeu de données résultant ne contient que des patientes avec un taux de données manquantes inférieur à 20%")
def then_verification_filtrage(when_filtrage_applique):
    assert all(when_filtrage_applique['taux_manquantes'] < 0.2)
```


# 3. Exemple : Prédiction endométriose avec XGBClassifier

### Fichier `features/prediction_endo.feature`

```gherkin
Feature: Prédiction de l’endométriose

  Scenario: Modèle XGBClassifier entraîné sur données filtrées
    Given un jeu de données structuré filtré avec moins de 20% de données manquantes
    When le modèle XGBClassifier est entraîné et testé
    Then la sensibilité est au moins 0.6
    And la spécificité est au moins 0.8
```


### Fichier `tests/test_prediction_endo.py`

```python
import pytest
from pytest_bdd import scenario, given, when, then
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import recall_score, precision_score, confusion_matrix

@scenario('features/prediction_endo.feature', 'Modèle XGBClassifier entraîné sur données filtrées')
def test_prediction_xgb():
    pass

@pytest.fixture
def jeu_donnees_filtre():
    # Jeu de données synthétique simplifié
    np.random.seed(42)
    n = 130
    X = pd.DataFrame({
        'age': np.random.randint(20, 50, n),
        'IMC': np.random.uniform(18, 35, n),
        'parite': np.random.randint(0, 3, n),
        'sympt1': np.random.randint(0, 2, n),
        'sympt2': np.random.randint(0, 2, n),
        'sympt3': np.random.randint(0, 2, n),
    })
    y = np.random.randint(0, 2, n)  # 0 = absence, 1 = présence d’endométriose
    return X, y

@given("un jeu de données structuré filtré avec moins de 20% de données manquantes")
def given_jeu_donnees(jeu_donnees_filtre):
    return jeu_donnees_filtre

@when("le modèle XGBClassifier est entraîné et testé")
def when_entrainement_modele(given_jeu_donnees):
    X, y = given_jeu_donnees
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    return sens, spec

@then("la sensibilité est au moins 0.6")
def then_verifier_sensibilite(when_entrainement_modele):
    sens, _ = when_entrainement_modele
    assert sens >= 0.6

@then("la spécificité est au moins 0.8")
def then_verifier_specificite(when_entrainement_modele):
    _, spec = when_entrainement_modele
    assert spec >= 0.8
```
