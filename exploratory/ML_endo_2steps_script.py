"""
Notebook to predict endometriosis in 2 steps:
Predict presence or absence of endometriosis in general, superficial or profoundAmong those patients where the model predcited endometriosis, predict the type of endometriosis (superficial or profound)
"""
### Imports ###

# Data manipulation and other stuff : 
import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 500)
from matplotlib import pyplot as plt
import seaborn as sns
#import eli5 # eli5 not working anymore for current stable version of sklearn
from IPython.display import display

import eli5

# Utils for NLP : 
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split

# Utils for encoding : 
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, OneHotEncoder

# Utils for regression : 
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC

# Utils for Multilabel classification :
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Utils for Metrics calculation : 
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, precision_recall_curve
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, roc_curve
from exploratory.utils.metrics_utils import rewrite_keys, rapport_metrics_decision_tree, multilabel_multioutput_svc, multilabel_multioutput_LR, custom_show_weights

from sklearn.tree import plot_tree

# Multiclass/Multilabel preparation :
from sklearn.base import BaseEstimator, ClassifierMixin

# Custom preprocessing : 
from exploratory.preprocessing.preprocess_NLP import preprocess_and_split

# Tensorflow/keras
import tensorflow as tf
import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras.models import Sequential
from keras.layers import *
from tensorflow.keras.utils import to_categorical
from keras.initializers import Constant
from keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split, GridSearchCV
from skopt import BayesSearchCV

# Utils for encoding : 
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, OneHotEncoder

# Utils for regression : 
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import accuracy_score, classification_report, make_scorer

# Utils for classification :
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier, XGBRFClassifier, plot_importance
from sklearn.ensemble import HistGradientBoostingClassifier

# Utils for Metrics calculation : 
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, precision_recall_curve
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, roc_curve
from exploratory.utils.metrics_utils import rewrite_keys, rapport_metrics_decision_tree
from sklearn.model_selection import cross_validate, cross_val_score, cross_val_predict

# Custom utils  
from exploratory.Opti_utils.ML_utils import find_best_threshold, from_model_to_bst_trhld, custom_metrics, scores, compare_results,find_error
from exploratory.Opti_utils.ML_utils import FP, TP, FN, TN, check_corr_col, filtre_list, fit_method, check_baseline, check_FP_FN, check_best_results
from exploratory.Opti_utils.ML_utils import ML_opti_recueil, univariate_ml, sensi, speci, mix_sensi_speci, custom_metric
from exploratory.Opti_utils.ML_utils import Binarisation, model_to_plot, clean_lists


## Choisir soit les données synthétisées où l'information des données gynéco a été priorisé, soit receuil
# (Priorisé = dans le cas d'un conflit entre gynéco et receuil, on prend soit l'info dans gynéco soit receuil)

data_select = 'gynéco'     # 'gynéco', 'receuil', ou 'original'

recueil_imc  = pd.read_excel('/home/nounou/endopath/Data/DATA_RAW/Recueil (1).xlsx').drop('Unnamed: 90', axis=1)
if data_select != 'original':
    columns_receuil_to_include = ['Numéro anonymat', 'age', 'imc', 'g', 'p', 'anapath.lusd','anapath.lusg','anapath.torus','anapath.autre']
    recueil_imc = recueil_imc[columns_receuil_to_include]
recueil_imc = recueil_imc.rename(columns={'Numéro anonymat': 'Anonymisation'})
# Remplace les manquantes par un np.nan
recueil_imc.replace(['Na', 'NA', 'nan', 'Nan', 'NAN'], np.nan, inplace=True)

if data_select in ['gynéco', 'receuil']:
    df = pd.read_excel(f'/home/nounou/endopath/Data/DATA_PROCESSED/data_synth_priorité_{data_select}.xlsx')
else:
    # If original data are used, we don't need the synthesized ones except to note the patients
    df = pd.read_excel(f'/home/nounou/endopath/Data/DATA_PROCESSED/data_synth_priorité_gynéco.xlsx')

# Note les information originales receuil, en prenant les mêmes patientes
# Explication: On a pas les même sets de patientes entre receuil et gynéco, comme les deux ont été synthétisés, on a perdu des patientes
patients_in_data_synth = list(pd.unique(df['Anonymisation']))
include_patients = 'in_synth'   # 'all' or 'in_synth'
recueil_imc_patients_in_synth = recueil_imc.loc[recueil_imc['Anonymisation'].isin(patients_in_data_synth)]
    
print(f'Number of original patients in receuil: {recueil_imc.shape[0]}')
print(f'Number of patients in synthesized data: {recueil_imc_patients_in_synth.shape[0]}')
if data_select == 'original':
    if include_patients == 'in_synth':
        df = recueil_imc_patients_in_synth.copy()
    else:
        df = recueil_imc.copy()

# On enlève les colonnes liés a la chirurgie : 
liste_colonnes_chir = ['date.chir', 'chir.macro.lusd', 'chir.macro.lusg', 'chir.macro.torus',  'chir.macro.oma', 'chir.macro.uro', 'chir.macro.dig',  'chir.macro.superf', 'resec.lusd', 'resec.lusg', 'resec.torus', 'resec.autre']
for col_to_drop in liste_colonnes_chir:
    if col_to_drop in df.columns:
        df = df.drop(col_to_drop, axis=1)

df_orig = df.copy()


# Note most important feautres = those that appear in top 10 of feature importances of model
features_important = ['imc', 'age', 'g', 'atcd.endo', 'atcd.endo.type', 'atcd.infertilite', 'atcd.absentéisme', 
                      'irm.adm', 'irm.lusg', 'echo.lusg', 'echo.lusd', 'echo.oma', 'echo.adm', 
                      'rectosonographie_réalisée', 'lésion LUS G à la rectosonographie', 'lésion rectum à la rectosonographie',
                      'échographie_réalisée', 'tv.douleur.lusg', 'tv.douloureux', 'tv.nodule.lusd', 
                      'nausées', 'douleurs_defecations', 'dysuries', 'sf.dig.diarrhée', 'spotting', 'sf.dig.constip', 'désir.G', ]

# Drop all columns with percentage missing values higher than a given threshold
features_select = 'all'  # 'by_perc_missing'
if features_select == 'all':
    thresh_missing = 100
    
    # Calculer le pourcentage de NaN pour chaque colonne
    na_percentage = df.isna().mean() * 100

    # Créer un nouveau DataFrame avec les résultats
    na_df = pd.DataFrame({
        'Column': na_percentage.index,
        'NaN Percentage': na_percentage.values
    })

    # Trier le DataFrame par pourcentage de NaN en ordre décroissant
    na_df_sorted = na_df.sort_values(by='NaN Percentage', ascending=False)
    na_df_filtered = na_df_sorted[na_df_sorted['NaN Percentage'] > 0]

    # Step 1: Sort columns based on NaN percentage in descending order
    sorted_na_columns = na_df_filtered.sort_values(by='NaN Percentage', ascending=False)['Column']
    columns_no_missing = list(na_df_sorted.loc[na_df_sorted['NaN Percentage']==0, 'Column'])
    columns_above_threshold = na_percentage[na_percentage >= thresh_missing].index.tolist()
    columns_to_keep = list(df.drop(columns_above_threshold, axis=1).columns)
    if 'Anonymisation' not in columns_to_keep:
        columns_to_keep = columns_to_keep + ['Anonymisation']
    print(f'{len(columns_to_keep)} columns remaining')

else:
    columns_to_keep = list(df.columns)

# Try including different patients, for example filtering out those with many missing values
patients_select = 'by_perc_missing'   #'by_perc_missing'
if patients_select == 'by_perc_missing':
    max_perc_missing = 20
    if features_select == 'by_perc_missing':
        df = df[columns_to_keep]
    # Recompute percentage missing df because features might have been rejected
    df_missing_per_patient = pd.DataFrame()
    df_missing_per_patient['Anonymisation'] = df['Anonymisation']
    df_missing_per_patient['nmissing'] = df.apply(lambda x: x.isna().sum(), axis=1)
    df_missing_per_patient['perc_missing'] = (df_missing_per_patient['nmissing'] / df.shape[0]) * 100
    df_missing_per_patient_sorted_ascending = df_missing_per_patient.sort_values(by='nmissing', ascending=True)
    patients_select = list(df_missing_per_patient_sorted_ascending.loc[df_missing_per_patient_sorted_ascending.perc_missing<=max_perc_missing, 'Anonymisation'])
    df = df.loc[df['Anonymisation'].isin(patients_select)]
    print(f'{len(patients_select)} patients remaining')

if data_select != 'original':
    df_with_target = pd.merge(df, recueil_imc, on=['Anonymisation'])
else:
    df_with_target = df.copy()
endometriose_generale = df_with_target.loc[:,['anapath.lusd','anapath.lusg','anapath.torus','anapath.autre']].sum(axis=1).apply(lambda x: Binarisation(x))
df_with_target['endometriose_generale'] = endometriose_generale

df_with_target_endometriosis_patients = df_with_target.loc[df_with_target.endometriose_generale==1]
endometriose_profonde = df_with_target_endometriosis_patients.loc[:,['anapath.lusd','anapath.lusg','anapath.torus']].sum(axis=1).apply(lambda x: Binarisation(x))
df_with_target_endometriosis_patients['endometriose_profonde'] = endometriose_profonde

print(df_with_target.head())

### Show how many patients we have for each set
npatients_total = df_with_target.shape[0]
npatients_with_endo = df_with_target.loc[df_with_target.endometriose_generale==1].shape[0]
npatients_without_endo = df_with_target.loc[df_with_target.endometriose_generale==0].shape[0]
npatients_superficielle = df_with_target.loc[df_with_target['anapath.autre']==1].shape[0]
npatients_only_superficielle = df_with_target_endometriosis_patients.loc[df_with_target_endometriosis_patients.endometriose_profonde==0].shape[0]
npatients_profonde_and_superficielle = df_with_target_endometriosis_patients.loc[(df_with_target_endometriosis_patients.endometriose_profonde==1) &
                                                                                (df_with_target_endometriosis_patients['anapath.autre']==1)].shape[0]
npatients_profonde = df_with_target_endometriosis_patients.loc[(df_with_target_endometriosis_patients.endometriose_profonde==1)].shape[0]
npatients_only_profonde = npatients_profonde - npatients_profonde_and_superficielle
print(f'Nombre de patientes au total: {npatients_total}')
print(f'Nombre de patientes sans endo: {npatients_without_endo}')
print(f'Nombre de patientes avec endo superficielle ou profonde: {npatients_with_endo}')
print(f'Nombre de patientes avec endo superficielle: {npatients_superficielle}')
print(f'Nombre de patientes avec endo profonde: {npatients_profonde}')
print(f'Nombre de patientes avec endo superficielle uniquement: {npatients_only_superficielle}')
print(f'Nombre de patientes avec endo profonde uniquement: {npatients_only_profonde}')
print(f'Nombre de patientes avec endo superficielle et profonde: {npatients_profonde_and_superficielle}')


#Exploration ---------------------------

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
axes = axes.flatten()
custom_palette = {'Positif': "maroon", 'Négatif': "darkgreen"}
sns.histplot(data=df_with_target.replace({0: "Négatif", 1: "Positif"}), x='endometriose_generale', hue='endometriose_generale', palette=custom_palette, ax=axes[0])
custom_palette = {'Profonde': "maroon", 'Superficielle': "darkgreen"}
sns.histplot(data=df_with_target_endometriosis_patients.replace({0: "Superficielle", 1: "Profonde"}), x='endometriose_profonde', hue='endometriose_profonde', palette=custom_palette, ax=axes[1])
axes[1].set_ylim(axes[0].get_ylim())
plt.ylabel('Nombre de patientes')


# ML ---------------------------------

df_with_target_endometriose_generale = df_with_target.copy()
df_with_target_endometriose_generale.set_index('Anonymisation', inplace=True)
df_with_target_endometriose_profonde = df_with_target_endometriosis_patients.copy()
df_with_target_endometriose_profonde.set_index('Anonymisation', inplace=True)
df_with_target_endometriose_superficielle = df_with_target.copy()
df_with_target_endometriose_superficielle['endometriose_superficielle'] = df_with_target_endometriose_superficielle['anapath.autre']

endometriose_generale = df_with_target_endometriose_generale['endometriose_generale']
endometriose_profonde = df_with_target_endometriose_profonde['endometriose_profonde']
endometriose_superficielle = df_with_target_endometriose_superficielle['endometriose_superficielle']
features_endometriose_generale = df_with_target.drop(['anapath.lusd', 'anapath.lusg', 'anapath.torus', 'anapath.autre', 'endometriose_generale'], axis=1).set_index('Anonymisation')
features_endometriose_profonde = df_with_target_endometriosis_patients.drop(['anapath.lusd', 'anapath.lusg', 'anapath.torus', 'anapath.autre', 'endometriose_generale', 'endometriose_profonde'], axis=1).set_index('Anonymisation')
features_endometriose_superficielle = df_with_target_endometriose_superficielle.drop(['anapath.lusd', 'anapath.lusg', 'anapath.torus', 'anapath.autre', 'endometriose_generale', 'endometriose_superficielle'], axis=1).set_index('Anonymisation')

## Init Metrics :
scorer = make_scorer(custom_metric, greater_is_better=True)
mix_recall = make_scorer(mix_sensi_speci, greater_is_better=True)
sensibilite = make_scorer(sensi, greater_is_better=True)
specificite = make_scorer(speci, greater_is_better=True)
scorers = { 'speci': specificite, 'sensi' : sensibilite}

# 1) Endometriosis in general vs. no endometriosis --------------------------------

# split
X_train, X_test, y_train, y_test = train_test_split(features_endometriose_generale, endometriose_generale, random_state=42, stratify=endometriose_generale)

seed = 42
tree_BL = XGBClassifier(random_state=seed, )
tree_BL.fit(X_train, y_train)
pred_BL = tree_BL.predict(X_test)
model_to_plot(tree_BL, X_test, y_test)
resultat_BL = pd.DataFrame(pd.Series(scores(y_test, pred_BL)), columns=['XGB_BL_sensi'])
if data_select == 'original':
    print(f'################ Données receuil origine ################')
else:
    print(f'################ Priorité = {data_select} ################')
print('threshold = 0,5')
print('Sensibilité : ', resultat_BL.loc['sensibilite','XGB_BL_sensi'])
print('Spécificité : ', resultat_BL.loc['specificité','XGB_BL_sensi'])
dict_ = from_model_to_bst_trhld(tree_BL, X_test, y_test)
print('_________________________________________________________')
print('Best threshold :', dict_['best_threshold'])
print('Sensibilité :', dict_['scores']['sensibilite'])
print('Spécificité :', dict_['scores']['specificité'])

# Show feature importance
xgb_fea_imp=pd.DataFrame(list(tree_BL.get_booster().get_fscore().items()),
columns=['feature','importance']).sort_values('importance', ascending=False)
print('',xgb_fea_imp)

ax = plot_importance(tree_BL, )
ax.figure.set_size_inches(10,8)

# 2) Profound or superficial endometriosis ---------------------------------------

# split
X_train, X_test, y_train, y_test = train_test_split(features_endometriose_profonde, endometriose_profonde, random_state=42, stratify=endometriose_profonde)

seed = 42
tree_BL = XGBClassifier(random_state=seed, )
tree_BL.fit(X_train, y_train)
pred_BL = tree_BL.predict(X_test)
model_to_plot(tree_BL, X_test, y_test)
resultat_BL = pd.DataFrame(pd.Series(scores(y_test, pred_BL)), columns=['XGB_BL_sensi'])
if data_select == 'original':
    print(f'################ Données receuil origine ################')
else:
    print(f'################ Priorité = {data_select} ################')
print('threshold = 0,5')
print('Sensibilité : ', resultat_BL.loc['sensibilite','XGB_BL_sensi'])
print('Spécificité : ', resultat_BL.loc['specificité','XGB_BL_sensi'])
dict_ = from_model_to_bst_trhld(tree_BL, X_test, y_test)
print('_________________________________________________________')
print('Best threshold :', dict_['best_threshold'])
print('Sensibilité :', dict_['scores']['sensibilite'])
print('Spécificité :', dict_['scores']['specificité'])

# Show feature importance
xgb_fea_imp=pd.DataFrame(list(tree_BL.get_booster().get_fscore().items()),
columns=['feature','importance']).sort_values('importance', ascending=False)
print('',xgb_fea_imp)

ax = plot_importance(tree_BL, )
ax.figure.set_size_inches(10,8)

#Superficial endometriosis (including patients with superficial & profound) vs. no endometriosis -------------------------

# split
X_train, X_test, y_train, y_test = train_test_split(features_endometriose_superficielle, endometriose_superficielle, random_state=42, stratify=endometriose_superficielle)

seed = 42
tree_BL = XGBClassifier(random_state=seed, )
tree_BL.fit(X_train, y_train)
pred_BL = tree_BL.predict(X_test)
model_to_plot(tree_BL, X_test, y_test)
resultat_BL = pd.DataFrame(pd.Series(scores(y_test, pred_BL)), columns=['XGB_BL_sensi'])
if data_select == 'original':
    print(f'################ Données receuil origine ################')
else:
    print(f'################ Priorité = {data_select} ################')
print('threshold = 0,5')
print('Sensibilité : ', resultat_BL.loc['sensibilite','XGB_BL_sensi'])
print('Spécificité : ', resultat_BL.loc['specificité','XGB_BL_sensi'])
dict_ = from_model_to_bst_trhld(tree_BL, X_test, y_test)
print('_________________________________________________________')
print('Best threshold :', dict_['best_threshold'])
print('Sensibilité :', dict_['scores']['sensibilite'])
print('Spécificité :', dict_['scores']['specificité'])