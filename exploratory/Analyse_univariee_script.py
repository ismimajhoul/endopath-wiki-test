"""
Contexte : Ce notebook a pour interêt de trouver les variables qui vont potentiellement orienter le machine learning vers une métrique ou l'autre (sensibilité vs spécificité).
Pour ce faire on prend chaque variable et on essaie de prédire la target puis on classe les réultats en 3 listes.Auteurs: Maxime Mock, Nicolai Wolpert
"""
# Imports :

import numpy as np
import pandas as pd
import string 
import pickle

# Plot : 
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost

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
from utils.metrics_utils import rewrite_keys, rapport_metrics_decision_tree
from sklearn.model_selection import cross_validate, cross_val_score, cross_val_predict

# Utils for Thresholding 
from Opti_utils.ML_utils import find_best_threshold, from_model_to_bst_trhld, custom_metrics, scores, compare_results,find_error
from Opti_utils.ML_utils import FP, TP, FN, TN, check_corr_col, filtre_list, fit_method, check_baseline, check_FP_FN, check_best_results
from Opti_utils.ML_utils import ML_opti_recueil, univariate_ml, sensi, speci, mix_sensi_speci, custom_metric
# For statistics
import scipy.stats as stats

#Ouverture des datas -------------------------------

recueil_imc  = pd.read_excel('/home/nounou/endopath/Data/DATA_RAW/Recueil (1).xlsx').drop('Unnamed: 90', axis=1)
print(recueil_imc.shape)

# On enlève les colonnes liés a la chirurgie : 
liste_colonnes_chir = ['date.chir', 'chir.macro.lusd', 'chir.macro.lusg', 'chir.macro.torus',  'chir.macro.oma', 'chir.macro.uro', 'chir.macro.dig',  'chir.macro.superf', 'resec.lusd', 'resec.lusg', 'resec.torus', 'resec.autre']
for col_to_drop in liste_colonnes_chir:
    recueil_imc = recueil_imc.drop(col_to_drop, axis=1)
# Remplace les manquantes par un np.nan
recueil_imc.replace(['Na', 'NA', 'nan', 'Nan', 'NAN'], np.nan, inplace=True)
# n_ano en Index
recueil_imc = recueil_imc.set_index('Numéro anonymat')
print(recueil_imc.shape)

#Statistiques des variables continues -------------------------------------

print(recueil_imc.loc[:, ['age', 'imc', 'g', 'p']].head(4))

#Split des features et des targets --------------------------

def Binarisation(x):
    if x>1:
        x=1
    return x

# recueil_imc.dropna(axis=0, inplace=True)
target = recueil_imc.iloc[:,-4:].copy()
features = recueil_imc.iloc[:,:-4].copy()

endometriose = target.loc[:,['anapath.lusd','anapath.lusg','anapath.torus']].sum(axis=1).apply(lambda x: Binarisation(x))

recueil_imc_endo = recueil_imc.copy()
recueil_imc_endo['endometriose'] = endometriose
recueil_imc_endo['endometriose'] = recueil_imc_endo['endometriose'].replace(0, 'Négatif').replace(1, 'Positif')
print(recueil_imc_endo.shape)

#Proportion des missing values par feature ---------------------------

percent_missing = recueil_imc.isnull().sum() * 100 / recueil_imc.shape[0]
missing_value_df = pd.DataFrame({'column_name': recueil_imc.columns,
                                 'percent_missing': percent_missing})
missing_value_df = missing_value_df.sort_values(by='percent_missing', ascending=False)
print(missing_value_df)

# Missing values in the 9 features of interest
features_of_interest = ['atcd.endo', 'irm.lusg', 'tv.douloureux', 'irm.externe', 'sf.dig.diarrhee', 'echo.lusg', 'echo.lusd', 'ef.hormone.dpc', 'effet.hormone']
print(missing_value_df.loc[missing_value_df.index.isin(features_of_interest)])

"""
plt.figure(figsize=(30, 5))
sns.barplot(x=missing_value_df.loc[missing_value_df.percent_missing>0].index, y=missing_value_df.loc[missing_value_df.percent_missing>0, 'percent_missing'].values)
plt.title('Percent Missing')
plt.ylabel('Percentage missing values', fontsize=12)
plt.xlabel('Features', fontsize=12)
plt.xticks(rotation=60)
plt.show()
"""

#Répartition des diagnostics d\'endométriose dans la cohorte -----------------------------
endometriose_complete = target.loc[:,['anapath.lusd','anapath.lusg','anapath.torus', 'anapath.autre']].sum(axis=1).apply(lambda x: Binarisation(x))
endometriose_complete = endometriose_complete.replace(0, 'Négatif').replace(1, 'Positif')

"""
plt.figure(figsize=(5,6))
ax = sns.histplot(data=endometriose_complete, shrink = 0.3, color='#00F4CB')
ax.bar_label(ax.containers[0])
ax.set_xlabel('Diagnostic', fontsize=14)
ax.set_ylabel('Nombre', fontsize=14)
ax.set_title('Répartition des diagnostics d\'endométriose dans la cohorte', pad=25, fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()
"""

serie = recueil_imc_endo['endometriose']
"""
plt.figure(figsize=(5,6))
ax = sns.histplot(data=serie, shrink=0.3, color='#00B2A2')
ax.bar_label(ax.containers[0])
ax.set_xlabel('Diagnostic', fontsize=14)
ax.set_ylabel('Nombre', fontsize=14)
ax.set_title('Répartition des diagnostics d\'endométriose profonde dans la cohorte', pad=25, fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()
"""

#Répartition des symtômes dans la cohorte-------------------------------

# Montre répartition de présence ou absence de symptômes d'intérêt
features_of_interest = ['atcd.endo', 'irm.lusg', 'tv.douloureux', 'irm.externe', 'sf.dig.diarrhee', 'echo.lusg', 'echo.lusd', 'ef.hormone.dpc', 'effet.hormone']

fig, axs = plt.subplots(figsize=(10, 10), nrows=3, ncols=3)
axs = axs.flatten()

for ifeat, feat in enumerate(features_of_interest):
    symptom_complete = recueil_imc_endo[feat].replace(0, 'Absent').replace(1, 'Présent')
    symptom_complete = pd.Categorical(symptom_complete, categories=['Absent', 'Présent'], ordered=True)  # Set the category order
    sns.histplot(data=symptom_complete, shrink = 0.3, color='#00F4CB', ax=axs[ifeat])
    axs[ifeat].bar_label(axs[ifeat].containers[0])
    axs[ifeat].set_xlabel('Diagnostic', fontsize=10)
    axs[ifeat].set_ylabel('Nombre', fontsize=10)
    axs[ifeat].set_title(feat, pad=25, fontsize=14)
#plt.tight_layout()
#plt.show()

#Calcul du % d'info manquantes ----------------------------------------------

n_nan = features.isna().sum().sum()
n_cellule = features.shape[0]*features.shape[1]
n_info = n_cellule - n_nan
n_cellule_dropna = features.dropna().shape[0]*features.dropna().shape[1]
print(f'% d\'informations manquantes dans le jeu de données :{round(n_nan/n_cellule*100,2)}')
print(f'% d\'informations avec un inputer : {round(n_cellule/n_info*100,2)}')
print(f'% d\'informations en supprimant les lignes : {round(n_cellule_dropna/n_info*100,2)}')

recueil_imc_endo_pos = recueil_imc_endo.loc[recueil_imc_endo.loc[:,'endometriose']=="Positif"].copy()
recueil_imc_endo_neg = recueil_imc_endo.loc[recueil_imc_endo.loc[:,'endometriose']=="Négatif"].copy()

print(recueil_imc_endo)


min_ = int(np.min(recueil_imc_endo['age'].dropna().unique()))
max_ = int(np.max(recueil_imc_endo['age'].dropna().unique()))
bins = list(range(min_,max_,1))

"""
plt.figure(figsize=(12,7))
sns.histplot(data=recueil_imc_endo, x='age', bins=bins, color='salmon')
plt.title('Distribution de la cohorte selon âge', fontsize=28)
plt.xlabel("Age", fontsize=20)
plt.ylabel('Nombre de patientes', fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
# plt.legend(fontsize=12)
#plt.savefig('Prix=f(Age).png')
# plt.text(87, 0.5, 'Source : dataset "Titanic" fourni par Kaggle')
plt.show()
"""

min_ = int(np.min(recueil_imc_endo['imc'].dropna().unique()))
max_ = int(np.max(recueil_imc_endo['imc'].dropna().unique()))
bins = list(range(min_,max_))
"""
plt.figure(figsize=(12,7))

sns.histplot(recueil_imc.loc[:,'imc'], bins=bins, color='royalblue')
plt.title('Distribution de la cohorte selon leur IMC', fontsize=28)
plt.xlabel("IMC", fontsize=20)
plt.ylabel('Nombre de patientes', fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
# plt.legend(fontsize=20)

#plt.savefig('Prix=f(Age).png')
# plt.text(87, 0.5, 'Source : dataset "Titanic" fourni par Kaggle')
plt.show()
"""


min_pos = int(np.min(recueil_imc_endo_pos['imc'].dropna().unique()))
max_pos = int(np.max(recueil_imc_endo_pos['imc'].dropna().unique()))
bins_pos = list(range(min_pos, max_pos))

min_neg = int(np.min(recueil_imc_endo_neg['imc'].dropna().unique()))
max_neg = int(np.max(recueil_imc_endo_neg['imc'].dropna().unique()))
bins_neg = list(range(min_neg, max_neg))
"""
palette ={0.0: "yellow", 1.0: "royalblue"}

fig, ax = plt.subplots(1, 1, figsize=(12, 7))

sns.histplot(ax=ax, data=recueil_imc_endo_pos, x='imc', palette=palette, stat='count', bins=bins_pos, alpha=0.5, kde=True, label='Positif', color='royalblue')
sns.histplot(ax=ax, data=recueil_imc_endo_neg, x='imc', palette=palette, stat='count', bins=bins_neg, alpha=0.5, kde=True, label='Negatif', color='gold')

plt.title('Distribution de la cohorte selon leur IMC', fontsize=28)
plt.xlabel("IMC", fontsize=20)
plt.ylabel('Nombre de patientes', fontsize=20)
plt.legend(title='Diagnostic :',loc='best')
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.legend(fontsize=20)
plt.show()
"""

max_g = int(np.max(list(recueil_imc.loc[:,'g'].dropna().unique())))
min_g = int(np.min(list(recueil_imc.loc[:,'g'].dropna().unique())))
bins = list(range(min_g, max_g))
palette ={"Positif": "mediumpurple", "Négatif": "darkorange"}
"""
plt.figure(figsize=(12,7))

sns.histplot(data=recueil_imc_endo, x='g', hue='endometriose', bins=bins, palette=palette, alpha=0.8, kde=True, hue_order=["Négatif", "Positif"])
plt.title('Distribution de la cohorte selon leur gestité', fontsize=20)
plt.xlabel("Gestité", fontsize=15)
plt.ylabel('Nombre de patientes', fontsize=15)
# plt.legend(fontsize=12)
#plt.savefig('Prix=f(Age).png')
plt.show()
"""

max_g = int(np.max(list(recueil_imc.loc[:,'g'].dropna().unique())))
min_g = int(np.min(list(recueil_imc.loc[:,'g'].dropna().unique())))
bins = list(range(min_g, max_g))
plt.figure(figsize=(12,7))
"""
sns.histplot(recueil_imc.loc[:,'g'], bins=bins, color='darkorange')
plt.title('Distribution de la cohorte selon leur gestité', fontsize=20)
plt.xlabel("Gestité", fontsize=15)
plt.ylabel('Nombre de patientes', fontsize=15)
# plt.legend(fontsize=12)
#plt.savefig('Prix=f(Age).png')
# plt.text(87, 0.5, 'Source : dataset "Titanic" fourni par Kaggle')
plt.show()
"""

max_p = int(np.max(list(recueil_imc.loc[:,'p'].dropna().unique())))
min_p = int(np.min(list(recueil_imc.loc[:,'p'].dropna().unique())))
bins = list(range(min_p, max_p))
"""
plt.figure(figsize=(12,7))

sns.histplot(recueil_imc.loc[:,'p'], bins=bins, color='limegreen')
plt.title('Distribution de la cohorte selon leur parité', fontsize=16)
plt.xlabel("Parité", fontsize=15)
plt.ylabel('Nombre de patientes', fontsize=15)

plt.show()
"""

max_p = int(np.max(list(recueil_imc.loc[:,'p'].dropna().unique())))
min_p = int(np.min(list(recueil_imc.loc[:,'p'].dropna().unique())))
bins = list(range(min_p, max_p))
palette ={"Négatif": "limegreen", "Positif": "red"}
"""

plt.figure(figsize=(12,7))

sns.histplot(data=recueil_imc_endo, x='p', hue='endometriose', bins=bins, palette=palette, alpha=0.8, kde=True, hue_order=["Négatif", "Positif"])
plt.title('Distribution de la cohorte selon leur parité', fontsize=20)
plt.xlabel("Parité", fontsize=15)
plt.ylabel('Nombre de patientes', fontsize=15)

plt.show()
"""

max_p = int(np.max(list(recueil_imc.loc[:,'sf.dsp.eva'].dropna().unique())))
min_p = int(np.min(list(recueil_imc.loc[:,'sf.dsp.eva'].dropna().unique())))
bins = list(range(min_p, max_p))
"""
plt.figure(figsize=(12,7))
sns.histplot(data=recueil_imc_endo, x='sf.dsp.eva', bins=bins, color = 'lightcoral', kde=True)
plt.title("Distribution de l'évaluation de la douleur pendant les rapports intimes", fontsize=20)
plt.xlabel("Evaluation de la douleur", fontsize=15)
plt.ylabel('Nombre de patientes', fontsize=15)

plt.show()
"""

max_p = int(np.max(list(recueil_imc.loc[:,'sf.dsp.eva'].dropna().unique())))
min_p = int(np.min(list(recueil_imc.loc[:,'sf.dsp.eva'].dropna().unique())))
bins = list(range(min_p, max_p))
palette ={"Négatif": "darkblue", "Positif": "lightcoral"}
"""
plt.figure(figsize=(12,7))
sns.histplot(data=recueil_imc_endo, x='sf.dsp.eva', hue='endometriose', bins=bins, palette=palette, alpha=0.4, kde=True, hue_order=["Négatif", "Positif"])
plt.title("Distribution de l'évaluation de la douleur pendant les rapports intimes", fontsize=20)
plt.xlabel("Evaluation de la douleur", fontsize=15)
plt.ylabel('Nombre de patientes', fontsize=15)

plt.show()
"""

max_p = int(np.max(list(recueil_imc.loc[:,'sf.dsm.eva'].dropna().unique())))
min_p = int(np.min(list(recueil_imc.loc[:,'sf.dsm.eva'].dropna().unique())))
bins = list(range(min_p, max_p))
"""
plt.figure(figsize=(12,7))

sns.histplot(data=recueil_imc_endo, x='sf.dsm.eva', bins=bins, color='navajowhite', kde=True)
plt.title('Distribution de l\'évaluation de la douleur des menstruations', fontsize=20)
plt.xlabel("Evaluation de la douleur", fontsize=15)
plt.ylabel('Nombre de patientes', fontsize=15)

plt.show()
"""

max_p = int(np.max(list(recueil_imc.loc[:,'sf.dsm.eva'].dropna().unique())))
min_p = int(np.min(list(recueil_imc.loc[:,'sf.dsm.eva'].dropna().unique())))
bins = list(range(min_p, max_p))
palette ={"Négatif": "navajowhite", "Positif": "saddlebrown"}

"""
plt.figure(figsize=(12,7))

sns.histplot(data=recueil_imc_endo, x='sf.dsm.eva', hue='endometriose', bins=bins, palette=palette, alpha=0.8, kde=True, hue_order=["Négatif", "Positif"])
plt.title('Distribution de l\'évaluation de la douleur des menstruations', fontsize=20)
plt.xlabel("Evaluation de la douleur", fontsize=15)
plt.ylabel('Nombre de patientes', fontsize=15)

plt.show()
"""
"""
plt.figure(figsize=(6,4))
plt.style.use('default')

mask = np.triu(np.ones_like(recueil_imc.loc[:, ['age', 'imc', 'g', 'p', 'sf.dsp.eva','sf.dsm.eva']].corr()))
sns.heatmap(recueil_imc.loc[:, ['age', 'imc', 'g', 'p', 'sf.dsp.eva','sf.dsm.eva']].corr(),cmap="rocket_r", annot=True, mask=mask, cbar=True, vmin=-1, vmax=1)
plt.yticks(rotation=0)
plt.title("Correlation entre les Variables continues", size=14, weight="bold")
plt.show()
"""

#Statistiques des variables catégorielles -----------------
#Préparation des données

features_chir_ONE = pd.get_dummies(features.loc[:,'chir'], prefix='chir')
features_dsptype_ONE = pd.get_dummies(features.loc[:,'sf.dsp.type'].replace(0, 'aucun'), prefix='dsp.type')
features_enc = pd.concat([features.drop('chir', axis=1).drop('sf.dsp.type', axis=1), features_chir_ONE, features_dsptype_ONE], axis=1)

print(features_enc.describe())

# split 
X_train, X_test, y_train, y_test = train_test_split(features_enc, endometriose, random_state=42, stratify=endometriose)

# split 
X_train2, X_test2, y_train2, y_test2 = train_test_split(features_enc, endometriose, random_state=42, stratify=endometriose)

scorer = make_scorer(custom_metric, greater_is_better=True)
mix_recall = make_scorer(mix_sensi_speci, greater_is_better=True)
sensibilite = make_scorer(sensi, greater_is_better=True)
specificite = make_scorer(speci, greater_is_better=True)
scorers = { 'speci': specificite, 'sensi' : sensibilite}

#Préparation du ML univarié ------------------------------

resultats = univariate_ml(X_train, X_test, y_train, y_test, scorers)
print(resultats)

#Résultats --------------------------

# Choix des seuils :
seuils_speci = [0.8, 0.5]
seuils_sensi = [0.8, 0.5]

# Constitutions des listes : 
liste_speci = list(resultats.loc[:,resultats.loc['speci',:] >=seuils_speci[0]].columns)
liste_sensi = list(resultats.loc[:,resultats.loc['sensi',:] >=seuils_sensi[0]].columns)
liste_mixte = list(resultats.loc[:,(resultats.loc['speci',:] >=seuils_speci[1])&(resultats.loc['sensi',:] >=seuils_sensi[1])&(resultats.loc['speci',:] <=seuils_speci[0])&(resultats.loc['sensi',:] <=seuils_sensi[0])].columns)

# Résultats : 
print(f'Liste des {len(liste_speci)} colonnes ou la spécificité est au dessus du seuil de {seuils_speci[0]}\n')
print(liste_speci)
print(f'\nListe des {len(liste_sensi)} colonnes ou la sensibilité est au dessus du seuil de {seuils_sensi[0]}\n')
print(liste_sensi)
print(f'\nListe des {len(liste_mixte)} colonnes ou la sensibilité ET la spécificité est respectivement au dessus du seuil de {seuils_sensi[1]} et {seuils_speci[1]}\n')
print(liste_mixte)

def check_corr_col(col, best_set):
    boolean=True
    # g and p are correlated : 
    if col == 'g' and 'p' in best_set:
        boolean=False       
    elif col == 'p'and 'g' in best_set:
        boolean=False
    
    # atcd.endo and atcd.chir.endo are correlated : 
    elif col == 'atcd.endo'and 'atcd.chir.endo' in best_set:
        boolean=False
    elif col == 'atcd.chir.endo'and 'atcd.endo' in best_set:
        boolean=False 
        
    # atcd.pma and atcd.infertilite are correlated :
    elif col == 'atcd.pma'and 'atcd.infertilite' in best_set:
        boolean=False
    elif col == 'atcd.infertilite'and 'atcd.pma' in best_set:
        boolean=False
        
    # sf.dsm.eva and sf.dsm are correlated :
    elif col == 'sf.dsm.eva'and 'sf.dsm' in best_set:
        boolean=False
    elif col == 'sf.dsm'and 'sf.dsm.eva' in best_set:
        boolean=False
        
    # sf.dsp.eva and sf.dsp are correlated :
    elif col == 'sf.dsp.eva'and 'sf.dsp' in best_set:
        boolean=False
    elif col == 'sf.dsp'and 'sf.dsp.eva' in best_set:
        boolean=False
               
    # g and p are correlated :
    elif col == 'ef.hormone.dsm'and 'effet.hormone' in best_set:
        boolean=False
    elif col == 'effet.hormone'and 'ef.hormone.dsm' in best_set:
        boolean=False

    # echo.oma and irm.oma are correlated :
    elif col == 'echo.oma'and 'irm.oma' in best_set:
        boolean=False   
    elif col == 'irm.oma'and 'echo.oma' in best_set:
        boolean=False
    
    # irm.xr and irm.externe are correlated :
    elif col == 'irm.xr'and 'irm.externe' in best_set:
        boolean=False
    elif col == 'irm.externe'and 'irm.xr' in best_set:
        boolean=False

    return boolean

def clean_lists(liste):
    '''
    Filters out features that are correlated with other features defined manually
    '''
    liste_temp = liste.copy()
    for col in liste:
        if not check_corr_col(col, liste_temp):
            liste_temp.remove(col)
    return liste_temp


"""
liste_col_sensi = clean_lists(liste_col_sensi)
liste_col_speci = clean_lists(liste_col_speci)
liste_col_mixte = clean_lists(liste_col_mixte)

/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\
ATTENTION: liste_col_sensi, liste_col_speci et liste_col_mixte ne sont pas définis
"""

#Save Results -------------------------------------

"""
dictionnary = { 'specifite':liste_col_speci,
                'sensibilite':liste_col_sensi,
                'moyenne':liste_col_mixte
              }
with open('../../Data/Generate/data.pkl', 'wb') as f:
    pickle.dump(dictionnary, f)

"""

#Ces listes pourront être réutilisées dans le cadre du feature engineering pour essayer d'améliorer les résultats de machine learning

def FN(to_test, ref):
    boolean = False
    if to_test > ref:
        boolean = True
    return boolean

def TP(to_test, ref):
    boolean = False
    if to_test == 1 and ref==1:
        boolean = True
    return boolean

def TN(to_test, ref):
    boolean = False
    if to_test == 0 and ref== 0:
        boolean = True
    return boolean

def FP(to_test, ref):
    boolean = False
    if to_test <ref:
        boolean = True
    return boolean


def compare_results(series_to_test, series_reference):
    liste=[]
    for idx in range(0,len(series_to_test)):
        to_test = series_to_test.iloc[idx]
        ref = series_reference.iloc[idx]
        if FP(to_test, ref):
            liste.append('FP')
        elif FN(to_test, ref):
            liste.append('FN')
        elif TN(to_test, ref):
            liste.append('TN')
        elif TP(to_test, ref):
            liste.append('TP')
    return pd.Series(liste)

#Lien statistiques des features avec l'endométriose ------------------------------------------------

### Show for all quantitative features

features_quantitative = ['age', 'imc', 'sf.dsp.eva', 'sf.dsm.eva']

fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(25, 5))
axes = axes.flatten()

for i, feature in enumerate(features_quantitative):

    # Perform t-test
    feature_with_endo = recueil_imc_endo[recueil_imc_endo['endometriose'] == 'Positif'][feature].dropna()
    feature_without_endo = recueil_imc_endo[recueil_imc_endo['endometriose'] == 'Négatif'][feature].dropna()
    results_ttest = stats.ttest_ind(feature_with_endo, feature_without_endo)

    # Determine p-value text and color
    if results_ttest.pvalue < 0.001:
        show_p = 'p < 0.001'
    elif results_ttest.pvalue < 0.01:
        show_p = 'p < 0.01'
    else:
        show_p = 'p = ' + str(round(results_ttest.pvalue, 2))

    color_text = 'red' if results_ttest.pvalue < 0.05 else 'black'

    custom_palette = {'Positif': "maroon", 'Négatif': "darkgreen"}
    sns.barplot(
        data=recueil_imc_endo, hue='endometriose', y=feature,
        palette=custom_palette, alpha=.6, ax=axes[i], errorbar="sd", hue_order=['Positif', 'Négatif']
    )
    axes[i].set_title(f'{feature}: t={round(results_ttest.statistic, 2)}, {show_p}', color=color_text)
    axes[i].set_xlabel('Endométriose')
    axes[i].set_ylabel(feature)
    handles, labels = axes[i].get_legend_handles_labels()

#plt.show()


### For all categorical features

features_categorical = ['g', 'p', 'atcd.endo', 'irm.lusg', 'tv.douloureux', 'irm.externe', 'sf.dig.diarrhee', 'echo.lusg', 'echo.lusd', 'ef.hormone.dpc', 'effet.hormone']  # features of interest
features_categorical = list(recueil_imc_endo.drop(features_quantitative + ['chir', 'anapath.lusd', 'anapath.lusg', 'anapath.torus', 'anapath.autre', 'endometriose'], axis=1).columns) # all features

ncols = 4
nrows = int(np.ceil(len(features_categorical) / ncols))
if len(features_categorical) <= 12:
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 15))
else:
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 50))
axes = axes.flatten()
features_significant = []
features_trend = []
for i, feature in enumerate(features_categorical):

    recueil_plot = recueil_imc_endo.copy()
    if feature not in ['g', 'p']:
        recueil_plot[feature] = recueil_plot[feature].replace({0: 'Négatif', 1: 'Positif'})
    ct_table_ind=pd.crosstab(recueil_imc_endo[feature],recueil_imc_endo["endometriose"])
    chi2_stat, p, dof, expected = stats.chi2_contingency(ct_table_ind)

    # Determine p-value text and color
    if p < 0.001:
        show_p = 'p < 0.001'
    elif p < 0.01:
        show_p = 'p < 0.01'
    else:
        show_p = 'p = ' + str(round(p, 2))

    if p < 0.05:
        features_significant = features_significant + [feature]
        color_text = 'red'
    elif p < 0.1:
        features_trend = features_trend + [feature]
        color_text = 'black'
    else:
        color_text = 'black'

    # Create the plot
    sns.histplot(data=recueil_plot, x='endometriose', hue=feature, hue_order=['Négatif', 'Positif'], multiple='stack', palette='viridis', ax=axes[i])

    # Customize the plot
    axes[i].set_title(f'{feature}: chi2_stat={round(chi2_stat, 2)}, {show_p}', color=color_text)
    axes[i].set_xlabel('endometriose')
    axes[i].set_ylabel('Count')
    #axes[i].set_xticks([0, 1], ['sans endometriose', 'avec endometriose'])
for ax in axes:
    if not ax.has_data():  # Check if the axis has any data
        fig.delaxes(ax)    # Remove the empty axis

#plt.tight_layout()
#plt.show()

### Show features statistically significant

ncols = 4
nrows = int(np.ceil(len(features_significant) / ncols))
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 10))
axes = axes.flatten()
for i, feature in enumerate(features_significant):

    recueil_plot = recueil_imc_endo.copy()
    if feature not in ['g', 'p']:
        recueil_plot[feature] = recueil_plot[feature].replace({0: 'Négatif', 1: 'Positif'})
    ct_table_ind=pd.crosstab(recueil_imc_endo[feature],recueil_imc_endo["endometriose"])
    chi2_stat, p, dof, expected = stats.chi2_contingency(ct_table_ind)

    # Determine p-value text and color
    if p < 0.001:
        show_p = 'p < 0.001'
    elif p < 0.01:
        show_p = 'p < 0.01'
    else:
        show_p = 'p = ' + str(round(p, 2))

    color_text = 'red' if p < 0.05 else 'black'

    # Create the plot
    #sns.histplot(data=recueil_plot, x='endométriose', hue=feature, hue_order=['Négatif', 'Positif'], multiple='stack', palette='viridis', ax=axes[i])  ERROR : ValueError: Could not interpret value `endométriose` for `x`. An entry with this name does not appear in `data`

    # Customize the plot
    axes[i].set_title(f'{feature}: chi2_stat={round(chi2_stat, 2)}, {show_p}', color=color_text)
    axes[i].set_xlabel('endométriose')
    axes[i].set_ylabel('Count')
    #axes[i].set_xticks([0, 1], ['sans endometriose', 'avec endometriose'])
for ax in axes:
    if not ax.has_data():  # Check if the axis has any data
        fig.delaxes(ax)    # Remove the empty axis

#plt.tight_layout()
#plt.show()


### Show features with trend

ncols = 4
nrows = int(np.ceil(len(features_trend) / ncols))
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 10))
axes = axes.flatten()
for i, feature in enumerate(features_trend):

    recueil_plot = recueil_imc_endo.copy()
    if feature not in ['g', 'p']:
        recueil_plot[feature] = recueil_plot[feature].replace({0: 'Négatif', 1: 'Positif'})
    ct_table_ind=pd.crosstab(recueil_imc_endo[feature],recueil_imc_endo["endometriose"])
    chi2_stat, p, dof, expected = stats.chi2_contingency(ct_table_ind)

    # Determine p-value text and color
    if p < 0.001:
        show_p = 'p < 0.001'
    elif p < 0.01:
        show_p = 'p < 0.01'
    else:
        show_p = 'p = ' + str(round(p, 2))

    color_text = 'red' if p < 0.05 else 'black'

    # Create the plot
    sns.histplot(data=recueil_plot, x='endometriose', hue=feature, hue_order=['Négatif', 'Positif'], multiple='stack', palette='viridis', ax=axes[i])

    # Customize the plot
    axes[i].set_title(f'{feature}: chi2_stat={round(chi2_stat, 2)}, {show_p}', color=color_text)
    axes[i].set_xlabel('endometriose')
    axes[i].set_ylabel('Count')
    #axes[i].set_xticks([0, 1], ['sans endometriose', 'avec endometriose'])
for ax in axes:
    if not ax.has_data():  # Check if the axis has any data
        fig.delaxes(ax)    # Remove the empty axis

#plt.tight_layout()
#plt.show()