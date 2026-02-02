#!/usr/bin/env python
# coding: utf-8

# # Recherche des variables les plus importantes :
# # Travail avec le Fichier recueil complet 

# Le but de ce Notebook est de faire de la réduction dimensionelle pour essayer  de voir si des variables auraient un impact plus que d'autres sur le diagnostic. On utilise la FAMD car on a un mélange de type de variable (catégorielle, discrete, continue)

# In[25]:


# IMPORTS :

## Plots and data manipulation : 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

## Préprocessing :
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA

import prince
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder

# In[26]:


recueil_imc = pd.read_excel('Data/DATA_RAW/Recueil (1).xlsx').drop('Unnamed: 90', axis=1)

# PCA pas relevant, T-SNE peut fonctionner

# ## Reduction dimensionnelle FAMD : 

# ### Préparation des données : 

# In[27]:


rows_recueil_imc, columns_recueil_imc = recueil_imc.shape
print(f'Le fichier recueil_imc, contient {rows_recueil_imc} lignes (patientes) et {columns_recueil_imc} colonnes (pathologies)')

# On enlève les colonnes liés a la chirurgie : 
liste_colonnes_chir = ['date.chir', 'chir.macro.lusd', 'chir.macro.lusg', 'chir.macro.torus',  'chir.macro.oma', 'chir.macro.uro', 'chir.macro.dig',  'chir.macro.superf', 'resec.lusd', 'resec.lusg', 'resec.torus', 'resec.autre']
for col_to_drop in liste_colonnes_chir:
    recueil_imc = recueil_imc.drop(col_to_drop, axis=1)
# Remplace les manquantes par un np.nan
recueil_imc.replace(['Na', 'NA', 'nan', 'Nan', 'NAN'], np.nan, inplace=True)
# n_ano en Index
recueil_imc = recueil_imc.set_index('Numéro anonymat')

# In[28]:


recueil_imc.dropna(axis=0, inplace=True)
# Dernières 4 colonnes contiennent le target (endometriose)
target = recueil_imc.iloc[:,-4:].copy()
features = recueil_imc.iloc[:,:-4].copy()

# In[29]:


features.dtypes

# #### Baseline : le moins de changement possible

# In[30]:


columns_quanti = ['age', 'imc', 'g', 'p', 'sf.dsp.eva', 'sf.dsm.eva']
def tri_colonnes_features(features):
    # Tri des colonnes : 
    ## On extrait la liste des colonnes du DF :
    columns = list(features.columns)
    ## Création des listes des colonnes : 
    columns_str = []
    columns_float = []
    columns_datetime = []
    ## Boucle pour trier les colonnes :
    for col in columns:
        if features.loc[:,col].dtypes == 'object':
            columns_str.append(col)
        elif features.loc[:,col].dtypes == 'float64' or features.loc[:,col].dtypes == 'int64':
            columns_float.append(col)
        elif features.loc[:,col].dtypes == 'datetime64[ns]':
            columns_datetime.append(col)
        else:
            print(features.loc[:,col].dtypes)
            print('Colonne non triée :',col)
    
    for col in columns_quanti:
        columns_float.remove(col)
    columns_float
    return columns_str, columns_float, columns_datetime


# In[31]:


columns_str, column_cat_float, columns_datetime = tri_colonnes_features(features)

# In[32]:


def preprocess(features, baseline=True, columns_quanti = ['age', 'imc', 'g', 'p', 'sf.dsp.eva', 'sf.dsm.eva']):
    
    columns_str, column_cat_float, columns_datetime = tri_colonnes_features(features)
    features.drop(columns_datetime, axis=1, inplace=True)
    if baseline==True:
        features[column_cat_float] = features[column_cat_float].astype('object')
        features.loc[:,column_cat_float].replace([1.0,0.0], ['positif','negatif'], inplace=True)
        cat = columns_str + column_cat_float
        features_cat= features.loc[:,cat].copy()
        features_cont= features.loc[:,columns_quanti].copy()
        
    else:
        features_chir_dsptype = pd.get_dummies(features.loc[:,['chir','sf.dsp.type']]).replace([1,0], ['positif','negatif'])
        features[column_cat_float] = features[column_cat_float].astype('object')
        features.loc[:,column_cat_float] = features.loc[:,column_cat_float].replace([1.0,0.0], ['positif','negatif'])
        cat = columns_str + column_cat_float
        features_cat= pd.concat([features_chir_dsptype, features.loc[:,cat]], axis=1)
        features_cont= features.loc[:,columns_quanti].copy()

    return features_cat, features_cont

# In[33]:


#On crée les deux versions des dataframes (ONE_enc, ou inchangé)
features_cat_enc, features_cont = preprocess(features, baseline=True)
features_cat_enc_2, features_cont_2 = preprocess(features, baseline=False)
# On standardise les données cont : 
scaler = StandardScaler()
features_cont = pd.DataFrame(scaler.fit_transform(features_cont), columns=columns_quanti, index=features_cont.index)
features_cont_2 = pd.DataFrame(scaler.fit_transform(features_cont_2), columns=columns_quanti, index=features_cont_2.index)
#On concatène l'information pour la FAMD :
features_enc = pd.concat([features_cat_enc, features_cont], axis=1)
features_ONE_enc = pd.concat([features_cat_enc_2, features_cont_2], axis=1)

# In[34]:


def Binarisation(x):
    if x>1:
        x=1
    return x
endometriose = target.loc[:,['anapath.lusd','anapath.lusg','anapath.torus']].sum(axis=1).apply(lambda x: Binarisation(x))
endometriose.replace([1.0,0.0], ['positif', 'négatif'], inplace=True)

# In[36]:

"""
famd_enc_1 = prince.FAMD(
     n_components=2,
     n_iter=10,
     copy=True,
     engine='auto',
     random_state=42)

famd_enc_2 = prince.FAMD(
     n_components=6,
     n_iter=10,
     copy=True,
     engine='auto',
     random_state=42)"""
# ...existing code...
famd_enc_1 = prince.FAMD(
     n_components=2,
     n_iter=10,
     copy=True,
     engine='sklearn',  # <-- changed from 'auto' to 'sklearn'
     random_state=42)

famd_enc_2 = prince.FAMD(
     n_components=6,
     n_iter=10,
     copy=True,
     engine='sklearn',  # <-- changed from 'auto' to 'sklearn'
     random_state=42)
# ...existing code...

# In[44]:


# Ensure all columns are displayed
pd.set_option('display.max_rows', None)

# Identify categorical columns
categorical_columns = features_cat_enc.columns.tolist()  # Ensure this is a list

# Convert to category type and ensure categories are strings
for col in categorical_columns:
    features_cat_enc[col] = features_cat_enc[col].astype('category')
    features_cat_enc[col] = features_cat_enc[col].cat.rename_categories(lambda x: str(x))

# Combine categorical and numerical features
features_enc = pd.concat([features_cat_enc, features_cont], axis=1)

# Verify the data types and unique values
for col in categorical_columns:
    print(f'{col}: {features_cat_enc[col].dtype}, unique values: {features_cat_enc[col].unique()}')

# Ensure no object columns remain
for col in features_enc.select_dtypes(include='object').columns:
    features_enc[col] = features_enc[col].astype('category')

# Ensure all indexing operations use lists
features_enc = features_enc.loc[:, features_enc.columns.tolist()]

# Fit the FAMD model
famd_enc_1 = prince.FAMD(n_components=2, n_iter=10, copy=True, engine='sklearn', random_state=42) # Changed from 'auto' to 'sklearn'
famd_enc_2 = prince.FAMD(n_components=6, n_iter=10, copy=True, engine='sklearn', random_state=42) # Changed from 'auto' to 'sklearn'

famd_enc_1.fit(features_enc)
famd_enc_2.fit(features_enc)

# In[16]:


famd_enc_1.row_coordinates(features_enc)

# In[17]:

"""
df_col_corr = famd_enc_2.column_correlations(features_enc)
df_col_corr = df_col_corr.applymap(lambda x: np.abs(x))
for integer in range(0, len(df_col_corr.columns)):
    print(round(df_col_corr.sort_values(integer, ascending=False).iloc[:5,integer],3), '\n')"""
# ...existing code...
col_contrib = famd_enc_2.column_contributions_
print(col_contrib)

# In[18]:


fig, ax = plt.subplots(5, 3, figsize=(30, 30))

"""# 1ere ligne : 
famd_enc_2.plot_row_coordinates(
     features_enc,
     ax=ax[0][0],
     # figsize=(10, 10),
     x_component=0,
     y_component=1,
     labels=features.index,
     color_labels = list(endometriose),
     ellipse_outline=False,
     ellipse_fill=True,
     show_points=True)

famd_enc_2.plot_row_coordinates(
     features_enc,
     ax=ax[0][1],
     x_component=0,
     y_component=2,
     labels=features.index,
     color_labels = list(endometriose),
     ellipse_outline=False,
     ellipse_fill=True,
     show_points=True)

famd_enc_2.plot_row_coordinates(
     features_enc,
     ax=ax[0][2],
     x_component=0,
     y_component=3,
     labels=features.index,
     color_labels = list(endometriose),
     ellipse_outline=False,
     ellipse_fill=True,
     show_points=True)

famd_enc_2.plot_row_coordinates(
     features_enc,
     ax=ax[1][0],
     x_component=0,
     y_component=4,
     labels=features.index,
     color_labels = list(endometriose),
     ellipse_outline=False,
     ellipse_fill=True,
     show_points=True)


famd_enc_2.plot_row_coordinates(
     features_enc,
     ax=ax[1][1],
     x_component=0,
     y_component=5,
     labels=features.index,
     color_labels = list(endometriose),
     ellipse_outline=False,
     ellipse_fill=True,
     show_points=True)

famd_enc_2.plot_row_coordinates(
     features_enc,
     ax=ax[1][2],
     x_component=1,
     y_component=2,
     labels=features.index,
     color_labels = list(endometriose),
     ellipse_outline=False,
     ellipse_fill=True,
     show_points=True)

famd_enc_2.plot_row_coordinates(
     features_enc,
     ax=ax[2][0],
     x_component=1,
     y_component=3,
     labels=features.index,
     color_labels = list(endometriose),
     ellipse_outline=False,
     ellipse_fill=True,
     show_points=True)

famd_enc_2.plot_row_coordinates(
     features_enc,
     ax=ax[2][1],
     x_component=1,
     y_component=4,
     labels=features.index,
     color_labels = list(endometriose),
     ellipse_outline=False,
     ellipse_fill=True,
     show_points=True)

famd_enc_2.plot_row_coordinates(
     features_enc,
     ax=ax[2][2],
     x_component=1,
     y_component=5,
     labels=features.index,
     color_labels = list(endometriose),
     ellipse_outline=False,
     ellipse_fill=True,
     show_points=True)

famd_enc_2.plot_row_coordinates(
     features_enc,
     ax=ax[3][0],
     x_component=2,
     y_component=3,
     labels=features.index,
     color_labels = list(endometriose),
     ellipse_outline=False,
     ellipse_fill=True,
     show_points=True)

famd_enc_2.plot_row_coordinates(
     features_enc,
     ax=ax[3][1],
     x_component=2,
     y_component=4,
     labels=features.index,
     color_labels = list(endometriose),
     ellipse_outline=False,
     ellipse_fill=True,
     show_points=True)

famd_enc_2.plot_row_coordinates(
     features_enc,
     ax=ax[3][2],
     x_component=2,
     y_component=5,
     labels=features.index,
     color_labels = list(endometriose),
     ellipse_outline=False,
     ellipse_fill=True,
     show_points=True)

famd_enc_2.plot_row_coordinates(
     features_enc,
     ax=ax[4][0],
     x_component=3,
     y_component=4,
     labels=features.index,
     color_labels = list(endometriose),
     ellipse_outline=False,
     ellipse_fill=True,
     show_points=True)

famd_enc_2.plot_row_coordinates(
     features_enc,
     ax=ax[4][1],
     x_component=3,
     y_component=5,
     labels=features.index,
     color_labels = list(endometriose),
     ellipse_outline=False,
     ellipse_fill=True,
     show_points=True)

famd_enc_2.plot_row_coordinates(
     features_enc,
     ax=ax[4][2],
     x_component=4,
     y_component=5,
     labels=features.index,
     color_labels = list(endometriose),
     ellipse_outline=False,
     ellipse_fill=True,
     show_points=True)

plt.show()"""

# existing code...
coords = famd_enc_2.row_coordinates(features_enc)
color_map = {'positif': 'red', 'négatif': 'blue'}
colors = [color_map[val] for val in endometriose]

components = coords.columns
k = 0
for i in range(5):
    for j in range(3):
        x_comp = i
        y_comp = j + i + 1
        if y_comp >= len(components):
            continue
        ax[i][j].scatter(
            coords.iloc[:, x_comp], coords.iloc[:, y_comp],
            c=colors
        )
        ax[i][j].set_xlabel(f'Component {x_comp+1}')
        ax[i][j].set_ylabel(f'Component {y_comp+1}')
        ax[i][j].set_title(f'FAMD - Comp {x_comp+1} vs {y_comp+1}')
plt.show()

# In[19]:


famd_enc_2.row_coordinates(features_enc)

# In[20]:


"""famd_enc_2.column_correlations(features_enc).apply(lambda x: np.abs(x)).sort_values(0, ascending=False)"""
col_contrib = famd_enc_2.column_contributions_
print(col_contrib)

# In[21]:

"""
famd_ONE_2 = prince.FAMD(
     n_components=6,
     n_iter=10,
     copy=True,
     check_input=True,
     engine='auto',
     random_state=42)"""
# ...existing code...
# Ensure all categorical columns are strings ---------- ajouté par moi--------------
for col in features_ONE_enc.select_dtypes(include=['object', 'category']).columns:
    features_ONE_enc[col] = features_ONE_enc[col].astype(str)


famd_ONE_2 = prince.FAMD(
     n_components=6,
     n_iter=10,
     copy=True,
     check_input=True,
     engine='sklearn',  # <-- changed from 'auto' to 'sklearn'
     random_state=42)

# In[22]:


famd_ONE_2.fit(features_ONE_enc)

# In[23]:


famd_ONE_2.row_coordinates(features_ONE_enc)

# In[24]:


#df_famd_2_col_corr = famd_ONE_2.column_correlations(features_ONE_enc)
# existing code...
col_contrib = famd_ONE_2.column_contributions_
print(col_contrib)

# In[25]:

#existing code...
#df_famd_2_col_corr.apply(lambda x: np.abs(x)).sort_values(0, ascending=False)

# In[26]:


fig2, ax2 = plt.subplots(5, 3, figsize=(30, 30))

"""# 1ere ligne : 
famd_ONE_2.plot_row_coordinates(
     features_ONE_enc,
     ax=ax2[0][0],
     # figsize=(10, 10),
     x_component=0,
     y_component=1,
     labels=features_ONE_enc.index,
     color_labels = list(endometriose),
     ellipse_outline=False,
     ellipse_fill=True,
     show_points=True)

famd_ONE_2.plot_row_coordinates(
     features_ONE_enc,
     ax=ax2[0][1],
     x_component=0,
     y_component=2,
     labels=features_ONE_enc.index,
     color_labels = list(endometriose),
     ellipse_outline=False,
     ellipse_fill=True,
     show_points=True)

famd_ONE_2.plot_row_coordinates(
     features_ONE_enc,
     ax=ax2[0][2],
     x_component=0,
     y_component=3,
     labels=features_ONE_enc.index,
     color_labels = list(endometriose),
     ellipse_outline=False,
     ellipse_fill=True,
     show_points=True)

famd_ONE_2.plot_row_coordinates(
     features_ONE_enc,
     ax=ax2[1][0],
     x_component=0,
     y_component=4,
     labels=features_ONE_enc.index,
     color_labels = list(endometriose),
     ellipse_outline=False,
     ellipse_fill=True,
     show_points=True)


famd_ONE_2.plot_row_coordinates(
     features_ONE_enc,
     ax=ax2[1][1],
     x_component=0,
     y_component=5,
     labels=features_ONE_enc.index,
     color_labels = list(endometriose),
     ellipse_outline=False,
     ellipse_fill=True,
     show_points=True)

famd_ONE_2.plot_row_coordinates(
     features_ONE_enc,
     ax=ax2[1][2],
     x_component=1,
     y_component=2,
     labels=features_ONE_enc.index,
     color_labels = list(endometriose),
     ellipse_outline=False,
     ellipse_fill=True,
     show_points=True)

famd_ONE_2.plot_row_coordinates(
     features_ONE_enc,
     ax=ax2[2][0],
     x_component=1,
     y_component=3,
     labels=features_ONE_enc.index,
     color_labels = list(endometriose),
     ellipse_outline=False,
     ellipse_fill=True,
     show_points=True)

famd_ONE_2.plot_row_coordinates(
     features_ONE_enc,
     ax=ax2[2][1],
     x_component=1,
     y_component=4,
     labels=features_ONE_enc.index,
     color_labels = list(endometriose),
     ellipse_outline=False,
     ellipse_fill=True,
     show_points=True)

famd_ONE_2.plot_row_coordinates(
     features_ONE_enc,
     ax=ax2[2][2],
     x_component=1,
     y_component=5,
     labels=features_ONE_enc.index,
     color_labels = list(endometriose),
     ellipse_outline=False,
     ellipse_fill=True,
     show_points=True)

famd_ONE_2.plot_row_coordinates(
     features_ONE_enc,
     ax=ax2[3][0],
     x_component=2,
     y_component=3,
     labels=features_ONE_enc.index,
     color_labels = list(endometriose),
     ellipse_outline=False,
     ellipse_fill=True,
     show_points=True)

famd_ONE_2.plot_row_coordinates(
     features_ONE_enc,
     ax=ax2[3][1],
     x_component=2,
     y_component=4,
     labels=features_ONE_enc.index,
     color_labels = list(endometriose),
     ellipse_outline=False,
     ellipse_fill=True,
     show_points=True)

famd_ONE_2.plot_row_coordinates(
     features_ONE_enc,
     ax=ax2[3][2],
     x_component=2,
     y_component=5,
     labels=features_ONE_enc.index,
     color_labels = list(endometriose),
     ellipse_outline=False,
     ellipse_fill=True,
     show_points=True)

famd_ONE_2.plot_row_coordinates(
     features_ONE_enc,
     ax=ax2[4][0],
     x_component=3,
     y_component=4,
     labels=features_ONE_enc.index,
     color_labels = list(endometriose),
     ellipse_outline=False,
     ellipse_fill=True,
     show_points=True)

famd_ONE_2.plot_row_coordinates(
     features_ONE_enc,
     ax=ax2[4][1],
     x_component=3,
     y_component=5,
     labels=features_ONE_enc.index,
     color_labels = list(endometriose),
     ellipse_outline=False,
     ellipse_fill=True,
     show_points=True)

famd_ONE_2.plot_row_coordinates(
     features_ONE_enc,
     ax=ax2[4][2],
     x_component=4,
     y_component=5,
     labels=features_ONE_enc.index,
     color_labels = list(endometriose),
     ellipse_outline=False,
     ellipse_fill=True,
     show_points=True)

plt.show()"""

#existing code...
# Get the coordinates
coords = famd_ONE_2.row_coordinates(features_ONE_enc)
color_map = {'positif': 'red', 'négatif': 'blue'}
colors = [color_map[val] for val in endometriose]

components = coords.columns
for i in range(5):
    for j in range(3):
        x_comp = i
        y_comp = j + i + 1
        if y_comp >= len(components):
            continue
        ax2[i][j].scatter(
            coords.iloc[:, x_comp], coords.iloc[:, y_comp],
            c=colors
        )
        ax2[i][j].set_xlabel(f'Component {x_comp+1}')
        ax2[i][j].set_ylabel(f'Component {y_comp+1}')
        ax2[i][j].set_title(f'FAMD - Comp {x_comp+1} vs {y_comp+1}')
plt.show()
