#!/usr/bin/env python
# coding: utf-8

# # test_chi2
# 
# Notebook to test for features most correlated with endometriosis
# 
# Author: Maxime Mock

# In[6]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import SimpleImputer

# Utils for classification :
import xgboost
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier, XGBRFClassifier

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, f_classif, mutual_info_classif

#   
# ## __Tests <font face = 'symbol'>c</font> <sup>2</sup> :__  
# 
# Permet de vérifier si un échantillon d'une variable aléatoire, donne des observations comparables à celle d'une loi de probabilité P définie a priori dont on pense, pour des raisons théoriques ou pratiques, qu'elle devrait être la loi de Y.
# L'hypothèse nulle HO est donc : la variable aléatoire suit la loi de Proba.
# En termes de valeur-p, l'hypothèse nulle (l'observation est suffisamment proche de la théorie) est généralement rejetée lorsque p ≤ 0,05.
# INDEPENDANCE des deux colonnes
# ## __Tests F
# 
# si pv<0.05 alors les 2 variables suivent la même loi normale de proba donc même espérance et ecart type (à confirmer)
# 

# # Open data : 

# In[7]:


recueil_imc  = pd.read_excel('Data/DATA_RAW/Recueil (1).xlsx').drop('Unnamed: 90', axis=1)
recueil_imc.shape

# In[8]:


# On enlève les colonnes liés a la chirurgie : 
liste_colonnes_chir = ['date.chir', 'chir.macro.lusd', 'chir.macro.lusg', 'chir.macro.torus',  'chir.macro.oma', 'chir.macro.uro', 'chir.macro.dig',  'chir.macro.superf', 'resec.lusd', 'resec.lusg', 'resec.torus', 'resec.autre']
for col_to_drop in liste_colonnes_chir:
    recueil_imc = recueil_imc.drop(col_to_drop, axis=1)
# Remplace les manquantes par un np.nan
recueil_imc.replace(['Na', 'NA', 'nan', 'Nan', 'NAN'], np.nan, inplace=True)
# n_ano en Index
recueil_imc = recueil_imc.set_index('Numéro anonymat')
recueil_imc.shape

# In[9]:


# recueil_imc.dropna(axis=0, inplace=True)
target = recueil_imc.iloc[:,-4:].copy()
features = recueil_imc.iloc[:,:-4].copy()

def Binarisation(x):
    if x>1:
        x=1
    return x
endometriose = target.loc[:,['anapath.lusd','anapath.lusg','anapath.torus']].sum(axis=1).apply(lambda x: Binarisation(x))

features_chir_ONE = pd.get_dummies(features.loc[:,'chir'], prefix='chir')
features_dsptype_ONE = pd.get_dummies(features.loc[:,'sf.dsp.type'].replace(0, 'aucun'), prefix='dsp.type')
features_enc = pd.concat([features.drop('chir', axis=1).drop('sf.dsp.type', axis=1), features_chir_ONE, features_dsptype_ONE], axis=1)

# In[10]:


features_enc.shape

# # Sans imputer : 

# ## Préparation des data :

# In[11]:


features_enc_dropna = features_enc.dropna(how='any', axis=0)
endometriose_dropna = endometriose.loc[list(features_enc_dropna.index)]

# ### Chi2 test : 

# In[12]:


#apply SelectKBest class to extract top 10 best features
bestfeatures = SelectKBest(score_func=chi2, k=20)
fit = bestfeatures.fit(features_enc_dropna,endometriose_dropna)
dfscores = pd.DataFrame(fit.scores_)
df_pv = pd.DataFrame(fit.pvalues_)
dfcolumns = pd.DataFrame(features_enc_dropna.columns)
#concat two dataframes for better visualization 
featureScores = pd.concat([dfcolumns,dfscores,df_pv],axis=1)
featureScores.columns = ['Features','Score','p values']  # naming the dataframe columns
print(featureScores.nlargest(20, 'Score'))  # print 20 best features

# In[13]:


featureScores.loc[featureScores.loc[:,'p values']<=0.05]

# In[58]:


featureScores.reset_index(drop=True, inplace=True)
featureScores.to_excel('Data/chi2.xlsx')

# In[15]:


featureScores.sort_values('Score', inplace=True, ascending=False)


fig, ax = plt.subplots(1,1,figsize=(20, 9))


ax.set_title('Résultats des scores du test chi2 sans inputer des variables de Recueil', fontsize=18)
ax.bar(featureScores['Features'], height=featureScores['Score'], width=0.5, color='deepskyblue')
# ax.set_ylim(0,1)
ax.tick_params(axis='x',rotation=90)



# In[16]:



fig, ax = plt.subplots(1,1,figsize=(20, 9))


ax.set_title('Top10 des variables issues du chi2 sans inputer', fontsize=18)
ax.bar(featureScores['Features'].iloc[:10], height=featureScores['Score'].iloc[:10], width=0.5, color='deepskyblue')
# ax.set_ylim(0,1)
ax.tick_params(axis='x',rotation=90)



# ### f_classif test : 

# In[17]:


#apply SelectKBest class to extract top 10 best features
bestfeatures_f_drop = SelectKBest(score_func=f_classif, k=20)
fit_f_drop = bestfeatures_f_drop.fit(features_enc_dropna,endometriose_dropna)
dfscores_f_drop = pd.DataFrame(fit_f_drop.scores_)
pvalues_f_drop = pd.DataFrame(fit_f_drop.pvalues_)
dfcolumns_f_drop = pd.DataFrame(features_enc_dropna.columns)
#concat two dataframes for better visualization 
featureScores_f_drop = pd.concat([dfcolumns_f_drop,dfscores_f_drop,pvalues_f_drop],axis=1)
featureScores_f_drop.columns = ['Features','Score','p values']  #naming the dataframe columns
featureScores_f_drop.nlargest(10, 'Score')

# In[18]:


featureScores_f_drop.reset_index(drop=True, inplace=True)
featureScores_f_drop.to_excel('Data/f_test.xlsx')

# In[19]:


featureScores.sort_values('Score', inplace=True, ascending=False)

fig, ax = plt.subplots(1,1,figsize=(20, 9))
ax.set_title('Résultats des scores du F-test sans inputer des variables de Recueil', fontsize=18)
ax.bar(featureScores['Features'], height=featureScores['Score'], width=0.5, color='mediumslateblue')
# ax.set_ylim(0,1)
ax.tick_params(axis='x',rotation=90)


# In[20]:


fig, ax = plt.subplots(1,1,figsize=(20, 9))

ax.set_title('Top10 des variables issues du F-Test sans inputer', fontsize=18)
ax.bar(featureScores['Features'].iloc[:10], height=featureScores['Score'].iloc[:10], width=0.5, color='mediumslateblue')
# ax.set_ylim(0,1)
ax.tick_params(axis='x',rotation=90)

# ### mutual_info_classif test : 

# In[21]:


#apply SelectKBest class to extract top 10 best features
bestfeatures = SelectKBest(score_func=mutual_info_classif, k=20)
fit = bestfeatures.fit(features_enc_dropna,endometriose_dropna)
dfscores = pd.DataFrame(fit.scores_)
pvalues = pd.DataFrame(fit.pvalues_)
dfcolumns = pd.DataFrame(features_enc_dropna.columns)
#concat two dataframes for better visualization 
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Features','Score']  #naming the dataframe columns
print(featureScores.nlargest(10,'Score'))  #print 20 best features

# In[22]:


featureScores.sort_values('Score', inplace=True, ascending=False)

fig, ax = plt.subplots(1,1,figsize=(20, 9))
ax.set_title('Résultats des scores du test d\'information mutuelle sans inputer des variables de Recueil', fontsize=18)
ax.bar(featureScores['Features'], height=featureScores['Score'], width=0.5, color='mediumblue')
# ax.set_ylim(0,1)
ax.tick_params(axis='x',rotation=90)

# In[23]:


fig, ax = plt.subplots(1,1,figsize=(20, 9))

ax.set_title('Top10 des variables issues de l\'information mutuelle sans inputer', fontsize=18)
ax.bar(featureScores['Features'].iloc[:10], height=featureScores['Score'].iloc[:10], width=0.5, color='mediumblue')
# ax.set_ylim(0,1)
ax.tick_params(axis='x',rotation=90)

# # Avec imputer

# ## préparation des données : 

# In[24]:


imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
imp_mode = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
imp_median = SimpleImputer(missing_values=np.nan, strategy='median')

features_enc_mean = imp_mean.fit_transform(features_enc)
features_enc_mode = imp_mode.fit_transform(features_enc)
features_enc_median = imp_median.fit_transform(features_enc)

features_enc_mean = pd.DataFrame(features_enc_mean, columns=imp_mean.get_feature_names_out())
features_enc_mode = pd.DataFrame(features_enc_mode, columns=imp_mode.get_feature_names_out())
features_enc_median = pd.DataFrame(features_enc_median, columns=imp_median.get_feature_names_out())

# ## Etude mean :

# ### Chi2 test : 

# In[25]:


#apply SelectKBest class to extract top 10 best features
bestfeatures_chi_mean = SelectKBest(score_func=chi2, k=20)
fit_chi_mean = bestfeatures_chi_mean.fit(features_enc_mean,endometriose)
dfscores_chi_mean = pd.DataFrame(fit_chi_mean.scores_)
pvalues_chi_mean = pd.DataFrame(fit_chi_mean.pvalues_)
dfcolumns_chi_mean = pd.DataFrame(features_enc_mean.columns)
#concat two dataframes for better visualization 
featureScores_chi_mean = pd.concat([dfcolumns_chi_mean,dfscores_chi_mean,pvalues_chi_mean],axis=1)
featureScores_chi_mean.columns = ['Features','Score','p value']  #naming the dataframe columns
print(featureScores_chi_mean.nlargest(10,'Score'))  #print 20 best features

# In[26]:


featureScores_chi_mean.sort_values('Score', inplace=True, ascending=False)

fig, ax = plt.subplots(1,1,figsize=(20, 9))
ax.set_title('Résultats des scores du test chi2 "mean" des variables de Recueil', fontsize=18)
ax.bar(featureScores_chi_mean['Features'], height=featureScores_chi_mean['Score'], width=0.5, color='deepskyblue')
# ax.set_ylim(0,1)
ax.tick_params(axis='x',rotation=90)

# In[27]:


fig, ax = plt.subplots(1,1,figsize=(20, 9))
ax.set_title('Top10 des variables issues du chi2 "mean"', fontsize=18)
ax.bar(featureScores_chi_mean['Features'].iloc[:10], height=featureScores_chi_mean['Score'].iloc[:10], width=0.5, color='deepskyblue')
# ax.set_ylim(0,1)
ax.tick_params(axis='x',rotation=90)

# ### f_classif test : 

# In[28]:


#apply SelectKBest class to extract top 10 best features
bestfeatures_f_mean = SelectKBest(score_func=f_classif, k=20)
fit_f_mean = bestfeatures_f_mean.fit(features_enc_mean, endometriose)
pvalues_f_mean = pd.DataFrame(fit_f_mean.pvalues_)
dfscores_f_mean = pd.DataFrame(fit_f_mean.scores_)
dfcolumns_f_mean = pd.DataFrame(features_enc_mean.columns)
#concat two dataframes for better visualization 
featureScores_f_mean = pd.concat([dfcolumns_f_mean,dfscores_f_mean,pvalues_f_mean],axis=1)
featureScores_f_mean.columns = ['Features','Score','p value']  #naming the dataframe columns
print(featureScores_f_mean.nlargest(10,'Score'))  #print 20 best features

# In[29]:


featureScores.sort_values('Score', inplace=True, ascending=False)

fig, ax = plt.subplots(1,1,figsize=(20, 9))
ax.set_title('Résultats des scores du F-test "mean" des variables de Recueil', fontsize=18)
ax.bar(featureScores['Features'], height=featureScores['Score'], width=0.5, color='mediumslateblue')
# ax.set_ylim(0,1)
ax.tick_params(axis='x',rotation=90)

# In[30]:


fig, ax = plt.subplots(1,1,figsize=(20, 9))
ax.set_title('Top10 des variables issues du F-test "mean"', fontsize=18)
ax.bar(featureScores['Features'].iloc[:10], height=featureScores['Score'].iloc[:10], width=0.5, color='mediumslateblue')
# ax.set_ylim(0,1)
ax.tick_params(axis='x',rotation=90)

# In[61]:


df_p_values_f_mean = pd.DataFrame(fit.pvalues_, index=features_enc_mean.columns, columns=['p_value']) #TODO a voir
df_p_values_f_mean.sort_values('p_value', inplace=True)
df_p_values_f_mean.head(4)
#df_p_values_f_mean.loc[df_p_values_f_mean.loc[:,'p_value']<=0.05]

# ### mutual_info_classif test : 

# In[32]:


#apply SelectKBest class to extract top 10 best features
bestfeatures_mut = SelectKBest(score_func=mutual_info_classif, k=20)
fit_mut_mean = bestfeatures_mut.fit(features_enc_mean, endometriose)
dfscores_mut_mean = pd.DataFrame(fit_mut_mean.scores_)
dfcolumns_mut_mean = pd.DataFrame(features_enc_mean.columns)
#concat two dataframes for better visualization 
featureScores_mut_mean = pd.concat([dfcolumns_mut_mean,dfscores_mut_mean],axis=1)
featureScores_mut_mean.columns = ['Features','Score']  #naming the dataframe columns
print(featureScores_mut_mean.nlargest(20,'Score'))  #print 20 best features

# In[33]:


featureScores.sort_values('Score', inplace=True, ascending=False)

fig, ax = plt.subplots(1,1,figsize=(20, 9))
ax.set_title('Résultats des scores du test d\'information mutuelle "mean" des variables de Recueil', fontsize=18)
ax.bar(featureScores['Features'], height=featureScores['Score'], width=0.5, color='mediumblue')
# ax.set_ylim(0,1)
ax.tick_params(axis='x',rotation=90)

# In[34]:


fig, ax = plt.subplots(1,1,figsize=(20, 9))
ax.set_title('Top10 des variables issues de l\'information mutuelle "mean"', fontsize=18)
ax.bar(featureScores['Features'].iloc[:10], height=featureScores['Score'].iloc[:10], width=0.5, color='mediumblue')
# ax.set_ylim(0,1)
ax.tick_params(axis='x',rotation=90)

# ## Etude mode :

# ### Chi2 test : 

# In[35]:


#apply SelectKBest class to extract top 10 best features
bestfeatures_chi_mode = SelectKBest(score_func=chi2, k=20)
fit_ch_mode = bestfeatures_chi_mode.fit(features_enc_mode,endometriose)
dfscores_chi_mode = pd.DataFrame(fit_ch_mode.scores_)
p_val_chi_mode = pd.DataFrame(fit_ch_mode.pvalues_)
dfcolumns_chi_mode = pd.DataFrame(features_enc_mode.columns)
#concat two dataframes for better visualization 
featureScores_chi_mode = pd.concat([dfcolumns_chi_mode,dfscores_chi_mode,p_val_chi_mode],axis=1)
featureScores_chi_mode.columns = ['Features','Score','p value']  #naming the dataframe columns
print(featureScores_chi_mode.nlargest(10,'Score'))  #print 20 best features

# In[36]:


featureScores.sort_values('Score', inplace=True, ascending=False)


fig, ax = plt.subplots(1,1,figsize=(20, 9))


ax.set_title('Résultats des scores du test chi2 "mode" des variables de Recueil', fontsize=18)
ax.bar(featureScores['Features'], height=featureScores['Score'], width=0.5, color='deepskyblue')
# ax.set_ylim(0,1)
ax.tick_params(axis='x',rotation=90)



# In[37]:


fig, ax = plt.subplots(1,1,figsize=(20, 9))
ax.set_title('Top10 des variables issues du chi2 "mode"', fontsize=18)
ax.bar(featureScores['Features'].iloc[:10], height=featureScores['Score'].iloc[:10], width=0.5, color='deepskyblue')
# ax.set_ylim(0,1)
ax.tick_params(axis='x',rotation=90)

# In[38]:


df_p_values_chi_mode = pd.DataFrame(fit.pvalues_, index=features_enc_mean.columns, columns=['p_value'])
df_p_values_chi_mode.sort_values('p_value', inplace=True)
df_p_values_chi_mode.loc[df_p_values_chi_mode.loc[:,'p_value']<=0.05]

# ### f_classif test : 

# In[39]:


#apply SelectKBest class to extract top 10 best features
bestfeatures_f_mode = SelectKBest(score_func=f_classif, k=20)
fit_f_mode = bestfeatures_f_mode.fit(features_enc_mode, endometriose)
dfscores_f_mode = pd.DataFrame(fit_f_mode.scores_)
pval_f_mode = pd.DataFrame(fit_f_mode.pvalues_)
dfcolumns_f_mode = pd.DataFrame(features_enc_mode.columns)
#concat two dataframes for better visualization 
featureScores_f_mode = pd.concat([dfcolumns_f_mode,dfscores_f_mode, pval_f_mode],axis=1)
featureScores_f_mode.columns = ['Features','Score', 'p value']  #naming the dataframe columns
print(featureScores_f_mode.nlargest(10,'Score'))  #print 20 best features

# In[40]:


featureScores_f_mode.sort_values('Score', inplace=True, ascending=False)


fig, ax = plt.subplots(1,1,figsize=(20, 9))


ax.set_title('Résultats des scores du F-test "mode" des variables de Recueil', fontsize=18)
ax.bar(featureScores_f_mode['Features'], height=featureScores_f_mode['Score'], width=0.5, color='mediumslateblue')
# ax.set_ylim(0,1)
ax.tick_params(axis='x',rotation=90)



# In[41]:


fig, ax = plt.subplots(1,1,figsize=(20, 9))
ax.set_title('Top10 des variables issues du F-test "mode"', fontsize=18)
ax.bar(featureScores_f_mode['Features'].iloc[:10], height=featureScores_f_mode['Score'].iloc[:10], width=0.5, color='mediumslateblue')
# ax.set_ylim(0,1)
ax.tick_params(axis='x',rotation=90)

# ### mutual_info_classif test : 

# In[42]:


#apply SelectKBest class to extract top 10 best features
bestfeatures = SelectKBest(score_func=mutual_info_classif, k=20)
fit = bestfeatures.fit(features_enc_mode, endometriose)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(features_enc_mode.columns)
#concat two dataframes for better visualization 
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Features','Score']  #naming the dataframe columns
print(featureScores.nlargest(20,'Score'))  #print 20 best features

# In[43]:


featureScores.sort_values('Score', inplace=True, ascending=False)

fig, ax = plt.subplots(1,1,figsize=(20, 9))
ax.set_title('Résultats des scores du test d\'information mutuelle "mode" des variables de Recueil', fontsize=18)
ax.bar(featureScores['Features'], height=featureScores['Score'], width=0.5, color='mediumblue')
# ax.set_ylim(0,1)
ax.tick_params(axis='x',rotation=90)

# In[44]:


fig, ax = plt.subplots(1,1,figsize=(20, 9))
ax.set_title('Top10 des variables issues de l\'information mutuelle "mode"', fontsize=18)
ax.bar(featureScores['Features'].iloc[:10], height=featureScores['Score'].iloc[:10], width=0.5, color='mediumblue')
# ax.set_ylim(0,1)
ax.tick_params(axis='x',rotation=90)

# ## Etude median :

# ### Chi2 test : 

# In[45]:


#apply SelectKBest class to extract top 10 best features
bestfeatures_chi_med = SelectKBest(score_func=chi2, k=20)
fit_chi_med = bestfeatures_chi_med.fit(features_enc_median,endometriose)
pval_chi_med = pd.DataFrame(fit_chi_med.pvalues_)
dfscores_chi_med = pd.DataFrame(fit_chi_med.scores_)
dfcolumns_chi_med = pd.DataFrame(features_enc_median.columns)
#concat two dataframes for better visualization 
featureScores_chi_med = pd.concat([dfcolumns_chi_med,dfscores_chi_med, pval_chi_med],axis=1)
featureScores_chi_med.columns = ['Features','Score', 'p values']  #naming the dataframe columns
print(featureScores_chi_med.nlargest(10,'Score'))  #print 20 best features

# In[46]:


featureScores_chi_med.sort_values('Score', inplace=True, ascending=False)

fig, ax = plt.subplots(1,1,figsize=(20, 9))
ax.set_title('Résultats des scores du test chi2 "mediane" des variables de Recueil', fontsize=18)
ax.bar(featureScores_chi_med['Features'], height=featureScores_chi_med['Score'], width=0.5, color='deepskyblue')
# ax.set_ylim(0,1)
ax.tick_params(axis='x',rotation=90)


# In[47]:


fig, ax = plt.subplots(1,1,figsize=(20, 9))
ax.set_title('Top10 des variables issues du chi2 "mediane"', fontsize=18)
ax.bar(featureScores['Features'].iloc[:10], height=featureScores['Score'].iloc[:10], width=0.5, color='deepskyblue')
# ax.set_ylim(0,1)
ax.tick_params(axis='x',rotation=90)

# In[48]:


df_p_values_chi_med = pd.DataFrame(fit.pvalues_, index=features_enc_mean.columns, columns=['p_value'])
df_p_values_chi_med.sort_values('p_value', ascending=False, inplace=True)
df_p_values_chi_med.iloc[:10,:]

# In[49]:


df_p_values_chi_med.sort_values('p_value', inplace=True)
df_p_values_chi_med.loc[df_p_values_chi_med.loc[:,'p_value']<=0.05]

# ### f_classif test : 

# In[50]:


#apply SelectKBest class to extract top 10 best features
bestfeatures_f_med = SelectKBest(score_func=f_classif, k=20)
fit_f_med = bestfeatures_f_med.fit(features_enc_median, endometriose)
dfscores_f_med = pd.DataFrame(fit_f_med.scores_)
pval_f_med = pd.DataFrame(fit_f_med.pvalues_)
dfcolumns_f_med = pd.DataFrame(features_enc_median.columns)
#concat two dataframes for better visualization 
featureScores_f_med = pd.concat([dfcolumns_f_med,dfscores_f_med, pval_f_med],axis=1)
featureScores_f_med.columns = ['Features','Score', 'p values']  #naming the dataframe columns
print(featureScores_f_med.nlargest(10,'Score'))  #print 20 best features

# In[51]:


featureScores_f_med.sort_values('Score', inplace=True, ascending=False)


fig, ax = plt.subplots(1,1,figsize=(20, 9))


ax.set_title('Résultats des scores du F-test "mediane" des variables de Recueil', fontsize=18)
ax.bar(featureScores_f_med['Features'], height=featureScores_f_med['Score'], width=0.5, color='mediumslateblue')
# ax.set_ylim(0,1)
ax.tick_params(axis='x',rotation=90)



# In[52]:


fig, ax = plt.subplots(1,1,figsize=(20, 9))
ax.set_title('Top10 des variables issues du F-test "mediane"', fontsize=18)
ax.bar(featureScores_f_med['Features'].iloc[:10], height=featureScores_f_med['Score'].iloc[:10], width=0.5, color='mediumslateblue')
# ax.set_ylim(0,1)
ax.tick_params(axis='x',rotation=90)

# In[53]:


df_p_values_f_med = pd.DataFrame(fit.pvalues_, index=features_enc_mean.columns, columns=['p_value'])
df_p_values_f_med.sort_values('p_value', inplace=True)

# In[54]:


df_p_values_f_med.loc[df_p_values_f_med.loc[:,'p_value']<=0.05]

# ### mutual_info_classif test : 

# In[55]:


#apply SelectKBest class to extract top 10 best features
bestfeatures = SelectKBest(score_func=mutual_info_classif, k=20)
fit = bestfeatures.fit(features_enc_median, endometriose)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(features_enc_mean.columns)
#concat two dataframes for better visualization 
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Features','Score']  #naming the dataframe columns
print(featureScores.nlargest(20,'Score'))  #print 20 best features

# In[56]:


featureScores.sort_values('Score', inplace=True, ascending=False)

fig, ax = plt.subplots(1,1,figsize=(20, 9))
ax.set_title('Résultats des scores du test d\'information mutuelle "mediane" des variables de Recueil', fontsize=18)
ax.bar(featureScores['Features'], height=featureScores['Score'], width=0.5, color='mediumblue')
# ax.set_ylim(0,1)
ax.tick_params(axis='x',rotation=90)

# In[57]:


fig, ax = plt.subplots(1,1,figsize=(20, 9))
ax.set_title('Top10 des variables issues de l\'information mutuelle "mediane"', fontsize=18)
ax.bar(featureScores['Features'].iloc[:10], height=featureScores['Score'].iloc[:10], width=0.5, color='mediumblue')
# ax.set_ylim(0,1)
ax.tick_params(axis='x',rotation=90)
