#!/usr/bin/env python
# coding: utf-8

# #  NLP_ENDOPATHS_Nicolai

# Author: Nicolai Wolpert
# Email: nicolai.wolpert@capgemini.com
# Date: June 2024

# ## Imports

# In[1]:


### Imports ###

# Data manipulation and other stuff : 
import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 500)
from matplotlib import pyplot as plt
import seaborn as sns
#import eli5 # eli5 not working anymore for current stable version of sklearn
from IPython.display import display

from exploratory.preprocessing.preprocess_NLP import from_X_split_get_Y_split
from sklearn.utils import resample, shuffle

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
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, precision_recall_curve, hamming_loss, accuracy_score, jaccard_score, classification_report, roc_auc_score
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, roc_curve
from exploratory.utils.metrics_utils import *
from sklearn.metrics import ConfusionMatrixDisplay

from sklearn.tree import plot_tree

# Multiclass/Multilabel preparation :
from sklearn.base import BaseEstimator, ClassifierMixin

# Kfold cross-validation with stratification
from exploratory.Opti_utils.ML_utils import kfold_cv_stratified

# Custom preprocessing : 
from exploratory.preprocessing.preprocess_NLP import preprocess_and_split
from exploratory.utils.metrics_utils import *
from exploratory.Opti_utils.ML_utils import Binarisation

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
from keras import Sequential
from keras.layers import Embedding, LSTM, Dropout, Dense


# ## Load data

# In[2]:


'''
# Les features qui ont été identifiés comme les plus pertinentes pour pouvoir prédire l'endiométrose profonde sont les suivants
# (cela veut dire qu'ils donnent une sensibilité et spécificité au dessus du seuil de 0.4 et 0.5):
features_of_interest = ['age', 'atcd.endo', 'sf.dsm.eva', 'sf.dpc', 'tv.douleur.lusd', 'tv.douleur.lusg', 'chir_GD', 'chir_SW']
'''
# See 'Explore_features_ML_NLP' for how these have been identified
features_of_interest = ['atcd.endo', 'irm.lusg', 'tv.douloureux', 'irm.externe', 'sf.dig.diarrhee', 'echo.lusg', 'echo.lusd', 'ef.hormone.dpc', 'effet.hormone']
features_of_interest = ['atcd.absenteisme']
#features_of_interest = 'all'

# In[3]:


# Loading X : 
df_nlp = pd.read_csv('Data/DATA_PROCESSED/donnees_entree_nlp_sans_endo.csv', usecols=['Anonymisation', 'Date', 'Nature', 'Résumé'])
print('X shape is :', df_nlp.shape)

# Use original receuil data or infos extracted manually from the gyneco files
receuil_data = 'manual_gyneco'     # 'orig', 'manual_gyneco'

# Loading Y : 
if receuil_data == 'manual_gyneco':
    # Read the receuil version created by manually extracting infos from gyneco files
    recueil = pd.read_excel('Data/DATA_PROCESSED/data_gynéco_manual_extraction.xlsx')
    recueil = recueil[['Anonymisation'] + features_of_interest]

else:
    recueil_orig  = pd.read_excel('Data/DATA_RAW/Recueil (1).xlsx').drop('Unnamed: 90', axis=1)
    recueil = recueil_orig.copy()
    recueil.replace(['Na', 'NA'], np.nan, inplace=True)
    recueil = recueil.rename(columns={'Numéro anonymat': 'Anonymisation'})
    # Note the target variable, if endometriosis is present or not (corresponds to the anapth columns but not 'autre')
    recueil['endometriose'] = recueil_orig.loc[:,['anapath.lusd','anapath.lusg','anapath.torus']].sum(axis=1).apply(lambda x: Binarisation(x))
    # Drop the anapath columns again
    recueil = recueil[[c for c in recueil.columns if not c.startswith('anapath')]]
    recueil = recueil[['Anonymisation'] + features_of_interest]
recueil.set_index('Anonymisation', inplace=True)
print('Y shape is :', recueil.shape)
print(f'Nombre de patientes dans le df_nlp : {len(df_nlp.Anonymisation.unique())}')

if 'DJ-055' in list(df_nlp['Anonymisation']):
    df_nlp.loc[df_nlp['Anonymisation']=='DJ-055', 'Anonymisation'] ='NJ-055'
'NJ-055' in list(df_nlp['Anonymisation'])
'DJ-055' in list(df_nlp['Anonymisation'])

# ## Preprocess 

# In[17]:


X_train, X_test, Y_train, Y_test, max_vocab, X, Y = preprocess_and_split( df_nlp,
                                                                    recueil,
                                                                    features_of_interest,
                                                                    42,                     # choix de la seed pour le random split
                                                                    0.20,                   # Choix du test_size
                                                                    special_char=True,      # supprime les caractères spéciaux  
                                                                    accents=False,          # supprime les accents
                                                                    lower=True,            # passe la casse du texte en minuscule
                                                                    token=False,            # tokenize le corpus
                                                                    remove_stopwords=True,
                                                                    drop_number=True,
                                                                    compress=True,          # un corpus par n_ano, shape = [200,2]
                                                                    preprocess_mode='multiclass',
                                                                    encoder_mode=True,
                                                                    anapath=True,
                                                                    correction_by_json=True
                                                                  )
print(f'Nombre de patientes inclues dans l\'étude NLP "baseline" : {X_train.shape[0] + X_test.shape[0]}')
print(f'Nombre de symptômes/caractéristiques à prédire dans l\'étude NLP : {Y_train.shape[1]}')

# ### Optional, for the case of one feature: Resampling to correct for class imbalance

# In[5]:


use_resampling = True
balance_method = 'upsampling'       # 'upsampling' or 'downsampling'
if len(features_of_interest) == 1:
    target_feature = features_of_interest[0]
    data_train = pd.merge(X_train, Y_train, on='Anonymisation')

    # Vérifier distribution des classes dans train/val/test
    fig, axs = plt.subplots(nrows=1, ncols=2)
    axs = axs.flatten()
    sns.histplot(data=Y_train[target_feature].map({0: 'absent', 1: 'present', 2: 'ambigue'}), shrink=0.3, color='#00B2A2', ax=axs[0])
    axs[0].bar_label(axs[0].containers[0])
    axs[0].set_xlabel('')
    axs[0].set_ylabel('Nombre', fontsize=14)
    axs[0].set_title('train', pad=25, fontsize=14)
    sns.histplot(data=Y_test[target_feature].map({0: 'absent', 1: 'present', 2: 'ambigue'}), shrink=0.3, color='#00B2A2', ax=axs[1])
    axs[1].bar_label(axs[1].containers[0])
    axs[1].set_xlabel('')
    axs[1].set_ylabel('Nombre', fontsize=14)
    axs[1].set_ylim(axs[0].get_ylim())
    axs[1].set_title('test', pad=25, fontsize=14)
    plt.suptitle('Distribution classes par patientes avant resampling')
    plt.tight_layout()
    plt.show()

    data_positive_train = (Y_train[target_feature] == 1.0).sum()
    data_negative_train = (Y_train[target_feature] == 0.0).sum()
    data_ambiguous_train = (Y_train[target_feature] == 2.0).sum()
    print('positive data in training:', data_positive_train)
    print('negative data in training:', data_negative_train)
    print('ambiguous data in training:', data_ambiguous_train)
    print()
    print('positive data in test:',(Y_test[target_feature] == 1.0).sum())
    print('negative data in test:',(Y_test[target_feature] == 0.0).sum())
    print('ambiguous data in test:',(Y_test[target_feature] == 2.0).sum())

    ### Now the resampling

    # Separate majority and minority classes in training data for upsampling
    if data_positive_train == np.max([data_positive_train, data_negative_train, data_ambiguous_train]):
        data_train_majority = data_train[data_train[target_feature] == 1.0]
        data_train_minority1 = data_train[data_train[target_feature] == 0.0]
        data_train_minority2 = data_train[data_train[target_feature] == 2.0]
    elif data_negative_train == np.max([data_positive_train, data_negative_train, data_ambiguous_train]):
        data_train_majority = data_train[data_train[target_feature] == 0.0]
        data_train_minority1 = data_train[data_train[target_feature] == 1.0]
        data_train_minority2 = data_train[data_train[target_feature] == 2.0]
    elif data_ambiguous_train == np.max([data_positive_train, data_negative_train, data_ambiguous_train]):
        data_train_majority = data_train[data_train[target_feature] == 2.0]
        data_train_minority1 = data_train[data_train[target_feature] == 0.0]
        data_train_minority2 = data_train[data_train[target_feature] == 1.0]
        
    if balance_method=='upsampling':
        
        print("majority class before upsampling:",data_train_majority.shape)
        print("minority class 1 before upsampling:",data_train_minority1.shape)
        print("minority class 2 before upsampling:",data_train_minority2.shape)

        # Upsample minority class 1
        data_train_minority1_upsampled = resample(data_train_minority1, 
                                        replace=True,     # sample with replacement
                                        n_samples= data_train_majority.shape[0],    # to match majority class
                                        random_state=123) # reproducible results
        # Upsample minority class 2
        data_train_minority2_upsampled = resample(data_train_minority2, 
                                        replace=True,     # sample with replacement
                                        n_samples= data_train_majority.shape[0],    # to match majority class
                                        random_state=123) # reproducible results

        # Combine majority class with upsampled minority class
        data_train_upsampled = pd.concat([data_train_majority, data_train_minority1_upsampled, data_train_minority2_upsampled])
        
        # Display new class counts
        print("After upsampling\n",data_train_upsampled[target_feature].value_counts(),sep = "")
        
        data_train = data_train_upsampled.copy()

    # Not recommended since it results in huge data loss - just to test out performance of model with completely balanced data distribution
    elif balance_method=='downsampling':

        minimum_number_of_patients_per_class = np.min([data_positive_train, data_negative_train, data_ambiguous_train])
        
        ### Remove samples from the majority class
        train_original = data_train.copy()
        train_negative = data_train.loc[data_train[target_feature]==0]
        train_positive = data_train.loc[data_train[target_feature]==1]
        train_missing = data_train.loc[data_train[target_feature]==2]
        
        if train_negative.shape[0] > train_positive.shape[0]:
            data_train_majority = train_negative
            data_train_minority = train_positive
        else:
            data_train_majority = train_positive
            data_train_minority = train_negative
        
        train_negative = train_negative.sample(n=minimum_number_of_patients_per_class, replace=False, random_state=1)
        train_positive = train_positive.sample(n=minimum_number_of_patients_per_class, replace=False, random_state=1)
        train_missing = train_missing.sample(n=minimum_number_of_patients_per_class, replace=False, random_state=1)
        
        # Combine majority class with downsampled majority class
        train_downsampled = pd.concat([train_negative, train_positive, train_missing])
        
        data_train = train_downsampled.copy()
    
    X_train = data_train[['Anonymisation', 'Résumé']]
    Y_train = data_train[['Anonymisation', target_feature]].set_index(['Anonymisation'])

    # Vérifier bonne distribution des classes dans train/val/test
    fig, axs = plt.subplots(nrows=1, ncols=2)
    axs = axs.flatten()
    sns.histplot(data=Y_train[target_feature].map({0: 'absent', 1: 'present', 2: 'ambigue'}), shrink=0.3, color='#00B2A2', ax=axs[0])
    axs[0].bar_label(axs[0].containers[0])
    axs[0].set_xlabel('')
    axs[0].set_ylabel('Nombre', fontsize=14)
    axs[0].set_title('train', pad=25, fontsize=14)
    sns.histplot(data=Y_test[target_feature].map({0: 'absent', 1: 'present', 2: 'ambigue'}), shrink=0.3, color='#00B2A2', ax=axs[1])
    axs[1].bar_label(axs[1].containers[0])
    axs[1].set_xlabel('')
    axs[1].set_ylabel('Nombre', fontsize=14)
    axs[1].set_ylim(axs[0].get_ylim())
    axs[1].set_title('test', pad=25, fontsize=14)
    plt.suptitle('Distribution classes par phrases après resampling')
    plt.tight_layout()
    plt.show()

else:
    print('You have more than one feature of interest. You cannot use upsampling')

# In[6]:


tfIdfVectorizer=TfidfVectorizer(use_idf=True, max_features=max_vocab, lowercase=False)
X_train_fitted = tfIdfVectorizer.fit_transform(X_train.Résumé)
X_train_fitted_df = pd.DataFrame(X_train_fitted.todense(), columns=tfIdfVectorizer.get_feature_names_out(), index= X_train.Anonymisation)
X_test_fitted = tfIdfVectorizer.transform(X_test.Résumé)
X_test_fitted_df = pd.DataFrame(X_test_fitted.todense(), columns=tfIdfVectorizer.get_feature_names_out(), index= X_test.Anonymisation)

# ## Algos

# ### Decision Trees

# In[7]:


####################################################################
multilabel_classifier = DecisionTreeClassifier(random_state=42)
labels_MLC = {0:'négatif', 1:'positif', 2:' Non mentionnées'}
multilabel_classifier.fit(X_train_fitted_df, Y_train)
####################################################################

# In[8]:


Y_pred = multilabel_classifier.predict(X_test_fitted_df)
Y_pred = pd.DataFrame(Y_pred, columns=Y_train.columns)
DT_Multi_index, CR_global = rapport_metrics_decision_tree(Y_test, Y_pred)

# In[9]:


if len(features_of_interest) == 1:
    target_feature = features_of_interest[0]
    print(classification_report(DT_Multi_index[target_feature]['y_true'], DT_Multi_index[target_feature]['y_pred']))
    cm = confusion_matrix(DT_Multi_index[target_feature]['y_true'], DT_Multi_index[target_feature]['y_pred'])
    if len(pd.unique(recueil[target_feature]))==2:
        labels=['non', 'oui']
    else:
        labels=['non', 'oui', 'ambigue']
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot()
else:
    score_decision_tree = evaluate_results_multilabel(DT_Multi_index, CR_global)
    show_best_and_lowest_scores(CR_global, score='f1_score')
    show_best_and_lowest_scores(CR_global, score='recall')
    show_best_and_lowest_scores(CR_global, score='precision')
    show_precision_vs_recall(CR_global)

# A faire: Corréler score ML de chaque feature avec score NLP respectif

# #### Decision Trees Kfold crossvalidation

# In[10]:


classifier = DecisionTreeClassifier(random_state=42)
Y_pred_folds, classifier = kfold_cv_stratified(X, Y, classifier, max_vocab, nfolds=5)
DT_cv_Multi_index, CR_global_DT_cv = rapport_metrics_decision_tree(Y, Y_pred_folds)
CR_global_DT_cv

if len(features_of_interest) == 1:
    target_feature = features_of_interest[0]
    print(classification_report(DT_cv_Multi_index[target_feature]['y_true'], DT_cv_Multi_index[target_feature]['y_pred']))
    cm = confusion_matrix(DT_cv_Multi_index[target_feature]['y_true'], DT_cv_Multi_index[target_feature]['y_pred'])
    if len(pd.unique(recueil[target_feature]))==2:
        labels=['non', 'oui']
    else:
        labels=['non', 'oui', 'ambigue']
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot()
else:
    score_decision_tree = evaluate_results_multilabel(DT_cv_Multi_index, CR_global_DT_cv)
    show_best_and_lowest_scores(CR_global_DT_cv, score='f1_score')
    show_best_and_lowest_scores(CR_global_DT_cv, score='recall')
    show_best_and_lowest_scores(CR_global_DT_cv, score='precision')
    show_precision_vs_recall(CR_global_DT_cv)

# ### Random Forests

# In[11]:


####################################################################
multilabel_classifier_RFC = RandomForestClassifier(random_state=42, class_weight = "balanced")         # Adjust for the imbalance in class distributions
# labels_MLC = {0:'négatif', 1:'positif', 2:' Non mentionnées'}
multilabel_classifier_RFC.fit(X_train_fitted_df, Y_train)
####################################################################

# In[12]:


Y_pred = multilabel_classifier_RFC.predict(X_test_fitted_df)
Y_pred = pd.DataFrame(Y_pred, columns=Y_train.columns)
RF_Multi_index, CR_global_RF = rapport_metrics_decision_tree(Y_test, Y_pred)
CR_global_RF

# In[13]:


if len(features_of_interest) == 1:
    target_feature = features_of_interest[0]
    print(classification_report(RF_Multi_index[target_feature]['y_true'], RF_Multi_index[target_feature]['y_pred']))
    cm = confusion_matrix(RF_Multi_index[target_feature]['y_true'], RF_Multi_index[target_feature]['y_pred'])
    if len(pd.unique(recueil[target_feature]))==2:
        labels=['non', 'oui']
    else:
        labels=['non', 'oui', 'ambigue']
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot()
else:
    score_decision_tree = evaluate_results_multilabel(RF_Multi_index, CR_global_RF)
    show_best_and_lowest_scores(CR_global_RF, score='f1_score')
    show_best_and_lowest_scores(CR_global_RF, score='recall')
    show_best_and_lowest_scores(CR_global_RF, score='precision')
    show_precision_vs_recall(CR_global_RF)

# #### RandomForests Kfold crossvalidation

# In[14]:


classifier = RandomForestClassifier(random_state=42, class_weight = "balanced")
Y_pred_folds, classifier = kfold_cv_stratified(X, Y, classifier, max_vocab, nfolds=5)
RF_cv_Multi_index, CR_global_RF_cv = rapport_metrics_decision_tree(Y, Y_pred_folds)
CR_global_RF_cv

# In[15]:


if len(features_of_interest) == 1:
    target_feature = features_of_interest[0]
    print(classification_report(RF_cv_Multi_index[target_feature]['y_true'], RF_cv_Multi_index[target_feature]['y_pred']))
    cm = confusion_matrix(RF_cv_Multi_index[target_feature]['y_true'], RF_cv_Multi_index[target_feature]['y_pred'])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
else:
    score_decision_tree = evaluate_results_multilabel(RF_cv_Multi_index, CR_global_RF_cv)
    show_best_and_lowest_scores(CR_global_RF_cv, score='f1_score')
    show_best_and_lowest_scores(CR_global_RF_cv, score='recall')
    show_best_and_lowest_scores(CR_global_RF_cv, score='precision')
    show_precision_vs_recall(CR_global_RF_cv)

# #### Random Forests Hyperparameter Search

# In[21]:


#from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
#  ...existing code...
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import MultiLabelBinarizer

# Custom cross-validator class using IterativeStratification
"""class CustomIterativeStratifiedCV:
    def __init__(self, n_splits=5):
        self.n_splits = n_splits

    def split(self, X, y, groups=None):
        mskf = MultilabelStratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        return mskf.split(X, y)

    def get_n_splits(self, X, y, groups=None):
        return self.n_splits"""

# Define the parameter grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

# Initialize the classifier
classifier = RandomForestClassifier(random_state=42, class_weight="balanced")

# Initialize the custom cross-validator
#custom_cv = CustomIterativeStratifiedCV(n_splits=5)
# ...existing code...
custom_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# ...existing code...

# Initialize GridSearchCV with the custom cross-validator
grid_search = GridSearchCV(estimator=classifier, param_grid=param_grid, 
                           cv=custom_cv, n_jobs=-1, verbose=2, scoring='f1_weighted')

tfIdfVectorizer=TfidfVectorizer(use_idf=True, max_features=max_vocab, lowercase=False)
X_fitted = tfIdfVectorizer.fit_transform(X.Résumé)
X_fitted_df = pd.DataFrame(X_fitted.todense(), columns=tfIdfVectorizer.get_feature_names_out(), index= X.Anonymisation)

# Exemple de conversion de Y en format multi-étiquette
mlb = MultiLabelBinarizer()
Y_multilabel = mlb.fit_transform(Y.values.reshape(-1, 1))

# Vérifiez le nouveau format de Y
print(Y_multilabel)

# Fit the model
#grid_search.fit(X_fitted_df.values, Y_multilabel)
# ...existing code...
# Fit the model
grid_search.fit(X_fitted_df.values, Y.values.ravel())

# Get the best parameters and the best model
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

print("Best parameters found: ", best_params)
print("Best model found: ", best_model)

# Cross-validation predictions using the best model
#Y_pred_folds = cross_val_predict(best_model, X_fitted_df.values, Y_multilabel, cv=custom_cv, method='predict')
#Y_pred_folds = pd.DataFrame(Y_pred_folds, columns=Y.columns, index=Y.index)
# ...existing code...
# Cross-validation predictions using the best model
Y_pred_folds = cross_val_predict(best_model, X_fitted_df.values, Y.values.ravel(), cv=custom_cv, method='predict')
Y_pred_folds = pd.DataFrame(Y_pred_folds, columns=Y.columns, index=Y.index)

# ...existing code...

# Compute classification report and confusion matrix
RF_cv_Multi_index, CR_global_RF_cv = rapport_metrics_decision_tree(Y, Y_pred_folds)

# Print the results
print(RF_cv_Multi_index)
print(CR_global_RF_cv)


# ### SVC

# In[60]:


CR_global_SVC, SVC_Multi_index, SVC_y_pred, dict_model_svc, SVC_Multi_index_proba = multilabel_multioutput_svc(X_train_fitted_df, X_test_fitted_df, Y_train, Y_test, tfIdfVectorizer)

# In[61]:


if len(features_of_interest) == 1:
    target_feature = features_of_interest[0]
    print(classification_report(SVC_Multi_index[target_feature]['y_true'], SVC_Multi_index[target_feature]['y_pred']))
    cm = confusion_matrix(SVC_Multi_index[target_feature]['y_true'], SVC_Multi_index[target_feature]['y_pred'])
    if len(pd.unique(recueil[target_feature]))==2:
        labels=['non', 'oui']
    else:
        labels=['non', 'oui', 'ambigue']
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot()
else:
    score_decision_tree = evaluate_results_multilabel(SVC_Multi_index, CR_global_SVC)
    show_best_and_lowest_scores(CR_global_SVC, score='f1_score')
    show_best_and_lowest_scores(CR_global_SVC, score='recall')
    show_best_and_lowest_scores(CR_global_SVC, score='precision')
    show_precision_vs_recall(CR_global_SVC)
