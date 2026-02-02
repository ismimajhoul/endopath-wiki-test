'''
Author : Maxime Mock
Date 09/03/2023
Project : ENDOPATHS

Purpose : Prepare some utils for grid search automation followed by thresholding variation for set the best results for Machine learning classification problems
'''

# Some utils for metrics : 


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from xgboost import XGBClassifier, XGBRFClassifier, plot_importance
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, accuracy_score, make_scorer, precision_recall_curve, roc_curve
from sklearn.model_selection import cross_validate
from sklearn.feature_extraction.text import TfidfVectorizer
from skmultilearn.model_selection import IterativeStratification

def clean_lists(liste):
    liste_temp = liste.copy()
    for col in liste:
        if not check_corr_col(col, liste_temp):
            liste_temp.remove(col)
    return liste_temp

def Binarisation(x):
    if x>1:
        x=1
    return x

def find_error(y_test, y_pred):
    '''
    Take target set and predicted set and return the index for False positive and False negative.
    Input : 
    ---------
    y_test : True value from the split of data.
    y_pred : Predicted value from the model
    
    Return :
    ---------
    dict_idx : A dictionnary with 'FP' and 'FN' keys and getting lists as value.
    '''
    # Initialisation : 
    dict_idx = {'FP':[], 'FN':[]}
    # Loop for reasearch FP and FN : 
    for idx, elem in enumerate(y_pred):
        if elem > y_test.iloc[idx]:
            dict_idx['FP'].append(idx)
        elif elem < y_test.iloc[idx]:
            dict_idx['FN'].append(idx)
    return dict_idx
# Metrics : 

def custom_metric(y_true, y_pred):
    specificite = recall_score(y_true, y_pred, pos_label=0)
    if specificite >=0.8:
        metric = recall_score(y_true, y_pred, pos_label=1)
    if specificite < 0.8:
        metric = (recall_score(y_true, y_pred, pos_label=0) + recall_score(y_true, y_pred, pos_label=1))/2
    return metric


def sensi(y_true, y_pred):
    return recall_score(y_true, y_pred, pos_label=1)

def speci(y_true, y_pred):
    return recall_score(y_true, y_pred, pos_label=0)

def mix_sensi_speci(y_true, y_pred):
    return (recall_score(y_true, y_pred, pos_label=0) + recall_score(y_true, y_pred, pos_label=1))/2

# scorer = make_scorer(custom_metric, greater_is_better=True)
# mix_recall = make_scorer(mix_sensi_speci, greater_is_better=True)
# sensibilite = make_scorer(sensi, greater_is_better=True)
# specificite = make_scorer(speci, greater_is_better=True)

def FN(to_test, ref):
    ''' Check if a prediction is a FN
    Return a boolean'''
    boolean = False
    if to_test > ref:
        boolean = True
    return boolean

def TP(to_test, ref):
    ''' Check if a prediction is a TP
    Return a boolean'''
    boolean = False
    if to_test == 1 and ref==1:
        boolean = True
    return boolean

def TN(to_test, ref):
    ''' Check if a prediction is a TN
    Return a boolean'''
    boolean = False
    if to_test == 0 and ref== 0:
        boolean = True
    return boolean

def FP(to_test, ref):
    ''' Check if a prediction is a FP
    Return a boolean'''
    boolean = False
    if to_test <ref:
        boolean = True
    return boolean


def compare_results(series_to_test, series_reference):
    '''Compare a predicted series with a series of reference.
    return a pd.Series with filled by FP,FN,TP,TN values
    '''
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

def scores(y_test, y_pred):
    '''
    Calculate selected metrics between predicted values and true values
    
    Input:
    -------------
    y_test : True value from the split of data
    y_pred : Predicted value from a model
    
    Return :
    -------------
    a dictionnary taking metrics name as keys
    '''
    CM = confusion_matrix(y_test, y_pred)
    TP = CM[1][1]
    FP = CM[0][1]
    TN = CM[0][0]
    FN = CM[1][0]
    specificite = TN/(FP+TN)
    sensibilite = recall_score(y_test, y_pred)
    return {'specificité':specificite, 'sensibilite':sensibilite, 'TP':TP, 'FP':FP, 'TN':TN, 'FN':FN}

def score(y_test, y_pred):
    '''
    Calculate selected metrics between predicted values and true values
    
    Input:
    -------------
    y_test : True value from the split of data
    y_pred : Predicted value from a model
    
    Return :
    -------------
    a dictionnary taking metrics name as keys
    '''
    CM = confusion_matrix(y_test, y_pred)
    TP = CM[1][1]
    FP = CM[0][1]
    TN = CM[0][0]
    FN = CM[1][0]
    specificite = TN/(FP+TN)
    sensibilite = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    return [specificite, sensibilite, precision, accuracy, f1, TP, FP, TN, FN]


def custom_metrics(y_test, y_pred):
    CM = confusion_matrix(y_test, y_pred)
    TP = CM[1][1]
    FP = CM[0][1]
    TN = CM[0][0]
    FN = CM[1][0]
    specificite = TN/(FP+TN)
    sensibilite = recall_score(y_test, y_pred)
    mean = round((specificite+sensibilite)/2,3)
    return mean


def find_best_threshold(df:pd.DataFrame(), seuil):
    try :
        # On isole la fraction du dataframe qui a une spécificité au dessus de 0,80 :
        df_speci_80 = df.loc[df.loc[:,'speci']>=seuil]
        # On cherche les maximums de la sensibilité : 
        df_sensi_max = df_speci_80.loc[df_speci_80.loc[:,'sensi']==df_speci_80.loc[:,'sensi'].max(),:]
        # On prend le max en specificité : 
        df_sensi_max_speci_max = df_sensi_max.loc[df_sensi_max.loc[:,'speci']==df_sensi_max.loc[:,'speci'].max(),:]
        return round(df_sensi_max_speci_max.iloc[:,0].values[0], 3)
    except:
        print('Tri impossible, possiblement aucune valeur supérieur au seuil')

def from_model_to_bst_trhld2(model, x_test, y_test):
    '''
    Take a model and find the best threshold for his prediction
    
    Input : 
    ---------------
    model : a fitted model
    x_test : X_test from the split
    y_test : y_test from the split
    
    Return :
    ---------------
    dict_resultat : a dictionnary for each threshold (from 0.05 to 0.95')
    '''
    # initialisation :    
    liste_resultats = []
    predict_proba = pd.DataFrame(model.predict_proba(x_test))
    # préparation des données : 
    index = list(y_test.index)
    predict_proba = pd.DataFrame(model.predict_proba(x_test), index=index)
    
    predict_proba['y_test'] = y_test
    predict_proba['y_pred'] = model.predict(x_test)
    #loop :
    for threshold in np.arange(0.05, 1, 0.01):
        dict_ = {}
        predict_proba_temp = (predict_proba.iloc[:,1]>threshold).apply(lambda x: round(x,0))
        sensib = sensi(y_test, predict_proba_temp)
        specic = speci(y_test, predict_proba_temp)
        dict_['trhd']=threshold
        dict_['sensi']=sensib
        dict_['speci']=specic
        liste_resultats.append(dict_)
    resultats = pd.DataFrame(liste_resultats)
    return resultats

def from_model_to_bst_trhld(model, x_test, y_test):
    '''
    from model to select the best threshold for get the best prediction
    
    Input : 
    ---------------
    model : a fitted model
    x_test : X_test from the split
    y_test : y_test from the split
    
    Return :
    ---------------
    resultats : a dataframe for each threshold (from 0.05 to 0.95')
    '''
    # initialisation :    
    liste_resultats = []
    predict_proba = pd.DataFrame(model.predict_proba(x_test))
    for threshold in np.arange(0.05, 1, 0.01):
        dict_ = {}
        predict_proba_temp = (predict_proba.iloc[:,1]>threshold).apply(lambda x: round(x,0))
        sensib = sensi(y_test, predict_proba_temp)
        specic = speci(y_test, predict_proba_temp)
        dict_['trhd']=threshold
        dict_['sensi']=sensib
        dict_['speci']=specic
        liste_resultats.append(dict_)
    resultats = pd.DataFrame(liste_resultats)
    thrld = find_best_threshold(resultats, 0.80)
    prdct_prb_fnl = (predict_proba.iloc[:,1]>thrld).apply(lambda x: round(x,0))
    score_final = scores(y_test, prdct_prb_fnl)
    return {'best_threshold':thrld, 'scores':score_final, 'predict_proba':prdct_prb_fnl}


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


def filtre_list(list_to_filter, list_elem):
    for elem in list_elem:
        if elem in list_to_filter:
            list_to_filter.remove(elem)
    return list_to_filter

def fit_method(model_to_train, X_train, X_test, y_train, liste_col):
    model = XGBClassifier()
    model.fit(X_train.loc[:,liste_col], y_train)
    y_pred = model.predict(X_test.loc[:,liste_col])
    return y_pred, model

def check_baseline(dict_score, seuil_FP, seuil_FN):
    FP = dict_score['FP']
    FN = dict_score['FN']
    print('FP:',FP,'FN:',FN)
    if seuil_FP>=FP and seuil_FN>=FN:
        print('Baseline done')
    else:
        raise Exception("please check the threshold values")
        
def check_FP_FN(dict_to_check, seuil_FP, seuil_FN):
    FP_to_check = dict_to_check['FP']
    FN_to_check = dict_to_check['FN']
    return FP_to_check<=seuil_FP and FN_to_check<seuil_FN
    
def check_best_results(dict_to_check, seuil_FP, seuil_FN):
    FP_to_check = dict_to_check['FP']
    FN_to_check = dict_to_check['FN']
    return FP_to_check<=seuil_FP and FN_to_check<seuil_FN

    
# ML_opti_receuil(X_train, X_test, y_train, y_test, liste_col_speci, XGBClassifier(), 3, 15)
def ML_opti_recueil(X_train, X_test, y_train, y_test, liste_col, model_to_train, seuil_FP, seuil_FN):
    '''
    take a Train/test split and a list of column for fit a baseline ML algorithm and try to improve it, column bu column.
    input:
    X_train : training features pd.DataFrame is expected
    X_test : test features pd.DataFrame is expected
    y_train : training labels pd.DataFrame is expected
    y_test : test labels pd.DataFrame is expected
    liste_col : liste of columns from the train, will be the baseline for ML
    model_to_train
    seuil_FP
    seuil_FN
    
    Return :
    Dictionary with fitted model, metrics' results, predictions
    
    '''
    # Initialisation : 
    dict_best_set = {}
    best_results ={}
    list_columns_to_try = filtre_list(list(X_train.columns), liste_col)
    y_pred_baseline, model_baseline  = fit_method(model_to_train, X_train, X_test, y_train, liste_col)
    dict_score_baseline = scores(y_test, y_pred_baseline) 
    check_baseline(dict_score_baseline, seuil_FP, seuil_FN)
    best_set = liste_col
    best_results['results'] = dict_score_baseline
    best_results['set_col'] = best_set
    
    # Loop for optimize the ML :
    for col in list_columns_to_try:
        if check_corr_col(col, best_set)==True:
            y_pred, model  = fit_method(model_to_train, X_train, y_train, best_set+[col])
            dict_score = scores(y_test, y_pred)
            if check_best_results(dict_score, seuil_FP, seuil_FN): ##### TO DO !!!!!!
                best_set = best_set + [col]
                best_results['results'] = dict_score
                dict_best_set[f'resultat_{len(dict_best_set.keys())}'] = {'set_col':best_set, 'resultat':dict_score}
        else:
            continue
    
    
    return best_results, dict_best_set, y_pred


def univariate_ml(X_train, X_test, y_train, y_test, scorer):
    '''Take splitted data set for make univariate machine learning
    
    use XGBClassifier as model
    
    Input : 
    -----------
    X_train : training data set
    X_test : test data set
    y_train : train target dataset 
    y_test : test target dataset
    scorer = dict('classic_metric':'metrics', 'custom_metric':makescorer(method))
    
    Return : 
    -----------
    resultats : dataframe getting features of dataset as columns, and metrics as index 
    '''   
    resultats = {}
    metric = {}
    for name, serie in X_train.items():     # Nicolai replaced 'iteritems' which is deprecated in pandas >= 2.0
        # Initialisation du modèle : 
        tree = XGBClassifier(random_state=42)
        # fit du modèle sur les données : 
        scores = cross_validate(tree, serie, y_train, cv=5, scoring=scorer)
        resultats[name] = {
                           'speci':round(scores['test_speci'].mean(),2),
                           'sensi':round(scores['test_sensi'].mean(), 2)
                          }
    return pd.DataFrame(resultats)

def plot(precisions, recalls, thresholds, fpr, tpr, resultat):
    fig = plt.figure(figsize=(12,10))
    
    ax1 = fig.add_subplot(221)   
    line1, = plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    line3, = plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    plt.setp(line1, linewidth=2, linestyle='--', color='b')
    plt.setp(line3, linewidth=2, linestyle='--', color='g')
    ax1.set_xlabel("Threshold")
    ax1.legend()
    ax1.set_ylim([0, 1])
    ax1.set_title('Precision and recall vs threshold')   
    
    ax2 = fig.add_subplot(222) 
    line2, = plt.step(recalls, precisions, linewidth=2)
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Evolution de la précision en fonction du recall')
    
    ax3 = fig.add_subplot(223)
    linea, = plt.plot(fpr, tpr, linewidth=2)
    lineb, = plt.plot([0, 1], [0, 1], 'k--')
    ax3.axis([0, 1, 0, 1])
    ax3.set_xlabel('False Positive Rate')
    ax3.set_ylabel('True Positive Rate (=Recall)')
    ax3.set_title('roc_curve')
    
    ax1 = fig.add_subplot(224)   
    line_5, = plt.plot(resultat.trhd, resultat.sensi, label="Sensibilité")
    line_6, = plt.plot(resultat.trhd, resultat.speci, label="Spécificité")
    plt.setp(line_5, linewidth=2, linestyle='--', color='orange')
    plt.setp(line_6, linewidth=2, linestyle='--', color='red')
    ax1.set_xlabel("Threshold")
    ax1.legend()
    ax1.set_ylim([0, 1])
    ax1.set_title('Sensibilité and Spécificité vs threshold')
    plt.tight_layout()
    plt.show()
    
def model_to_plot(model, x_test, y_test):
    y_pred = model.predict(x_test)
    y_test_proba = model.predict_proba(x_test)   
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_test_proba[:,1])
    fpr, tpr, thresholds_ = roc_curve(y_test, y_test_proba[:,1])
    resultat = from_model_to_bst_trhld2(model, x_test, y_test)
    plot(precisions, recalls, thresholds, fpr, tpr, resultat)

def kfold_cv_stratified(X, Y, classifier, max_vocab, nfolds=5):
    """
    Fits a k-fold cross validation while making sure that the distribution of labels in the features are the same in train and test set.
    
    Parameters
    ----------
    X : pd.Dataframe
        Dataframe containing the features
    Y : pd.DataFrame
        Dataframe containing the target labels
    max_vocab : int
        Maximum vocabulary of the text
    nfolds : int
        Number of folds

    Returns
    -------
    Y_pred_folds : pd.Dataframe
        Predictions
    classifier : Fitted classifier

    Author: Nicolai Wolpert
    Date: 25.06.2024
    """

    X.reset_index(inplace=True, drop=True)
    Y.reset_index(inplace=True, drop=True)
    
    Y_pred_folds = Y.copy().reset_index(drop=True)
    kfold = IterativeStratification(n_splits = nfolds)
    for idx_train, idx_test in kfold.split(X, Y):
        X_train_kfold = X.loc[idx_train]
        X_test_kfold = X.loc[idx_test]
        Y_train_kfold = Y.loc[idx_train]
        Y_test_kfold = Y.loc[idx_test]
        tfIdfVectorizer=TfidfVectorizer(use_idf=True, max_features=max_vocab, lowercase=False)
        X_train_kfold_fitted = tfIdfVectorizer.fit_transform(X_train_kfold.Résumé)
        X_train_kfold_fitted_df = pd.DataFrame(X_train_kfold_fitted.todense(), columns=tfIdfVectorizer.get_feature_names_out(), index= X_train_kfold.Anonymisation)
        X_test_kfold_fitted = tfIdfVectorizer.transform(X_test_kfold.Résumé)
        X_test_kfold_fitted_df = pd.DataFrame(X_test_kfold_fitted.todense(), columns=tfIdfVectorizer.get_feature_names_out(), index= X_test_kfold.Anonymisation)

        classifier.fit(X_train_kfold_fitted_df, Y_train_kfold)
        if Y.shape[1]==1:
            Y_pred_folds.loc[idx_test] = classifier.predict(X_test_kfold_fitted_df).reshape(len(idx_test), 1)
        else:
            Y_pred_folds.loc[idx_test] = classifier.predict(X_test_kfold_fitted_df)

    return Y_pred_folds, classifier