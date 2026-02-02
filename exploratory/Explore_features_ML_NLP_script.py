"""
Author: Nicolai WolpertEmail: nicolai.wolpert@capgemini.com
October 2024
Ce notebook utilise le meilleur modèle d'apprentissage automatique pour prédire l'endométriose et explore l'impact des différentes features/symptômes des patientes sur la classification. 
L'objectif est d'avoir le moins de symptômes possible tout en obtenant la meilleure prédiction possible.
Car ces symptômes devront être prédits en utilisant du NLP, et pour cela, moins il y a de symptômes, plus cela est facile.

"""


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

# Custom utils  
from Opti_utils.ML_utils import find_best_threshold, from_model_to_bst_trhld, custom_metrics, scores, compare_results,find_error
from Opti_utils.ML_utils import FP, TP, FN, TN, check_corr_col, filtre_list, fit_method, check_baseline, check_FP_FN, check_best_results
from Opti_utils.ML_utils import ML_opti_recueil, univariate_ml, sensi, speci, mix_sensi_speci, custom_metric
from Opti_utils.ML_utils import Binarisation, model_to_plot, clean_lists

def data_processing():
     #OPENING AND DATA MANAGEMENT
    """-------------------------modif------------------------------"""
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

    #PREPARATION POUR LE ML
    #SPLIT DES FEATURES ET TARGET

    # recueil_imc.dropna(axis=0, inplace=True)
    target = recueil_imc.iloc[:,-4:].copy()
    features = recueil_imc.iloc[:,:-4].copy()

    endometriose = target.loc[:,['anapath.lusd','anapath.lusg','anapath.torus']].sum(axis=1).apply(lambda x: Binarisation(x))

    recueil_imc_endo = recueil_imc.copy()
    recueil_imc_endo['endometriose'] = endometriose
    print(recueil_imc_endo.shape)

    # Voici les colonnes qui ne sont pas disponible pour le NLP normalement
    features_to_drop_for_nlp = ['age', 'imc', 'g', 'p', 'sf.dsp.eva', 'sf.dsm.eva']

    #PREPARATION DES DONNEES
    features_chir_ONE = pd.get_dummies(features.loc[:,'chir'], prefix='chir')
    features_dsptype_ONE = pd.get_dummies(features.loc[:,'sf.dsp.type'].replace(0, 'aucun'), prefix='dsp.type')
    features_enc = pd.concat([features.drop('chir', axis=1).drop('sf.dsp.type', axis=1), features_chir_ONE, features_dsptype_ONE], axis=1)

    #print(features_enc)

    # split 
    X_train, X_test, y_train, y_test = train_test_split(features_enc, endometriose, random_state=42, stratify=endometriose)

    #print(features_enc)

    scorer = make_scorer(custom_metric, greater_is_better=True)
    mix_recall = make_scorer(mix_sensi_speci, greater_is_better=True)
    sensibilite = make_scorer(sensi, greater_is_better=True)
    specificite = make_scorer(speci, greater_is_better=True)
    scorers = { 'speci': specificite, 'sensi' : sensibilite}

    #RESULTATS

    with open('/home/nounou/endopath/archives/Data/DATA_PROCESSED/data.pkl', 'rb') as f1:
        dictionnary_list = pickle.load(f1)
        
    liste_col_speci= dictionnary_list['specifite']
    liste_col_sensi = dictionnary_list['sensibilite']
    liste_col_mixte = dictionnary_list['moyenne']

    #print(liste_col_speci)

    liste_col_sensi = clean_lists(liste_col_sensi)
    liste_col_speci = clean_lists(liste_col_speci)
    liste_col_mixte = clean_lists(liste_col_mixte)


    """--------------------modif----------------------------------"""
    #if __name__ == "__main__":
    # Code qui ne doit pas être exécuté lors de l'importation
    

    #ESSAIS DE ML AVEC LES COLONNES

    seed=42

    ## Init Metrics :
    scorer = make_scorer(custom_metric, greater_is_better=True)
    mix_recall = make_scorer(mix_sensi_speci, greater_is_better=True)
    sensibilite = make_scorer(sensi, greater_is_better=True)
    specificite = make_scorer(speci, greater_is_better=True)

    # Params :
    param1 = {'min_child_weight': [1, 5, 10],
    'gamma': [0, 0.5, 1, 1.5, 2, 5],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'max_depth': [3, 4, 5, 6, 7, 8],
    'n_estimators': [50, 100, 150, 200, 300, 500]}

    # MACHINE LEARNING

    # On enlève les colonnes correlées : 
    X_train_ = X_train.copy()
    X_test_ = X_test.copy()

    liste_to_drop = ['p','atcd.chir.endo','atcd.infertilite','sf.dsm','sf.dsp','ef.hormone.dsm','echo.oma','irm.xr']
    for col in liste_to_drop:
        X_test_.drop(col, axis=1, inplace=True)
        X_train_.drop(col, axis=1, inplace=True)

    tree_1 = XGBClassifier(random_state=seed)
    test_cv = cross_validate(tree_1, X_train_, y_train, cv=5, scoring=scorers, return_estimator =True)

    print('sensi : ',round(test_cv['test_sensi'].mean(), 2))
    print('speci : ',round(test_cv['test_speci'].mean(),2))

    #Training
    print("Training")

    tree_BL = XGBClassifier(random_state=seed, )
    tree_BL.fit(X_train_, y_train)
    pred_BL = tree_BL.predict(X_test_)
    #model_to_plot(tree_BL, X_test_, y_test)
    resultat_BL = pd.DataFrame(pd.Series(scores(y_test, pred_BL)), columns=['XGB_BL_sensi'])
    print('threshold = 0,5')
    print('Sensibilité : ', resultat_BL.loc['sensibilite','XGB_BL_sensi'])
    print('Spécificité : ', resultat_BL.loc['specificité','XGB_BL_sensi'])
    dict_ = from_model_to_bst_trhld(tree_BL, X_test_, y_test)
    print('_________________________________________________________')
    print('Best threshold :', dict_['best_threshold'])
    print('Sensibilité :', dict_['scores']['sensibilite'])
    print('Spécificité :', dict_['scores']['specificité'])



    # Show feature importance
    xgb_fea_imp=pd.DataFrame(list(tree_BL.get_booster().get_fscore().items()),
    columns=['feature','importance']).sort_values('importance', ascending=False)
    #print('',xgb_fea_imp)
    #xgb_fea_imp.to_csv('xgb_fea_imp.csv')

    #ax = plot_importance(tree_BL, )
    #ax.figure.set_size_inches(10,8)

    # Show feature importance
    xgb_fea_imp=pd.DataFrame(list(tree_BL.get_booster().get_fscore().items()),
    columns=['feature','importance']).sort_values('importance', ascending=False)
    #print('',xgb_fea_imp)
    #ax = plot_importance(tree_BL, max_num_features=20)


    print("Encore des bonnes performances si on enlève les colonnes pas disponible pour la partie NLP?")

    X_train_nlp_features = X_train.copy()
    X_test_nlp_features = X_test.copy()

    for col in features_to_drop_for_nlp:
        X_test_nlp_features.drop(col, axis=1, inplace=True)
        X_train_nlp_features.drop(col, axis=1, inplace=True)


    tree_BL = XGBClassifier(random_state=seed)
    tree_BL.fit(X_train_nlp_features, y_train)
    pred_BL = tree_BL.predict(X_test_nlp_features)
    #model_to_plot(tree_BL, X_test_nlp_features, y_test)
    resultat_BL = pd.DataFrame(pd.Series(scores(y_test, pred_BL)), columns=['XGB_BL_sensi'])
    print('threshold = 0,5')
    print('Sensibilité : ', resultat_BL.loc['sensibilite','XGB_BL_sensi'])
    print('Spécificité : ', resultat_BL.loc['specificité','XGB_BL_sensi'])
    dict_ = from_model_to_bst_trhld(tree_BL, X_test_nlp_features, y_test)
    print('_________________________________________________________')
    print('Best threshold :', dict_['best_threshold'])
    print('Sensibilité :', dict_['scores']['sensibilite'])
    print('Spécificité :', dict_['scores']['specificité'])

    print("Garder infos démographiques mais pas les evaluations")

    X_train_without_eva = X_train.copy()
    X_test_without_eva = X_test.copy()

    for col in X_train_without_eva.columns:
        if 'eva' in col:
            X_train_without_eva.drop(col, axis=1, inplace=True)
            X_test_without_eva.drop(col, axis=1, inplace=True)

    tree_BL = XGBClassifier(random_state=seed)
    tree_BL.fit(X_train_without_eva, y_train)
    pred_BL = tree_BL.predict(X_test_without_eva)
    #model_to_plot(tree_BL, X_test_without_eva, y_test)
    resultat_BL = pd.DataFrame(pd.Series(scores(y_test, pred_BL)), columns=['XGB_BL_sensi'])
    print('threshold = 0,5')
    print('Sensibilité : ', resultat_BL.loc['sensibilite','XGB_BL_sensi'])
    print('Spécificité : ', resultat_BL.loc['specificité','XGB_BL_sensi'])
    dict_ = from_model_to_bst_trhld(tree_BL, X_test_without_eva, y_test)
    print('_________________________________________________________')
    print('Best threshold :', dict_['best_threshold'])
    print('Sensibilité :', dict_['scores']['sensibilite'])
    print('Spécificité :', dict_['scores']['specificité'])

    print("Modèle avec que les infos démographiques")

    X_train_only_demographic = X_train[['age', 'imc', 'g']].copy()
    X_test_only_demographic = X_test[['age', 'imc', 'g']].copy()

    tree_BL = XGBClassifier(random_state=seed)
    tree_BL.fit(X_train_only_demographic, y_train)
    pred_BL = tree_BL.predict(X_test_only_demographic)
    #model_to_plot(tree_BL, X_test_only_demographic, y_test)
    resultat_BL = pd.DataFrame(pd.Series(scores(y_test, pred_BL)), columns=['XGB_BL_sensi'])
    print('threshold = 0,5')
    print('Sensibilité : ', resultat_BL.loc['sensibilite','XGB_BL_sensi'])
    print('Spécificité : ', resultat_BL.loc['specificité','XGB_BL_sensi'])
    dict_ = from_model_to_bst_trhld(tree_BL, X_test_only_demographic, y_test)
    print('_________________________________________________________')
    print('Best threshold :', dict_['best_threshold'])
    print('Sensibilité :', dict_['scores']['sensibilite'])
    print('Spécificité :', dict_['scores']['specificité']) 

    print("Modèle avec que les infos démographiques +  eva")

    X_train_only_demographic_eva = X_train[['age', 'imc', 'g','sf.dsp.eva', 'sf.dsm.eva']].copy()
    X_test_only_demographic_eva = X_test[['age', 'imc', 'g', 'sf.dsp.eva', 'sf.dsm.eva']].copy()

    tree_BL = XGBClassifier(random_state=seed)
    tree_BL.fit(X_train_only_demographic_eva, y_train)
    pred_BL = tree_BL.predict(X_test_only_demographic_eva)
    #model_to_plot(tree_BL, X_test_only_demographic_eva, y_test)
    resultat_BL = pd.DataFrame(pd.Series(scores(y_test, pred_BL)), columns=['XGB_BL_sensi'])
    print('threshold = 0,5')
    print('Sensibilité : ', resultat_BL.loc['sensibilite','XGB_BL_sensi'])
    print('Spécificité : ', resultat_BL.loc['specificité','XGB_BL_sensi'])
    dict_ = from_model_to_bst_trhld(tree_BL, X_test_only_demographic_eva, y_test)
    print('_________________________________________________________')
    print('Best threshold :', dict_['best_threshold'])
    print('Sensibilité :', dict_['scores']['sensibilite'])
    print('Spécificité :', dict_['scores']['specificité'])

    ######################

    features_importances_symptoms = xgb_fea_imp.loc[~xgb_fea_imp['feature'].isin(features_to_drop_for_nlp)]
    #print(features_importances_symptoms)

    ######################

    print("Explorer performance du modèle en fonction de quels symptômes sont inclus")

    cutoff_feature_importance = 2     # Symptômes avec feature importance en dessous sont rejetés
    include_demographic = True
    include_eva = False

    important_symptoms = list(features_importances_symptoms.loc[features_importances_symptoms.importance >= cutoff_feature_importance]['feature'].values)

    features_subset = important_symptoms
    if include_demographic:
        features_subset = ['age', 'imc', 'g'] + features_subset
    if include_eva:
        features_subset = ['sf.dsp.eva', 'sf.dsm.eva'] + features_subset
    X_train_feature_subset = X_train[features_subset].copy()
    X_test_feature_subset = X_test[features_subset].copy()

    tree_BL = XGBClassifier(random_state=seed)
    tree_BL.fit(X_train_feature_subset, y_train)
    pred_BL = tree_BL.predict(X_test_feature_subset)
    #model_to_plot(tree_BL, X_test_feature_subset, y_test)
    resultat_BL = pd.DataFrame(pd.Series(scores(y_test, pred_BL)), columns=['XGB_BL_sensi'])
    print('threshold = 0,5')
    print('Sensibilité : ', resultat_BL.loc['sensibilite','XGB_BL_sensi'])
    print('Spécificité : ', resultat_BL.loc['specificité','XGB_BL_sensi'])
    dict_ = from_model_to_bst_trhld(tree_BL, X_test_feature_subset, y_test)
    print('_________________________________________________________')
    print('Best threshold :', dict_['best_threshold'])
    print('Sensibilité :', dict_['scores']['sensibilite'])
    print('Spécificité :', dict_['scores']['specificité'])


    print("Systematically change the feature importance cutoff")
    feature_importance_cutoffs = list(range(round(np.floor(np.min(features_importances_symptoms['importance']))), round(np.ceil(np.max(features_importances_symptoms['importance'])))))


    nsymptoms = []
    best_thresholds = []
    sensitivities = []
    specificities = []
    features_included = []
    for feat_imp_cuttoff in feature_importance_cutoffs:

        important_symptoms = list(features_importances_symptoms.loc[features_importances_symptoms.importance >= feat_imp_cuttoff]['feature'].values)
        nsymptoms = nsymptoms + [len(important_symptoms)]

        features_subset = important_symptoms
        if include_demographic:
            features_subset = ['age', 'imc', 'g'] + features_subset
        if include_eva:
            features_subset = ['sf.dsp.eva', 'sf.dsm.eva'] + features_subset
        X_train_feature_subset = X_train[features_subset].copy()
        X_test_feature_subset = X_test[features_subset].copy()
        
        tree_BL = XGBClassifier(random_state=seed)
        tree_BL.fit(X_train_feature_subset, y_train)
        pred_BL = tree_BL.predict(X_test_feature_subset)
        #model_to_plot(tree_BL, X_test_feature_subset, y_test)
        resultat_BL = pd.DataFrame(pd.Series(scores(y_test, pred_BL)), columns=['XGB_BL_sensi'])
        dict_ = from_model_to_bst_trhld(tree_BL, X_test_feature_subset, y_test)

        best_thresholds = best_thresholds + [dict_['best_threshold']]
        sensitivities = sensitivities + [dict_['scores']['sensibilite']]
        specificities = specificities + [dict_['scores']['specificité']]
    results_by_feature_imp_cutoff = pd.DataFrame({'feature_importance_cutoffs': feature_importance_cutoffs, 'nsymptoms': nsymptoms, 'best_thresholds': best_thresholds, 'sensitivities': sensitivities, 'specificities': specificities})
    results_by_feature_imp_cutoff.set_index('feature_importance_cutoffs', inplace=True)
    print(results_by_feature_imp_cutoff)

    """
    code pour les graphs a ajouter ?
    """

    cutoff_feature_importance = 14
    important_symptoms = list(features_importances_symptoms.loc[features_importances_symptoms.importance >= cutoff_feature_importance]['feature'].values)
    print(f'Included symptoms with a cutoff of {cutoff_feature_importance}:')
    print(important_symptoms)

    pass


    """------------------------------------------------------"""

"""-------------------------test main ------------------------------"""
"""def main():
    print("Lancement du script Explore_features_ML_NLP_script.py en mode test principal.")
    try:
        # Appel de la fonction principale de traitement des données si elle existe
        data_processing()
        print("Traitement des données terminé avec succès.")
    except Exception as e:
        print("Erreur lors du test :", e)

if __name__ == "__main__":
    main()"""