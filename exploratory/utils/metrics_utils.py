'''
Author : Maxime Mock
Date : 25/01/2023

Purpose :  Functions for edit classification report and metric report
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#import eli5
import sys
from sklearn.metrics import classification_report, recall_score, f1_score, precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from IPython.display import display

def rewrite_keys(dictionary: dict(), nom):
    dict_copy = dictionary.copy()
    for key in dictionary.keys():
        dict_copy[(nom, key)] = dict_copy.pop(key)
    return dict_copy

# def from_CM_to_recall(CM, rank=1):
#     sum_ = CM[rank].sum()
#     return CM[rank][rank]/sum_

# def from_CM_to_precision(CM, rank=1):
#     CM = CM.T
#     sum_ = CM[rank].sum()
#     return CM[rank][rank]/sum_

def recalls(y_true, y_pred):
    scores = recall_score(y_true, y_pred, average=None, zero_division=0)
    return dict_resultats(y_true, y_pred, scores, 'recall')

def precisions(y_true, y_pred):
    scores = precision_score(y_true, y_pred, average=None, zero_division=0)
    return dict_resultats(y_true, y_pred, scores, 'precision')

def dict_resultats(y_true, y_pred, score, string):
    classes = np.unique(np.concatenate((y_true, y_pred)))
    length = len(classes)
    # classes_t = np.unique(y_true)
    # n_class_t = len(classes_t)
    # classes_p = np.unique(y_pred)
    # n_class_p = len(classes_p)
    if length == len(score):
        dictionnaire = {('metrics', string+'_'+str(int(class_))):score_ for (class_, score_) in zip(classes, score)}
    else:
        print('classes :', classes, 'score :', score)
        raise ValueError('The number of classes and the number of score don\'t match')
            
    return dictionnaire

def F1_scores(y_true, y_pred):
    scores = f1_score(y_true, y_pred, average=None, zero_division=0)
    return dict_resultats(y_true, y_pred, scores, 'f1_score')

def rapport_metrics_decision_tree(Y_test, Y_pred):
    #init :
    dict_pred = {}
    dict_metric = {}
    # liste_pred= []
    for nom, labels in Y_test.items(): 
        # Calcul des métriques :
        dict_f1 = F1_scores(Y_test.loc[:,nom], Y_pred.loc[:,nom])
        dict_precision = precisions(Y_test.loc[:,nom], Y_pred.loc[:,nom])
        dict_recall = recalls(Y_test.loc[:,nom], Y_pred.loc[:,nom])
        dict_f1.update(dict_precision)
        dict_f1.update(dict_recall)
        dict_metric[nom] = dict_f1
        # Préparation des dictionnaires pour concaténations : 
        dict_pred[(nom, 'y_true')]=list(Y_test[nom])
        dict_pred[(nom, 'y_pred')]=list(Y_pred.loc[:,nom])
        # liste_pred.append(Y_pred.loc[:,nom])
    # concaténation des informations dans des datasets : 
    DT_Multi_index = pd.DataFrame(dict_pred, index=Y_test.index)
    CR_global = pd.DataFrame(dict_metric)
    CR_global.sort_index(inplace=True)
    CR_global.sort_values(by=('metrics','f1_score_1'), axis=1, ascending=False, inplace=True)
    return DT_Multi_index, CR_global

def evaluate_results_multilabel(DT_Multi_index, CR_global, label=None):
    """
    Shows evaluations of predictions from any multilabel classification model
    
    Parameters
    ----------
    DT_Multi_index : pd.Dataframe
        Dataframe containing the true values and predictions for each feature
    CR_global : pd.DataFrame
        Dataframe containing the f1, precision and recall scores

    label : list
        Custom list of labels

    Returns
    -------
    scores : json dictionnary
        results for all metrics

    Author: Nicolai Wolpert
    Date: 20.06.2024
    """
    
    all_features = [c[0] for c in DT_Multi_index.columns]

    FPs = []
    FNs = []
    TPs = []
    TNs = []
    for feat in all_features:
        FP = DT_Multi_index[(DT_Multi_index[feat]['y_true'] == 0) & (DT_Multi_index[feat]['y_pred'] == 1)].shape[0] / len(DT_Multi_index[feat]['y_true'])
        FN = DT_Multi_index[(DT_Multi_index[feat]['y_true'] == 1) & (DT_Multi_index[feat]['y_pred'] == 0)].shape[0] / len(DT_Multi_index[feat]['y_true'])
        TP = DT_Multi_index[(DT_Multi_index[feat]['y_true'] == 1) & (DT_Multi_index[feat]['y_pred'] == 1)].shape[0] / len(DT_Multi_index[feat]['y_true'])
        TN = DT_Multi_index[(DT_Multi_index[feat]['y_true'] == 0) & (DT_Multi_index[feat]['y_pred'] == 0)].shape[0] / len(DT_Multi_index[feat]['y_true'])
        FPs += [FP]
        FNs += [FN]
        TPs += [TP]
        TNs += [TN]

    average_precision_micro = round(np.sum(TPs) / (np.sum(TPs) + np.sum(FPs)), 2)
    average_recall_micro = round(np.sum(TPs) / (np.sum(TPs) + np.sum(FNs)), 2)
    average_f1_micro = round(2 * average_precision_micro * average_recall_micro / (average_precision_micro + average_recall_micro), 2)

    average_f1_macro = round(np.nanmean(CR_global.loc['metrics'].loc['f1_score_1']), 2)
    average_precision_macro = round(np.nanmean(CR_global.loc['metrics'].loc['precision_1']), 2)
    average_recall_macro = round(np.nanmean(CR_global.loc['metrics'].loc['recall_1']), 2)

    accuracies = []
    exact_matches = []
    for patient in DT_Multi_index.index:
        ncorrect = 0
        for feat in all_features:
            if DT_Multi_index.loc[patient, feat]['y_true'] == DT_Multi_index.loc[patient, feat]['y_pred']:
                ncorrect += 1
        accuracy = ncorrect/len(all_features)
        accuracies += [accuracy]
    average_accuracy = np.round(np.mean(accuracies), 2)

    y_true = DT_Multi_index[[col for col in DT_Multi_index.columns if 'y_true' in col]]
    y_pred = DT_Multi_index[[col for col in DT_Multi_index.columns if 'y_pred' in col]]
    exact_matches = (y_true.values == y_pred.values).all(axis=1)
    n_exact_matches = np.sum(exact_matches)
    exact_match_ratio = n_exact_matches / DT_Multi_index.shape[0]

    print(f'##### Macro-averaged f1-score:  {average_f1_macro} #####')
    print(f'##### Macro-averaged precision: {average_precision_macro} #####')
    print(f'##### Macro-averaged recall:    {average_recall_macro} #####')
    print()
    print(f'##### Micro-averaged f1-score:  {average_f1_micro} #####')
    print(f'##### Micro-averaged precision: {average_precision_micro} #####')
    print(f'##### Micro-averaged recall:    {average_recall_micro} #####')
    print()
    print(f'##### Average accuracy across patients:  {average_accuracy} #####')
    print()
    print(f'##### Exact match ratio:  {exact_match_ratio} #####')

    df_scores = pd.DataFrame(columns=['f1', 'precision', 'recall'])
    df_scores['f1'] = CR_global.loc['metrics'].loc['f1_score_1']
    df_scores['precision'] = CR_global.loc['metrics'].loc['precision_1']
    df_scores['recall'] = CR_global.loc['metrics'].loc['recall_1']
    df_scores = df_scores.reset_index().melt(id_vars='index', var_name='metric', value_name='score')
    df_scores.rename(columns={'index': 'feature'}, inplace=True)
    df_scores = df_scores.dropna()

    fig, axs = plt.subplots(nrows=1, ncols=2, gridspec_kw={'width_ratios': [5, 2]}, figsize=(10, 5))
    axs = axs.flatten()

    sns.barplot(x='metric', y='score', hue='metric', data=df_scores, ax=axs[0])
    axs[0].set_title('Macro-averaged metrics across all features for positive class')
    axs[0].set_xlabel('Metric')
    axs[0].set_ylabel('Score')

    sns.histplot(accuracies, ax=axs[1])
    axs[1].set_title(f'Accuracies across patients (avg = {average_accuracy})')

    plt.tight_layout()
    plt.show()

    scores = {'f1_macroaverage': average_f1_macro, 'precision_macroaverage': average_precision_macro, 'recall_macroaverage': average_recall_macro,
            'f1_microaverage': average_f1_macro, 'precision_microaverage': average_precision_macro, 'recall_microaverage': average_recall_macro,
            'accuracy_avg': average_accuracy, 'exact_match_ratio': exact_match_ratio}

    return scores

def show_best_and_lowest_scores(CR_global, score='f1_score', nlargest_and_lowest=20):
    """
    Plots the distribution of classif. score of interest
    
    Parameters
    ----------
    CR_global : pd.DataFrame
        pd.Dataframe containing the true values and predictions for each feature
    score : str
        Score of interest ('f1_score', 'precision' or 'recall')
    nlargest_and_lowest : int
        The N largest and N lowest scores will be plotted

    Returns
    -------
    /

    Author: Nicolai Wolpert
    Date: 20.06.2024
    """
    
    score = score + '_1'
    df_plot = CR_global.loc['metrics'].T.reset_index().rename(columns={'index': 'feature'}).sort_values(by=score, ascending=False)

    if nlargest_and_lowest >= len(CR_global.columns):
        fig, ax = plt.subplots(figsize=(18, 5))
        sns.barplot(x='feature', y=score, data=df_plot, orient='v', ax=ax)
        ax.set_ylabel(score[:-2])
        ax.set_title(f'{score[:-2]} for Class 1 by feature')
        ax.tick_params(rotation=70)
    else:
        df_plot = df_plot.loc[df_plot[score]>0]
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(18, 5))
        axs = axs.flatten()
        sns.barplot(x='feature', y=score, data=df_plot.nlargest(nlargest_and_lowest, score), orient='v', ax=axs[0])
        axs[0].set_ylabel(score[:-2])
        axs[0].set_title(f'{score[:-2]} for Class 1 by feature')
        axs[0].tick_params(rotation=70)


        sns.barplot(x='feature', y='f1_score_1', data=df_plot.nsmallest(nlargest_and_lowest, score).iloc[::-1], orient='v', ax=axs[1])
        axs[1].set_ylabel(score[:-2])
        axs[1].set_title(f'{score[:-2]} for Class 1 by feature')
        axs[1].set_ylim(axs[0].get_ylim())
        axs[1].tick_params(rotation=70)
    plt.show()

def show_precision_vs_recall(CR_global, features_of_interest=None, normalize_scale=True):
    """
    Plots the distribution of classif. score of interest
    
    Parameters
    ----------
    CR_global : pd.DataFrame
        pd.Dataframe containing the true values and predictions for each feature
    features_of_interest : list
        List of strings of features that are to be highlighted in the figure (e.g. those features most valuable for endometriosis prediction)

    Returns
    -------
    /
    
    Author: Nicolai Wolpert
    Date: 20.06.2024
    """

    # Extract precision_1 and recall_1
    df_plot = CR_global.loc['metrics'].copy()
    precision_1 = df_plot.loc['precision_1']
    recall_1 = df_plot.loc['recall_1']

    # Filter out NaN values
    valid_indices = precision_1.notna() & recall_1.notna()
    precision_1 = precision_1[valid_indices]
    recall_1 = recall_1[valid_indices]

    fig, ax = plt.subplots(figsize=(8, 8))
    if features_of_interest != None:
        sns.scatterplot(x=precision_1, y=recall_1, hue=precision_1.index.isin(features_of_interest), palette={True: 'red', False: 'blue'}, s=100, edgecolor='w')
    else:
        sns.scatterplot(x=precision_1, y=recall_1, s=100, edgecolor='w')

    # Ajouter des labels pour chaque point
    for feature in precision_1.index:
        plt.text(precision_1[feature], recall_1[feature], feature, fontsize=9, ha='right')

    if normalize_scale:
        plt.xlim([0, 1])
        plt.ylim([0, 1])
    plt.xlabel('Precision (Class 1)')
    plt.ylabel('Recall (Class 1)')
    plt.title('Precision vs Recall for Class 1')
    if features_of_interest != None:
        handles, labels = ax.get_legend_handles_labels()
        labels = ['Other Features', 'Best features for endo. prediction']
        plt.legend(handles=handles, labels=labels, loc='best')
    plt.show()
    return fig, ax

def reshape_df_explain(df_to_transform):
    shape = df_to_transform.shape
    cols = list(df_to_transform)
    rows = list(df_to_transform.index)
    col_len = len(cols)
    row_len = len(rows)
    dict_ = {}
    for row in range(row_len):
        for col in range(col_len):        
            dict_[(rows[row],cols[col])] = df_to_transform.iloc[row,col]
    return dict_

def multilabel_multioutput_svc(X_train, X_test, Y_train, Y_test, vec):        
    #init :
    dict_pred = {}
    dict_pred_proba = {}
    dict_metric = {}
    liste_pred= []
    dict_model = {}
    # Boucle pour fit les SVC
    for idx, (nom, labels) in enumerate(Y_train.items()):
        model_to_fit = SVC(kernel='linear', random_state=42)
        model_to_fit_proba = SVC(kernel='linear', probability=True)
        label = labels.unique()
        # Fit du model : 
        model_to_fit.fit(X_train, labels)
        model_to_fit_proba.fit(X_train, labels)
        dict_model[nom]=model_to_fit
        # Prédiction : 
        Y_pred = model_to_fit.predict(X_test)
        Y_pred_proba = model_to_fit_proba.predict_proba(X_test)
        # Calcul des métriques :
        dict_f1 = F1_scores(Y_test.loc[:,nom].values, Y_pred)
        dict_precision = precisions(Y_test.loc[:,nom].values, Y_pred)
        dict_recall = recalls(Y_test.loc[:,nom].values, Y_pred)
        dict_f1.update(dict_precision)
        dict_f1.update(dict_recall)
        '''
        # eli5 package not working anymore
        #explications NLP :
        df_to_transform = eli5.explain_weights_df(model_to_fit, vec=vec, top=6, target_names=label)
        if type(df_to_transform)==pd.core.frame.DataFrame:
            dict_explication = reshape_df_explain(df_to_transform.iloc[0:6,:])
            dict_f1.update(dict_explication)
        '''
        dict_metric[nom] = dict_f1
        # Préparation des dictionnaires/listes pour concaténations : 
        dict_pred[(nom, 'y_true')]=list(Y_test[nom])
        dict_pred[(nom, 'y_pred')]=list(Y_pred)
        dict_pred_proba[(nom, 'y_true')]=list(Y_test[nom])
        dict_pred_proba[(nom, 'y_pred_proba')]=list(Y_pred_proba)
        liste_pred.append(Y_pred)
    # concaténation des informations dans des datasets : 
    LR_Multi_index = pd.DataFrame(dict_pred, index=Y_test.index)
    LR_Multi_index_proba = pd.DataFrame(dict_pred_proba, index=Y_test.index)
    LR_y_pred = pd.DataFrame(liste_pred,columns=Y_test.index, index=Y_train.columns).T
    CR_global = pd.DataFrame(dict_metric)
    CR_global.sort_index(inplace=True)
    CR_global.sort_values(by=('metrics','f1_score_1'), axis=1, ascending=False, inplace=True)
    print('done')
    return CR_global, LR_Multi_index, LR_y_pred, dict_model, LR_Multi_index_proba

def multilabel_multioutput_LR(X_train, X_test, Y_train, Y_test, vec):   
    #init :
    # storage for y_pred and y_true :
    dict_pred = {}
    # storage for metrics :
    dict_metric = {}
    # Storage for Y_pred
    liste_pred= []
    # Storage for models : 
    dict_model = {}
    # Boucle pour fit les logReg
    for idx, (nom, labels) in enumerate(Y_train.items()):
        model_to_fit = LogisticRegression(multi_class='auto')
        label = labels.unique()
        # Fit du model : 
        model_to_fit.fit(X_train, labels)
        dict_model[nom]=model_to_fit
        # Prédiction : 
        Y_pred = model_to_fit.predict(X_test)
        # Calcul des métriques :
        dict_f1 = F1_scores(Y_test.loc[:,nom].values, Y_pred)
        dict_precision = precisions(Y_test.loc[:,nom].values, Y_pred)
        dict_recall = recalls(Y_test.loc[:,nom].values, Y_pred)
        dict_f1.update(dict_precision)
        dict_f1.update(dict_recall)
        '''
        # eli5 package not working anymore
        #explications NLP :
        df_to_transform = eli5.explain_weights_df(model_to_fit, vec=vec, top=6, target_names=label)
        if type(df_to_transform)==pd.core.frame.DataFrame:
            dict_explication = reshape_df_explain(df_to_transform.iloc[0:6,:])
            dict_f1.update(dict_explication)
        '''
        dict_metric[nom] = dict_f1
        # Préparation des dictionnaires/listes pour concaténations : 
        dict_pred[(nom, 'y_true')]=list(Y_test[nom])
        dict_pred[(nom, 'y_pred')]=list(Y_pred)
        liste_pred.append(Y_pred)
    # concaténation des informations dans des datasets : 
    LR_Multi_index = pd.DataFrame(dict_pred, index=Y_test.index)
    LR_y_pred = pd.DataFrame(liste_pred,columns=Y_test.index, index=Y_train.columns).T
    CR_global = pd.DataFrame(dict_metric)
    CR_global.sort_index(inplace=True)
    CR_global.sort_values(by=('metrics','f1_score_1'), axis=1, ascending=False, inplace=True)
    print('done')
    return CR_global, LR_Multi_index, LR_y_pred, dict_model

def custom_show_weights(dict_model, vec, nom):
    clf = dict_model[nom]
    display(eli5.show_weights(clf, top=10, vec=vec))



# df_to_transform = eli5.explain_weights_df(logregcv, vec=tfIdfVectorizer, top=6, target_names=label_temp)
# cols = list(df_to_transform)
# rows = list(df_to_transform.index)
# col_len = len(cols)
# row_len = len(rows)
# dict_ = {}
# for row in range(row_len):
#     for col in range(col_len):        
#         dict_[f'{str(rows[row])}_{cols[col]}'] = df_to_transform.iloc[row,col]