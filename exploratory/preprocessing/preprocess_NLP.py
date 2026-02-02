'''
Author : Maxime Mock
Date : 30/11/2022

Purpose :  Functions for prepocessing of features and target for NLP process
'''
## Imports : 
import pandas as pd
import numpy as np
import string
import nltk
nltk.download('stopwords')
import re
from spellchecker import SpellChecker
from unidecode import unidecode
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder

## Select same index for X and Y : 

def check_n_ano(recueil, df_nlp):
    '''
    Vérifie que dans les deux jeux de données il y ait bien les mêmes numéros d'anonymisation afin de faire un train_test_split correct.
    Input :
    -----------------
    recueil = Target
    df_nlp = Features
    Output :
    -----------------
    Recueil_checked =  le recueil avec les n_ano correspondant à celui du df_nlp
    df_nlp_checked = le df_nlp avec les n_ano correspondant à celui du recueil
    '''
    # Remove missing information from recueil : 
    n_nan_patiente = recueil.isnull().sum(axis=1)
    liste_patiente_to_drop = [patiente for patiente, count_nan in n_nan_patiente.items() if count_nan >80]
    recueil.drop(liste_patiente_to_drop, axis=0, inplace=True)
    
    set_n_ano_nlp = set(df_nlp.Anonymisation)
    set_n_ano_recueil = set(recueil.index)
    
    # Intersection :
    intersection = set_n_ano_nlp.intersection(set_n_ano_recueil)
    
    # Resize data : 
    Recueil_checked = recueil.loc[list(intersection),:]
    df_nlp_checked = df_nlp.set_index('Anonymisation').loc[list(intersection),:].reset_index()
    
    
    return Recueil_checked, df_nlp_checked


## Preprocess Y : 
def missing_string_data(x):
    if x == 2:
        x=0
    return x

def encoding_recueil(recueil, ONE_HOT=False):
    recueil_copy = recueil.copy()
    if ONE_HOT == True:
        encoder = OneHotEncoder(sparse_output=False)
        # Column's name preparation :
        if 'chir' in recueil.columns: 
            liste_contenu_chir = list(recueil_copy.loc[:,'chir'].unique())
        else:
            liste_contenu_chir = [] 
        if 'sf.dsp.type' in recueil.columns: 
            liste_contenu_dsp_type = list(recueil_copy.loc[:,'sf.dsp.type'].unique())
        else:
            liste_contenu_dsp_type = []
        liste_col_str = liste_contenu_chir + liste_contenu_dsp_type    
        
        if ('chir' in recueil.columns) and ('sf.dsp.type' in recueil.columns):
            liste_str = ['chir', 'sf.dsp.type']
            liste_col_str = liste_contenu_chir + liste_contenu_dsp_type
        elif ('chir' in recueil.columns):
            liste_str = ['chir']
            liste_col_str = liste_contenu_chir + liste_contenu_dsp_type
        elif ('sf.dsp.type' in recueil.columns):
            liste_str = ['sf.dsp.type']
            liste_col_str = liste_contenu_chir + liste_contenu_dsp_type
        else:
            liste_str = []
        

        if len(liste_str)==0:
            #print('Nothing to one-hot encode')
            return recueil_copy
        else:
            # Encoding : 
            recueil_fit = encoder.fit_transform(recueil_copy.loc[:,liste_str])
            # Prepare data : 
            if 'chir' in recueil.columns: recueil_copy.pop('chir');
            if 'sf.dsp.type' in recueil.columns: recueil_copy.pop('sf.dsp.type');
            recueil_fit = pd.DataFrame(recueil_fit, columns=liste_col_str, index=recueil_copy.index)
            recueil_fit = pd.concat([recueil_fit, recueil_copy], axis=1)

    else :
        encoder = OrdinalEncoder()
        recueil_fit = encoder.fit_transform(recueil)
        recueil_fit = pd.DataFrame(recueil_fit, columns=recueil.columns, index=recueil.index)

    return recueil_fit

def multiclass_modificateur(x):
    '''
    Nécessite un jeu de données uniquement en float et en int   
    Transforme le jeu de données en un jeu multiclass, multilabel :   
    0 : données manquantes
    1 : négatif
    2 : positif
    '''
    if np.isnan(x):
        x = 2
    elif pd.api.types.is_integer(x) or pd.api.types.is_float(x):
        x = x+1
    else:
        raise Exception('Type non identifié')
    return x  

def preprocess_recueil(recueil, preprocess_mode, encoder_mode, features_of_interest = 'all'): 
    '''
    Preprocessing :
    1) Transform string features into numerical
    2) Preprocess data
    
    Input : 
    ------------------
    recueil : data
    preprocess_mode : {'multiclass', 'dropna'}
     >>> dropna : for let the classification as a binary multilabel classification
     >>> multiclass : for change the classifisation as a multiclass multilabel/multioutputs classification
    encoder_mode : {True, False}
     >>> True : active One Hot encoder 
     >>> False : let Ordinal encoder active
    features_of_interest : list
    Return : 
    ----------------
    recueil : preprocessed recueil
    '''
    
    if features_of_interest == 'all':
        recueil_copy = recueil.copy()
    elif type(features_of_interest) == str:
        recueil_copy = recueil[[features_of_interest]].copy()     # S'il y a juste une colonne d'intérêt, assurer que 'receuil' reste un dataframe
    else:
        recueil_copy = recueil[features_of_interest].copy()

    ##### Change string features to numérical : 
    # Replace 0 by 'aucune' in sf.dsp.type column :
    if 'sf.dsp.type' in recueil_copy.columns: recueil_copy.loc[:,'sf.dsp.type'] = recueil_copy.loc[:,'sf.dsp.type'].replace(0, 'aucune');
    liste_colonnes_chir = ['date.chir', 'chir.macro.lusd', 
                          'chir.macro.lusg', 'chir.macro.torus', 
                          'chir.macro.oma', 'chir.macro.uro', 
                          'chir.macro.dig', 'chir.macro.superf', 
                          'resec.lusd', 'resec.lusg', 
                          'resec.torus', 'resec.autre']
    for col_chir in liste_colonnes_chir:
        if col_chir in recueil_copy.columns:
            recueil_copy = recueil_copy.drop(col_chir, axis=1)
    
    # drop continuous and discrete features :
    list_to_drop = ['age', 'imc', 'g', 'p', 'sf.dsp.eva', 'sf.dsm.eva']
    for col in list_to_drop:
        if col in recueil_copy.columns:
            recueil_copy = recueil_copy.drop(col, axis=1)
    
    # Encode recueil :
    recueil_copy = encoding_recueil(recueil_copy, encoder_mode)
    
    ##### Preprocess : 
    if preprocess_mode == 'multiclass':
        recueil_copy = recueil_copy.fillna(2)
        #recueil_copy = recueil_copy.map(multiclass_modificateur)
        #recueil_copy = recueil_copy.replace(0, 3)-1 #### 0 : Négatif // 1 : Positif // 2 : Données manquantes
    elif preprocess_mode == 'dropna':
        nsamples_before = recueil_copy.shape[0]
        recueil_copy.dropna(inplace=True)
        nsamples_after = recueil_copy.shape[0]
        nsamples_lost = nsamples_before - nsamples_after
        perc_lost = (nsamples_lost / nsamples_before) * 100
        print(f'Lost {nsamples_lost} samples by dropping nan (={round(perc_lost, 2)}%)')
    for f in recueil_copy.columns:
        recueil_copy[f] = recueil_copy[f].astype(float)
    return recueil_copy



## Preprocess X 

def remove_special_characters(text):
    """
    Input: str : A string to clean from non alphanumeric characters
    Output: str : The same strings without non alphanumeric characters
    """
    # faire attention à " ' "
    text_temp = text
    for char in string.punctuation:
        if char in text:
            text_temp = text_temp.replace(char, ' ')
    return text_temp

def remove_accents(text):
    """
    Input: str : A string to clean from non alphanumeric characters
    Output: str : The same strings without non alphanumeric characters
    Author: Nicolai Wolpert
    Date: 20.06.2024
    """
    text_temp = unidecode(text)
    return text_temp

def lowercase_text(text):
    """
    Input: str : A string to lowercase
    Output: str : The same string lowercased
    """
    return text.lower()

def tokenize_words(text):
    """
    Input: str : A string to tokenize
    Output: list of str : A list of the tokens splitted from the input string
    """
    #TOFILL
    #list_words = text.split(' ')
    # Nicolai corrected 24.06.:
    list_words = re.findall(r"[\w']+", text)
    return list_words

def correction_series(text):
    dict_json = load_dict_correction()
    list_ = appliquer_correction(text, dict_json)
    return contract_string(list_)
    
def apply_spellcheck(text):
    spell = SpellChecker(language='fr')
    text_temp = spell.correction(text)
    if text_temp is not None:
        return text_temp
    else:
        return text

def load_dict_correction():
    path = '/home/nounou/endopath/archives/Data/DATA_PROCESSED/Correction_mots/dictionnaire_correction.json'
    with open(path) as json_file:
        dict_json = json.load(json_file)
    return dict_json

def appliquer_correction(text, dict_json):
    #return list(map(lambda x: replace_word(x, dict_json), text.lower().split(' ')))
    return list(map(lambda x: replace_word(x, dict_json), re.split(r'[ /,.-]', text)))

def replace_word(x, dict_json):
    if x in set(dict_json.keys()):
        x= x.replace(x, dict_json[x])
    return x

def contract_list(liste:list()):
    liste_unique = []
    for elem in liste:
        liste_unique = liste_unique + elem
    return liste_unique

def contract_string(liste:list):
    string = ''
    for elem in liste:
        string = string + elem + ' '
    return string

def compress_n_ano(df):
    liste_rows = []
    liste_n_ano = df.Anonymisation.unique()
    type_ = type(df.loc[0,'Résumé'])
    for i, n_ano in enumerate(liste_n_ano):
        if type_ == list:
            liste_token = contract_list(list(df.loc[df.Anonymisation == n_ano, 'Résumé']))
            row = [n_ano, liste_token]
        elif  type_ == str:
            string = contract_string(list(df.loc[df.Anonymisation == n_ano, 'Résumé']))
            row = [n_ano, string]
        liste_rows.append(row)
    df_concat = pd.DataFrame(liste_rows, columns=['Anonymisation','Résumé'])
    return df_concat

def vocab(df):
    type_ = type(df.loc[0,'Résumé'])
    liste_doc = list(df.Résumé)
    if type_ ==list:
         vocab = list(set(contract_list(liste_doc)))
    elif type_ == str:
        vocab = contract_string(list(df.Résumé))
        vocab = list(set(vocab.split(' ')))
    return vocab

def from_X_split_get_Y_split(X, Y_to_split:pd.DataFrame()):
    Y = Y_to_split.loc[X.Anonymisation,:]
    return Y 

def drop_anapath(Y):
    if 'anapath.lusd' in Y.columns: Y.drop('anapath.lusd', axis=1, inplace=True);
    if 'anapath.lusg' in Y.columns: Y.drop('anapath.lusg', axis=1, inplace=True);
    if 'anapath.torus' in Y.columns: Y.drop('anapath.torus', axis=1, inplace=True);
    if 'anapath.autre' in Y.columns: Y.drop('anapath.autre', axis=1, inplace=True);
    return Y


def stop_words(liste):
    stopwords = nltk.corpus.stopwords.words('french')
    for w in stopwords:
        if w in liste:
            liste.remove(w)
    return liste

def contain_number(word):
    return any(char.isdigit() for char in word)

def remove_number(liste):
    '''
    Take a list or a string and remove word containing digits   
    '''
    # If liste is a string and not a list:
    if type(liste) == str:
        # from string to list : 
        liste = liste.split(' ') 
        liste_copy = liste.copy()
        for w in liste:                # each string of the list is checked
            if contain_number(w):      # if there is any number in the string
                liste_copy.remove(w)   # the string is removed
        liste_copy = contract_string(liste_copy)
        
        
    # if liste if a list :    
    elif type(liste) ==list:   
        liste_copy = liste.copy()
        for w in liste:                # each string of the list is checked
            if contain_number(w):      # if there is any number in the string
                liste_copy.remove(w)   # the string is removed
    return liste_copy

## Preprocess X and Y 

def preprocess_and_split(X_, Y_, features_of_interest='all', seed=42, test_size=0.20, special_char=True, accents=True, lower=True, token=True, drop_number=True, remove_stopwords = True, compress=True, preprocess_mode='dropna', encoder_mode=True, anapath=False, correction_by_json=False, spellcheck=False):
    '''
    X_ : NLP tokenized data
    Y_ : recueil
    features_of_interest : list
    seed : for split data, int.
    test_size : target size of test, float
    
    
    preprocess_mode : {'multiclass', 'dropna'}
     >>> dropna : for let the classification as a binary multilabel classification
     >>> multiclass : for change the classifisation as a multiclass multilabel/multioutputs classification
    encoder_mode : {True, False}
     >>> True : active One Hot encoder 
     >>> False : let Ordinal encoder active
    
    '''
    X = X_.copy()
    Y = Y_.copy()
      
    # Preprocessing Y :
    Y = preprocess_recueil(Y, preprocess_mode, encoder_mode, features_of_interest)

    # Get same index for X and Y (and drop missing values in Y)
    Y, X = check_n_ano(Y, X)

    if anapath==True:
        Y = drop_anapath(Y)
    
    # Preprocessing X :
    # IMPORTANT: Lowercase and removal of special characters has to be applied before 'correction_series', else words will not be found in the correction dictionnary
    if lower:
        X.Résumé = X.Résumé.apply(lowercase_text)
    if special_char:
        X.Résumé = X.Résumé.apply(remove_special_characters)
    if correction_by_json:
        X.Résumé = X.Résumé.apply(correction_series)
    '''
    # Spellcheck possible pour corriger les mots mais prend trop de temps...
    if spellcheck:
        print('Running spellcheck. This takes a while...')
        X.Résumé = X.Résumé.apply(apply_spellcheck)
    '''
    if accents:
        X.Résumé = X.Résumé.apply(remove_accents)
        
    if drop_number:
        X.Résumé = X.Résumé.apply(remove_number)
    if compress:
        X = compress_n_ano(X)
    if remove_stopwords:
        # Removing stopwords requires tokenized words
        X.Résumé = X.Résumé.apply(tokenize_words)
        X.Résumé = X.Résumé.apply(stop_words)
        # Concatenate back into non-tokenized form after
        X.Résumé = X.Résumé.apply(lambda words: ' '.join(words))
    if token:
        X.Résumé = X.Résumé.apply(tokenize_words)

    # Max Vocab :    
    vocabulaire = vocab(X)
    max_vocab = len(vocabulaire)
        
    # Split : 
    X_train, X_test = train_test_split(X, random_state=seed, test_size=test_size)    
    Y_train = from_X_split_get_Y_split(X_train, Y)
    Y_test = from_X_split_get_Y_split(X_test, Y)
        
    return X_train, X_test, Y_train, Y_test, max_vocab, X, Y