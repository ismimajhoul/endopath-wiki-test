#!/usr/bin/env python
# coding: utf-8

# #  NLP_ENDOPATHS_Nicolai

# Author: Nicolai Wolpert  
# Email: nicolai.wolpert@capgemini.com  
# Date: June 2024
# 
# Script to test LSTM models to predict patient symptoms from gynéco text files

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
from exploratory.Opti_utils.ML_utils import kfold_cv_stratified, sensi, speci

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
from keras import Sequential
from keras.layers import Embedding, LSTM, Dropout, Dense

import pickle

# In[2]:


model_dir = '/home/nounou/endopath/exploratory/models'
pred_dir = '/home/nounou/endopath/exploratory/predictions/'

# ## Load data

# In[3]:


# Loading X : 
df_nlp = pd.read_csv('/home/nounou/endopath/Data/DATA_PROCESSED/donnees_entree_nlp_sans_endo.csv', usecols=['Anonymisation', 'Date', 'Nature', 'Résumé'])
print('X shape is :', df_nlp.shape)

# Loading Y : 
# Two options:
# 1) Take the original 'receuil' file as ground truth. In that case select 'receuil_orig'
# 2) Take the infos extracted manually from the gyneco file by a human observer as ground truth. In that case select 'gyneco_manual'
data_option = 'gyneco_manual'
if data_option=='receuil_orig':
    recueil  = pd.read_excel('./../../Data/Raw/Recueil (1).xlsx').drop('Unnamed: 90', axis=1)
    recueil.replace(['Na', 'NA'], np.nan, inplace=True)
    recueil.set_index('Numéro anonymat', inplace=True)
elif data_option=='gyneco_manual':
    recueil  = pd.read_excel('/home/nounou/endopath/Data/DATA_PROCESSED/data_gynéco_manual_extraction.xlsx')
    recueil.set_index('Anonymisation', inplace=True)
print('Y shape is :', recueil.shape)
print(f'Nombre de patientes dans le df_nlp : {len(df_nlp.Anonymisation.unique())}')

if 'DJ-055' in list(df_nlp['Anonymisation']):
    df_nlp.loc[df_nlp['Anonymisation']=='DJ-055', 'Anonymisation'] ='NJ-055'
'NJ-055' in list(df_nlp['Anonymisation'])
'DJ-055' in list(df_nlp['Anonymisation'])

# In[4]:


'''
# Les features qui ont été identifiés comme les plus pertinentes pour pouvoir prédire l'endiométrose profonde sont les suivants
# (cela veut dire qu'ils donnent une sensibilité et spécificité au dessus du seuil de 0.4 et 0.5):
features_of_interest = ['age', 'atcd.endo', 'sf.dsm.eva', 'sf.dpc', 'tv.douleur.lusd', 'tv.douleur.lusg', 'chir_GD', 'chir_SW']
'''
# See 'Explore_features_ML_NLP' for how these have been identified
features_of_interest = ['atcd.endo', 'irm.lusg', 'tv.douloureux', 'irm.externe', 'sf.dig.diarrhee', 'echo.lusg', 'echo.lusd', 'ef.hormone.dpc', 'effet.hormone']
features_of_interest = ['irm.lusg']
#features_of_interest = 'all'

# In[5]:


if features_of_interest != 'all':
    recueil = recueil[features_of_interest]

# In[6]:


percent_missing = recueil.isnull().sum() * 100 / recueil.shape[0]
missing_value_df = pd.DataFrame({'column_name': recueil.columns,
                                 'percent_missing': percent_missing})
missing_value_df

# ### LSTM

# In[7]:


use_validation_set = True
train_proportion = 0.7

# #### LSTM for one feature

# In[13]:


target_feature = 'irm.lusg'
"""_, _, _, _, max_vocab, X, Y = preprocess_and_split( df_nlp,
                                                   recueil,
                                                   target_feature,
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
Y = Y[[target_feature]]
nclasses = len(np.unique(Y.values.flatten()))
if nclasses==2:
    Y_one_hot = to_categorical(Y, num_classes=2)
    Y_one_hot = pd.DataFrame(Y_one_hot, columns=['négatif', 'positif'], index=Y.index)
else:
    Y_one_hot = to_categorical(Y, num_classes=3)
    Y_one_hot = pd.DataFrame(Y_one_hot, columns=['négatif', 'positif', 'valeur_manquante'], index=Y.index)

# Split :
X_train, X_test = train_test_split(X, random_state=42, test_size=0.2)
indeces_test = list(X_test.index)
Y_train = from_X_split_get_Y_split(X_train, Y_one_hot)
Y_test = from_X_split_get_Y_split(X_test, Y_one_hot).values

if use_validation_set:
    X_val = X_train.iloc[X_train.shape[0]-10:,:]
    indeces_val = list(X_val.index)
    Y_val = Y_train.iloc[Y_train.shape[0]-10:,:].values
    X_train = X_train.iloc[:X_train.shape[0]-10,:]
    indeces_train = list(X_train.index)
    Y_train = Y_train.iloc[:Y_train.shape[0]-10,:].values
print(f'Number of training samples: {X_train.shape[0]}')
if use_validation_set: print(f'Number of validation samples: {X_val.shape[0]}');
print(f'Number of testing samples: {X_test.shape[0]}')
print(f'Number of classes: {nclasses}')"""

#modify
# ...existing code...

# Define get_lstm_model BEFORE your for loop
def get_lstm_model(vocab_size, embedding_dim, lstm_out_dim, dropout_rate, n_dense, nclasses):
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim))
    model.add(LSTM(units=lstm_out_dim, return_sequences=True))
    model.add(Dropout(dropout_rate))
    model.add(LSTM(units=lstm_out_dim))
    model.add(Dropout(dropout_rate))
    model.add(Dense(units=128, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(units=64, activation='relu'))
    model.add(Dropout(dropout_rate))
    if nclasses==2:
        print('2 classes, using binary crossentropy')
        model.add(Dense(n_dense, input_dim=lstm_out_dim, activation='sigmoid'))
        model.compile(loss = 'binary_crossentropy', optimizer='adam', metrics = ['accuracy', 'Recall'])
    else:
        print(f'{nclasses} classes, using categorical crossentropy')
        model.add(Dense(n_dense, input_dim=lstm_out_dim, activation='softmax'))
        model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics = ['accuracy', 'Recall'])
    return model

# ...now your for preprocess_mode in ['multiclass', 'dropna']: loop...
embedding_dim = 100
lstm_out_dim = 200
dropout_rate = 0.2
for preprocess_mode in ['multiclass', 'dropna']:
    print(f"Running for preprocess_mode: {preprocess_mode}")
    # Prepare data
    _, _, _, _, max_vocab, X, Y = preprocess_and_split(
        df_nlp,
        recueil,
        target_feature,
        42,
        0.20,
        special_char=True,
        accents=False,
        lower=True,
        token=False,
        remove_stopwords=True,
        drop_number=True,
        compress=True,
        preprocess_mode=preprocess_mode,
        encoder_mode=True,
        anapath=True,
        correction_by_json=True
    )
    Y = Y[[target_feature]]
    nclasses = len(np.unique(Y.values.flatten()))
    if nclasses == 2:
        Y_one_hot = to_categorical(Y, num_classes=2)
        Y_one_hot = pd.DataFrame(Y_one_hot, columns=['négatif', 'positif'], index=Y.index)
    else:
        Y_one_hot = to_categorical(Y, num_classes=3)
        Y_one_hot = pd.DataFrame(Y_one_hot, columns=['négatif', 'positif', 'valeur_manquante'], index=Y.index)

    # Split
    X_train, X_test = train_test_split(X, random_state=42, test_size=0.2)
    Y_train = from_X_split_get_Y_split(X_train, Y_one_hot)
    Y_test = from_X_split_get_Y_split(X_test, Y_one_hot).values

    if use_validation_set:
        X_val = X_train.iloc[X_train.shape[0]-10:,:]
        Y_val = Y_train.iloc[Y_train.shape[0]-10:,:].values
        X_train = X_train.iloc[:X_train.shape[0]-10,:]
        Y_train = Y_train.iloc[:Y_train.shape[0]-10,:].values

    # Tokenization and padding
    tokenizer = Tokenizer(num_words=max_vocab, split=' ', oov_token='<unw>', filters=' ')
    if use_validation_set:
        tokenizer.fit_on_texts(pd.concat([X_train, X_val]).loc[:, 'Résumé'])
    else:
        tokenizer.fit_on_texts(X_train['Résumé'])
    train_seqs = tokenizer.texts_to_sequences(X_train.loc[:, 'Résumé'])
    if use_validation_set:
        valid_seqs = tokenizer.texts_to_sequences(X_val.loc[:, 'Résumé'])
    test_seqs = tokenizer.texts_to_sequences(X_test.loc[:, 'Résumé'])
    max_seq_length = max([len(seq) for seq in train_seqs + (valid_seqs if use_validation_set else [])])
    X_train_pad = pad_sequences(train_seqs, max_seq_length)
    if use_validation_set:
        X_valid_pad = pad_sequences(valid_seqs, max_seq_length)
    X_test_pad = pad_sequences(test_seqs, max_seq_length)

    # Model
    model = get_lstm_model(max_vocab, embedding_dim, lstm_out_dim, dropout_rate, n_dense=Y_train.shape[1], nclasses=nclasses)
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, verbose=1)
    batch_size = 64
    max_epochs = 10
    if use_validation_set:
        history = model.fit(X_train_pad, Y_train, epochs=max_epochs, batch_size=batch_size,
                            verbose=0, validation_data=(X_valid_pad, Y_val), callbacks=[early_stopping])
    else:
        history = model.fit(X_train_pad, Y_train, epochs=max_epochs, batch_size=batch_size, verbose=0)

    # Predict and save
    predictions = model.predict(X_test_pad)
    n_pred_classes = predictions.shape[1]
    real_labels = np.argmax(Y_test, axis=1)
    pred_labels = np.argmax(predictions, axis=1)
    confidences = np.max(predictions, axis=1)
    predictions_df = pd.DataFrame({
        f"{target_feature}_real": real_labels,
        f"{target_feature}_predicted": pred_labels,
        f"{target_feature}_confidence": confidences
    })
    predictions_df.to_csv(
        pred_dir + f'predictions_lstm_upsampled_{preprocess_mode}_{target_feature}'.replace('.', '_') + '.csv',
        index=False
    )
    print(f"Saved predictions for {preprocess_mode}")

# In[14]:


max_vocab_size = max_vocab 

tokenizer = Tokenizer(num_words=max_vocab_size, split=' ', oov_token='<unw>', filters=' ')
if use_validation_set:
    tokenizer.fit_on_texts(pd.concat([X_train,X_val]).loc[:,'Résumé'])
else:
    tokenizer.fit_on_texts(X_train['Résumé'])

# This encodes our sentence as a sequence of integer
# each integer being the index of each word in the vocabulary
train_seqs = tokenizer.texts_to_sequences(X_train.loc[:,'Résumé'])
if use_validation_set: valid_seqs = tokenizer.texts_to_sequences(X_val.loc[:,'Résumé']);
test_seqs = tokenizer.texts_to_sequences(X_test.loc[:,'Résumé'])

# We need to pad the sequences so that they are all the same length :
# the length of the longest one
max_seq_length = max( [len(seq) for seq in train_seqs + valid_seqs] )

X_train_pad = pad_sequences(train_seqs, max_seq_length)
X_valid_pad = pad_sequences(valid_seqs, max_seq_length)
X_test_pad = pad_sequences(test_seqs, max_seq_length)

embedding_dim = 100
lstm_out_dim = 200
dropout_rate = 0.2

def get_lstm_model(vocab_size, embedding_dim, lstm_out_dim, dropout_rate, n_dense, nclasses):
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim))

    model.add(LSTM(units=lstm_out_dim, return_sequences=True))
    model.add(Dropout(dropout_rate))

    model.add(LSTM(units=lstm_out_dim))
    model.add(Dropout(dropout_rate))

    model.add(Dense(units=128, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(units=64, activation='relu'))
    model.add(Dropout(dropout_rate))

    if nclasses==2:
        print('2 classes, using binary crossentropy')
        model.add(Dense(n_dense, input_dim=lstm_out_dim, activation='sigmoid'))
        model.compile(loss = 'binary_crossentropy', optimizer='adam', metrics = ['accuracy', 'Recall'])
    else:
        print(f'{nclasses} classes, using categorical crossentropy')
        model.add(Dense(n_dense, input_dim=lstm_out_dim, activation='softmax'))
        model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics = ['accuracy', 'Recall'])
    
    return model

model = get_lstm_model(max_vocab_size, embedding_dim, lstm_out_dim, dropout_rate, n_dense=Y_train.shape[1], nclasses=nclasses)
# print(model.summary())
early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, verbose=1)
batch_size = 64
max_epochs = 10

'''
if use_validation_set:
    history = model.fit(X_train_pad, Y_train, epochs=max_epochs, batch_size=batch_size, verbose=1, validation_data = (X_valid_pad, Y_val), callbacks=[early_stopping])
else:
    history = model.fit(X_train_pad, Y_train, epochs=max_epochs, batch_size=batch_size, verbose=1, callbacks=[early_stopping])
'''

# In[15]:


predictions = model.predict(X_test_pad)
predictions

# In[11]:


predictions_df = pd.DataFrame(columns=[target_feature + '_predicted', target_feature + '_real'])
#predictions_df[target_feature + '_predicted_proba'] = predictions
predictions_df[target_feature + '_predicted'] = np.argmax(predictions, axis=1).tolist()
predictions_df[target_feature + '_real'] = np.argmax(Y_test, axis=1).tolist()
predictions_df.head()

# In[12]:


print(classification_report(predictions_df[target_feature + '_real'], predictions_df[target_feature + '_predicted']))

# In[13]:


"""cm = confusion_matrix(predictions_df[target_feature + '_real'], predictions_df[target_feature + '_predicted'])
if len(pd.unique(recueil[target_feature]))==2:
    labels=['non', 'oui']
else:
    labels=['non', 'oui', 'ambigue']
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot()"""
#modify
cm = confusion_matrix(predictions_df[target_feature + '_real'], predictions_df[target_feature + '_predicted'])
unique_classes = np.unique(predictions_df[target_feature + '_real'])
if len(unique_classes) == 2:
    labels = ['non', 'oui']
elif len(unique_classes) == 3:
    labels = ['non', 'oui', 'ambigue']
else:
    labels = [str(i) for i in unique_classes]
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot()

# #### Upsampling to account for class imbalances

# In[ ]:


target_feature = 'irm.lusg'
preprocess_mode = 'multiclass'      # 'multiclass' = garder les nan et les traiter comme classe, 'dropnan' = enlever les nan
_, _, _, _, max_vocab, X, Y = preprocess_and_split( df_nlp,
                                                   recueil,
                                                   target_feature,
                                                   42,                     # choix de la seed pour le random split
                                                   0.20,                   # Choix du test_size
                                                   special_char=True,      # supprime les caractères spéciaux  
                                                   accents=False,          # supprime les accents
                                                   lower=True,            # passe la casse du texte en minuscule
                                                   token=False,            # tokenize le corpus
                                                   remove_stopwords=True,
                                                   drop_number=True,
                                                   compress=True,          # un corpus par n_ano, shape = [200,2]
                                                   preprocess_mode=preprocess_mode,
                                                   encoder_mode=True,
                                                   anapath=True,
                                                   correction_by_json=True
                                                   )
Y = Y[[target_feature]]
nclasses = len(np.unique(Y.values.flatten()))
if nclasses==2:
    Y_one_hot = to_categorical(Y, num_classes=2)
    Y_one_hot = pd.DataFrame(Y_one_hot, columns=['négatif', 'positif'], index=Y.index)
else:
    Y_one_hot = to_categorical(Y, num_classes=3)
    Y_one_hot = pd.DataFrame(Y_one_hot, columns=['négatif', 'positif', 'valeur_manquante'], index=Y.index)

# Split :
X_train, X_test = train_test_split(X, random_state=42, test_size=0.2)
indeces_test = list(X_test.index)
Y_train = from_X_split_get_Y_split(X_train, Y_one_hot)
Y_test = from_X_split_get_Y_split(X_test, Y_one_hot).values

if use_validation_set:
    X_val = X_train.iloc[X_train.shape[0]-10:,:]
    indeces_val = list(X_val.index)
    Y_val = Y_train.iloc[Y_train.shape[0]-10:,:].values
    X_train = X_train.iloc[:X_train.shape[0]-10,:]
    indeces_train = list(X_train.index)
    Y_train = Y_train.iloc[:Y_train.shape[0]-10,:].values
print(f'Number of samples in total: {X_train.shape[0] + X_val.shape[0] + X_test.shape[0]}')
print(f'Number of training samples: {X_train.shape[0]}')
if use_validation_set: print(f'Number of validation samples: {X_val.shape[0]}');
print(f'Number of testing samples: {X_test.shape[0]}')
print(f'Number of classes: {nclasses}')

# In[11]:


data = pd.merge(X, Y_one_hot.reset_index().rename(columns={'Numéro anonymat': 'Anonymisation'}), on='Anonymisation')
data = pd.merge(data, Y.reset_index().rename(columns={'Numéro anonymat': 'Anonymisation'}), on='Anonymisation')
data.head()

# In[12]:


# Calculate the distribution of classes for the selected feature
class_counts = data[target_feature].value_counts()

# Create a bar plot
plt.figure(figsize=(8, 6))
class_counts.plot(kind='bar', color=['skyblue', 'orange'])
plt.title(f'Original distribution of Classes for {target_feature}')
plt.xlabel('Class')
plt.ylabel('Count')
plt.xticks(rotation=0)  # Rotate x-axis labels if necessary
plt.grid(axis='y', linestyle='--', alpha=0.7)

# In[13]:


data_negative = data.loc[data[target_feature]==0.0]
data_positive = data.loc[data[target_feature]==1.0]
data_missing = data.loc[data[target_feature]==2.0]

if len(data_negative) > len(data_positive):
    data_majority = data_negative
    data_minority = data_positive
else:
    data_majority = data_positive
    data_minority = data_negative

bias = data_minority.shape[0]/data_majority.shape[0]

# split train/test data first 
data_train = pd.concat([data_majority.sample(frac=train_proportion,random_state=200),
         data_minority.sample(frac=train_proportion,random_state=200), data_missing.sample(frac=train_proportion,random_state=200)])
data_test = pd.concat([data_majority.drop(data_majority.sample(frac=train_proportion,random_state=200).index),
        data_minority.drop(data_minority.sample(frac=train_proportion,random_state=200).index), data_missing.drop(data_missing.sample(frac=train_proportion,random_state=200).index)])


data_test_negative = data_test.loc[data_test[target_feature]==0.0]
data_test_positive = data_test.loc[data_test[target_feature]==1.0]
data_test_missing = data_test.loc[data_test[target_feature]==2.0]
if use_validation_set:
    data_val = pd.concat([data_test_negative.sample(frac=0.5,random_state=200),
            data_test_positive.sample(frac=0.5,random_state=200), data_test_missing.sample(frac=0.5,random_state=200)])
    data_test = pd.concat([data_test_negative.drop(data_test_negative.sample(frac=0.5,random_state=200).index),
            data_test_positive.drop(data_test_positive.sample(frac=0.5,random_state=200).index), data_test_missing.drop(data_test_missing.sample(frac=0.5,random_state=200).index)])

data_train = shuffle(data_train)
if use_validation_set: data_val = shuffle(data_val)
data_test = shuffle(data_test)

# In[14]:


data_positive_train = (data_train[target_feature] == 1.0).sum()
data_negative_train = (data_train[target_feature] == 0.0).sum()
data_ambiguous_train = (data_train[target_feature] == 2.0).sum()
print('positive data in training:', data_positive_train)
print('negative data in training:', data_negative_train)
print('ambiguous data in training:', data_ambiguous_train)
print()
if use_validation_set:
    print('positive data in validation:',(data_val[target_feature] == 1.0).sum())
    print('negative data in validation:',(data_val[target_feature] == 0.0).sum())
    print('ambiguous data in validation:',(data_val[target_feature] == 2.0).sum())
    print()
print('positive data in test:',(data_test[target_feature] == 1.0).sum())
print('negative data in test:',(data_test[target_feature] == 0.0).sum())
print('ambiguous data in test:',(data_test[target_feature] == 2.0).sum())

# In[15]:


# Separate majority and minority classes in training data for upsampling
if nclasses==2:
    if len(data_train[data_train[target_feature] == 0.0]) > len(data_train[data_train[target_feature] == 1.0]):
        data_train_majority = data_train[data_train[target_feature] == 0.0]
        data_train_minority = data_train[data_train[target_feature] == 1.0]
    else:
        data_train_majority = data_train[data_train[target_feature] == 1.0]
        data_train_minority = data_train[data_train[target_feature] == 0.0]

    print("majority class before upsample:",data_train_majority.shape)
    print("minority class before upsample:",data_train_minority.shape)

    # Upsample minority class
    data_train_minority_upsampled = resample(data_train_minority, 
                                    replace=True,     # sample with replacement
                                    n_samples= data_train_majority.shape[0],    # to match majority class
                                    random_state=123) # reproducible results
    
    # Combine majority class with upsampled minority class
    data_train_upsampled = pd.concat([data_train_majority, data_train_minority_upsampled])
    
elif nclasses==3:
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

    print("majority class before upsample:",data_train_majority.shape)
    print("minority class 1 before upsample:",data_train_minority1.shape)
    print("minority class 2 before upsample:",data_train_minority2.shape)

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

# In[16]:


# Calculate the distribution of classes for the selected feature
class_counts = data_train_upsampled[target_feature].value_counts()

# Create a bar plot
plt.figure(figsize=(8, 6))
class_counts.plot(kind='bar', color=['skyblue', 'orange'])
plt.title(f'Distribution of Classes in training for {target_feature} after upsampling')
plt.xlabel('Class')
plt.ylabel('Count')
plt.xticks(rotation=0)  # Rotate x-axis labels if necessary
plt.grid(axis='y', linestyle='--', alpha=0.7)

# In[17]:


if nclasses==2:
    X_train = data_train_upsampled[['Anonymisation', 'Résumé']]
    Y_train = data_train_upsampled[['négatif', 'positif']].values
    if use_validation_set:
        X_val = data_val[['Anonymisation', 'Résumé']]
        Y_val = data_val[['négatif', 'positif']].values
    X_test = data_test[['Anonymisation', 'Résumé']]
    Y_test = data_test[['négatif', 'positif']].values
else:
    X_train = data_train_upsampled[['Anonymisation', 'Résumé']]
    Y_train = data_train_upsampled[['négatif', 'positif', 'valeur_manquante']].values
    if use_validation_set:
        X_val = data_val[['Anonymisation', 'Résumé']]
        Y_val = data_val[['négatif', 'positif', 'valeur_manquante']].values
    X_test = data_test[['Anonymisation', 'Résumé']]
    Y_test = data_test[['négatif', 'positif', 'valeur_manquante']].values

print(f'Number of training samples: {X_train.shape[0]}')
if use_validation_set: print(f'Number of validation samples: {X_val.shape[0]}');
print(f'Number of testing samples: {X_test.shape[0]}')

# In[19]:


max_vocab_size = max_vocab 

tokenizer = Tokenizer(num_words=max_vocab_size, split=' ', oov_token='<unw>', filters=' ')
tokenizer.fit_on_texts(pd.concat([X_train,X_val]).loc[:,'Résumé'])

# This encodes our sentence as a sequence of integer
# each integer being the index of each word in the vocabulary
train_seqs = tokenizer.texts_to_sequences(X_train.loc[:,'Résumé'])
if use_validation_set: valid_seqs = tokenizer.texts_to_sequences(X_val.loc[:,'Résumé']);
test_seqs = tokenizer.texts_to_sequences(X_test.loc[:,'Résumé'])

# We need to pad the sequences so that they are all the same length :
# the length of the longest one
max_seq_length = max( [len(seq) for seq in train_seqs + valid_seqs] )

X_train_pad = pad_sequences(train_seqs, max_seq_length)
if use_validation_set: X_valid_pad = pad_sequences(valid_seqs, max_seq_length);
X_test_pad = pad_sequences(test_seqs, max_seq_length)

embedding_dim = 100
lstm_out_dim = 200
dropout_rate = 0.2

def get_lstm_model(vocab_size, embedding_dim, lstm_out_dim, dropout_rate, n_dense, nclasses):
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, trainable=True))

    model.add(LSTM(units=lstm_out_dim, return_sequences=True))
    model.add(Dropout(dropout_rate))
    
    model.add(LSTM(units=lstm_out_dim))
    model.add(Dropout(dropout_rate))

    model.add(Dense(units=128, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(units=64, activation='relu'))
    model.add(Dropout(dropout_rate))

    if nclasses==2:
        print('2 classes, using binary crossentropy')
        model.add(Dense(n_dense, input_dim=lstm_out_dim, activation='sigmoid'))
        model.compile(loss = 'binary_crossentropy', optimizer='adam', metrics = ['accuracy', 'Recall'])
    else:
        print(f'{nclasses} classes, using categorical crossentropy')
        model.add(Dense(n_dense, input_dim=lstm_out_dim, activation='softmax'))
        model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics = ['accuracy', 'Recall'])
    
    return model


model = get_lstm_model(max_vocab_size, embedding_dim, lstm_out_dim, dropout_rate, n_dense=Y_train.shape[1], nclasses=nclasses)
# print(model.summary())
early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, verbose=1)
batch_size = 64
max_epochs = 20
if use_validation_set:
    history = model.fit(X_train_pad, Y_train, epochs=max_epochs, batch_size=batch_size, 
                        verbose=1, validation_data = (X_valid_pad, Y_val), callbacks=[early_stopping])
else:
    history = model.fit(X_train_pad, Y_train, epochs=max_epochs, batch_size=batch_size, 
                    verbose=1)
pickle.dump(model, open(model_dir + f'lstm_upsampled_{preprocess_mode}_{target_feature}'.replace('.', '_'), 'wb'))

# In[20]:


model.summary()

# In[33]:


#model = pickle.load(open(model_dir + f'lstm_upsampled_{preprocess_mode}_{target_feature}'.replace('.', '_'), 'rb'))

# In[21]:


predictions = model.predict(X_test_pad)
predictions

# In[22]:


predictions_df = pd.DataFrame(columns=[target_feature + '_real', target_feature + '_predicted', target_feature + '_confidence'])
#predictions_df[target_feature + '_predicted_proba'] = predictions
predictions_df[target_feature + '_real'] = np.argmax(Y_test, axis=1).tolist()
predictions_df[target_feature + '_predicted'] = np.argmax(predictions[:, :2], axis=1).tolist()      # Ignorer la catégorie "données manquantes"
predictions_df[target_feature + '_confidence'] = np.max(predictions[:, :2], axis=1).tolist()
predictions_df.to_csv(pred_dir + f'predictions_lstm_upsampled_{preprocess_mode}_{target_feature}'.replace('.', '_') + '.csv')
predictions_df.head()

# In[23]:


print(classification_report(predictions_df[target_feature + '_real'], predictions_df[target_feature + '_predicted']))

# In[24]:


cm = confusion_matrix(predictions_df[target_feature + '_real'], predictions_df[target_feature + '_predicted'])
if len(pd.unique(recueil[target_feature]))==2:
    labels=['non', 'oui']
else:
    labels=['non', 'oui', 'ambigue']
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot()

# In[25]:


if use_validation_set:
    fig, axes = plt.subplots(1, 2, figsize = (16,6))
    axes[0].plot(history.history['accuracy'])
    axes[0].plot(history.history['val_accuracy'],'--')
    axes[0].set_title('model accuracy')
    axes[0].set_ylabel('accuracy')
    axes[0].set_xlabel('epoch')
    axes[0].legend(['train', 'val'], loc='lower right')

    axes[1].plot(history.history['loss'])
    axes[1].plot(history.history['val_loss'],"--")
    axes[1].set_title('model loss')
    axes[1].set_ylabel('loss')
    axes[1].set_xlabel('epoch')
    axes[1].legend(['val_loss', 'loss'], loc='lower right')

# #### Thresholding approach

# In[37]:


liste_resultats = []
for threshold in np.arange(0.05, 1, 0.01):
    dict_ = {}
    predicted_values = [1 if pred_proba > threshold else 0 for pred_proba in predictions[:, 1]]
    true_values = np.argmax(Y_test, axis=1).tolist()
    # Remove missing data
    predicted_values = [x for i, x in enumerate(predicted_values) if true_values[i] != 2]
    true_values = [x for x in true_values if x != 2]
    sensib = sensi(true_values, predicted_values)
    specic = speci(true_values, predicted_values)
    dict_['trhd']=threshold
    dict_['sensi']=sensib
    dict_['speci']=specic
    liste_resultats.append(dict_)
resultats = pd.DataFrame(liste_resultats)

# In[38]:


plt.figure()
plt.plot(resultats.trhd, resultats.speci, '--', color='blue')
plt.plot(resultats.trhd, resultats.sensi, '--', color='green')
plt.legend(['Precision', 'Recall'])
plt.title(f'{target_feature}')
plt.show()

# 

# #### LSTM for all features

# In[5]:


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
X_val = X_train.iloc[X_train.shape[0]-10:,:]
Y_val = Y_train.iloc[Y_train.shape[0]-10:,:]
X_train = X_train.iloc[:X_train.shape[0]-10,:]
Y_train = Y_train.iloc[:Y_train.shape[0]-10,:]
nclasses = len(np.unique(Y.values.flatten()))
print(f'Number of training samples: {X_train.shape[0]}')
print(f'Number of validation samples: {X_val.shape[0]}')
print(f'Number of testing samples: {X_test.shape[0]}')
print(f'Number of classes: {nclasses}')


# In[6]:


X_train.head()

# In[7]:


Y_train.head()

# In[8]:


max_vocab_size = max_vocab 

tokenizer = Tokenizer(num_words=max_vocab_size, split=' ', oov_token='<unw>', filters=' ')
tokenizer.fit_on_texts(pd.concat([X_train,X_val]).loc[:,'Résumé'])

# This encodes our sentence as a sequence of integer
# each integer being the index of each word in the vocabulary
train_seqs = tokenizer.texts_to_sequences(X_train.loc[:,'Résumé'])
valid_seqs = tokenizer.texts_to_sequences(X_val.loc[:,'Résumé'])
test_seqs = tokenizer.texts_to_sequences(X_test.loc[:,'Résumé'])

# In[9]:


# We need to pad the sequences so that they are all the same length :
# the length of the longest one
max_seq_length = max( [len(seq) for seq in train_seqs + valid_seqs] )

X_train_pad = pad_sequences(train_seqs, max_seq_length)
X_valid_pad = pad_sequences(valid_seqs, max_seq_length)
X_test_pad = pad_sequences(test_seqs, max_seq_length)

embedding_dim = 100
lstm_out_dim = 200
dropout_rate = 0.2
nclasses = len(np.unique(Y_train.values.flatten()))

def get_lstm_model(vocab_size, embedding_dim, seq_length, lstm_out_dim, dropout_rate, n_dense, nclasses):
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=seq_length))

    '''
    model.add(LSTM(units=lstm_out_dim, return_sequences=True))
    model.add(Dropout(dropout_rate))
    '''

    model.add(LSTM(units=lstm_out_dim))
    model.add(Dropout(dropout_rate))

    '''
    model.add(Dense(units=128, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(units=64, activation='relu'))
    model.add(Dropout(dropout_rate))
    '''

    if nclasses==2:
        print('2 classes, using binary crossentropy')
        model.add(Dense(n_dense, input_dim=lstm_out_dim, activation='sigmoid'))
        model.compile(loss = 'binary_crossentropy', optimizer='adam', metrics = ['accuracy', 'Recall'])
    else:
        print(f'{nclasses} classes, using categorical crossentropy')
        model.add(Dense(n_dense, input_dim=lstm_out_dim, activation='softmax'))
        model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics = ['accuracy', 'Recall'])
    return model


model = get_lstm_model(max_vocab_size, embedding_dim, max_seq_length, lstm_out_dim, dropout_rate, n_dense=Y_train.shape[1], nclasses=nclasses)
# print(model.summary())
early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, verbose=1)
batch_size = 64
max_epochs = 10
history = model.fit(X_train_pad, Y_train, epochs=max_epochs, batch_size=batch_size, 
                    verbose=0, validation_data = (X_valid_pad, Y_val), callbacks=[early_stopping])

# In[ ]:


predictions = model.predict(X_test_pad)
predictions_df = pd.DataFrame(predictions, columns=Y_test.columns, index=Y_test.index)
predictions_df

# In[38]:


Y_test

# In[49]:


# Select the feature to visualize
#feature = 'atcd.endo'
#modify
feature = 'irm.lusg'

# Calculate the distribution of classes for the selected feature
#class_counts = Y_train[feature].value_counts()
# Use the original Y (before one-hot encoding);;;;;;;;;modify
class_counts = Y[feature].value_counts()

# Create a bar plot
plt.figure(figsize=(8, 6))
class_counts.plot(kind='bar', color=['skyblue', 'orange'])
plt.title(f'Distribution of Classes for {feature}')
plt.xlabel('Class')
plt.ylabel('Count')
plt.xticks(rotation=0)  # Rotate x-axis labels if necessary
plt.grid(axis='y', linestyle='--', alpha=0.7)

# ## Show results across features

# In[8]:


upsampling = True
preprocess_mode = 'multiclass'

results_per_feat = pd.DataFrame(columns=['feature', 'precision_0', 'recall_0', 'f1_score_0', 'precision_1', 'recall_1', 'f1_score_1'])
for feature in features_of_interest:
    feature_read = feature.replace('.', '_')
    if upsampling:
        predictions_df = pd.read_csv(pred_dir + f'predictions_lstm_upsampled_{preprocess_mode}_{feature_read}.csv')
    else:
        predictions_df = pd.read_csv(pred_dir + f'predictions_lstm_{preprocess_mode}_{feature_read}.csv')

    classif_report = classification_report(predictions_df[feature + '_real'], predictions_df[feature + '_predicted'], output_dict=True)

    results_per_feat = pd.concat([results_per_feat, pd.DataFrame({'feature': [feature], 
                                                                'precision_0': [classif_report['0']['precision']], 'recall_0': [classif_report['0']['recall']], 'f1_score_0': [classif_report['0']['f1-score']],
                                                                'precision_1': [classif_report['1']['precision']], 'recall_1': [classif_report['1']['recall']], 'f1_score_1': [classif_report['1']['f1-score']]})])
    
results_per_feat = pd.concat({'metrics': results_per_feat.set_index('feature').transpose()})
results_per_feat

# In[9]:


show_best_and_lowest_scores(results_per_feat, score='f1_score')

# In[10]:


show_best_and_lowest_scores(results_per_feat, score='recall')

# In[11]:


show_best_and_lowest_scores(results_per_feat, score='precision')

# In[9]:


show_precision_vs_recall(results_per_feat)

# In[13]:


### Nicer plot for presentations
#features_of_interest=None
#modify
features_of_interest = ['irm.lusg']
normalize_scale=True

# Extract precision_1 and recall_1
df_plot = results_per_feat.loc['metrics'].copy()
precision_1 = df_plot.loc['precision_1']
recall_1 = df_plot.loc['recall_1']

# Filter out NaN values
valid_indices = precision_1.notna() & recall_1.notna()
precision_1 = precision_1[valid_indices]
recall_1 = recall_1[valid_indices]

fig, ax = plt.subplots(figsize=(7, 7))
if features_of_interest != None:
    sns.scatterplot(x=precision_1, y=recall_1, hue=precision_1.index.isin(features_of_interest), palette={True: 'red', False: 'blue'}, s=100, edgecolor='w')
else:
    sns.scatterplot(x=precision_1, y=recall_1, s=100, edgecolor='w')

# Ajouter des labels pour chaque point
for feature in precision_1.index:
    plt.text(precision_1[feature], recall_1[feature], feature, fontsize=12, ha='right')

if normalize_scale:
    plt.xlim([0, 1])
    plt.ylim([0, 1])
plt.xlabel('Spécificité', fontsize=14)
plt.ylabel('Sensitivité', fontsize=14)
plt.title('', fontsize=15)
if features_of_interest != None:
    handles, labels = ax.get_legend_handles_labels()
    labels = ['Other Features', 'Best features for endo. prediction']
    plt.legend(handles=handles, labels=labels, loc='best')
plt.show()

# ### Compare performance dropping vs. keeping nans

# In[40]:


upsampling = True
preprocess_mode = 'multiclass'

results_per_feat_keepna = pd.DataFrame(columns=['feature', 'precision_0', 'recall_0', 'f1_score_0', 'precision_1', 'recall_1', 'f1_score_1'])
for feature in features_of_interest:
    feature_read = feature.replace('.', '_')
    if upsampling:
        predictions_df = pd.read_csv(pred_dir + f'predictions_lstm_upsampled_{preprocess_mode}_{feature_read}.csv')
    else:
        predictions_df = pd.read_csv(pred_dir + f'predictions_lstm_{preprocess_mode}_{feature_read}.csv')

    classif_report = classification_report(predictions_df[feature + '_real'], predictions_df[feature + '_predicted'], output_dict=True)

    results_per_feat_keepna = pd.concat([results_per_feat_keepna, pd.DataFrame({'feature': [feature], 
                                                                'precision_0': [classif_report['0']['precision']], 'recall_0': [classif_report['0']['recall']], 'f1_score_0': [classif_report['0']['f1-score']],
                                                                'precision_1': [classif_report['1']['precision']], 'recall_1': [classif_report['1']['recall']], 'f1_score_1': [classif_report['1']['f1-score']]})])
    
results_per_feat_keepna = pd.concat({'metrics': results_per_feat_keepna.set_index('feature').transpose()})
results_per_feat_keepna

# In[41]:


upsampling = True
preprocess_mode = 'dropna'

results_per_feat_dropna = pd.DataFrame(columns=['feature', 'precision_0', 'recall_0', 'f1_0', 'precision_1', 'recall_1', 'f1_1'])
for feature in features_of_interest:
    feature_read = feature.replace('.', '_')
    if upsampling:
        predictions_df = pd.read_csv(pred_dir + f'predictions_lstm_upsampled_{preprocess_mode}_{feature_read}.csv')
    else:
        predictions_df = pd.read_csv(pred_dir + f'predictions_lstm_{preprocess_mode}_{feature_read}.csv')

    classif_report = classification_report(predictions_df[feature + '_real'], predictions_df[feature + '_predicted'], output_dict=True)

    results_per_feat_dropna = pd.concat([results_per_feat_dropna, pd.DataFrame({'feature': [feature], 
                                                                'precision_0': [classif_report['0']['precision']], 'recall_0': [classif_report['0']['recall']], 'f1_score_0': [classif_report['0']['f1-score']],
                                                                'precision_1': [classif_report['1']['precision']], 'recall_1': [classif_report['1']['recall']], 'f1_score_1': [classif_report['1']['f1-score']]})])
    
results_per_feat_dropna = pd.concat({'metrics': results_per_feat_dropna.set_index('feature').transpose()})
results_per_feat_dropna

# In[112]:


show_precision_vs_recall(results_per_feat_keepna)

# In[42]:


show_precision_vs_recall(results_per_feat_dropna)

# ### 2-step procedure: 1) Predict absence or presence of info in gyneco file 2) For those where info present, predict whether symptom absent or present

# ### 1) Info present or not

# In[56]:


target_feature = 'atcd.endo'

# In[57]:


# For this take the data where infos have been extracted manually from gyneco files
recueil  = pd.read_excel('/home/nounou/endopath/Data/DATA_PROCESSED/data_gynéco_manual_extraction.xlsx')
recueil.set_index('Anonymisation', inplace=True)
recueil[target_feature] = recueil[target_feature].replace({np.nan: 2.0, 'NaN': 2.0})

# In[58]:


# Calculate the distribution of classes for the selected feature
class_counts = recueil[target_feature].value_counts()

nmissing = recueil[target_feature].isnull().sum()
npresent = recueil.loc[recueil[target_feature]==1].shape[0]
nabsent = recueil.loc[recueil[target_feature]==0].shape[0]
print(f'Number of patients with info missing: {nmissing}')
print(f'Number of patients with symptom present: {npresent}')
print(f'Number of patients with symptom absent: {nabsent}')

# Create a bar plot
plt.figure(figsize=(8, 6))
class_counts.plot(kind='bar', color=['skyblue', 'orange', 'red'])
plt.title(f'Original distribution of Classes for {target_feature}')
plt.xlabel('Class')
plt.ylabel('Count')
plt.xticks(rotation=0)  # Rotate x-axis labels if necessary
plt.grid(axis='y', linestyle='--', alpha=0.7)

# In[59]:


recueil_feature_present = recueil[[target_feature]].copy()
recueil_feature_present[target_feature] = recueil_feature_present[target_feature].replace({2.0: 0, 0.0: 1, 1.0: 1})
_, _, _, _, max_vocab, X, Y = preprocess_and_split( df_nlp,
                                                   recueil_feature_present,
                                                   'all',
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

Y = Y[[target_feature]]

Y_one_hot = to_categorical(Y, num_classes=2)
Y_one_hot = pd.DataFrame(Y_one_hot, columns=['négatif', 'positif'], index=Y.index)

# Split :
X_train, X_test = train_test_split(X, random_state=42, test_size=0.2)
indeces_test = list(X_test.index)
Y_train = from_X_split_get_Y_split(X_train, Y_one_hot)
Y_test = from_X_split_get_Y_split(X_test, Y_one_hot).values

if use_validation_set:
    X_val = X_train.iloc[X_train.shape[0]-10:,:]
    indeces_val = list(X_val.index)
    Y_val = Y_train.iloc[Y_train.shape[0]-10:,:].values
    X_train = X_train.iloc[:X_train.shape[0]-10,:]
    indeces_train = list(X_train.index)
    Y_train = Y_train.iloc[:Y_train.shape[0]-10,:].values
print(f'Number of training samples: {X_train.shape[0]}')
if use_validation_set: print(f'Number of validation samples: {X_val.shape[0]}');
print(f'Number of testing samples: {X_test.shape[0]}')

# In[60]:


data = pd.merge(X, Y_one_hot.reset_index().rename(columns={'Numéro anonymat': 'Anonymisation'}), on='Anonymisation')
data = pd.merge(data, Y.reset_index().rename(columns={'Numéro anonymat': 'Anonymisation'}), on='Anonymisation')
data_mentioned = data.loc[data[target_feature]==1.0]
data_not_mentioned = data.loc[data[target_feature]==0.0]

# In[61]:


# Create a bar plot
fig, ax = plt.subplots(figsize=(8, 6))
plt.bar([1, 2], [data_mentioned.shape[0], data_not_mentioned.shape[0]], color=['green', 'orange'])
plt.title(f'Original distribution of Classes for {target_feature}')
plt.ylabel('Count')
plt.xticks([1, 2])
ax.set_xticklabels(['Info mentioned', 'Info not mentioned'])
plt.xticks(rotation=0)  # Rotate x-axis labels if necessary
plt.grid(axis='y', linestyle='--', alpha=0.7)

# In[62]:


if len(data_not_mentioned) > len(data_mentioned):
    data_majority = data_not_mentioned
    data_minority = data_mentioned
else:
    data_majority = data_mentioned
    data_minority = data_not_mentioned

bias = data_minority.shape[0]/data_majority.shape[0]

# split train/test data first 
data_train = pd.concat([data_majority.sample(frac=train_proportion,random_state=200), data_minority.sample(frac=train_proportion,random_state=200)])
data_test = pd.concat([data_majority.drop(data_majority.sample(frac=train_proportion,random_state=200).index), data_minority.drop(data_minority.sample(frac=train_proportion,random_state=200).index)])

data_mentioned_test = data_test.loc[data_test[target_feature] == 1.0]
data_not_mentioned_test = data_test.loc[data_test[target_feature] == 0.0]
if use_validation_set:
    data_val = pd.concat([data_not_mentioned_test.sample(frac=0.5,random_state=200),
            data_mentioned_test.sample(frac=0.5,random_state=200)])
    data_test = pd.concat([data_not_mentioned_test.drop(data_not_mentioned_test.sample(frac=0.5,random_state=200).index),
            data_mentioned_test.drop(data_mentioned_test.sample(frac=0.5,random_state=200).index)])

data_train = shuffle(data_train)
if use_validation_set: data_val = shuffle(data_val)
data_test = shuffle(data_test)

# In[63]:


print('data with info present in training:',(data_train[target_feature] == 1.0).sum())
print('data with info missing in training:',(data_train[target_feature] == 0.0).sum())
if use_validation_set:
    print('data with info present in validation:',(data_val[target_feature] == 1.0).sum())
    print('data with info missing in validation:',(data_val[target_feature] == 0.0).sum())
print('data with info present in test:',(data_test[target_feature] == 1.0).sum())
print('data with info missing in test:',(data_test[target_feature] == 0.0).sum())

# In[64]:


# Separate majority and minority classes in training data for upsampling 
if len(data_train[data_train[target_feature] < 2]) > len(data_train[data_train[target_feature] == 2.0]):
    data_train_majority = data_train[data_train[target_feature] == 1.0]
    data_train_minority = data_train[data_train[target_feature] == 0.0]
else:
    data_train_majority = data_train[data_train[target_feature] == 1.0]
    data_train_minority = data_train[data_train[target_feature] == 0.0]

print("majority class before upsample:",data_majority.shape)
print("minority class before upsample:",data_minority.shape)

# Upsample minority class
data_train_minority_upsampled = resample(data_train_minority, 
                                 replace=True,     # sample with replacement
                                 n_samples= data_train_majority.shape[0],    # to match majority class
                                 random_state=123) # reproducible results
 
# Combine majority class with upsampled minority class
data_train_upsampled = pd.concat([data_train_majority, data_train_minority_upsampled])
 
# Display new class counts
print("After upsampling\n",data_train_upsampled[target_feature].value_counts(),sep = "")

# In[65]:


# Calculate the distribution of classes for the selected feature
data_mentioned = data_train_upsampled.loc[data_train_upsampled[target_feature] == 1.0]
data_not_mentioned = data_train_upsampled.loc[data_train_upsampled[target_feature] == 0.0]

# Create a bar plot
fig, ax = plt.subplots(figsize=(8, 6))
plt.bar([1, 2], [data_mentioned.shape[0], data_not_mentioned.shape[0]], color=['green', 'orange'])
plt.title(f'Distribution of Classes for {target_feature} in training after upsampling')
plt.ylabel('Count')
plt.xticks([1, 2])
ax.set_xticklabels(['Info mentioned', 'Info not mentioned'])
plt.xticks(rotation=0)  # Rotate x-axis labels if necessary
plt.grid(axis='y', linestyle='--', alpha=0.7)

# In[25]:


X_train = data_train_upsampled[['Anonymisation', 'Résumé']]
Y_train = data_train_upsampled[['négatif', 'positif']].values
if use_validation_set:
    X_val = data_val[['Anonymisation', 'Résumé']]
    Y_val = data_val[['négatif', 'positif']].values
X_test = data_test[['Anonymisation', 'Résumé']]
Y_test = data_test[['négatif', 'positif']].values

# In[69]:


max_vocab_size = max_vocab 

tokenizer = Tokenizer(num_words=max_vocab_size, split=' ', oov_token='<unw>', filters=' ')
tokenizer.fit_on_texts(pd.concat([X_train,X_val]).loc[:,'Résumé'])

# This encodes our sentence as a sequence of integer
# each integer being the index of each word in the vocabulary
train_seqs = tokenizer.texts_to_sequences(X_train.loc[:,'Résumé'])
if use_validation_set: valid_seqs = tokenizer.texts_to_sequences(X_val.loc[:,'Résumé']);
test_seqs = tokenizer.texts_to_sequences(X_test.loc[:,'Résumé'])

# We need to pad the sequences so that they are all the same length :
# the length of the longest one
max_seq_length = max( [len(seq) for seq in train_seqs + valid_seqs] )

X_train_pad = pad_sequences(train_seqs, max_seq_length)
if use_validation_set: X_valid_pad = pad_sequences(valid_seqs, max_seq_length);
X_test_pad = pad_sequences(test_seqs, max_seq_length)

embedding_dim = 100
lstm_out_dim = 200
dropout_rate = 0.2

def get_lstm_model(vocab_size, embedding_dim, lstm_out_dim, dropout_rate, n_dense, nclasses):
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, trainable=True))

    model.add(LSTM(units=lstm_out_dim, return_sequences=True))
    model.add(Dropout(dropout_rate))
    
    model.add(LSTM(units=lstm_out_dim))
    model.add(Dropout(dropout_rate))

    model.add(Dense(units=128, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(units=64, activation='relu'))
    model.add(Dropout(dropout_rate))

    if nclasses==2:
        print('2 classes, using binary crossentropy')
        model.add(Dense(n_dense, input_dim=lstm_out_dim, activation='sigmoid'))
        model.compile(loss = 'binary_crossentropy', optimizer='adam', metrics = ['accuracy', 'Recall'])
    else:
        print(f'{nclasses} classes, using categorical crossentropy')
        model.add(Dense(n_dense, input_dim=lstm_out_dim, activation='softmax'))
        model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics = ['accuracy', 'Recall'])
    
    return model


model = get_lstm_model(max_vocab_size, embedding_dim, lstm_out_dim, dropout_rate, n_dense=Y_train.shape[1], nclasses=2)
# print(model.summary())
early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, verbose=1)
batch_size = 64
max_epochs = 20

if use_validation_set:
    history = model.fit(X_train_pad, Y_train, epochs=max_epochs, batch_size=batch_size, 
                        verbose=1, validation_data = (X_valid_pad, Y_val), callbacks=[early_stopping])
else:
    history = model.fit(X_train_pad, Y_train, epochs=max_epochs, batch_size=batch_size, verbose=1)

# In[71]:


pickle.dump(model, open(model_dir + f'lstm_upsampled_{target_feature}_info_absent_vs_present'.replace('.', '_'), 'wb'))

# In[72]:


predictions = model.predict(X_test_pad)
predictions

# In[73]:


predictions_df = pd.DataFrame(columns=[target_feature + '_predicted', target_feature + '_real'])
#predictions_df[target_feature + '_predicted_proba'] = predictions
predictions_df[target_feature + '_predicted'] = np.argmax(predictions, axis=1).tolist()
predictions_df[target_feature + '_real'] = np.argmax(Y_test, axis=1).tolist()
predictions_df.head()

# In[74]:


print(classification_report(predictions_df[target_feature + '_real'], predictions_df[target_feature + '_predicted']))
