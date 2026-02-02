#!/usr/bin/env python
# coding: utf-8

# #  Context : 

# Le but de ce notebook est de mettre en place l'utilisation du texte résumé dans les notebook précédents pour essayer de faire la prédiction en symptomes.
# 
# La première étape est d'utiliser un TfIdf en guise de baseline afin d'obtenir des données utilisable par un algorithme pour la prédiction.
# 
# Cette étape de prédiction doit être couplée avec le package eli5 et nous ''oblige'' à passer par une voie multiclass et donc un modèle par symptomes à prédire, plutot qu'une voie multilabel-multioutputs qui utiliserait 1 modèle.
# 
# - baseline bruité
# - sans le bruit
# - etude N-gram

# # Import : 

# In[3]:


### Imports ###

# Data manipulation and other stuff : 
import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 500)
from matplotlib import pyplot as plt
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


# # NLP 

# ## Load data : 

# In[4]:


# Loading X : 
df_nlp = pd.read_csv('/home/nounou/endopath/Data/DATA_PROCESSED/donnees_entree_nlp_sans_endo.csv', usecols=['Anonymisation', 'Date', 'Nature', 'Résumé'])
print('X shape is :', df_nlp.shape)

# Loading Y : 
recueil  = pd.read_excel('/home/nounou/endopath/Data/DATA_RAW/Recueil (1).xlsx').drop('Unnamed: 90', axis=1)
recueil.replace(['Na', 'NA'], np.nan, inplace=True)
recueil.set_index('Numéro anonymat', inplace=True)
print('Y shape is :', recueil.shape)
print(f'Nombre de patientes dans le df_nlp : {len(df_nlp.Anonymisation.unique())}')

# In[8]:


if 'DJ-055' in list(df_nlp['Anonymisation']):
    df_nlp.loc[df_nlp['Anonymisation']=='DJ-055', 'Anonymisation'] ='NJ-055'

# In[9]:


'NJ-055' in list(df_nlp['Anonymisation'])

# In[10]:


'DJ-055' in list(df_nlp['Anonymisation'])

# ## Preprocess 

# In[1]:


# Préprocess les datas : 
X_train, X_test, Y_train, Y_test, max_vocab = preprocess_and_split( df_nlp,
                                                                    recueil,
                                                                    'all',
                                                                    42,                     # choix de la seed pour le random split
                                                                    0.20,                   # Choix du test_size
                                                                    special_char=True,      # supprime les caractères spéciaux  
                                                                    accents=True,           # supprime les accents
                                                                    lower=False,            # passe la casse du texte en minuscule
                                                                    token=False,            # tokenize le corpus
                                                                    remove_stopwords=False,
                                                                    drop_number=True,
                                                                    compress=True,          # un corpus par n_ano, shape = [200,2]
                                                                    preprocess_mode='multiclass',
                                                                    encoder_mode=True,
                                                                    anapath=True
                                                                  )
print(f'Nombre de patientes inclues dans l\'étude NLP "baseline" : {X_train.shape[0] + X_test.shape[0]}')
print(f'Nombre de symptômes/caractéristiques à prédire dans l\'étude NLP "baseline" : {Y_train.shape[1]}')

# In[24]:


X_train.head()

# In[25]:


# Préprocess les datas : 
X_train_sw, X_test_sw, Y_train_sw, Y_test_sw, max_vocab_sw = preprocess_and_split( df_nlp,
                                                                    recueil,
                                                                    'all',
                                                                    42,                     # choix de la seed pour le random split
                                                                    0.20,                   # Choix du test_size
                                                                    special_char=True,      # supprime les caractères spéciaux  
                                                                    accents=True,    # supprime les accents
                                                                    lower=False,            # passe la casse du texte en minuscule
                                                                    token=True,            # tokenize le corpus
                                                                    remove_stopwords=True,
                                                                    compress=True,          # un corpus par n_ano, shape = [200,2]
                                                                    preprocess_mode='multiclass',
                                                                    encoder_mode=True,
                                                                    anapath=True
                                                                  )
print(f'Nombre de patientes inclues dans l\'étude NLP sans stopword : {X_train_sw.shape[0] + X_test_sw.shape[0]}')

# In[26]:


X_train_sw.head()

# In[15]:


### Rappel des classes : 
# 0 : Negatif 
# 1 : Positif
# 2 : Non mentionnées

### Guidelines de l'études : 
#1 baseline bruité
#2 sans le bruit
#3 etude N-gram

                                # ne pas prendre en compte les chiffres  

# >>> classe 1 precision rappel UNIQUEMENT 
# F1_score pour chaque label

#Classer du F1 score le + au -

# mot qui participe à la pred de la classe 1 pour chaque target (donc chaque colonne)
#eli5 package !!!! 

# ____

# ## TF-IDF
# 

# In[9]:


### avec stopwords

# In[16]:


#max_vocab = 25000
tfIdfVectorizer=TfidfVectorizer(use_idf=True, max_features=max_vocab, lowercase=False)
X_train_fitted = tfIdfVectorizer.fit_transform(X_train.Résumé)
X_train_fitted_df = pd.DataFrame(X_train_fitted.todense(), columns=tfIdfVectorizer.get_feature_names_out(), index= X_train.Anonymisation)
X_test_fitted = tfIdfVectorizer.transform(X_test.Résumé)
X_test_fitted_df = pd.DataFrame(X_test_fitted.todense(), columns=tfIdfVectorizer.get_feature_names_out(), index= X_test.Anonymisation)

# In[17]:


X_test_fitted_df.head()

# In[18]:


X_train_fitted_df.shape

# ## Test ELI5  MULTILABEL MULTICLASS :

# In[13]:


# class MultiLabelProbClassifier(BaseEstimator, ClassifierMixin):

#     def __init__(self, clf):
#         self.clf = clf

#     def fit(self, X, y):
#         self.clf.fit(X, y)

#     def predict(self, X):
#         ret = self.clf.predict(X)
#         return ret

#     def predict_proba(self, X):
#         if X.shape[0] == 1:
#             self.probas_ = self.clf.predict_proba(X)[0]
#             sums_to = sum(self.probas_)
#             new_probs = [x / sums_to for x in self.probas_]
#             return new_probs
#         else:
#             self.probas_ = self.clf.predict_proba(X)
#             print(self.probas_)
#             ret_list = []
#             for list_of_probs in self.probas_:
#                 sums_to = sum(list_of_probs)
#                 print(sums_to)
#                 new_probs = [x / sums_to for x in list_of_probs]
#                 ret_list.append(np.asarray(new_probs))
#             return np.asarray(ret_list)


# model_test  = DecisionTreeClassifier(random_state=42)
# the_model = MultiLabelProbClassifier(model_test)
# pipe = Pipeline([('Tfidf', TfidfVectorizer(use_idf=True, max_features=max_vocab, lowercase=False)), ('model', the_model)])
# pipe.fit(X_train.Résumé, Y_train)

# pred_test = pipe.predict(X_test.Résumé)


# te = TextExplainer(random_state=42, n_samples=300, position_dependent=True)

# def explain_pred(sentence):
#     te.fit(sentence, pipe.predict_proba)
#     t_pred = te.explain_prediction()
#     #t_pred = te.explain_prediction(top = 20, target_names=["ANB", "CAP", "ECON", "EDU", "ENV", "EX", "FED", "HEG", "NAT", "POL", "TOP", "ORI", "QER","COL","MIL", "ARMS", "THE", "INTHEG", "ABL", "FEM", "POST", "PHIL", "ANAR", "OTHR"])
#     txt = format_as_text(t_pred)
#     html = format_as_html(t_pred)
#     html_file = open("latest_prediction.html", "a+")
#     html_file.write(html)
#     html_file.close()
#     print(te.metrics_)

# In[14]:


# pipe.predict_proba(X_test.loc[:,"Résumé"])

# In[15]:


len(X_test.Résumé)

# In[16]:


# explain_pred(X_test.Résumé.iloc[0])

# ## Prédiction

# ### DecisionTree

# In[19]:


####################################################################
multilabel_classifier = DecisionTreeClassifier(random_state=42)
labels_MLC = {0:'négatif', 1:'positif', 2:' Non mentionnées'}
multilabel_classifier.fit(X_train_fitted_df, Y_train)
####################################################################

# In[29]:


plot_tree(multilabel_classifier)
plt.show()

# In[30]:


feature_imp = pd.Series(multilabel_classifier.feature_importances_, index=multilabel_classifier.feature_names_in_)
feature_imp_short = feature_imp.loc[feature_imp !=0]

plt.figure(figsize=(25,10))
plt.bar(list(feature_imp_short.index), list(feature_imp_short.values))
plt.xticks(rotation =45)
plt.show()

# In[22]:


# eli5 not working anymore for current stable version of sklearn
eli5.explain_weights_df(multilabel_classifier, feature_names=multilabel_classifier.feature_names_in_, top=10)

# In[22]:


# eli5.show_weights(multilabel_classifier, feature_names=list(multilabel_classifier.feature_names_in_), top=10)

# In[31]:


X_train_fitted_df.shape

# In[32]:


Y_train.shape

# In[33]:


Y_pred = multilabel_classifier.predict(X_test_fitted_df)
Y_pred = pd.DataFrame(Y_pred, columns=Y_train.columns)
DT_Multi_index, CR_global = rapport_metrics_decision_tree(Y_test, Y_pred)

# In[34]:


DT_Multi_index

# In[35]:


CR_global

# In[18]:


Y_pred.shape

# In[20]:


X_test_fitted_df

# ### RFC

# In[28]:


####################################################################
multilabel_classifier_RFC = RandomForestClassifier(random_state=42)
# labels_MLC = {0:'négatif', 1:'positif', 2:' Non mentionnées'}
multilabel_classifier_RFC.fit(X_train_fitted_df, Y_train)
####################################################################

# In[29]:


feature_imp_RF = pd.Series(multilabel_classifier_RFC.feature_importances_, index=multilabel_classifier_RFC.feature_names_in_)
feature_imp_RF_short = feature_imp_RF.loc[feature_imp_RF >0.003]

plt.figure(figsize=(25,10))
plt.bar(list(feature_imp_RF_short.index), list(feature_imp_RF_short.values))
plt.xticks(rotation =45)
plt.show()

# In[29]:


# # df_to_transform = eli5.explain_weights_df(logregcv, vec=tfIdfVectorizer, top=6, target_names=label_temp)

# def reshape_df_explain(df_to_transform):
#     cols = list(df_to_transform)
#     rows = list(df_to_transform.index)
#     col_len = len(cols)
#     row_len = len(rows)
#     dict_ = {}
#     for row in range(row_len):
#         for col in range(col_len):        
#             dict_[f'{str(rows[row])}_{cols[col]}'] = df_to_transform.iloc[row,col]
#     return dict_

# import eli5
# from eli5.lime import TextExplainer
# from IPython.display import display

# # display(eli5.show_weights(multilabel_classifier))

# PP_test = multilabel_classifier.predict_proba(pd.DataFrame(X_test_fitted_df.iloc[0,:]).T)

# # multilabel_classifier.feature_names_in_

# def list_to_array(list_):
#     array = np.zeros(len(list_))
#     for idx, array in list_:
#         array[idx] = list_[idx]

# max_ = 0
# max_value=-1
# for array in PP_test:
#     length = len(array[0])
#     max_temp = np.max(array[0])
#     if max_<length:
#         max_ = length
#     if max_temp > max_value:
#         max_value = max_temp
# print(max_, max_value)



# array = np.zeros(2)
# print(len(array))
# array = np.pad(array, (1,0))
# array





# # eli5.show_prediction(estimator=multilabel_classifier, doc=pd.DataFrame(X_test_fitted_df.iloc[0,:]).T,
# #                     feature_names=list(X_test_fitted_df.columns),
# #                     show_feature_values=True)

# # eli5.sklearn.explain_weights.explain_decision_tree(multilabel_classifier, vec = tfIdfVectorizer)


# # eli5.explain_prediction()

# def pad_and_contract(array):
#     liste_array=[]
#     for array in array_to_pad:
#         if len(array)<=2:
#             array= list(array[0])
#             array.append(0.)
#             if len(array)<=2:
#                 array.append(0.)
                
#         liste_array.append(array)
#     return np.asarray(liste_array)

# def PP(self, doc):
#     self.transform = tfIdfVectorizer.transform(doc)
#     predicted_proba = multilabel_classifier.predict_proba(doc_fitted)
#     return pad_and_contract(array_to_pad)

# array = PP(X_test.Résumé.iloc[0])

# type(X_test.Résumé.iloc[0])

# eli5.sklearn.explain_prediction_sklearn(multilabel_classifier_RFC, X_test.Résumé.iloc[0], vec=tfIdfVectorizer)

# exp = TextExplainer(n_samples=5000, clf=multilabel_classifier, vec=tfIdfVectorizer, char_based=False,random_state=42, position_dependent=False)
# exp.fit(X_test.Résumé.iloc[0], 

# PP

# )

# eli5.sklearn.explain_weights_sklearn(multilabel_classifier, vec=tfIdfVectorizer)



# eli5.show_weights(multilabel_classifier, vec=tfIdfVectorizer)

# eli5.show_weights(multilabel_classifier, vec=tfIdfVectorizer)

# In[30]:


CR_global

# https://blog.octo.com/nlp-une-classification-multilabels-simple-efficace-et-interpretable/

# In[34]:


#RandomForest, voir xgBoost

# ### LogisticRegressionCV

# In[35]:


# Rappel des classes : 
# 0 : Negatif 
# 1 : Positif
# 2 : Données manquantes 

# In[36]:


# ONE hot pas possible parce que pas assez de chir présent ....
# supprimer les chir peu présent ? donc drop des colonnes ?

# In[31]:


if len(list(Y_train['PSH'].unique()))<=1:
    print('yep')
else:
    print('nope')

# In[36]:


# drop des colonnes uniques : 
for nom, values in Y_train.items():
    if len(list(values.unique())) <=1: 
        print(nom)
        Y_train.drop(nom, axis=1, inplace=True)

# In[37]:


def find_labels(liste):
    liste = pd.Series(np.unique(liste))
    liste.replace(labels_MLC, inplace=True)
    return list(liste)

# In[40]:


# eli5.explain_weights_df(logregcv, feature_names=X_train_fitted_df.columns, 
#                   top=10, target_names=label_temp)

# In[41]:


# display(eli5.explain_weights(logregcv, vec=tfIdfVectorizer, 
#                   top=6, target_names=label_temp))
# print(nom)
# df_to_transform = eli5.explain_weights_df(logregcv, vec=tfIdfVectorizer, top=6, target_names=label_temp)

# In[42]:


# df_to_transform = eli5.explain_weights_df(logregcv, vec=tfIdfVectorizer, top=6, target_names=label_temp)
# cols = list(df_to_transform)
# rows = list(df_to_transform.index)
# col_len = len(cols)
# row_len = len(rows)
# dict_ = {}
# for row in range(row_len):
#     for col in range(col_len):        
#         dict_[f'{str(rows[row])}_{cols[col]}'] = df_to_transform.iloc[row,col]
# pd.DataFrame({'col':dict_})

# In[42]:


# Nicolai: doesn't work because of eli5 incompatibility with sklearn
#CR_global_LR, LR_Multi_index, LR_y_pred, dict_model_LR = multilabel_multioutput_LR(X_train_fitted_df, X_test_fitted_df, Y_train, Y_test, tfIdfVectorizer)
#CR_global_LR.loc['metrics',:]

# In[45]:


#LR_Multi_index

# In[46]:


#LR_y_pred

# In[47]:


#dict_model_LR

# ### SCV 

# In[3]:


CR_global_SVC, SVC_Multi_index, SVC_y_pred, dict_model_svc = multilabel_multioutput_svc(X_train_fitted_df, X_test_fitted_df, Y_train, Y_test, tfIdfVectorizer)

# In[49]:


CR_global_SVC, SVC_Multi_index, SVC_y_pred, dict_model_svc = multilabel_multioutput_svc(X_train_fitted_df, X_test_fitted_df, Y_train, Y_test, tfIdfVectorizer)

# In[50]:


CR_global_SVC

# In[51]:


for nom in Y_train.columns:
    print(nom)
    custom_show_weights(dict_model_LR, tfIdfVectorizer, nom)

# In[52]:


CR_global_SVC.loc['metrics',:]

# ## NLP avec correction : 

# In[14]:


# Préprocess les datas :
X_train_corr, X_test_corr, Y_train, Y_test, max_vocab = preprocess_and_split( df_nlp,
                                                                    recueil,
                                                                    'all',
                                                                    42,                     # choix de la seed pour le random split
                                                                    0.20,                   # Choix du test_size
                                                                    special_char=True,      # supprime les caractères spéciaux
                                                                    accents=True,    # supprime les accents
                                                                    lower=False,            # passe la casse du texte en minuscule
                                                                    token=False,            # tokenize le corpus
                                                                    remove_stopwords=False,
                                                                    drop_number=True,
                                                                    compress=True,          # un corpus par n_ano, shape = [200,2]
                                                                    preprocess_mode='multiclass',
                                                                    encoder_mode=True,
                                                                    anapath=True,
                                                                    correction=True
                                                                  )
print(f'Nombre de patientes inclues dans l\'étude NLP "baseline" : {X_train.shape[0] + X_test.shape[0]}')
print(f'Nombre de symptômes/caractéristiques à prédire dans l\'étude NLP "Correction" : {Y_train.shape[1]}')

# In[16]:


# drop des colonnes uniques : 
for nom, values in Y_train.items():
    if len(list(values.unique())) <=1: 
        print(nom)
        Y_train.drop(nom, axis=1, inplace=True)

# In[17]:


#max_vocab = 25000
tfIdfVectorizer_corr = TfidfVectorizer(use_idf=True, max_features=max_vocab, lowercase=False)
X_train_corr_fitted = tfIdfVectorizer_corr.fit_transform(X_train_corr.Résumé)
X_train_corr_fitted_df = pd.DataFrame(X_train_corr_fitted.todense(), columns=tfIdfVectorizer_corr.get_feature_names_out(), index= X_train.Anonymisation)
X_test_corr_fitted = tfIdfVectorizer_corr.transform(X_test.Résumé)
X_test_corr_fitted_df = pd.DataFrame(X_test_corr_fitted.todense(), columns=tfIdfVectorizer_corr.get_feature_names_out(), index= X_test.Anonymisation)

# In[18]:


CR_global_SVC_corr, SVC_Multi_index_corr, SVC_y_pred_corr, dict_model_svc_corr = multilabel_multioutput_svc(X_train_corr_fitted_df, X_test_corr_fitted_df, Y_train, Y_test, tfIdfVectorizer_corr)

# In[57]:


CR_global_LR_corr, LR_Multi_index_corr, LR_y_pred_corr, dict_model_LR_corr = multilabel_multioutput_LR(X_train_corr_fitted_df, X_test_corr_fitted_df, Y_train, Y_test, tfIdfVectorizer_corr)

# In[58]:


CR_global_SVC_corr

# In[59]:


CR_global_LR_corr

# ## LSTM

# In[2]:


recueil

# In[11]:


X_train2, X_test2, Y_train2, Y_test2, max_vocab2 = preprocess_and_split( df_nlp,
                                                                    recueil,
                                                                    42,                     # choix de la seed pour le random split
                                                                    0.20,                   # Choix du test_size
                                                                    special_char=False,     # supprime les caractères spéciaux  
                                                                    lower=False,            # passe la casse du texte en minuscule
                                                                    token=False,            # tokenize le corpus
                                                                    remove_stopwords=False,
                                                                    compress=True,          # un corpus par n_ano, shape = [200,2]
                                                                    preprocess_mode='multiclass',
                                                                    encoder_mode=True,
                                                                    anapath=True
                                                                  )
X_val2 = X_train2.iloc[117:,:]
Y_val = Y_train2.iloc[117:,:]
X_train2 = X_train2.iloc[:117,:]
Y_train2 = Y_train2.iloc[:117,:]
# Sauvegarder les files pour pouvoir les réouvrir et repartir d'ici !!!!!!

# In[8]:


max_vocab_size = max_vocab2 

tokenizer = Tokenizer(num_words=max_vocab_size, split=' ', oov_token='<unw>', filters=' ')
tokenizer.fit_on_texts(pd.concat([X_train2,X_val2]).loc[:,'Résumé'])

# This encodes our sentence as a sequence of integer
# each integer being the index of each word in the vocabulary
train_seqs = tokenizer.texts_to_sequences(X_train2.loc[:,'Résumé'])
valid_seqs = tokenizer.texts_to_sequences(X_val2.loc[:,'Résumé'])
test_seqs = tokenizer.texts_to_sequences(X_test2.loc[:,'Résumé'])

# In[63]:


# # We need to pad the sequences so that they are all the same length :
# # the length of the longest one
# max_seq_length = max( [len(seq) for seq in train_seqs + valid_seqs] )

# X_train_pad = pad_sequences(train_seqs, max_seq_length)
# X_valid_pad = pad_sequences(valid_seqs, max_seq_length)
# X_test_pad = pad_sequences(test_seqs, max_seq_length)

# def get_lstm_model_2(vocab_size, embedding_dim, seq_length, lstm_out_dim, dropout_rate, n_dense=99):
#     model = Sequential()
#     model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=seq_length))
#     model.add(LSTM(units=lstm_out_dim))
#     model.add(Dropout(dropout_rate))
#     model.add(Dense(n_dense, input_dim=lstm_out_dim, activation='sigmoid'))
    
#     # NE PAS OUBLIER DE DEFINIR LA METRIQUE !!!!!
#     model.compile(loss = 'binary_crossentropy', optimizer='adam',metrics = ['accuracy', 'Recall'])
#     return model

# embedding_dim = 100
# lstm_out_dim = 200
# dropout_rate = 0.2

# model = get_lstm_model_2(max_vocab_size, embedding_dim, max_seq_length, lstm_out_dim, dropout_rate)
# # print(model.summary())
# early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, verbose=1)
# batch_size = 64
# max_epochs = 10
# history = model.fit(X_train_pad, Y_train2, epochs=max_epochs, batch_size=batch_size, 
#                     verbose=0, validation_data = (X_valid_pad, Y_val), callbacks=[early_stopping])
    

# In[64]:


       
# # Définir la metrique dans le .compile !!!!!
# test_acc = model.evaluate(X_test2, Y_test2, verbose=0)     

# In[ ]:




# In[ ]:




# In[ ]:



