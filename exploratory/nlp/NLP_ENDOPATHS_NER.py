#!/usr/bin/env python
# coding: utf-8

# #  NLP_ENDOPATHS_NER

# Perform Named Entity Recognition (NER) on the Endopath dataset

# Author: Nicolai Wolpert
# Email: nicolai.wolpert@capgemini.com
# Date: July 2024

# ## Imports

# In[1]:


### Imports ###

# Data manipulation and other stuff : 
import numpy as np
import pandas as pd
import re
import string
import os
#pd.set_option('display.max_rows', 10)
import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns
from IPython.display import display
from tqdm import tqdm

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

from sklearn.tree import plot_tree

# Multiclass/Multilabel preparation :
from sklearn.base import BaseEstimator, ClassifierMixin

# Kfold cross-validation with stratification
from exploratory.Opti_utils.ML_utils import kfold_cv_stratified
from exploratory.Opti_utils.ML_utils import Binarisation

# Custom preprocessing : 
from exploratory.preprocessing.preprocess_NLP import correction_series, lowercase_text, remove_special_characters, remove_number

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

from pprint import pprint
import functools

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from transformers import AutoModelForSequenceClassification, CamembertForMaskedLM, AutoTokenizer, AutoConfig
from datasets import load_dataset
from sklearn.metrics import confusion_matrix, f1_score

import plotly.express as px
from tqdm.notebook import tqdm

from datasets.dataset_dict import DatasetDict
from datasets import Dataset

import pickle

from nlstruct.recipes import train_ner
from nlstruct import load_pretrained
from nlstruct.datasets import load_from_brat, export_to_brat
from nlstruct.recipes import train_qualified_ner

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from collections import Counter

# In[2]:


model_dir = 'models/'
pred_dir = 'predictions/'
data_dir = 'Data/'

# ## Load data

# In[3]:


# Loading X : 
df_nlp = pd.read_csv('/home/nounou/endopath/Data/DATA_PROCESSED/donnees_entree_nlp_sans_endo.csv', usecols=['Anonymisation', 'Date', 'Nature', 'Résumé'])
df_nlp_orig = df_nlp.copy()
print('X shape is :', df_nlp.shape)

# Loading Y : 
recueil_orig  = pd.read_excel('/home/nounou/endopath/Data/DATA_RAW/Recueil (1).xlsx').drop('Unnamed: 90', axis=1)
recueil_orig = recueil_orig[['Numéro anonymat', 'atcd.endo', 'irm.lusg', 'tv.douloureux', 'irm.externe', 'sf.dig.diarrhee', 'echo.lusg', 'echo.lusd', 'ef.hormone.dpc', 'effet.hormone', 'anapath.lusd', 'anapath.lusg', 'anapath.torus', 'anapath.autre']]
recueil = recueil_orig.copy()
recueil.replace(['Na', 'NA'], np.nan, inplace=True)
recueil = recueil.rename(columns={'Numéro anonymat': 'Anonymisation'})
# Note the target variable, if endometriosis is present or not (corresponds to the anapth columns but not 'autre')
recueil['endometriose'] = recueil.loc[:,['anapath.lusd','anapath.lusg','anapath.torus']].sum(axis=1).apply(lambda x: Binarisation(x))
# Drop the anapath columns again
recueil = recueil[[c for c in recueil.columns if not c.startswith('anapath')]]

print(f'Nombre de patientes dans le df_nlp : {len(df_nlp.Anonymisation.unique())}')

if 'DJ-055' in list(df_nlp['Anonymisation']):
    df_nlp.loc[df_nlp['Anonymisation']=='DJ-055', 'Anonymisation'] ='NJ-055'
'NJ-055' in list(df_nlp['Anonymisation'])
'DJ-055' in list(df_nlp['Anonymisation'])

# Text correction/preparation
df_nlp.Résumé = df_nlp.Résumé.apply(remove_special_characters)
df_nlp.Résumé = df_nlp.Résumé.apply(lowercase_text)
df_nlp.Résumé = df_nlp.Résumé.apply(correction_series)

# Merge receuil and nlp dataframes and rename columns
#df_nlp = df_nlp.groupby('Anonymisation')['Résumé'].agg(' '.join).reset_index()  # join all texts for patients
data = pd.merge(df_nlp, recueil, on='Anonymisation', how='inner')
data = data.rename(columns={'Anonymisation': 'patient', 'Résumé': 'text'})
data = data.replace({0: 'absent', 1: 'present'})

# In[ ]:


data

# ## NER

# ### Gliner

# In[5]:


from gliner import GLiNER

model = GLiNER.from_pretrained("almanach/camembert-bio-gliner-v0.1")

# In[6]:


labels = ["Âge", "Patient", "Maladie", "Symptômes", "Médicament"]
if os.path.isfile(data_dir + 'Generate/df_entities.csv'):
    df_entities = pd.read_csv(data_dir + 'Generate/df_entities.csv')
else:
    df_entities = pd.DataFrame(columns=list(data.columns) + ['term', 'label', 'score'])
    for idx in tqdm(data.index):  # Loop through desired indices
        entities = model.predict_entities(data.loc[idx, 'text'], labels, threshold=0.5)
        for entity in entities:
            df_entity = data.loc[idx].copy()
            df_entity['term'] = entity["text"]
            df_entity['label'] = entity["label"]
            df_entity['score'] = entity["score"]
            df_entity = df_entity.to_frame().T  # Convert to DataFrame to concat
            df_entities = pd.concat([df_entities, df_entity], ignore_index=True)
    df_entities.to_csv(data_dir + 'Generate/df_entities.csv')
# Keep only unique terms one per patient
if os.path.isfile(data_dir + 'Generate/df_entities_unique.csv'):
    df_entities_unique = pd.read_csv(data_dir + 'Generate/df_entities_unique.csv')
else:
    df_entities_unique = pd.DataFrame()
    for patient in tqdm(list(pd.unique(df_entities.patient))):
        
        df_entities_unique_patient = df_entities.loc[df_entities.patient==patient]
        df_entities_unique_patient.drop_duplicates(subset=['term'], keep='first',inplace=True)
        df_entities_unique = pd.concat([df_entities_unique, df_entities_unique_patient])
    df_entities_unique.to_csv(data_dir + 'Generate/df_entities_unique.csv')
df_entities_unique

# In[7]:


# Extract all terms by type of label
medications_patients_with_endometriosis = list(df_entities_unique.loc[(df_entities_unique['endometriose']=='present') & (df_entities_unique['label']=='Médicament'), 'term'])
medications_patients_without_endometriosis = list(df_entities_unique.loc[(df_entities_unique ['endometriose']=='absent') & (df_entities_unique['label']=='Médicament'), 'term'])

symptoms_patients_with_endometriosis = list(df_entities_unique.loc[(df_entities_unique['endometriose']=='present') & (df_entities_unique['label']=='Symptômes'), 'term'])
symptoms_patients_without_endometriosis = list(df_entities_unique.loc[(df_entities_unique ['endometriose']=='absent') & (df_entities_unique['label']=='Symptômes'), 'term'])

sicknesses_patients_with_endometriosis = list(df_entities_unique.loc[(df_entities_unique['endometriose']=='present') & (df_entities_unique['label']=='Maladie'), 'term'])
sicknesses_patients_without_endometriosis = list(df_entities_unique.loc[(df_entities_unique ['endometriose']=='absent') & (df_entities_unique['label']=='Maladie'), 'term'])

# Cleaning
medications_patients_with_endometriosis = [term for term in medications_patients_with_endometriosis if len(term) > 2]
medications_patients_without_endometriosis = [term for term in medications_patients_without_endometriosis if len(term) > 2]
symptoms_patients_with_endometriosis = [term for term in symptoms_patients_with_endometriosis if len(term) > 2]
symptoms_patients_without_endometriosis = [term for term in symptoms_patients_without_endometriosis if len(term) > 2]
sicknesses_patients_with_endometriosis = [term for term in sicknesses_patients_with_endometriosis if len(term) > 2]
sicknesses_patients_without_endometriosis = [term for term in sicknesses_patients_without_endometriosis if len(term) > 2]

# Merge across types
terms_patients_with_endometriosis = medications_patients_with_endometriosis + symptoms_patients_with_endometriosis + sicknesses_patients_with_endometriosis
terms_patients_without_endometriosis = medications_patients_without_endometriosis + symptoms_patients_without_endometriosis + sicknesses_patients_without_endometriosis

# In[9]:


### Wordclouds, by type and presence of endometriosis

wordcloud_medications_with_endo = WordCloud(width = 1000, height = 500, background_color='white').generate_from_frequencies(Counter(medications_patients_with_endometriosis))
wordcloud_medications_without_endo = WordCloud(width = 1000, height = 500, background_color='white').generate_from_frequencies(Counter(medications_patients_without_endometriosis))

wordcloud_symptoms_with_endo = WordCloud(width = 1000, height = 500, background_color='white').generate_from_frequencies(Counter(symptoms_patients_with_endometriosis))
wordcloud_symptoms_without_endo = WordCloud(width = 1000, height = 500, background_color='white').generate_from_frequencies(Counter(symptoms_patients_without_endometriosis))

wordcloud_sicknesses_with_endo = WordCloud(width = 1000, height = 500, background_color='white').generate_from_frequencies(Counter(sicknesses_patients_with_endometriosis))
wordcloud_sicknesses_without_endo = WordCloud(width = 1000, height = 500, background_color='white').generate_from_frequencies(Counter(sicknesses_patients_without_endometriosis))

fig, axs = plt.subplots(3, 2, figsize=(17, 10))
axs[0, 0].imshow(wordcloud_medications_with_endo, interpolation='bilinear')
axs[0, 0].axis("off")
axs[0, 0].set_title('médications, patientes avec endometriose', fontsize=20)
axs[0, 1].imshow(wordcloud_medications_without_endo, interpolation='bilinear')
axs[0, 1].axis("off")
axs[0, 1].set_title('médications, patientes sans endometriose', fontsize=20)

axs[1, 0].imshow(wordcloud_symptoms_with_endo, interpolation='bilinear')
axs[1, 0].axis("off")
axs[1, 0].set_title('symptômes, patientes avec endometriose', fontsize=20)
axs[1, 1].imshow(wordcloud_symptoms_without_endo, interpolation='bilinear')
axs[1, 1].axis("off")
axs[1, 1].set_title('symptômes, patientes sans endometriose', fontsize=20)

axs[2, 0].imshow(wordcloud_sicknesses_with_endo, interpolation='bilinear')
axs[2, 0].axis("off")
axs[2, 0].set_title('maladies, patientes avec endometriose', fontsize=20)
axs[2, 1].imshow(wordcloud_sicknesses_without_endo, interpolation='bilinear')
axs[2, 1].axis("off")
axs[2, 1].set_title('maladies, patientes sans endometriose', fontsize=20)
plt.subplots_adjust(wspace=0.05)
#plt.tight_layout()
plt.show()

# In[10]:


Counter(medications_patients_with_endometriosis)
Counter(medications_patients_without_endometriosis)
Counter(symptoms_patients_with_endometriosis)
Counter(symptoms_patients_without_endometriosis)
Counter(sicknesses_patients_with_endometriosis)
Counter(sicknesses_patients_without_endometriosis)

# In[16]:


Counter(sicknesses_patients_without_endometriosis)

# In[10]:


### Show barplot with most common words
nmostfrequent = 50
typelabel = 'Tous'

if typelabel == 'Médicaments':
    with_endo = medications_patients_with_endometriosis
    without_endo = medications_patients_without_endometriosis
if typelabel == 'Symptômes':
    with_endo = symptoms_patients_with_endometriosis
    without_endo = symptoms_patients_without_endometriosis
elif typelabel == 'Maladies':
    with_endo = sicknesses_patients_with_endometriosis
    without_endo = sicknesses_patients_without_endometriosis
elif typelabel == 'Tous':
    with_endo = terms_patients_with_endometriosis
    without_endo = terms_patients_without_endometriosis

freq_medications_with_endo = Counter(with_endo)
df_word_freq_with_endo  = pd.DataFrame(freq_medications_with_endo.items(), columns=['word', 'frequency']).sort_values(by='frequency', ascending=False).head(nmostfrequent)

freq_without_endo = Counter(without_endo)
df_word_freq_without_endo  = pd.DataFrame(freq_without_endo.items(), columns=['word', 'frequency']).sort_values(by='frequency', ascending=False).head(nmostfrequent)

# Plot the horizontal barplot
fig, axs = plt.subplots(1, 2, figsize=(10, 7))
axs = axs.flatten()
sns.barplot(y='word', x='frequency', data=df_word_freq_with_endo, palette='viridis', ax = axs[0])
axs[0].set_title(f'{typelabel}, patientes avec endométriose', fontsize=15)
axs[0].set_xlabel('fréquence', fontsize=12)
axs[0].set_ylabel('mot', fontsize=12)
sns.barplot(y='word', x='frequency', data=df_word_freq_without_endo, palette='viridis', ax = axs[1])
axs[1].set_title(f'{typelabel}, patientes sans endométriose', fontsize=15)
axs[1].set_xlabel('fréquence', fontsize=12)
axs[1].set_ylabel('mot', fontsize=12)
plt.tight_layout()
plt.show()

# In[11]:


### Terms specific to endometriosis, by type

medications_patients_only_endometriosis = [term for term in medications_patients_with_endometriosis if term in list(set(medications_patients_with_endometriosis) - set(medications_patients_without_endometriosis))]
symptoms_patients_only_endometriosis = [term for term in symptoms_patients_with_endometriosis if term in list(set(symptoms_patients_with_endometriosis) - set(symptoms_patients_without_endometriosis))]
sicknesses_patients_only_endometriosis = [term for term in sicknesses_patients_with_endometriosis if term in list(set(sicknesses_patients_with_endometriosis) - set(sicknesses_patients_without_endometriosis))]

wordcloud_medications_patients_only_endometriosis = WordCloud(width = 1000, height = 500, background_color='white').generate_from_frequencies(Counter(medications_patients_only_endometriosis))
wordcloud_symptoms_patients_only_endometriosis = WordCloud(width = 1000, height = 500, background_color='white').generate_from_frequencies(Counter(symptoms_patients_only_endometriosis))
wordcloud_sicknesses_patients_only_endometriosis = WordCloud(width = 1000, height = 500, background_color='white').generate_from_frequencies(Counter(sicknesses_patients_only_endometriosis))

fig, axs = plt.subplots(1, 3, figsize=(15, 5))
axs = axs.flatten()
axs[0].imshow(wordcloud_medications_patients_only_endometriosis, interpolation='bilinear')
axs[0].axis("off")
axs[0].set_title('médications spécifiques à l\'endométriose', fontsize=15)
axs[1].imshow(wordcloud_symptoms_patients_only_endometriosis, interpolation='bilinear')
axs[1].axis("off")
axs[1].set_title('symptômes spécifiques à l\'endométriose', fontsize=15)
axs[2].imshow(wordcloud_sicknesses_patients_only_endometriosis, interpolation='bilinear')
axs[2].axis("off")
axs[2].set_title('maladies spécifiques à l\'endométriose', fontsize=15)

plt.tight_layout()
plt.show()

# In[17]:




# In[23]:


### Show barplot with most common words
nmostfrequent = 10

patients_with_endo = list(pd.unique(df_entities_unique.loc[(df_entities_unique['endometriose']=='present'), 'patient']))
npatients_with_endo = len(patients_with_endo)

freq_medications_only_endo = Counter(medications_patients_only_endometriosis)
df_freq_medications_only_endo  = pd.DataFrame(freq_medications_only_endo.items(), columns=['word', 'frequency']).sort_values(by='frequency', ascending=False).head(nmostfrequent)
df_freq_medications_only_endo['percentage'] = df_freq_medications_only_endo['frequency'] / npatients_with_endo

freq_symptoms_only_endo = Counter(symptoms_patients_only_endometriosis)
df_freq_symptoms_only_endo  = pd.DataFrame(freq_symptoms_only_endo.items(), columns=['word', 'frequency']).sort_values(by='frequency', ascending=False).head(nmostfrequent)
df_freq_symptoms_only_endo['percentage'] = df_freq_symptoms_only_endo['frequency'] / npatients_with_endo

freq_sicknesses_only_endo = Counter(sicknesses_patients_only_endometriosis)
df_freq_sicknesses_only_endo  = pd.DataFrame(freq_sicknesses_only_endo.items(), columns=['word', 'frequency']).sort_values(by='frequency', ascending=False).head(nmostfrequent)
df_freq_sicknesses_only_endo['percentage'] = df_freq_sicknesses_only_endo['frequency'] / npatients_with_endo

# Choose if to plot absolute number of patients or percentage
plot_type = 'absolute'

# Plot the horizontal barplot
fig, axs = plt.subplots(1, 3, figsize=(17, 5))
axs = axs.flatten()
if plot_type == 'absolute':
    sns.barplot(y='frequency', x='word', data=df_freq_medications_only_endo, ax = axs[0])
    axs[0].set_ylabel(f'fréquence sur {npatients_with_endo} patientes', fontsize=12)
elif plot_type == 'percentage':
    sns.barplot(y='percentage', x='word', data=df_freq_medications_only_endo, ax = axs[0])
    axs[0].set_ylabel('percentage des patientes', fontsize=12)
axs[0].set_title(f'Médications spécifiques endométriose', fontsize=15)

axs[0].set_xlabel('', fontsize=12)
axs[0].yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
for tick in axs[0].get_xticklabels():
    tick.set_rotation(60)
if plot_type == 'absolute':
    sns.barplot(y='frequency', x='word', data=df_freq_symptoms_only_endo, ax = axs[1])
    axs[1].set_ylabel(f'fréquence sur {npatients_with_endo} patientes', fontsize=12)
elif plot_type == 'percentage':
    sns.barplot(y='percentage', x='word', data=df_freq_medications_only_endo, ax = axs[1])
    axs[1].set_ylabel('percentage des patientes', fontsize=12)
axs[1].set_title(f'Symptômes spécifiques endométriose', fontsize=15)
axs[1].set_xlabel('', fontsize=12)
axs[1].yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
for tick in axs[1].get_xticklabels():
    tick.set_rotation(60)
if plot_type == 'absolute':
    sns.barplot(y='frequency', x='word', data=df_freq_sicknesses_only_endo, ax = axs[2])
    axs[2].set_ylabel(f'fréquence sur {npatients_with_endo} patientes', fontsize=12)
elif plot_type == 'percentage':
    sns.barplot(y='percentage', x='word', data=df_freq_medications_only_endo, ax = axs[2])
    axs[2].set_ylabel('percentage des patientes', fontsize=12)
axs[2].set_title(f'Maladies spécifiques endométriose', fontsize=15)
axs[2].set_xlabel('', fontsize=12)
axs[2].yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
for tick in axs[2].get_xticklabels():
    tick.set_rotation(60)
plt.tight_layout()
plt.show()

# In[24]:


'spasfon' in medications_patients_only_endometriosis

# In[42]:


npatients_with_endo = len(pd.unique(data.loc[data['endometriose']=='present', 'patient']))
npatients_without_endo = len(pd.unique(data.loc[data['endometriose']=='absent', 'patient']))
npatients_without_endo

# In[43]:


npatients_with_endo

# In[44]:


### Compare frequency of termes in patients with vs. without endometriosis

'''
# Select only terms with at least a frequency of this number
cutofffreq = 10
'''

freq_with_endo = Counter(terms_patients_with_endometriosis)
df_word_freq_with_endo  = pd.DataFrame(freq_with_endo.items(), columns=['word', 'absolute_frequency']).sort_values(by='absolute_frequency', ascending=False)
#df_word_freq_with_endo = df_word_freq_with_endo.loc[df_word_freq_with_endo['absolute_frequency'] >= cutofffreq]

freq_without_endo = Counter(terms_patients_without_endometriosis)
df_word_freq_without_endo  = pd.DataFrame(freq_without_endo.items(), columns=['word', 'absolute_frequency']).sort_values(by='absolute_frequency', ascending=False)
#df_word_freq_without_endo = df_word_freq_without_endo.loc[df_word_freq_without_endo['absolute_frequency'] >= cutofffreq]

### Compute term frequency relative to the corpus

absolute_frequencies_with_endo = df_word_freq_with_endo['absolute_frequency']
nwords_total_with_endometriosis = sum(data.loc[data['endometriose']=='present', 'text'].str.count(' ') + 1)
relative_frequencies_with_endo = [(freq/nwords_total_with_endometriosis) for freq in absolute_frequencies_with_endo]
relative_frequencies_with_endo = [(freq/npatients_with_endo) for freq in absolute_frequencies_with_endo]
df_word_freq_with_endo['relative_frequency'] = relative_frequencies_with_endo

absolute_frequencies_without_endo = df_word_freq_without_endo['absolute_frequency']
nwords_total_without_endometriosis = sum(data.loc[data['endometriose']=='absent', 'text'].str.count(' ') + 1)
relative_frequencies_without_endo = [(freq/nwords_total_without_endometriosis) for freq in absolute_frequencies_without_endo]
relative_frequencies_without_endo = [(freq/npatients_without_endo) for freq in absolute_frequencies_without_endo]
df_word_freq_without_endo['relative_frequency'] = relative_frequencies_without_endo

# Combine the dataframes
all_words = pd.concat([df_word_freq_with_endo['word'], df_word_freq_without_endo['word']]).unique()
df_freq_with_vs_without_endo = pd.DataFrame({'word': all_words})

# Merge frequencies from both dataframes
df_freq_with_vs_without_endo = df_freq_with_vs_without_endo.merge(df_word_freq_with_endo, on='word', how='left', suffixes=('', '_with_endo'))
df_freq_with_vs_without_endo = df_freq_with_vs_without_endo.merge(df_word_freq_without_endo, on='word', how='left', suffixes=('_with_endo', '_without_endo'))

df_freq_with_vs_without_endo.fillna(0, inplace=True)

# Sort by total frequency for better visualization
df_freq_with_vs_without_endo['total_frequency'] = df_freq_with_vs_without_endo['absolute_frequency_with_endo'] + df_freq_with_vs_without_endo['absolute_frequency_without_endo']
df_freq_with_vs_without_endo['total_frequency'] = df_freq_with_vs_without_endo['total_frequency'].astype(int)
df_freq_with_vs_without_endo.sort_values('total_frequency', ascending=False, inplace=True)

# Select only terms with at least a frequency of this number
cutofffreq = 10
df_freq_with_vs_without_endo = df_freq_with_vs_without_endo.loc[(df_freq_with_vs_without_endo.absolute_frequency_with_endo >= cutofffreq) | (df_freq_with_vs_without_endo.absolute_frequency_without_endo >= cutofffreq)]

df_freq_with_vs_without_endo

# In[45]:


### Show frequency of words in patients with vs. without endometriosis

frequency = 'relative'      # 'absolute' or 'relative'

# Plotting
num_words = df_freq_with_vs_without_endo.shape[0]
ncols = 6
nrows = int(np.ceil(num_words/ncols))
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(17, 8))
axes = axes.flatten()
for i, idx in enumerate(df_freq_with_vs_without_endo.index):
    axes[i].bar(['avec endo', 'sans endo'], [df_freq_with_vs_without_endo.loc[idx, f'{frequency}_frequency_with_endo'], df_freq_with_vs_without_endo.loc[idx, f'{frequency}_frequency_without_endo']], color=['blue', 'orange'])
    axes[i].set_title(df_freq_with_vs_without_endo.loc[idx, 'word'])
    if frequency == 'absolute':
        axes[i].set_ylabel('Fréquence absolue')
    else:
        axes[i].set_ylabel('Fréquence relative')
    axes[i].yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
for i in range(i+1, nrows*ncols):
    fig.delaxes(axes[i])

plt.tight_layout()
plt.show()

# In[75]:


df = df_entities_unique.copy()
features = ['atcd.endo', 'irm.lusg', 'tv.douloureux', 'irm.externe', 
                  'sf.dig.diarrhee', 'echo.lusg', 'echo.lusd', 'ef.hormone.dpc', 'effet.hormone']

# Set the number of top terms to plot
N = 20
normalize = False

# Count the occurrences of each term
term_counts = df['term'].value_counts()

# Select the top N most frequent terms
top_terms = term_counts.head(N).index

# Filter the dataframe to include only the top N terms
df_top_terms = df[df['term'].isin(top_terms)]

# List of binary columns to analyze
binary_columns = ['atcd.endo', 'irm.lusg', 'tv.douloureux', 'irm.externe', 
                  'sf.dig.diarrhee', 'echo.lusg', 'echo.lusd', 'ef.hormone.dpc', 'effet.hormone']

# Create subplots
ncols = 3
nrows = int(np.ceil(len(features)/ncols))
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(18, 12))
axes = axes.flatten()
# Iterate over binary columns and plot the presence vs absence for each
for ax, feature in zip(axes, features):
    feature_terms = df_top_terms.groupby(['term', feature]).size().unstack(fill_value=0)
    if normalize:
        npatientspresent = df.drop_duplicates(subset=['patient']).value_counts([feature])['present']
        npatientsabsent = df.drop_duplicates(subset=['patient']).value_counts([feature])['absent']
        feature_terms['absent'] = feature_terms['absent'] / npatientspresent
        feature_terms['present'] = feature_terms['present'] / npatientsabsent
    feature_terms.plot(kind='bar', stacked=True, ax=ax)
    ax.set_title(feature)
    ax.set_xlabel("")
    if normalize:
        ax.set_ylabel("fréquence normalisée")
    else:
        ax.set_ylabel("fréquence")
    ax.legend(title=feature, labels=['Absent', 'Present'])

plt.tight_layout()
plt.show()

# In[68]:


feature_terms['absent'] / nabsent

# In[76]:


feature = 'effet.hormone'
df.drop_duplicates(subset=['patient']).value_counts([feature])
