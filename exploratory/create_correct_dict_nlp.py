#!/usr/bin/env python
# coding: utf-8

# # create_correct_dict_nlp
# 
# Create a dictionnary for correcting text data in the patient files using spellchecker
# 
# Author: Maxime Mock

# In[2]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm
import re

import nltk
nltk.download('stopwords')
from spellchecker import SpellChecker

from exploratory.preprocessing.preprocess_NLP import load_dict_correction, tokenize_words
from exploratory.preprocessing.preprocess_NLP import *

# In[3]:


df_nlp = pd.read_csv('Data/DATA_PROCESSED/donnees_entree_nlp_sans_endo.csv', usecols=['Anonymisation', 'Date', 'Nature', 'Résumé'])
dict_correction = load_dict_correction()
dict_correction_extended = dict_correction.copy()

# In[4]:


all_words = df_nlp.Résumé.apply(tokenize_words).explode().sort_values().drop_duplicates().reset_index(drop=True)
all_words.to_csv('Data/DATA_PROCESSED/all_words.csv', index=False)

# In[7]:


spell = SpellChecker(language='fr')
for idx in tqdm(df_nlp.index):
    text_uncorrected = df_nlp.loc[idx, 'Résumé']
    words_uncorrected = re.findall(r"[\w']+", text_uncorrected)
    for word_uncorrected in words_uncorrected:
        word_corrected = spell.correction(word_uncorrected)
        if (word_corrected is not None) and (word_corrected != word_uncorrected) and (len(word_corrected) > 3) and (len(word_uncorrected) > 3) and ('\'' not in word_uncorrected):
            if word_uncorrected not in dict_correction_extended.keys():
                print(f'Adding corrected word {word_corrected} for original word: {word_uncorrected}')
                dict_correction_extended[word_uncorrected] = word_corrected

# In[8]:


dict_correction

# ## Check for abbreviations

# In[9]:


#df_nlp = pd.read_csv('./../../Data/Generate/donnees_entree_nlp_sans_endo.csv', usecols=['Anonymisation', 'Date', 'Nature', 'Résumé'])
df_nlp = pd.read_csv('Data/DATA_PROCESSED/Donnees_avec_endo_concat.csv', usecols=['Anonymisation', 'Résumé'])
df_nlp.Résumé = df_nlp.Résumé.apply(remove_special_characters)
df_nlp.Résumé = df_nlp.Résumé.apply(lowercase_text)
df_nlp.Résumé = df_nlp.Résumé.apply(correction_series)
df_nlp.head()

# In[10]:


all_words = df_nlp.Résumé.apply(tokenize_words).explode().sort_values().drop_duplicates().reset_index(drop=True)
all_words = all_words[all_words.str.isalpha()]
all_words = all_words.reindex(all_words.str.len().sort_values().index)
pd.set_option('display.max_rows', 500)
all_words.loc[all_words.apply(lambda x: len(x) > 2)].head(100)

# In[11]:


# Make a list of all short words (= potential abbreviations) that have more than one occurence
potential_abbreviations = all_words.loc[all_words.apply(lambda x: 1 < len(x) < 5)]
abbreviations = []
for abb in potential_abbreviations:
    noccurences = 0
    for phrase in df_nlp.Résumé:
        if abb in phrase.split(' '):
            noccurences += 1
    if noccurences > 2:
        abbreviations = abbreviations + [abb]
abbreviations

# In[12]:


# Show the phrases containing the word
word_of_interest = 'dl'
for phrase in df_nlp.Résumé:
    if word_of_interest in phrase.split(' '):
        print(phrase)

# In[13]:


# Give out all the texts containing a word, and the patient ID
word_of_interest = 'métrorragies'
df_texts_with_word = df_nlp[df_nlp['Résumé'].str.contains(word_of_interest, case=False, na=False)]
print(list(df_texts_with_word.Anonymisation))

# In[14]:


df_nlp.loc[df_nlp.Anonymisation=='AM-038', 'Résumé'].values[0]

# In[15]:


for idx in df_nlp.loc[df_nlp.Anonymisation=='AA-071'].index:
    print(df_nlp.loc[idx, 'Résumé'])
