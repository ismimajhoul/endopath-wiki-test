#!/usr/bin/env python
# coding: utf-8

# ## Scan given patient for symptoms/tests/medications
# This is to speed up the manual inspection of patient files for extraction of features in file 'infos_dossier_gyneco'.
# This manual creation of information was done to compare NLP performance to ML performance using these informations (that NLP is supposed to extract)

# In[1]:


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

# In[2]:


df_nlp = pd.read_csv('/home/nounou/endopath/Data/DATA_PROCESSED/Donnees_avec_endo_concat.csv', usecols=['Anonymisation', 'Résumé'])
df_nlp.Résumé = df_nlp.Résumé.apply(remove_special_characters)
df_nlp.Résumé = df_nlp.Résumé.apply(lowercase_text)
df_nlp.Résumé = df_nlp.Résumé.apply(correction_series)
df_nlp.head()

# In[3]:


### Helper functions to print out relevants bits from the text for each feature/column

from fuzzywuzzy import fuzz    # library that helps finding matching words that might be misspelled or so
import re

def extract_context(text, keywords, nwords_around=10, threshold=80):
    # Split the text into words
    words = re.findall(r'\b\w+\b', text)
    
    # Initialize an empty list to store the results
    results = []
    
    # Iterate over the words with their positions
    for i, word in enumerate(words):
        # Check fuzzy match against each keyword
        for keyword in keywords:
            if fuzz.ratio(word.lower(), keyword) >= threshold:
                # Extract words before and after the matched word
                start_index = max(i - nwords_around, 0)
                end_index = i + nwords_around + 1
                context = words[start_index:end_index]
                results.append(' '.join(context))
                break  # Stop checking other keywords if one is matched
    
    return results


def extract_context_prefix(text, prefix, nwords_around=5):
    # Split the text into words
    words = re.findall(r'\b\w+\b', text)
    
    # Initialize an empty list to store the results
    results = []
    
    # Iterate over the words with their positions
    for i, word in enumerate(words):
        # Check if the word starts with the given prefix (default: 'dys')
        if word.lower().startswith(prefix):
            # Extract the window of words before and after the matched word
            start_index = max(i - nwords_around, 0)
            end_index = i + nwords_around + 1
            context = words[start_index:end_index]
            results.append(' '.join(context))
    
    return results

import re


def extract_context_abbreviation(text, abbreviation, nwords_around=3):
    # Split the text into words
    words = re.findall(r'\b\w+\b', text)
    
    # Initialize an empty list to store the results
    results = []
    
    # Iterate over the words with their positions
    for i, word in enumerate(words):
        # Check if the word matches the abbreviation (case-insensitive)
        if word.lower() == abbreviation.lower():
            # Extract words before and after the matched word
            start_index = max(i - nwords_around, 0)
            end_index = i + nwords_around + 1
            context = words[start_index:end_index]
            results.append(' '.join(context))
    
    return results

# In[ ]:


patient = 'AC-131'
text_patient = df_nlp.loc[df_nlp.Anonymisation==patient, 'Résumé'].values[0]

### Scan the text and print out any piece of text that could be relevant for the symptom in question

print(f'##################### Patient: {patient} #####################\n')

print('############ Douleurs: ############')
keywords=["douleur", "douloureux"]
if len(extract_context(text_patient, keywords=keywords)) == 0: print('/')
else:
    for phrase in extract_context(text_patient, keywords=keywords): print(phrase)
    print()

print('############ Dysménorrhées: ############')
keywords=["dysménorrhées", "dysme", "menhorr", "menorr", "règles"]
if len(extract_context(text_patient, keywords)) == 0: print('/')
else:
    for phrase in extract_context(text_patient, keywords): print(phrase)
    print()

print('############ Douleur exonération de selles: ############')
keywords=["exoneration", "selles", "défécation", "defec", "défec"]
if len(extract_context(text_patient, keywords=keywords)) == 0: print('/')
else:
    for phrase in extract_context(text_patient, keywords=keywords): print(phrase)
    print()

print('############ Toucher vaginal: ############')
if len(extract_context(text_patient, keywords=["vaginal", "toucher", "tv"])) == 0: print('/')
else:
    for phrase in extract_context(text_patient, keywords=["vaginal", "toucher"]): print(phrase)
    print()

print('############ douleurs mictionnelles/ dysurie: ############')
if len(extract_context(text_patient, keywords=["miction", "dysurie"])) == 0: print('/')
else:
    for phrase in extract_context(text_patient, keywords=["miction", "dysurie"]): print(phrase)
    print()

print('############ rectorragie: ############')
if len(extract_context(text_patient, keywords=["rectorragie", "rectorr"])) == 0: print('/')
else:
    for phrase in extract_context(text_patient, keywords=["rectorragie", "rectorr"]): print(phrase)
    print()

print('############ Douleurs rapports intimes / dyspareunies: ############')
keywords=["rapport", "sexe", "sexuel", "intime", "dyspareunies", "dyspareunie", "coital", "coitales"]
if len(extract_context(text_patient, keywords=keywords)) == 0: print('/')
else:
    for phrase in extract_context(text_patient, keywords=keywords, nwords_around=20): print(phrase)
    print()

print('############ Spotting: ############')
if len(extract_context(text_patient, keywords=["spotting", "spottings"])) == 0: print('/')
else:
    for phrase in extract_context(text_patient, keywords=["spotting", "spottings"]): print(phrase)
    print()

print('############ Amenorrhées: ############')
if len(extract_context(text_patient, keywords=["amenorrhées"])) == 0: print('/')
else:
    for phrase in extract_context(text_patient, keywords=["amenorrhées"]): print(phrase)
    print()

print('############ Ménorragies: ############')
if len(extract_context(text_patient, keywords=["ménorragies", "ménorragie", "ménorr", "menorr"])) == 0: print('/')
else:
    for phrase in extract_context(text_patient, keywords=["ménorragies", "ménorragie", "ménorr", "menorr"]): print(phrase)
    print()

print('############ Métrorragies: ############')
if len(extract_context(text_patient, keywords=["métrorragies"])) == 0: print('/')
else:
    for phrase in extract_context(text_patient, keywords=["métrorragies"]): print(phrase)
    print()

print('############ Hyperménorrhées: ############')
if len(extract_context(text_patient, keywords=["hyperménorrhées", "hyperménorrhée", "hypermenorrhees", "hypermeno"])) == 0: print('/')
else:
    for phrase in extract_context(text_patient, keywords=["hyperménorrhées", "hyperménorrhée", "hypermenorrhees", "hypermeno"]): print(phrase)
    print()

print('############ Mictions excessives: ############')
if len(extract_context(text_patient, keywords=["miction", "excessives"])) == 0: print('/')
else:
    for phrase in extract_context(text_patient, keywords=["miction", "excessives"]): print(phrase)
    print()

print('############ Spasmes abdominales: ############')
if len(extract_context(text_patient, keywords=["spasmes", "abdominales", "cramp", "crampes"])) == 0: print('/')
else:
    for phrase in extract_context(text_patient, keywords=["spasmes", "abdominales"]): print(phrase)
    print()

print('############ pollakiurie: ############')
if len(extract_context(text_patient, keywords=["pollakiurie", "pollakiuries", "uriner"])) == 0: print('/')
else:
    for phrase in extract_context(text_patient, keywords=["pollakiurie", "pollakiuries", "uriner"]): print(phrase)
    print()

print('############ faux besoins: ############')
if len(extract_context(text_patient, keywords=["faux"])) == 0: print('/')
else:
    for phrase in extract_context(text_patient, keywords=["faux"]): print(phrase)
    print()

print('############ infection urinaire: ############')
if len(extract_context(text_patient, keywords=["infection", "urinaire"])) == 0: print('/')
else:
    for phrase in extract_context(text_patient, keywords=["infection", "urinaire"]): print(phrase)
    print()

print('############ perte appétit: ############')
if len(extract_context(text_patient, keywords=["appetit", "appétit", "mange"])) == 0: print('/')
else:
    for phrase in extract_context(text_patient, keywords=["appetit", "appétit", "mange"]): print(phrase)
    print()

print('############ nausées: ############')
if len(extract_context(text_patient, keywords=["nausées", "nausée"])) == 0: print('/')
else:
    for phrase in extract_context(text_patient, keywords=["nausées", "nausée"]): print(phrase)
    print()

print('############ vomissements: ############')
keywords=["vomissements", "vomit", "vomir"]
if len(extract_context(text_patient, keywords=keywords)) == 0: print('/')
else:
    for phrase in extract_context(text_patient, keywords=keywords): print(phrase)
    print()

print('############ dyspnée: ############')
if len(extract_context(text_patient, keywords=["dyspnée"])) == 0: print('/')
else:
    for phrase in extract_context(text_patient, keywords=["dyspnée"]): print(phrase)
    print()

print('############ constipation / troubles transit : ############')
keywords = ["constipation", "constipé", "transit"]
if len(extract_context(text_patient, keywords=keywords)) == 0: print('/')
else:
    for phrase in extract_context(text_patient, keywords=keywords): print(phrase)
    print()

print('############ diarrhées / diarrhées cataméniales : ############')
keywords = ["diarrhées", "cataméniales"]
if len(extract_context(text_patient, keywords=keywords)) == 0: print('/')
else:
    for phrase in extract_context(text_patient, keywords=keywords): print(phrase)
    print()

print('############ incontinence : ############')
keywords = ["incontinence", "incontinent"]
if len(extract_context(text_patient, keywords=keywords)) == 0: print('/')
else:
    for phrase in extract_context(text_patient, keywords=keywords): print(phrase)
    print()

print('############ hyperthermie : ############')
keywords = ["hyperthermie", "therm"]
if len(extract_context(text_patient, keywords=keywords)) == 0: print('/')
else:
    for phrase in extract_context(text_patient, keywords=keywords): print(phrase)
    print()

print('############ vessie spastique : ############')
keywords = ["vessie", "spastique"]
if len(extract_context(text_patient, keywords=keywords)) == 0: print('/')
else:
    for phrase in extract_context(text_patient, keywords=keywords): print(phrase)
    print()

print('############ Désir/essai de grossesse : ############')
keywords = ["désir", "essai", "grossesse", "enceinte"]
if len(extract_context(text_patient, keywords=keywords)) == 0: print('/')
else:
    for phrase in extract_context(text_patient, keywords=keywords): print(phrase)
    print()

print('############ infertilité : ############')
keywords = ["infertilité", "infertile", "infert"]
if len(extract_context(text_patient, keywords=keywords)) == 0: print('/')
else:
    for phrase in extract_context(text_patient, keywords=keywords): print(phrase)
    print()

print('############ PMA / FIV : ############')
abbreviation = "PMA"
if len(extract_context_abbreviation(text_patient, abbreviation=abbreviation)) == 0: print('/')
else:
    for phrase in extract_context_abbreviation(text_patient, abbreviation=abbreviation, nwords_around = 20): print(phrase)
    print()
abbreviation = "FIV"
if len(extract_context_abbreviation(text_patient, abbreviation=abbreviation)) == 0: print('/')
else:
    for phrase in extract_context_abbreviation(text_patient, abbreviation=abbreviation, nwords_around = 20): print(phrase)
    print()
keywords = ["fécondation", "vitro"]
if len(extract_context(text_patient, keywords=keywords)) == 0: print('/')
else:
    for phrase in extract_context(text_patient, keywords=keywords): print(phrase)
    print()

print('############ fausses couches : ############')
keywords = ["fausses", "couches"]
if len(extract_context(text_patient, keywords=keywords)) == 0: print('/')
else:
    for phrase in extract_context(text_patient, keywords=keywords): print(phrase)
    print()

print('############ Antécédent d''endométriose : ############')
keywords = ["endométriose", "endo", "antécédent", "ant", "atcd"]
if len(extract_context(text_patient, keywords=keywords)) == 0: print('/')
else:
    for phrase in extract_context(text_patient, keywords=keywords, nwords_around = 20): print(phrase)
    print()

print('############ Antécédent de chirurgie d''endométriose : ############')
keywords = ["chirurgie", "chirurgies", "chir", "opération", "opéré", "ablation", "Coelioscopie", "coelio", "coelio", "adhesiolyse", "adhes", "hysteroscopie", "exérèse", "ligament", "lus", "gauche", "droite", "g", "endométriose", "endo"]
if len(extract_context(text_patient, keywords=keywords)) == 0: print('/')
else:
    for phrase in extract_context(text_patient, keywords=keywords, nwords_around = 20): print(phrase)
    print()

print('############ Nodule : ############')
keywords = ["nodule"]
if len(extract_context(text_patient, keywords=keywords)) == 0: print('/')
else:
    for phrase in extract_context(text_patient, keywords=keywords): print(phrase)
    print()

print('############ Epaississement : ############')
keywords = ["épaississement", "épais"]
if len(extract_context(text_patient, keywords=keywords)) == 0: print('/')
else:
    for phrase in extract_context(text_patient, keywords=keywords): print(phrase)
    print()

print('############ Chirurgie bariatrique : ############')
keywords = ["bariatrique"]
if len(extract_context(text_patient, keywords=keywords)) == 0: print('/')
else:
    for phrase in extract_context(text_patient, keywords=keywords, nwords_around = 20): print(phrase)
    print()

print('############ résection : ############')
keywords = ["résection"]
if len(extract_context(text_patient, keywords=keywords)) == 0: print('/')
else:
    for phrase in extract_context(text_patient, keywords=keywords): print(phrase)
    print()

print('############ anapath : ############')
keywords = ["anapath"]
if len(extract_context(text_patient, keywords=keywords)) == 0: print('/')
else:
    for phrase in extract_context(text_patient, keywords=keywords): print(phrase)
    print()

print('############ frottis cervico vaginal : ############')
keywords = ["frottis"]
if len(extract_context(text_patient, keywords=keywords)) == 0: print('/')
else:
    for phrase in extract_context(text_patient, keywords=keywords, nwords_around=20): print(phrase)
    print()

print('############ échographie : ############')
keywords = ["échographie", "écho", "echo"]
if len(extract_context(text_patient, keywords=keywords)) == 0: print('/')
else:
    for phrase in extract_context(text_patient, keywords=keywords, nwords_around = 40): print(phrase)
    print()

print('############ IRM : ############')
abbreviation = "IRM"
if len(extract_context_abbreviation(text_patient, abbreviation=abbreviation, nwords_around=30)) == 0: print('/')
else:
    for phrase in extract_context_abbreviation(text_patient, abbreviation=abbreviation, nwords_around = 20): print(phrase)
    print()

print('############ rectosonographie : ############')
keywords = ["rectosonographie", "recto", "sonogr", "rsg"]
if len(extract_context(text_patient, keywords=keywords)) == 0: print('/')
else:
    for phrase in extract_context(text_patient, keywords=keywords, nwords_around = 30): print(phrase)
    print()

print('############ test hpv: ############')
keywords = ["hpv"]
if len(extract_context(text_patient, keywords=keywords)) == 0: print('/')
else:
    for phrase in extract_context(text_patient, keywords=keywords): print(phrase)
    print()

print('############ absentéisme scolaire/professionnel: ############')
keywords = ["absentéisme", "absentisme", "scolaire", "professionnel", "école", "travail"]
if len(extract_context(text_patient, keywords=keywords)) == 0: print('/')
else:
    for phrase in extract_context(text_patient, keywords=keywords): print(phrase)
    print()

### Print out all the medications appearing:

print('############ Medications: ############')

keywords = ["antalgique", "antiinflammatoire", "inflammatoire", "anti", "antispasmodique", "spasmodique", "hormonal", "hormon", "antadys", "dihydrocodéine",
            "paracétamol", "ibuprofène", "dafalgan", "efferalgan", "diclofénac", "kétoprofène", "celecoxib", "naproxène", "indométacine", "piroxicam", "flurbiprofène", "voltarène", "profenid", "celebrex", "advil", "apranax", "indocid", "feldène", "cebutid", "néfopam", "paralyoc",    # antalgiques palier 1
            "codéine", "codoliprane", "algisedal", "antalvic", "dicodin", "tramadol", "topalgic", "contramal", "zamudol", "ixprim", "zaldiar", "contramal", "topalgic",                                                                                                  # antalgiques palier 2
            "actiskenan", "sévrédol", "skénan", "moscontin", "kapanol", "sophidone", "oxycodone", "oxynorm", "oxycontin", "morphine", "hydromorphone", "oxycodone", "tapentadol", "buprénorphine", "fentanyl", "méthadone",         # antalgiques palier 3
            "lamaline", "izalgi", "ains", "biprofenid", "ponstyl", "nifluril", "acupan", "skénan", "utrogestan", "surgestone", "norlevo", "naproxène", "progestérone", "progestatif", "oestro",
            "toviaz", "spasfon", "vesicare", "meteospasmyl", "optilova", "optidril", "minidril", "mirena", "izéane", "optimizette", "décapeptyl", "duphaston", "estréva", "visanne", "lutenyl", "lutéran", "ditropan", "microval", "misolfa",
            "cp", "prendre"]
if len(extract_context(text_patient, keywords=keywords)) == 0: print('/')
else:
    for phrase in extract_context(text_patient, keywords=keywords): print(phrase)
    print()

from gliner import GLiNER
model = GLiNER.from_pretrained("almanach/camembert-bio-gliner-v0.1")
entities = model.predict_entities(text_patient, labels=["médicaments"], threshold=0.5)
for e in entities: print(e['text'])

# ## Compare results of manual extraction with receuil data

# In[52]:


recueil_imc  = pd.read_excel('/home/nounou/endopath/Data/DATA_RAW/Recueil (1).xlsx').drop('Unnamed: 90', axis=1)
recueil_imc = recueil_imc.rename(columns={"Numéro anonymat": "Anonymisation"})

infos_dossier_gyneco  = pd.read_excel('/home/nounou/endopath/Data/Generate/infos_dossiers_gyneco.xlsx')
infos_dossier_gyneco

# In[65]:


columns_to_compare = ['ttt.p', "traitement progestatif"] #['sf.dpc', 'douleurs_pelviennes'] # ['tv.douleur.lusg', "douleurs lus gauche"]
df = pd.merge(infos_dossier_gyneco, recueil_imc, on=['Anonymisation'])
try: 
    df[columns_to_compare[0]] = pd.array(df[columns_to_compare[0]], dtype=pd.Int64Dtype())
except:
    pass
try: 
    df[columns_to_compare[1]] = pd.array(df[columns_to_compare[1]], dtype=pd.Int64Dtype())
except:
    pass
pd.set_option('display.max_rows', df.shape[0]+10)
df[['Anonymisation'] + columns_to_compare]

# In[64]:


single_column = 'test_hpv'
df_single = pd.merge(infos_dossier_gyneco[['Anonymisation'] + [single_column]], recueil_imc, on=['Anonymisation'])[['Anonymisation']+[single_column]]
try: 
    df_single[single_column] = pd.array(df_single[single_column], dtype=pd.Int64Dtype())
except:
    pass
df_single

# In[8]:


list(df.columns)

# In[9]:


column_of_interest = 'sf.dpc'
df = pd.merge(df_nlp, recueil_imc, on=['Anonymisation'])
pd.set_option('display.max_rows', df.shape[0]+10)
df[['Anonymisation', column_of_interest]]
