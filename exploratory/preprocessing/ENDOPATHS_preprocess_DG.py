### ENDOPATHS_preprocess_DG
# Script for preprocessing of the 'dossier-gyneco' file. Puts it into the form that is used for all analysis and classifications.
# Not automatized so far, but specific to data available so far (state: July 2024)
# Author: Maxime Mock

# IMPORTS : 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import openpyxl
import msoffcrypto
import io
import re
import datetime
import itertools
import string 
from dateutil.parser import parse
from io import StringIO
from preprocess_dossier_gyneco import traitement_DF, traitement_des_separateurs_OR, df_sans_bool, extraction_des_données, preparation_des_dates_a_tronquer, troncature_des_datas
from extraction_endometriose import get_data_endo, endométriose
from src.utils.gestion_lexique import from_serie_to_list, remove_duplicat
pd.options.display.max_columns = 88
pd.options.display.max_rows = 250
pd.set_option('display.width', 1000)

print('Hello')

PASSWORD_1  =  pd.read_csv('Data/DATA_RAW/Nouveau dossier/PASSWORD_1.txt').columns[0]
decrypted = io.BytesIO()
with open("Data/DATA_RAW/dossier-gyneco-23-03-2022.xlsx", "rb") as f:
    file = msoffcrypto.OfficeFile(f)
    file.load_key(password=PASSWORD_1)  # Use password
    file.decrypt(decrypted)

dossier_gyneco = pd.read_excel(decrypted)
rows_gyneco, columns_gyneco = dossier_gyneco.shape
print(f'Le fichier dossier_gyneco, contient {rows_gyneco} lignes (patientes) et {columns_gyneco} colonnes (pathologies)')

### __*Ouverture du mapper pour anonymiser les IPP*__ :

mapper_IPP_Ano_df = pd.read_excel('Data/DATA_RAW/INCLUSION RECHERCHE CLINIQUE.xlsx', index_col='*IPP*')
mapper_Ano_IPP_df = pd.read_excel('Data/DATA_RAW/INCLUSION RECHERCHE CLINIQUE.xlsx', index_col='Numéro inclusion')
mapper_temp1 = mapper_IPP_Ano_df.to_dict()
mapper_temp2 = mapper_Ano_IPP_df.to_dict()
mapper_IPP_Ano = mapper_temp1['Numéro inclusion']
mapper_Ano_IPP = mapper_temp2['*IPP*']

### __*Traitement des célulles  avec les lignes CSV piégées*__ : 

copy = dossier_gyneco.copy()
dossier_gyneco = traitement_DF(dossier_gyneco)
print(dossier_gyneco.shape)


### On cherche les colonnes avec le séparateur | 
#### Sur le Df d'origine  => ## NBS: helper function potentially existent ?
coordonnees_df_copy = []
for elem in itertools.product(range(len(copy)), range(1687)):
    coordonnees_df_copy.append(elem)
liste_col_separator_str_copy = []
for coord in coordonnees_df_copy:
    if type(copy.iloc[coord]) == str and '|' in copy.iloc[coord]:
        liste_col_separator_str_copy.append(coord[1])
liste_col_separator_str_copy=list(set(liste_col_separator_str_copy))
liste_col_separator_str_copy.sort()
print("AAAAAAA",len(liste_col_separator_str_copy))

##

#### Sur le Df après traitement
coordonnees_df = []
for elem in itertools.product(range(2561), range(1687)): # NBS: not scalable 
    coordonnees_df.append(elem)

liste_col_separator_str = []
for coord in coordonnees_df:
    if type(dossier_gyneco.iloc[coord]) == str and '|' in dossier_gyneco.iloc[coord]:
        liste_col_separator_str.append(coord[1])
liste_col_separator_str=list(set(liste_col_separator_str))

liste_col_separator_str.sort()
len(liste_col_separator_str)
  
liste_sep_col_named = []
for elem in liste_col_separator_str:
    liste_colonnes = list(dossier_gyneco.columns)
    liste_sep_col_named.append(liste_colonnes[elem])

print('done')
print(len(liste_col_separator_str))

# TODO NBS: à comprendre
liste_liste_sep_x = []
liste_sep_1 = liste_col_separator_str[:4]
liste_liste_sep_x.append(liste_sep_1)
liste_sep_2 = liste_col_separator_str[4:11]
liste_liste_sep_x.append(liste_sep_2)
liste_sep_3 = liste_col_separator_str[11:19]
liste_liste_sep_x.append(liste_sep_3)
liste_sep_4 = liste_col_separator_str[19:21]
liste_liste_sep_x.append(liste_sep_4)
liste_sep_5 = liste_col_separator_str[21:23]
liste_liste_sep_x.append(liste_sep_5)
liste_sep_6 = liste_col_separator_str[23:29]
liste_liste_sep_x.append(liste_sep_6)
liste_sep_7 = liste_col_separator_str[30:34]
liste_liste_sep_x.append(liste_sep_7)
liste_sep_8 = liste_col_separator_str[34:38]
liste_liste_sep_x.append(liste_sep_8)
liste_sep_9 = liste_col_separator_str[38:53]
liste_liste_sep_x.append(liste_sep_9)
liste_sep_10 = liste_col_separator_str[53:56]
liste_liste_sep_x.append(liste_sep_10)
liste_sep_11 = liste_col_separator_str[57:61]
liste_liste_sep_x.append(liste_sep_11)
liste_sep_12 = liste_col_separator_str[61:64]
liste_liste_sep_x.append(liste_sep_12)
liste_sep_13 = liste_col_separator_str[64:72]
liste_liste_sep_x.append(liste_sep_13)
liste_sep_14 = liste_col_separator_str[72:]
liste_liste_sep_x.append(liste_sep_14)

# Grâce aux indices, on créée une liste avec les nom des colonnes, que l'on split au format du futur multi-index :

liste_liste_sep_x_name = []
for liste in liste_liste_sep_x:
    liste_temp = []
    for elem in liste:
        str_temp = list(dossier_gyneco.columns)[elem]
        str_temp = str_temp.replace('Gynécologie > ', '')  # NBS: same as line 157
        part_1, part_2 = str_temp.split('>')
        liste_temp.append((part_1, part_2))
        
    liste_liste_sep_x_name.append(liste_temp)
print('Done')

#### Recapitulatif des listes et automatisation du traitement :


# Anonymisation du DF : 
dossier_gyneco = dossier_gyneco.replace(mapper_IPP_Ano)

liste_colonnes = list(dossier_gyneco.columns)

# pattern = '[A-Z][A-Z][-][0-9][0-9][0-9]'

new_liste_colonnes = []
for str_ in liste_colonnes:
    new_liste_colonnes.append(str_.replace('Gynécologie > ', ''))
new_liste_colonnes[0]='Anonymisation'   

dossier_gyneco = dossier_gyneco.rename(columns = dict(zip(liste_colonnes, new_liste_colonnes)))
dossier_gyneco.head(2)

dossier_gyneco = dossier_gyneco.drop(['Sexe', '¤Age'], axis=1)
dossier_gyneco = dossier_gyneco.dropna(how='all',axis=1)
dossier_gyneco = dossier_gyneco.dropna(how='all',axis=0)

### __*Quelques observations en regardant certaines cellules/colonnes*__ :
# ID non utilisé pour le moment par le fichier dossier_gyneco
print('ID non utilisés : ')
for key, value in mapper_IPP_Ano.items():
    if value not in dossier_gyneco['Anonymisation'].dropna().unique():
        print(value, key)
print('Done')
print('DG.shape =', dossier_gyneco.shape)

## __Préprocessing__ : 

### __*Préparation d'un multiindex pour le dataframe*__ :

# On enregistre la colonne Anonymisation comme index :
dossier_gyneco = dossier_gyneco.set_index('Anonymisation')
index_anonym = dossier_gyneco.index

# On utilise la liste_colonnes pour générer les tuples qui serviront pour le multiindex : 
liste_colonnes_new_index = list(dossier_gyneco.columns)

liste_new_col = []
for col in liste_colonnes_new_index:
    if '>' in col:
        (index_1, index_2) = col.split('>')
        liste_new_col.append((index_1, index_2))
    else :
        liste_new_col.append(col)
        
# On transforme le dataframe avec le multiindex et l'index_anonym :        
multi_index = pd.MultiIndex.from_tuples(liste_new_col, names=['Nature', 'Acte']) 
dossier_gyneco.columns = multi_index
dossier_gyneco = dossier_gyneco.reset_index()

# On crée les variables des listes des colonnes pour la suite du notebook :
Nature = dossier_gyneco.columns.get_level_values(0)
Acte = dossier_gyneco.columns.get_level_values(1)
multi_index_final = list(dossier_gyneco.columns)

print('Traitement des séparateurs | ')
df_test_4 = traitement_des_separateurs_OR(dossier_gyneco, liste_liste_sep_x_name)
Multiindex = df_test_4.columns

df_sans_col_bool = df_sans_bool(df_test_4)
print('Avant :', df_test_4.shape)
print('Après :', df_sans_col_bool.shape)

to_drop = [
    ('Consultation', 'id_fiche_prog_op'),
    ('Consultation', 'Intervenant txt'),
    ('Consultation', 'Intervenant'),
    ('Consultation', 'Consultation d annonce'),
    ('Consultation', "Ordonnances / Id fiche"), # ID de l'ordonnance,
    ('Consultation', "Ordonnances / Resume short"), # qui redige l'ordonnance
    ('Consultation', "Ordonnances / Date ordo"),
    ('Consultation', 'Ordonnances / Type'),
    ('Ordonnance', 'Consigne ordonnance'),
    ('Ordonnance', 'Prescripteur'), # Dr
    ('Ordonnance', 'Prescripteur autre'), # autre Dr
    ('Ordonnance', 'Prescripteur id'), # Id du Dr
    ('Ordonnance', 'Type ordonnance'),
    ('Ordonnance', 'Autre type ordonnance'),
    ('Ordonnance', 'OAR (1)'),
    ('Ordonnance', 'QSP (1)'), 
    ('Ordonnance', 'Date taille'),
    ('Ordonnance', 'Date poids'),
    ('Mots', 'Inclusion mots / Intervenant id'), # ID du Dr
    ('Mots', 'Inclusion mots / Intervenant txt'), # Dr 
    ('Mots', 'Inclusion mots / Intervenant'),
    ('Mots', 'Inclusion mots / Type mot'),
    ('Mots', 'Inclusion mots / Createur id'),
    ('Mots', 'Inclusion mots affichage / Intervenant txt'),
    ('Mots', 'Inclusion mots affichage / Intervenant'),
    ('Mots', 'Inclusion mots affichage / Type mot'),
    ('Mots', 'Inclusion mots affichage / Fiche id'),
    ('Mots', 'Inclusion mots affichage / Createur id'),
    ('Programmation opératoire', 'Sp'),    
    ('Programmation opératoire', 'Tel'),
    ('Programmation opératoire', 'Type hospitalisation'),
    ('Programmation opératoire', 'Operateur'),
    ('Programmation opératoire', 'operateur_new_id'),
    ('Programmation opératoire', 'Opérateur txt'),
    ('Programmation opératoire', 'Operateur.1'),
    ('Programmation opératoire', 'operateur_new_2_id'),  
    ('Programmation opératoire', 'Opérateur autre'),     
    ('Programmation opératoire', 'Délégué'),
    ('Programmation opératoire', 'Supérieur inférieur 21j'),
    ('Programmation opératoire', 'Date prévue'),
    ('Programmation opératoire', 'Anesthesie'),
    ('Programmation opératoire', 'Anesthesie txt'),
    ('Programmation opératoire', 'Informee diag'),
    ('Programmation opératoire', 'Informee risque'),
    ('Programmation opératoire', 'Risque'),
    ('Programmation opératoire', 'Chambre implantable'),
    ('Programmation opératoire', 'Maison'),
    ('Programmation opératoire', 'Duree operatoire'),
    ('Programmation opératoire', 'Allergie autre txt'),
    ('CR opératoire (Fiche)', 'Durée opératoire'),
    ('CR opératoire (Fiche)', 'Type anesthésie'),
    ('CR opératoire (Fiche)', 'Autre type anesthésie'),
    ('CR opératoire (Fiche)', 'Geste ids'),
    ('CR opératoire (Fiche)', 'Compte textiles'),
    ('CR opératoire (Fiche)', 'Opérateurs / Opérateur'),
    ('CR opératoire (Fiche)', 'Opérateurs / Operateur.1'),
    ('CR opératoire (Fiche)', 'Aides / Aide.1'),
    ('CR opératoire (Fiche)', 'Aides / Aide.2'),
    ('CR opératoire (Fiche)', 'Ibodes / Ibode.1'),
    ('CR opératoire (Fiche)', 'Ibodes / Ibode.2'),
    ('CR opératoire (Fiche)', 'Anesthésistes / Anesthesiste.1'),
    ('CR opératoire (Fiche)', 'Anesthésistes / Anesthesiste.2'),
    ('CR opératoire (Fiche)', 'Cmb soignants / Soignant int'),
    ('CR opératoire (Fiche)', 'Cmb soignants / Soignant txt'),    
    ('Hospitalisation', 'Date entree'),
    ('Hospitalisation', 'Date sortie'),
    ('Hospitalisation', 'Inclusion mots / Date mot'),
    ('Hospitalisation', 'Inclusion mots / Intervenant id'),
    ('Hospitalisation', 'Inclusion mots / Intervenant txt'),
    ('Hospitalisation', 'Inclusion mots / Type mot'),
    ('Hospitalisation', 'Inclusion mots / Createur id'),
    ("Fiche d'hospitalisation ACHA", 'Id vue'),
    ("Fiche d'hospitalisation ACHA", 'Vue com dossier asur existant'),
    ("Fiche d'hospitalisation ACHA", 'Liste specialites'),
    ("Fiche d'hospitalisation ACHA", 'Declare mesure nomen'),
    ("Fiche d'hospitalisation ACHA", 'Declare mesure chumlea nomen'),
    ("Fiche d'hospitalisation ACHA", 'Date CRH'),
    ("Fiche d'hospitalisation ACHA", 'Date dernière génération'),
    ("Fiche d'hospitalisation ACHA", 'Date de génération du document de sortie'),
    ("Fiche d'hospitalisation ACHA",
    'Date de génération du document de sortie valant CRH'),
    ("Fiche d'hospitalisation ACHA", 'Date de génération du CRH'),
    ("Fiche d'hospitalisation ACHA", 'IdDocSortieCRH'),
    ("Fiche d'hospitalisation ACHA", 'Date entrée'),
    ("Fiche d'hospitalisation ACHA", 'Date sortie'),
    ("Fiche d'hospitalisation ACHA", 'Unité'),
    ("Fiche d'hospitalisation ACHA", 'Responsable'),
    ("Fiche d'hospitalisation ACHA", 'Responsable txt'),
    ("Fiche d'hospitalisation ACHA", 'Responsable.1'),
    ("Fiche d'hospitalisation ACHA", 'Medecin'),
    ("Fiche d'hospitalisation ACHA", 'Medecin txt'),
    ("Fiche d'hospitalisation ACHA", 'Medecin id'),
    ("Fiche d'hospitalisation ACHA", 'Date poids'),
    ("Fiche d'hospitalisation ACHA", 'Date taille'),
    ("Fiche d'hospitalisation ACHA", "Motif d'entrée"),
    ("Fiche d'hospitalisation ACHA", 'Histoire de la maladie'),
    ("Fiche d'hospitalisation ACHA", 'Conclusion examen clinique initial'),
    ("Fiche d'hospitalisation ACHA", 'Commentaire transfusion'),
    ("Fiche d'hospitalisation ACHA", 'Antécédent transfusion'),
    ("Fiche d'hospitalisation ACHA", 'Transfusion.1'),
    ("Fiche d'hospitalisation ACHA", 'Allergie'),
    ("Fiche d'hospitalisation ACHA", 'Commentaire allergie'),
    ("Fiche d'hospitalisation ACHA",
    'Actes techniques, biologiques ou imagerie marquants'),
    ("Fiche d'hospitalisation ACHA", "Pose d'un dispositif medico implantable"),
    ("Fiche d'hospitalisation ACHA",
    "Commentaire pose d'un dispositif medico implantablePose d'un dispositif medico implantable"),
    ("Fiche d'hospitalisation ACHA", 'Evenemements indesirables - complications'),
    ("Fiche d'hospitalisation ACHA", 'Autre evènement'),
    ("Fiche d'hospitalisation ACHA", 'Destination de sortie'),
    ("Fiche d'hospitalisation ACHA", 'Dest sortie txt'),
    ("Fiche d'hospitalisation ACHA", 'Commune traitement habituel'),
    ("Fiche d'hospitalisation ACHA", 'Continuité des soins'),
    ("Fiche d'hospitalisation ACHA", 'RDV à prendre par le patient'),
    ("Fiche d'hospitalisation ACHA", 'RDV pris pour le patient'),
    ("Fiche d'hospitalisation ACHA", 'Autre continuité'),
    ("Fiche d'hospitalisation ACHA", 'Consignes'),
    ("Fiche d'hospitalisation ACHA", 'Prise en charge sociale'),
    ("Fiche d'hospitalisation ACHA",
    'Conseil, recommandation ou surveillance particulière'),
    ("Fiche d'hospitalisation ACHA", 'UniqueId'),
    ("Fiche d'hospitalisation ACHA", 'Surface corporelle'),
    ("Fiche d'hospitalisation ACHA", 'Gen crh'),
    ("Fiche d'hospitalisation ACHA", 'Gen doc sortie val crh'),
    ("Fiche d'hospitalisation ACHA", 'Gen doc sortie'),
    ("Fiche d'hospitalisation ACHA", 'Patient porteur/contact de BMR ou BHRe.1'),
    ("Fiche d'hospitalisation ACHA", 'Specialite fiche intro / Titre'),
    ("Fiche d'hospitalisation ACHA", 'Specialite fiche contenu / Titre'),
    ("Fiche d'hospitalisation ACHA", 'Specialite fiche conclusion / Titre'),
    ("Fiche d'hospitalisation ACHA", 'Sejour selectionne / Id venue'),
    ("Fiche d'hospitalisation ACHA", 'Antecedent / Id'),
    ("Fiche d'hospitalisation ACHA", 'Antecedent / Type antecedent'),
    ("Fiche d'hospitalisation ACHA", 'Antecedent / Libelle antecedent sans cim'),
    ("Fiche d'hospitalisation ACHA", 'Incl historique ordonnance / Prescripteur'),
    ("Fiche d'hospitalisation ACHA",
    'Incl historique ordonnance / Date ordonnance'),
    ("Fiche d'hospitalisation ACHA", 'Evol et suivi / Date mot'),
    ("Fiche d'hospitalisation ACHA", 'Evol et suivi / Intervenant mot'),
    ("Fiche d'hospitalisation ACHA", 'Evol et suivi / Intervenant mot id'),
    ("Fiche d'hospitalisation ACHA", 'Evol et suivi / Intervenant mot txt'),
    ("Fiche d'hospitalisation ACHA", 'Evol et suivi croissant / Date mot'),
    ("Fiche d'hospitalisation ACHA", 'Evol et suivi croissant / Intervenant mot'),
    ("Fiche d'hospitalisation ACHA",
    'Evol et suivi croissant / Intervenant mot id'),
    ("Fiche d'hospitalisation ACHA",
    'Evol et suivi croissant / Intervenant mot txt'),
    ("Fiche d'hospitalisation ACHA", 'Liste rdv / Date'),
    ("Fiche d'hospitalisation ACHA", 'Liste rdv / Type rdv'),
    ("Fiche d'hospitalisation ACHA", "Liste rdv / Nom de l'agenda"),
    ("Fiche d'hospitalisation réglementaire", 'Id vue'),
    ("Fiche d'hospitalisation réglementaire", 'Vue com dossier asur existant'),
    ("Fiche d'hospitalisation réglementaire", 'Liste specialites'),
    ("Fiche d'hospitalisation réglementaire", 'Declare mesure nomen'),
    ("Fiche d'hospitalisation réglementaire", 'Declare mesure chumlea nomen'),
    ("Fiche d'hospitalisation réglementaire", 'Date CRH'),
    ("Fiche d'hospitalisation réglementaire", 'Date dernière génération'),
    ("Fiche d'hospitalisation réglementaire",
    'Date de génération du document de sortie'),
    ("Fiche d'hospitalisation réglementaire",
    'Date de génération du document de sortie valant CRH'),
    ("Fiche d'hospitalisation réglementaire", 'Date de génération du CRH'),
    ("Fiche d'hospitalisation réglementaire", 'IdDocSortieCRH'),
    ("Fiche d'hospitalisation réglementaire", 'Date entrée'),
    ("Fiche d'hospitalisation réglementaire", 'Date sortie'),
    ("Fiche d'hospitalisation réglementaire", 'Unité'),
    ("Fiche d'hospitalisation réglementaire", 'Responsable'),
    ("Fiche d'hospitalisation réglementaire", 'Responsable txt'),
    ("Fiche d'hospitalisation réglementaire", 'Responsable.1'),
    ("Fiche d'hospitalisation réglementaire", 'Medecin'),
    ("Fiche d'hospitalisation réglementaire", 'Medecin txt'),
    ("Fiche d'hospitalisation réglementaire", 'Medecin id'),
    ("Fiche d'hospitalisation réglementaire", 'Date poids'),
    ("Fiche d'hospitalisation réglementaire", 'Date taille'),
    ("Fiche d'hospitalisation réglementaire", 'Imc'),
    ("Fiche d'hospitalisation réglementaire",
    'Patient porteur/contact de BMR ou BHRe'),
    ("Fiche d'hospitalisation réglementaire", 'Transfusion'),
    ("Fiche d'hospitalisation réglementaire", 'Commentaire transfusion'),
    ("Fiche d'hospitalisation réglementaire", 'Antécédent transfusion'),
    ("Fiche d'hospitalisation réglementaire", 'Transfusion.1'),
    ("Fiche d'hospitalisation réglementaire", 'Allergie'),
    ("Fiche d'hospitalisation réglementaire", 'Commentaire allergie'),
    ("Fiche d'hospitalisation réglementaire",
    'Actes techniques, biologiques ou imagerie marquants'),
    ("Fiche d'hospitalisation réglementaire",
    "Pose d'un dispositif medico implantable"),
    ("Fiche d'hospitalisation réglementaire",
    'Evenemements indesirables - complications'),
    ("Fiche d'hospitalisation réglementaire",
    'Commentaire Evenemements indesirables - complications'),
    ("Fiche d'hospitalisation réglementaire", 'Autre evènement'),
    ("Fiche d'hospitalisation réglementaire", 'Destination de sortie'),
    ("Fiche d'hospitalisation réglementaire", 'Dest sortie txt'),
    ("Fiche d'hospitalisation réglementaire", 'Commune traitement habituel'),
    ("Fiche d'hospitalisation réglementaire", 'Continuité des soins'),
    ("Fiche d'hospitalisation réglementaire", 'RDV à prendre par le patient'),
    ("Fiche d'hospitalisation réglementaire", 'RDV pris pour le patient'),
    ("Fiche d'hospitalisation réglementaire", 'Autre continuité'),
    ("Fiche d'hospitalisation réglementaire", 'Consignes'),
    ("Fiche d'hospitalisation réglementaire", 'Prise en charge sociale'),
    ("Fiche d'hospitalisation réglementaire",
    'Conseil, recommandation ou surveillance particulière'),
    ("Fiche d'hospitalisation réglementaire", 'UniqueId'),
    ("Fiche d'hospitalisation réglementaire", 'Surface corporelle'),
    ("Fiche d'hospitalisation réglementaire", 'Gen crh'),
    ("Fiche d'hospitalisation réglementaire", 'Gen doc sortie val crh'),
    ("Fiche d'hospitalisation réglementaire", 'Gen doc sortie'),
    ("Fiche d'hospitalisation réglementaire",
    'Patient porteur/contact de BMR ou BHRe.1'),
    ("Fiche d'hospitalisation réglementaire", 'Specialite fiche intro / Titre'),
    ("Fiche d'hospitalisation réglementaire", 'Specialite fiche contenu / Titre'),
    ("Fiche d'hospitalisation réglementaire",
    'Specialite fiche conclusion / Titre'),
    ("Fiche d'hospitalisation réglementaire", 'Sejours / Id venue'),
    ("Fiche d'hospitalisation réglementaire", 'Sejour selectionne / Id venue'),
    ("Fiche d'hospitalisation réglementaire", 'Antecedent / Type antecedent'),
    ("Fiche d'hospitalisation réglementaire",
    'Antecedent / Libelle antecedent sans cim'),
    ("Fiche d'hospitalisation réglementaire",
    'Incl historique ordonnance / Prescripteur'),
    ("Fiche d'hospitalisation réglementaire",
    'Incl historique ordonnance / Date ordonnance'),
    ("Fiche d'hospitalisation réglementaire", 'Evol et suivi / Date mot'),
    ("Fiche d'hospitalisation réglementaire", 'Evol et suivi / Intervenant mot'),
    ("Fiche d'hospitalisation réglementaire",
    'Evol et suivi / Intervenant mot id'),
    ("Fiche d'hospitalisation réglementaire",
    'Evol et suivi / Intervenant mot txt'),
    ("Fiche d'hospitalisation réglementaire",
    'Evol et suivi croissant / Date mot'),
    ("Fiche d'hospitalisation réglementaire",
    'Evol et suivi croissant / Intervenant mot'),
    ("Fiche d'hospitalisation réglementaire",
    'Evol et suivi croissant / Intervenant mot id'),
    ("Fiche d'hospitalisation réglementaire",
    'Evol et suivi croissant / Intervenant mot txt'),
    ("Fiche d'hospitalisation réglementaire", 'Liste rdv / Date'),
    ("Fiche d'hospitalisation réglementaire", 'Liste rdv / Type rdv'),
    ("Fiche d'hospitalisation réglementaire", "Liste rdv / Nom de l'agenda")      
]

df_sans_col_bool.drop(to_drop, axis=1, inplace=True)
print('df sans col bool', df_sans_col_bool.shape)


## __*Création de fonctions pour sortir les informations au format voulu*__  :

Donnees_concat_dossier_gyneco  = extraction_des_données(df_sans_col_bool)
Donnees_concat_dossier_gyneco.to_csv('Data/DATA_PROCESSED/Donnees_concat_dossier_gyneco.csv', index=False)
# __*Ouverture du fichier PMSI*__ :
PASSWORD_2 = pd.read_csv('Data/DATA_RAW/Nouveau dossier/PASSWORD_2.txt').columns[0]                                                 ## Ouverture du MDP
decrypted_2 = io.BytesIO()                                                                                                 ## Création d'un fichier io 
  
with open("Data/DATA_RAW/2022 - Donnees PMSI - Protocole ENDOPATHS - GHN..ALTRAN.xlsx", "rb") as f:                            ## Décryptage du fichier excel
    file_2 = msoffcrypto.OfficeFile(f)
    file_2.load_key(password=PASSWORD_2)  # Use password
    file_2.decrypt(decrypted_2)
    
                                                                                                                           ## Ouverture des feuilles du fichier excel 
pmsi_tab_hospit = pd.read_excel(decrypted_2, sheet_name='TAB_HOSPIT', header = 1)
pmsi_tab_hospit_diags = pd.read_excel(decrypted_2, sheet_name='TAB_HOSPIT_DIAGS', header = 2)
pmsi_tab_hospit_actes = pd.read_excel(decrypted_2, sheet_name='TAB_HOSPIT_ACTES', header = 2)
pmsi_tab_consult = pd.read_excel(decrypted_2, sheet_name='TAB_CONSULT', header = 2)
pmsi_dict = pd.read_excel(decrypted_2, sheet_name='Dictionnaire de données')

dict_pmsi = {}                                                                                                ## Enregistrement dans un dictionnaire
dict_pmsi['pmsi_tab_hospit'] = pmsi_tab_hospit
dict_pmsi['pmsi_tab_hospit_diags'] = pmsi_tab_hospit_diags
dict_pmsi['pmsi_tab_hospit_actes'] = pmsi_tab_hospit_actes
dict_pmsi['pmsi_tab_consult'] = pmsi_tab_consult
dict_pmsi['pmsi_dict'] = pmsi_dict

for key, value in dict_pmsi.items():
    rows_elem, columns_elem = value.shape
    print(f'Le fichier {key} contient {rows_elem} lignes et {columns_elem} colonnes')

liste_consult = from_serie_to_list(pmsi_tab_consult.CAM_LIBELLE_COMPLET)
liste_diags = from_serie_to_list(pmsi_tab_hospit_diags.CIM_LIBELLE_COMPLET)
liste_actes = from_serie_to_list(pmsi_tab_hospit_actes.CAM_LIBELLE_COMPLET)
liste_lexique_medical = liste_consult + liste_diags + liste_actes
liste_lexique_medical = remove_duplicat(liste_lexique_medical)
liste_lexique_medical = liste_lexique_medical[1:]


pd.Series(liste_lexique_medical).to_csv('Data/DATA_PROCESSED/lexique_medical.csv', index=False)
liste_pathologies = list(pmsi_tab_hospit_diags.loc[:,'CIM_LIBELLE_COMPLET'].unique())

donnees_endometrioses = get_data_endo(pmsi_tab_hospit_diags, pmsi_tab_hospit_actes)
donnees_endometrioses.to_csv('Data/DATA_PROCESSED/donnees_endometriose.csv')

serie_test = preparation_des_dates_a_tronquer(donnees_endometrioses)
df_to_troncate = pd.read_csv('Data/DATA_PROCESSED/Donnees_concat_dossier_gyneco.csv')
print(df_to_troncate.shape)
données_sans_endo  = troncature_des_datas(df_to_troncate, serie_test)
print(données_sans_endo.shape)
données_sans_endo.to_csv('Data/DATA_PROCESSED/donnees_entree_nlp_sans_endo.csv')
