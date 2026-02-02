'''
Maxime Mock
05/10/2022

Le but : Mettre toutes les fonctions utiles pour gérer la création d'une liste de mot, sa vérification orthographique, les corrections, le traitement de liste de mots, de chaines de caractères.

'''
# NBS: Subset of preprocess_dossier_gyneco. Does not comply with DRY principle

import pandas as pd
import string



def bien_othographie(string, lexique):
    booleen = string in lexique
    return booleen

def mot_le_plus_proche(string:str, lexique):
    mot, score, indice = process.extractOne(string, lexique, scorer = fuzz.token_set_ratio)
    # Answer est de la forme : (mot_le_plus_proche, score, indice dans le lexique)
    return mot, score 

def Series_to_list(Serie:pd.DataFrame()):
    liste = []
    for value in Serie.values:
        liste = liste + value
    return liste

def df_nlp(df:pd.DataFrame()):
    dict_={}
    liste_key = list(df.Anonymisation.unique())
    for key in liste_key:
        dict_[key] = Series_to_list(df.loc[df.Anonymisation==key, 'Résumé'])
    Serie_nlp = pd.Series(dict_)
    return Serie_nlp

def clear_string(strg:str):
    string_temp = strg
    for cara in strg:
        if cara in string.punctuation:
            string_temp = string_temp.replace(cara,' ')
    return string_temp.lower()

def from_serie_to_list(serie):
    '''Prend un sérié, récupère les chaines de caractère, enlève la ponctuation, concatène tout les mots dans une liste unique
    Input :
    Serie: pd.Series
    Return :
    liste: list()
    '''
    liste = []
    for value in serie:
        value_cleared = clear_string(value)
        liste_temp = value_cleared.split(' ')
        liste = liste + liste_temp
        liste = list(set(liste))
    return liste

def remove_duplicat(liste:list()):
    return list(set(liste))

def extraction_des_strings(df:pd.DataFrame()):
    liste_full_string = []
    for liste in df['Résumé'].values:
        liste_temp = liste[0].split(' ')
        liste_full_string = liste_full_string +liste_temp
    return liste_full_string

def liste_unique(liste:list()):
    liste_unique  =list(set(liste))
    return liste_unique

def creation_regex(string:str):
    str_temp = '('
    for c in string:
        if c =='?':
            str_temp = str_temp+')['+c+']('
        else:
            str_temp = str_temp+'['+c+']'    
    str_temp = str_temp +')'
    return str_temp

def is_all_punctuation(string:str):
    booleen = True    
    for char in string:
        if char.isalnum() == True :
            booleen = booleen and False
    return booleen

def ponctu_avant(string:str):
    to_remove=''
    length = len(string)
    for i in range(length):
        c = string[i]
        if c.isalnum() ==False:
                to_remove = to_remove + c
        if c.isalnum() == True:
            break

    string = string.replace(to_remove, '')
    return string

def ponctu_apres(string:str):
    to_remove = ''
    length = len(string)
    for i in range(1, (length+1)):
        c = string[-i]
        if c.isalnum() ==False:
                to_remove = to_remove + c
        if c.isalnum() == True:
            break
    to_remove = to_remove[::-1]
    string = string.replace(to_remove, '')
    return string

def traitement_interrogation(mot, mot_temp:str):
    dictionnaire = dict()
    regex_temp = creation_regex(mot)
    recherche = re.search(regex_temp, mot)
    length = len(recherche.groups())
    length_1 = len(recherche.group(1))
    length_2 = len(recherche.group(2))
    avant = recherche.group(1)
    apres = recherche.group(length)
    estradiol = re.compile(r"[?][s][t][r][a][d][i][o][l]", re.IGNORECASE)
    bool_11 = bool(estradiol.search(mot_temp))
    coelioscopie = re.compile(r"([\w']{1,})([c])[?]([l][i][o][a-z]*)", re.IGNORECASE)
    bool_12 = bool(coelioscopie.search(mot_temp))

    if bool_11:
        mot_temp = 'estradiol'           
    elif bool_12 and len(coelioscopie.search(mot_temp).group(1)) >1:
        mot_temp= coelioscopie.search(mot_temp).group(1) +' coelioscopie'
    elif is_all_punctuation(avant) and length !=0:
        mot_temp = mot_temp.replace(avant, "")
        mot_temp = mot_temp.replace('?', "")
    elif is_all_punctuation(apres):
        mot_temp = mot_temp.replace(apres, "")
        mot_temp = mot_temp.replace('?', "")

    elif recherche.group(1).isalnum() == False and len(recherche.group(1)) ==1:
        mot_temp = mot_temp.replace(recherche.group(length), "")
        mot_temp = mot_temp.replace('?', "")
    elif length_1 == 1 and recherche.group(1).isalnum() == True: 
        coelioscopie = re.compile(r"([c])[?]([l][i][o][a-z]*)", re.IGNORECASE)
        bool_1 = bool(coelioscopie.search(mot_temp)) 

        soeur = re.compile(r"([s])[?]([u][r])", re.IGNORECASE)
        bool_2 = bool(soeur.search(mot_temp))

        atteinte = re.compile(r"([a])[?]([e][i])", re.IGNORECASE)
        bool_3 = bool(atteinte.search(mot_temp))

        cest = re.compile(r'[C][?][e][s][t]', re.IGNORECASE)
        bool_7=bool(cest.search(mot_temp))

        cétait = re.compile(r'[C][?][é][t][a][i][t]', re.IGNORECASE)
        bool_8= bool(cétait.search(mot_temp))

        if bool_1:             
            length = len(coelioscopie.search(mot_temp).group(1))
            mot_temp = 'coelioscopie'
        elif bool_2:
            mot_temp = 'soeur'
        elif bool_3:
            mot_temp = 'atteinte'
        elif bool_7:
            mot_temp = "c'est"
        elif bool_8:
            mot_temp = "c'était"
        else : 
            mot_temp= mot_temp.replace('?', "'")

    elif length_1>1 :
        manoeuvre = re.compile(r'([m][a][n])[?]([u][v][r])', re.IGNORECASE)
        bool_4 = bool(manoeuvre.search(mot_temp))

        a_l_irm =  re.compile(r'([à])([l])[?]([I][R][M])', re.IGNORECASE)
        bool_5 = bool(a_l_irm.search(mot_temp))

        qu_ =  re.compile(r'([\w]{0,})([q][u])[?]([\w]*)', re.IGNORECASE)
        bool_6 = bool(qu_.search(mot_temp))

        optimizette = re.compile(r'[o][p][?][m][i][z][e][?][e]', re.IGNORECASE)
        bool_9 = bool(optimizette.search(mot_temp))

        atteinte = re.compile(r"([d]['][a])[?]([e][i][n][t][e])", re.IGNORECASE)
        bool_10 = bool(atteinte.search(mot_temp))

        if bool_4:
            mot_temp = 'manoeuvre'
        elif bool_5:
            mot_temp = "à l'IRM"
        elif bool_9:
            mot_temp = "optimizette"
        elif bool_6:
            mot_temp = mot_temp.replace('?', "'")
        elif bool_10:
            mot_temp = "d'atteinte"
        elif length_2 == 0:
            mot_temp = mot_temp.replace('?', '')
        else:
            mot_temp =  mot_temp.replace('?', 'ti')   

    elif length_2 == 0 or length_1 == 0 :
        mot_temp = mot_temp.replace('?', '')

    else:
        mot_temp =  mot_temp.replace('?', 'ti')

    return mot_temp



def dictionnaire_remplacement(df:pd.DataFrame()):
    dictionnaire_to_replace = dict()
    dictionnaire_unchanged = dict()
    
    # dict_ = {'valeur_a_remplacer': 'nouvelle_valeur'}
    liste_full_string = extraction_des_strings(df)
    liste_to_replace = liste_unique(liste_full_string)
    liste_apostrophe = ["d", "s", "m", "n", "l", "j", "c"]
    exception = ['op?mize?e', 'c?lioscopie', 's?ur', 'àl?IRM', 'man?uvre']
    # reg_interro = r'([\w]*)[?]([\w]*)'
    for mot in liste_to_replace:
        booleen = is_all_punctuation(mot)
        mot_temp = ponctu_avant(mot)
        mot_temp = ponctu_apres(mot_temp)
    
        if '?' in mot_temp and booleen == False: 
            mot_temp = traitement_interrogation(mot, mot_temp)
        elif mot != mot_temp:
            dictionnaire_to_replace[mot] = mot_temp
        else:
            dictionnaire_unchanged[mot] = mot_temp
    return dictionnaire_to_replace, dictionnaire_unchanged


def ponctu_apres(string:str):
    to_remove = ''
    length = len(string)
    for i in range(1, (length+1)):
        c = string[-i]
        if c.isalnum() ==False:
                to_remove = to_remove + c
        if c.isalnum() == True:
            break
    to_remove = to_remove[::-1]
    string = string.replace(to_remove, '')
    return string

def check_if_string(string:str):
    return string.isalpha()

def split_words(liste:list()):
    liste_mot = []
    liste_pas_mot = []
    for word in liste:
        booleen = check_if_string(word)
        if booleen == True:
            liste_mot.append(word)
        else:
            liste_pas_mot.append(word)
    return liste_mot, liste_pas_mot