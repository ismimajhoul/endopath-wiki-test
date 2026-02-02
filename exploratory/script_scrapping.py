'''
@author = Maxime Mock

This script is edited for getting a spelling correction thanks to google website requests
help : https://www.scraperapi.com/blog/headers-and-cookies-for-web-scraping/
'''


import requests
from lxml import html, etree
from io import StringIO, BytesIO
from bs4 import BeautifulSoup
import pickle
import numpy as np
import time
import os
import json

#update la fin : checked
#update le début : to do 


def save_dict(dictionnaire:dict, path_name_file:str):
    with open(path_name_file, "w") as f2:
        json.dump(dictionnaire, f2)

def load_json(path):
    with open(path) as json_file:
        return json.load(json_file)

def merge_dict(dict_1, dict_2):
    dict_1.update(dict_2)
    return dict_1

def save_list(liste, path_list):
    with open(path_list, "wb") as fp:  
        pickle.dump(liste, fp)

def load_list(path):
    with open(path, "rb") as fp:
        return pickle.load(fp)

def delay():
    int_ = np.random.randint(1,6)
    time.sleep(int_)


def get_corr(mot, headers):
        response = requests.get(f"https://www.google.com/search?q={mot}", headers=headers).text
        soup = BeautifulSoup(response, 'html.parser').select_one('a.gL9Hy')
        soup2 = BeautifulSoup(response, 'html.parser').select("span[data-dobid]")
        return soup, soup2
    
def init(path):
    dictionnaire = {}
    dict_dictionnaire = {'nom':'', 'value':dictionnaire}
    dict_liste={'nom':'', 'value':None}
    headers = { 'authority' : 'www.google.com',
        'accept-language' : 'fr',
        'user-agent' : 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.0.0 Safari/537.36',
        'cookies' : 'CONSENT=PENDING+354' }
    list_dir = os.listdir(path)
    for elem in list_dir:
        if 'dict' in elem:
            dictionnaire =load_json(os.path.join(path+elem))
            dict_dictionnaire['value'] = dictionnaire
            dict_dictionnaire['nom'] = os.path.join(path+elem)
        elif 'list' in elem:
            liste = load_list(os.path.join(path+elem))
            dict_liste['value'] = liste
            dict_liste['nom'] = os.path.join(path+elem)
    return headers, dict_liste, dict_dictionnaire,

def correction(path_of_file):
    headers, dict_liste, dict_dictionnaire = init(path_of_file)
    liste_path = dict_liste['nom']
    liste =  dict_liste['value']
    dict_path = dict_dictionnaire['nom']
    dictionnaire =  dict_dictionnaire['value']
    print(f'Il reste {len(liste)} mots à corriger')
    if dict_path == '':
        dict_path = path_of_file+'/dictionnaire_correction'

    liste_copy = liste.copy()
    if len(liste)!=0:
        for mot in liste:
            correction, correction2 = get_corr(mot, headers)
            delay()
            if correction:
                print('Done')
                dict_temp = {mot:correction.text}
                dictionnaire.update(dict_temp)
                save_dict(dictionnaire, dict_path)
                liste_copy.remove(mot)
                save_list(liste_copy, liste_path)
            elif len(correction2)!=0:
                print('done2')
                dict_temp = {mot:correction2[0].contents[0]}
                dictionnaire.update(dict_temp)
                save_dict(dictionnaire, dict_path)
                liste_copy.remove(mot)
                save_list(liste_copy, liste_path)
            else:
                print('mot correct')
                liste_copy.remove(mot)
                save_list(liste_copy, liste_path)
    else:
        print('Plus de mot à corriger')



# def correction_scrap_google(liste_mot_a_corrige:list()):
#     '''
#     Cette fonction permet de lancer des requêtes sur le site google afin de verifier une liste de mot.
    
#     Input:
#     --------
#     liste_mot_a_corrige = liste de mots à corriger
    
#     Return :
#     --------
#     dictionnaire : contient les correspondances entre les mots à corriger et les mots corrigés
#     liste de mot sans correction : liste contenant les mots ne trouvant pas de corrections
#     '''
#     headers = {
#     'authority' : 'www.google.com',
#     'accept-language' : 'fr',
#     'user-agent' : 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.0.0 Safari/537.36',
#     'cookies' : '    CONSENT=PENDING+354; SOCS=CAESHAgCEhJnd3NfMjAyMjA2MjItMF9SQzMaAmZyIAEaBgiA8viVBg; SEARCH_SAMESITE=CgQIsZYB; SID=QwjVtg04YiDbpOV1P3h33clRtiWGKAo_a66xatAkEZHYniZoE5J0kFSIWbzh9Tx4KMKFHQ.; __Secure-1PSID=QwjVtg04YiDbpOV1P3h33clRtiWGKAo_a66xatAkEZHYniZoeP6IjDjsTBhbR6dUc9Sblw.; __Secure-3PSID=QwjVtg04YiDbpOV1P3h33clRtiWGKAo_a66xatAkEZHYniZodvRXL8g2f4UFGnWsngzNkA.; HSID=A204aawd684bbN6sa; SSID=AVx-TSTDL6RGDZcnC; APISID=YJ0DD_aai33mqVoX/AuHnTKuVTJBPKoU33; SAPISID=kMr0vpMRJ6byAsdU/Ahjgk4TUTf9b7Ch0b; __Secure-1PAPISID=kMr0vpMRJ6byAsdU/Ahjgk4TUTf9b7Ch0b; __Secure-3PAPISID=kMr0vpMRJ6byAsdU/Ahjgk4TUTf9b7Ch0b; OTZ=6778558_52_52_123900_48_436380; NID=511=NkzLgLBkIu5mPM2mY8hR7zCcVT8XzmjsCPifoZDakiD99W95DJEOwCkrzwCZsho2Dr5H6X9izp89cEBJokHqWxR4pRdLihYOimk8Jv1DD2409kQGLauLczuM_J098a2xY3Y1m91f9X1pt2FSnh7h87qAZ2nNAzFbLgrHyu0POZufAbAQgS1jDJETMa93u9OJeN4SUsZ8INriDqApauN0yHvyGZ0-iPY5eWVdqKRhedoHmSOr; __Secure-ENID=8.SE=h8Nq-diMESxnObplMucDr9lB9ZrL2IS3skNsuMR-6VxyQyZiEmrW7e8Bi5cypXiJqjSKzYfiwcU7HGGqm6bcBXqxQaVbNsoXjLWHiv8d1EZV5hWpICo67GyFnGP_IrjgT_0XMJK0u2PmHY_veuw556exCLDANex0Ykji2cpY1izbfQTi5_vTq5aDhjnIEy6ufU4N7NgXF-B2KP-BAAsDoM3GtQwW-vZECbredpALEg; AEC=AakniGMYehzURS1bi6eu568OPQETBwYmNvkgz45RjWuz7N-dmssvAKDDaw; DV=Y72v80O0njsSAJonGalrPPVlwE07Shg; SIDCC=AIKkIs0sogd8gMkrY_8Ibo84ePrnpinwka-nF28hQNFto1zgdxn6ghYhoRdYQgrTj9visvr_Hg; __Secure-1PSIDCC=AIKkIs2v3uqqzLcLhMYyxm6DMTqSbKC_reNez2DgNzL9hlSg0dgrVwfSWwhlQpqk0Nk9kT96nw; __Secure-3PSIDCC=AIKkIs36h2cERSsP45brOH9dOUpC_oDOUDzTeDf6wzOMWdIFvvLuJ2-GMp2w3TZmZZMGpIRp7Q'
# }


#     liste_mot_sans_correction=[]
#     dictionnaire= {}
#     for mot_a_corriger in liste_mot_a_corrige:
#         correction = get_corr(mot_a_corriger, headers)
#         if correction !=None:
#             mot_corrige = correction.text
#             dictionnaire[mot_a_corriger] = mot_corrige
#         else:
#             liste_mot_sans_correction.append(mot_a_corriger)
            
#     return dictionnaire, liste_mot_sans_correction

# def correction_scrap_google2(liste_mot_a_corrige:list()):
#     '''
#     Cette fonction permet de lancer des requêtes sur le site google afin de verifier une liste de mot.
    
#     Input:
#     --------
#     liste_mot_a_corrige = liste de mots à corriger
    
#     Return :
#     --------
#     dictionnaire : contient les correspondances entre les mots à corriger et les mots corrigés
#     liste de mot sans correction : liste contenant les mots ne trouvant pas de corrections
#     '''
#     headers = {
#     'authority' : 'www.google.com',
#     'accept-language' : 'fr',
#     'user-agent' : 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.0.0 Safari/537.36',
#     'cookies' : 'CONSENT=PENDING+354'
# }


#     liste_mot_sans_correction=[]
#     dictionnaire= {}
#     for mot_a_corriger in liste_mot_a_corrige:
#         correction = get_corr(mot_a_corriger, headers)
#         if correction !=None:
#             mot_corrige = correction.text
#             dictionnaire[mot_a_corriger] = mot_corrige
#         else:
#             liste_mot_sans_correction.append(mot_a_corriger)
            
#     return dictionnaire, liste_mot_sans_correction


