Ci joint une explication sur les fichier contenu dans le fichier compressé.

Liste des fichiers : 

- liste_tous_les_mots.txt : contient tous les mots du jeux de données.

- feature_text_tokenized.txt : contient la liste de features textuelles que j'obtiens après le traitement du dossier gynéco, elle a une longueur de 2295 éléments

- liste_mot_à_corriger.txt : contient la liste des mots que j'ai à corriger (mélange abréviations, fautes orthographique, noms propres)

__________________________________________________________
Les fichiers ont été enregistrés via la libraire pickle.
Si jamais pour ouvrir un fichier : 
'''
with open('path_of_my_list'rb') as fp:
    ma_liste_de_mots = pickle.load(fp)
'''	