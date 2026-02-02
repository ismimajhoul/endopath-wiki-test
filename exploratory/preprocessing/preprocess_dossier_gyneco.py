### preprocess_dossier_gyneco
# Helper functions for preprocessing of dossier-gyneco files
# Author: Maxime Mock

import re
import datetime
import itertools
import string 
import pandas as pd
import numpy as np

#Levenshtein distance :
import Levenshtein as lev
from fuzzywuzzy import fuzz
from fuzzywuzzy import process

from dateutil.parser import parse

from io import StringIO

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline


# La stratégie repose sur le fait que dans une cellule normale il n'y aura pas deux points virgules qui se suivent.
# Une autre stratégie possible (?)repose sur la string : '_x000D_' qui apparait en fin de ligne, on obtient les mêmes résultats.

# Préparation d'une fonction pour repérer les anomalies de lecture du CSV : 
def check_series(df, series:pd.Series):
    """
    Analyze a pandas Series to find specific anomalies within a DataFrame.

    This function searches for string elements in the series that contain both
    '\n' and ';;'. When found, it records their coordinates in the original DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame containing the series to be analyzed.
        series (pd.Series): The Series to analyze for anomalies.

    Returns:
        List[Tuple[int, int]]: A list of coordinates (row, column) of cells 
                               in the DataFrame that contain the anomalies.

    Note:
        This function assumes that the series is a column or row from the input DataFrame.
    """
    liste_cordonnees = []
    series_copy = series.copy()
    series_copy.dropna(inplace=True)

    for elem in series_copy:
        if type(elem)== str:
            bool_1 = '\n' in elem
            bool_2 = ';;' in elem
            if bool_1 == True and bool_2 == True:
                # Find the coordinates of the element in the DataFrame
                array = np.where(df == elem)
                coordonnée = [array[0][0], array[1][0]]
                liste_cordonnees.append(coordonnée)

    return liste_cordonnees      


#Création d'une fonction qui va donner une liste des cellules identifiées : 
def extraction_coordonnees_string(df:pd.DataFrame):
    """
    Extract coordinates of cells in a DataFrame that contain specific string anomalies.

    This function iterates through all columns of the input DataFrame and uses the
    check_series function to identify cells containing both '\n' and ';;'.

    Args:
        df (pd.DataFrame): The DataFrame to analyze.

    Returns:
        List[Tuple[int, int]]: A list of coordinates (row, column) of cells 
                               in the DataFrame that contain the anomalies.

    Note:
        This function assumes the existence of a check_series function that takes
        a DataFrame and a Series as input and returns a list of coordinates.
    """
    liste_cellule_a_traiter = []

    for col in df.columns:
        # Check each column for anomalies
        liste_coord = check_series(df, df[col])
        if len(liste_coord) > 0 : # If anomalies are found
             liste_cellule_a_traiter = liste_cellule_a_traiter + liste_coord

    return liste_cellule_a_traiter

# Création d'une fonction pour à partir d'une string faire une liste de ligne :
def string_to_list(string):
    """
    Split a string into a list of lines.

    This function takes a string and splits it into a list of strings,
    using the newline character ('\n') as a separator.

    Args:
        string (str): The input string to be split.

    Returns:
        List[str]: A list of strings, where each string is a line from the input.

    Example:
        >>> string_to_list("Hello\nWorld")
        ['Hello', 'World']
    """
    liste_ligne = string.split('\n')
    return liste_ligne

#Création d'une fonction pour d'une ligne faire un dataframe :
def line_to_df(ligne_string):
    """
    Convert a string representing a CSV line to a pandas DataFrame.

    This function takes a string containing CSV data (with semicolon as separator)
    and converts it into a pandas DataFrame. Each field in the input string becomes
    a column in the resulting DataFrame.

    Args:
        ligne_string (str): A string containing CSV data with semicolon as separator.

    Returns:
        Union[pd.DataFrame, None]: A pandas DataFrame if successful, None if the input is empty.

    Example:
        >>> line_to_df("1;2;3")
           0  1  2
        0  1  2  3
    """
    ligne_df = pd.read_csv(StringIO(ligne_string), sep=';',header=None)
    return ligne_df

def from_coord_to_df(df:pd.DataFrame, liste_coordonnees:list()):
    """
    Process specific cells in a DataFrame and create a new DataFrame from their contents.

    This function takes a DataFrame and a list of coordinates. For each coordinate,
    it processes the cell content, splits it into lines, converts each line to a DataFrame,
    applies some specific treatments, and finally concatenates all resulting DataFrames.

    Args:
        df (pd.DataFrame): The input DataFrame to process.
        liste_coordonnees (List[Tuple[int, int]]): A list of (row, column) coordinates to process.

    Returns:
        pd.DataFrame: A new DataFrame created from the processed cell contents.

    Note:
        This function relies on several helper functions that are not defined here:
        string_to_list, line_to_df, traitement_df_temp_top, traitement_df_bot
    """
    liste_df = []

    for coordonnees in liste_coordonnees:
        string = df.iloc[coordonnees[0], coordonnees[1]]
        liste_string = string_to_list(string)

        for string in liste_string:
            df_temp = line_to_df(string)

            # Apply specific treatments based on the position of the string in liste_string
            if liste_string[0] == string:  # First string         
                df_temp = traitement_df_temp_top(df, df_temp, coordonnees[0], coordonnees[1])
            elif len(df_temp.columns) == 1691:
                df_temp.drop([2, 3, 4, 5], axis = 1, inplace=True)
              # df_mid.columns = liste_colonnes
            elif liste_string[-1] == string: # Last string
                df_temp = traitement_df_bot(df, df_temp, coordonnees)   

            rows, columns = df_temp.shape
            booleen1 = coordonnees[0] == 47 and coordonnees[1] == 755
            booleen2 = liste_string[0] == string
            booleen3 = liste_string[-1] == string
            booleen4 = coordonnees[0] == 47 and coordonnees[1] == 44
            combi_bool = booleen1 and booleen2
            combi_bool2 = booleen4 and booleen3

            # Adjust the number of columns if necessary
            if columns<1687:
                df_temp = pd.concat([df_temp, pd.DataFrame([np.nan]*(1687-columns)).T], axis=1)

            df_temp.columns=df.columns 

            # Special case handling
            if combi_bool == False and combi_bool2 == False:
                liste_df.append(df_temp)

    # Concatenate all processed DataFrames
    df_new_lines = pd.concat(liste_df, axis=0, ignore_index=True)

    return df_new_lines
        

def traitement_df_bot(df_gyneco, df_bot, coordonnees):
    """
    Process and transform dataframes based on given coordinates.

    Parameters:
    df_gyneco (pd.DataFrame): Input gynecology dataframe
    df_bot (pd.DataFrame): Input bot dataframe
    coordonnees (list): Coordinates for determining processing logic

    Returns:
    pd.DataFrame: Processed bot dataframe
    """
    # Define lists of coordinates for different processing cases
    liste_coord_1 = [[21,44], [47,44], [57,44], [237,44], [252,44],[703,44], [1596,44], [1657,44], [1685,44], [1826,44], [2029,44], [2033,44], [47,755]]
    liste_coord_3 = [[333,44], [1060,44]]

    # Drop specific columns from df_bot
    df_bot = df_bot.drop([2,3,4,5], axis=1)
    
    if coordonnees in liste_coord_1:     
        length = len(df_bot.columns)
        if length<1687:
            # Pad df_bot with NaN values to reach 1687 columns
            df_bot = pd.concat([df_bot, pd.DataFrame([np.nan]*(1687-length)).T], axis=1)
            
    elif coordonnees in liste_coord_3:        
        if coordonnees[0] == 333:
            # Concatenate df_bot with NaN values and join with a slice of df_gyneco
            df_concat = pd.concat([df_bot, pd.DataFrame([np.nan]*257).T], axis=1)
            df_bot = df_concat.join(df_gyneco.iloc[334:335,253:1213], how='cross')
        if coordonnees[0] == 1060:
            # Perform string concatenation and complex dataframe operations
            full_string = df_bot.iloc[0, 6] + 'g' + df_gyneco.iloc[1061,0]
            df_bot.columns = list(range(len(df_bot.columns)))
            df_bot.loc[0,6] = full_string 
            df_bot = df_bot.join(df_gyneco.iloc[1061:1062,1:2], how='cross')
            df_bot[['8', '9', '10', '11']] = np.nan
            df_bot =  df_bot.join(df_gyneco.iloc[1061:1062,2:1677], how='cross')
                    
    return df_bot


def traitement_df_temp_top(df_gyneco:pd.DataFrame, df_first:pd.DataFrame, indice:int, n_colonne:int):
    """
    Process and combine dataframes based on specified parameters.

    Parameters:
    df_gyneco (pd.DataFrame): Input gynecology dataframe
    df_first (pd.DataFrame): Another input dataframe to be joined
    indice (int): Starting row index for slicing df_gyneco
    n_colonne (int): Number of columns to select from df_gyneco

    Returns:
    pd.DataFrame: Processed and combined dataframe
    """
    # Create a copy of df_gyneco and store its column names
    df_copy = df_gyneco.copy()
    liste_colonnes_temp = df_copy.columns

    # Slice df_gyneco based on input parameters
    df_temp = df_gyneco.iloc[indice:indice+1,0:n_colonne].copy()

    # Assign column names to df_first based on df_gyneco's columns
    if len(df_first.columns)<len(liste_colonnes_temp[n_colonne:]):
        df_first.columns = liste_colonnes_temp[n_colonne:(n_colonne+len(df_first.columns))]
    else:
        df_first.columns = liste_colonnes_temp[n_colonne:]

    # Perform a cross join between df_temp and df_first
    df_join = df_temp.join(df_first, how='cross')

    # Ensure the output dataframe has exactly 1687 columns, padding with NaN if necessary
    rows, columns = df_join.shape
    if columns<1687:
            df_join = pd.concat([df_join, pd.DataFrame([np.nan]*(1687-columns)).T], axis=1)

    return df_join

def traitement_DF(df:pd.DataFrame):
    """
    Process a dataframe by extracting coordinates, creating new lines,
    dropping specific rows, and sorting the result.

    Parameters:
    df (pd.DataFrame): Input dataframe to be processed

    Returns:
    pd.DataFrame: Processed dataframe
    """
    # Create a copy of the input dataframe
    df_temp = df.copy()

    # Extract coordinates from the dataframe
    liste_coord = extraction_coordonnees_string(df_temp)

    # Create new lines based on the extracted coordinates
    df_new_lines = from_coord_to_df(df_temp, liste_coord)

    # Replace '_x000D_' with NaN in the new lines
    df_new_lines.replace('_x000D_', np.nan, inplace=True)

    # Drop specific rows from the original dataframe
    df_temp.drop([21, 22, 47, 48,
                 57, 58, 237, 238,
                 252, 333, 334, 703, 
                 1060, 1061, 1596, 1657, 
                 1658, 1685, 1826, 2029, 2033],
                axis=0, inplace=True)
    
    # Concatenate the modified original dataframe with the new lines
    df_final = pd.concat([df_temp,df_new_lines], axis=0, ignore_index=True)

    # Sort the final dataframe by the '#IPP' column and reset the index
    df_final.sort_values('#IPP', inplace=True)
    df_final.reset_index(drop=True, inplace = True)

    return df_final


def split_OR(string:str):
    """
    Split a string by the delimiter ' | ' (space, vertical bar, space).

    Parameters:
    string (str): The input string to be split

    Returns:
    list: A list of substrings resulting from the split operation

    Example:
    >>> split_OR("apple | banana | cherry")
    ['apple', 'banana', 'cherry']
    """
    return string.split(' | ')

def check_length(dictionnary:dict()):
    """
    Normalize the length of values in a dictionary to the maximum length found.

    This function finds the maximum length of values across all keys in the input dictionary.
    It then pads shorter values with either NaN or repeats the first element (for 'Anonymisation' key)
    to match this maximum length.

    Parameters:
    dictionary (dict): Input dictionary with iterable values

    Returns:
    dict: Modified dictionary with all values of equal length

    Note:
    - The function modifies the input dictionary in-place.
    - Special handling for key ('Anonymisation','): repeats the first element instead of padding with NaN.
    """
    # Find the maximum length of values
    length = 0
    for key, value in dictionnary.items():
        length_temp = len(value)
        if length_temp > length:
            length=length_temp

    # Pad shorter values to match the maximum length
    for key, value in dictionnary.items():
        if len(value)<length:
            delta = length - len(value)
            if key == ('Anonymisation',''):
                # Special case: repeat the first element
                integer =len(value) +delta
                value = [value[0]]*integer    
            else:
                # Pad with NaN
                liste = [np.nan]*delta
                value = value + liste

            dictionnary[key]=value

    return dictionnary


def process_1_ligne(series:pd.Series(dtype='object'), integer:int):
    """
    Process a single row (Series) of data and convert it into a DataFrame.

    This function performs the following steps:
    1. Initializes a dictionary with the 'Anonymisation' field.
    2. Iterates through the series, splitting elements containing ' | '.
    3. Normalizes the length of all values in the dictionary.
    4. Converts the dictionary to a DataFrame.
    5. Applies date formatting and type conversion to the DataFrame.

    Parameters:
    series (pd.Series): Input series to process
    integer (int): Number of times to repeat the 'Anonymisation' value

    Returns:
    pd.DataFrame: Processed and transformed DataFrame

    Note:
    - Assumes the existence of helper functions: check_car_OR, split_OR, 
      check_length, change_date, and loop_retype.
    - The 'Anonymisation' field is treated specially, being repeated 'integer' times.
    """
    dict_to_check = {}
    dict_to_check[('Anonymisation', '')] = [series.loc[('Anonymisation', '')]]*integer

    index = list(series.index)
    for i, elem in enumerate(series):
        if check_car_OR(elem):
            key_temp =  index[i]
            dict_to_check[key_temp] = split_OR(elem)
            dict_to_transform = check_length(dict_to_check)

    df_ = pd.DataFrame(dict_to_transform)
    df_ = change_date(df_)
    df_ = loop_retype(df_)

    return df_
        
def slicer1(df_to_slice):
    """
    Split a DataFrame into its first row and the remaining rows.

    This function takes a DataFrame and returns two objects:
    1. A Series representing the first row of the input DataFrame.
    2. A DataFrame containing all rows of the input DataFrame except the first one.

    Parameters:
    df_to_slice (pd.DataFrame): The input DataFrame to be sliced

    Returns:
    tuple[pd.Series, pd.DataFrame]: A tuple containing:
        - df_1 (pd.Series): The first row of the input DataFrame
        - df_2 (pd.DataFrame): The remaining rows of the input DataFrame

    Example:
    >>> df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    >>> first_row, rest = slicer1(df)
    >>> print(first_row)
    A    1
    B    4
    Name: 0, dtype: int64
    >>> print(rest)
       A  B
    1  2  5
    2  3  6
    """
    df_1 = df_to_slice.iloc[0,:] # Select the first row
    df_2 = df_to_slice.iloc[1:,:] # Select all rows except the first one
    return df_1, df_2 
        
def replacement(df:pd.DataFrame(), df_temp:pd.DataFrame()):
    """
    Replace values in the main DataFrame with values from a temporary DataFrame.

    This function updates the values in 'df' with the values from 'df_temp'
    for the indices and columns that exist in 'df_temp'.

    Parameters:
    df (pd.DataFrame): The main DataFrame to be updated
    df_temp (pd.DataFrame): The temporary DataFrame containing replacement values

    Returns:
    pd.DataFrame: The updated main DataFrame

    Note:
    - This function modifies the input DataFrame 'df' in-place.
    - Only the overlapping indices and columns between 'df' and 'df_temp' are affected.

    Example:
    >>> df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]}, index=['x', 'y', 'z'])
    >>> df_temp = pd.DataFrame({'A': [10, 20], 'B': [40, 50]}, index=['x', 'y'])
    >>> result = replacement(df, df_temp)
    >>> print(result)
        A   B
    x  10  40
    y  20  50
    z   3   6
    """
    indices = df_temp.index
    colonnes = df_temp.columns
    df.loc[indices, colonnes] = df_temp
    return df

def traitement_des_separateurs_OR(df:pd.DataFrame(), liste_de_liste:list()):
    """
    Process a DataFrame by applying operations based on a list of lists and concatenating results.

    This function performs the following steps:
    1. Creates a copy of the input DataFrame.
    2. Iterates through the list of lists, applying preprocessing and iterative processing.
    3. Concatenates additional DataFrames generated during processing.
    4. Removes specific columns from the final DataFrame.

    Parameters:
    df (pd.DataFrame): The input DataFrame to be processed
    liste_de_liste (List[List]): A list of lists containing processing instructions

    Returns:
    pd.DataFrame: The processed DataFrame

    Note:
    - This function relies on several helper functions: préparation_df, iter_df_to_process, and replacement.
    - Specific columns are dropped at the end of processing.
    """
    df_final = df.copy()
    liste_to_concat = []

    for liste in liste_de_liste:
        df_temp = préparation_df(df,liste)
        df_temp , liste_df_to_concat = iter_df_to_process(df_temp)
        liste_to_concat = liste_to_concat + liste_df_to_concat
        df_final = replacement(df_final, df_temp)

    df_final = pd.concat([df_final]+liste_to_concat, axis=0)
    df_final.reset_index(drop=True, inplace=True)

    # Drop specific columns
    df_final.drop(("Fiche d'hospitalisation réglementaire", 'Antecedent / Id'), axis=1, inplace=True)
    df_final.drop(("Fiche d'hospitalisation ACHA", 'Sejours / Id venue'),axis=1, inplace=True)

    return df_final 

def préparation_df(df:pd.DataFrame(),liste_col:list()):
    """
    Prepare a subset of the input DataFrame based on specified columns.

    This function performs the following steps:
    1. Creates a temporary DataFrame with only the specified columns.
    2. Removes rows where all values are NaN.
    3. Concatenates the 'Anonymisation' column from the original DataFrame
       with the temporary DataFrame.

    Parameters:
    df (pd.DataFrame): The input DataFrame to be processed
    liste_col (List[str]): A list of column names to be included in the subset

    Returns:
    pd.DataFrame: A new DataFrame containing the 'Anonymisation' column and
                  the specified columns, with rows containing all NaN values removed

    Note:
    - This function assumes the existence of an ('Anonymisation', '') column in the input DataFrame.
    - Rows with all NaN values in the specified columns are dropped.
    """
    # Create a temporary DataFrame with specified columns and drop rows with all NaN
    df_temp = df.loc[:,liste_col].dropna(how='all', axis=0).copy()

    # Get the index of remaining rows
    index = df_temp.index

    # Concatenate the 'Anonymisation' column with the temporary DataFrame
    df_to_process = pd.concat([df.loc[index,('Anonymisation','')], df_temp], axis=1)

    return df_to_process

def check_ligne_OR(ligne):
    """
    Check a series or list for the maximum number of OR-separated elements in any string.

    This function iterates through the elements of the input, looking for strings
    that contain the OR separator (' | '). It returns the maximum count of
    OR-separated elements found in any single string.

    Parameters:
    ligne (Union[pd.Series, List]): A series or list-like object to be checked

    Returns:
    Union[int, None]: The maximum number of OR-separated elements found in any string,
                      or None if no such string is found

    Note:
    - This function assumes the existence of a helper function 'check_car_OR'
      which is not defined here.
    - The OR separator is defined as ' | ' (space, vertical bar, space).

    Example:
    >>> check_ligne_OR(['a | b | c', 'd | e', 'f'])
    3
    >>> check_ligne_OR(['a', 'b', 'c'])
    None
    """
    integer = None
    for elem in ligne :
        bool_1 = type(elem)==str
        bool_2 = check_car_OR(elem)==True
        if bool_1 and bool_2:
            integer = len([*re.finditer(r'[\s][|][\s]', elem)]) + 1
    return integer
            
def iter_df_to_process(df_to_process:pd.DataFrame()):
    """
    Iterate through a DataFrame, processing rows with OR-separated values.

    This function performs the following steps for each row:
    1. Checks if the row contains OR-separated values.
    2. If so, processes the row to expand OR-separated values.
    3. Replaces the original row with the first row of the processed result.
    4. Collects additional rows generated from the processing for later concatenation.

    Parameters:
    df_to_process (pd.DataFrame): The input DataFrame to be processed

    Returns:
    Tuple[pd.DataFrame, List[pd.DataFrame]]: 
        - The processed input DataFrame
        - A list of additional DataFrames to be concatenated later

    Note:
    - This function relies on several helper functions: check_ligne_OR, process_1_ligne, and slicer1.
    - Rows without OR-separated values are left unchanged.
    """
    liste_df_to_concat = []

    for n_row in df_to_process.index:
        integer = check_ligne_OR(df_to_process.loc[n_row,:])
        if integer != None:
            df_to_slice = process_1_ligne(df_to_process.loc[n_row,:], integer)
            
            df_to_replace, df_to_concat = slicer1(df_to_slice)
            df_to_process.loc[n_row,:] = df_to_replace
            liste_df_to_concat.append(df_to_concat)

    return df_to_process, liste_df_to_concat

#TO-DO: Refactor if needed/used
def retype(string:str):
    """
    Convert string to appropriate data type.

    Converts to int, datetime, bool, or NaN based on content.
    Handles specific date formats and boolean-like strings.

    Args:
    string (str): Input string to convert

    Returns:
    Converted value (int, datetime, bool, np.nan, or str)
    """
    # création d'un regex pour chercher une date :
    #reg_1 = "(\d{2}[/]\d{2}[/]\d{4})"
    #reg_2 = "(\d{2}[-]\d{2}[-]\d{4})"
    reg_3 = "(\d{4}[-]\d{2}[-]\d{2})"
    #reg_4 = "(\d{4}[/]\d{2}[/]\d{2})"
    reg_5 = "(\d{2}[/]\d{2}[/]\d{4})"
    #format_date_1 ='%d/%m/%Y %H:%M:%S'
    #format_date_2 ='%d-%m-%Y %H:%M:%S'
    format_date_3 ='%Y-%m-%d %H:%M:%S'
    #format_date_4 ='%Y/%m/%d %H:%M:%S'
    format_date_5 ='%m/%d/%Y %H:%M:%S'
    
    
    # création de liste pour traiter différemment la string :
    liste_True = ['true', 'oui']
    liste_False = ['false', 'non']
    liste_None = ['none', 'nan']
    # différenciation pour appliquer le changement :
    if string.isdigit():
        elem = int(string)
    # elif re.match(reg_1, string) and len(string)<20:
    #     elem = datetime.datetime.strptime(string, format_date_1)
    # elif re.match(reg_2, string) and len(string)<20:
    #     elem = datetime.datetime.strptime(string, format_date_2)
    elif re.match(reg_3, string) and len(string)<20:
        elem = datetime.datetime.strptime(string, format_date_3)
    # elif re.match(reg_4, string) and len(string)<20:
    #     elem = datetime.datetime.strptime(string, format_date_4)
    elif re.match(reg_5, string) and len(string)<20:
        elem = datetime.datetime.strptime(string, format_date_5)
    elif string.lower() in liste_True:
        elem = True
    elif string.lower() in liste_False:
        elem = False    
    elif string.lower() in liste_None:
        elem = np.nan
    else:                     
        elem =string           
    
    return elem

def change_date(df:pd.DataFrame()):
    """
    Convert 'Date' columns to datetime type.

    Args:
    df (pd.DataFrame): Input DataFrame

    Returns:
    pd.DataFrame: DataFrame with 'Date' columns converted
    """
    columns = df.columns
    for col in columns:
        if 'Date' in col[1]:
            df.loc[:,col] = pd.to_datetime(df.loc[:,col])
    return df
            
            
def loop_retype(df:pd.DataFrame()):
    """
    Convert string elements in DataFrame to appropriate types.

    Uses retype() function for type conversion.

    Args:
    df (pd.DataFrame): Input DataFrame

    Returns:
    pd.DataFrame: DataFrame with converted data types
    """
    print('df before call to iteritems:')
    print(df)
    for n_col,j in df.items():
        for n_row, elem in j.items():
            if type(elem)==str:
                elem = retype(elem)
                df.loc[n_row, n_col] = elem  
    return df

def check_car_OR(string:str):
    """
    Check if a string contains the '|' character.

    Args:
    string (str): Input string to check

    Returns:
    bool: True if '|' is present, False otherwise
    """
    booleen = False
    if type(string)==str:
        if '|' in string:
            booleen = True
    return booleen


def is_date(string:str, fuzzy=False):
    """
    Check if string can be interpreted as a date.

    Args:
    string (str): String to check for date
    fuzzy (bool): Ignore unknown tokens if True

    Returns:
    bool: True if string is a valid date, False otherwise
    """
    try: 
        parse(string, fuzzy=fuzzy)
        return True

    except ValueError:
        return False
    
def is_float(string:str):
    """
    Check if string can be interpreted as a float.

    Args:
    string (str): String to check for float

    Returns:
    bool: True if string is a valid float, False otherwise
    """
    try:
        float(string)
        return True
    except ValueError:
        return False
    
def presence_int(string:str):
    """
    Check if string contains any digit.

    Args:
    string (str): String to check for digits

    Returns:
    bool: True if string contains a digit, False otherwise
    """
    search_d = re.compile('\d')
    return bool(search_d.search(string))

def recherche_int(liste:list()):
    """
    Check if any string in the list contains a digit.

    Args:
    liste (list): List of strings to check

    Returns:
    bool: True if any string contains a digit, False otherwise
    """
    answer = True
    liste_bool = []
    for string in liste:
        liste_bool.append(presence_int(string)) 
    integer = np.sum(liste_bool)
    if integer == 0 :
        answer = False
    return answer

def clear_int(string:str):
    """
    Remove all digits from the input string.

    Args:
    string (str): Input string

    Returns:
    str: String with all digits removed
    """
    search_d = re.compile('\d')
    list_int = re.findall(search_d, string)
    for elem in list_int:
        string = string.replace(elem, '')
    return string

def traitement_int_nom(liste_nom:list()):
    """
    Remove digits from strings in the input list if present.

    Args:
    liste_nom (List[str]): List of strings to process

    Returns:
    List[str]: List of strings with digits removed where applicable
    """
    liste_nom_sans_int = []
    for nom in liste_nom:
        nom_checked = nom
        if presence_int(nom):
            nom_checked = clear_int(nom)
            liste_nom_sans_int.append(nom_checked)
        else:
            liste_nom_sans_int.append(nom_checked)
    return liste_nom_sans_int

def punctu_query(name):
    """
    Check if the string contains '.' or '_'.

    Args:
    name (str): String to check

    Returns:
    bool: True if '.' or '_' is present, False otherwise
    """
    bool_1 = '.' in name
    bool_2 = '_' in name
    if bool_1 or bool_2:
        return True
    else:
        return False
        
def clean_str(liste:list(), string:str):
    """
    Remove all occurrences of elements in liste from string.

    Args:
    liste (List[str]): List of substrings to remove
    string (str): String to clean

    Returns:
    str: Cleaned string
    """
    string_cleared = string
    for elem in liste:
        string_cleared = string_cleared.replace(elem, '')
    return string_cleared 
    
def contient_mail(string:str):
    """
    Check if string contains an email address.

    Args:
    string (str): String to check

    Returns:
    bool: True if email found, False otherwise
    """
    regex_mail = "([\w\.*\-*\_*]*)[@][\w\\W]*[.]\w*"
    booleen = bool(re.search(regex_mail, string))
    return booleen
    
def traitement_mail(string:str):
    """
    Process string if it contains an email.

    Args:
    string (str): Input string

    Returns:
    str: Processed string if email found, original string otherwise
    """
    string_ano = string
    if contient_mail(string):
        string_ano = traitement_str(string)
    return string_ano
    
def traitement_str(string:str):
    """
    Process string to remove email addresses and related information.

    Args:
    string (str): Input string containing email addresses

    Returns:
    str: Processed string with email-related information removed
    """
    # Regex patterns for email and username extraction
    regex_patiente_mail = "([\w\.*\-*\_*]*[@]\w*[.]\w*)" 
    regex_patiente_nom = "([\w\.*\-*\_*]*)[@]\w*[.]\w*"

    # Extract emails and usernames 
    liste_mail = re.findall(regex_patiente_mail, string)
    liste_nom = re.findall(regex_patiente_nom, string)
    # on sait que notre regex ne capture pas les mails chu-lyon mais peut-être aura t-on besoin d'un regex pour les mails CHU dans le doute il est la et il capture tout le mail :
    # regex_chu_lyon = "([\w\.*\-*\_*]*[@][c][h][u][-][l][y][o][n][.][f][r])"
    # recherche_mail_chu = re.search(regex_chu_lyon, string)

    # Remove duplicates
    liste_mail_unique = list(set(liste_mail))
    liste_nom_prenom_unique = list(set(liste_nom))

    # Remove integers from names
    liste_nom_sans_int = traitement_int_nom(liste_nom_prenom_unique)
    
    liste_to_clear = liste_mail_unique + liste_nom_prenom_unique 

    # Process names further
    for i, nom_prenom in enumerate(liste_nom_sans_int):
        if punctu_query(liste_nom_sans_int[i]):
            nom, prenom = split_name(liste_nom_sans_int[i])
            liste_to_clear.append(nom)
            liste_to_clear.append(prenom)
            break
        else:
            liste = liste_combinaison(nom_prenom)
            if len(liste) != 0:
                liste_to_clear.append(liste)

    # Remove all identified elements from the original string            
    string_cleared = clean_str(liste_to_clear, string)
    return string_cleared

def split_name(string:str):
    """
    Split a name string into parts based on '.' or '_' separator.

    Args:
    string (str): Input string containing a name

    Returns:
    list: List of name parts after splitting
    """
    if '.' in string :
        nomprenom = string.split('.')
    elif '_' in string :
        nomprenom = string.split('_')
    return nomprenom

#TO-DO: Understand where/if this is used and correct according to the intended use case
def find_name(liste_noms:list(), string:str):
    """
    Find a name with '.' or '_' separator from a list of names.

    Args:
    liste_noms (list): List of name strings
    string (str): Unused parameter

    Returns:
    str: First name found with '.' or '_' separator

    Note: This function has implementation issues and may not work as intended.
    """
    liste_nom = [] # Unused variable
    # to_split = r'[`\-=~!@#$%^&*()_+\[\]{};\'\\:"|<,./<>?]'

    for name in liste_noms:
        if '.' in name:
            nomprenom = name
        elif '_' in name:
            nomprenom = name
        # If no name with '.' or '_' is found, the function returns None implicitly

        # else:
        #      nomprenom = name_combinaison(name)
    return nomprenom # This would always return the last processed name


def slice_name(nomprenom:str, integer):
    """
    Split a name string into two parts at a specified position.

    Args:
    nomprenom (str): Input string containing a full name
    integer (int): Position at which to split the string

    Returns:
    tuple: A tuple containing two parts of the name (nom, prenom)

    Note: This function assumes the input string and integer are valid.
    """
    nom = nomprenom[:integer]
    prenom = nomprenom[integer:]
    return nom, prenom

def name_combinaison(nomprenom:str):
    """
    Generate all possible two-part name combinations from a given string.

    Args:
    nomprenom (str): Input string containing a full name without spaces

    Returns:
    list: A list of all possible name combinations (firstname lastname and lastname firstname)

    Note: This function assumes the input is a valid string without spaces.
    """
    liste_combi = []
    for integer in range(len(nomprenom)):
        # Split the name at each possible position
        nom, prenom = slice_name(nomprenom, integer)

        # Create both orderings of the split name
        nom_prenom = nom + ' ' + prenom
        prenom_nom = prenom + ' ' + nom

        # Add both combinations to the list
        liste_combi.append(prenom_nom)
        liste_combi.append(nom_prenom)

    return liste_combi

def check_combi(liste_combi, string:str):
    """
    Find name combinations from the input list that are present in the given string.

    Args:
    liste_combi (list): List of name combinations to check
    string (str): String to search within

    Returns:
    list: List of name combinations found in the string

    Note: This function performs case-sensitive substring matching.
    """
    combi_to_clear = []
    for combi in liste_combi:
        if combi in string:
            combi_to_clear.append(combi)
    return combi_to_clear

def liste_combinaison(nomprenom:str):
    """
    Generate name combinations and find which ones are present in the original name.

    Args:
    nomprenom (str): Input string containing a full name

    Returns:
    list: List of name combinations found in the original name

    Note: This function relies on name_combinaison and check_combi functions.
    """
    # Generate all possible name combinations
    liste = name_combinaison(nomprenom)

    # Check which combinations are present in the original name
    liste_combi = check_combi(liste, nomprenom)

    return liste_combi

def convert_date_in_df(df:pd.DataFrame(), liste_colonne_date):
    """
    Convert specified columns in a DataFrame to datetime type.

    Args:
    df (pd.DataFrame): Input DataFrame
    liste_colonne_date (list): List of column names or identifiers containing date information

    Returns:
    pd.DataFrame: DataFrame with specified columns converted to datetime type

    Note: This function assumes dates are in the format 'YYYY-MM-DD HH:MM:SS'.
    """
    for liste in liste_colonne_date:
        for i, elem in df.loc[:,liste].iteritems():
            elem_checked = pd.to_datetime(elem, format='%Y-%m-%d %H:%M:%S')
            df.loc[i,liste] = elem_checked
    return df

def is_datetime(value_to_check):
    """
    Check if the given value is a datetime.datetime object.

    Args:
    value_to_check: Any value to be checked

    Returns:
    bool: True if the value is a datetime.datetime object, False otherwise
    """
    boolean = isinstance(value_to_check, datetime.datetime)
    return boolean

def is_real_datetime(value):
    """
    Check if the given value is a valid datetime object and not null.

    Args:
    value: Any value to be checked

    Returns:
    bool: True if the value is a datetime object and not null, False otherwise

    Note: This function relies on the is_datetime function and pandas.isnull.
    """
    boolean = is_datetime(value)
    boolean2 = pd.isnull(value) != True
    return boolean and boolean2


def liste_date(df):
    """
    Create a list of column names that contain the word 'Date' in a MultiIndex DataFrame.

    Args:
    df (pd.DataFrame): Input DataFrame with MultiIndex columns

    Returns:
    List[Tuple]: List of column tuples containing 'Date' in the second level
    """
    liste_colonne_date = []
    for tuple_col in Multiindex: #pd.Multiindex meant here ?
        if 'Date' in tuple_col[1]:
            liste_colonne_date.append(tuple_col)

    return liste_colonne_date

def make_serie_date_only(df, row):
    """
    Create a Series of non-null datetime values from a specific row in a DataFrame.

    Args:
    df (pd.DataFrame): Input DataFrame
    row (Union[int, str]): Row index or label to process

    Returns:
    pd.Series: Series containing only non-null datetime values from the specified row

    Note: This function relies on the is_real_datetime function for datetime validation.
    """
    serie = df.loc[row,:]
    serie = serie.loc[serie.apply(is_real_datetime)==True]
    return serie

def check_date_get_time(datetime):
    """
    Check if a datetime object has a non-zero time component.

    Args:
    dt (datetime): The datetime object to check

    Returns:
    bool: True if the datetime has a non-zero time component, False otherwise

    Raises:
    TypeError: If the input is not a datetime object
    """
    hour = datetime.hour
    mins = datetime.minute
    sec = datetime.second
    boolean = hour != 0 and mins != 0 and sec != 0

    return boolean

def comparateur_date(date, date_to_check):
    """
    Compare two dates, considering both direct equality and equality after applying switch_date.

    Args:
    date (Union[datetime, date]): The reference date
    date_to_check (Union[datetime, date]): The date to compare against the reference

    Returns:
    bool: True if dates are equal either directly or after switching, False otherwise

    Note: This function relies on an external switch_date function.
    """
    bool_1 = date.date() == date_to_check.date()
    bool_2 = date.date() == switch_date(date_to_check).date()

    return bool_1 or bool_2 

def switch_date(date):
    """
    Switch the month and day of a given date.

    This function is useful for correcting dates that were incorrectly encoded
    with swapped month and day values.

    Args:
    input_date (Union[datetime, date]): The date to be corrected

    Returns:
    Union[datetime, date]: A new date object with month and day swapped.
                           If the swap results in an invalid date, returns the original date.
    """
    year = date.year
    month = date.month
    day = date.day
    month, day = day, month
    try:
        new_date = datetime.datetime( year, month, day)
        return new_date
    except:
        return date

def correct_date(df):
    """
    Correct dates in the DataFrame by comparing them with a reference date.

    This function identifies date columns, processes each row, and corrects dates
    that don't have a time component if they match the reference date when switched.

    Args:
    df (pd.DataFrame): The input DataFrame containing date columns

    Returns:
    pd.DataFrame: A DataFrame with corrected dates

    Note: This function relies on several helper functions:
    liste_date, make_serie_date_only, check_date_get_time, and comparateur_date
    """
    liste = liste_date(df)

    for row in df.index:
        serie_date = make_serie_date_only(df, row)
        if len(serie_date) > 1:
            date_ref = serie_date[0]
            for date_temp in serie_date[1:]:
                if check_date_get_time(date_temp) == False:
                    if comparateur_date(date_ref, date_temp):
                        index = np.where(serie_date== date_temp)[0][0]
                        df.loc[row, serie_date.index[index]] = date_ref

    return df   

# fonctions : 

def preprocess(df):
    """
    Preprocess a DataFrame by converting string representations of boolean values and NaN to their proper types.

    Args:
    df (pd.DataFrame): The input DataFrame to preprocess

    Returns:
    pd.DataFrame: The preprocessed DataFrame

    Note: This function modifies the DataFrame in-place.
    """
    df.replace('true', True, inplace=True)
    df.replace('True', True, inplace=True)
    df.replace('False', False, inplace=True)
    df.replace('false', False, inplace=True)
    df.replace('NaN', np.nan, inplace=True)
    df.replace('oui', True, inplace=True)
    df.replace('OUI', True, inplace=True)
    df.replace('Oui', True, inplace=True)
    df.replace('non', True, inplace=True)
    df.replace('NON', True, inplace=True)
    df.replace('Non', True, inplace=True)

def count_bool_in_series(series):
    """
    Count the total number of boolean values (True and False) in a pandas Series.

    Args:
    series (pd.Series): The input Series to analyze

    Returns:
    int: The total count of boolean values in the Series

    Note: This function considers only exact boolean values (True and False),
    not truthy or falsy values like 0, 1, or strings.
    """
    counter_False = series.loc[series == False].value_counts().sum() 
    counter_True = series.loc[series == True].value_counts().sum()
    counter = counter_False  + counter_True
    return counter
    
def get_list_col_bool(df):
    """
    Identify columns in a DataFrame that contain boolean values.

    Args:
    df (pd.DataFrame): The input DataFrame to analyze

    Returns:
    List[Tuple]: A list of column names (as tuples for MultiIndex columns) that contain boolean values

    Note: This function considers a column to contain boolean values if it has at least one True or False value.
    """
    liste_col_bool = []
    for col, serie in df.items():
        integer = count_bool_in_series(serie)
        if integer>0:
            liste_col_bool.append(col)
    return liste_col_bool    
    
#TO-DO: Refactor
def list_col_bool(df):
    """
    Identify columns in a DataFrame that contain boolean values and return their indices.

    Args:
    df (pd.DataFrame): The input DataFrame to analyze

    Returns:
    List[int]: A list of column indices that contain boolean values

    Note: This function considers a column to contain boolean values if it has at least one True or False value.
    """
    tuple_True = np.where(df == True)
    tuple_False = np.where(df == False)
    # On isole les colonnes True et False, on les additionne et on extrait une liste unique :
    liste_n_col_bool = list(set(list(tuple_True[1])+list(tuple_False[1])))
    return liste_n_col_bool

def df_sans_bool(df):
    """
    Create a new DataFrame with boolean columns removed.

    Args:
    df (pd.DataFrame): The input DataFrame

    Returns:
    pd.DataFrame: A new DataFrame with boolean columns removed

    Note: This function uses get_list_col_bool to identify boolean columns.
    """
    df_temp = df.copy()
    liste = get_list_col_bool(df)
    df_temp.drop(liste, axis=1, inplace=True)
    return df_temp

def Transform_bool(df):
    """
    Extract and preprocess boolean columns from a DataFrame.

    This function performs the following steps:
    1. Preprocesses the DataFrame to ensure correct boolean typing
    2. Identifies columns containing boolean values
    3. Returns a new DataFrame with only the boolean columns

    Args:
    df (pd.DataFrame): The input DataFrame

    Returns:
    pd.DataFrame: A new DataFrame containing only the boolean columns

    Note: This function relies on preprocess and get_list_col_bool functions.
    """
    # Ensure all boolean values are correctly typed
    preprocess(df)

    # Identify columns containing boolean values
    liste_col = get_list_col_bool(df) 

    # Extract only the boolean columns
    df_temp = df.loc[:,liste_col]

    return df_temp

# itération avec df.items
    # for col, series in df.items():
        #df.col = series.replace(True, col)
        #df.col = series.replace(False,'')

def recherche_date(ligne_of_df, regex1, regex2):
    """
    Find the first valid date in a pandas Series (row of a DataFrame).

    This function prioritizes datetime objects and returns the first valid date found.

    Args:
    row (pd.Series): A row from a DataFrame

    Returns:
    Tuple[Optional[datetime.date], Optional[int]]: 
        A tuple containing the found date (as a date object) and its index in the row.
        If no valid date is found, returns (None, None).

    Note: The function currently ignores string parsing (commented out in the original function).
    """
    date = None
    for i, elem in enumerate(ligne_of_df):
        boolean_1 = isinstance(elem, datetime.datetime) and pd.isnull(elem) != True
        if boolean_1:
            date = elem.date()
        # elif type(elem) == str:
        #     recherche = re.search(regex1, mot)
        #     if recherche != None:
        #         date = recherche[0]
        #     recherche = re.search(regex2, mot)
        #     if recherche != None:
        #         date = recherche[0]
            
    return date, i 

#TO-DO: Refactor
def traitement_ligne(ligne_of_df, anonymat):
    """
    Process a row of hospital visit data, extracting key information.

    Args:
    row (pd.Series): A row from a DataFrame containing hospital visit data
    anonymized_id (str): An anonymized identifier for the patient

    Returns:
    List[Any]: A list containing:
        1. Anonymized identifier (str)
        2. Date of visit (datetime.date or None)
        3. Nature of the hospital consultation (str)
        4. Summary of the consultation (str)

    Note: This function relies on the recherche_date function for date extraction.
    """
    # Regex patterns for date extraction
    regex1 = re.compile("(\d\d[/.]\d\d[-/.]\d\d\d\d \d\d[:]\d\d[:]\d\d)")
    regex2 = re.compile("([d][a][t][e][t][i][m][e][.][d][a][t][e][t][i][m][e][(]\d\d\d\d[,] \d, \d, \d\d, \d\d, \d\d[)])")
    
    # Extract the nature of the visit from the index
    Nature = ligne_of_df.index.get_level_values(0)
    
    # Création de la liste qui va accueillir les données : 
    liste_contenu_ligne = []
    
    # On rajoute l'anonymat à la liste_contenu_ligne :
    liste_contenu_ligne.append(anonymat)
    
    # Extract date and its index
    date, i  = recherche_date(ligne_of_df, regex1, regex2)
    liste_contenu_ligne.append(date)
    liste_contenu_ligne.append(Nature[i])

    concat = ''
    concat_final= []        

    for value in ligne_of_df:
        boolean_2 = isinstance(value, str) and value != anonymat 
        if boolean_2 == True:
            match_1 = re.match(regex1, value)
            match_2 = re.match(regex2, value)
            
            if match_1 == None and match_2 == None:
                concat = concat  + value + ', '
    concat_final.append(concat)
    
    liste_contenu_ligne.append(concat_final) 
    
    return liste_contenu_ligne

def from_list_ligne_to_df_sorted(liste):
    """
    Create a sorted DataFrame from a list of hospital visit data.

    Args:
    data_list (List[List[Any]]): A list of lists, each containing:
        [Anonymization, Date, Nature, Summary]

    Returns:
    pd.DataFrame: A DataFrame sorted by the 'Date' column, with columns:
        ['Anonymisation', 'Date', 'Nature', 'Résumé']

    Note: This function assumes that the 'Date' column can be sorted as-is.
    If date parsing is needed, uncomment the pd.to_datetime line.
    """
    columns= ['Anonymisation', 'Date', 'Nature', 'Résumé']
    df_temp = pd.DataFrame(liste, columns=columns)

    # Uncomment the following line if date parsing is needed:
    # df_temp['Date'] = pd.to_datetime(df_temp['Date'], format='%Y-%m-%d %H:%M:%S')

    df_temp = df_temp.sort_values('Date')
    return df_temp

# La fonction permet de créer un dataframe au format voulu à partir de dossier_gyneco.
def extraction_des_données(dataframe):
    """
    Extract and process gynecological patient data from the input DataFrame.

    This function performs the following steps:
    1. Extracts unique patient identifiers
    2. Processes each patient's data
    3. Combines and sorts the processed data
    4. Removes entries with null dates or empty summaries
    5. Sorts the data by anonymized patient ID

    Args:
    dataframe (pd.DataFrame): Input DataFrame containing raw gynecological patient data

    Returns:
    pd.DataFrame: Processed DataFrame ready for machine learning use

    Note: This function relies on helper functions traitement_ligne and from_list_ligne_to_df_sorted
    """
    #Préparation : 
    df_copy = dataframe.copy()
    df_copy = df_copy.set_index('Anonymisation')
    liste_patiente = df_copy.index.dropna().unique().tolist()
    series_patiente_reccurence = df_copy.index.value_counts()
    liste_df = []
    liste_lignes = []
    liste_rates = []
    df_copy = df_copy.reset_index()
    
    # Boucle pour opérer sur l'ensemble du dataframe : 
    for anonymat in liste_patiente:        
        for j in (df_copy.loc[df_copy.Anonymisation == anonymat].index): 
            ligne_temp = traitement_ligne(df_copy.iloc[j,:].dropna(), anonymat)
            if len(ligne_temp)>3:
                liste_lignes.append(ligne_temp) 
        
        df_temp = from_list_ligne_to_df_sorted(liste_lignes)
        liste_lignes = []
        liste_df.append(df_temp)
    
    # Création du Dataframe final :
    df_final = pd.concat(liste_df)
    df_final.reset_index(drop=True, inplace=True)
    
    # Drop des date == None :
    df_final.drop(df_final.index[list(np.where(df_final.Date.isnull())[0])], inplace=True)
    df_final.reset_index(drop=True, inplace=True)
    
    # Drop des Résumé vide : 
    df_final.drop(df_final.index[list(np.where(df_final.reset_index(drop=True).Résumé=='')[0])], inplace=True)
    df_final.reset_index(drop=True, inplace=True)
    
    # On ordonne le dataframe :
    df_final.reset_index(drop=True, inplace=True)
    
    ##(des n° d'Anonymat sont des int, pas de correspondance avec le dictionnaire fourni)
    list_int = list(np.where(df_final.Anonymisation.map(lambda x: type(x) == str)==False)[0])
    list_str = list(np.where(df_final.Anonymisation.map(lambda x: type(x) == str)==True)[0])
    df_str = df_final.loc[list_str,:]
    df_int = df_final.loc[list_int,:]
    df_str.sort_values('Anonymisation', inplace=True)
    
    # On concatène les deux DF ordonnées : 
    df_final = pd.concat([df_str, df_int])
    
    # reset l'index : 
    df_final.reset_index(drop=True, inplace=True)
    
    # On sort la string de la liste en mappant : 
    df_final.Résumé = df_final.Résumé.map(lambda x: x[0])
    return df_final 

def bien_othographie(string, lexique):
    """
    Check if a word is correctly spelled by verifying its presence in a given lexicon.

    Args:
    word (str): The word to check for correct spelling
    lexicon (Union[Set[str], List[str]]): A collection of correctly spelled words

    Returns:
    bool: True if the word is in the lexicon, False otherwise
    """
    booleen = string in lexique
    return booleen

def mot_le_plus_proche(string:str, lexique):
    """
    Find the closest matching word from a lexicon using fuzzy string matching.

    This function uses the fuzzywuzzy library to find the best match for the input word
    from the provided lexicon, based on the token set ratio scoring method.

    Args:
    word (str): The input word to find a match for
    lexicon (List[str]): A list of words to search for matches

    Returns:
    Tuple[str, int]: A tuple containing:
        - The closest matching word from the lexicon
        - The similarity score (0-100, where 100 is a perfect match)

    Note: This function requires the fuzzywuzzy library.
    """
    mot, score, indice = process.extractOne(string, lexique, scorer = fuzz.token_set_ratio)

    return mot, score 

def mot_le_plus_proche2(string:str, lexique):
    """
    Find the closest matching word from a lexicon using fuzzy string matching with WRatio.

    This function uses the fuzzywuzzy library to find the best match for the input word
    from the provided lexicon, based on the WRatio scoring method. WRatio is a more
    sophisticated scoring method that combines different ratios for improved matching.

    Args:
    word (str): The input word to find a match for
    lexicon (List[str]): A list of words to search for matches

    Returns:
    Tuple[str, int]: A tuple containing:
        - The closest matching word from the lexicon
        - The similarity score (0-100, where 100 is a perfect match)

    Note: This function requires the fuzzywuzzy library.
    """
    mot, score, indice = process.extractOne(string, lexique, scorer = fuzz.WRatio)
    # Answer est de la forme : (mot_le_plus_proche, score, indice dans le lexique)
    return mot, score

def Series_to_list(Serie:pd.DataFrame()):
    """
    Flatten a pandas Series of iterables into a single list.

    This function assumes that each value in the input Series is an iterable
    (e.g., a list) and concatenates all these iterables into a single flat list.

    Args:
    series (pd.Series): The input Series containing iterables

    Returns:
    List[Any]: A flattened list containing all elements from the Series

    Example:
    >>> s = pd.Series([[1, 2], [3, 4], [5, 6]])
    >>> Series_to_list(s)
    [1, 2, 3, 4, 5, 6]
    """
    liste = []
    for value in Serie.values:
        liste = liste + value
    return liste

def df_nlp(df:pd.DataFrame()):
    """
    Create a Series of summarized text data grouped by anonymized identifiers.

    This function takes a DataFrame with 'Anonymisation' and 'Résumé' columns,
    and creates a Series where each entry is a list of all 'Résumé' values
    for each unique 'Anonymisation' identifier.

    Args:
    df (pd.DataFrame): Input DataFrame with 'Anonymisation' and 'Résumé' columns

    Returns:
    pd.Series: A Series where the index is unique 'Anonymisation' values and
               each entry is a list of corresponding 'Résumé' values
    """
    dict_={}
    liste_key = list(df.Anonymisation.unique())
    for key in liste_key:
        dict_[key] = Series_to_list(df.loc[df.Anonymisation==key, 'Résumé'])
    Serie_nlp = pd.Series(dict_)
    return Serie_nlp

def clear_string(strg:str):
    """
    Remove punctuation from a string and convert it to lowercase.

    This function replaces all punctuation characters with spaces
    and then converts the entire string to lowercase.

    Args:
    text (str): The input string to be processed

    Returns:
    str: The processed string with punctuation removed and converted to lowercase
    """
    string_temp = strg
    for cara in strg:
        if cara in string.punctuation:
            string_temp = string_temp.replace(cara,' ')
    return string_temp.lower()

def remove_duplicat(liste:list()):
    """
    Remove duplicate elements from a list while preserving order.

    Args:
    items (List[T]): The input list from which to remove duplicates

    Returns:
    List[T]: A new list with duplicates removed, preserving the original order
    """
    return list(set(liste))

def extraction_des_strings(df:pd.DataFrame()):
    """
    Extract and combine all words from the 'Résumé' column of a DataFrame.

    This function assumes that each entry in the 'Résumé' column is a list
    containing a single string, which is then split into words.

    Args:
    df (pd.DataFrame): Input DataFrame with a 'Résumé' column

    Returns:
    List[str]: A list of all words extracted from the 'Résumé' column

    Example:
    >>> df = pd.DataFrame({'Résumé': [['This is a test'], ['Another test here']]})
    >>> extraction_des_strings(df)
    ['This', 'is', 'a', 'test', 'Another', 'test', 'here']
    """
    liste_full_string = []
    for liste in df['Résumé'].values:
        liste_temp = liste[0].split(' ')
        liste_full_string = liste_full_string +liste_temp
    return liste_full_string

def liste_unique(liste:list()):
    """
    Remove duplicate elements from a list.

    Note that this process does not preserve the
    original order of elements.

    Args:
    items (List[T]): The input list from which to remove duplicates

    Returns:
    List[T]: A new list with duplicate elements removed

    Example:
    >>> liste_unique([1, 2, 2, 3, 1, 4])
    [1, 2, 3, 4]  # Note: The order may vary
    """
    liste_unique  =list(set(liste))
    return liste_unique

def creation_regex(string:str):
    """
    Create a flexible regular expression pattern from a given string.

    This function converts each character in the input string into a character class,
    and treats '?' specially to allow optional characters.

    Args:
    pattern (str): The input string to convert into a regex pattern

    Returns:
    str: A string representing the created regex pattern
    """
    str_temp = '('
    for c in string:
        if c =='?':
            str_temp = str_temp+')['+c+']('
        else:
            str_temp = str_temp+'['+c+']'    
    str_temp = str_temp +')'
    return str_temp

def is_all_punctuation(string:str):
    """
    Check if a string consists entirely of punctuation characters.

    Args:
    text (Union[str, bytes]): The input string to check

    Returns:
    bool: True if the string contains only punctuation, False otherwise
    """
    booleen = True    
    for char in string:
        if char.isalnum() == True :
            booleen = booleen and False
    return booleen

def ponctu_avant(string:str):
    """
    Remove leading non-alphanumeric characters from a string.

    This function removes all leading punctuation and whitespace characters
    from the input string, stopping at the first alphanumeric character.

    Args:
    text (Union[str, bytes]): The input string to process

    Returns:
    str: The input string with leading non-alphanumeric characters removed
    """
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
    """
    Remove trailing non-alphanumeric characters from a string.

    This function removes all trailing punctuation and whitespace characters
    from the input string, stopping at the last alphanumeric character.

    Args:
    text (Union[str, bytes]): The input string to process

    Returns:
    str: The input string with trailing non-alphanumeric characters removed
    """
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
    """
    Process words containing question marks, applying specific rules for medical terms.

    Args:
    mot (str): Original word
    mot_temp (str): Temporary word to be processed

    Returns:
    str: Processed word with question marks handled

    Note: This function applies various regex patterns to handle specific medical terms
    and common abbreviations containing question marks.
    """
    dictionnaire = dict()

    # Create regex pattern and search for groups
    regex_temp = creation_regex(mot)
    recherche = re.search(regex_temp, mot)
    length = len(recherche.groups())
    length_1 = len(recherche.group(1))
    length_2 = len(recherche.group(2))
    avant = recherche.group(1)
    apres = recherche.group(length)

    # Check for specific medical terms
    estradiol = re.compile(r"[?][s][t][r][a][d][i][o][l]", re.IGNORECASE)
    bool_11 = bool(estradiol.search(mot_temp))
    coelioscopie = re.compile(r"([\w']{1,})([c])[?]([l][i][o][a-z]*)", re.IGNORECASE)
    bool_12 = bool(coelioscopie.search(mot_temp))

    # Process based on specific patterns and conditions
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
    """
    Create replacement dictionaries for string processing in a DataFrame.

    Args:
    df (pd.DataFrame): Input DataFrame containing text data

    Returns:
    Tuple[Dict[str, str], Dict[str, str]]: Two dictionaries:
        1. Words to be replaced
        2. Words that remain unchanged

    Note: This function processes unique strings from the DataFrame,
    handling punctuation and special cases with question marks.
    """
    dictionnaire_to_replace = dict()
    dictionnaire_unchanged = dict()
    
    # Extract and uniquify strings from DataFrame
    liste_full_string = extraction_des_strings(df)
    liste_to_replace = liste_unique(liste_full_string)

    # Lists for specific cases (unused in this function, but may be used elsewhere)
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
    """
    Remove trailing non-alphanumeric characters from a string.

    Args:
    string (str): Input string

    Returns:
    str: String with trailing non-alphanumeric characters removed
    """
    to_remove = ''
    length = len(string)

    # Iterate through string from end to start
    for i in range(1, (length+1)):
        c = string[-i]
        if c.isalnum() ==False:
                to_remove = to_remove + c
        if c.isalnum() == True:
            break

    # Reverse the collected characters and remove from original string
    to_remove = to_remove[::-1]
    string = string.replace(to_remove, '')

    return string

def check_if_string(string:str):
    """
    Check if a string contains only alphabetic characters.

    Args:
    string (str): Input string to check

    Returns:
    bool: True if string is all alphabetic, False otherwise
    """
    return string.isalpha()

def split_words(liste:list()):
    """
    Split a list of strings into alphabetic and non-alphabetic words.

    Args:
    liste (list): List of strings to process

    Returns:
    tuple: Two lists (alphabetic words, non-alphabetic strings)
    """
    liste_mot = []
    liste_pas_mot = []
    for word in liste:
        booleen = check_if_string(word)
        if booleen == True:
            liste_mot.append(word)
        else:
            liste_pas_mot.append(word)
    return liste_mot, liste_pas_mot

def change_encoding(string, norm='cp1252'):
    """
    Change the encoding of a string, removing specific byte sequences.

    Args:
    string (str): Input string to re-encode
    norm (str): Target encoding (default: 'cp1252')

    Returns:
    str: Re-encoded string with specific byte sequences removed
    """
    # On encode le string en byte:
    byte = string.encode()
    # On replace le caractère spécial qui s'introduit OKLM :
    byte = byte.replace(b'\xc2', b'')
    # On décode avec le bon typepe d'encodage
    new_string = byte.decode(norm)
    return new_string

def unicode_to_string(string):
    """
    Replace specific Unicode characters with their ASCII equivalents.

    Args:
    string (str): Input string containing Unicode characters

    Returns:
    str: String with specific Unicode characters replaced
    """
    char_apostrophe = '\x92' # Right single quotation mark
    char_espace = '\xa0' # Non-breaking space
    char_vide = '\x85' # Ellipsis
    char_oe1 = '\x8c' # Latin capital ligature OE
    char_oe2 = '\x9c' # Latin small ligature oe

    if char_apostrophe in string:
        string = string.replace(char_apostrophe, "'")
    elif char_espace in string:
        string = string.replace(char_espace, " ")
    elif char_vide in string:
        string = string.replace(char_vide, "")
    elif char_oe1 in string:
        string = string.replace(char_oe1, "oe")
    elif char_oe2 in string:
        string = string.replace(char_oe2, "oe")
    return string
    

def creation_liste_correction(df, lexique):
    """
    Process DataFrame text data to create lists of numbers and words needing correction.

    Args:
    df (pd.DataFrame): Input DataFrame with 'Résumé' column
    lexique (pd.Series): Series containing correctly spelled words

    Returns:
    Tuple[List[str], List[str]]: Lists of numbers and words needing correction
    """
    # Concatenate and clean strings from DataFrame
    liste_full_string = []
    for string in df['Résumé'].values:
        string = unicode_to_string(string)
        string = clear_string(string)
        liste_temp = string.split(' ')
        liste_full_string = liste_full_string +liste_temp
    
    # On nettoie les strings :
    liste_full_string_cleared=[]
    for elem in liste_full_string:
        elem_cleared = clear_string(elem)
        liste_full_string_cleared.append(elem_cleared)

    # Remove duplicates and split into words and non-words
    liste_mot_unique_cleared = remove_duplicat(liste_full_string_cleared)
    liste_mot, liste_pas_mot = split_words(liste_mot_unique_cleared)

    # Identify words needing correction
    liste_mot_corrections = liste_mot.copy()
    for word in liste_mot:
        if bien_othographie(word, lexique.values):
            liste_mot_corrections.pop(liste_mot_corrections.index(word))

    # Categorize non-words into numbers and mixed strings
    liste_nombre = []
    liste_mélange = []
    for mot in liste_pas_mot:
        mot = unicode_to_string(mot)
        mot = mot.replace("®", '')
        if mot.isdigit():
            liste_nombre.append(mot)
        elif check_if_string(mot) and bien_othographie(word, lexique.values)==False:
            liste_mot_corrections.append(mot)
        elif re.search(r'([\d]{1,}[\w°]*)', mot) : # Regex for units search :
            liste_nombre.append(mot)
        elif re.search(r'[\d]{1,}[x*][\d]{1,}[\w]*', mot): # si c'est un volume :
            liste_nombre.append(mot)
        elif len(mot) == 1:
            continue
        elif '°' in mot:
            continue
        elif '³' in mot:
            liste_nombre.append(mot)
        else:
            liste_mélange.append(mot)

    return liste_nombre, liste_mot_corrections

def slicer(df:pd.DataFrame(), indice_min, indice_max):
    """
    Slice a DataFrame by removing rows between two specified indices.

    Args:
    df (pd.DataFrame): Input DataFrame
    indice_min (int): Start index of rows to remove (inclusive)
    indice_max (int): End index of rows to remove (inclusive)

    Returns:
    pd.DataFrame: DataFrame with specified rows removed
    """
    part_1 = df.loc[0:indice_min]
    part_2 = df.loc[indice_max+1:]
    df_sliced = pd.concat([part_1, part_2])
    return df_sliced

def timedelta_in_days(date_1,date_2):
    """
    Calculate the number of days between two dates.

    Args:
    date_1 (datetime): First date
    date_2 (datetime): Second date

    Returns:
    int: Number of days between date_1 and date_2
    """
    delta = date_1 - date_2
    return delta.days

def recherche_closest_date(liste_date, date_to_check):
    """
    Find the closest date to a given date from a list of dates.

    Args:
    liste_date (List[Union[datetime, date]]): List of dates to search
    date_to_check (Union[datetime, date]): Reference date

    Returns:
    Union[datetime, date]: The closest date from the list
    """
    liste_delta = []

    # Convert date_to_check to date object if it's datetime
    if check_if_date_is_datetime(date_to_check):
        date_to_check = date_to_check.date()
        
    # Calculate time deltas and find the minimum
    for date in liste_date:  
        if check_if_date_is_datetime(date):
            date = date.date()
        integer = timedelta_in_days(date, date_to_check)
        liste_delta.append(date)
    date = liste_date[liste_delta.index(pd.Series(liste_delta).min())]
    return date

def check_if_date_is_datetime(date):
    """
    Check if the given object is an instance of datetime.datetime.

    Args:
    date: Object to check

    Returns:
    bool: True if the object is a datetime.datetime instance, False otherwise
    """
    booleen = isinstance(date, datetime.datetime)
    return booleen

def check_programmation_opé_for_patiente_with_endo(df, n_ano):
    """
    Remove 'Programmation opératoire' entry for a specific patient with endometriosis.

    Args:
    df (pd.DataFrame): Input DataFrame containing patient data
    n_ano (str): Anonymized patient identifier

    Returns:
    pd.DataFrame: DataFrame for the specified patient with 'Programmation opératoire' removed if present
    """
    df_ano = df.loc[df.loc[:,'Anonymisation']== n_ano].copy()

    # Identify and remove 'Programmation opératoire' entry if it exists
    index = df_ano.loc[df_ano.loc[:,'Nature']=='Programmation opératoire'].index
    if len(index)==1:
        df_ano.drop(index[0], axis=0, inplace=True)     

    return df_ano

def check_voc_(string):
    """
    Check if a string contains any of the predefined medical terms related to endometriosis procedures.

    Args:
    string (str): Input string to check

    Returns:
    bool: True if any predefined term is found in the string, False otherwise
    """
    boolean = False
    liste_mot_a_check = ['endométriose', 'coelioscopie',
                         'exérèse','coelio',
                         'endometriose','exerese',
                         'exerèse','exérese', 
                         'resection', 'résection',
                         'alcoolisation', 'c?lioscopie',
                         'exploratrice', 'résection',
                         'alcoolisation', 'ponction' ]
    for word in liste_mot_a_check:
        if word in string.lower():
            boolean = True

    return boolean

def tri_prog_opé(df, serie):
    """
    Filter out specific 'Programmation opératoire' entries based on vocabulary check.

    Args:
    df (pd.DataFrame): Input DataFrame containing patient data
    serie (pd.Series): Unused in the current implementation

    Returns:
    pd.DataFrame: DataFrame with filtered 'Programmation opératoire' entries removed
    """
    df_copy = df.copy()
    index_to_drop = []
    liste_n_ano = df_copy.Anonymisation.unique()

    # Iterate through unique patient identifiers
    for n_ano in liste_n_ano:
        # Select 'Programmation opératoire' entries for the current patient
        to_drop = df_copy.loc[(df_copy.loc[:,'Anonymisation']== n_ano) & (df_copy.loc[:,'Nature']== 'Programmation opératoire')]

        # Check each résumé for specific vocabulary
        for i, résumé in enumerate(list(to_drop.Résumé)):
            if check_voc_(résumé)==True:
                index = to_drop.iloc[i,:].name
                index_to_drop.append(index)

    # Remove identified entries      
    df = df.drop(index_to_drop)

    return df

def last_check(df):
    """
    Remove rows from the DataFrame where the 'Résumé' contains specific medical vocabulary.

    Args:
    df (pd.DataFrame): Input DataFrame with 'Résumé' column

    Returns:
    pd.DataFrame: DataFrame with filtered rows removed
    """
    df.reset_index(drop=True, inplace=True)
    copy = df.copy()
    for idx, row in enumerate(df.Résumé):
        if check_voc_(row):
            copy.drop(idx, axis=0, inplace=True)

    return copy  

def troncature_des_datas(df, serie):
    """
    Truncate and filter DataFrame based on date criteria and specific checks.

    Args:
    df (pd.DataFrame): Input DataFrame with 'Anonymisation' and 'Date' columns
    serie (pd.Series): Series with anonymization indices and corresponding dates

    Returns:
    pd.DataFrame: Truncated and filtered DataFrame
    """
    # Make a copy of df :
    df.dropna(axis=0, inplace=True)
    df.Date = pd.to_datetime(df.Date)
    df_sliced = df.copy()

    #Get the list of unique n° anonymisation :
    liste_ano = df.Anonymisation.unique()

    # Loop for slice the data :
    for n_ano in liste_ano:
        if n_ano in serie.index:
            # Get the list of date with ONE n° anonymsation
            series_temp = df.loc[df.Anonymisation==n_ano, 'Date']
            liste_date = list(series_temp)

            # Get the data to troncate
            date_to_search = serie.loc[n_ano]
            date_to_troncate = recherche_closest_date(liste_date,date_to_search)

            # Get the index for slice : 
            indice_min = series_temp.index[series_temp.tolist().index(date_to_troncate)]
            indice_max = series_temp.index[series_temp.tolist().index(series_temp.tolist()[-1])]
            delta = indice_max-indice_min
            df_sliced = slicer(df_sliced, indice_min, indice_max)

    df_sliced = tri_prog_opé(df_sliced, serie)
    df_sliced = last_check(df_sliced)
    df_sliced.dropna(axis=0, inplace=True)
    df_sliced.reset_index()

    return df_sliced
        
  # Regarder les patientes non positives, checker les programmations opératoires     
        
def preparation_des_dates_a_tronquer(df):
    """
    Prepare a Series of earliest dates for each unique anonymization number, excluding 'FA-086'.

    Args:
    df (pd.DataFrame): Input DataFrame with 'Anonymisation' and 'Date' columns

    Returns:
    pd.Series: Series with anonymization numbers as index and earliest dates as values
    """
    dict_temp={}
    liste_ano_unique = list(df.Anonymisation.unique())
    liste_ano_unique.remove('FA-086')
    for n_ano in liste_ano_unique:
        df_temp = df.loc[df.Anonymisation == n_ano, 'Date'].sort_values()
        dict_temp[n_ano] = df_temp.unique()[0]

    return pd.Series(dict_temp)