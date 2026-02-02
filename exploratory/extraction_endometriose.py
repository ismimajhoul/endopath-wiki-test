'''
author : Maxime Mock
date : 04/10/2022 
Le but de ce srcipt est de permettre d'éditer un dataframe compilant les informations lié au diagnostic d'endométriose des patientes de la cohorte

'''
import pandas as pd
import datetime



def endométriose(liste:list()):
    """
    Take a list of pathologie, return a list of endometriosis' pathologie
    """
    liste_endo = []
    for pathologie in liste:
        # Check if 'endométriose' is in the pathology name (case-insensitive)
        if 'endométriose' in pathologie.lower():
            liste_endo.append(pathologie)
    return liste_endo

def create_series_of_endo(df:pd.DataFrame(), liste_endo:list(), col_1:str, col_2):
    """
    Create a pandas Series by concatenating data from a DataFrame based on endometriosis pathologies.
 
    Args:
        df (pd.DataFrame): The input DataFrame containing the data.
        liste_endo (list): A list of endometriosis pathologies to filter by.
        col_1 (str): The name of the column to filter on (containing pathology names).
        col_2 (str): The name of the column to select data from.
 
    Returns:
        pd.Series: A concatenated Series containing data from col_2 where col_1 matches the endometriosis pathologies.
    """
    liste_serie = []
    for endo in liste_endo:
        # Filter the DataFrame for each endometriosis pathology
        serie_temp = df.loc[df.loc[:,col_1] == endo, col_2]
        liste_serie.append(serie_temp)
    # Concatenate all the temporary Series
    serie = pd.concat(liste_serie, axis=0)
    return serie

def liste_from_serie_to_list(serie:pd.Series(dtype=object)):
    """
    Convert a pandas Series to a Python list.
    """
    return list(serie)

def get_non_unique_list(liste:list()):
    """
    Separate a list into non-unique and unique elements.

    Args:
        liste (list): The input list to be processed.

    Returns:
        tuple[list, list]: A tuple containing two lists:
            - The first list contains non-unique elements (appearing more than once in the input).
            - The second list contains unique elements (appearing exactly once in the input).

    Note:
        The order of elements in the returned lists may not match the order in the input list.
    """
    non_unique = []
    unique = []
    for elem in liste:
        if liste.count(elem)>1 and elem not in non_unique:
            # Element appears more than once and hasn't been added to non_unique yet
            non_unique.append(elem)
        elif liste.count(elem)==1:
            # Element appears exactly once  
            unique.append(elem)    
    return non_unique, unique

def remove_values_from_list(the_list, val):
    """
    Remove all occurrences of a specific value from a list.

    Args:
        the_list (List[Any]): The input list from which to remove values.
        val (Any): The value to be removed from the list.

    Returns:
        List[Any]: A new list with all occurrences of 'val' removed.

    Note:
        This function does not modify the original list but returns a new one.
    """
    return [value for value in the_list if value != val]

def remove_exception(liste_to_process, list_of_exception):
    """
    Remove all elements from liste_to_process that are present in list_of_exception.

    Args:
        liste_to_process (List[Any]): The input list to be processed.
        list_of_exception (List[Any]): The list of elements to be removed from liste_to_process.

    Returns:
        List[Any]: A new list with all elements from list_of_exception removed from liste_to_process.

    Note:
        This function assumes the existence of a remove_values_from_list function.
        It creates a new list for each element in list_of_exception, which may be inefficient for large lists.
    """
    for elem in list_of_exception:
        # Remove each exception element from the list
        liste_to_process = remove_values_from_list(liste_to_process, elem)
    return liste_to_process

def contract_data_getting_unique_index(df:pd.DataFrame(), col):
    """
    Create a DataFrame with a unique index by contracting data in the specified column for duplicate indices.

    Args:
        df (pd.DataFrame): The input DataFrame with potentially non-unique index.
        col (str): The name of the column to contract for duplicate indices.

    Returns:
        pd.DataFrame: A new DataFrame with a unique index. For duplicate indices, the data in the specified
                      column is combined into a list.

    Note:
        This function assumes the existence of a get_non_unique_list function.
    """
    dict_temp = {}
    non_unique, unique = get_non_unique_list(list(df.index))
    # Combine data for non-unique indices
    for row in non_unique:
        dict_temp[row] = list(df.loc[row, col])

    # Create a Series with the combined data
    df_unique = pd.Series(dict_temp, name='CIM_LIBELLE_COMPLET')

    # Concatenate the unique rows with the combined data
    df_final = pd.concat([df.loc[unique], pd.DataFrame(df_unique)], axis=0)

    return df_final
    
def get_data_endo(df_1:pd.DataFrame(), df_2:pd.DataFrame()):
    """
    Process PMSI data to collect information where endometriosis is found.

    Args:
        df_1 (pd.DataFrame): DataFrame containing diagnosis information.
        df_2 (pd.DataFrame): DataFrame containing surgical operation information.

    Returns:
        pd.DataFrame: A processed DataFrame containing endometriosis-related data with columns:
                      'Anonymisation', 'Date', 'Acte', and 'Diagnostic'.

    Note:
        This function assumes the existence of several helper functions:
        endométriose, create_series_of_endo, contract_data_getting_unique_index, remove_exception.
    """
    # Exception for NUMANORSS that don't match (women who already had endometriosis before HCL):
    liste_exception = [323, 127, 41, 31]
    
    
    # Get list of endometriosis diagnoses
    liste_pathologies = list(df_1.loc[:,'CIM_LIBELLE_COMPLET'].unique())
    liste_endo = endométriose(liste_pathologies)

    # Collect endometriosis diagnoses (NUMANROSS and CIM_LIBELLE_COMPLET) from diagnosis
    df_diag_numanorss_endo = create_series_of_endo(df_1, liste_endo, 'CIM_LIBELLE_COMPLET', ['NUMANORSS', 'CIM_LIBELLE_COMPLET'])
    diagnosis = contract_data_getting_unique_index(df_diag_numanorss_endo.set_index('NUMANORSS'),'CIM_LIBELLE_COMPLET')
    
    # Collect surgical operations
    unique = list(df_diag_numanorss_endo.NUMANORSS.unique())
    unique = remove_exception(unique, liste_exception)
    surgical = create_series_of_endo(df_2, unique, 'NUMANORSS', ['NUM_INCLUSION', 'DREALAN','CAM_LIBELLE_COMPLET', 'NUMANORSS'])
    
    # Change index for matching
    surgical.set_index('NUMANORSS', inplace=True)
    
    # Matching
    surgical['Diagnostic'] = diagnosis
    
    # Reshape df
    surgical.sort_values('NUM_INCLUSION', inplace=True)
    surgical.columns = ['Anonymisation', 'Date', 'Acte', 'Diagnostic']

    # Add exception cases
    date1 = datetime.datetime.strptime('2017-11-23', '%Y-%m-%d')
    date2 = datetime.datetime.strptime('2017-07-18', '%Y-%m-%d')
    date3 = datetime.datetime.strptime('2018-03-07', '%Y-%m-%d')
    date4 = datetime.datetime.strptime('2017-06-29', '%Y-%m-%d')
    dict_exception = {'Anonymisation':['GV-076','AM-038','GC-021','FA-086'], 
                      'Date':[date1, date2, date3, date4], 
                      'Acte':['-','-','-','-'], 
                      'Diagnostic':['Endométriose depuis 2000','Endométriose depuis 2014',"Antécédent d'endométriose",'Endométriose depuis 2011']}
    surgical = pd.concat([surgical, pd.DataFrame(dict_exception)])
    surgical.reset_index(drop=True, inplace=True)
    
    return surgical