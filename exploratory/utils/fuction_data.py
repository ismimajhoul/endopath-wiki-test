import pandas as pd

def convert_date_in_df(df:pd.DataFrame(), liste_colonne_date):
    for liste in liste_colonne_date:
        for i, elem in df.loc[:,liste].iteritems():
            elem_checked = pd.to_datetime(elem, format='%Y-%m-%d %H:%M:%S')
            df.loc[i,liste] = elem_checked
    return df

def is_datetime(value_to_check):
    boolean = isinstance(value_to_check, datetime.datetime)
    return boolean

def is_real_datetime(value):
    boolean = is_datetime(value)
    boolean2 = pd.isnull(value) != True
    return boolean and boolean2


def liste_date(df):
    '''
    Création d'une liste qui contient toutes les colonnes avec le mot 'Date'
    --------------------
    Input 
    dataframe : pd.DataFrame()
    ---------------------
    Return 
    liste_colonne_date : list()
    '''
    liste_colonne_date = []
    for tuple_col in Multiindex:
        if 'Date' in tuple_col[1]:
            liste_colonne_date.append(tuple_col)
    return liste_colonne_date

def make_serie_date_only(df, row):
    '''
    Utilise un indice de ligne correspondant à une ligne  et un dataframe pour retourner une serie ne comprennant que des valeurs non nulles.
    ---------------
    Input:
    row : int
    df : pd.DataFrame()
    ---------------
    Return : 
    Serie: pd.Series()
    '''
    serie = df.loc[row,:]
    serie = serie.loc[serie.apply(is_real_datetime)==True]
    return serie


def check_date_get_time(datetime):
    '''
    check if a datetime get a time in the datetime
    -------------------------
    Input : 
    datetime
    -------------------------
    Return :
    boolean (True if the date get a time False if not)
    '''
    hour = datetime.hour
    mins = datetime.minute
    sec = datetime.second
    boolean = hour != 0 and mins != 0 and sec != 0
    return boolean


def comparateur_date(date, date_to_check):
    bool_1 = date.date() == date_to_check.date()
    bool_2 = date.date() == switch_date(date_to_check).date()
    return bool_1 or bool_2 

def switch_date(date):
    '''
    Création d'une fonction pour inverser les mois et les jours lorsque des dates ont été males encodées à l'ouverture du fichier
    '''
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
    # Création de la liste des colonnes contenant le mot 'Date'
    liste = liste_date(df)
    for row in df.index:
        serie_date = make_serie_date_only(df, row)
        # boolean = check_date_get_time(date_ref) valable si la première date n'est pas fiable, 1ere approximation : la 1ere date est fiable
        if len(serie_date) > 1:
            date_ref = serie_date[0]
            for date_temp in serie_date[1:]:
                if check_date_get_time(date_temp) == False:
                    if comparateur_date(date_ref, date_temp):
                        index = np.where(serie_date== date_temp)[0][0]
                        df.loc[row, serie_date.index[index]] = date_ref
    return df  