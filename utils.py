import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    print("--> Tratando datos NaN")
    df = tratamiento_columnas_nan(df)

    print("--> Tratando columnas nominales")
    df = tratamiento_columnas_nominales(df)

    print("--> Tratamiento de columnas booleanas")
    df = tratamiento_booleanos(df)

    print("--> Rellenando valores NaN con la media")
    df = tratamiento_columnas_numericas(df)

    print("--> Unificando las lineas por cliente")
    df = combinar_filas_por_cliente(df)

    print("--> Escalando los valores")
    df = escalado_datos(df)
    return df


def tratamiento_columnas_nan(df: pd.DataFrame, threshold=80) -> pd.DataFrame:
    """
    Tratamiento de valores NaN:
        - Drop todas las columnas que tengas todos los valores en NaN
        - Drop todas las columnas que tengas mas de un threshold de valores NaN (default threshold 80)
    """
    df_nuevo = df.dropna(axis=1, how="all")
    df_nulos = nan_percentaje(df_nuevo, threshold)
    df_nuevo = df_nuevo.drop(columns=df_nulos.index)
    return df_nuevo


def escalado_datos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Escalamos todos los valores numericos
    """
    df_nuevo = pd.DataFrame(df)
    scaler = StandardScaler()
    numerical_columns = df_nuevo.select_dtypes(include=['number']).columns
    df_nuevo[numerical_columns] = scaler.fit_transform(
        df_nuevo[numerical_columns])

    return df_nuevo


def tratamiento_columnas_numericas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rellenar los valores NaN de las columnas numericas con la media
    """
    df_nuevo = pd.DataFrame(df)

    numerical_columns = df_nuevo.select_dtypes(include=['number']).columns
    imputer = SimpleImputer(strategy='mean')
    df_nuevo[numerical_columns] = imputer.fit_transform(
        df_nuevo[numerical_columns])

    return df_nuevo


def tratamiento_columnas_nominales(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tratamiento de columnas nominales
        - Frequency map para las que tengas valores que tenga sentido mapear a numeros
        - One Hot Encoding en las que no tenaga sentido convertir en valors numericos,
        para estas columnas, primero rellenaremos los valores NaN con el valor mas repetido
        y luego realizaremos el OHE.
    """
    frequency_map = {
        'very_low': 0,
        'low': 1,
        'moderate_low': 2,
        'moderate': 3,
        'moderate_high': 4,
        'high': 5,
        'very_high': 6
    }
    df_nuevo = pd.DataFrame(df)
    for col in ['Infraction_CLH', 'Base_67254', 'Infraction_TEN']:
        df_nuevo[col] = df_nuevo[col].map(frequency_map)

    df_nuevo['Infraction_DQLY'] = df_nuevo['Infraction_DQLY'].fillna(
        df_nuevo['Infraction_DQLY'].value_counts().head(1))

    columns_to_encode = ['Infraction_YFSG', 'Infraction_DQLY']
    df_nuevo = pd.get_dummies(df_nuevo, columns=columns_to_encode)
    return df_nuevo


def tratamiento_booleanos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tratamiento de columnas booleanas:
        - Rellenar los valores NaN de las columnas booleanas con el valor mas repetido
        - Booleanos a int
    """
    df_nuevo = pd.DataFrame(df)
    boolean_columns = df_nuevo.select_dtypes('bool').columns

    for col in boolean_columns:
        df_nuevo[col] = df_nuevo[col].fillna(
            df_nuevo[col].value_counts().head(1))
        df_nuevo[col] = df_nuevo[col].astype(int)

    return df_nuevo


def nan_percentaje(df, threshold=0):
    porcentaje_nulos = df.isnull().mean() * 100
    porcentaje_nulos = porcentaje_nulos[porcentaje_nulos > threshold]
    return porcentaje_nulos


def combinar_filas_por_cliente(df: pd.DataFrame) -> pd.DataFrame:
    """
    Unificamos todas las transacciones de cada cliente en una linea
    """
    numerical_columns = df.select_dtypes(include=['number']).columns
    client_dict = {col: 0.0 for col in numerical_columns}
    client_dict['ID'] = ""
    df_dict = {col: [] for col in numerical_columns}
    df_dict['ID'] = []
    client = df['ID'].iloc[0]

    for i, row in df.iterrows():
        if client != row['ID']:
            for key in client_dict:
                df_dict[key].append(client_dict[key])
            client_dict = {col: 0.0 for col in numerical_columns}
            client = row['ID']
        row_dict = row.to_dict()
        client_dict['ID'] = row['ID']
        for key in client_dict.keys():
            client_dict[key] += row_dict[key]
    for key in client_dict:
        df_dict[key].append(client_dict[key])

    new_df = pd.DataFrame(df_dict)
    return new_df
