import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tratamiento de valores NaN:
        - Drop todas las columnas que tengas todos los valores en NaN
        - Drop todas las columnas que tengas mas de un 80% de valores NaN
    """
    print("--> Tratando datos NaN")
    df = df.dropna(axis=1, how="all")
    df_nulos = nan_percentaje(df, threshold=80)
    df = df.drop(columns=df_nulos.index)

    """
    Tratamiento de columnas nominales
        - Frequency map para las que tengas valores que tenga sentido mapear a numeros
        - One Hot Encoding en las que no tenaga sentido convertir en valors numericos,
        para estas columnas, primero rellenaremos los valores NaN con el valor mas repetido
        y luego realizaremos el OHE.
    """
    print("--> Tratando columnas nominales")
    frequency_map = {
        'very_low': 0,
        'low': 1,
        'moderate_low': 2,
        'moderate': 3,
        'moderate_high': 4,
        'high': 5,
        'very_high': 6
    }
    for col in ['Infraction_CLH', 'Base_67254', 'Infraction_TEN']:
        df[col] = df[col].map(frequency_map)

    df['Infraction_DQLY'] = df['Infraction_DQLY'].fillna(
        df['Infraction_DQLY'].value_counts().head(1))

    columns_to_encode = ['Infraction_YFSG', 'Infraction_DQLY']
    df = pd.get_dummies(df, columns=columns_to_encode)

    """
    Asiganar tipo a las filas:
        - Fechas a Datatime
        - Booleanos a int, antes de convertirlos en int rellenamos los valores NaN con el valor mas repetido
    """
    print("--> Asignando tipo a filas")
    df['Expenditure_AHF'] = pd.to_datetime(df['Expenditure_AHF'])

    boolean_columns = df.select_dtypes('bool').columns

    for col in boolean_columns:
        df[col] = df[col].fillna(df[col].value_counts().head(1))
        df[col] = df[col].astype(int)

    """
    Rellenar los valores NaN de las columnas numericas con la media
    """
    print("--> Rellenando valores NaN con la media")
    numerical_columns = df.select_dtypes(include=['number']).columns
    imputer = SimpleImputer(strategy='mean')
    df[numerical_columns] = imputer.fit_transform(df[numerical_columns])

    """
    Unificamos todas las transacciones de cada cliente en una linea
    """
    print("--> Unificando las lineas por cliente")
    df = combinar_filas_por_cliente(df)

    """
    Escalamos todos los valores numericos en un rango de 0 a 1
    """
    print("--> Escalando los valores")
    scaler = StandardScaler()
    numerical_columns = df.select_dtypes(include=['number']).columns
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
    return df


def nan_percentaje(df, threshold=0):
    porcentaje_nulos = df.isnull().mean() * 100
    porcentaje_nulos = porcentaje_nulos[porcentaje_nulos > threshold]
    return porcentaje_nulos


def combinar_filas_por_cliente(df: pd.DataFrame) -> pd.DataFrame:
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
