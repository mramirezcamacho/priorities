import os
import numpy as np
import pandas as pd
import csv

mainFolder = 'comparacionesCSVJuneDynamic'
getDataFromFolder = 'dataDynamicDivided'


def createFolders():
    paises = ['PE', 'CO', 'MX', 'CR']
    prioridades = ['1', '2', '3', '4', '5']
    for pais in paises:
        for prioridad in prioridades:
            os.makedirs(
                f'{mainFolder}/{pais}/p{prioridad}', exist_ok=True)


def get_df_from_country(country: str = 'PE'):
    # Especifica la ruta de la carpeta

    # Obtén la lista de todos los archivos en la carpeta
    file_list = os.listdir(getDataFromFolder)

    # Filtra la lista para incluir solo archivos (no directorios)
    file_list = [f for f in file_list if os.path.isfile(
        os.path.join(getDataFromFolder, f))]
    country_list = [f for f in file_list if country.upper() in f]
    countryDF = {}
    for dataFile in country_list:
        if "Old" in dataFile:
            countryDF['old'] = pd.read_csv(
                os.path.join(getDataFromFolder, dataFile))
        else:
            countryDF['new'] = pd.read_csv(
                os.path.join(getDataFromFolder, dataFile))
    return countryDF


def comparation_per_column(priority: str = '5', column: str = 'daily_online_hours', country: str = 'PE'):
    df_dict = get_df_from_country(country)
    old_df = df_dict['old']
    new_df = df_dict['new']
    months = sorted(old_df['month_'].unique())

    # Filtrar por prioridad
    priorityNumber = int(priority)
    if priorityNumber == 1:
        old_df = old_df[old_df['priority'] == 'Priority 1']
        new_df = new_df[new_df['priority'] == 'New Rs']
        priority = 'Old_1_vs_New_Rs'
    elif priorityNumber == 2:
        old_df = old_df[old_df['priority'] == 'Priority 2']
        new_df = new_df[new_df['priority'] == 'Priority 1']
        priority = 'Old_2_vs_New_1'
    elif priorityNumber == 3:
        old_df = old_df[old_df['priority'] == 'Priority 3']
        new_df = new_df[new_df['priority'] == 'Priority 2']
        priority = 'Old_3_vs_New_2'
    elif priorityNumber == 4:
        old_df = old_df[old_df['priority'] == 'Priority 4']
        new_df = new_df[new_df['priority'] == 'Priority 4']
        priority = 'priority_4'
    elif priorityNumber == 5:
        old_df = old_df[old_df['priority'] == 'Priority 5']
        new_df = new_df[new_df['priority'] == 'Priority 5']
        priority = 'priority_5'

    comparation_results = []

    for month in months:
        old_month_df = old_df[old_df['month_'] == month]
        new_month_df = new_df[new_df['month_'] == month]

        if not old_month_df.empty and not new_month_df.empty:
            old_value = old_month_df[column].values[0] if len(
                old_month_df[column].values) > 0 else np.nan
            new_value = new_month_df[column].values[0] if len(
                new_month_df[column].values) > 0 else np.nan
            comparation_results.append(
                {'month': month, 'Old priority': old_value, 'New priority': new_value})

    DFcomparation = pd.DataFrame(comparation_results)
    fileName = f'{mainFolder}/{country}/p{
        str(priorityNumber)}/comparacion_{column}.csv'
    DFcomparation.to_csv(fileName, index=False)
    with open(fileName, 'r', newline='') as file:
        reader = csv.reader(file)
        existing_data = list(reader)

    new_row = [column.replace('_', ' ').capitalize(), priority]
    existing_data.insert(0, new_row)

    with open(fileName, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(existing_data)
        print(f"Comparación guardada en '{fileName}'")

    return DFcomparation


def unificarCSV(pais: str, prioridad: str):
    carpeta = f'{mainFolder}/{pais}/p{prioridad}'
    archivo_salida = f'total_{pais}_p{prioridad}.csv'

    # Lista para almacenar todos los datos de los archivos CSV
    datos_combinados = []

    # Obtener la lista de archivos CSV en la carpeta
    archivos_csv = [archivo for archivo in os.listdir(
        carpeta) if archivo.endswith('.csv')]

    # Leer cada archivo CSV y combinar datos
    for archivo in archivos_csv:
        ruta_archivo = os.path.join(carpeta, archivo)
        with open(ruta_archivo, 'r', newline='') as file:
            reader = csv.reader(file)
            datos_archivo = list(reader)
            datos_combinados.extend(datos_archivo)
            # Agregar una fila vacía entre archivos
            datos_combinados.append([])

    # Escribir datos combinados en un solo archivo CSV
    ruta_archivo_salida = os.path.join(carpeta, archivo_salida)
    with open(ruta_archivo_salida, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(datos_combinados)


# Ejemplo de uso
def start():
    paises = ['PE', 'CO', 'MX', 'CR']
    prioridades = ['1', '2', '3', '4', '5']
    columns = ['daily_orders', 'orders_per_eff_online', 'daily_online_hours',
               'b_cancel_rate', 'bad_rating_rate', "healthy_stores", 'imperfect_orders', 'ted', 'ted_gmv', 'exposure_uv', 'b_p1p2', "r_burn", "r_burn_gmv", "r_burn_per_order", "b2c_total_burn", "b2c_gmv", "b2c_per_order"]
    a = ['eff_online_rs', 'daily_orders', 'imperfect_order_rate']
    columns = columns + a

    # add exposure and p1p2 etc
    createFolders()
    for pais in paises:
        for prioridad in prioridades:
            for columna in columns:
                comparation_per_column(prioridad, columna, pais)
            unificarCSV(pais, prioridad)


start()
