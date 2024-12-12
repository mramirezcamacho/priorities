import os
import numpy as np
import pandas as pd
import csv
from centralizedData import dividedData, comparationData


folderToGetData = dividedData
folderToSendData = comparationData


def createFolders(paises, prioridades):
    for pais in paises:
        for prioridad in prioridades:
            if prioridad == 'New Rs':
                prioridad = '0'
            os.makedirs(
                f'{folderToSendData}/{pais}/p{prioridad}', exist_ok=True)


def comparacionAllPriorities(country: str, priorities: list, columns: list):
    df_data = pd.read_csv(f'{folderToGetData}/{country}_data.csv')
    months = sorted(df_data['month_'].unique())
    for columna in columns:
        newDF = pd.DataFrame(columns=['month', 'New Rs', 'Priority 1',
                                      'Priority 2', 'Priority 3', 'Priority 4', 'Priority 5'])

        for month in months:

            row2append = {'month': month}
            for priority in priorities:
                if priority == 'New Rs':
                    priority = '0'
                meanWhile = df_data[(df_data['month_'] == month)]
                if priority == '0':
                    meanWhile = meanWhile[(
                        meanWhile['priority'] == 'New Rs')]
                else:
                    meanWhile = meanWhile[(
                        meanWhile['priority'] == f'Priority {priority}')]
                meanWhile = meanWhile[meanWhile[columna].notna()]
                value = meanWhile[columna].values[0]
                if columna == 'eff_online_rs':
                    value = int(value)
                if priority == '0':
                    row2append['New Rs'] = value
                else:
                    row2append[f'''Priority {
                        priority}'''] = value

            newDF.loc[len(newDF)] = row2append
        os.makedirs(f'{folderToSendData}/{country}/All', exist_ok=True)
        fileName = f'{folderToSendData}/{country}/All/{columna}.csv'
        newDF.to_csv(fileName, index=False)
        with open(fileName, 'r', newline='') as file:
            reader = csv.reader(file)
            existing_data = list(reader)
        new_row = [columna.replace('_', ' ').capitalize(), 'Priority 0', 'Priority 1',
                   'Priority 2', 'Priority 3', 'Priority 4', 'Priority 5']
        existing_data.insert(0, new_row)
        with open(fileName, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(existing_data)
            print(f"Comparación guardada en '{fileName}'")


def comparation_per_column(priority: str, column: str, country: str):
    df_data = pd.read_csv(f'{folderToGetData}/{country}_data.csv')
    months = sorted(df_data['month_'].unique())
    # Filtrar por prioridad
    try:
        priorityNumber = int(priority)
    except:
        priorityNumber = 0
    if priorityNumber == 0:
        realDF = df_data[df_data['priority'] == 'New Rs']
    else:
        realDF = df_data[df_data['priority'] == f'Priority {priorityNumber}']

    comparation_results = []

    for month in months:
        month_df = realDF[realDF['month_'] == month]

        if not month_df.empty:
            real_value = month_df[column].values[0] if len(
                month_df[column].values) > 0 else np.nan
            comparation_results.append(
                {'month': month, 'Priority': real_value})

    if column == 'eff_online_rs':
        # put values int
        for i in range(len(comparation_results)):
            comparation_results[i]['Priority'] = int(
                comparation_results[i]['Priority'])

    DFcomparation = pd.DataFrame(comparation_results)
    fileName = f'{folderToSendData}/{country}/p{
        str(priorityNumber)}/{column}.csv'

    DFcomparation.to_csv(fileName, index=False)
    with open(fileName, 'r', newline='') as file:
        reader = csv.reader(file)
        existing_data = list(reader)

    new_row = [column.replace('_', ' ').capitalize(),
               f'Priority {priorityNumber}']
    existing_data.insert(0, new_row)

    with open(fileName, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(existing_data)
        print(f"Comparación guardada en '{fileName}'")

    return DFcomparation


def unificarCSV(pais: str, prioridad: str):
    if prioridad == 'New Rs':
        prioridad = '0'
    carpeta = f'{folderToSendData}/{pais}/p{prioridad.replace(" ", "_")}'
    archivo_salida = f'total_{pais}_p{prioridad.replace(" ", "_")}.csv'

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


def getColumns(country: str):
    df = pd.read_csv(f'{folderToGetData}/{country}_data.csv')
    return df.columns[3:]
# Ejemplo de uso


def start():
    paises = ['PE', 'CO', 'MX', 'CR']
    prioridades = ['New Rs', '1', '2', '3', '4', '5',]

    # add exposure and p1p2 etc
    createFolders(paises, prioridades)
    for pais in paises:
        columns = getColumns(pais)
        for prioridad in prioridades:
            for columna in columns:
                comparation_per_column(prioridad, columna, pais)
            unificarCSV(pais, prioridad)
        comparacionAllPriorities(pais, prioridades, columns)


if __name__ == '__main__':
    start()
