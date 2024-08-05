import os
from pprint import pprint
from matplotlib import pyplot as plt
import pandas as pd


def saveData2Csv():
    # Load the CSV data into a DataFrame
    df = pd.read_csv('priorityChanges/changesOfPriorities.csv')
    df = df.fillna('No Registered')

    monthsColumns = df.columns[2:]
    uniqueRs = df['shop_id'].unique()
    data = {}

    # Process the data
    for rs in uniqueRs:
        rsData = df[df['shop_id'] == rs]
        pais = rsData['country_code'].values[0]
        for i, monthPriority in enumerate(monthsColumns[:-1]):
            month = monthPriority.split('_')[0]
            presentPriority = rsData[monthPriority].values[0]
            nextPriority = rsData[monthsColumns[i+1]].values[0]
            if pais not in data:
                data[pais] = {}
            if presentPriority not in data[pais]:
                data[pais][presentPriority] = {}
            if month not in data[pais][presentPriority]:
                data[pais][presentPriority][month] = {}
            if presentPriority == nextPriority:
                if 'Stays' not in data[pais][presentPriority][month]:
                    data[pais][presentPriority][month]['Stays'] = 0
                data[pais][presentPriority][month]['Stays'] += 1
            else:
                if nextPriority not in data[pais][presentPriority][month]:
                    data[pais][presentPriority][month][nextPriority] = 0
                data[pais][presentPriority][month][nextPriority] += 1

    # Print the data (optional)
    pprint(data)

    rows = []
    for country, priorities in data.items():
        for priority, months in priorities.items():
            for month, transitions in months.items():
                for key, value in transitions.items():
                    row = {'Country': country, 'Priority': priority,
                           'Month': month, 'ChangeTo': key, '# Rs': value}
                    rows.append(row)

    df_data = pd.DataFrame(rows)

    # Save the DataFrame to a CSV file
    df_data.to_csv(
        'priorityChanges/processed_priority_changes.csv', index=False)


def df_to_dict():
    loadedData = pd.read_csv('priorityChanges/processed_priority_changes.csv')
    data = {}
    for index, row in loadedData.iterrows():
        country = row['Country']
        priority = row['Priority']
        month = row['Month']
        changeTo = row['ChangeTo']
        rs = row['# Rs']
        if country not in data:
            data[country] = {}
        if priority not in data[country]:
            data[country][priority] = {}
        if month not in data[country][priority]:
            data[country][priority][month] = {}
        data[country][priority][month][changeTo] = rs
    return data


def makePercentages(data):
    for country, priorities in data.items():
        for priority_initial, months_data in priorities.items():
            for month, transitions in months_data.items():
                total = sum(transitions.values())
                for key in transitions:
                    transitions[key] = (transitions[key] / total) * 100
    return data


def priorityFolderName(priority):
    if priority == 'Priority 1':
        return 'p1'
    if priority == 'Priority 2':
        return 'p2'
    if priority == 'Priority 3':
        return 'p3'
    if priority == 'Priority 4':
        return 'p4'
    if priority == 'Priority 5':
        return 'p5'
    if priority == 'Priority 6':
        return 'p6'
    if priority == 'No Registered':
        return 'noReg'


def makePlots():
    Data = df_to_dict()
    Data = makePercentages(Data)
    priority_colors = {
        'Priority 1': 'blue',
        'Priority 2': 'orange',
        'Priority 3': 'green',
        'Priority 4': '#1eaae7',
        'Priority 5': 'purple',
        'Priority 6': 'brown',
        'Stays': '#e7411e',
        'No Registered': 'gray'
    }

    # Iterar sobre cada país en el diccionario
    for country, priorities in Data.items():
        # Iterar sobre cada prioridad inicial
        for priority_initial, months_data in priorities.items():
            output_folder = f'plotsJuneDynamic/{country}/{
                priorityFolderName(priority_initial)}'
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            plt.figure(figsize=(12, 8))  # Tamaño de la figura

            months = list(months_data.keys())
            priority_labels = list(months_data[months[0]].keys())

            # Crear un array de valores por prioridad destino para cada mes
            for priority_target in priority_labels:
                values = [months_data[month].get(
                    priority_target, 0) for month in months]
                # Obtener color, 'black' si no está definido
                color = priority_colors.get(priority_target, 'black')
                plt.plot(months, values, label=priority_target,
                         linewidth=4, color=color)

                # Añadir etiquetas de texto a cada punto
                for i in range(len(months)):
                    plt.text(months[i], values[i] + 0.3, f'{values[i]:.2f}%', color=color, ha='center', va='bottom', bbox=dict(
                        facecolor='white', edgecolor=color, boxstyle='round,pad=0.3'))

            # Configuraciones de la gráfica
            plt.xticks(rotation=45)
            plt.xlabel('Month')
            plt.ylabel('% of change')
            plt.title(f'''Flow of priority changes for {
                      priority_initial} in {country}''')
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
                       ncol=4, frameon=False, prop={'size': 12})
            plt.grid(True)
            plt.tight_layout()

            file_name = f'changesPriority.png'.replace(" ", "_")
            if priorityFolderName(priority_initial) != 'p6':
                plt.savefig(os.path.join(output_folder, file_name))
            plt.close()

            # Mostrar la gráfica
            # plt.show()
makePlots()
