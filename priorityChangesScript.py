from pprint import pprint
import pandas as pd
import matplotlib.pyplot as plt


def saveData2Csv():
    # Load the CSV data into a DataFrame
    df = pd.read_csv('priorityChanges/changesOfPriorities.csv')
    df = df.fillna('no registered')

    monthsColumns = df.columns[2:]
    uniqueRs = df['shop_id'].unique()
    paises = df['country_code'].unique()
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
                if 'stays' not in data[pais][presentPriority][month]:
                    data[pais][presentPriority][month]['stays'] = 0
                data[pais][presentPriority][month]['stays'] += 1
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
