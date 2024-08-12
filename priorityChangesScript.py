import calendar
import os
from pprint import pprint
from matplotlib import pyplot as plt
import pandas as pd
from centralizedData import plotsFolder

SUPEROUTPUT = plotsFolder


def saveData2Csv():
    # Load the CSV data into a DataFrame
    df = pd.read_csv('priorityChanges/changesOfPriorities3.csv')
    df = df.fillna('No Registered')

    monthsColumns = df.columns[2:]
    uniqueRs = df['shop_id'].unique()
    data = {}

    # Process the data
    for rs in uniqueRs:
        rsData = df[df['shop_id'] == rs]
        pais = rsData['country_code'].values[0]
        for i, monthPriority in enumerate(monthsColumns[1:], start=1):
            month = monthPriority.split('_')[0]
            presentPriority = rsData[monthPriority].values[0]
            previousPriority = rsData[monthsColumns[i-1]].values[0]
            if pais not in data:
                data[pais] = {}
            if presentPriority not in data[pais]:
                data[pais][presentPriority] = {}
            if month not in data[pais][presentPriority]:
                data[pais][presentPriority][month] = {}
            if presentPriority == previousPriority:
                if 'Stays' not in data[pais][presentPriority][month]:
                    data[pais][presentPriority][month]['Stays'] = 0
                data[pais][presentPriority][month]['Stays'] += 1
            else:
                if previousPriority not in data[pais][presentPriority][month]:
                    data[pais][presentPriority][month][previousPriority] = 0
                data[pais][presentPriority][month][previousPriority] += 1
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
        month = month.capitalize()
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
            changes = set()
            for month, transitions in months_data.items():
                total = sum(transitions.values())
                for key in transitions:
                    transitions[key] = (transitions[key] / total) * 100
                    changes.add(key)
            for change in changes:
                for month, transitions in months_data.items():
                    if change not in transitions:
                        transitions[change] = 0
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


def saveTXT(folder_path: str, file_name: str, content: str):
    os.makedirs(folder_path, exist_ok=True)
    output_file = os.path.join(folder_path, file_name)
    try:
        with open(output_file, 'w') as f:
            f.write(content)
            f.close()
    except Exception as e:
        print(e)


def makePlots2():
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

    # Create a mapping of month names to their respective order
    month_order = {month: index for index,
                   month in enumerate(calendar.month_name) if month}

    # Iterate over each country in the dictionary
    for country, priorities in Data.items():
        # Iterate over each initial priority
        for priority_initial, months_data in priorities.items():
            output_folder = f'''{SUPEROUTPUT}/{country}/{
                priorityFolderName(priority_initial)}/priorityChanges'''
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            plt.figure(figsize=(12, 10.5))  # Figure size

            months = sorted(months_data.keys(), key=lambda x: month_order[x])
            priority_labels = list(months_data[months[0]].keys())

            # Create an array of values by target priority for each month
            for priority_target in priority_labels:
                values = [months_data[month].get(
                    priority_target, 0) for month in months]
                color = priority_colors.get(priority_target, 'black')
                plt.plot(months, values, label=priority_target,
                         linewidth=4, color=color)

                # Add text labels to each point
                for i in range(len(months)):
                    plt.text(months[i], values[i] + 0.3, f'{values[i]:.2f}%', color=color, ha='center', va='bottom', fontsize=16,
                             bbox=dict(facecolor='white', edgecolor=color, boxstyle='round,pad=0.3'))

            # Plot configurations
            plt.xlabel(' ')
            plt.ylabel(' ')
            plt.title(' ')
            plt.xticks(fontsize=20)  # Set X-axis tick labels font size
            plt.yticks(fontsize=20)  # Set Y-axis tick labels font size
            # plt.title(f'''Flow of priority changes for {
            #           priority_initial} in {country}''')
            plt.legend(loc='upper center', bbox_to_anchor=(
                0.5, 1.15), ncol=4, frameon=False, prop={'size': 18})
            plt.grid(True)
            plt.tight_layout()

            plt.savefig(os.path.join(output_folder, "All.png"))
            saveTXT(output_folder, "note.txt", "Example text")
            plt.close()


if __name__ == '__main__':
    makePlots2()
