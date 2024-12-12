from pprint import pprint
from matplotlib.ticker import FuncFormatter
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import pandas as pd
from priorityChangesScript import makePlots2
from matplotlib.ticker import MaxNLocator, FuncFormatter
from centralizedData import plotsFolder, comparationData
from divideData import getColumns
import calendar


MONTHS = [3, 9]
mainFolder = comparationData
mainPlotFolder = plotsFolder
serio = 1


def getInitialData(serio: bool):
    if serio:
        paises = ['MX', 'CO', 'PE', 'CR']
        prioridades = ['0', '1', '2', '3', '4',]
        columns = getColumns()
        columns = ['orders_per_eff_online', 'eff_online_rs', 'daily_orders',
                   'exposure_per_eff_online', 'b_p1p2', 'ted_gmv', 'r_burn_gmv', 'b2c_gmv', 'p2c_gmv',
                   'imperfect_order_rate', 'bad_rating_rate', 'eff_online_rs', 'healthy_stores',
                   # 'overdue_orders_per_total_orders',
                   'exposure_uv', 'asp', 'aop', 'b_cancel_rate',
                   ]

    else:
        paises = ['MX']
        prioridades = ['0', '1', '2', '3', '4',]
        columns = ['orders_per_eff_online', 'eff_online_rs', 'daily_orders',
                   'exposure_per_eff_online', 'b_p1p2', 'ted_gmv', 'r_burn_gmv', 'b2c_gmv', 'p2c_gmv',
                   'imperfect_order_rate', 'bad_rating_rate', 'eff_online_rs', 'healthy_stores',
                   # 'overdue_orders_per_total_orders',
                   'exposure_uv', 'asp', 'aop', 'b_cancel_rate',
                   ]
        columns = ['daily_orders',
                   'ted_gmv',
                   ]
    yLabelsPerColumn = {
        'daily_orders': 'Ordenes diarias',
        'orders_per_eff_online': 'Ordenes por eficiencia online',
        'daily_online_hours': 'Horas online diarias',
        'b_cancel_rate': 'Tasa de cancelación',
        'bad_rating_rate': 'Tasa de calificación negativa',
        "healthy_stores": 'Tiendas saludables',
        'imperfect_orders': 'Ordenes imperfectas',
        'ted_gmv': 'TED/GMV',
        'exposure_uv': 'Exposición UV',
        'b_p1p2': 'P1P2'
    }
    return paises, prioridades, columns, yLabelsPerColumn


def getCombinatoryData():
    combinatoryData = [
        (['ted_gmv', 'r_burn_gmv', 'b2c_gmv', 'p2c_gmv'], 'Basic'),
        (['imperfect_order_rate', 'b_cancel_rate', 'bad_rating_rate'], 'Basic'),
        (['imperfect_order_rate', 'bad_rating_rate'], 'Basic'),
        (['imperfect_order_rate', 'orders_per_eff_online'], 'Basic'),
        (['eff_online_rs', 'healthy_stores'], 'Basic'),
        # (['overdue_orders_per_total_orders', 'b_cancel_rate'], 'Dual'),
        (['exposure_per_eff_online', 'b_p1p2'], 'Dual'),
        (['ted_gmv', 'asp'], 'Dual'),

    ]
    return combinatoryData


def savePlot(folder_path: str, file_name: str, dpi: int = 150):
    os.makedirs(folder_path, exist_ok=True)
    output_file = os.path.join(folder_path, file_name)
    plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
    plt.close()  # Close the plot to free up memory


def saveTXT(folder_path: str, file_name: str, content: str):
    os.makedirs(folder_path, exist_ok=True)
    output_file = os.path.join(folder_path, file_name)
    try:
        with open(output_file, 'w') as f:
            f.write(content)
            f.close()
    except Exception as e:
        print(e)


def makeNewPriorityPlot(pais: str, prioridad: str, columna: str, yLabels: dict, config: tuple = ('bottom', 'top'), tasaCambio=0.02):
    folder = f'{mainFolder}/{pais}/p{prioridad}/'
    file_path = folder + f'{columna}.csv'
    data = pd.read_csv(file_path, skiprows=1)

    data = data.iloc[MONTHS[0]-1:]
    data.reset_index(drop=True, inplace=True)

    Month = data['month']
    PriorityData = data['Priority']

    # Create a DataFrame
    data = pd.DataFrame({'Month': Month, 'PriorityData': PriorityData})

    data['Month'] = data['Month'].apply(lambda x: calendar.month_name[x])
    # Plot using seaborn
    sns.set_theme(style="whitegrid")  # Set style, optional
    plt.figure(figsize=(12, 10.5))  # Set figure size, optional

    # Plot PriorityData
    sns.lineplot(x='Month', y='PriorityData',
                 data=data, label=f'{columna.replace("_", " ").replace("gmv", "/gmv").capitalize()}', linewidth=4, color='#fc4c02')

    # Mean and range calculations
    meanLastXMonthsNew = np.mean(PriorityData[-MONTHS[1]+MONTHS[0]:])
    maxValue = max(PriorityData)
    minValue = min(PriorityData)
    meanValue = (maxValue - minValue)

    if meanValue < 0.05:
        margin = meanValue + 0.01
    elif meanValue < 0.1:
        margin = meanValue + 0.04
    elif meanValue < 1:
        margin = meanValue * 0.30
    else:
        margin = meanValue * 0.20

    if minValue - margin < 0:
        plt.ylim([0, maxValue + margin])
    else:
        plt.ylim([minValue - margin, maxValue + margin])

    # Annotate data points
    for i, (mes, new_priority) in enumerate(zip(data['Month'], PriorityData)):
        plt.text(mes, new_priority + ((-meanValue*tasaCambio) if config[1] == 'bottom' else (meanValue*tasaCambio)),
                 f'''{round(new_priority/1000000, 2)}M''' if maxValue // 1000000 > 0 else (
            f'''{round(new_priority * 100, 2)
                 }%''' if (maxValue < 1 and columna != 'orders_per_eff_online') else f"{new_priority:,.2f}"
        ), color='#fc4c02', ha='center', va=config[1], fontsize=22,
                 bbox=dict(facecolor='white', edgecolor='#fc4c02', boxstyle='round,pad=0.3'))

    # Format y-axis
    if maxValue > 30:
        formatter = FuncFormatter(lambda x, pos: '{:,.0f}'.format(x))
    elif maxValue > 1:
        formatter = FuncFormatter(lambda x, pos: '{:,.1f}'.format(x))
    else:
        formatter = FuncFormatter(
            lambda x, pos: str(round(float(x)*100, 2))+'%')
    if columna == 'orders_per_eff_online':
        formatter = FuncFormatter(lambda x, pos: '{:,.1f}'.format(x))
    plt.gca().yaxis.set_major_formatter(formatter)

    # Draw the line between the last two months and calculate the percentage change
    if len(PriorityData) >= 2:
        last_value = PriorityData.iloc[-1]
        second_last_value = PriorityData.iloc[-3]
        percentage_change = (last_value - second_last_value) / \
            second_last_value * 100

        # Coordinates for the line
        x_coords = [data.index[-3], data.index[-1]]
        y_coords = [second_last_value, last_value]

        # Draw the line
        plt.plot(x_coords, y_coords, color='#007acc',
                 linewidth=2, linestyle='--')

        # Annotate the percentage change
        mid_x = (x_coords[0] + x_coords[1]) / 2
        mid_y = (y_coords[0] + y_coords[1]) / 2
        plt.text(mid_x, mid_y, f'{percentage_change:.2f}%', color='#007acc',
                 ha='center', va='bottom', fontsize=22, fontweight='bold',
                 bbox=dict(facecolor='white', edgecolor='#007acc', boxstyle='round,pad=0.3'))

    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
               ncol=2, frameon=False, prop={'size': 20})

    plt.xticks(range(len(data['Month'])))
    plt.xlabel(' ')
    plt.ylabel(' ')

    plot_folder = f'{mainPlotFolder}/{pais}/p{prioridad}/{columna}/'
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xticks(fontsize=20)  # Set X-axis tick labels font size
    plt.yticks(fontsize=20)  # Set Y-axis tick labels font size

    # Create note for the last 4 months
    note = ''
    # format it to millions if it is bigger than 1M, and to percentage if it is less than 1
    highest = f'''Highest: {round(maxValue/1000000, 2)}M''' if maxValue // 1000000 > 0 else (
        f'''Highest: {round(maxValue * 100, 2)}%\n''' if maxValue < 1 and columna != 'orders_per_eff_online' else f'''Highest: {maxValue:,.2f}''')

    lowest = f'''Lowest: {round(minValue/1000000, 2)}M''' if minValue // 1000000 > 0 else (
        f'''Lowest: {round(minValue * 100, 2)}%''' if minValue < 1 and columna != 'orders_per_eff_online' else f'''Lowest: {minValue:,.2f}''')

    highestAndLowest = highest + ', ' + lowest + '\n'
    # Format the mean of the last 4 months
    meanLastXMonths = f'''Mean of the last 4 months: {
        round(meanLastXMonthsNew / 1000000, 2)}M\n''' if maxValue // 1000000 > 0 else (
        f'''Mean of the last 4 months: {round(meanLastXMonthsNew * 100, 2)}%\n''' if maxValue < 1 and columna != 'orders_per_eff_online' else f'''Mean of the last 4 months: {meanLastXMonthsNew:,.2f}\n''')

    note += highestAndLowest + meanLastXMonths

    # Save note and plot
    saveTXT(plot_folder, 'note.txt', note)
    savePlot(plot_folder, f'{config[0][0].upper()}{config[1][0].upper()}.png')


def makeMultiMetricPlot(pais: str, prioridad: str, columnas: list, yLabels: dict, config: tuple = ('bottom', 'top'), tasaCambio=0.02):
    folder = f'{mainFolder}/{pais}/p{prioridad}/'
    data_dict = {}

    for columna in columnas:
        file_path = folder + f'{columna}.csv'
        data = pd.read_csv(file_path, skiprows=1)

        data = data.iloc[MONTHS[0]-1:]
        data.reset_index(drop=True, inplace=True)

        Month = data['month']
        PriorityData = data['Priority']

        data_dict[columna] = pd.DataFrame(
            {'Month': Month, 'PriorityData': PriorityData})
        data_dict[columna]['Month'] = data_dict[columna]['Month'].apply(
            lambda x: calendar.month_name[x])

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 10.5))

    for columna in columnas:
        data = data_dict[columna]
        sns.lineplot(x='Month', y='PriorityData', data=data, label=f'{
                     columna.replace("_", " ").replace("gmv", "/gmv").capitalize()}', linewidth=4)

    max_values = [max(data_dict[col]['PriorityData']) for col in columnas]
    max_value = max(max_values)
    min_values = [min(data_dict[col]['PriorityData']) for col in columnas]
    min_value = min(min_values)

    meanValue = (max_value - min_value)
    margin = 0
    if meanValue < 0.05:
        margin = meanValue + 0.02
    else:
        margin = meanValue * 0.20

    if min_value - margin < 0:
        plt.ylim([0, max_value + margin])
    else:
        plt.ylim([min_value - margin, max_value + margin])

    bigData = max_value > 1

    last_positions = {}
    offset = meanValue * tasaCambio * 3

    for columna in columnas:
        data = data_dict[columna]
        PriorityData = data['PriorityData']

        for i, (mes, new_priority) in enumerate(zip(data['Month'], PriorityData)):
            text_label = f'{round(new_priority/1000000, 2)}M' if max_value // 1000000 > 0 else (
                f'{round(new_priority * 100, 2)}%' if (max_value < 1) and not bigData else f"{new_priority:,.2f}")

            if mes in last_positions:
                last_position = last_positions[mes]
                if abs(new_priority - last_position) < offset:
                    new_priority += offset if new_priority > last_position else -offset

            last_positions[mes] = new_priority

            plt.text(mes, new_priority + ((-meanValue*tasaCambio) if config[1] == 'bottom' else (meanValue*tasaCambio)),
                     text_label, color='black', ha='center', va=config[1], fontsize=22,
                     bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3', linewidth=1))

    plt.xlabel(' ')
    plt.ylabel(' ')

    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
               ncol=2, frameon=False, prop={'size': 20})

    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

    if bigData:
        formatter = FuncFormatter(lambda x, pos: '{:,.2f}'.format(x))
        plt.gca().yaxis.set_major_formatter(formatter)
    else:
        formatter = FuncFormatter(lambda x, pos: str(round(x*100, 2)) + '%')
        plt.gca().yaxis.set_major_formatter(formatter)

    plot_folder = f'{mainPlotFolder}/{pais}/p{prioridad}/{"_".join(columnas)}/'

    savePlot(plot_folder,  f'''{config[0][0].upper()}{
             config[1][0].upper()}.png''')

    noteText = ''
    for columna in columnas:
        # put the average of the last 4 months, and the percentage change between the last 2 months
        noteText += (
            f'''{columna.replace("_", " ").replace("gmv", "/gmv").capitalize()} average in the last 4 months: {
                round(np.mean(data_dict[columna].iloc[-MONTHS[1]:, 1])/1000000, 2)}M'''
            if max_value // 1000000 > 0
            else f'''{columna.replace("_", " ").replace("gmv", "/gmv").capitalize()} average in the last 4 months: {round(np.mean(data_dict[columna].iloc[-MONTHS[1]:, 1]) * 100, 2)} %'''
            if max_value < 1 and columna != 'orders_per_eff_online'
            else f'''{columna.replace("_", " ").replace("gmv", "/gmv").capitalize()} average in the last 4 months: {np.mean(data_dict[columna].iloc[-MONTHS[1]:, 1]):,.2f}'''
        )
        noteText += (
            f''' and the percentage change between the last 2 months: {round(
                (data_dict[columna].iloc[-1, 1] - data_dict[columna].iloc[-2, 1]) / data_dict[columna].iloc[-3, 1] * 100, 2)}%.\n'''
        )
    if len(columnas) > 2:
        noteText = ''

    saveTXT(plot_folder, 'note.txt', noteText)


def RBurnGraphs(pais: str, prioridad: str):
    if prioridad == '0':
        return

    miniData = {}

    groups = {
        'Co-Subsidized Orders': ['All_3_burns_orders', 'B2C_R_Burn_orders'],
        'R burn': ['R_Burn_Only_orders', 'R_Burn_and_P2C_orders'],
        'Others': ['B2C_Only_orders', 'B2C_and_P2C_orders', 'P2C_Only_orders'],
        'Organic': ['No_Burn_orders']
    }

    for group in groups:
        for columna in groups[group]:
            dataColumn = pd.read_csv(
                f'{mainFolder}/{pais}/p{prioridad}/{columna}.csv', skiprows=1)
            dataColumn = dataColumn.iloc[MONTHS[0]-1:]
            monthsDF = dataColumn['month']
            for monthDF in monthsDF:
                if monthDF not in miniData:
                    miniData[monthDF] = {}
                if group not in miniData[monthDF]:
                    miniData[monthDF][group] = 0
                miniData[monthDF][group] += dataColumn[dataColumn['month']
                                                       == monthDF]['Priority'].values[0]

    df = pd.DataFrame(miniData).T
    bestColumns = list(groups.keys())
    monthsDF = np.array(monthsDF.apply(lambda x: calendar.month_name[x]))
    # Replace missing values with 0 and normalize the data
    df.fillna(0, inplace=True)
    data = df[bestColumns].values
    totals = np.sum(data, axis=1)

    # Normalize data to make sure it sums to 100%
    data_normalized = data / totals[:, None] * 100

    # Plotting the stacked bar chart
    fig, ax = plt.subplots(figsize=(12, 10.5))

    # Define the bottom of each bar segment
    bottom = np.zeros(len(monthsDF))

    # Plot each category
    for i, columna in enumerate(bestColumns):
        bars = ax.bar(
            monthsDF, data_normalized[:, i], label=columna, bottom=bottom)

        # Add percentage labels
        for j, bar in enumerate(bars):
            height = bar.get_height()
            percentage = f'{data_normalized[j, i]:.1f}%'
            ax.text(bar.get_x() + bar.get_width() / 2, bottom[j] + height / 2,
                    percentage, ha='center', va='center', color='black', fontsize=22,
                    bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3', linewidth=1))

        bottom += data_normalized[:, i]

    # Adding labels and title
    ax.set_xlabel(' ', )
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
              ncol=3, frameon=False, prop={'size': 20})
    plt.xticks(fontsize=20)  # Set X-axis tick labels font size
    plt.yticks(fontsize=20)  # Set Y-axis tick labels font size

    # Adjust layout to prevent clipping
    plt.tight_layout()

    # Save the plot
    plot_folder = f'''{
        mainPlotFolder}/{pais}/p{prioridad}/DistributionOrdersDiscounts/'''
    savePlot(plot_folder, 'All.png')
    # this note is more complicated, we'll show the nominal values from the csv, do check this out
    noteText = 'Last Month Data:\n'
    dfNomalValues = pd.read_csv('newColumns/newColumns.csv')
    dfNomalValues['date'] = dfNomalValues['date'].str.split(
        '-').str[1].astype(int)
    dfNomalValues = dfNomalValues[dfNomalValues['date'] == MONTHS[1]]
    for group, columns in groups.items():
        sumPerGroup = 0
        for column in columns:
            data2check = dfNomalValues[(dfNomalValues['Country'] == pais) & (
                dfNomalValues['priority'] == f'Priority {prioridad}')]
            if len(data2check) > 0:
                sumPerGroup += int(data2check[column].values[0])
        noteText += f'{group}: {sumPerGroup/1000000}M orders\n' if sumPerGroup // 1000000 > 0 else (
            f'{group}: {sumPerGroup:,.0f} orders\n')
    saveTXT(plot_folder, 'note.txt', noteText)


def makeDualYPlot(pais: str, prioridad: str, columnas: list, yLabels: dict, config: tuple = ('bottom', 'top'), tasaCambio=0.02):
    if len(columnas) != 2:
        raise ValueError("This function requires exactly two columns.")

    folder = f'{mainFolder}/{pais}/p{prioridad}/'
    data_dict = {}

    for columna in columnas:
        file_path = folder + f'{columna}.csv'
        data = pd.read_csv(file_path, skiprows=1)

        data = data.iloc[MONTHS[0]-1:]
        data.reset_index(drop=True, inplace=True)

        Month = data['month']
        PriorityData = data['Priority']

        data_dict[columna] = pd.DataFrame(
            {'Month': Month, 'PriorityData': PriorityData})
        data_dict[columna]['Month'] = data_dict[columna]['Month'].apply(
            lambda x: calendar.month_name[x])

    sns.set_theme(style="ticks")
    fig, ax1 = plt.subplots(figsize=(12, 10.5))

    # Plot for the first column on the left Y-axis
    data1 = data_dict[columnas[0]]
    color1 = 'tab:blue'
    ax1.set_xlabel(' ')
    # ax1.set_ylabel(yLabels.get(columnas[0], columnas[0]), color=color1)
    ax1.plot(data1['Month'], data1['PriorityData'], label=columnas[0].replace("_", " ").replace("gmv", "/gmv").capitalize(),
             color=color1, linewidth=4)
    ax1.tick_params(axis='y', labelcolor=color1,
                    labelsize=22)  # Set y-ticks font size
    ax1.tick_params(axis='x', labelsize=22)  # Set x-ticks font size
# Plot for the second column on the right Y-axis
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    data2 = data_dict[columnas[1]]
    color2 = 'tab:red'
    # ax2.set_ylabel(yLabels.get(columnas[1], columnas[1]), color=color2)
    ax2.plot(data2['Month'], data2['PriorityData'], label=columnas[1].replace("_", " ").replace("gmv", "/gmv").capitalize(),
             color=color2, linewidth=4)
    ax2.tick_params(axis='y', labelcolor=color2, labelsize=22)

    # Add legend
    fig.legend(loc='upper center', bbox_to_anchor=(0.5, 1),
               ncol=2, frameon=False, prop={'size': 22})

    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    i = 0
    # Format Y-axis
    for ax, data, color in [(ax1, data1, color1), (ax2, data2, color2)]:
        max_value = max(data['PriorityData'])
        min_value = min(data['PriorityData'])
        meanValue = (max_value - min_value)
        if meanValue < 0.05:
            margin = meanValue + 0.04
        elif meanValue < 1:
            margin = meanValue * 0.30
        else:
            margin = meanValue * 0.20

        if i == 0:
            if min_value - margin*3 < 0:
                ax.set_ylim([0, max_value + margin])
            else:
                ax.set_ylim([min_value - margin*3, max_value + margin])
            i = 1
        else:
            ax.set_ylim([min_value - margin, max_value + margin*3])
            i = 0

        bigData = max_value > 1

        for i, (mes, new_priority) in enumerate(zip(data['Month'], data['PriorityData'])):
            ax.text(mes, new_priority + ((-meanValue * tasaCambio) if config[1] == 'bottom' else (meanValue * tasaCambio)),
                    f'{round(new_priority/1000000, 2)}M' if max_value // 1000000 > 0 else (
                        f'{round(new_priority * 100, 2)}%' if (max_value <
                                                               1) and not bigData else f"{new_priority:,.2f}"
            ),
                color=color, ha='center', va=config[1], fontsize=22,
                bbox=dict(facecolor='white', edgecolor=color, boxstyle='round,pad=0.3', linewidth=1))

        if bigData:
            formatter = FuncFormatter(lambda x, pos: '{:,.2f}'.format(x))
        else:
            formatter = FuncFormatter(
                lambda x, pos: str(round(x * 100, 2)) + '%')

        ax.yaxis.set_major_formatter(formatter)
        ax.tick_params(axis='y', labelcolor=color, labelsize=20)
    plt.xticks(fontsize=22)  # Set X-axis tick labels font size
    plt.yticks(fontsize=22)  # Set Y-axis tick labels font size

    plot_folder = f'{mainPlotFolder}/{pais}/p{prioridad}/{'_'.join(columnas)}/'
    savePlot(plot_folder,  f'''{config[0][0].upper()}{
             config[1][0].upper()}.png''')
    noteText = ''
    for columna in columnas:
        # put the average of the last 4 months, and the percentage change between the last 2 months
        noteText += (
            f'''{columna.replace("_", " ").replace("gmv", "/gmv").capitalize()} average in the last 4 months: {
                round(np.mean(data_dict[columna].iloc[-MONTHS[1]:, 1])/1000000, 2)}M'''
            if max_value // 1000000 > 0
            else f'''{columna.replace("_", " ").replace("gmv", "/gmv").capitalize()} average in the last 4 months: {round(np.mean(data_dict[columna].iloc[-MONTHS[1]:, 1]) * 100, 2)} %'''
            if max_value < 1 and columna != 'orders_per_eff_online'
            else f'''{columna.replace("_", " ").replace("gmv", "/gmv").capitalize()} average in the last 4 months: {np.mean(data_dict[columna].iloc[-MONTHS[1]:, 1]):,.2f}'''
        )
        noteText += (
            f''' and the percentage change between the last 2 months: {round(
                (data_dict[columna].iloc[-1, 1] - data_dict[columna].iloc[-2, 1]) / data_dict[columna].iloc[-3, 1] * 100, 2)}%.\n'''
        )
    saveTXT(plot_folder, 'note.txt', noteText)


def graphsForAllPriorities(country: str, columns: list):
    for nominal in [True, False]:
        for columna in columns:
            df = pd.read_csv(
                f'{mainFolder}/{country}/All/{columna}.csv', skiprows=1)
            df = df.iloc[MONTHS[0]-1:]
            df.reset_index(drop=True, inplace=True)
            monthsDF = df['month']
            monthsDF = monthsDF.apply(lambda x: calendar.month_name[x])
            categories = df.columns[1:].values
            data = df.iloc[:, 1:].values
            totals = np.sum(data, axis=1)
            percentages = data / totals[:, None] * 100

            # Plotting the stacked bar chart
            fig, ax = plt.subplots(figsize=(12, 10.5))

            # Define the bottom of each bar segment
            bottom = np.zeros(len(monthsDF))

            # Plot each category
            for i in range(len(categories)):
                bars = ax.bar(monthsDF, data[:, i],
                              label=categories[i], bottom=bottom)

                # Add percentage labels
                for j, bar in enumerate(bars):
                    height = bar.get_height()
                    if not nominal:
                        percentage = f'{percentages[j, i]:.1f}%'
                    else:
                        percentage = f'{data[j, i]:,.1f}'
                    ax.text(bar.get_x() + bar.get_width() / 2, bottom[j] + height / 2,
                            percentage, ha='center', va='center', color='black', fontsize=22,
                            bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3', linewidth=1))

                bottom += data[:, i]
            if not nominal:
                for j in range(len(monthsDF)):
                    total_height = bottom[j]  # Total height of the stacked bar
                    ax.text(j, total_height*1.02, f'''{
                            total_height:,.1f}''', ha='center', va='bottom', fontsize=19, color='black')
            # Adding labels and title
            ax.set_ylabel(
                f'{columna.replace("_", " ").replace("gmv", "/gmv").capitalize()}', fontsize=22)
            ax.set_xlabel(' ', fontsize=18)
            ax.legend(loc='upper center',
                      bbox_to_anchor=(0.5, 1.15), ncol=3, frameon=False,
                      prop={'size': 22})

            # Adjust layout to prevent clipping
            ax.set_ylim(0, np.max(bottom) * 1.1)

            plt.xticks(fontsize=20)  # Set X-axis tick labels font size
            plt.yticks(fontsize=20)  # Set Y-axis tick labels font size
            plt.tight_layout()

            # Save the plot
            plot_folder = f'''{mainPlotFolder}/{country}/All/{columna}{
                '_nominal' if nominal else '_percentages'}/'''

            savePlot(plot_folder, f'''All.png''')
            # for the text, put the nominal numbers of the last 2 months
            if not nominal:
                textForNote = 'Last Month Data:\n'
                for i in range(len(categories)):
                    textForNote += f'{categories[i]}: {round(data[-1, i] / 1000000, 2)}M, ' if data[-1, i] // 1000000 > 0 else (
                        f'{categories[i]}: {round(data[-1, i] * 100, 2)}%, ' if data[-1, i] < 1 and columna != 'orders_per_eff_online' else f'{categories[i]}: {data[-1, i]:,.2f}, ')
            else:
                textForNote = 'Last Month percentages:\n'
                for i in range(len(categories)):
                    textForNote += f'{categories[i]
                                      }: {round(percentages[-1, i], 2)}%, '

            textForNote = textForNote[:-2]
            saveTXT(plot_folder, 'note.txt', textForNote)


def main():
    paises, prioridades, columns, yLabelsPerColumn = getInitialData(serio)
    combinatory = getCombinatoryData()

    configs = [('top', 'top'), ]
    for pais in paises:
        for prioridad in prioridades:
            for config in configs:
                for columna in columns:
                    makeNewPriorityPlot(pais, prioridad, columna,
                                        yLabelsPerColumn, config)
                for combination in combinatory:
                    if combination[1] == 'Basic':
                        makeMultiMetricPlot(pais, prioridad, combination[0],
                                            yLabelsPerColumn, config)
                    else:
                        makeDualYPlot(pais, prioridad, combination[0],
                                      yLabelsPerColumn, config)
            #     pass
            # if prioridad != '0':
                # RBurnGraphs(pais, prioridad)
            print(f'Plots de {pais} p{prioridad} creado')
        graphsForAllPriorities(pais, columns)
        print(f'{pais} done!')


if __name__ == '__main__':
    main()
    makePlots2()
