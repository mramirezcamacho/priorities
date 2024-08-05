from matplotlib.ticker import FuncFormatter
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import pandas as pd
from priorityChangesScript import makePlots2
from matplotlib.ticker import MaxNLocator, FuncFormatter


months = [3, 6]
mainFolder = 'comparacionesCSVJuneDynamic'
mainPlotFolder = 'plotsJuneDynamic'
mesesAtras = 3
serio = 1


def getInitialData(serio: bool):
    if serio:
        paises = ['PE', 'CO', 'MX', 'CR']
        prioridades = ['1', '2', '3', '4', '5']
        columns = ['daily_orders', 'orders_per_eff_online', 'daily_online_hours', 'b2c_per_order',
                   'b_cancel_rate', 'bad_rating_rate', "healthy_stores", 'imperfect_orders', 'ted', 'ted_gmv', 'exposure_uv', 'b_p1p2', "r_burn", "r_burn_gmv", "r_burn_per_order", "b2c_total_burn", "b2c_gmv", "b2c_per_order", 'eff_online_rs', 'imperfect_order_rate']
        # prioridades = ['4', '5']
        # columns = ['eff_online_rs', 'imperfect_order_rate']
    else:
        paises = ['PE']
        prioridades = ['2', '3']
        columns = ['orders_per_eff_online', 'ted_gmv', 'b_p1p2',
                   "b_cancel_rate", 'bad_rating_rate', 'imperfect_orders', ]
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
        ['ted_gmv', 'r_burn_gmv', 'b2c_gmv', 'p2c_gmv'],
        ['imperfect_order_rate', 'b_cancel_rate', 'bad_rating_rate'],
        ['imperfect_order_rate', 'orders_per_eff_online'],
        ['eff_online_rs', 'healthy_stores']

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


def makeNewPriorityPlot(pais: str, prioridad: str, columna: str, yLabels: dict, config: tuple = ('bottom', 'top'), tasaCambio=0.06):
    folder = f'{mainFolder}/{pais}/p{prioridad}/'
    file_path = folder + f'comparacion_{columna}.csv'
    data = pd.read_csv(file_path, skiprows=1)

    data = data.iloc[months[0]-1:]
    data.reset_index(drop=True, inplace=True)

    Month = data['month']
    NewPriority = data['New priority']

    # Create a DataFrame
    data = pd.DataFrame({'Month': Month, 'NewPriority': NewPriority})

    # Plot using seaborn
    sns.set_theme(style="whitegrid")  # Set style, optional
    plt.figure(figsize=(9, 6))  # Set figure size, optional

    # Plot NewPriority
    sns.lineplot(x='Month', y='NewPriority',
                 data=data, label=f'{columna.replace("_", " ").replace("gmv", "/gmv").capitalize()}', linewidth=4, color='#fc4c02')

    # Mean and range calculations
    meanLast4MonthsNew = np.mean(NewPriority[-4:])
    maxValue = max(NewPriority)
    minValue = min(NewPriority)
    meanValue = (maxValue - minValue)

    for i, (mes, new_priority) in enumerate(zip(Month, NewPriority)):
        plt.text(mes, new_priority + ((-meanValue*tasaCambio) if config[1] == 'bottom' else (meanValue*tasaCambio)),
                 f'''{round(new_priority/1000000, 2)}M''' if maxValue // 1000000 > 0 else (
            f'''{round(new_priority * 100, 2)
                 }%''' if (maxValue < 1) else f"{new_priority:,.2f}"
        ), color='#fc4c02', ha='center', va=config[1],
                 bbox=dict(facecolor='white', edgecolor='#fc4c02', boxstyle='round,pad=0.3'))

    # Format y-axis
    if maxValue > 10:
        formatter = FuncFormatter(lambda x, pos: '{:,.0f}'.format(x))
    elif maxValue > 1:
        formatter = FuncFormatter(lambda x, pos: '{:,.1f}'.format(x))
    else:
        formatter = FuncFormatter(
            lambda x, pos: str(round(float(x)*100, 2))+'%')
    plt.gca().yaxis.set_major_formatter(formatter)

    # Labels and title
    plt.xlabel('Month')
    if columna in ['bobobo']:
        plt.ylabel(yLabels[columna])
    else:
        plt.ylabel(' ')

    if pais == 'PE':
        paisNombre = 'Perú'
    elif pais == 'CO':
        paisNombre = 'Colombia'
    elif pais == 'MX':
        paisNombre = 'México'
    elif pais == 'CR':
        paisNombre = 'Costa Rica'

    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
               ncol=2, frameon=False, prop={'size': 16})

    if prioridad == '2':
        plt.xticks(range(3, months[1]+1))
    else:
        plt.xticks(range(months[0], months[1]+1))

    plot_folder = f'{mainPlotFolder}/{pais}/p{prioridad}/{columna}/'
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

    # Create note for the last 4 months
    if len(NewPriority) >= 4:
        note = f'Media de los últimos 4 meses: {
            round(meanLast4MonthsNew, 2)}'
    else:
        note = 'No hay suficientes datos para calcular la media de los últimos 4 meses.'

    # Save note and plot
    saveTXT(plot_folder, 'note.txt', note)
    savePlot(plot_folder, f'{config[0][0].upper()}{config[1][0].upper()}.png')


def makeMultiMetricPlot(pais: str, prioridad: str, columnas: list, yLabels: dict, config: tuple = ('bottom', 'top'), tasaCambio=0.06):
    folder = f'{mainFolder}/{pais}/p{prioridad}/'
    data_dict = {}

    for columna in columnas:
        file_path = folder + f'comparacion_{columna}.csv'
        data = pd.read_csv(file_path, skiprows=1)

        data = data.iloc[months[0]-1:]
        data.reset_index(drop=True, inplace=True)

        Month = data['month']
        OldPriority = data['Old priority']
        NewPriority = data['New priority']

        data_dict[columna] = pd.DataFrame(
            {'Month': Month, 'OldPriority': OldPriority, 'NewPriority': NewPriority})

    sns.set_theme(style="whitegrid")  # Set style, optional
    plt.figure(figsize=(12, 8))  # Set figure size, optional

    for columna in columnas:
        data = data_dict[columna]
        sns.lineplot(x='Month', y='NewPriority', data=data, label=f'{
                     columna.replace("_", " ").replace("gmv", "/gmv").capitalize()}', linewidth=4)

    max_values = [max(data_dict[col]['NewPriority']) for col in columnas]
    max_value = max(max_values)
    min_values = [min(data_dict[col]['NewPriority']) for col in columnas]
    min_value = min(min_values)

    meanValue = (max_value - min_value)
    bigData = max_value > 1

    for columna in columnas:
        data = data_dict[columna]
        NewPriority = data['NewPriority']

        for i, (mes, new_priority) in enumerate(zip(data['Month'], NewPriority)):
            plt.text(mes, new_priority + ((-meanValue*tasaCambio) if config[1] == 'bottom' else (meanValue*tasaCambio)),
                     f'{round(new_priority/1000000, 2)}M' if max_value // 1000000 > 0 else (
                f'{round(new_priority * 100, 2)}%' if (max_value <
                                                       1) and not bigData else f"{new_priority:,.2f}"
            ),
                # Adjusted fontsize
                color='black', ha='center', va=config[1], fontsize=12,
                # Added bbox
                # Added bbox with border
                bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3', linewidth=1))

    plt.xlabel('Month')
    plt.ylabel(' ')

    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
               ncol=2, frameon=False, prop={'size': 16})

    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

    if bigData:
        formatter = FuncFormatter(lambda x, pos: '{:,.2f}'.format(x))
        plt.gca().yaxis.set_major_formatter(formatter)
    else:
        formatter = FuncFormatter(lambda x, pos: str(round(x*100, 2)) + '%')
        plt.gca().yaxis.set_major_formatter(formatter)

    plot_folder = f'{mainPlotFolder}/{pais}/p{prioridad}/{'_'.join(columnas)}/'

    savePlot(plot_folder, f'''TT.png''')
    savePlot(plot_folder, f'''TB.png''')
    savePlot(plot_folder, f'''BT.png''')
    savePlot(plot_folder, f'''BB.png''')
    saveTXT(plot_folder, f'note.txt', 'Example text')


def makeDualYPlot(pais: str, prioridad: str, columnas: list, yLabels: dict, config: tuple = ('bottom', 'top'), tasaCambio=0.06):
    if len(columnas) != 2:
        raise ValueError("This function requires exactly two columns.")

    folder = f'{mainFolder}/{pais}/p{prioridad}/'
    data_dict = {}

    for columna in columnas:
        file_path = folder + f'comparacion_{columna}.csv'
        data = pd.read_csv(file_path, skiprows=1)

        data = data.iloc[months[0]-1:]
        data.reset_index(drop=True, inplace=True)

        Month = data['month']
        OldPriority = data['Old priority']
        NewPriority = data['New priority']

        data_dict[columna] = pd.DataFrame(
            {'Month': Month, 'OldPriority': OldPriority, 'NewPriority': NewPriority})

    sns.set_theme(style="whitegrid")
    fig, ax1 = plt.subplots(figsize=(12, 8))

    # Plot for the first column on the left Y-axis
    data1 = data_dict[columnas[0]]
    color1 = 'tab:blue'
    ax1.set_xlabel('Month')
    ax1.set_ylabel(yLabels.get(columnas[0], columnas[0]), color=color1)
    ax1.plot(data1['Month'], data1['NewPriority'], label=columnas[0].replace("_", " ").replace("gmv", "/gmv").capitalize(),
             color=color1, linewidth=4)
    ax1.tick_params(axis='y', labelcolor=color1)

    # Plot for the second column on the right Y-axis
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    data2 = data_dict[columnas[1]]
    color2 = 'tab:red'
    ax2.set_ylabel(yLabels.get(columnas[1], columnas[1]), color=color2)
    ax2.plot(data2['Month'], data2['NewPriority'], label=columnas[1].replace("_", " ").replace("gmv", "/gmv").capitalize(),
             color=color2, linewidth=4)
    ax2.tick_params(axis='y', labelcolor=color2)

    # Add legend
    fig.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
               ncol=2, frameon=False, prop={'size': 16})

    # Format Y-axis
    for ax, data, color in [(ax1, data1, color1), (ax2, data2, color2)]:
        max_value = max(data['NewPriority'])
        min_value = min(data['NewPriority'])
        meanValue = (max_value - min_value)
        bigData = max_value > 1

        for i, (mes, new_priority) in enumerate(zip(data['Month'], data['NewPriority'])):
            ax.text(mes, new_priority + ((-meanValue * tasaCambio) if config[1] == 'bottom' else (meanValue * tasaCambio)),
                    f'{round(new_priority/1000000, 2)}M' if max_value // 1000000 > 0 else (
                        f'{round(new_priority * 100, 2)}%' if (max_value <
                                                               1) and not bigData else f"{new_priority:,.2f}"
            ),
                color=color, ha='center', va=config[1], fontsize=12,
                bbox=dict(facecolor='white', edgecolor=color, boxstyle='round,pad=0.3', linewidth=1))

        if bigData:
            formatter = FuncFormatter(lambda x, pos: '{:,.2f}'.format(x))
        else:
            formatter = FuncFormatter(
                lambda x, pos: str(round(x * 100, 2)) + '%')
        ax.yaxis.set_major_formatter(formatter)

    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

    # Save plots
    plot_folder = f'{mainPlotFolder}/{pais}/p{prioridad}/{'_'.join(columnas)}/'

    savePlot(plot_folder, f'''DualY.png''')
    saveTXT(plot_folder, f'note.txt', 'Example text')

    plt.show()  # Show the plot

# Example usage
# makeDualYPlot('CO', '1', ['column1', 'column2'], {'column1': 'Label 1', 'column2': 'Label 2'})


def main():
    paises, prioridades, columns, yLabelsPerColumn = getInitialData(serio)
    combinatory = getCombinatoryData()

    configs = [('bottom', 'top'), ('top', 'bottom'),
               ('bottom', 'bottom'), ('top', 'top')]
    for pais in paises:
        for prioridad in prioridades:
            for config in configs:
                for columna in columns:
                    makeNewPriorityPlot(pais, prioridad, columna,
                                        yLabelsPerColumn, config)
            for combination in combinatory:
                makeMultiMetricPlot(pais, prioridad, combination,
                                    yLabelsPerColumn, ('top', 'top'))
            print(f'Plots de {pais} p{prioridad} creado')


main()
makePlots2()
