from matplotlib.ticker import FuncFormatter
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import pandas as pd
from priorityChangesScript import makePlots
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


def getCombinatoryData(serio: bool = True):
    combinatoryData = [
        ['ted_gmv', 'r_burn_gmv', 'b2c_gmv', 'p2c_gmv'],
        ['imperfect_order_rate', 'b_cancel_rate', 'bad_rating_rate'],
        ['imperfect_order_rate', 'orders_per_eff_online']

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


def checkForNanValues(npArray):
    for i in range(len(npArray)):
        if npArray[i] == None:
            npArray[i] = 0
    return npArray


def makeNewPriorityPlot(pais: str, prioridad: str, columna: str, yLabels: dict, config: tuple = ('bottom', 'top'), tasaCambio=0.06):
    folder = f'{mainFolder}/{pais}/p{prioridad}/'
    file_path = folder + f'comparacion_{columna}.csv'
    data = pd.read_csv(file_path, skiprows=1)

    if prioridad == '2':
        data = data.iloc[2:]
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
        note = f'El rendimiento de la nueva prioridad en los últimos 4 meses es:\n- Media de los últimos 4 meses: {
            round(meanLast4MonthsNew, 2)}'
    else:
        note = 'No hay suficientes datos para calcular la media de los últimos 4 meses.'

    # Save note and plot
    saveTXT(plot_folder, 'note.txt', note)
    savePlot(plot_folder, f'{config[0][0].upper()}_{config[1][0].upper()}.png')


def makePlot(pais: str, prioridad: str, columna: str, yLabels: dict, config: tuple = ('bottom', 'top'), tasaCambio=0.06):
    folder = f'{mainFolder}/{pais}/p{prioridad}/'
    file_path = folder + f'comparacion_{columna}.csv'
    data = pd.read_csv(file_path, skiprows=1)

    if prioridad == '2':
        data = data.iloc[2:]
        data.reset_index(drop=True, inplace=True)
    Month = data['month']
    OldPriority = data['Old priority']
    NewPriority = data['New priority']

    # Create a DataFrame
    data = pd.DataFrame(
        {'Month': Month, 'OldPriority': OldPriority, 'NewPriority': NewPriority})

    # Plot using seaborn
    sns.set_theme(style="whitegrid")  # Set style, optional
    plt.figure(figsize=(9, 6))  # Set figure size, optional

    bigData = False
    for i in range(len(OldPriority)):
        if OldPriority[i] > 1 or NewPriority[i] > 1:
            bigData = True
            break
    if columna in ['orders_per_eff_online']:
        bigData = True
    # Plot OldPriority
    sns.lineplot(x='Month', y='OldPriority',
                 data=data, label=f'Previous {columna.replace("_", " ").replace("gmv", "/gmv").capitalize()}', linewidth=4)

    # Plot NewPriority
    sns.lineplot(x='Month', y='NewPriority',
                 data=data, label=f'New {columna.replace("_", " ").replace("gmv", "/gmv").capitalize()}', linewidth=4, color='#fc4c02')

    #! AAAAA
    meanLast3MonthsOld = np.mean(OldPriority[-mesesAtras:])
    meanLast3MonthsNew = np.mean(NewPriority[-mesesAtras:])
    maxValue = max(max(OldPriority), max(NewPriority))
    minValue = min(min(OldPriority), min(NewPriority))
    meanValue = (maxValue - minValue)

    for i, (mes, old_priority, new_priority) in enumerate(zip(Month, OldPriority, NewPriority)):
        plt.text(mes, old_priority + ((-meanValue*tasaCambio) if config[0] == 'bottom' else (meanValue*tasaCambio)),
                 f'''{round(old_priority/1000000, 2)}M''' if maxValue // 1000000 > 0 else (
            f'''{round(old_priority * 100, 2)
                 }%''' if (maxValue < 1) and not bigData else f"{old_priority:,.2f}"
        ),
            color='blue', ha='center', va=config[0],
            bbox=dict(facecolor='white', edgecolor='blue', boxstyle='round,pad=0.3'))
        plt.text(mes, new_priority + ((-meanValue*tasaCambio) if config[1] == 'bottom' else (meanValue*tasaCambio)),
                 f'''{round(new_priority/1000000, 2)}M''' if maxValue // 1000000 > 0 else (
            f'''{round(new_priority * 100, 2)
                 }%''' if (maxValue < 1) and not bigData else f"{new_priority:,.2f}"
        ), color='#fc4c02', ha='center', va=config[1],
                 bbox=dict(facecolor='white', edgecolor='#fc4c02', boxstyle='round,pad=0.3'))
    #! AAAAA

    # Add labels and title
    if bigData:
        if max(max(OldPriority), max(NewPriority)) > 10:
            formatter = FuncFormatter(lambda x, pos: '{:,.0f}'.format(x))
            plt.gca().yaxis.set_major_formatter(formatter)
        elif max(max(OldPriority), max(NewPriority)) > 1:
            formatter = FuncFormatter(lambda x, pos: '{:,.1f}'.format(x))
            plt.gca().yaxis.set_major_formatter(formatter)
        else:
            formatter = FuncFormatter(lambda x, pos: '{:,.2f}'.format(x))
            plt.gca().yaxis.set_major_formatter(formatter)
    else:
        formatter = FuncFormatter(lambda x, pos: str(round(x*100, 2)) + '%')
        plt.gca().yaxis.set_major_formatter(formatter)
    if bigData:
        if list(data['NewPriority'].values)[-mesesAtras-1] == 0 or list(data['NewPriority'].values)[-1] == 0 or list(data['NewPriority'].values)[-mesesAtras-1] == None or list(data['NewPriority'].values)[-1] == None or np.isnan(list(data['NewPriority'].values)[-mesesAtras-1]) or np.isnan(list(data['NewPriority'].values)[-1]):
            cambiosNew = None
        else:
            cambiosNew = (list(data['NewPriority'].values)
                          [-1] / list(data['NewPriority'].values)[-mesesAtras-1])-1
        if list(data['OldPriority'].values)[-mesesAtras-1] == 0 or list(data['OldPriority'].values)[-1] == 0 or list(data['OldPriority'].values)[-mesesAtras-1] == None or list(data['OldPriority'].values)[-1] == None or np.isnan(list(data['OldPriority'].values)[-mesesAtras-1]) or np.isnan(list(data['OldPriority'].values)[-1]):
            cambiosOld = None
        else:
            cambiosOld = (list(data['OldPriority'].values)
                          [-1] / list(data['OldPriority'].values)[-mesesAtras-1])-1
    else:
        if list(data['NewPriority'].values)[-mesesAtras-1] == 0 or list(data['NewPriority'].values)[-1] == 0 or list(data['NewPriority'].values)[-mesesAtras-1] == None or list(data['NewPriority'].values)[-1] == None or np.isnan(list(data['NewPriority'].values)[-mesesAtras-1]) or np.isnan(list(data['NewPriority'].values)[-1]):
            cambiosNew = None
        else:
            cambiosNew = list(data['NewPriority'].values)[-1] - \
                list(data['NewPriority'].values)[-mesesAtras-1]
        if list(data['OldPriority'].values)[-mesesAtras-1] == 0 or list(data['OldPriority'].values)[-1] == 0 or list(data['OldPriority'].values)[-mesesAtras-1] == None or list(data['OldPriority'].values)[-1] == None or np.isnan(list(data['OldPriority'].values)[-mesesAtras-1]) or np.isnan(list(data['OldPriority'].values)[-1]):
            cambiosOld = None
        else:
            cambiosOld = list(data['OldPriority'].values)[-1] - \
                list(data['OldPriority'].values)[-mesesAtras-1]
    if cambiosNew is None or cambiosOld is None:
        note = 'The change cannot be calculated over the last {mesesAtras} months'
    elif cambiosNew > cambiosOld:
        if bigData:
            if cambiosNew < 0 and cambiosOld < 0:
                note = f'The new priority has@a better tendency@than the old one over the last {mesesAtras} months ({
                    round(cambiosNew*100, 2)}% > {round(cambiosOld*100, 2)}%)'
            else:
                note = f'The new priority has@a better tendency@than the old one over the last {mesesAtras} months ({
                    round(cambiosNew*100, 2)}% > {round(cambiosOld*100, 2)}%)'
        else:
            if cambiosNew < 0 and cambiosOld < 0:
                note = f'The new priority has@a better tendency@than the old one over the last {mesesAtras} months ({round(
                    cambiosNew*100, 2)} > {round(cambiosOld*100, 2)} percentage points)'
            else:
                note = f'The new priority has@a better tendency@than the old one over the last {mesesAtras} months ({round(
                    cambiosNew*100, 2)} > {round(cambiosOld*100, 2)} percentage points)'
    elif cambiosNew < cambiosOld:
        if bigData:
            if cambiosNew < 0 and cambiosOld < 0:
                note = f'The new priority has@a worse tendency@than the old one over the last {mesesAtras} months ({
                    round(cambiosNew*100, 2)}% < {round(cambiosOld*100, 2)}%)'
            else:
                note = f'The new priority has@a worse tendency@than the old one over the last {mesesAtras} months ({
                    round(cambiosNew*100, 2)}% < {round(cambiosOld*100, 2)}%)'
        else:
            if cambiosNew < 0 and cambiosOld < 0:
                note = f'The new priority has@a worse tendency@than the old one over the last {mesesAtras} months ({round(
                    cambiosNew*100, 2)} < {round(cambiosOld*100, 2)} percentage points)'
            else:
                note = f'The new priority has@a worse tendency@than the old one over the last {mesesAtras} months ({round(
                    cambiosNew*100, 2)} < {round(cambiosOld*100, 2)} percentage points)'
    else:
        note = 'The new priority has performed the same as the old one over the last {mesesAtras} months'

    plt.xlabel('Month')
    # if columna in yLabels:
    if columna in ['lala']:
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
    # plt.title(f"""Comparación de la prioridad {prioridad} en {paisNombre} respecto a
    #           {columna.replace("_", " ").replace("gmv", "/gmv").capitalize()} por mes""")

    #! here we put the note under, now we'll make a txt with that data

    # plt.annotate(
    #     note,
    #     xy=(-0.1, -0.2),
    #     xycoords="axes fraction",
    #     bbox=dict(boxstyle="round,pad=0.3",
    #               edgecolor="black", facecolor="aliceblue")
    # )

    plt.legend(loc='upper center', bbox_to_anchor=(
        0.5, 1.15), ncol=2, frameon=False, prop={'size': 16})
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

    if prioridad == '2':
        plt.xticks(range(3, months[1]+1))
    else:
        plt.xticks(range(months[0], months[1]+1))

    plot_folder = f'{mainPlotFolder}/{pais}/p{prioridad}/{columna}/'

    if bigData:
        note += f'\n- The last 3 months mean of the old priority is {round(
            meanLast3MonthsOld, 2)} and the new priority is {round(meanLast3MonthsNew, 2)}'
    else:
        note += f'\n- The last 3 months mean of the old priority is {round(
            meanLast3MonthsOld*100, 2)}% and the new priority is {round(meanLast3MonthsNew*100, 2)}%'
    note = '- '+note
    saveTXT(plot_folder, f'note.txt', note)
    savePlot(plot_folder, f'''{config[0][0].upper()}{
             config[1][0].upper()}.png''')


def makeMultiMetricPlot(pais: str, prioridad: str, columnas: list, yLabels: dict, config: tuple = ('bottom', 'top'), tasaCambio=0.06):
    folder = f'{mainFolder}/{pais}/p{prioridad}/'
    data_dict = {}

    for columna in columnas:
        file_path = folder + f'comparacion_{columna}.csv'
        data = pd.read_csv(file_path, skiprows=1)

        if prioridad == '2':
            data = data.iloc[2:]
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
                     columna.replace("_", " ").replace("gmv", "/gmv").capitalize()}', linewidth=2)

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
    plt.ylabel('Metrics')
    plt.title(f'Comparison of Various Metrics for Priority {
              prioridad} in {pais}')

    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
               ncol=2, frameon=False, prop={'size': 12})

    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

    if bigData:
        formatter = FuncFormatter(lambda x, pos: '{:,.2f}'.format(x))
        plt.gca().yaxis.set_major_formatter(formatter)
    else:
        formatter = FuncFormatter(lambda x, pos: str(round(x*100, 2)) + '%')
        plt.gca().yaxis.set_major_formatter(formatter)

    plot_folder = f'{mainPlotFolder}/{pais}/p{prioridad}/multi_metrics/'

    savePlot(plot_folder, f'''{'_'.join(columnas)}.png''')


def makePlotWith2Columns(pais: str, prioridad: str, columna: str, columna2: str, yLabels: dict):
    folder = f'{mainFolder}/{pais}/p{prioridad}/'
    file_path = folder + f'comparacion_{columna}.csv'
    data = pd.read_csv(file_path, skiprows=1)

    Month = data['month']
    OldPriorityFirstFeature = data['Old priority']
    NewPriorityFirstFeature = data['New priority']

    file_path2 = folder + f'comparacion_{columna2}.csv'
    data2 = pd.read_csv(file_path2, skiprows=1)
    OldPrioritySecondFeature = data2['Old priority']
    NewPrioritySecondFeature = data2['New priority']

    df = pd.DataFrame({
        'Month': Month,
        'Y1': OldPriorityFirstFeature,
        'Y2': NewPriorityFirstFeature,
        'Y3': OldPrioritySecondFeature,
        'Y4': NewPrioritySecondFeature
    })

    # Crear el gráfico
    fig, ax1 = plt.subplots(figsize=(9, 6))

    # Primer eje Y
    sns.lineplot(data=df, x='Month', y='Y1', ax=ax1, label=f'''P_old en {
                 columna.replace("_", " ").capitalize()}''', color='#aec7e8')
    sns.lineplot(data=df, x='Month', y='Y2', ax=ax1, label=f'''P_new en {
                 columna.replace("_", " ").capitalize()}''', color='#1f77b4')
    if columna in yLabels:
        ax1.set_ylabel(yLabels[columna])
    else:
        ax1.set_ylabel(' ')

    # Segundo eje Y
    ax2 = ax1.twinx()
    sns.lineplot(data=df, x='Month', y='Y3', ax=ax2, label=f'''P_old en {
                 columna2.replace("_", " ").capitalize()}''', color='#98df8a')
    sns.lineplot(data=df, x='Month', y='Y4', ax=ax2, label=f'''P_new en {
                 columna2.replace("_", " ").capitalize()}''', color='#2ca02c')
    if columna2 in yLabels:
        ax2.set_ylabel(yLabels[columna2])
    else:
        ax2.set_ylabel(' ')

    # Ajustar leyendas
    plt.xticks(range(months[0], months[1]+1))
    if pais == 'PE':
        paisNombre = 'Perú'
    elif pais == 'CO':
        paisNombre = 'Colombia'
    elif pais == 'MX':
        paisNombre = 'México'
    elif pais == 'CR':
        paisNombre = 'Costa Rica'
    # plt.title(f"""Comparación de la prioridad {prioridad} en {paisNombre} respecto a
    #           {columna.replace("_", " ").replace("gmv", "/gmv").capitalize()} y {columna2.replace("_", " ").capitalize()} por mes""")
    plt.xticks(range(months[0], months[1]+1))
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles=handles1, labels=labels1, loc='upper left')
    ax2.legend(handles=handles2, labels=labels2, loc='upper right')
    plot_folder = f'{mainPlotFolder}/{pais}/p{prioridad}/combinaciones/'
    savePlot(plot_folder, f'comparacion_{columna}_X_{columna2}.png')


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
                                    yLabelsPerColumn, config)
            print(f'Plots de {pais} p{prioridad} creado')


main()
makePlots()
