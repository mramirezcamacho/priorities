import csv
import pandas as pd

paises = ['PE', 'CO', 'MX', 'CR']
prioridades = ['1', '2', '3', '4', '5']

for pais in paises:
    initialList = []
    first = True
    for prioridad in prioridades:
        csv_file = f'comparacionesCSVJuneDynamic/{pais}/p{
            prioridad}/total_{pais}_p{prioridad}.csv'
        with open(csv_file, mode='r', newline='') as file:
            reader = csv.reader(file)
            data = list(reader)
        if first:
            initialList = data.copy()
            first = False
        else:
            space = ['',]
            for index in range(len(initialList)):
                if index % 9 == 0:
                    initialList[index] = initialList[index] + \
                        space+space + data[index]
                else:
                    initialList[index] = initialList[index] + \
                        space + data[index]
    for i in range(len(initialList)):
        for j in range(len(initialList[i])):
            if '.' in initialList[i][j]:
                initialList[i][j] = str(float(initialList[i][j])*100)+'%'
    with open(f'comparacionesCSVJuneDynamic/{pais}/total_{pais}.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(initialList)
    print(
        f"New row added to 'comparacionesCSVJuneDynamic/{pais}/total_{pais}.csv' successfully.")
