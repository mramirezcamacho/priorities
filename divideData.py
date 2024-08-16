import pandas as pd
from centralizedData import mainData, dividedData
import os


folderToGetData = mainData+"/"
folderToSendData = dividedData


def getColumns(filePath: str = folderToGetData):
    files = os.listdir(filePath)
    columns = []
    for file in files:
        data = pd.read_csv(filePath+file)
        for column in data.columns[3:]:
            if column not in columns:
                columns.append(column)
    return columns


def addNewColumns(df: pd.DataFrame):
    newColumnsDF = pd.read_csv('newColumns/newColumns.csv')
    newColumnsToAppend = newColumnsDF.columns[3:]
    for column in newColumnsToAppend:
        df[column] = None
    newColumnsDF['date'] = newColumnsDF['date'].str.split(
        '-').str[1].astype(int)
    prioridades = newColumnsDF['priority'].unique()
    months = newColumnsDF['date'].unique()
    countries = newColumnsDF['Country'].unique()
    # now, lets append the new columns to df with the values of newColumnsDF that match with the df
    for country in countries:
        for month in months:
            for priority in prioridades:
                totalSumOfTheNewColumns = 0
                for column in newColumnsToAppend:
                    value = newColumnsDF[(newColumnsDF['Country'] == country) & (
                        newColumnsDF['date'] == month) & (newColumnsDF['priority'] == priority)][column].values
                    if len(value) > 0:
                        totalSumOfTheNewColumns += value[0]
                if totalSumOfTheNewColumns == 0:
                    continue
                for column in newColumnsToAppend:
                    value = newColumnsDF[(newColumnsDF['Country'] == country) & (
                        newColumnsDF['date'] == month) & (newColumnsDF['priority'] == priority)][column].values
                    if len(value) > 0:
                        df.loc[(df['country_code'] == country) & (df['month_'] == month) & (
                            df['priority'] == priority), column] = value[0] / totalSumOfTheNewColumns
    for i, row in df.iterrows():
        for newColumn in newColumnsToAppend:
            if df.loc[i, newColumn] is None:
                df.loc[i, newColumn] = 0
    return df


def addNewColumns2(df: pd.DataFrame):
    newColumnsDF = pd.read_csv('newColumns/newColumns2.csv')
    newColumnToAppend = 'overdue_orders_per_total_orders'
    df[newColumnToAppend] = None
    months = newColumnsDF['month_'].unique()
    countries = newColumnsDF['country_code'].unique()
    prioridades = newColumnsDF['priority'].unique()
    for country in countries:
        for month in months:
            for priority in prioridades:
                valueNumerator = newColumnsDF[(newColumnsDF['country_code'] == country) & (
                    newColumnsDF['month_'] == month) & (newColumnsDF['priority'] == priority)]['overdue_orders'].values
                valueDenominador = newColumnsDF[(newColumnsDF['country_code'] == country) & (
                    newColumnsDF['month_'] == month) & (newColumnsDF['priority'] == priority)]['total_complete_orders_for_overdue'].values
                if len(valueNumerator) == 0 or len(valueDenominador) == 0:
                    df.loc[(df['country_code'] == country) & (df['month_'] == month) & (
                        df['priority'] == priority), newColumnToAppend] = 0
                else:
                    value = valueNumerator[0] / valueDenominador[0]
                    df.loc[(df['country_code'] == country) & (df['month_'] == month) & (
                        df['priority'] == priority), newColumnToAppend] = value
    return df


def dividePerCountry(filePath: str = folderToGetData, comparation=False):
    """
    Divide the data per country and save it in a new folder
    """
    # Create a new folder to save the data
    if not os.path.exists(folderToSendData):
        os.makedirs(folderToSendData)
    # Get the list of files in the folder
    files = os.listdir(filePath)
    # Loop over the files
    for file in files:
        # Read the data
        data = pd.read_csv(filePath+file)
        # Get the list of countries
        countries = data['country_code'].unique()
        # Loop over the countries
        for country in countries:
            # Get the data per country
            dataCountry = data[data['country_code'] == country]
            dataCountry = dataCountry[dataCountry['priority'].notna()]
            dataCountry = dataCountry[dataCountry['priority'] != ' ']
            dataCountry = dataCountry[dataCountry['priority'] != '']
            if 'total_exp_uv' in dataCountry.columns:
                dataCountry.rename(
                    columns={'total_exp_uv': 'exposure_uv'}, inplace=True)
            try:
                dataCountry = dataCountry.sort_values(
                    by=['priority', 'month_'], ascending=[False, True])
            except Exception as e:
                dataCountry = dataCountry.sort_values(
                    by=['week', 'Country', 'priority'], ascending=[False, True, False])
            # Save the data
            dataCountry = addNewColumns(dataCountry)
            dataCountry = addNewColumns2(dataCountry)
            if comparation:
                dataCountry.to_csv(
                    f'{folderToSendData}/'+f'{file[:3]}{country}_data'+'.csv', index=False)
            else:
                dataCountry.to_csv(
                    f'{folderToSendData}/'+f'{country}_data.csv', index=False)

    return


if __name__ == '__main__':
    dividePerCountry()
    print('Data divided successfully')
