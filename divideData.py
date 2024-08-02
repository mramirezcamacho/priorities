import os

import pandas as pd


folderToGetData = 'dataDynamic/'
folderToSendData = 'dataDynamicDivided'


def dividePerCountry(filePath: str = folderToGetData):
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
            dataCountry.to_csv(
                f'{folderToSendData}/'+f'{file[:3]}{country}_data'+'.csv', index=False)
    return


dividePerCountry()
