import pandas as pd

csvToOpen = 'dataDynamic/NewPriorityAnalysisDynamic.csv'
toSaveFolder = 'tables'
df = pd.read_csv(csvToOpen)

paises = df['country_code'].unique()
prioridades = df['priority'].unique()
columnasImportantes = ['priority', 'daily_orders', 'eff_online_rs', 'orders_per_eff_online', 'ted_gmv', 'b_p1p2',
                       'bad_rating_rate', 'imperfect_order_rate', 'b_cancel_rate',]
columns = columnasImportantes
df = df[df['month_'] == 6]
df = df.dropna(subset=['priority'])
for pais in paises:
    newDF = df[df['country_code'] == pais]
    # put a order by priority
    newDF = newDF.sort_values(by=['priority'], ascending=True)
    # show just the columns columns
    newDF = newDF[columns]
    # put % in the rate columns
    newDF['b_cancel_rate'] = newDF['b_cancel_rate'].apply(
        lambda x: f'{x*100:.2f}%')
    newDF['bad_rating_rate'] = newDF['bad_rating_rate'].apply(
        lambda x: f'{x*100:.2f}%')
    newDF['imperfect_order_rate'] = newDF['imperfect_order_rate'].apply(
        lambda x: f'{x*100:.2f}%')
    newDF['ted_gmv'] = newDF['ted_gmv'].apply(
        lambda x: f'{x*100:.2f}%')
    newDF['b_p1p2'] = newDF['b_p1p2'].apply(
        lambda x: f'{x*100:.2f}%')
    newDF = newDF.reset_index(drop=True)
    # newDF = newDF[newDF['priority'] != 'New Rs']
    newDF.to_csv(f'{toSaveFolder}/{pais}_priority.csv', index=False)
