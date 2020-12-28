import pandas as pd 
import os
import pyEX as p
from datetime import date
# https://github.com/d-eremeev/ADM-VRP
print(p.chart)

c = p.Client(api_token="pk_6ff6a3397f864fa99013ac76f64173c6")
# https://pyex.readthedocs.io/en/latest/#

watchlist = ['AAPL', 'FB', 'TSLA', 'COF', 'MSFT', 'AMD', 'AMZN','SPY','UBER','MSI','UAL','BRK.B','GOOGL','RUN','NET','FSLY', 'NFLX','OKTA','SHOP','QQQ','NIO']

today = date.today()
for w in watchlist:
    csv_name = './dataset/' + w + '_' 'NovDec' + '.csv'
    if not os.path.exists(csv_name):
        print('geting infor for symbol: ', w)
        run_df_detailed_daily = c.chartDF(w, timeframe='1mm')
        run_df_detailed_daily.to_csv(csv_name)

    
