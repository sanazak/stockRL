import pandas as pd 
import os
import glob
import datetime
import numpy as np
from collections import defaultdict
from scipy.special import softmax
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (20,5)

class Acount:
    def __init__(self, cash=1000):
        self.cash = cash
        self.positions =  defaultdict(int)
        self.n_trades = 0
    def buy(self, sym, N, price):
        cash_needed = price * N
        if self.cash < cash_needed:
            N = 0
            # print('Not enough cash to buy {}  of {} ({} needed, cash balanace: ${}'.format(N, sym, round(cash_needed, 2), round(self.cash,2)))
        else:
            self.positions[sym] += N
            self.cash -= cash_needed
            self.n_trades += 1
            # print('Bought {}  of {}, spent {}, new cash balance: ${}'.format(N, sym, round(cash_needed, 2), round(self.cash)))
    
    def sell(self, sym, N, price):
        cash_value = price * N
        if self.positions[sym] < N:
            N = N
            # print('Requested to sell {} of {}, but only have {}'.format(N, sym, self.positions[sym]))
        else:
            self.positions[sym] -= N
            self.cash += cash_value
            self.n_trades += 1
            # print('Sold {}  of {}, gained {}, new cash balance: ${}'.format(N, sym, cash_value, round(self.cash, 2)))
       
    def get_total_assets(self):
        return 


def buy_decision(df_s, cash):
    # df_s.sort_values(by=['%change'], ascending=False)
    df_s['rand'] =  np.random.randint(1, 100, df_s.shape[0])
    df_s = df_s[df_s['%change'] > 0.02]
    if len(df_s):
        df_s['weight'] = softmax(df_s['%change'].values)
        df_s['cash_per_sym'] = df_s['weight'] * cash / 3
        df_s['N_buy'] = np.round(df_s['cash_per_sym'] / df_s['trdingPrice'])
        df_s = df_s[df_s['N_buy']>0]
        return list(zip(df_s.sym, df_s.N_buy, df_s.trdingPrice))
    else:
        return []

def sell_decision(df_s, acc):
    # df_s.sort_values(by=['%change'], ascending=False)
    df_s['N_sell'] = df_s['sym'].apply(lambda x: acc.positions[x])
    df_s = df_s[df_s['N_sell']>0]
    return list(zip(df_s.sym, df_s.N_sell, df_s.trdingPrice))




my_acc = Acount(5000)
# my_acc.buy('AAPL', 10, 22)
print(my_acc.positions)


df = pd.read_csv('normalized_dataset.csv', index_col=0)
df = df[df['sym']!='TSLA']
df = df[df['sym']!='AAPL']
df = df[df['sym']!='RUN']


print(len(df))
df = df.dropna()
print(len(df))
df = df.sort_values(by=['date'])
times_list = df['date'].unique()
acc_val_history = []
cass_history = []
k = 0
for t in times_list:
    if '00' in t or '30' in t:
        cur_df = df[(df['date']==t)]



        order_list = buy_decision(cur_df, my_acc.cash)
        for order in order_list:
            my_acc.buy( order[0], order[1], order[2])

        sell_df = df[(df['date']==t) & (df['%change'] < - 0.02)].copy()
        order_list = sell_decision(sell_df, my_acc)
        for order in order_list:
            my_acc.sell( order[0], order[1], order[2])
        
        acc_val = round(my_acc.cash  + np.sum([N * cur_df[cur_df['sym']==sym]['trdingPrice'] for sym, N in my_acc.positions.items()]), 1)
    
        pos_={k: v for k, v in my_acc.positions.items() if v}
        print(t, pos_, 'acc_val = ', acc_val)

    acc_val = round(my_acc.cash  + np.sum([N * cur_df[cur_df['sym']==sym]['trdingPrice'] for sym, N in my_acc.positions.items()]), 1)
    acc_val_history.append(acc_val)
    cass_history.append(my_acc.cash)

    k = (k+1) % 1


plt.plot(acc_val_history)
# plt.plot(cass_history)
plt.xticks(range(0,len(times_list)), times_list,  rotation = 'vertical')
plt.gca().xaxis.set_major_locator(plt.MultipleLocator(10))
plt.show()
print(my_acc.n_trades)
