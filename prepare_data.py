import pandas as pd 
import os
import glob
import datetime
import numpy as np

max_hours_back = 3
day_start = 9


def adjust_pre_split(dfi, split_factor=1):
    price_feats = ['marketOpen', 'marketHigh', 'marketLow','marketClose', 'marketAverage',]
    for f in price_feats:
        dfi[f] = dfi[f] / split_factor

    vol_feats = ['marketVolume']
    for f in price_feats:
        dfi[f] = dfi[f] * split_factor 
    return dfi

# marketNotional = df['marketAverage'] * df['marketVolume'].plot()
# marketVolume = marketNumberOfTrades * Number of shares per trade
all_data_list = []
for csv in glob.glob('./dataset/*.csv'):
    symbol = os.path.basename(csv).split('_')[0]
    df = pd.read_csv(csv)
    df['sym'] = symbol
    df = df[['date','minute', 'sym','marketOpen', 'marketHigh', 'marketLow',
             'marketClose', 'marketAverage', 'marketVolume', 'marketNumberOfTrades']]
    print(csv, csv.endswith('TSLA_July.csv'))
    if csv.endswith('TSLA_July.csv'):
        print('================================')
        df = adjust_pre_split(df.copy(), split_factor=5)
    if csv=='AAPL_Sep.csv':
        df = adjust_pre_split(df.copy(), split_factor=4)
    
    df['volumeReference'] = df['marketVolume'].median()
    df['ntradesReference'] = df['marketNumberOfTrades'].median()
    df['day_close'] = np.nan
    df['next_slot_close'] = df['marketClose'].shift(-1)
    for ks in range(max_hours_back):
        df['prevClose_'+str(ks)] = df['marketClose'].shift(ks*2)

    dates = df['date'].unique()
    for d in dates:
        sub_df = df[df['date']==d]
        day_close = sub_df.iloc[-1]['marketClose']
        df.loc[df['date']==d, ['day_close']] = day_close

        date_inds = np.where(df['date']==d)[0] 
        df.loc[date_inds[-1],['next_slot_close']] = df.loc[date_inds[-2],['next_slot_close']]


    df['date'] = df.apply(lambda row: row[0] + ' ' + row['minute'], axis=1)
    df['hour'] = df['date'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%d %H:%M").hour)
    df['min'] = df['date'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%d %H:%M").minute)
    ref_val = df.iloc[0]['marketOpen']
    for k, row in df.iterrows():
        if row['hour']==9 and row['min']==30:
            ref_val = row['marketOpen']
        row['day_ref_val'] = ref_val
        if row['hour'] >= day_start:
            all_data_list.append(row)

full_df = pd.DataFrame(all_data_list)
full_df = full_df.drop(['minute', 'hour'], axis=1)

full_df['%change'] = 100 * (full_df['marketClose'] - full_df['marketOpen'])  / full_df['marketOpen']
full_df['%change_next'] = 100 * (full_df['next_slot_close'] - full_df['marketOpen'])  / full_df['marketOpen']


full_df.to_csv('dataset.csv', float_format='%.2f')

def convert_change_to_target(perc_ch):
    target = 'hold'
    if perc_ch < -.1:
        target = 'sell'
    elif perc_ch >= 0.1 :
        target = 'buy'
    return target

full_df['target'] = full_df['%change_next'].apply(lambda x: convert_change_to_target(x))

# normalized dataset
ndf = full_df.copy()
normalized_feats = [
    'marketOpen', 'marketHigh', 'marketLow', 'marketClose', 'marketAverage',
    'day_close', 'next_slot_close', 'prevClose_0', 'prevClose_1', 'prevClose_2']
    # 'prevClose_3', 'prevClose_4', 'prevClose_5', 'prevClose_6', 'prevClose_7',
    # 'prevClose_8', 'prevClose_9']
ndf['trdingPrice'] = ndf['marketClose']
for f in normalized_feats:
    ndf[f] = ndf[f] / ndf['day_ref_val'] - 1
ndf['marketVolume_normalized'] = ndf['marketVolume'] / ndf['volumeReference']
ndf['marketNumberOfTrades'] = ndf['marketNumberOfTrades'] / ndf['ntradesReference']
ndf = ndf.drop(['volumeReference', 'ntradesReference', 'min', 'day_close', 'next_slot_close'], axis=1)
ndf = ndf.sort_values(by=['sym', 'date'])
ndf.reset_index(drop=True, inplace=True)

ndf.to_csv('normalized_dataset.csv', float_format='%.3f')




print('mean absoloute error in estimating % change, assuming no change:', np.abs(full_df['%change']).mean())
print('mean absoloute error in estimating % next change, assuming no change:', np.abs(full_df['%change_next']).mean())

full_df['%change_prev'] = full_df['%change'].shift(-1)
full_df['naive_error'] = full_df['%change_prev'] - full_df['%change']
print('mean absoloute error in estimating % change, assuming constant change:', np.abs(full_df['naive_error']).mean())
full_df['naive_error'] = full_df['%change_next'] - full_df['%change']
print('mean absoloute error in estimating % next change, assuming constant change:', np.abs(full_df['naive_error']).mean())

