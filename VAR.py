import numpy as np
import yfinance as yf
from scipy.stats import norm
import pandas as pd
import datetime
import math
def download_data(stock, start_date, end_date):
    data={}
    ticker=yf.download(stock, start_date, end_date)
    data[stock]=ticker['Adj Close']
    return pd.DataFrame(data)

def calculate_var_tom(position, c, mu, sigma):
    var= position*(mu - sigma*norm.ppf(1-c))
    return var

def calculate_var_n(position, c, mu, sigma, n):
    var= position*(mu*n - sigma*math.sqrt(n)*norm.ppf(1-c))
    return var

if __name__=='__main__':
    start=datetime.datetime(2014,1,1)
    end=datetime.datetime(2018,1,1)
    stockdata=download_data('C',start,end)
    stockdata['returns']= np.log(stockdata['C']/stockdata['C'].shift(1))
    stockdata=stockdata[1:]
    print(stockdata)
    S = 1e6
    c = 0.99
    mu=np.mean(stockdata['returns'])
    sigma=np.std(stockdata['returns'])
    print('Value at risk tomorrow: $%0.2f' %calculate_var_tom(S,c,mu,sigma))
    print('Value at risk in future: $%0.2f' % calculate_var_n(S, c, mu, sigma, 10))