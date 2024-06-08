import numpy as np
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as optimization
NUM_TRADINGDAYS=252
NUM_Portfolios=10000
stocks=['AAPL','WMT','TSLA','GE','AMZN','DB']
start_date='2012-01-01'
end_date='2017-01-01'

def downloaddata():
    stockdata={}
    for stock in stocks:
        ticker=yf.Ticker(stock)
        stockdata[stock]=ticker.history(start=start_date,end=end_date)['Close']
    return pd.DataFrame(stockdata)
def showdata(data):
    data.plot(figsize=(10,5))
    plt.show()
def calcreturn(data):
    log_return=np.log(data/data.shift(1))
    return log_return[1:]
def showstats(returns):
    print(returns.mean()*NUM_TRADINGDAYS)
    print(returns.cov()*NUM_TRADINGDAYS)
def showmeanvar(returns,weights):
    portfolioreturn=np.sum(returns.mean()*weights)*NUM_TRADINGDAYS
    portfoliovolatility=np.sqrt(np.dot(weights.T,np.dot(returns.cov()*NUM_TRADINGDAYS,weights)))
    print(portfolioreturn)
    print(portfoliovolatility)
def generateportfolios(returns):
    portfoliomeans=[]
    portfoliorisks=[]
    portfolioweights=[]
    for _ in range(NUM_Portfolios):
        w=np.random.random(len(stocks))
        w/=np.sum(w)
        portfolioweights.append(w)
        portfoliomeans.append(np.sum(returns.mean()*w)*NUM_TRADINGDAYS)
        portfoliorisks.append(np.sqrt(np.dot(w.T,np.dot(returns.cov()*NUM_TRADINGDAYS,w))))
    return np.array(portfolioweights), np.array(portfoliomeans), np.array(portfoliorisks)
def showportfolios(returns,volatilities):
    plt.figure(figsize=(10,6))
    plt.scatter(volatilities,returns,c=returns/volatilities,marker='o')
    plt.grid(True)
    plt.xlabel('Expected Volatility')
    plt.ylabel('Expected Return')
    plt.colorbar(label='Sharpe Ratio')
    plt.show()
def stats(weights,returns):
    portfolioreturn = np.sum(returns.mean() * weights) * NUM_TRADINGDAYS
    portfoliovolatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * NUM_TRADINGDAYS, weights)))
    return np.array([portfolioreturn,portfoliovolatility,portfolioreturn/portfoliovolatility])
def minfuncsharpe(weights,returns):
    return -stats(weights,returns)[2]
def optimizeportfolio(weights,returns):
    constraints={'type':'eq', 'fun':lambda x:np.sum(x)-1}
    bounds=tuple((0,1) for _ in range(len(stocks)))
    return optimization.minimize(fun=minfuncsharpe,x0=weights[0],args=returns,method='SLSQP',bounds=bounds,constraints=constraints)
def printoptimalportfolio(optimum,returns):
    print("Optimal Portfolio:",optimum['x'].round(3))
    print("Expected return,volatility,Sharpe ratio:",stats(optimum['x'].round(3),returns))
def showoptportfolios(opt,rets,portfoliorets,portfoliovols):
    plt.figure(figsize=(10,6))
    plt.scatter(portfoliovols,portfoliorets,c=portfoliorets/portfoliovols,marker='o')
    plt.grid(True)
    plt.xlabel('Expected Volatility')
    plt.ylabel('Expected Return')
    plt.colorbar(label='Sharpe Ratio')
    plt.plot(stats(opt['x'],rets)[1],stats(opt['x'],rets)[0],'g*',markersize=20.)
    plt.show()
if __name__=='__main__':
    dataset=downloaddata()
    showdata(dataset)
    log_dailyreturns=calcreturn(dataset)
    showstats((log_dailyreturns))
    weights,means,risks=generateportfolios(log_dailyreturns)
    showportfolios(means,risks)
    optimum=optimizeportfolio(weights,log_dailyreturns)
    printoptimalportfolio(optimum,log_dailyreturns)
    showoptportfolios(optimum,log_dailyreturns,means,risks)