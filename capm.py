import numpy as np
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
Riskfreerate=0.05
Months=12
class CAPM:
    def __init__(self,stocks,startdate,enddate):
        self.stocks=stocks
        self.data=None
        self.start_date=startdate
        self.end_date=enddate
    def downloaddata(self):
        data={}
        for stock in self.stocks:
            ticker=yf.download(stock,self.start_date,self.end_date)
            data[stock]=ticker['Adj Close']
        return pd.DataFrame(data)
    def initialise(self):
        stockdata=self.downloaddata()
        stockdata=stockdata.resample('ME').last()
        self.data=pd.DataFrame({'s_adjclose': stockdata[self.stocks[0]],
                                'm_adjclose': stockdata[self.stocks[1]]})
        self.data[['s_returns','m_returns']]=np.log(self.data[['s_adjclose','m_adjclose']]/
                                                    self.data[['s_adjclose','m_adjclose']].shift(1))
        self.data=self.data[1:]
    def calcbeta(self):
        covmatrix=np.cov(self.data['s_returns'],self.data['m_returns'])
        beta=covmatrix[0,1]/covmatrix[1,1]
        print("Beta from formula:", beta)
    def plotregression(self,alpha,beta):
        fig,axis=plt.subplots(1,figsize=(20,10))
        axis.scatter(self.data['m_returns'],self.data['s_returns'],label="Data Points")
        axis.plot(self.data['m_returns'],beta*self.data['m_returns']+alpha,color='red',label="CAPM line")
        plt.title("CAPM")
        plt.xlabel('Market return $R_m$',fontsize=18)
        plt.ylabel('Stock return $R_a$')
        plt.text(0.08,0.05,r'$R_a=\beta*R_m + \alpha$',fontsize=18)
        plt.legend()
        plt.grid(True)
        plt.show()
    def regression(self):
        beta,alpha=np.polyfit(self.data['m_returns'],self.data['s_returns'],deg=1)
        print("Beta from regression:",beta)
        expectedreturn=Riskfreerate+beta*(self.data['m_returns'].mean()*Months-Riskfreerate)
        print("Expected return:",expectedreturn)
        self.plotregression(alpha,beta)

if __name__=='__main__':
    capm=CAPM(['IBM','^GSPC'],'2010-01-01','2017-01-01')
    capm.initialise()
    capm.calcbeta()
    capm.regression()