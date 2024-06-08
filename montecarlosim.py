import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

Numsimulations=100

def stockmontecarlo(S0,mu,sigma,N=252):
    result=[]
    for _ in range(Numsimulations):
        prices=[S0]
        for _ in range(N):
            stockprice=prices[-1]*np.exp((mu-0.5*sigma**2)+sigma*np.random.normal())
            prices.append(stockprice)
        result.append(prices)
    simulationdata=pd.DataFrame(result)
    simulationdata=simulationdata.T
    plt.plot(simulationdata)
    plt.show()
    simulationdata['mean']=simulationdata.mean(axis=1)
    plt.plot(simulationdata['mean'])
    plt.show()
    print("Predicted future stock price:$%.2f"%simulationdata['mean'].tail(1))

if __name__=="__main__":
    stockmontecarlo(50,0.0002,0.01)