import numpy.random as npr
import numpy as np
import matplotlib.pyplot as plt

def wienerprocess(dt=0.1,x0=0,n=1000):
    W=np.zeros(n+1)
    t=np.linspace(x0,n,n+1)
    W[1:n+1]=np.cumsum(np.random.normal(0,np.sqrt(dt),n))
    return t,W
def plotprocess(t,W):
    plt.plot(t,W)
    plt.xlabel("Time(t)")
    plt.ylabel("Wiener Process W(t)")
    plt.title("WIENER PROCESS")
    plt.show()

if __name__=="__main__":
    time,data=wienerprocess()
    plotprocess(time,data)