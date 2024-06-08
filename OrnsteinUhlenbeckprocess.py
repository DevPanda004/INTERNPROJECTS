import numpy as np
import matplotlib.pyplot as plt

def generateprocess(dt=0.1, theta=1.2, sigma=0.3, mu=0.9, n=10000):
    x=np.zeros(n)
    for t in range(1,n):
        x[t]=x[t-1] + theta*(mu-x[t-1])*dt + sigma*np.random.normal(0,np.sqrt(dt))
    return x
def plotprocess(x):
    plt.plot(x)
    plt.xlabel('t')
    plt.ylabel('x(t)')
    plt.title('Ornstein-Uhlenbeck Process')
    plt.show()

if __name__=="__main__":
    data=generateprocess()
    plotprocess(data)