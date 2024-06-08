import matplotlib.pyplot as plt
import numpy as np

def simgeometricrandomwalk(S0,T=2,N=1000,mu=0.1,sigma=0.05):
    dt=T/N
    t=np.linspace(0,T,N)
    W=np.random.standard_normal(size=N)
    W=np.cumsum(W)*np.sqrt(dt)
    X=(mu-0.5*sigma**2)*t + sigma*W
    S=S0*np.exp(X)
    return t,S
def plotsim(t,S):
    plt.plot(t,S)
    plt.xlabel("Time (t)")
    plt.ylabel("Stock Price S(t)")
    plt.title("Geometric Brownian Motion")
    plt.show()

if __name__=="__main__":
    time,data=simgeometricrandomwalk(55)
    plotsim(time,data)