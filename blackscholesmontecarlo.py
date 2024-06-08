import numpy as np
class OptionPricing:
    def __init__(self,S0,E,rf,T,sigma,iterations):
        self.S0=S0
        self.E=E
        self.T=T
        self.rf=rf
        self.iterations=iterations
        self.sigma=sigma
    def calloptionsim(self):
        optiondata=np.zeros([self.iterations,2])
        rand=np.random.normal(0,1,[1,self.iterations])
        stockprice=self.S0*np.exp(self.T*(self.rf-0.5*self.sigma**2)+self.sigma*rand*np.sqrt(self.T))
        optiondata[:,1]=stockprice-self.E
        average=np.sum(np.amax(optiondata,axis=1))/float(self.iterations)
        return average*np.exp(-1.0*self.rf*self.T)
    def putoptionsim(self):
        optiondata=np.zeros([self.iterations,2])
        rand=np.random.normal(0,1,[1,self.iterations])
        stockprice=self.S0*np.exp(self.T*(self.rf-0.5*self.sigma**2)+self.sigma*rand*np.sqrt(self.T))
        optiondata[:,1]=self.E-stockprice
        average=np.sum(np.amax(optiondata,axis=1))/float(self.iterations)
        return average*np.exp(-1.0*self.rf*self.T)
if __name__=="__main__":
    model=OptionPricing(100,100,0.05,1,0.2,10000)
    print("Call option price:",model.calloptionsim())
    print("Put option price:",model.putoptionsim())