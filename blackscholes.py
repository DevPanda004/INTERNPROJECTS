from scipy import stats
from numpy import log,exp,sqrt

def calloptionprice(S,E,T,rf,sigma):
    d1=(log(S/E)+(rf+sigma*sigma/2.0)*T)/(sigma*sqrt(T))
    d2=d1-sigma*sqrt(T)
    print("The d1 and d2 parameters are: %s,%s"%(d1,d2))
    return S*stats.norm.cdf(d1)-E*exp(-rf*T)*stats.norm.cdf(d2)
def putoptionprice(S,E,T,rf,sigma):
    d1=(log(S/E)+(rf+sigma*sigma/2.0)*T)/(sigma*sqrt(T))
    d2=d1-sigma*sqrt(T)
    return -S*stats.norm.cdf(-d1)+E*exp(-rf*T)*stats.norm.cdf(-d2)
if __name__=="__main__":
    S0=100
    E=100
    rf=0.05
    T=1
    sigma=0.2
    print(calloptionprice(S0,E,T,rf,sigma))
    print(putoptionprice(S0,E,T,rf,sigma))