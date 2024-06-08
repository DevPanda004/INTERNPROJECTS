from math import exp
class ZeroCouponBonds:
    def __init__(self,principal,maturity,marketinterest):
        self.principal=principal
        self.maturity=maturity
        self.interest=marketinterest/100
    def presentval(self,x,n):
        return x*exp(-self.interest*n)
    def calculateprice(self):
        return self.presentval(self.principal,self.maturity)
class CouponBonds:
    def __init__(self,principal,rate,maturity,marketinterest):
        self.principal=principal
        self.rate=rate/100
        self.maturity=maturity
        self.marketinterest=marketinterest/100
    def presentval(self,x,n):
        return x*exp(-self.marketinterest*n)
    def calculateprice(self):
        price=0
        for t in range(1,self.maturity+1):
            price+=self.presentval(self.principal*self.rate,t)
        price+=self.presentval(self.principal,self.maturity)
        return price
if __name__=='__main__':
    bond=ZeroCouponBonds(1000,2,4)
    print(bond.calculateprice())

