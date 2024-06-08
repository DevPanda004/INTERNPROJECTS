from math import exp
def futurediscrete(x,r,n):
    return x*(1+r)**n
def presentdiscrete(x,r,n):
    return x*(1+r)**-n
def futurecont(x,r,n):
    return x*exp(r*n)
def presentcont(x,r,n):
    return x*exp(-r*n)

if __name__== '__main__':
    x=100
    r=0.05
    n=5
print(futurediscrete(x,r,n))
print(futurecont(x,r,n))
print(presentdiscrete(x,r,n))
print(presentcont(x,r,n))
