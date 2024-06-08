import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

NUM_SIMULATIONS=1000
NUM_POINTS=200
def montecarlo(x,r0,kappa,theta,sigma,T=1.):
    dt=T/float(NUM_POINTS)
    result=[]

    for _ in range(NUM_SIMULATIONS):
        rates=[r0]
        for _ in range(NUM_POINTS):
            dr = kappa * (theta - rates[-1]) * dt + sigma * np.sqrt(dt) * np.random.normal()
            rates.append(rates[-1] + dr)
        result.append(rates)
    simulation_data=pd.DataFrame(result)
    simulation_data=simulation_data.T
    integral_sum=simulation_data.sum()*dt
    present_integralsum=np.exp(-integral_sum)
    bond_price=x*np.mean(present_integralsum)
    print("Bond price based on Monte-Carlo simulation is: %.2f" %bond_price)

if __name__=='__main__':
    montecarlo(1000,0.1,0.3,0.3,0.03)