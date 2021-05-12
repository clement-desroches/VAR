"""
Created on Wed May 5 09:47:30 2021

@author: CD8590
based on the work of AD0541

"""


from Portfolio import *
import matplotlib.pyplot as plt

## intializing the object
p = Portfolio()
## importing historical data
p.import_historical_data()
## filtering historical data with portfolio positions
p.interpolate_portfolio_historic()
## interpolating risk free rates with positions
p.interpolate_rates()
## computing all simulating varibales for each position
p.simulate_variables()
## computing the value of the porfolio each day
p.pricing_portfolio()
## results 
plt.plot(p.valuation.index, p.valuation.Values)
plt.xticks(rotation = 45)

print(f"\n\n---- Var95% is {p.get_var(95)}-----\n\n")

print(p.get_positions_var(95))