
"""
Created on Wed May 5 09:47:30 2021

@author: CD8590
based on the work of AD0541
code for representing portfolios, 
positions are grouped by underlying and product type
"""

import datetime as dt

import numpy as np
import pandas as pd
import scipy.stats as si
from Pricing import *
from progress.bar import ChargingBar
from tqdm import tqdm

class Portfolio:

    def __init__(self, overview_path="Copy of 20210504_EQD_VaR_New.xlsm", sheet='Positions'):
        # print(overview_path)
        self.portfolio = pd.read_excel(overview_path, sheet, header=0, index_col = None)
        self.portfolio = self.portfolio[self.portfolio.columns[:17]].loc[:93]
        self.portfolio.rename(columns = {'Balance' : 'Quantity', 'Contract close date' : 'Expiry', 'Strike (K)' : 'Strike',
                                        'Underlying' : 'Undr_Bloomberg', 'Curncy' : 'Currency' , 'Security type' : 'Type',
                                        'Description' : 'Name'}, inplace = True)
        print(self.portfolio)
        self.portfolio = self.portfolio[(self.portfolio['Expiry'] > dt.datetime.now()) | (self.portfolio['Expiry'].isnull())] # drop the positions expirated
        ## Changing name of columns to ease calling
        #self.portfolio.rename(columns = {'@KDR': 'KDR', 'Id Instrument' : 'Id_Instrument','Undr Name': 'Undr_Name',
#                             'Avg price' : 'Avg_Price','BC Market value' : 'BC_Market_value',
#                             'Date in' : 'Date_in', 'BC DTD P&L' : 'BC_DTD_P&L',
#                             'BC YTD P&L' : 'BC_YTD_P&L', 'Interest rate': 'Interest_rate',
#                             'BC Delta Cash': 'BC_Delta_Cash', 'Delta Cash' : 'Delta_Cash', 'Gamma Cash': 'Gamma_Cash',
#                             'OTM Add-on' : 'OTM_Add-on', 'Undr@Bloomberg' : 'Undr_Bloomberg',
#                             '@Bloomberg' : 'Bloomberg'}, inplace = True)
#        ## re-initializing index
        self.portfolio.set_index([list(range(self.portfolio.shape[0]))], inplace = True)
        ## Changing types :
        self.portfolio['Expiry'] = pd.to_datetime(self.portfolio['Expiry'])
        self.portfolio['Undr_Bloomberg'] = self.portfolio['Undr_Bloomberg'].str.lower()

        ## Date on which VaR will be computed
        self.date = dt.datetime(2021,5,4)# dt.datetime(int(overview_path[-13:-9]), int(overview_path[-9:-7]), int(overview_path[-7:-5]))
        
        
        
    def get_underlyings(self):
        """
        Method to get the underlyings of the different positions
        Inputs :
            - No inputs
        Outputs :
            - np.array of the underlyings
            IMPORTANT : self.import_historical_data() and self.interpolate_portfolio_historic must have been called"""
        return self.portfolio['Undr_Bloomberg'].unique()
    
    
    def get_missing_information(self):
        """
        Method to get the underlyings of which information is missing
        IMPORTANT : self.import_historical_data() and self.interpolate_portfolio_historic must have been called
        """
        missing=[]
        ## Test if the underlyings of positions are in the columns of historical data
        test= [self.get_underlyings()[i] in self.historic.indexes.dropna(axis = 1, how = 'any').columns.values for i in range(len(self.get_underlyings()))]
        ## keep only the underlyings which aren't in historical data
        for i in range(len(test)):
            if not test[i]:
                missing.append(self.get_underlyings()[i])
        if len(missing) == 0:
            return 'No missing information'
        return (f"We don't have infomation about {missing}")
    
    
    
    def import_historical_data(self, path = 'Copy of 20210504_EQD_VaR_New.xlsm', sheet = 'VaRData'):
        """
        Method to get historical values from 1000 days of equities 
        in order to prepare the pricing of the portfolio
        Inputs :
            - self : Porfolio object containing information about the positions
            - path (string) : path to get the data (should be the same model as VaR-data.xlsm
            - sheet (string) : sheet of the excel to get the data
            - date (datetime) : date limit of the data (should be equal to the portfolio date)
            # optionnel, je vais surement l'enlever et stocker la date dès le départ
        Outputs :
            -No return : create the attribute historic"""
        ## IMPORTING
        self.historic = pd.read_excel(path, sheet)
        #self.info = pd.read_excel(path, 'Macro', header = 62)
        ## Changing date tyope into dt.datetime and set it as index
        self.historic['Date'] = pd.to_datetime(self.historic['Date'])
        ## keeping only 1000 days before self.date
        j = self.historic.index[self.historic[self.historic.columns[0]] == self.date][0]
        i = j-1001
        self.historic = self.historic.loc[i:j]
        ## set Date as index
        self.historic.set_index('Date', inplace = True)
        ## lowering every name of underlyings in order to avoid errors after
        self.historic.rename(str.lower, axis='columns', inplace = True)
        print("\n\n\n----- Done importing historical Data -----\n\n\n")
    
    
    
    def interpolate_portfolio_historic(self):
        """
        Method to filter the hitorical data in order to keep only current positions
        Inputs :
            - self : Portfolio obect
        Outputs : 
            - no return: modified directly object (only self.historic is modified)"""
        ## gathering underlyings and volatility to keep
        undr = self.get_underlyings()
        vol = self.get_underlyings()+'_12mo_call_imp_vol'
        rates  = ['eonia index',	'eusa1 curncy',	'eusa2 curncy',	'eusa3 curncy',	'eusa4 curncy',	'eusa5 curncy',	'eusa7 curncy',
                  'eusa10 curncy',	'eusa15 curncy',	'eusa20 curncy',	'eusa30 curncy',
                  'sfdr1t curncy',	'sfsw1 curncy',	'sfsw2 curncy',	'sfsw3 curncy',	'sfsw4 curncy',	'sfsw5 curncy',	'sfsw7 curncy',	'sfsw10 curncy',	
                  'sfsw15 curncy',	'sfsw20 curncy',	'sfsw30 curncy',
                  'usdr1t curncy',	'ussw1 curncy',	'ussw2 curncy',	'ussw3 curncy',	'ussw4 curncy',	'ussw5 curncy',	'ussw7 curncy',	'ussw10 curncy',
                  'ussw15 curncy',	'ussw20 curncy',	'ussw30 curncy']
        ## filtering historical data to keep underlyings of positions and their 1Y volatility
        self.historic.indexes = self.historic.loc[:, undr]
        self.historic.vol = self.historic.loc[:,  vol]
        self.historic.rates = self.historic.loc[:, rates]
        self.historic.rates.rename(columns = {'eonia index' : 'eur0 curncy','eusa1 curncy' : 'eur1 curncy',	'eusa2 curncy' : 'eur2 curncy',	
                                              'eusa3 curncy': 'eur3 curncy',	'eusa4 curncy' : 'eur4 curncy',	'eusa5 curncy' : 'eur5 curncy',
                                              'eusa7 curncy': 'eur7 curncy', 'eusa10 curncy' : 'eur10 curncy',	'eusa15 curncy': 'eur15 curncy',	
                                              'eusa20 curncy' : 'eur20 curncy',	'eusa30 curncy': 'eur30 curncy',
                                              'usdr1t curncy' : 'usd0 curncy','ussw1 curncy': 'usd1 curncy',	'ussw2 curncy' : 'usd2 curncy',	
                                              'ussw3 curncy' : 'usd3 curncy',	'ussw4 curncy' : 'usd4 curncy',	'ussw5 curncy' : 'usd5 curncy',	
                                              'ussw7 curncy' : 'usd7 curncy',	'ussw10 curncy' : 'usd10 curncy', 'ussw15 curncy': 'usd15 curncy',	
                                              'ussw20 curncy' : 'usd20 curncy',	'ussw30 curncy': 'usd30 curncy',
                                              'sfdr1t curncy' : 'chf0 curncy',	'sfsw1 curncy' : 'chf1 curncy',	'sfsw2 curncy' : 'chf2 curncy',	
                                              'sfsw3 curncy': 'chf3 curncy',	'sfsw4 curncy' : 'chf4 curncy',	'sfsw5 curncy' : 'chf5 curncy',	
                                              'sfsw7 curncy' : 'chf7 curncy',	'sfsw10 curncy' : 'chf10 curncy', 'sfsw15 curncy' : 'chf15 curncy',	
                                              'sfsw20 curncy' : 'chf20 curncy',	'sfsw30 curncy' : 'chf30 curncy',}, inplace = True)
        ## Changes rates
        self.changes = self.historic.loc[:, ['chf', 'usd']]
        self.changes['eur'] = 1
        self.changes.rename(columns = {'chf' : 'CHF', 'usd' : 'USD', 'eur' : 'EUR'}, inplace = True)
        self.portfolio = self.portfolio[self.portfolio['Undr_Bloomberg'].isin(undr)]
        self.portfolio.set_index([list(range(self.portfolio.shape[0]))], inplace = True)
        ## Creation of objects regarding each type of positions :
        self.options = self.portfolio.loc[self.portfolio['Type'].str.contains('Option')]
        self.options.loc[:,'option_type']= 1
        self.options.loc[:,'option_type'].loc[self.options.loc[:,'Type'].str.contains('Put')] = -1
        self.options.loc[:,'Quotity']=1
        self.equities = self.portfolio.loc[self.portfolio['Type'].str.contains('Equity')]
        print("\n\n\n----- Done interpolating Data -----\n\n\n")

    def interpolate_rates(self):
        """
        Method which creates the interpolation between options and their rates
        inputs :
            -no inputs
        Outputs:
            - self.rates (pd.dataframe) : values of rates for each option in columns everyday in rows
        IMPORTANT : self.import_historical_data() and self.interpolate_portfolio_historic must have been called"""
        dates = self.historic.index
        self.rates = pd.DataFrame( columns = self.options['Name'] + '_rfr')
        
        ttm = ((self.options['Expiry'] - self.date).dt.days / 365).tolist()
        ## Rates indexes we have
        L = [0,1,2,3,4,5,7,10,15,20,30]
        ## Nearest neighbour of ttm in L
        distances = []
        for i in range(len(ttm)):
            dist = {str(index) : np.abs(index - ttm[i]) for index in L}
            dist = sorted(dist.items(), key=lambda x: x[1])
            distances.append(dist)
        currencies = self.options['Currency'].str.lower().values
        ## gathering values of each rate for each day
        for i, day in enumerate(dates) :
            df = (self.historic.rates.loc[day, currencies + np.array([distances[i][0][0] for i in range(len(distances))]) + np.array([' curncy']*len(ttm))].values * np.array([distances[i][1][1] for i in range(len(distances))]) + self.historic.rates.loc[day, currencies + np.array([distances[i][1][0] for i in range(len(distances))]) + np.array([' curncy']*len(ttm))].values * np.array([distances[i][0][1] for i in range(len(distances))]))/ np.abs(np.array([int(distances[i][1][0]) for i in range(len(distances))])-np.array([int(distances[i][0][0]) for i in range(len(distances))]))
            self.rates = self.rates.append(pd.DataFrame(df, index = self.options.Name + '_rfr').transpose(), ignore_index = True)
        ## reindexing with dates
        self.rates['Date'] = dates.values
        self.rates.set_index('Date', inplace = True)
        return self.rates
    
    
    def simulate_variables(self):
        """
        Simulate prices, vol, and rates of each positions each day
        Inputs :
            - No inputs
        Outputs : No return but stocks the results in the object Portfolio
        IMPORTANT : self.import_historical_data() and self.interpolate_portfolio_historic and self.interpolate_rates must have been called"""
        print("----- Simulation start -----\n\n")
        ## reation of different dataframe
        dates = self.historic.index
        self.simulate_prices_options = pd.DataFrame(columns = self.options['Undr_Bloomberg'])
        self.simulate_prices_equities = pd.DataFrame(columns = self.equities['Undr_Bloomberg'])        
        self.simulate_vol = pd.DataFrame(columns = self.options['Name']+'_vol')
        self.simulate_rfr = pd.DataFrame(columns = self.options['Name']+'_rfr')
        ## starting compute
        for i, day in enumerate(tqdm(dates[1:], desc = 'Simulating : ', colour = 'green', ncols =100)) :
            day2 = dates[:i+1][-1] # business day before day
            ## simulate prices in euro of every underlying of the options (changing rate to simulate too)
            prices_options = self.historic.loc[self.date, self.options['Undr_Bloomberg']].values * self.changes.loc[self.date, 
                                              self.options['Currency']].values * self.changes.loc[day, 
                                                          self.options['Currency']].values / self.changes.loc[day2, 
                                                                      self.options['Currency']].values * self.historic.loc[day, 
                                                                                  self.options['Undr_Bloomberg']].values / self.historic.loc[day2, 
                                                                                              self.options['Undr_Bloomberg']].values
            ## formatting and appending to simulate_prices_options
            self.simulate_prices_options = self.simulate_prices_options.append(pd.DataFrame(prices_options, index = self.options['Undr_Bloomberg']).transpose(), ignore_index = True)
            ## simulate prices in euro of every underlying of the equities (changing rate to simulate too)
            prices_equities = self.historic.loc[self.date, self.equities['Undr_Bloomberg']].values * self.changes.loc[self.date, 
                                               self.equities['Currency']].values * self.changes.loc[day, 
                                                            self.equities['Currency']].values / self.changes.loc[day2, 
                                                                         self.equities['Currency']].values * self.historic.loc[day, 
                                                                                      self.equities['Undr_Bloomberg']].values / self.historic.loc[day2, 
                                                                                                   self.equities['Undr_Bloomberg']].values
            ## formatting and appending to simulate_prices_equities
            self.simulate_prices_equities = self.simulate_prices_equities.append(pd.DataFrame(prices_equities, index = self.equities['Undr_Bloomberg']).transpose(), ignore_index = True)
            ## simulate volatility of every option with reference volatility of self.date
            vol = self.options['Volatility sigma'].values * self.historic.loc[day, 
                              self.options['Undr_Bloomberg'] + '_12mo_call_imp_vol'].values / self.historic.loc[day2, 
                                          self.options['Undr_Bloomberg'] + '_12mo_call_imp_vol'].values
            ## formatting and appending to simulate_vol
            self.simulate_vol = self.simulate_vol.append(pd.DataFrame(vol, index = self.options['Name']+'_vol').transpose(), ignore_index = True)
            ## simulate Risk free rates of every option with reference RFR of self.date
            rfr =  self.rates.loc[self.date].values/100 + self.rates.loc[day].values/100-self.rates.loc[day2].values/100
            ## formatting and appending to simulate_rfr
            self.simulate_rfr = self.simulate_rfr.append(pd.DataFrame(rfr, index = self.options['Name']+'_rfr').transpose(), ignore_index = True)
        ## Reindexing with dates
        self.simulate_prices_options['Date'] = dates[1:].values
        self.simulate_prices_options.set_index('Date', inplace = True)
        self.simulate_prices_equities['Date'] = dates[1:].values
        self.simulate_prices_equities.set_index('Date', inplace = True)
        self.simulate_vol['Date'] = dates[1:].values
        self.simulate_vol.set_index('Date', inplace = True)
        self.simulate_rfr['Date'] = dates[1:].values
        self.simulate_rfr.set_index('Date', inplace = True)
        print("\n\n\n----- Done simulating variables -----\n\n\n")


    def pricing_positions(self, i, day): 
        """
        Compute the price of every position at the date : day
        Inputs :
            - day (datetime) : Date to compute prices
        Outputs : No return but stocks the results in the object Portfolio
        IMPORTANT : self.import_historical_data() and self.interpolate_portfolio_historic must have been called"""
        ## Pricing de chacune des positions, see Pricing.py for the formulas 
        dates = self.historic.index
        day2 = dates[:i+1][-1]
        self.options.loc[:, 'pricing_' + day.strftime('%Y%m%d')] = price_option(option_type = self.options['option_type'].values, ttm = ((self.options['Expiry'] - self.date).dt.days / 365).values, 
                                                                 strike = self.options['Strike'].values * self.changes.loc[self.date, self.options['Currency']].values,
                                                                 spot = self.simulate_prices_options.loc[day].values,
                                                                 interest = self.simulate_rfr.loc[day].values,
                                                                 vol = self.simulate_vol.loc[day].values,
                                                                 quotity = self.options.loc[:,'Quotity'].values, size = self.options.loc[:,'Quantity'].values,
                                                                 dividend = self.options['Div Yield (q)'].values)
        
        self.equities.loc[:, 'pricing_' + day.strftime('%Y%m%d')] = price_equity(price = self.simulate_prices_equities.loc[day].values ,  quantity = self.equities.loc[:,'Quantity'].values)  

        
    def pricing_portfolio(self):
        """
        Return a dataframe with values of the portfolio over 1000 days before the date : self.date
        intputs : 
            - No inputs
        Outputs : 
        IMPORTANT : self.import_historical_data() and self.interpolate_portfolio_historic must have been called"""
        print("----- Pricing start -----\n\n")
        Values = []
        ## Dates ton which the portfolio will be computed
        dates = self.historic.index
        dates = dates[1:]
        ## Pricing
        for i, day in enumerate(tqdm(dates, desc = 'Pricing : ', colour = 'green', ncols = 100)):
            
            self.pricing_positions(i, day)
            Values.append(self.options['pricing_' + day.strftime('%Y%m%d')].sum() + 
                                                 self.equities['pricing_' + day.strftime('%Y%m%d')].sum())
        ## formatting results
        self.valuation = pd.DataFrame({'Values' : Values}, index = dates )
        self.portfolio = pd.concat([self.options, self.equities], sort = True)
       # self.portfolio.sort_index(inplace = True)
        print("\n\n\n----- Done pricing the portfolio -----\n\n\n")
        
        
    
    def get_var(self, conf):
        """
        Compute the VAR with a confidence : conf
        Inputs :
            - conf (float) : confidence of the Var wanted.
        Outputs :
            - Var (float) : Value of Var(conf%)
        IMPORTANT : self.import_historical_data() and self.interpolate_portfolio_historic and self.pricing_portfolio must have been called"""
        ## Var computing
        df = self.valuation
        self.diff = df.loc[:df.index.values[-2]].values - df.loc[self.date].values[0]
        self.diff = pd.DataFrame(self.diff, columns = ['diff'], index = self.historic.index[1:-1])
        values = np.sort(list(self.diff['diff']))
        return values[(100-conf)*10]
        
        
    def get_underlyings_var(self, conf):
        """
        Method to get variation of the underlyings between today and date of which the var conf% has been selected
        Inputs :
            - conf (float) : confidence of the Var wanted.
        Outputs :
            - self.underlyings_var (pandas.DataFrame) : Variations of the underlyings"""
        ## Getting VaR conf%
        var = self.get_var(conf)
        ## Getting the date of VaR conf%
        dates = self.historic.index
        date1 = self.diff[self.diff == var].index[0]
        
        ## Computing difference between prices of underlyings of the date of Var conf% and self.date
        var_underlyings =  self.historic.loc[date1, self.get_underlyings()] - self.historic.loc[self.date, self.get_underlyings()]
        self.underlyings_var = pd.DataFrame({'Vars' : var_underlyings}, index = self.get_underlyings())
        return self.underlyings_var
    
    
    
    def get_positions_var(self, conf):
        """
        Method to get variation of the positions between today and date of which the var conf% has been selected
        Inputs :
            - conf (float) : confidence of the Var wanted.
        Outputs :
            - self.positions_var (pandas.DataFrame) : Variations of the positions"""
        ## Getting VaR conf%
        var = self.get_var(conf)
        ## Getting the date of VaR conf%
        dates = self.historic.index
        dates = dates[1:]
        date1 = self.diff[self.diff == var].index[0]
        
         ## Computing difference between prices of positions of the date of Var conf% and self.date
        var_pos = self.portfolio['pricing_' + date1.strftime('%Y%m%d')] - self.portfolio['pricing_' + self.date.strftime('%Y%m%d')]
        self.positions_var = pd.DataFrame({'Var'+str(conf): var_pos.values}, index = self.portfolio['Name'])
        return self.positions_var
    
    
    
    def get_underlyings_greeks(self):
        """
        Method to get the cumulative greeks of positions group by underlyings
        Inputs : 
            - no inputs
        Outputs:
            - self.underlyings_greeks (pandas.DataFrame) : DataFrame of cumulatives greeks (index = Undr_Bloomberg)"""
        self.underlyings_greeks = self.portfolio[['Undr_Bloomberg', 'Volatility', 'Interest_rate',
                                                 'Delta_Cash',  'Gamma_Cash', 'Vega', 'Rho', 'Epsilon',
                                                 'Theta',]].groupby(['Undr_Bloomberg']).sum()
        return self.underlyings_greeks
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    