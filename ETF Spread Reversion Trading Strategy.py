#!/usr/bin/env python
# coding: utf-8

# # ETF Spread Reversion Trading

# # 1. Introduction:

# Spread trading strategies involve exploiting the price differences between two related financial instruments. It's very common in futures trading. By looking at the difference of the returns of these two instruments, it helps the trader to trade on the convergence of the spread. The following work look at a specific spread trading strategy for two Exchange-traded funds (ETFs), in this case PBE/XBI. The strategy works on the concept of mean reversion, where the trades take place based on two different parameters that guide the entry and exit to the market.
#     
# 
# The following work is a comprehensive analysis of the proposed trading strategy over a specified period, from January 01, 2022, to November 15, 2023. The strategy's performance is evaluated against various parameters, including the stop-loss threshold, trading costs, and the capital employed. Additionally, the strategy is analyzed for its correlation with Fama-French factor returns, providing insights into its risk profile and market sensitivities. The objective is to ascertain the efficacy of the strategy under different market conditions and parameter settings, thus providing a thorough understanding of its potential as a trading tool. 

# 
# Import various python modules that are to be used for obtaining, preparing, analysing the data and implement the trading strategy.

# In[60]:


import os
import datetime
import shutil
import requests

import pandas as pd
import numpy as np
import quandl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as sp
import plotnine as p9
import functools
import itertools
import random
import warnings
warnings.filterwarnings('ignore')
from statsmodels.graphics.tsaplots import plot_acf
from scipy.stats import probplot
from datetime import datetime
from io import BytesIO
from io import StringIO
from zipfile import ZipFile
from pandas.plotting import register_matplotlib_converters
from plotnine import ggplot, aes, geom_line, labs, scale_x_date, theme_minimal, element_text, theme, geom_ribbon, geom_point
from mizani.breaks import date_breaks
from mizani.formatters import date_format
register_matplotlib_converters()


# # 2. Obtain Data from Quandl server and Data Preparation for analysis:

# For obtaining data from the Quandl server, two tables have been used grab_quandl_table and fetch_quandl_table. grab_quandl_table function is responsible for downloading the table from the Quandl database and storing the data in the workstation. The other function fetch_quandl_table which is a wrapper that uses grab-quandl_table function to fetch and read a Quandl data table into a pandas dataframe.

# In[2]:


def grab_quandl_table(
    table_path,
    avoid_download=False,
    replace_existing=False,
    date_override=None,
    allow_old_file=False,
    **kwargs,
):
    root_data_dir = os.path.join("C:", "quandl_data_table_downloads")
    data_symlink = os.path.join(root_data_dir, f"{table_path}_latest.zip")
    if avoid_download and os.path.exists(data_symlink):
        print(f"Skipping any possible download of {table_path}")
        return data_symlink
    
    table_dir = os.path.dirname(data_symlink)
    if not os.path.isdir(table_dir):
        print(f'Creating new data dir {table_dir}')
        os.makedirs(table_dir, exist_ok=True)

    if date_override is None:
        my_date = datetime.now().strftime("%Y%m%d")
    else:
        my_date = date_override
    data_file = os.path.join(root_data_dir, f"{table_path}_{my_date}.zip")

    if os.path.exists(data_file):
        file_size = os.stat(data_file).st_size
        if replace_existing or not file_size > 0:
            print(f"Removing old file {data_file} size {file_size}")
        else:
            print(
                f"Data file {data_file} size {file_size} exists already, no need to download"
            )
            return data_file

    dl = quandl.export_table(table_path, filename=data_file, api_key="9v9zUkZARMYqBCzFPXzZ", **kwargs)
    file_size = os.stat(data_file).st_size
    if os.path.exists(data_file) and file_size > 0:
        print(f"Download finished: {file_size} bytes")
        if not date_override:
            if os.path.exists(data_symlink):
                print(f"Removing old symlink")
                os.unlink(data_symlink)
            print(f"Creating symlink: {data_file} -> {data_symlink}")
            shutil.copyfile(data_file, data_symlink)
    else:
        print(f"Data file {data_file} failed download")
        return
    return data_symlink if (date_override is None or allow_old_file) else "NoFileAvailable"


def fetch_quandl_table(table_path, avoid_download=True, **kwargs):
    return pd.read_csv(
        grab_quandl_table(table_path, avoid_download=avoid_download, **kwargs)
    )


# In[3]:


fetch_quandl_table('QUOTEMEDIA/TICKERS', avoid_download=False).head()


# # The ETF pair to create a spread trading strategy is X=PBE and Y= XBI.

# After fetching the relevant tables for PBE, XBI and SVOL tickers. The data has been read in the following code snippets.

# In[4]:


price1 = pd.read_csv(r"C:\Users\nihar\Desktop\pbe_price.csv")


# In[5]:


price2 = pd.read_csv(r"C:\Users\nihar\Desktop\xbi_price.csv")


# In[6]:


price3 = pd.read_csv(r"C:\Users\nihar\Desktop\svol_price.csv")


# Let's look at few rows of the required data:

# In[7]:


price1.head()


# In[8]:


price2.head()


# In[9]:


price3.head()


# Less liquid of PBE/XBI pair is PBE. Daily dollar volume and running 15-day median has been calculated for PBE future prices data. Daily dollar volume is the product of the adjusted close price and volume for the future for the same day.

# In[10]:


#Calculating daily dollar volume for less liquid PBE:
price1['daily_dollar_volume'] = price1['adj_close'] * price1['adj_volume']

#Calculating running 15-day median for less liquid PBE: Nt= Median [{Vt−16, Vt−15, . . . , Vt−1}]
price1['Nt'] = price1['daily_dollar_volume'].rolling(window=15).median().shift(1)


# The PBE price data has been filtered for the required dates 02 Dec 2021 until 15 Nov 2023.

# In[11]:


price1_copy = price1.copy()

# Converting the date column to datetime in the futures data
price1_copy['date'] = pd.to_datetime(price1_copy['date'])

# Date range: 02 Dec 2021 to 15 Nov 2023
start_date = datetime(2021, 12, 2)
end_date = datetime(2023, 11, 15)

#Filtering the data by the given dates
pbe_price = price1_copy.loc[(price1_copy['date']>=start_date) & (price1_copy['date']<=end_date)]


# In[12]:


pbe_price.shape


# Let's look at the sorted pbe_price table:

# In[13]:


pbe_price.sort_values(by='date').head()


# The XBI price data has been filtered for the required dates 02 Dec 2021 until 15 Nov 2023.

# In[14]:


price2_copy = price2.copy()

# Converting the date column to datetime in the futures data
price2_copy['date'] = pd.to_datetime(price2_copy['date'])

# Date range: 02 Dec 2021 to 15 Nov 2023
start_date = datetime(2021, 12, 2)
end_date = datetime(2023, 11, 15)

#Filtering the data by the given dates
xbi_price = price2_copy.loc[(price2_copy['date']>=start_date) & (price2_copy['date']<=end_date)]


# In[15]:


xbi_price.shape


# In[16]:


xbi_price.sort_values(by='date').head()


# The SVOL price data has been filtered for the required dates 02 Dec 2021 until 15 Nov 2023.

# In[17]:


price3_copy = price3.copy()

# Converting the date column to datetime in the futures data
price3_copy['date'] = pd.to_datetime(price3_copy['date'])

# Date range: 02 Dec 2021 to 15 Nov 2023
start_date = datetime(2021, 12, 2)
end_date = datetime(2023, 11, 15)

#Filtering the data by the given dates
svol_price = price3_copy.loc[(price3_copy['date']>=start_date) & (price3_copy['date']<=end_date)]


# In[18]:


svol_price.shape


# In[19]:


svol_price.sort_values(by='date').head()


# # 3. Initial Data Analysis:

# Let's look at the statistics and null values of the three price data we have PBE, XBI and SVOL.

# In[20]:


pbe_price.describe()


# In[21]:


xbi_price.describe()


# In[22]:


svol_price.describe()


# In[23]:


# Counting null values in pbe_price 
pbe_null_counts = pbe_price.isna().sum()
print("Null values in pbe_price:")
print(pbe_null_counts)

# Counting null values in xbi_price 
xbi_null_counts = xbi_price.isna().sum()
print("\nNull values in xbi_price:")
print(xbi_null_counts)


# # 4. Spread-reversion Trading Strategy:

# Spread is defined as the difference between the M-day return between PBE and XBI. There are different functions to succesfully develeoped a spread-reversion trading strategy. Four functions have been used for the strategy which are check_stop_loss function, buy_spread function, short_spread function and run_trading strategy function. Details of these functions are described in the following few cells of this notebook.

# A dataframe has been for the spread with the date, adj_close prices for PBE and XBI and Nt that is to be used to position the trades.

# In[24]:


#Spread

spread_df = pd.merge(pbe_price[['date', 'adj_close', 'Nt']], xbi_price[['date','adj_close']], on='date')


# In[25]:


spread_df.head()


# In[26]:


spread_df.shape


# In[27]:


spread_null_counts = spread_df.isna().sum()
print("Null values in spread_df:")
print(spread_null_counts)


# # Stop Loss Function:

# Let's describe the stop loss function. The function takes in the following parameters: 
# 1. entry_value_buy: the price at which one of the two futures has been bought which depends on buying or shorting of the spread
# 2. entry_value_short: the price at which one of the two futures has been shorted which depends buying or shorting of the spread
# 3. current_value_buy: the current price at which one of the two futures has been bought based on buying or shorting of the spread
# 4. current_value_short: the current price at which of one of the two futures has been shorted based on buying or shorting of the spread
# 5. s: stop-loss parameter: used to compare the loss with a portion of the gross traded cash at position entry
# 6. G (not a parameter to the function, calculated in the function): Gross traded cash at entry: sum of entry_value_buy and entry_value_short
# 
# Now, in the function first G is calculated and then current_pnl (current profit or loss) is calculated. If current_pnl is less than 0 then we check how much we lost based on the proportion of G using s*G. The function returns a boolean parameter depending on the check.

# In[28]:


#Stop loss function
def check_stop_loss(entry_value_buy, entry_value_short, current_value_buy, current_value_short, s):
    
    G = entry_value_buy + entry_value_short
    # Calculate the current loss
    current_pnl = (current_value_short - entry_value_buy) + (entry_value_short - current_value_buy)

    # Check if the loss exceeds the threshold
    if current_pnl < 0:
        if abs(current_pnl) > s * G:
            return True
        else:
            return False


# # Buy and Short Spread Functions:

# Buy Spread Function: Parameters for the function are trade_value, adj_close_pbe and adj_close_xbi. Trade_value is calculated from the running 15-day median of daily dollar volume (Nt). Trade_value is Nt/100. Trade_value is used to trade equal-sized dollar amounts of PBE or XBI i.e. $Nt/100. It takes in the adj_close price for PBE and XBI for the dates when the trade is happening. First we calculate the no. of shares from trade_value and adj_close prices. Then value_buy is the amount of futures bought and value_short is the amount of futures shorted have been returned by the function. In buy spread function X:PBE is bought and Y:XBI is shorted.

# In[29]:


#Buy Spread Function
def buy_spread(trade_value, adj_close_pbe, adj_close_xbi):
    pbe_shares = np.round(trade_value / adj_close_pbe)
    xbi_shares = np.round(trade_value / adj_close_xbi)
    value_buy = pbe_shares * adj_close_pbe
    value_short = xbi_shares * adj_close_xbi
    return value_buy, value_short


# Short Spread Function: Parameters for the function are trade_value, adj_close_pbe and adj_close_xbi. Trade_value is calculated from the running 15-day median of daily dollar volume (Nt). Trade_value is Nt/100. Trade_value is used to trade equal-sized dollar amounts of PBE or XBI i.e. $Nt/100. It takes in the adj_close price for PBE and XBI for the dates when the trade is happening. First we calculate the no. of shares from trade_value and adj_close prices. Then value_buy is the amount of futures bought and value_short is the amount of futures shorted have been returned by the function.  In short spread function X:PBE is sorted and Y:XBI is bought.

# In[30]:


#Short Spread Function
def short_spread(trade_value, adj_close_pbe, adj_close_xbi):
    pbe_shares = np.round(trade_value / adj_close_pbe)
    xbi_shares = np.round(trade_value / adj_close_xbi)
    value_short = pbe_shares * adj_close_pbe
    value_buy = xbi_shares * adj_close_xbi
    return value_buy, value_short


# In[31]:


spread_df = spread_df.reset_index()


# In[32]:


spread_df


# # Trading strategy function ETF Pair PBE/XBI:

# Let's talk about the run_trading_strategy function. The function has the following parameters:
# 1. spread_df: initial spread function that have the dates, adj_close prices for PBE and XBI and Nt
# 2. M: the periods to calculate the M-day return of the future prices. used to calculate the spread_Mday_return which is the difference between the M-day return of PBE and XBI.
# 3. s: stop-loss parameter
# 4. g: the parameter is used to decide when to place a trade or enter the market
# 5. j: the parameter that is used to exit the market (j<g)
# 6. zeta: trading cost parameter: used to calculate the price for entry and exit trades: used to calculate the net pnl (by default set to 0, but can be changed as per requirement)

# What the function does?
# 
# Let's start with the spread_df dataframe. Add the pbe_returns and xbi_returns to the dataframe that is the percentage change in the price for M periods. These returns are used to create the spread for M-day return. The rolling mean and standard deviation are calculated for the M-day period and these are to create market entry and exit signals.
# 
# The spread_df dataframe with all the required columns is then filtered for 01 Jan 2022 and 15 Nov 2023 and sorted for further analysis.
# 
# The maximum available capital is calculated by twice the maximum of the running 15-day median of the daily dollar volume. All the trading parameters and variables are initialized. Variables we have are positions to tell us whether it is long or short, pnl is a list to add our profit or loss when we close our positions, trade_value is the current Nt/100, trade_value_at_entry tell the trade value at entry of a position which is to be used to exit the position by same amount, long_positions_count and short_positions_count are to be used to track the number of long and short positions. Current_spread is the spread at the current date which is the spread_Mday_return from the spread_df database.
# 
# The strategy runs a loop by the date and rows of the spread_df dataframe. At the condition to enter the market it check for the position if its none then it enters condition to place a trade. 
# 
# Entry Conditions:
# 
# First check is whether position is none, if yes then the following condition is activated:
# 
# If the current spread is greater than mean_spread + g*std_spread we short the spread, the short_spread function is called to get the entry prices and position is set to short. If the current spread is less than mean_spread - g*std_spread we buy the spread and buy_spread is called to get the entry prices at buy_spread and positions is updated to long.
# 
# Exit Conditions:
# 
# If position is not none:
# 
# We calculate the exit prices at the trade_value_entry based on the position and also the current prices at trade_value based on the position. Put the exit_flag as false, then we check the following conditions:
# 
# If current spread is greater than mean_spread - j * std_spread and position is long, exit_flag is turned to True; also if current spread is less than mean_spread + j * std_spread and position is hort exit_flag is turned to True and 
# 
# When exit flag is true:
# 
# pnl_value is (exit_value_short - entry_value_buy) + (entry_value_short - exit_value_buy) and append the pnl list.
# 
# Check for stop loss:
# 
# if check_stop_loss(entry_value_buy, entry_value_short, current_value_buy, current_value_short, s) is true close the position and record the pnl.
# 
# The function convert the pnl list to a dataframe and return total profit.
# 

# In[33]:


def run_trading_strategy(spread_df, M, s, j, g, zeta):
    
    # Calculate M-day returns for both ETFs
    spread_df['pbe_returns'] = spread_df['adj_close_x'].pct_change(periods=M).fillna(0)
    spread_df['xbi_returns'] = spread_df['adj_close_y'].pct_change(periods=M).fillna(0)

    # Calculate the spread
    spread_column_name = f'spread_{M}day_return'
    spread_df[spread_column_name] = spread_df['pbe_returns'] - spread_df['xbi_returns']
    spread_df['spread_mean'] = spread_df[spread_column_name].rolling(window=M, min_periods = 0).mean()
    spread_df['spread_std'] = spread_df[spread_column_name].rolling(window=M, min_periods = 0).std()

    # Define trading period
    start_date = pd.to_datetime("2022-01-01")
    end_date = pd.to_datetime("2023-11-15")
    spread_df = spread_df[(spread_df['date'] >= start_date) & (spread_df['date'] <= end_date)]
    
    spread_df = spread_df.sort_values(by = 'date')
    last_date = spread_df.index[-1] if spread_df.index.name == 'date' else spread_df['date'].iloc[-1]
    
    #Maximum capital available
    max_Nt = spread_df['Nt'].max()
    capital_K = 2 * max_Nt  # Setting the capital K

    # Initialize trading parameters and variables
    position = None
    pnl = []
    total_profit = 0
    entry_value_short, entry_value_buy, trade_value_at_entry = 0,0,0
    
     # Initialize counters for long and short positions
    long_positions_count = 0
    short_positions_count = 0
    
    for date, row in spread_df.iterrows():
        current_spread = row[spread_column_name]
        mean_spread = row['spread_mean']
        std_spread = row['spread_std']
        trade_value = row['Nt'] / 100
        adj_close_pbe = row['adj_close_x']
        adj_close_xbi = row['adj_close_y']

        # Check for entry conditions
        if position is None:
            trade_value_at_entry = trade_value
            if current_spread < mean_spread - g * std_spread:
                # Buy Spread
                position = 'long'
                long_positions_count += 1
                entry_value_buy, entry_value_short = buy_spread(trade_value_at_entry, adj_close_pbe, adj_close_xbi)
                
            elif current_spread > mean_spread + g * std_spread:
                # Short Spread
                position = 'short'
                short_positions_count += 1
                entry_value_buy, entry_value_short = short_spread(trade_value_at_entry, adj_close_pbe, adj_close_xbi)

        # Check for exit conditions
        elif position:
            exit_value_buy, exit_value_short = buy_spread(trade_value_at_entry, adj_close_pbe, adj_close_xbi) if position == 'short' else short_spread(trade_value_at_entry, adj_close_pbe, adj_close_xbi)
            current_value_buy, current_value_short = buy_spread(trade_value, adj_close_pbe, adj_close_xbi) if position == 'short' else short_spread(trade_value, adj_close_pbe, adj_close_xbi)
            exit_flag = False
            if position == 'long' and current_spread > mean_spread - j * std_spread:
                exit_flag = True
            elif position == 'short' and current_spread <  mean_spread + j * std_spread:
                exit_flag = True

            if exit_flag is True:
                pnl_value = (exit_value_short - entry_value_buy) + (entry_value_short - exit_value_buy)
                pnl.append({'date': date, 'pnl': pnl_value})
                 # Decrement position counters
                if position == 'long':
                    long_positions_count -= 1
                elif position == 'short':
                    short_positions_count -= 1

                position = None
                
        #Check for stop loss
            if check_stop_loss(entry_value_buy, entry_value_short, current_value_buy, current_value_short, s):
                pnl_value = (current_value_short-entry_value_buy) + (entry_value_short - current_value_buy)
                pnl.append({'date': date, 'pnl': pnl_value})
                    
                # Decrement position counters
                if position == 'long':
                    long_positions_count -= 1
                elif position == 'short':
                    short_positions_count -= 1
                        
                position = None
                
        # Check if it's the last date in the dataset
        if row['date'] == last_date:
            if short_positions_count is not None:
                # Force close the position on the last day
                final_value_buy, final_value_short = buy_spread(trade_value_at_entry, adj_close_pbe, adj_close_xbi) 
                final_pnl_value = (final_value_short - entry_value_buy) + (entry_value_short - final_value_buy)
                pnl.append({'date': date, 'pnl': final_pnl_value})
                short_positions_count = 0
                            
            elif long_positions_count is not None:
                final_value_buy, final_value_short = short_spread(trade_value_at_entry, adj_close_pbe, adj_close_xbi) 
                final_pnl_value = (final_value_short - entry_value_buy) + (entry_value_short - final_value_buy)
                pnl.append({'date': date, 'pnl': final_pnl_value})
                long_positions_count = 0
                    
    if long_positions_count != 0 or short_positions_count != 0:
        print(f"Error: Not all positions are closed by the end of the trading period.{long_positions_count} {short_positions_count} ")

    # Calculate total profit
    pnl_df = pd.DataFrame(pnl)
    total_profit = pnl_df['pnl'].sum() if not pnl_df.empty else 0
    return total_profit


# In[34]:


#Testing trading strategy function:
try:
    profit = run_trading_strategy(spread_df, 4, 0.1, 0.005, 0.03, 0)
    print(f"Profit: {profit}")
except TypeError as e:
    print(f"An error occurred: {e}")


# # Random search to find the parameters to start with grid search optimization:

# Random search starts with a range of values for all parameters and set the maximum number of iterations. A for loop runs with random choices of the parameters and the profit is calculated for all parameters. The profit and the parameters are appended to a list. Max of all the parameters and profit are returned which are to be used for the Grid Search Optimization.

# In[35]:


#Random search to find the parameters to start with in grid search optimization

# Define ranges for each parameter
M_range = range(1, 15)  
s_range = np.arange(0.01, 0.15, 0.01)  
j_range = np.arange(0.005, 0.05, 0.005) 
g_range = np.arange(0.01, 0.1, 0.01)   

# Number of iterations for Random Search
n_iter = 200

# Random Search
random_results = []
for _ in range(n_iter):
    M = random.choice(M_range)
    s = random.choice(s_range)
    j = random.choice(j_range)
    g = random.choice(g_range)
    
    # Run your trading strategy function here
    profit = run_trading_strategy(spread_df, M, s, j, g, 0)
    
    # Store results
    random_results.append((profit, M, s, j, g))

# Find the best parameters from Random Search
best_profit, best_M, best_s, best_j, best_g = max(random_results, key=lambda item: item[0])
print(best_profit, best_M, best_s, best_j, best_g)


# # Grid Search Optimization:

# The parameters from random search are used to create a range of parameters for grid search optimization. The parameters from random search are modified around the boundaries for grid search algorithm. Then we run the loops for all values of M, s, j and g and get the profit for all the parameters and get the maximum of all values and those will be the best parameters for running the trading strategy. The algorithm is maximizing the profit.

# In[37]:


# Define narrower ranges based on Random Search results
M_range_grid = range(best_M-1, best_M+2)
s_range_grid = np.arange(best_s-0.03, best_s+0.02, 0.01)
j_range_grid = np.arange(best_j-0.02, best_j+0.02, 0.005)
g_range_grid = np.arange(best_g-0.01, best_g+0.02, 0.005)

# Grid Search
grid_results = []
for M in M_range_grid:
    for s in s_range_grid:
        for j in j_range_grid:
            for g in g_range_grid:
                
                profit = run_trading_strategy(spread_df, M, s, j, g, 0)
                
                # Store results with rounded parameters
                grid_results.append((profit, M, s, j, g,))

# Find the best parameters from Grid Search
best_profit_grid, best_M_grid, best_s_grid, best_j_grid, best_g_grid = max(grid_results, key=lambda item: item[0])

# Printing best parameters rounded to two decimal places
print(f"Best Profit: {best_profit_grid}, Best M: {best_M_grid}, Best s: {best_s_grid}, Best j: {best_j_grid}, Best g: {best_g_grid}")


# # Best parameter:: Profit= 308.4; M= 10; s=0.04; j=0.045; g=0.075

# # 6. Various plots and correlation heatmaps:

# After getting the best parameters, we have to use a function similar to run_trading_strategy to get the pnl dataframe, trade_values dataframe to visualise the trading signals on the spread data. Moreover, we have to check the correlation of the different parameters, profitability and trading rate with Fama French Factors and SVOL data.

# In[38]:


#Best parameters from grid search optimization
M_best = best_M_grid
s_best = round(best_s_grid, 2)
j_best = round(best_j_grid, 3)
g_best = round(best_g_grid, 3)
print(f"Best M: {M_best}, Best s: {s_best}, Best j: {j_best}, Best g: {g_best}")


# In[39]:


def get_various_dataframes(spread_df, M, s, j, g, zeta):
    
    # Calculate M-day returns for both ETFs
    spread_df['pbe_returns'] = spread_df['adj_close_x'].pct_change(periods=M).fillna(0)
    spread_df['xbi_returns'] = spread_df['adj_close_y'].pct_change(periods=M).fillna(0)

    # Calculate the spread
    spread_column_name = f'spread_{M}day_return'
    spread_df[spread_column_name] = spread_df['pbe_returns'] - spread_df['xbi_returns']
    spread_df['spread_mean'] = spread_df[spread_column_name].rolling(window=M, min_periods = 0).mean()
    spread_df['spread_std'] = spread_df[spread_column_name].rolling(window=M, min_periods = 0).std()

    # Define trading period
    start_date = pd.to_datetime("2022-01-01")
    end_date = pd.to_datetime("2023-11-15")
    spread_df = spread_df[(spread_df['date'] >= start_date) & (spread_df['date'] <= end_date)]
    
    spread_df = spread_df.sort_values(by = 'date')
    last_date = spread_df.index[-1] if spread_df.index.name == 'date' else spread_df['date'].iloc[-1]
    
    #Maximum capital available
    max_Nt = spread_df['Nt'].max()
    capital_K = 2 * max_Nt  # Setting the capital K

    # Initialize trading parameters and variables
    position = None
    pnl = []
    total_profit = 0
    entry_value_short, entry_value_buy, trade_value_at_entry = 0,0,0
    trades = []
    
     # Initialize counters for long and short positions
    long_positions_count = 0
    short_positions_count = 0
    
    for date, row in spread_df.iterrows():
        current_spread = row[spread_column_name]
        mean_spread = row['spread_mean']
        std_spread = row['spread_std']
        trade_value = row['Nt'] / 100
        adj_close_pbe = row['adj_close_x']
        adj_close_xbi = row['adj_close_y']

        # Check for entry conditions
        if position is None:
            trade_value_at_entry = trade_value
            if current_spread < mean_spread - g * std_spread:
                # Buy Spread
                position = 'long'
                long_positions_count += 1
                trades.append({'date':row['date'], 'position': 'long', 'spread':current_spread})
                entry_value_buy, entry_value_short = buy_spread(trade_value_at_entry, adj_close_pbe, adj_close_xbi)
                
            elif current_spread > mean_spread + g * std_spread:
                # Short Spread
                position = 'short'
                short_positions_count += 1
                trades.append({'date':row['date'], 'position': 'short', 'spread':current_spread})
                entry_value_buy, entry_value_short = short_spread(trade_value_at_entry, adj_close_pbe, adj_close_xbi)

        # Check for exit conditions
        elif position:
            exit_value_buy, exit_value_short = buy_spread(trade_value_at_entry, adj_close_pbe, adj_close_xbi) if position == 'short' else short_spread(trade_value_at_entry, adj_close_pbe, adj_close_xbi)
            current_value_buy, current_value_short = buy_spread(trade_value, adj_close_pbe, adj_close_xbi) if position == 'short' else short_spread(trade_value, adj_close_pbe, adj_close_xbi)
            exit_flag = False
            if position == 'long' and current_spread > mean_spread - j * std_spread:
                exit_flag = True
            elif position == 'short' and current_spread <  mean_spread + j * std_spread:
                exit_flag = True

            if exit_flag is True:
                pnl_value = (exit_value_short - entry_value_buy) + (entry_value_short - exit_value_buy)
                pnl.append({'date': row['date'], 'pnl': pnl_value})
                 # Decrement position counters
                if position == 'long':
                    long_positions_count -= 1
                    trades.append({'date':row['date'], 'position': 'short', 'spread':current_spread})
                elif position == 'short':
                    short_positions_count -= 1
                    trades.append({'date':row['date'], 'position': 'long', 'spread':current_spread})

                position = None
                
        #Check for stop loss
            if check_stop_loss(entry_value_buy, entry_value_short, current_value_buy, current_value_short, s):
                pnl_value = (current_value_short-entry_value_buy) + (entry_value_short - current_value_buy)
                pnl.append({'date': row['date'], 'pnl': pnl_value})
                    
                # Decrement position counters
                if position == 'long':
                    long_positions_count -= 1

                elif position == 'short':
                    short_positions_count -= 1
                    
                position = None
                
        # Check if it's the last date in the dataset
        if row['date'] == last_date:
            if short_positions_count is not None:
                # Force close the position on the last day
                final_value_buy, final_value_short = buy_spread(trade_value_at_entry, adj_close_pbe, adj_close_xbi) 
                final_pnl_value = (final_value_short - entry_value_buy) + (entry_value_short - final_value_buy)
                pnl.append({'date': row['date'], 'pnl': final_pnl_value})
                short_positions_count = 0
                            
            elif long_positions_count is not None:
                final_value_buy, final_value_short = short_spread(trade_value_at_entry, adj_close_pbe, adj_close_xbi) 
                final_pnl_value = (final_value_short - entry_value_buy) + (entry_value_short - final_value_buy)
                pnl.append({'date': row['date'], 'pnl': final_pnl_value})
                long_positions_count = 0
                    
   
    # Calculate total profit
    pnl_df = pd.DataFrame(pnl)
    trades_df = pd.DataFrame(trades)
    total_profit = pnl_df['pnl'].sum() if not pnl_df.empty else 0
    return total_profit, pnl_df, trades_df, spread_df, long_positions_count, short_positions_count


# In[40]:


total_profit, pnl_df, trades_df, spread_df,l, s = get_various_dataframes(spread_df, M_best, s_best, j_best, g_best, 0)


# In[41]:


total_profit


# In[42]:


trades_df


# In[43]:


pnl_df


# # Plot 1: How PnL varies during the trading period?

# In[44]:


# Create the plot
plot = (
    ggplot(pnl_df, aes(x='date', y='pnl')) +
    geom_line(color='green', size=0.5) +
    labs(title='PnL Over Time', x='Date', y='PnL') +
    scale_x_date(breaks=date_breaks('2 month'), labels=date_format('%Y-%m')) +
    theme_minimal() +
    theme(axis_text_x=element_text(rotation=45, hjust=1))
)
print(plot)


# The plot shows the variation of profit and loss over the whole trading period.

# # Plot 2: Spread M-day Return Analysis

# In[45]:


spread_df_copy = spread_df


# In[46]:


# Ensure 'date' is a column and is in datetime format in spread_df_copy
spread_df_copy['date'] = pd.to_datetime(spread_df_copy['date'])

# Create the plot
plot = (
    ggplot(spread_df_copy, aes(x='date', y='spread_4day_return')) +
    geom_line(color='blue', size=0.5) +
    labs(title='Spread 4-Day Return Analysis', x='Date', y='Spread 4-Day Return') +
    scale_x_date(breaks=date_breaks('2 month'), labels=date_format('%Y-%m')) +
    theme_minimal() +
    theme(axis_text_x=element_text(rotation=45, hjust=1))
)

# Display the plot
print(plot)


# # Fetching the Fama French Factors:

# The fetch_fama_french_factors function has been used to get the Fama French Factors. The function return the daily Fama-French 3 Factors for developed markets.

# In[47]:


def fetch_fama_french_factors(start_date, end_date):
   

   # URL for Fama-French 3 Factors for developed markets (Daily)
    ff_url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/Developed_3_Factors_Daily_CSV.zip"

    # Fetch the data
    response = requests.get(ff_url)
    if response.status_code != 200:
        raise Exception("Failed to fetch Fama-French factors for developed markets")

    zip_file = ZipFile(BytesIO(response.content))
    csv_name = zip_file.namelist()[0] 
    csv_content = zip_file.open(csv_name).read().decode('utf-8')

    ff_factors = pd.read_csv(StringIO(csv_content), skiprows=3, index_col=0)

    return ff_factors


# In[48]:


start_date = '2021-12-02'
end_date = '2023-11-15'
fama_french_factors = fetch_fama_french_factors(start_date, end_date)


# In[49]:


fama_french_factors.reset_index(inplace=True)
fama_french_factors.rename(columns={'index': 'date', 'Mkt-RF': 'Mkt_RF'}, inplace=True)

fama_french_factors['date'] = pd.to_datetime(fama_french_factors['date'], format='%Y%m%d')


# In[50]:


# Date range: 02 Dec 2021 to 15 Nov 2023
start_date = datetime(2021, 12, 2)
end_date = datetime(2023, 11, 15)

#Filtering the data by the given dates
fama_french_factors = fama_french_factors.loc[(fama_french_factors['date']>=start_date) & (fama_french_factors['date']<=end_date)]


# In[51]:


fama_french_factors = fama_french_factors.sort_values(by = 'date')


# In[52]:


fama_french_factors.head()


# In[53]:


pnl_df['date'] = pd.to_datetime(pnl_df['date'])
fama_french_factors['date'] = pd.to_datetime(fama_french_factors['date'])


# # Plot 3: Correlation heatmap with PnL and Fama French Factors:

# In[54]:


merged_df = pd.merge(pnl_df, fama_french_factors, on='date', how='inner')

# Calculate correlation matrix
correlation_matrix = merged_df[['pnl', 'Mkt_RF', 'SMB', 'HML', 'RF']].corr()

# Create a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap between PnL and Fama-French Factors")
plt.show()


# There is not any substantial correlation between the profit or loss with the Fama French Factors.

# In[61]:


svol_price['date'] = pd.to_datetime(svol_price['date'])
svol_price.sort_values(by='date', inplace=True)

# Calculate daily returns for SVOL
svol_price['SVOL_returns'] = svol_price['adj_close'].pct_change()

# Drop the first row as it will have NaN return
svol_price.dropna(subset=['SVOL_returns'], inplace=True)


# In[62]:


pnl_df['date'] = pd.to_datetime(pnl_df['date'])

# Merge the two DataFrames on 'date'
merged_df = pd.merge(pnl_df, svol_price[['date', 'SVOL_returns']], on='date', how='inner')


# # Plot 4: Correlation heatmap between PnL and SVOL:

# In[63]:


# Calculate correlation between PnL and SVOL returns
correlation_results = merged_df[['pnl', 'SVOL_returns']].corr()

print(correlation_results)
# Create a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_results, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap between PnL and SVOL Returns")
plt.show()


# The correlation heatmap shows less correlation between the profit and loss and the market volatility for the same period. That means our pnl does not vary much with the market volatility.

# # 7. Conclusion:

# The analysis of the spread trading strategy reveals significant insights into its performance and correlation with market factors. By adjusting the parameters such as the stop-loss threshold, entry and exit points, and the duration of the returns, the strategy's robustness and adaptability to market conditions were thoroughly evaluated. 
# 
# The strategy's correlation with Fama-French factor returns and SVOL provides a deeper understanding of its market sensitivities. This tells the strategy's exposure to different market risks and its alignment with broader market movements. The results underscore the importance of parameter optimization and the need for continuous monitoring and adjustment based on market dynamics.
# 
# In summary, this study offers valuable insights into the application and effectiveness of a spread trading strategy. It highlights the potential of such strategies in exploiting market inefficiencies while also emphasizing the need for careful parameter selection and risk management. As with any trading strategy, the importance of aligning it with market conditions and individual risk tolerance cannot be overstated.
