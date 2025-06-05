import pandas as pd
import pandas_datareader.data as web
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas_datareader as pdr
import numpy as np

# set the start and end dates for the data
start_date = '1994-01-01'
end_date = '2024-10-01'

# download the data from FRED using pandas_datareader
gdp = web.DataReader('JPNRGDPEXP', 'fred', start_date, end_date)
log_gdp = np.log(gdp)

# apply a Hodrick-Prescott filter to the data to extract the cyclical component
cycle, trend = sm.tsa.filters.hpfilter(log_gdp, lamb=1600)

# グラフ1：元データとトレンド成分
plt.figure(figsize=(12, 6))
plt.plot(log_gdp, label='Log GDP', color='black', linewidth=2)
plt.plot(trend, label=f'Trend', linestyle='--')
plt.title('Log Real GDP and Trend Components (HP Filter)')
plt.xlabel('Year')
plt.ylabel('Log GDP')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# グラフ2：循環成分の比較
plt.figure(figsize=(12, 6))
plt.plot(cycle, label=f'Cycle')
plt.title('Cyclical Components of Log Real GDP (HP Filter)')
plt.xlabel('Year')
plt.ylabel('Cyclical Component')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
