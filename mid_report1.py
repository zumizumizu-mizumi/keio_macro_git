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
UK_gdp = web.DataReader('NGDPRSAXDCGBQ', 'fred', start_date, end_date)
UK_log_gdp = np.log(UK_gdp)

# apply a Hodrick-Prescott filter to the data to extract the cyclical component
UK_cycle, UK_trend = sm.tsa.filters.hpfilter(UK_log_gdp, lamb=1600)

# download the data from FRED using pandas_datareader
JP_gdp = web.DataReader('JPNRGDPEXP', 'fred', start_date, end_date)
JP_log_gdp = np.log(JP_gdp)

# apply a Hodrick-Prescott filter to the data to extract the cyclical component
JP_cycle, JP_trend = sm.tsa.filters.hpfilter(JP_log_gdp, lamb=1600)

# ① UKとJPの循環変動成分の標準偏差を計算
uk_std = UK_cycle.std()
jp_std = JP_cycle.std()
print(f"UK GDP cyclical component standard deviation: {uk_std:.4f}")
print(f"JP GDP cyclical component standard deviation: {jp_std:.4f}")

# ② UKとJPの循環変動成分の相関係数を計算（共通期間で）
combined = pd.concat([UK_cycle, JP_cycle], axis=1).dropna()
combined.columns = ['UK_cycle', 'JP_cycle']
correlation = combined.corr().iloc[0, 1]
print(f"Correlation between UK and JP cyclical components: {correlation:.4f}")

# ③ UKとJPの循環変動成分の時系列グラフをプロット
plt.figure(figsize=(12, 6))
plt.plot(combined.index, combined['UK_cycle'], label='UK Cycle', linestyle='-', color='blue')
plt.plot(combined.index, combined['JP_cycle'], label='JP Cycle', linestyle='--', color='red')
plt.title('Cyclical Components of UK and JP Real GDP (HP Filter, λ=1600)')
plt.xlabel('Year')
plt.ylabel('Cyclical Component (Log GDP)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
