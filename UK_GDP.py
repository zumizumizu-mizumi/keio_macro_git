import pandas as pd
import pandas_datareader.data as web
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas_datareader as pdr
import numpy as np

# データ期間の設定
start_date = '1994-01-01'
end_date = '2024-10-01'

# UKの実質GDPをFREDからダウンロードし、対数変換
gdp = web.DataReader('NGDPRSAXDCGBQ', 'fred', start_date, end_date)
log_gdp = np.log(gdp)

#λを３通り設定
lambdas = [10, 100, 1600]
trends = {}
cycles = {}

# 各λに対してHPフィルターを適用
for lam in lambdas:
    cycle, trend = sm.tsa.filters.hpfilter(log_gdp, lamb=lam)
    trends[lam] = trend
    cycles[lam] = cycle

# グラフ1：元データとトレンド成分
plt.figure(figsize=(12, 6))
plt.plot(log_gdp, label='Log GDP', color='black', linewidth=2)
for lam in lambdas:
    plt.plot(trends[lam], label=f'Trend (λ={lam})', linestyle='--')
plt.title('Log Real GDP and Trend Components (HP Filter)')
plt.xlabel('Year')
plt.ylabel('Log GDP')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# グラフ2：循環成分の比較
plt.figure(figsize=(12, 6))
for lam in lambdas:
    plt.plot(cycles[lam], label=f'Cycle (λ={lam})')
plt.title('Cyclical Components of Log Real GDP (HP Filter)')
plt.xlabel('Year')
plt.ylabel('Cyclical Component')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()