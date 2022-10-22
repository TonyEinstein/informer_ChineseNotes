import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
import numpy as np


# 读取数据
df = pd.read_csv("more_samll_test.csv")
df_3 = df[["date","X1","X2","X3"]]
df_4 = df[["date","X1","X2","X3","X4"]]
df_X4 = df[["date","X4"]]

# 1.在excel和代码制图中分别查看数据分布以及规律
print("1.数据的内容以及分布:\n",df_X4.tail(4))
df_3.plot()
df_4.plot()
df_X4.plot()


# 2.查看有无缺失值：因为是生成的，所以无缺失值
print("--------------"*5)
print("2.数据中是否有缺失值：\n",df.isna().any())

# 3.异常值处理，由于在1步骤中可以观察到数据是周期性规律性比较强的，且数据是生成的，所以无需处理异常值。
print("--------------"*5)
print("3.由于在1步骤中可以观察到数据是周期性规律性比较强的，且数据是生成的，所以无需处理异常值。")

# 4.数据检验：平稳性检验、白噪声检验。
print("--------------"*5)
print("4.数据检验：")
data_columns = ["X1","X2","X3","X4"]

print("4.1 平稳性检验：")
"""
autolag参数含义：自动选择滞后数目（AIC：赤池信息准则，默认；BIC：贝叶斯信息准则；
t-stat：基于maxlag，从maxlag开始并删除一个滞后直到最后一个滞后长度基于 t-statistic 显著性小于5%为止；None：使用maxlag指定的滞后）

第一个值：表示Test Statistic ， 即T值，表示T统计量。
第二个值：p-value，即p值，表示T统计量对应的概率值。
第三个值：Lags Used，即表示延迟阶数。
第四个值：Number of Observations Used，即表示测试的次数。
第五个值：
    配合第一个一起看的，是在99%，95%，90%置信区间下的临界的ADF检验的值，若是第一个值比第五个值小证实平稳，反之证实不平稳。根据结果看出来；
    大括号中的值，分别表示1%， 5%， 10% 的三个level，一般来说5%是常用的阈值，也可以根据自己的实际需求进行设定。p-value的值大于5%，将认为数据是不平稳的。
第六个值：未知
"""
adf = [adfuller(np.array(df[x]),autolag='AIC') for x in data_columns]
for i in range(len(adf)):
    print("{}列平稳性检验结果：\t".format(data_columns[i]),"T统计量：%.2f\t" % adf[i][0],"T统计量对应的概率值p-value：{}\t".format(adf[i][1]),
          "延迟阶数：%.0f\t" % adf[i][2],"测试的次数(ADF回归和计算的观测值的个数)：%.0f\t" % adf[i][3],"95%置信区间临界值：{}\t".format(round(adf[i][4]["5%"],2)),
          "是否平稳：{}\t".format(True if adf[i][1]< 0.05 else False))


print("4.2 白噪声检验（纯随机检验）：")
"""
参数解释：
lags为延迟期数，如果为整数，则是包含在内的延迟期数，如果是一个列表或数组，那么所有时滞都包含在列表中最大的时滞中；
boxpierce为True时表示除开返回LB统计量还会返回 Box-Pierce统计量；
auto_lag:指示是否根据最大相关值的阈值自动确定最佳滞后长度的标志。

返回值解释：
lb-value:测试的统计量,即Ljung-Box统计量的值；
p-value:基于卡方分布的p值；
bp-value:Box-Pierce 检验的检验统计量的值;
bpp-value:基于卡方分布下的Box-Pierce检验的p值；
"""
als = [acorr_ljungbox(np.array(df[x]),lags=[1,2,6,20,50],boxpierce=False) for x in data_columns]
for i in range(len(als)):
    print("{}列纯随机检验结果：\t".format(data_columns[i]),"min_Ljung-Box统计量的P值：%.2f\t" % min(als[i]["lb_pvalue"]),"是否纯随机：{}\t".format(True if min(als[i]["lb_pvalue"]) > 0.05 else False))

# 5.数据定阶（确定特征长度），根据数据周期性，计算自相关系数以及偏自相关系数确定阶数。
"""
根据'aic', 'bic', 'hqic'准则
"""
print("--------------"*5)
import statsmodels.tsa.stattools as st
# 直接根据分布图取一个周期的长度即可，因为很明显。

df_st = df[data_columns]
print(df_st.columns)
for x in df_st.columns:
    order = st.arma_order_select_ic(df[x], max_ar=500, max_ma=500, ic=['bic'])
    # order = st.arma_order_select_ic(df[x], max_ar=500, max_ma=500, ic=['aic', 'bic'])
    # order = st.arma_order_select_ic(df[x], max_ar=500, max_ma=500, ic=['aic', 'bic', 'hqic'])
    # print(order.bic_min_order,order.aic_min_order)
    print(order.bic_min_order)

plt.show()







