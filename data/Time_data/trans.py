import numpy as np
import pandas as pd
import time
from datetime import datetime, date, timedelta




# 给数据加时间
def data_plus(path,idx):
    df = pd.read_excel(path)
    df[["X1","X2","X3","X4"]] = df[["X1","X2","X3","X4"]].round(2)
    times = pd.date_range('2000-01-01 00:00:00', '2022-01-01 00:00:00', freq='15min')
    print("生成的总长度：", len(times))
    times = times[-idx:]
    df['date'] = times
    order = ['date', 'X1','X2','X3','X4']
    df = df[order]
    df.to_csv("data.csv",sep=',',index=False,header=True)
    print(df.head(3))

# data数据添加时间
# data_plus("data.xlsx",idx=300000)



# 差分数据
def chaifen(path,idx,pred_len):
    df = None
    sign = False
    try:
        df = pd.read_excel(path)
        sign = True
    except:
        df = pd.read_csv(path)
    df[["X1", "X2", "X3", "X4"]] = df[["X1", "X2", "X3", "X4"]].round(2)
    # 训练数据
    train_df = df[:idx]
    #验证数据（预测未来的真实值）
    future_df = df[idx:idx+pred_len]
    future_df.reset_index(inplace=True,drop=True)
    if sign == False:
        train_df.to_csv("训练集_{}.csv".format(path[:-4]))
        future_df.to_csv("未来数据真实值_{}.csv".format(path[:-4]))
    else:
        train_df.to_csv("训练集_{}.csv".format(path[:-5]))
        future_df.to_csv("未来数据真实值_{}.csv".format(path[:-5]))

df = pd.read_csv("samll_test.csv")
print(df.corr())


# df = pd.read_csv("more_samll_test.csv")
# dfs = np.mean(np.array(df[['X1']])-np.array(df[['X2']]))
# print(dfs)

# df = pd.read_excel("most_samll_test_1个变量.xlsx")
# df_train = df[:5000]
# df_vail = df[5000:].reset_index(drop=True)

# df_train.to_csv("most_samll_test_1个变量.csv",sep=",",encoding="utf-8",index=False)
# df_vail.to_excel("most_samll_test_1个变量真实值.xlsx",encoding="utf-8",index=False)
# print(df_train)
# print(df_vail)