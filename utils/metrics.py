import sys

import numpy as np

def RSE(pred, true):
    return np.sqrt(np.sum((true-pred)**2)) / np.sqrt(np.sum((true-true.mean())**2))

def CORR(pred, true):
    u = ((true-true.mean(0))*(pred-pred.mean(0))).sum(0) 
    d = np.sqrt(((true-true.mean(0))**2*(pred-pred.mean(0))**2).sum(0))
    return (u/d).mean(-1)

# 平均绝对误差（Mean Absolute Error）,观测值与真实值的误差绝对值的平均值。
def MAE(pred, true):
    # print(pred)
    # print(true)
    # print(pred.shape,true.shape)
    # sys.exit()
    return np.mean(np.abs(pred-true))

# 均方误差(Mean Square Error)
def MSE(pred, true):
    return np.mean((pred-true)**2)

# 均方根误差(Root Mean Square Error),其实就是MSE加了个根号，这样数量级上比较直观，比如RMSE=10，可以认为回归效果相比真实值平均相差10
def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))

# 平均绝对百分比误差（Mean Absolute Percentage Error）；范围[0,+∞)，MAPE 为0%表示完美模型，MAPE 大于 100 %则表示劣质模型。
def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))

 # 均方百分比误差 MSPE
def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))


# 没有量纲影响的,对称平均绝对百分比误差（Symmetric Mean Absolute Percentage Error）
# smape越小越好
def SMAPE(pred,true):
    smape = np.mean(np.abs((pred - true)) / ((true+pred) * 1/2))
    return smape

# 决定系数：R2受到样本数量的影响
def R2_score(pred,true):
    # r2 = metrics.r2_score(pred,true)
    r2 = 1 - np.sum((true - pred) ** 2) / np.sum((true - np.mean(true)) ** 2)
    return r2

# 校正决定系数：抵消样本数量对 R-Square的影响，做到了真正的 0~1，越大越好。；消除了样本数目的影响；
def Adjusted_R_Square(pred,true,r2):
    ad_r2 = 1-(1-r2)* (len(true)-1-1)/(len(true)-1)
    return ad_r2

def metric(pred, true):
    # 平均绝对误差（Mean Absolute Error）
    mae = MAE(pred, true)
    # 均方误差(Mean Square Error)
    mse = MSE(pred, true)
    # 均方根误差(Root Mean Square Error)
    rmse = RMSE(pred, true)
    # 平均绝对百分比误差（Mean Absolute Percentage Error）
    mape = MAPE(pred, true)
    #
    mspe = MSPE(pred, true)

    r2  = R2_score(pred,true)
    # 没有样本数目的影响：
    # 校正决定系数
    ad_r2 = Adjusted_R_Square(pred,true,r2)
    # 对称平均绝对百分比误差（Symmetric Mean Absolute Percentage Error）
    smape = SMAPE(pred,true)

    return mae,rmse,smape,r2,ad_r2

def var_true(df,index):
    # 有true，那么从第3列开始取实验列
    df_ex = df.iloc[:, index:]
    # print(df_ex)
    df['var_true'] = 0.00
    df['std_true'] = 0.00
    # 遍历行
    for i in range(len(df_ex)):
        # 便遍历列
        sum_diff = []
        for c in df_ex.columns.tolist():
            sum_diff.append((float(df['true'][i])-float(df[c][i]))**2)
        df['var_true'][i] = sum(sum_diff)/float(len(sum_diff))
        df['std_true'][i] = (sum(sum_diff)/float(len(sum_diff)))**0.5
        # print(len(sum_diff))
        # print(sum_diff)
        # sys.exit()
    return df
# 暂时没用到
def var_true_multivariate(df,tmp,index,args,j):
    # 有true
    df["std_true_{}".format(args.columns[j + 1])] = 0.00
    df["var_true_{}".format(args.columns[j + 1])] = 0.00
    # 遍历行
    for k in range(len(tmp[j])):
        # 便遍历列
        sum_diff = []
        for c in tmp[j].iloc[:, index:].columns.tolist():
            sum_diff.append((float(tmp[j].iloc[:, 0][k]) - float(tmp[j][c][k])) ** 2)
        df["var_true_{}".format(args.columns[j + 1])][k] = sum(sum_diff) / float(len(sum_diff))
        df["std_true_{}".format(args.columns[j + 1])][k] = (sum(sum_diff) / float(len(sum_diff))) ** 0.5
    # 保留小数点
    df["std_true_{}".format(args.columns[j + 1])] = round(df["std_true_{}".format(args.columns[j + 1])], 1)
    df["var_true_{}".format(args.columns[j + 1])] = round(df["var_true_{}".format(args.columns[j + 1])], 1)
    return df

# 计算标准差标准误
def calculate_var(df,args):
    if args.features != 'M':
        index = 0
        if 'true' in df.columns.tolist():
            index = 3
        else:
            index = 2
        # print(df.iloc[:,index:])
        std_self = df.iloc[:,index:].std(axis=1)
        var_self = df.iloc[:,index:].var(axis=1,ddof=1)
        # print(var_self)
        # print(std_self)
        # sys.exit()
        if index == 3:
            df = var_true(df, index)
            # sys.exit()
        df["std_true"] = round(df["std_true"], 1)
        df["var_true"] = round(df["var_true"], 1)
        df['std_self'] = std_self
        df['var_self'] = var_self
        df["std_self"] = round(df["std_self"], 1)
        df["var_self"] = round(df["var_self"], 1)
    if args.features == 'M':
        tmp = []
        for i in range(args.c_out):
            tmp.append(df[[s for s in df.columns if args.columns[i + 1] in s and "pred" not in s]])
        for j in range(len(tmp)):
            # print(tmp[j].columns)
            # continue
            index = 0
            if 'true' in tmp[j].columns.tolist()[0]:
                index = 1
            else:
                index = 0
            # 计算组内方差
            # print(tmp[j].iloc[:, index:].columns)
            std_self = tmp[j].iloc[:, index:].std(axis=1)
            var_self = tmp[j].iloc[:, index:].var(axis=1, ddof=1)
            df['std_self_{}'.format(args.columns[j + 1])] = std_self
            df['var_self_{}'.format(args.columns[j + 1])] = var_self
            df["std_self_{}".format(args.columns[j + 1])] = round(df["std_self_{}".format(args.columns[j + 1])], 1)
            df["var_self_{}".format(args.columns[j + 1])] = round(df["var_self_{}".format(args.columns[j + 1])], 1)
            # print(tmp[j].columns)
            # sys.exit()
            # print(df.columns)
            # 计算组数据和真实值的方差,index=2
            if index == 1:
                # df = var_true_multivariate(df,tmp[j], index,args,j)
                # 有true
                df["std_true_{}".format(args.columns[j + 1])] = 0.00
                df["var_true_{}".format(args.columns[j + 1])] = 0.00
                # 遍历行
                for k in range(len(tmp[j])):
                    # 便遍历列
                    sum_diff = []
                    for c in tmp[j].iloc[:, index:].columns.tolist():
                        sum_diff.append((float(tmp[j].iloc[:,0][k]) - float(tmp[j][c][k])) ** 2)
                    df["var_true_{}".format(args.columns[j + 1])][k] = sum(sum_diff) / float(len(sum_diff))
                    df["std_true_{}".format(args.columns[j + 1])][k] = (sum(sum_diff) / float(len(sum_diff))) ** 0.5
            # 保留小数点
            df["std_true_{}".format(args.columns[j + 1])] = round(df["std_true_{}".format(args.columns[j + 1])], 1)
            df["var_true_{}".format(args.columns[j + 1])] = round(df["var_true_{}".format(args.columns[j + 1])], 1)
    return df