# !/usr/bin python3                                 
# encoding    : utf-8 -*-                            
# @author     :   陈汝海                               
# @software   : PyCharm      
# @file       :   multi_lag_processor.py
# @Time       :   2021/12/3 8:39
import pandas as pd

def decimal_xls_2(path,to_path):
    df = pd.read_excel(path)
    df['price'] = round(df['price'],1)
    # 数据缩放
    df['estimate_amount'] = df['estimate_amount']/400
    # 小数点
    df['estimate_amount'] = round(df['estimate_amount'],1)
    df= df.loc[:, ['date', 'estimate_amount', 'price']]
    # print(df.tail(3))
    # df.to_csv(to_path,sep=',',index=False,encoding='utf-8')
    # print("------------------未进行滞后处理-价格-供求数据文件保存完成！存储在{} ----------------------".format(to_path))
    return df

def multi_lag(df_mutil,df_single,lag):
    df_single['date'] = pd.to_datetime(df_single['date'])
    lenght_mutil = len(df_mutil)
    tmp = df_mutil[['date','price']]
    tmp = tmp[lag:]
    tmp.reset_index(drop=True,inplace=True)
    try:
        # 获取tmp的最后一个日期：
        index_last_mutil_in_sigle = df_single[(df_single.date == tmp['date'][lenght_mutil - lag - 1])].index.tolist()[0]
        # print(df_single.iloc[index_last_mutil_in_sigle+1:index_last_mutil_in_sigle+lag+1,:])
        # 新增加多行数据
        tmp = tmp.append(df_single.iloc[index_last_mutil_in_sigle + 1:index_last_mutil_in_sigle + lag + 1, :],ignore_index=True)
        assert len(df_mutil) == len(tmp)
        df_mutil[['date', 'price']] = tmp[['date', 'price']]
    except Exception as e:
        print("------------**------------进行滞后性处理出现报错！检查df_single中是否包含了df_mutil的滞后性元素------------**------------")
        print(e)
        return df_mutil
    return df_mutil

def lag_processor_main(original_multi_path,output_multi_path,single_path,lag, result_path):
    df_mutil = decimal_xls_2(original_multi_path, output_multi_path)
    df_single = pd.read_csv(single_path,encoding='utf-8')
    df_mut = multi_lag(df_mutil=df_mutil,df_single=df_single,lag=lag)
    df_mut.to_csv(result_path, sep=',', index=False, encoding='utf-8')
    print("-----------------滞后处理文件-价格-供求数据处理完成！存储在：{} ------------------".format(result_path))
    print("完成进行滞后处理！")