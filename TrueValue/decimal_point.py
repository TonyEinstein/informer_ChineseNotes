# !/usr/bin python3                                 
# encoding    : utf-8 -*-                            
# @author     :   陈汝海                               
# @software   : PyCharm      
# @file       :   decimal_point.py
# @Time       :   2021/11/18 16:12


import pandas as pd

def decimal_2(path,to_path):
    df = pd.read_csv(path,encoding='utf-8')
    df['price'] = round(df['price'],1)
    df.to_csv(to_path,sep=',',index=False,encoding='utf-8')


# decimal_2('cleaned_data/原始值-月均价.csv','月均价.csv')
# decimal_2('cleaned_data/原始值-日均价.csv','日均价.csv')