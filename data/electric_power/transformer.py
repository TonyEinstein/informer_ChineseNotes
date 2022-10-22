


import pandas as pd

def jianyan(data):
    pass

def xiangguan(data):
    pass

def fi(data):


    return data
    pass

def transformer_data(path,topath):
    data = pd.read_csv('TexasTurbine.csv')
    tmp = pd.date_range(start='20181231', end='20191231', freq='1H')
    tmp = tmp.tolist()
    tmp.pop()
    data['time'] = tmp
    data.drop('Time stamp', inplace=True, axis=1)
    order = ['time', 'Wind speed | (m/s)', 'Wind direction | (deg)','Pressure | (atm)',"Air temperature | ('C)","System power generated | (kW)"]
    data = data[order]
    data.columns = ['date', 'wind_speed', 'wind_direc','pressure','air_temperature','spg']
    data['date'] = pd.to_datetime(data['date'])
    data['wind_direc'] = round(data['wind_direc'],2)
    # data = data[:]

    # 异常值检测、填充
    # ...

    # 相关系数
    # 1

    # 平稳性检验---+
    # 1

    # 白噪声检验---+
    # 1

    data.to_csv(topath,index=False,encoding='utf-8')

transformer_data('TexasTurbine.csv','electric.csv')

