
import torch
import argparse
import datetime
import json
import os
import shutil
import sys
import time
# 加载torch也是需要时间的，算在里面。
import numpy as np
start = time.time()
import pandas as pd
from exp.exp_informer import Exp_Informer
from utils.visualization import *
from utils.initialize_random_seed import *
from utils.metrics import *
from utils.multi_lag_processor import *
from pyecharts.globals import CurrentConfig, OnlineHostType
import warnings

# np.set_printoptions(precision=2)
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)   #显示完整的列
pd.set_option('display.max_rows', None)  #显示完整的行


# 未来的那段时间的真实值
def get_true_data(sheet_name,true_file,arg):
    data = pd.read_excel(true_file, sheet_name=sheet_name)
    if args.features != 'M':
        data["{}".format(arg.target)] = round(data["{}".format(arg.target)], 1)
    if args.features == 'M':
        data_columns = data.columns.values.tolist()
        for i in range(args.c_out):
            data[data_columns[i + 1]] = round(data[data_columns[i + 1]], 1)
    return data

def initialize_parameter():
    parser = argparse.ArgumentParser(description='[Informer] Long Sequences Forecasting')
    parser.add_argument('--root_path', type=str, default='./data/electric_power/', help='数据文件的根路径（root path of the data file）')
    # parser.add_argument('--root_path', type=str, default='./data/ETT/', help='数据文件的根路径（root path of the data file）')
    # parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--target', type=str, default='spg', help='S或MS任务中的目标特征列名（target feature in S or MS task）')
    parser.add_argument('--freq', type=str, default='h', help='时间特征编码的频率（freq for time features encoding）, '
                                                              '选项（options）:[s:secondly, t:minutely, h:hourly, d:daily, b:工作日（business days）, w:weekly, m:monthly], '
                                                              '你也可以使用更详细的频率，比如15分钟或3小时（you can also use more detailed freq like 15min or 3h）')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/',
                        help='模型检查点的位置（location of model checkpoints）')
    # Informer decoder input: concat[start token series(label_len), zero padding series(pred_len)]
    parser.add_argument('--enc_in', type=int, default=1, help='编码器输入大小（encoder input size）')
    parser.add_argument('--dec_in', type=int, default=1, help='解码器输入大小（decoder input size）')
    parser.add_argument('--c_out', type=int, default=1, help='输出尺寸（output size）')
    parser.add_argument('--d_model', type=int, default=32, help='模型维数（dimension of model）默认是512-------------------------模型维数')
    parser.add_argument('--n_heads', type=int, default=8, help='（num of heads）multi-head self-attention的head数')
    parser.add_argument('--e_layers', type=int, default=2, help='编码器层数（num of encoder layers）-------------------编码器层数')
    parser.add_argument('--d_layers', type=int, default=1, help='解码器层数（num of decoder layers）---------------------解码器层数')
    parser.add_argument('--s_layers', type=str, default='3,2,1', help='堆栈编码器层数（num of stack encoder layers）---------------堆栈编码器层数')
    parser.add_argument('--d_ff', type=int, default=128, help='fcn维度（dimension of fcn），默认是2048--------------------FCN维度')
    """
    预测未来短期时间1~3个月的时候，d_model和d_ff进行设置的小，如16、32或者16,16；
    预测未来短期时间4个月及以上的时候，d_model和d_ff进行设置的稍微大一点点，如16、64或者32,64；32,128。
    """
    parser.add_argument('--factor', type=int, default=5, help='probsparse attn factor')
    parser.add_argument('--padding', type=int, default=0, help='padding type')
    parser.add_argument('--distil', action='store_false', help='是否在编码器中不使用知识蒸馏，使用此参数意味着不使用蒸馏'
                                                               '（whether to use distilling in encoder, using this argument means not using distilling）',
                        default=True)
    parser.add_argument('--attn', type=str, default='prob', help='用于编码器的注意力机制，选项：[prob, full]'
                                                                 '（attention used in encoder, options:[prob, full]）')

    # 时间特征编码【未知】
    parser.add_argument('--embed', type=str, default='timeF', help='时间特征编码，选项：[timeF, fixed, learned]'
                                                                   '（time features encoding, options:[timeF, fixed, learned]）')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true',default=True, help='是否在编码器中输出注意力'
                                                                        '（whether to output attention in ecoder）')
    parser.add_argument('--do_predict', action='store_true', default=True, help='是否预测看不见的未来数据'
                                                                                '（whether to predict unseen future data）')
    parser.add_argument('--mix', action='store_true', help='在生成解码器中使用混合注意力'
                                                            '（use mix attention in generative decoder）', default=True)
    parser.add_argument('--cols', type=str, nargs='+', help='将数据文件中的某些cols作为输入特性'
                                                            '（certain cols from the data files as the input features）')
    parser.add_argument('--num_workers', type=int, default=0, help='工作的数据加载器数量 data loader num workers')
    parser.add_argument('--train_epochs', type=int, default=50, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='训练输入数据的批大小 batch size of train input data--------------------批次大小')
    parser.add_argument('--patience', type=int, default=10, help='提前停止的连续轮数 early stopping patience')
    parser.add_argument('--des', type=str, default='forecasting', help='实验描述 exp description')

    parser.add_argument('--loss', type=str, default='mse', help='损失函数选项：loss function【mse、huberloss、L1loss】--------------------损失函数')

    parser.add_argument('--lradj', type=str, default='type1', help='校正的学习率adjust learning rate----------------------学习率更新算法')
    parser.add_argument('--use_amp', action='store_true', help='使用自动混合精度训练 use automatic mixed precision training--------',
                        default=True)
    parser.add_argument('--output', type=str, default='./output', help='输出路径')
    # 想要获得最终预测的话这里应该设置为True；否则将是获得一个标准化的预测。
    parser.add_argument('--inverse', action='store_true', help='逆标准化输出数据 inverse output data', default=True)
    parser.add_argument('--scale', action='store_true', help='是否进行标准化，默认是True', default=True)
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
    parser.add_argument('--itr', type=int, default=2, help='次实验 experiments times----------------------------------多少次实验')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='optimizer learning rate-----------------------------初始学习率')
    parser.add_argument('--save_model_choos', type=bool, default=False, help='是否保存模型，不保存的话不占用IO')
    parser.add_argument('--is_show_label', type=bool, default=False, help='是否显示图例数值')
    # seq_len其实就是n个滑动窗口的大小，pred_len就是一个滑动窗口的大小
    parser.add_argument('--seq_len', type=int, default=96,
                        help='Informer编码器的输入序列长度（input sequence length of Informer encoder）原始默认为96------------------------编码器输入序列长度seq_len')
    parser.add_argument('--label_len', type=int, default=48,
                        help='inform解码器的开始令牌长度（start token length of Informer decoder），原始默认为48-------------------------解码器的开始令牌起始位置label_len')
    parser.add_argument('--pred_len', type=int, default=76, help='预测序列长度（prediction sequence length）原始默认为24------------------预测序列长度pred_len')
    # pred_len就是要预测的序列长度（要预测未来多少个时刻的数据），也就是Decoder中置零的那部分的长度
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='dropout，长序列预测用0.5，短期预测用0.05~0.2(一般是0.05)，如果shuffle_flag的训练部分为True，那么该值直接设置为0;模型参数多设置为0.5，要在0.5范围内；视情况而定。----')

    parser.add_argument('--train_proportion', type=float, default=0.8, help='训练集比例')
    parser.add_argument('--test_proportion', type=float, default=0.1, help='测试集比例')

    parser.add_argument('--seed', type=int, default=12345, help='random seed 随机数种子')
    parser.add_argument('--random_choos', type=bool, default=True, help='random seed 随机数种子，是否随机，为True一般用于多次实验')
    parser.add_argument('--sub_them', type=str, default='5变量多对一', help='单次运行的存储文件夹字后面的内容--------------------存储数据父文件夹名字')
    # parser.add_argument('--sub_them', type=str, default='月度', help='单次运行的存储文件夹的月字后面的内容--------------------存储数据父文件夹名字')
    parser.add_argument('--true_sheetname', type=str, default='Sheet1', help='真实值的月份名称,execl文件的sheetname--------------------------真实值的月份数值')
    # parser.add_argument('--true_price', type=str, default='7月第二第三周', help='真实值的月份名称,execl文件的sheetname--------------------------真实值的月份数值')
    # parser.add_argument('--true_price', type=str, default='1-6月', help='真实值的月份名称,execl文件的sheetname--------------------------真实值的月份数值')
    parser.add_argument('--model', type=str, required=False, default='informer',
                        help='model of experiment, options: [informer, informerstack]')
    parser.add_argument('--data', type=str, required=False, default='electric_power', help='data them，取决了在data parse中寻找的是哪个数据文件的配置,很重要')
    # parser.add_argument('--data', type=str, required=False, default='chicken_MS',help='data them，取决了在data parse中寻找的是哪个数据文件的配置,很重要')

    # parser.add_argument('--true_file', type=str, required=False, default='./TrueValue/2020年真实值.xls', help='真实值数据的文件名')
    # parser.add_argument('--true_file', type=str, required=False, default='./TrueValue/周粒度实验的真实价格.xls', help='真实值数据的文件名')
    # parser.add_argument('--true_file', type=str, required=False, default='./TrueValue/月粒度实验的真实价格.xls', help='真实值数据的文件名')
    # parser.add_argument('--true_file', type=str, required=False, default='./TrueValue/日粒度实验的真实价格.xls', help='真实值数据的文件名')
    # parser.add_argument('--true_file', type=str, required=False, default='./TrueValue/日粒度与月粒度对比实验的真实价格.xls', help='真实值数据的文件名')
    parser.add_argument('--true_file', type=str, required=False, default='./TrueValue/electric_power真实值.xls', help='真实值数据的文件名')

    # parser.add_argument('--data_path', type=str, default='周粒度-多特征数据汇总.csv', help='data file')
    parser.add_argument('--data_path', type=str, default='electric.csv', help='data file')

    parser.add_argument('--columns', type=list, required=False, default=["date",'spg'], help='存储预测数据的时候的列名，多对多M')
    # parser.add_argument('--columns', type=list, required=False, default=["time", 'GZ_maize_prince','CD_maize_price','CD_SBM_price','ZJ_SBM_prince','price'], help='存储预测数据的时候的列名，多对一MS、一对一S任务')
    # parser.add_argument('--shuffle_flag_train', type=str, required=False, default=True, help='训练的时候是否打乱数据[未完成该定义]')
    parser.add_argument('--features', type=str, default='MS', help='预测任务选项（forecasting task, options）:[M, S, MS]; '
                                                                   'M:多变量预测多元（multivariate predict multivariate）, '
                                                                   'S:单变量预测单变量（univariate predict univariate）, '
                                                                   'MS:多变量预测单变量（multivariate predict univariate）')
    #----------------S任务下:下面的配置项不用修改,如果需要再进行修改-------------------
    parser.add_argument('--lag_sign', type=bool,required=False, default=False, help="是否进行滞后性处理，只需要进行一次即可。开启此选项进行一次处理后修改回为False，才有效。-------")
    parser.add_argument('--lag', type=int, default=0, help="滞后性处理的数值，代表滞后了多少，仅仅用于M或者MS模式----------")
    parser.add_argument('--original_multi_path', type=str, default='./Time_data/Uncleaned_data/价格-供求数据.xls',
                        help="供求价格的excel文件所在的路径")
    parser.add_argument('--output_multi_originalPath', type=str, default="./Time_data/Uncleaned_data/未进行滞后处理-价格-供求数据.csv",
                        help="生成供求价格的csv文件路径")
    parser.add_argument('--single_path', type=str, default="./Time_data/价格.csv", help="完整月均价数据的所在路径")
    parser.add_argument('--laged_multi_path', type=str, default="./data/Time_data/供求-价格.csv", help="经过滞后处理后的价格-供求数据")
    args = parser.parse_args()
    return args

"""
enc_in: informer的encoder的输入维度
dec_in: informer的decoder的输入维度
c_out: informer的decoder的输出维度
d_model: informer中self-attention的输入和输出向量维度
n_heads: multi-head self-attention的head数
e_layers: informer的encoder的层数
d_layers: informer的decoder的层数
d_ff: self-attention后面的FFN的中间向量表征维度
factor: probsparse attention中设置的因子系数
padding: decoder的输入中，作为占位的x_token是填0还是填1
distil: informer的encoder是否使用注意力蒸馏
attn: informer的encoder和decoder中使用的自注意力机制
embed: 输入数据的时序编码方式
activation: informer的encoder和decoder中的大部分激活函数
output_attention: 是否选择让informer的encoder输出attention以便进行分析

小数据集的预测可以先使用默认参数或适当减小d_model和d_ff的大小

"""

# 原理参考：https://blog.csdn.net/fluentn/article/details/115392229
# 文件格式：时间列名是date,XXX
if __name__ == '__main__':
    # 进行parser的变量初始化，获取实例。
    args = initialize_parameter()

    print("model：\t",args.model)
    if args.lag_sign:
        lag_processor_main(args.original_multi_path, args.output_multi_originalPath, args.single_path, args.lag, args.laged_multi_path)
        print("已经处理完 滞后性数值进程---回退args.lag_sign参数为False并且建议定制好实验才可继续往下进行~")
        sys.exit()
    # 判断GPU是否能够使用，并获取标识
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    # 判断是否使用多块GPU，默认不使用多块GPU
    if args.use_gpu and args.use_multi_gpu:
        # 获取显卡列表，type：str
        args.devices = args.devices.replace(' ', '')
        # 拆分显卡获取列表，type：list
        device_ids = args.devices.split(',')
        # 转换显卡id的数据类型
        args.device_ids = [int(id_) for id_ in device_ids]
        print("显卡设备id：", args.device_ids)
        # 获取第一块显卡
        args.gpu = args.device_ids[0]
    # 初始化数据解析器，用于定义训练模式、预测模式、数据粒度的初始化选项。
    """
    字典格式：{数据主题：{data：数据路径，'T':目标字段列名,'M'：，'S'：，'MS':}}

    'M:多变量预测多元（multivariate predict multivariate）'，
    'S:单变量预测单变量（univariate predict univariate）'，
    'MS:多变量预测单变量（multivariate predict univariate）'。
    """
    data_parser = {
        'ETTh1': {'data': 'ETTh1.csv', 'T': 'OT', 'M': [7, 7, 7], 'S': [1, 1, 1], 'MS': [7, 7, 1]},
        'ETTh2': {'data': 'ETTh2.csv', 'T': 'OT', 'M': [7, 7, 7], 'S': [1, 1, 1], 'MS': [7, 7, 1]},
        'ETTm1': {'data': 'ETTm1.csv', 'T': 'OT', 'M': [7, 7, 7], 'S': [1, 1, 1], 'MS': [7, 7, 1]},
        'ETTm2': {'data': 'ETTm2.csv', 'T': 'OT', 'M': [7, 7, 7], 'S': [1, 1, 1], 'MS': [7, 7, 1]},
        'WTH': {'data': 'WTH.csv', 'T': 'WetBulbCelsius', 'M': [12, 12, 12], 'S': [1, 1, 1], 'MS': [12, 12, 1]},
        'ECL': {'data': 'ECL.csv', 'T': 'MT_320', 'M': [321, 321, 321], 'S': [1, 1, 1], 'MS': [321, 321, 1]},
        'Solar': {'data': 'solar_AL.csv', 'T': 'POWER_136', 'M': [137, 137, 137], 'S': [1, 1, 1], 'MS': [137, 137, 1]},
        'Time_data': {'data': 'most_samll_test_3个变量.csv', 'T': 'X3', 'M': [3, 3, 3], 'S': [1, 1, 1], 'MS': [3, 3, 1]},
        'C5': {'data': 'C5_data.csv', 'T': 'NH3', 'M': [4, 4, 4], 'S': [1, 1, 1], 'MS': [4, 4, 1]},
        'electric_power': {'data': 'electric.csv', 'T': 'spg', 'M': [4, 4, 4], 'S': [1, 1, 1], 'MS': [5, 5, 1]},
    }
    # 判断在parser中定义的数据主题是否在解析器中
    if args.data in data_parser.keys():
        # 根据args里面定义的数据主题，获取对应的初始化数据解析器info信息，type：dict
        data_info = data_parser[args.data]
        # 获取该数据主题的数据文件的路径
        args.data_path = data_info['data']
        # 从数据解析器中获取 S或MS任务中的目标特征列名。
        args.target = data_info['T']
        # 从数据解析器中 根据变量features的初始化信息 获取 编码器输入大小，解码器输入大小，输出尺寸
        args.enc_in, args.dec_in, args.c_out = data_info[args.features]
    # 堆栈编码器层数，type：list
    args.s_layers = [int(s_l) for s_l in args.s_layers.replace(' ', '').split(',')]
    # 时间特征编码的频率，就是进行特征工程的时候时间粒度选取多少
    args.detail_freq = args.freq
    args.freq = args.freq[-1:]
    print('Args in experiment:')
    print(args)
    now_time = datetime.datetime.now().strftime('%mM_%dD %HH:%Mm:%Ss').replace(" ", "_").replace(":", "_")
    # 获取模型实例
    Exp = Exp_Informer
    # 获取page实例
    page_loss = get_page_loss(args.itr)
    page_predict_true = get_page_value(args.itr)
    page_predict = get_page_noTrue(args.itr)
    page_test = get_page_Test(args.itr)

    """
    存储数据的字典，为了将预测和均值和真实值存储到本地,(若是没有真实值，那么不存储真实值)
    存储未来预测值的真实数据，为了做可视化和评估未来
    存储模型信息的json文件
    存储预测未来的时候生成的时间
    """
    data_dict = dict()
    test_dict = dict()
    true = []
    info_dict = dict()
    pred_dates = []
    pred_dates2 = []
    try:
        data_true = get_true_data(args.true_sheetname,args.true_file,args)
        pred_dates2 = data_true["date"].astype(str).tolist()
        args.pred_len = len(data_true)
        print("真实值数据长度：", len(data_true))
        if args.features != 'M':
            true = data_true["{}".format(args.target)].values.tolist()
            data_dict["true"] = data_true["{}".format(args.target)].values.tolist()
        if args.features == 'M':
            data_columns = data_true.columns.values.tolist()
            print(data_columns)
            for i in range(args.c_out):
                true.append(data_true[data_columns[i + 1]].values.tolist())
                data_dict["true_{}".format(data_columns[i + 1])] = data_true[data_columns[i + 1]].values.tolist()
    except Exception as e:
        print("提示：由于未来还没有发生，在真实值数据中没有这个月份数据，故而无法画出未来预测值~未来值的对比图!")
        print(e)
    finally:
        print("Program to continue！>>>")
    print("实验预测未来的真实值：",data_dict)
    print("实验预测未来的时间：",pred_dates2)
    print(data_dict.keys())
    # sys.exit()
    # 构建单次运行的存储路径：
    run_name_dir_old = args.true_sheetname + "_" + args.model + "_" + str(args.itr) + "_" + now_time + "_" + args.data
    args.output = os.path.join(args.output,args.data+"_" + args.sub_them + "_"+args.freq)
    run_name_dir = os.path.join(args.output, run_name_dir_old)
    if not os.path.exists(run_name_dir):
        os.makedirs(run_name_dir)
    # 单次运行的n个实验的模型存储的路径：需要判断是否存在，训练的时候已经判断了
    run_name_dir_ckp_main = os.path.join(args.checkpoints, args.data+"_" + args.sub_them + "_"+args.freq)
    run_name_dir_ckp = os.path.join(run_name_dir_ckp_main, run_name_dir_old)
    # 存储整个实验的info信息
    info_file = os.path.join(run_name_dir, "{}_info_{}_{}.json".format(args.true_sheetname, args.model, args.data))
    df_columns = []
    test_columns = []
    # 要进行多少次实验，一次实验就是完成一个模型的训练-测试-预测 过程。默认2次
    for ii in range(args.itr):
        print("-------------.....第{}次实验.....------------".format(ii+1))
        run_ex_dir = os.path.join(run_name_dir, "第_{}_次实验记录".format(ii + 1))
        if args.random_choos == True:
            pass
        else:
            setup_seed(args.seed)
        if not os.path.exists(run_ex_dir):
            os.makedirs(run_ex_dir)
        # 添加实验info
        info_dict["实验序号"] = ii+1
        info_dict["model"] = args.model
        info_dict["data_them"] = args.data
        info_dict["编码器的输入序列长度 seq_len【滑动窗口大小】"] = args.seq_len
        info_dict["解码器的开始解码令牌起始位置 label_len"] = args.label_len
        info_dict["预测未来序列长度 pred_len"] = args.pred_len
        info_dict["时间特征编码的频率【数据粒度】freq"] = args.freq
        info_dict["dorpout"] = args.dropout
        info_dict["批次大小 batch_size"] = args.batch_size
        info_dict["提前停止的连续轮数 patience"] = args.patience
        info_dict["随机种子seed"] = args.seed
        info_dict["损失函数loss"] = args.loss
        info_dict["是否随机实验 random_choos"] = args.random_choos
        info_dict["滞后数值 lag"] = args.lag
        info_dict["编码器输入大小 enc_in"] = args.enc_in
        info_dict["解码器输入大小 dec_in"] = args.dec_in
        info_dict["输出尺寸 c_out"] = args.c_out
        info_dict["模型维数 d_model"] = args.d_model
        info_dict["多头部注意力机制的头部个数 n_heads"] = args.n_heads
        info_dict["编码器层数 e_layers"] = args.e_layers
        info_dict["解码器层数 d_layers"] = args.d_layers
        info_dict["堆栈编码器层数 s_layers"] = str(args.s_layers)
        info_dict["self-attention后面的FFN的中间向量表征维度 d_ff"] = args.d_ff
        info_dict["probsparse attn factor"] = args.factor
        info_dict["是否在编码器中不使用知识蒸馏 distil"] = args.distil
        info_dict["编码器的注意力机制 attn"] = args.attn
        info_dict["填充的值 padding"] = args.padding
        info_dict["时间特征编码 embed"] = args.embed
        info_dict["激活函数 activation"] = args.activation
        info_dict["是否在编码器中输出注意力 output_attention"] = args.output_attention
        info_dict["是否预测看不见的未来数据 do_predict"] = args.do_predict
        info_dict["在生成解码器中使用混合注意力 mix"] = args.mix
        info_dict["实验次数 itr"] = args.itr
        info_dict["校正的学习率 lradj"] = args.lradj
        info_dict["使用自动混合精度训练 use_amp"] = args.use_amp
        info_dict["逆标准化输出数据 inverse"] = args.inverse
        info_dict["优化器初始学习率 learning_rate"] = args.learning_rate

        # 实验设置记录要点，方便打印，同时也作为文件名字传入参数，setting record of experiments
        setting = '{}_{}_{}_{}'.format(ii+1,args.model,args.data,args.features,)
        # 设置实验，将数据参数和模型变量传入实例
        exp = Exp(args)  # set experiments

        # 训练模型
        print('>>>>>>>start training :  {}  >>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        model, info_dict, all_epoch_train_loss, all_epoch_vali_loss, all_epoch_test_loss, epoch_count = exp.train(
            setting, info_dict, run_name_dir_ckp, run_ex_dir,args)

        # 模型测试：存在问题：无法取到时间，boder不确定
        print('>>>>>>>testing :  {}  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        # test： 返回的是数组
        # sys.exit()
        info_dict, test_pred, test_true = exp.test(setting, info_dict, run_ex_dir,args)

        future_pred, pred_date = 0, 0
        # 做预测
        if args.do_predict:
            print('>>>>>>>predicting :  {}  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            # 模型预测未来
            future_pred, pred_date = exp.predict(setting, run_name_dir_ckp, run_ex_dir,args, load=args.save_model_choos)
            pred_dates = pred_date
            # assert pred_dates == pred_dates2
        # 存储实验的info信息：
        with open(info_file, mode='a', encoding='utf-8') as f:
            json.dump(info_dict, f, indent=4, ensure_ascii=False)
        # 存储数据：
        df = pd.DataFrame(columns=args.columns)
        if args.features != 'M':
            # 存储未来预测值
            df = pd.DataFrame(data={args.columns[0]: pred_date,args.columns[1]: future_pred,}, columns=args.columns)
            df.to_csv(os.path.join(run_ex_dir, "第{}次实验_未来预测结果.csv".format(ii + 1)), index=False, encoding='utf-8')

            # 预存储测试集评估值
            df2 = pd.DataFrame(data={args.columns[0]: list(range(len(test_pred))), args.columns[1]: test_pred, }, columns=args.columns)
            df2.to_csv(os.path.join(run_ex_dir, "第{}次实验_测试集预测结果.csv".format(ii + 1)), index=False, encoding='utf-8')

            # 存储预测结果到字典
            data_dict["实验{}".format(ii + 1)] = future_pred
            # 添加字段名字
            df_columns.append("实验{}".format(ii + 1))

            # 测试集操作
            test_dict['true'] = test_true.tolist()
            # 存储预测结果到字典
            test_dict["实验{}".format(ii + 1)] = test_pred.tolist()
            test_columns.append("实验{}".format(ii + 1))
            # 可视化：
            line_p = chart_predict(pred_date, future_pred, run_ex_dir, args, ii + 1)
            line_loss = chart_loss(all_epoch_train_loss, all_epoch_vali_loss, all_epoch_test_loss, epoch_count,
                                   run_ex_dir,
                                   args, ii + 1)
            # print(type(test_pred),test_pred)
            # print(type(test_true),test_true)
            # sys.exit()
            line_test = chart_test(range(len(test_true)), test_pred, test_true, run_ex_dir, args, ii+1)
            page_test.add(line_test)
            # 将预测的预测和未来的真实值一起进行可视化
            if true != []:
                line_pt = chart_predict_and_true(pred_date, future_pred, true, run_ex_dir, args, ii + 1)
                page_predict_true.add(line_pt)
            # 将图表加入page
            page_loss.add(line_loss)
            page_predict.add(line_p)
        # print(test_dict)
        # sys.exit()
        if args.features == 'M':
            # 构建dict
            df_dict = dict()
            df_dict2 = dict()
            df_dict2[args.columns[0]]= list(range(len(test_pred)))
            df_dict[args.columns[0]]=pred_date
            print(pred_date)
            # 处理test_true

            for i in range(args.c_out):
                df_dict[args.columns[i+1]] = [round(f,1) for f in future_pred[:,i].flatten().tolist()]
                df_dict2[args.columns[i+1]] = [round(f,1) for f in test_pred[:,i].flatten().tolist()]
                # 存储预测结果到字典
                data_dict["实验{}_{}".format(ii + 1,args.columns[i+1])] = [round(f,1) for f in future_pred[:,i].flatten().tolist()]
                # 添加字段名字
                df_columns.append("实验{}_{}".format(ii + 1,args.columns[i+1]))

                # 测试集操作
                test_dict['true_{}'.format(args.columns[i+1])] = test_true[:,i].tolist()
                # print('true_{}'.format(args.columns[i+1]),"\t",test_true[:,i])
                # sys.exit()
                # 存储预测结果到字典
                test_dict["实验{}_{}".format(ii + 1,args.columns[i+1])] = [round(f,1) for f in test_pred[:,i].flatten().tolist()]
                test_columns.append("实验{}_{}".format(ii + 1, args.columns[i + 1]))
            #--------------------------------------------------
            df = pd.DataFrame(data=df_dict, columns=args.columns)
            df.to_csv(os.path.join(run_ex_dir, "第{}次实验_未来预测结果.csv".format(ii + 1)), index=False, encoding='utf-8')
            # -------------------------------------
            # 预存储测试集评估值
            df2 = pd.DataFrame(data=df_dict2,columns=args.columns)
            df2.to_csv(os.path.join(run_ex_dir, "第{}次实验_测试集预测结果.csv".format(ii + 1)), index=False, encoding='utf-8')
            # print(pred_date)
            # sys.exit()
            # 可视化：
            line_p = chart_predict(pred_date, future_pred, run_ex_dir, args, ii + 1)
            line_loss = chart_loss(all_epoch_train_loss, all_epoch_vali_loss, all_epoch_test_loss, epoch_count,run_ex_dir,args, ii + 1)
            line_test = chart_test(range(len(test_true)), test_pred, test_true, run_ex_dir, args, ii + 1)
            page_test.add(line_test)
            # 将预测的预测和未来的真实值一起进行可视化
            if true != []:
                line_pt = chart_predict_and_true(pred_date, future_pred, true, run_ex_dir, args, ii + 1)
                page_predict_true.add(line_pt)
            # 将图表加入page
            page_loss.add(line_loss)
            page_predict.add(line_p)
        # 清除cuda的缓存
        torch.cuda.empty_cache()
        if args.save_model_choos==False:
            # 删除存储的模型
            shutil.rmtree(run_name_dir_ckp)
        #-------------------------------------------------------------------------------------------------
    # print(test_dict)
    # sys.exit()
    if args.save_model_choos == False:
        shutil.rmtree(run_name_dir_ckp_main)
    visual_test_path = os.path.join(run_name_dir,'visual_tmp')
    if not os.path.exists(visual_test_path):
        os.makedirs(visual_test_path)
    # 可视化page
    page_loss.render(os.path.join(visual_test_path, "训练-验证-损失可视化-test.html"))
    page_loss.save_resize_html(source=os.path.join(visual_test_path, "训练-验证-损失可视化-test.html"),
                               cfg_file=os.path.join('./utils/', "chart_config.json"),
                               dest=os.path.join(run_name_dir, "训练-验证-损失可视化.html"))

    page_predict.render(os.path.join(visual_test_path, "predict-test.html"))
    page_predict.save_resize_html(source=os.path.join(visual_test_path, "predict-test.html"),
                            cfg_file=os.path.join('./utils/', "chart_config.json"),
                            dest=os.path.join(run_name_dir, "predict.html"))

    page_test.render(os.path.join(visual_test_path, "Test_size-test.html"))
    page_test.save_resize_html(source=os.path.join(visual_test_path, "Test_size-test.html"),
                                  cfg_file=os.path.join('./utils/', "chart_config.json"),
                                  dest=os.path.join(run_name_dir, "Test_size.html"))

    # 存储字典文件
    data_dict["date"] = pred_dates
    df = pd.DataFrame(data_dict)

    test_dict['date'] = list(range(1, len(test_true.tolist()) + 1))
    df_test = pd.DataFrame(test_dict)
    # print(test_dict)
    # print("****"*5)
    # print(list(range(1, len(test_true.tolist()) + 1)))
    # sys.exit()
    # ----------------------------
    if true != []:
        page_predict_true.render(os.path.join(visual_test_path, "predict-true-test.html"))
        page_predict_true.save_resize_html(source=os.path.join(visual_test_path, "predict-true-test.html"),
                                 cfg_file=os.path.join('./utils/', "chart_config.json"),
                                 dest=os.path.join(run_name_dir, "predict-true.html"))
    # 计算均值
    if args.features != 'M':
        df["pred_mean_{}".format(args.target)] = df[df_columns].mean(axis=1)
        df_columns.insert(0, 'pred_mean_{}'.format(args.target))
        df_columns.insert(0, 'true')
        df["pred_mean_{}".format(args.target)] = round(df["pred_mean_{}".format(args.target)], 1)

        df_test["pred_mean_{}".format(args.target)] = df_test[test_columns].mean(axis=1)
        test_columns.insert(0, 'pred_mean_{}'.format(args.target))
        test_columns.insert(0, 'true')
        df_test["pred_mean_{}".format(args.target)] = round(df_test["pred_mean_{}".format(args.target)], 1)

    if args.features == 'M':
        for i in range(args.c_out):
            df["pred_mean_{}".format(args.columns[i+1])] = df[[s for s in df_columns if args.columns[i+1] in s]].mean(axis=1)
            df_columns.insert(0, "pred_mean_{}".format(args.columns[i + 1]))
            df_columns.insert(0, list(data_dict.keys())[i])

            df_test["pred_mean_{}".format(args.columns[i + 1])] = df_test[[s for s in test_columns if args.columns[i + 1] in s]].mean(axis=1)
            test_columns.insert(0, "pred_mean_{}".format(args.columns[i + 1]))
            test_columns.insert(0, list(data_dict.keys())[i])

    df_columns.insert(0, 'date')
    df = df[df_columns]
    df = calculate_var(df,args)
    print("--------"*3)
    df.iloc[:,1:] = df.iloc[:,1:].round(1)
    df.to_csv(os.path.join(run_name_dir, "{}次实验_{}未来预测结果.csv".format(args.itr, args.true_sheetname)), index=False, encoding='utf-8',sep=',')

    # -------------------------------------
    test_columns.insert(0, 'date')
    df_test = df_test[test_columns]
    # print(df_test)
    # sys.exit()
    df_test = calculate_var(df_test, args)
    # print("--------" * 3)
    df.iloc[:, 1:] = df.iloc[:, 1:].round(1)
    df_test.to_csv(os.path.join(run_name_dir, "{}次实验_{}测试集预测结果.csv".format(args.itr, args.true_sheetname)), index=False,encoding='utf-8', sep=',')

    try:
        # 可视化预测的均值与真实值,df是为了输入预测值
        chart_avg_predict_and_true(df,run_name_dir, args)
    except Exception as e:
        print("没有读取到真实值文件夹里面有真实值!")
        print(e)
    print("--------" * 3)
    # 计算测试集、预测集的平均MAE指标
    MAE_dict = dict()
    df_test_mae_calculation = df_test[df_test.columns[1:len(args.columns)*2-1]]
    test_columns = df_test_mae_calculation.columns.tolist()
    df_mae_calculation = df[df.columns[1:len(args.columns)*2-1]]
    df_columns = df_mae_calculation.columns.tolist()
    out_columns = args.columns.copy()
    out_columns.reverse()
    #准备一个临时
    # 0 1 2 3
    for i in range(len(args.columns)-1):
        MAE_dict["test_mae_{}".format(out_columns[i])] = \
            np.around(np.abs(np.mean(np.array(df_test_mae_calculation[test_columns[i*2]]) - np.array(df_test_mae_calculation[test_columns[i*2+1]]))),3)
        try:
            MAE_dict["predict_mae_{}".format(out_columns[i])] = \
                np.around(np.abs(np.mean(np.array(df_mae_calculation[df_columns[i * 2]]) - np.array(df_mae_calculation[df_columns[i * 2 + 1]]))),3)
        except:
            print("没有读取到真实值文件夹里面有真实值!")
    print(MAE_dict)
    json_data = json.dumps(MAE_dict,ensure_ascii=False, indent=4)
    with open(os.path.join(run_name_dir, "{}次实验平均评估指标.csv".format(args.itr)), 'a',encoding='utf8') as f_six:
        f_six.write(json_data)
end = time.time()
print("程序运行时间：\t",str(end-start))