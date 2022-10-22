
# 数据加载器
import datetime
import sys

# 在自定义的data模块中
import pandas as pd

from data.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Pred
#
from exp.exp_basic import Exp_Basic
# 导入模型
from models.model import Informer, InformerStack

# 提前停止策略、修正学习率
from utils.tools import EarlyStopping, adjust_learning_rate
# 评价指标
from utils.metrics import metric

import numpy as np

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

import os
import time

import warnings
warnings.filterwarnings('ignore')

# 继承Exp_Basic类
class Exp_Informer(Exp_Basic):
    def __init__(self, args):
        super(Exp_Informer, self).__init__(args)

    # 构造模型
    def _build_model(self):
        model_dict = {
            'informer':Informer,
            'informerstack':InformerStack,
        }
        if self.args.model=='informer' or self.args.model=='informerstack':
            e_layers = self.args.e_layers if self.args.model=='informer' else self.args.s_layers
            model = model_dict[self.args.model](
                self.args.enc_in,
                self.args.dec_in, 
                self.args.c_out, 
                self.args.seq_len, 
                self.args.label_len,
                self.args.pred_len, 
                self.args.factor,
                self.args.d_model, 
                self.args.n_heads, 
                e_layers, # self.args.e_layers,
                self.args.d_layers, 
                self.args.d_ff,
                self.args.dropout, 
                self.args.attn,
                self.args.embed,
                self.args.freq,
                self.args.activation,
                self.args.output_attention,
                self.args.distil,
                self.args.mix,
                self.device
            ).float()
        
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    # 获取数据并进行处理，返回符合输入格式的数据
    def _get_data(self, flag):
        args = self.args

        data_dict = {
            'ETTh1':Dataset_ETT_hour,
            'ETTh2':Dataset_ETT_hour,
            'ETTm1':Dataset_ETT_minute,
            'ETTm2':Dataset_ETT_minute,
            'WTH':Dataset_Custom,
            'ECL':Dataset_Custom,
            'Solar':Dataset_Custom,
            '{}'.format(self.args.data):Dataset_Custom,
            'custom':Dataset_Custom,
        }
        # 下面这个Data，此时是一个Dataset_Custom。
        # self.args.data：chicken;    Data是Dataset_Custom对象
        Data = data_dict[self.args.data]
        #
        timeenc = 0 if args.embed!='timeF' else 1

        # flag:设置任务类型
        # 根据flag设置训练设置和数据操作设置
        # 做测试的时候
        if flag == 'test':
            shuffle_flag = False; drop_last = True; batch_size = 1; freq=args.freq
        # 做预测的时候
        elif flag=='pred':
            # 如果是预测未来的任务
            shuffle_flag = False; drop_last = False; batch_size = 1; freq=args.detail_freq
            # 因为是预测任务，所以Data被赋值为Dataset_Pred对象
            Data = Dataset_Pred
        # 做数据的时候: train和vali的选项:打乱数据
        elif flag == 'val':
            shuffle_flag = False; drop_last = True; batch_size = 1; freq=args.freq
        else:
            shuffle_flag = True; drop_last = True; batch_size = args.batch_size; freq=args.freq
        # 使用Dataset_Custom进行读取数据集，并转换为数组.:
        # 实例化Dataset_Custom对象
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            inverse=args.inverse,
            timeenc=timeenc,
            freq=freq,
            scale=args.scale,
            cols=args.cols,
            args=args
        )

        """
        (96, 1)
        (72, 1)
        (96, 3)
        (72, 3)
        """
        """
        返回读取的数据且是一个iterable，可迭代对象。这个可迭代对象里面是4个数组，对应了
        """
        # sys.exit()
        # print(flag,":\t", len(data_set))
        # 对data_set使用DataLoader，这里的shuffle决定了是否把数据打乱
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)
        """
        drop_last代表将不足一个batch_size的数据是否保留，即假如有4条数据，batch_size的值为3，将取出一个batch_size之后剩余的1条数据是否仍然作为训练数据，即是否丢掉这条数据。
        """

        """
        torch.Size([32, 96, 1])
        torch.Size([32, 72, 1])
        torch.Size([32, 96, 3])
        torch.Size([32, 72, 3])
        """
        """
        DataLoader就是将数据data_set组装起来成input的格式，且是一个iterable，可迭代对象。这个输入格式是序列的输入格式，[批次大小batch_size，输入序列长度seq_len，特征(有多少列)数量]。
        其中，输入序列长度seq_len相当于是滑动窗口的大小。
        """
        return data_set, data_loader

    # 选择模型优化器（这里是adam）
    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    # 选择损失标准(损失函数)
    def _select_criterion(self):
        # https://pytorch.org/docs/stable/nn.functional.html
        # 默认是mse
        criterion = nn.MSELoss()
        if self.args.loss == 'mse':
            criterion = nn.MSELoss()
        if self.args.loss == 'L1loss':
            criterion = nn.L1Loss()
        if self.args.loss == 'huberloss':
            criterion = nn.SmoothL1Loss()
        return criterion

    # 验证集的验证
    def vali(self,vali_data,vali_loader,criterion,args):
        self.model.eval()
        total_loss = []
        for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(vali_loader):
            pred, true = self._process_one_batch(
                vali_data, batch_x, batch_y, batch_x_mark, batch_y_mark,args)
            pred = pred[:, :, -1:] if args.features == 'MS' else pred
            # print(type(pred), pred.shape)
            # print(pred)
            # print(type(true), true.shape)
            # print(true)
            # print("-----------" * 4)
            # sys.exit()
            loss = criterion(pred.detach().cpu(), true.detach().cpu())
            total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    # 训练集的训练
    def train(self,setting,info_dict,run_name_dir_ckp,run_ex_dir,args):
        # 做训练的时候这里面已经测试集评估功能 和 验证集的验证功能了,args.save_model_choos
        global scaler
        train_data, train_loader = self._get_data(flag = 'train')
        vali_data, vali_loader = self._get_data(flag = 'val')
        test_data, test_loader = self._get_data(flag = 'test')
        # 存储模型的位置
        path = os.path.join(run_name_dir_ckp, setting)
        # path = os.path.join(run_ex_dir, setting)#将模型和可视化文件存储在一起
        if not os.path.exists(path):
            os.makedirs(path)
        time_now = time.time()
        # 训练步数
        train_steps = len(train_loader)
        # 提前停止（保存模型代码在里面）
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True,save_model_choos=args.save_model_choos)
        # 模型优化器
        model_optim = self._select_optimizer()
        # 损失函数
        criterion =  self._select_criterion()
        if self.args.use_amp:
            # autocast + GradScaler 可以达到自动混合精度训练的目的；
            # GradScaler是梯度
            scaler = torch.cuda.amp.GradScaler()
        # 训练的时候记录每个epoch产生的损失，包括训练集损失、验证集损失、测试集(评估集)损失
        all_epoch_train_loss = []
        all_epoch_vali_loss = []
        all_epoch_test_loss = []
        # 训练args.train_epochs个epoch，每一个epoch循环一遍整个数据集
        epoch_count = 0
        for epoch in range(self.args.train_epochs):
            epoch_count += 1
            iter_count = 0
            # 存储当前epoch下的每个迭代步的训练损失
            train_loss = []
            """
            模型中有BN层(Batch Normalization）和Dropout，需要在训练时添加model.train()，在测试时添加model.eval()。
            
            其中model.train()是保证BN层用每一批数据的均值和方差，而model.eval()是保证BN用全部训练数据的均值和方差；
            而对于Dropout，model.train()是随机取一部分网络连接来训练更新参数，而model.eval()是利用到了所有网络连接。
            """
            self.model.train()
            epoch_time = time.time()
            # 在每个epoch里面迭代数据训练模型：遍历一遍数据
            for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(train_loader):
                # 累计迭代次数
                iter_count += 1
                # 把模型的参数梯度设置为0:
                model_optim.zero_grad()
                # 训练集的预测值和真实值 : 这里的真实值是输入数据-滑动窗口，预测值是滑动川口里面的对应预测值。[批次,预测长度,1]
                pred, true = self._process_one_batch(train_data, batch_x, batch_y, batch_x_mark, batch_y_mark,args)
                # 对于多变量，把数组打平，【然后归一化】，然后再计算损失。
                pred = pred[:, :, -1:] if args.features == 'MS' else pred
                # print(type(pred),pred.shape)
                # print(pred)
                # print(type(true),true.shape)
                # print(true)
                # print("-----------"*4)
                # sys.exit()
                # 计算损失
                # print(type(true),true.dtype)
                # print(type(pred),pred.dtype)
                """
                true:    <class 'torch.Tensor'> torch.float32
                pred:    <class 'torch.Tensor'> torch.float16
                """
                loss = criterion(pred.float(), true.float())
                # loss = criterion(pred.double(), true.double())
                # sys.exit()
                # 将每个迭代步的loss添加到train_loss列表
                train_loss.append(loss.item())
                # 每迭代一百个样本就打印一次
                if (i+1) % 100==0:
                    # 查看迭代100个样本所花费的时间，和这100个样本的训练损失值，还有当前所在epoch
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time()-time_now)/iter_count
                    left_time = speed*((self.args.train_epochs - epoch)*train_steps - i)
                    # 查看处理速度
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
                if self.args.use_amp:
                    # 达到自动混合精度训练的目的
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()
            # 打印遍历一遍整个训练集 所需要的时间，也就是此次epoch所需要的时间
            print("Epoch: {} cost time: {}".format(epoch+1, time.time()-epoch_time))
            # 对训练集损失求均值
            train_loss = np.average(train_loss)
            # 验证集验证
            vali_loss = self.vali(vali_data, vali_loader, criterion,args)
            # 测试集进行评估模型，其实这里也是达到验证的作用
            test_loss = self.vali(test_data, test_loader, criterion,args)
            # 添加到列表中留存
            all_epoch_train_loss.append(float(round(train_loss,1)))
            all_epoch_vali_loss.append(float(round(vali_loss,1)))
            all_epoch_test_loss.append(float(round(test_loss,1)))
            # 完成每个epoch的训练就打印一次
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            # 判断是否提前停止
            early_stopping(vali_loss, self.model, path,args.save_model_choos)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            # 更新学习率
            adjust_learning_rate(model_optim, epoch+1, self.args)
        # 存储该次实验的更新迭代中的最优模型
        if args.save_model_choos==True:
            best_model_path = path+'/'+'checkpoint.pth'
            # 下面是加载模型，（这个模型最终在预测完之后要删除，因为占用内存大）
            self.model.load_state_dict(torch.load(best_model_path))
        # 实验记录
        info_dict["【训练】本次实验训练的train平均损失"] = round(float(np.mean(all_epoch_train_loss)),1)
        info_dict["【验证】本次实验训练的vali平均损失"]  = round(float(np.mean(all_epoch_vali_loss)),1)
        info_dict["【验证】本次实验训练的test平均损失"]  = round(float(np.mean(all_epoch_test_loss)),1)
        info_dict["----实际训练的epoch-------"] = epoch_count

        return self.model,info_dict,all_epoch_train_loss,all_epoch_vali_loss,all_epoch_test_loss,epoch_count

    # 测试集测试
    def test(self,setting,info_dict,run_ex_dir,args):
        test_data, test_loader = self._get_data(flag='test')#做测试的时候
        # 不启用 BatchNormalization 和 Dropout，因为不是训练模式
        self.model.eval()
        preds = []
        trues = []
        # batch_x是输入的一个批次的x数据，
        for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(test_loader):
            # print(batch_x.shape, batch_y.shape, batch_x_mark.shape, batch_y_mark.shape)
            # print(batch_x, batch_y)
            # 返回的是数组,注意：loader里面已经把数据打乱了
            pred, true = self._process_one_batch(
                test_data, batch_x, batch_y, batch_x_mark, batch_y_mark,args)
            pred = pred[:, :, -1:] if args.features == 'MS' else pred
            # print(type(pred),pred.shape)
            # print(pred)
            # print(type(true),true.shape)
            # print(true)
            # print("-----------"*4)
            # sys.exit()
            # 把数组添加到列表
            preds.append(pred.detach().cpu().numpy())
            trues.append(true.detach().cpu().numpy())
            # if args.inverse == False:
            #     inverse_true = Standardization.inverse_transform(true)
            #     inverse_pred = Standardization.inverse_transform(pred)
        preds = np.array(preds)
        trues = np.array(trues)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        """
        test shape: (29, 32, 24, 1) (29, 32, 24, 1)
        test shape: (928, 24, 1) (928, 24, 1)
        """
        # result save
        folder_path = run_ex_dir+'/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        if args.features == 'M':
            # 取到最后一个值，因为那个才是要预测的。
            trues = trues[:,-1:,:]
            preds = preds[:,-1:,:]
            trues = trues.reshape(len(trues),trues.shape[-1])
            preds = preds.reshape(len(preds),preds.shape[-1])
            # print(trues)
            # sys.exit()
            trues = np.around(trues,decimals=1)
            preds = np.around(preds,decimals=1)
            trues = trues.tolist()
            preds = preds.tolist()
            preds = np.array(preds)
            trues = np.array(trues)

        if args.features != 'M':
            trues = trues[:, -1:, :]
            preds = preds[:, -1:, :]
            trues = trues.flatten()
            preds = preds.flatten()
            trues = np.around(trues, decimals=1)
            preds = np.around(preds, decimals=1)
            trues = trues.tolist()
            preds = preds.tolist()
            preds = [round(i, 1) for i in preds]
            trues = [round(i, 1) for i in trues]
            preds = np.array(preds)
            trues = np.array(trues)

        # 评估指标：测试集评估模型
        mae,rmse,smape,r2,ad_r2 = metric(preds, trues)
        mae, rmse, smape, r2, ad_r2 = round(float(mae),1),round(float(rmse),1),round(float(smape),1),round(float(r2),1),round(float(ad_r2),1)
        print('测试集评估结果：\t平均绝对误差 MAE:{}，均方根误差RMSE:{}，对称平均绝对百分比误差SMAPE:{}，决定系数R²：{}，校正R²:{} \n'.format(mae,rmse,smape,r2,ad_r2))
        # 存储评估指标
        info_dict["【评估】本次实验的test集平均绝对误差MAE"] = mae
        info_dict["【评估】本次实验的test集均方根误差RMSE"] = rmse
        info_dict["【评估】本次实验的test集对称平均绝对百分比误差SMAPE"] = smape
        info_dict["【评估】本次实验的test集决定系数R²"] = r2
        info_dict["【评估】本次实验的test集校正决定系数Ad_R²"] = ad_r2
        # 存储评估指标和向量
        np.save(folder_path+'metrics.npy', np.array([mae,rmse,smape,r2,ad_r2]))
        np.save(folder_path+'pred.npy', preds)
        np.save(folder_path+'true.npy', trues)
        if args.inverse == False:
            pass
        return info_dict,preds,trues

    # 预测未来
    def predict(self, setting,run_name_dir_ckp, run_ex_dir,args,load=False):
        # 从_get_data获取数据，【这句代码的返回结果搞不明白】
        pred_data, pred_loader = self._get_data(flag='pred')
        pred_date = pred_data.pred_date
        if args.freq[-1] == "t" or args.freq[-1] == 'h' or args.freq[-1] == 's':
            pred_date = [str(p) for p in pred_date[1:]]
        else:
            pred_date = [str(p).split(" ")[0] for p in pred_date[1:]]
        print("本次实验预测未来的时间范围：",pred_date)
        # 加载模型
        if load:
            path = os.path.join(run_name_dir_ckp ,setting)
            # path = os.path.join(run_ex_dir ,setting)
            best_model_path = path+'/'+'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))
        # 清楚缓存
        self.model.eval()
        preds = []
        
        for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(pred_loader):
            # print(batch_x.shape,batch_y.shape,batch_x_mark.shape,batch_y_mark.shape)
            # torch.Size([1, 96, 1]) torch.Size([1, 48, 1]) torch.Size([1, 96, 3]) torch.Size([1, 72, 3])
            """
            [1, 96, 1]是输入的一个批次的X数据，可以认为是滑动窗口为96的X。
            [1, 48, 1]是输入的一个批次的Y数据，可以认为是滑动窗口为96的X的标签数据，48是inform解码器的开始令牌长度label_len，多步预测的展现。
            
            [1, 96, 3]是输入的X数据的Q、K、V向量的数组。
            [1, 72, 3]是输入的Y数据的Q、K、V向量的数组,其中，72=48+24，48是label_len，24是预测序列长度pred_len，也就是说24是被预测的，这里是作为已知输入的。
            """
            # print(batch_x.shape, batch_y.shape, batch_x_mark.shape, batch_y_mark.shape)
            # sys.exit()
            pred, true = self._process_one_batch(pred_data, batch_x, batch_y, batch_x_mark, batch_y_mark,args)
            preds.append(pred.detach().cpu().numpy())


        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        preds = preds[:, :, -1:] if args.features == 'MS' else preds
        # print(preds)
        # print(type(preds),len(preds),preds.shape,preds)
        # sys.exit()
        # result save
        folder_path = run_ex_dir+'/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        if args.features == 'M':
            preds = preds[0]
            print("本次实验预测未来的结果：", preds)
            # 存储未来的预测结果到npy文件
            np.save(folder_path + 'real_prediction.npy', preds)
            assert len(preds) == len(pred_date)
            return preds, pred_date
        if args.features != 'M':
            # 这里要修改
            preds = preds.flatten().tolist()
            preds = [round(i, 1) for i in preds]
            print("本次实验预测未来的结果：",preds)
            # 存储未来的预测结果到npy文件
            np.save(folder_path+'real_prediction.npy', preds)
            assert len(preds) == len(pred_date)
            return preds, pred_date
        return preds,pred_date

    # 对一个batch进行的编码解码操作，就是训练模型
    def _process_one_batch(self, dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark,args):
        global dec_inp
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float()

        batch_x_mark = batch_x_mark.float().to(self.device)
        batch_y_mark = batch_y_mark.float().to(self.device)

        # decoder input
        if self.args.padding==0:
            # 返回一个形状为为size，size是一个list，代表了数组的shape,类型为torch.dtype，里面的每一个值都是0的tensor
            dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float()
        elif self.args.padding==1:
            dec_inp = torch.ones([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float()
        # 在给定维度上对输入的张量序列seq 进行连接操作。
        """
        outputs = torch.cat(inputs, dim=0) → Tensor
        
        inputs : 待连接的张量序列，可以是任意相同Tensor类型的python 序列，可以是列表或者元组。
        dim : 选择的扩维, 必须在0到len(inputs[0])之间，沿着此维连接张量序列。
        """
        dec_inp = torch.cat([batch_y[:,:self.args.label_len,:], dec_inp], dim=1).float().to(self.device)
        # encoder - decoder（编码器-解码器）
        # 假如使用自动混合精度训练
        if self.args.use_amp:
            # pytorch 使用autocast半精度进行加速训练
            with torch.cuda.amp.autocast():
                # 假如在编码器中输出注意力
                if self.args.output_attention:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        # 假如不使用自动混合精度训练
        else:
            if self.args.output_attention:
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
            else:
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        # 逆标准化输出数据
        if self.args.inverse:
            outputs = dataset_object.inverse_transform(outputs)
        f_dim = -1 if self.args.features=='MS' else 0
        # 如果是MS。那么只留有一列输出
        # outputs = outputs[:, :, 1:] if args.features == 'MS' else outputs
        # 对y进行解码
        batch_y = batch_y[:,-self.args.pred_len:,f_dim:].to(self.device)

        # 如果是M任务，那么进行打平再输出去计算梯度
        return outputs, batch_y
