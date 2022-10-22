# !/usr/bin python3
# encoding    : utf-8 -*-
# @author     :   陈汝海
# @software   : PyCharm
# @file       :   visualization.py
# @Time       :   2021/10/14 10:50

# 可视化测试集
import os
import sys

import pandas as pd
from pyecharts.charts import Line,Page,Grid
from pyecharts import options as opts
from pyecharts.globals import ThemeType
from pyecharts.globals import CurrentConfig, OnlineHostType

# https://pyecharts.org/#/zh-cn/themes

# 可视化损失值
def chart_loss(train_loss,val_loss,test_loss,epoch_count,run_ex_dir,args,i):
    # 查看真实值 与预测值的差距：为训练集和测试集准备的函数
    line = Line(init_opts=opts.InitOpts(theme=ThemeType.SHINE))
    line.set_global_opts(
        tooltip_opts=opts.TooltipOpts(is_show=True),
        xaxis_opts=opts.AxisOpts(type_="category",name="epoch"),
        yaxis_opts=opts.AxisOpts(
            type_="value",
            name="损失值",
            min_='dataMin',
            axistick_opts=opts.AxisTickOpts(is_show=True),
            splitline_opts=opts.SplitLineOpts(is_show=True),
        ),
        legend_opts=opts.LegendOpts(pos_right='10%'),
        datazoom_opts=opts.DataZoomOpts(is_show=True,),
        # datazoom_opts=opts.DataZoomOpts(is_show=True,type_="inside"),
        title_opts=opts.TitleOpts(title='第{}次实验'.format(i))
    )
    epoch_count = [i+1 for i in range(epoch_count)]
    line.add_xaxis(xaxis_data=epoch_count)
    line.add_yaxis(
        series_name="{}_loss".format("train"),
        y_axis=train_loss,
        is_smooth=True,
        label_opts=opts.LabelOpts(is_show=False),
        markpoint_opts=opts.MarkPointOpts(data=[opts.MarkPointItem(type_="min"),opts.MarkLineItem(type_="average")])
    )
    line.add_yaxis(
        series_name="{}_loss".format("val"),
        y_axis=val_loss,
        is_smooth=True,
        label_opts=opts.LabelOpts(is_show=False),
        markpoint_opts=opts.MarkPointOpts(data=[opts.MarkPointItem(type_="min"),opts.MarkLineItem(type_="average")])
    )
    line.add_yaxis(
        series_name="{}_loss".format("test"),
        y_axis=test_loss,
        is_smooth=True,
        label_opts=opts.LabelOpts(is_show=False),
        markpoint_opts=opts.MarkPointOpts(data=[opts.MarkPointItem(type_="min"),opts.MarkLineItem(type_="average")]),
    )
    line.render(os.path.join(run_ex_dir, '{}_Loss_{}_{}_loss.html'.format(i,args.model,args.data)))
    print("loss chart action is completed!")
    return line

def chart_test(time_list,pred,true,run_ex_dir,args,i):
    # 查看真实值 与预测值的差距：为训练集和测试集准备的函数
    line = Line(init_opts=opts.InitOpts(theme=ThemeType.SHINE))
    line.set_global_opts(
        tooltip_opts=opts.TooltipOpts(is_show=True),
        xaxis_opts=opts.AxisOpts(type_="category",name="time",axislabel_opts=opts.LabelOpts(is_show=True,rotate=10)),
        yaxis_opts=opts.AxisOpts(
            type_="value",
            name="{}".format(args.target),
            min_='dataMin',
            axistick_opts=opts.AxisTickOpts(is_show=True),
            splitline_opts=opts.SplitLineOpts(is_show=True),
        ),
        datazoom_opts=opts.DataZoomOpts(is_show=True),
        # datazoom_opts=opts.DataZoomOpts(is_show=True,type_="inside"),
        title_opts=opts.TitleOpts(title='模型评估：测试集表现')
    )
    line.add_xaxis(xaxis_data=time_list)
    if args.features != 'M':
        line.add_yaxis(
            series_name="Test True {}".format(args.target),
            y_axis=true.tolist(),
            label_opts=opts.LabelOpts(is_show=args.is_show_label),
            markline_opts=opts.MarkLineOpts(data=[opts.MarkLineItem(type_="average")]),
        )
        line.add_yaxis(
            series_name="Test Predicted {}".format(args.target),
            y_axis=pred.tolist(),
            symbol="emptyCircle",
            is_symbol_show=True,
            label_opts=opts.LabelOpts(is_show=args.is_show_label),
            markline_opts=opts.MarkLineOpts(data=[opts.MarkLineItem(type_="average")]),
        )
    if args.features == 'M':
        for i in range(args.c_out):
            line.add_yaxis(
                series_name='pred_{}'.format(args.columns[i + 1]),
                y_axis=[round(f, 1) for f in pred[:, i].flatten().tolist()],
                is_smooth=True,
                label_opts=opts.LabelOpts(is_show=args.is_show_label),
                markline_opts=opts.MarkLineOpts(data=[opts.MarkLineItem(type_="average")]),
            )
            line.add_yaxis(
                series_name='true_{}'.format(args.columns[i + 1]),
                y_axis=[round(f, 1) for f in true[:, i].flatten().tolist()],
                is_smooth=True,
                label_opts=opts.LabelOpts(is_show=args.is_show_label),
                markline_opts=opts.MarkLineOpts(data=[opts.MarkLineItem(type_="average")]),
            )
    line.render(os.path.join(run_ex_dir, '{}_Test_{}_{}_evaluation.html'.format(i,args.model,args.data)))
    print("test evaluation chart action is completed!")
    return line

# 可视化预测未来和真实值的结果
def chart_predict_and_true(time_list,pred,true,run_ex_dir,args,i):
    # 查看真实值 与预测值的差距：为训练集和测试集准备的函数
    line = Line(init_opts=opts.InitOpts(theme=ThemeType.SHINE))
    line.set_global_opts(
        tooltip_opts=opts.TooltipOpts(is_show=True),
        xaxis_opts=opts.AxisOpts(type_="category",name="time"),
        # xaxis_opts=opts.AxisOpts(type_="category",name="时间",axislabel_opts=opts.LabelOpts(is_show=True,rotate=10)),
        yaxis_opts=opts.AxisOpts(
            type_="value",
            name="{}".format(args.target),
            min_='dataMin',
            axistick_opts=opts.AxisTickOpts(is_show=True),
            splitline_opts=opts.SplitLineOpts(is_show=True),
        ),
        datazoom_opts=opts.DataZoomOpts(is_show=True),
        # datazoom_opts=opts.DataZoomOpts(is_show=True,type_="inside"),
        title_opts=opts.TitleOpts(title='实验{}'.format(i))
    )
    line.add_xaxis(xaxis_data=time_list)
    if args.features != 'M':
        line.add_yaxis(
            series_name="Predict {}".format(args.target),
            y_axis=pred,
            is_smooth=True,
            label_opts=opts.LabelOpts(is_show=args.is_show_label),
            markline_opts=opts.MarkLineOpts(data=[opts.MarkLineItem(type_="average")]),
        )
        line.add_yaxis(
            series_name="feature True {}".format(args.target),
            y_axis=true,
            is_smooth=True,
            label_opts=opts.LabelOpts(is_show=args.is_show_label),
            markline_opts=opts.MarkLineOpts(data=[opts.MarkLineItem(type_="average")]),
        )
    if args.features == 'M':
        for i in range(args.c_out):
            line.add_yaxis(
                series_name='pred_{}'.format(args.columns[i + 1]),
                y_axis=[round(f,1) for f in pred[:,i].flatten().tolist()],
                is_smooth=True,
                label_opts=opts.LabelOpts(is_show=args.is_show_label),
                markline_opts=opts.MarkLineOpts(data=[opts.MarkLineItem(type_="average")]),
            )
            line.add_yaxis(
                series_name='true_{}'.format(args.columns[i + 1]),
                y_axis=true[i],
                is_smooth=True,
                label_opts=opts.LabelOpts(is_show=args.is_show_label),
                markline_opts=opts.MarkLineOpts(data=[opts.MarkLineItem(type_="average")]),
            )
    line.render(os.path.join(run_ex_dir, '{}_True_Pred_{}_{}_feature.html'.format(i,args.model, args.data)))
    print("True and Pred predict chart action is completed!")
    return line

# 可视化预测未来的结果
def chart_predict(time_list,pred,run_ex_dir,args,i):
    # 查看真实值 与预测值的差距：为训练集和测试集准备的函数
    line = Line(init_opts=opts.InitOpts(theme=ThemeType.SHINE))
    line.set_global_opts(
        tooltip_opts=opts.TooltipOpts(is_show=True),
        xaxis_opts=opts.AxisOpts(type_="category",name="time"),
        # xaxis_opts=opts.AxisOpts(type_="category",name="时间",axislabel_opts=opts.LabelOpts(is_show=True,rotate=10)),
        yaxis_opts=opts.AxisOpts(
            type_="value",
            name="{}".format(args.target),
            min_='dataMin',
            axistick_opts=opts.AxisTickOpts(is_show=True),
            splitline_opts=opts.SplitLineOpts(is_show=True),
        ),
        datazoom_opts=opts.DataZoomOpts(is_show=True),
        # datazoom_opts=opts.DataZoomOpts(is_show=True,type_="inside"),
        title_opts=opts.TitleOpts(title='{}预测未来情况'.format(i))
    )
    line.add_xaxis(xaxis_data=time_list)
    if args.features != 'M':
        line.add_yaxis(
            series_name="Predict {}".format(args.target),
            y_axis=pred,
            is_smooth=True,
            label_opts=opts.LabelOpts(is_show=args.is_show_label),
            markline_opts=opts.MarkLineOpts(data=[opts.MarkLineItem(type_="average")]),
        )
    if args.features == 'M':
        for i in range(args.c_out):
            line.add_yaxis(
                series_name='pred_{}'.format(args.columns[i + 1]),
                y_axis=[round(f,1) for f in pred[:,i].flatten().tolist()],
                is_smooth=True,
                label_opts=opts.LabelOpts(is_show=args.is_show_label),
                markline_opts=opts.MarkLineOpts(data=[opts.MarkLineItem(type_="average")]),
            )
    line.render(os.path.join(run_ex_dir, '{}_Prediction_{}_{}_feature.html'.format(i,args.model, args.data)))
    print("predict chart action is completed!")
    return line

# 可视化预测未来和真实值的结果
def chart_avg_predict_and_true(df,run_name_dir,args):
    # 查看真实值 与预测值的差距：为训练集和测试集准备的函数
    line = Line(init_opts=opts.InitOpts(theme=ThemeType.SHINE))
    line.set_global_opts(
        tooltip_opts=opts.TooltipOpts(is_show=True),
        xaxis_opts=opts.AxisOpts(type_="category",name="time"),
        # xaxis_opts=opts.AxisOpts(type_="category",name="时间",axislabel_opts=opts.LabelOpts(is_show=True,rotate=10)),
        yaxis_opts=opts.AxisOpts(
            type_="value",
            name="{}".format(args.target),
            min_='dataMin',
            # min_='dataMin',
            axistick_opts=opts.AxisTickOpts(is_show=True),
            splitline_opts=opts.SplitLineOpts(is_show=True),
        ),
        datazoom_opts=opts.DataZoomOpts(is_show=True),
        # datazoom_opts=opts.DataZoomOpts(is_show=True,type_="inside"),
        title_opts=opts.TitleOpts(title='avg_predict_true')
    )
    #
    line.add_xaxis(xaxis_data=df['date'].tolist())
    if args.features != 'M':
        line.add_yaxis(
            series_name="Avg Predict {}".format(args.target),
            y_axis=df['pred_mean_{}'.format(args.target)].tolist(),
            is_smooth=True,
            label_opts=opts.LabelOpts(is_show=args.is_show_label),
            markline_opts=opts.MarkLineOpts(data=[opts.MarkLineItem(type_="average")]),
        )
        line.add_yaxis(
            series_name="feature True {}".format(args.target),
            y_axis=df['true'].tolist(),
            is_smooth=True,
            label_opts=opts.LabelOpts(is_show=args.is_show_label),
            markline_opts=opts.MarkLineOpts(data=[opts.MarkLineItem(type_="average")]),
        )
    if args.features == 'M':
        # 选出含有avg字段的列
        tmp_pred = df[[s for s in df.columns if s.startswith("pred_mean_")]]
        tmp_true = df[[s for s in df.columns if s.startswith('true_')]]
        # print(tmp_pred.columns)
        # print(tmp_true.columns)
        for i in range(args.c_out):
            line.add_yaxis(
                series_name='pred_{}'.format(args.columns[1:][i - 1]),
                y_axis=tmp_pred.iloc[:,i].tolist(),
                is_smooth=True,
                label_opts=opts.LabelOpts(is_show=args.is_show_label),
                markline_opts=opts.MarkLineOpts(data=[opts.MarkLineItem(type_="average")]),
            )
            line.add_yaxis(
                series_name='true_{}'.format(args.columns[1:][i - 1]),
                y_axis=tmp_true.iloc[:,i].tolist(),
                is_smooth=True,
                label_opts=opts.LabelOpts(is_show=args.is_show_label),
                markline_opts=opts.MarkLineOpts(data=[opts.MarkLineItem(type_="average")]),
            )
    line.render(os.path.join(run_name_dir, 'Avg_Pred_True_{}_{}_feature.html'.format(args.model, args.data)))
    print("AVG True and Pred predict chart action is completed!")
    return line

def get_page_loss(n=20):
    # 损失组图
    page = Page(layout=Page.SimplePageLayout,page_title="{}次实验-损失值可视化".format(n))
    return page

def get_page_value(n=20):
    # 未来预测值的 组图，有真实值
    page = Page(layout=Page.SimplePageLayout,page_title="{}次实验-Pred-True".format(n))
    return page

def get_page_noTrue(n=20):
    # 未来预测值的组图，没有真实值
    page = Page(layout=Page.SimplePageLayout,page_title="{}次实验-Pred".format(n))
    return page

def get_page_Test(n=20):
    # 测试集的组图
    page = Page(layout=Page.SimplePageLayout,page_title="{}次实验-Test".format(n))
    return page

def get_page_Train(n=20):
    # 训练集预测值的组图: 暂时未实现
    page = Page(layout=Page.SimplePageLayout,page_title="{}次实验-Pred".format(n))
    return page

def grid_line():
    grid = Grid()


