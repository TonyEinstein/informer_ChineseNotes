# informer_ChineseNotes

![Python 3.7](https://img.shields.io/badge/python-3.7-green.svg?style=plastic)
![PyTorch LTS](https://img.shields.io/badge/PyTorch-lts%20-%23EE4C2C.svg?style=plastic)
![cudatoolkit 11.1](https://img.shields.io/badge/cudatoolkit-11.1-green.svg?style=plastic)

[informer_ChineseNotes](https://github.com/TonyEinstein/informer_ChineseNotes) has added Chinese annotations and some visual additions to the [informer2020](https://github.com/zhouhaoyi/Informer2020) project.


## 使用场景
* 时间序列预测
* 股票预测
* 市场预测
* 电力预测
* 天气预测
* ....

## Requirements
- torch_lts
- pyecharts
- Python 3.7
- matplotlib
- numpy == 1.19.4
- pandas
- scikit_learn
- xlrd
- ....

You can install it directly from the environment pack version file:
```
pip install -r requirements.txt
or
conda env create -f torch_lts.yaml
```

## Quick Configuration Environment

### Method 1：
Create a virtual environment in Anaconda Powershell Prompt (Anaconda3) :
```
conda create -n torch_lts python=3.7
```
Then enter the virtual environment:
```
activate torch_lts
```
Finally, install the environment dependency package using the requirements. TXT file:
```
pip install -r requirements.txt

```

### Method 2
Create virtual environments and install dependencies using conda and YAMl files directly from Anaconda Powershell Prompt (Anaconda3) :
```
conda env create -f torch_lts.yaml
```

### Method 3
Execute the following commands directly in your environment to install:
```
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch-lts -c conda-forge

pip install numpy == 1.19.4
pip install pyecharts
pip install xlrd
pip install openpyxl
pip install matplotlib
pip install pandas

```



### Import virtual environment in PyCharm:
When selecting the interpreter, do not create a new virtual environment and use the virtual environment you already created:
![image](https://user-images.githubusercontent.com/47185449/176341835-506057a2-479b-414b-a88b-45ff0f1650db.png)


## FAQ
If you run into a problem like `RuntimeError: The size of tensor a (98) must match the size of tensor b (96) at non-singleton dimension 1`, you can check torch version or modify code about `Conv1d` of `TokenEmbedding` in `models/embed.py` as the way of circular padding mode in Conv1d changed in different torch versions.

## Interesting and useful tool recommendations
* [streamlit](https://awesome-streamlit.org/)


## Thank you
* Thanks to [@zhouhaoyi](https://github.com/zhouhaoyi) open source algorithm paper and code repository.


## reference
* Original repository: [informer2020](https://github.com/zhouhaoyi/Informer2020)
* Zhouhaoyi paper: [Informer：Beyond Efficient Transformer for Long Sequence Time-Series Forecasting (AAAI'21 Best Paper)](https://arxiv.org/abs/2012.07436)

## Tips
数据是网上找来的，已做脱敏处理，如有冒犯到您的请联系e-mail或者从[Tony's blog](https://blog.csdn.net/qq_42658739?type=blog)中私聊我侵删。

## To contact me
* e-mail 1：crh19981229@gmail.com
* e-mail 2：ruhai.chen@foxmail.com 【优先建议】

## 关于报错 main_**运行后报错KeyError: "['true'] not in index" 的解决方法

详细看这里：
https://github.com/TonyEinstein/informer_ChineseNotes/issues/1
https://github.com/TonyEinstein/informer_ChineseNotes/issues/4



