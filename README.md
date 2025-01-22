# DAF-C
论文名字为Enhanced Edge Detection via Dual-Branch Attention Fusion with Canny-Assisted Supervision，目前投稿在The visual computer,下面是我们代码的一些细节

## 目录
- [开发前的配置要求](#开发前的配置要求)
- [安装步骤](#安装步骤)
- [部署](#部署)
- [数据集](#数据集)
- [训练与推理](#执行)
- [作者](#作者)
- [引用](#引用)




### 开发前的配置要求

1. python 3.10
2. pytorch cuda12.1版本

### **安装步骤**

1. Clone the repo

```sh
git clone https://github.com/sfg-oss/daf-c.git
```


### 部署

linux 3090



### 数据集

BSDSdataset：https://vcl.ucsd.edu/hed/HED-BSDS.tar![image](https://github.com/user-attachments/assets/0cd1ef50-5958-44c4-a4bd-ae38ea7f28a9)
PascalVoc dataset:https://pan.baidu.com/s/1d9CTR9w1MTcVrBvG-WIIXw?pwd=83cv![image](https://github.com/user-attachments/assets/67a49194-c007-4cf1-8140-b00b97038b46)


### 执行
1.train 

```sh
python tarin.py
```
2.test 在test.py中将输入输出放入正确的位置
```sh
python test.py
```



### 作者

wht，有问题可通过邮箱联系，后续代码会持续更新ing

 *您也可以在贡献者名单中参看所有参与该项目的开发者。*

## Citation
If you find this work useful, please cite our paper:

**Enhanced Edge Detection via Dual-Branch Attention Fusion with Canny-Assisted Supervision**  
*The Visual Computer*







