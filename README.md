## Bert sentiment analysis

### 简介

这是2022秋季学期，SJTU互联网信息抽取技术课程的课程设计

本小组针对中文和英文的数据分别训练了两个模型，为方便测试，本文件对测试方式进行了详细说明，详见“测试方式”



### 目录结构说明

+ model/：模型储存目录
  + en/：储存了英文对应模型的参数和config文件等
  + cn/：储存了中文对应模型的参数和config文件等
+ test/：测试脚本目录
  + test_en.py：英文模型的测试脚本
  + test_cn.py：中文模型的测试脚本
+ train/：训练脚本目录
  + train_en.py：英文模型的训练脚本
  + train_cn.py：中文模型的训练脚本
+ bert.yaml：conda环境配置文件
+ README.md：README文档



### 测试方式

在测试前，请确保所需依赖项都已安装完毕，具体可见环境配置文件`bert.yaml`，也可以在已安装anaconda的情况下，使用如下命令创建一个新的可用环境：

~~~shell
conda env create -f ./bert.yaml
~~~

由于微调模型较大，我们没有放在仓库中，在测试前请从[Google Drive](https://drive.google.com/drive/folders/1hSxFnXghs2yMUf2qqpx6vYYdSZ7oqLTZ)下载`model/`文件夹并放在仓库根目录中

此外，两个测试文件都实现了GPU加速，若有可用GPU则会优先使用GPU进行运算，否则使用CPU。需要注意，该模型占用显存较大，测试过程大概需要占用2.5G显存。





#### 英文测试

进入`test/`目录，使用如下命令进行测试：

~~~shell
python ./test_en.py -i <input_file_path> -o <output_file_path>
~~~

`test_en.py `文件接收三个参数：

+ `'-m','--model_path'`：待测试的模型路径，默认为`"../model/en/model.pth"`，故处于`test/`目录中时，未改变目录结构的情况下无需额外指定
+ `'-i','--input'`：输入文件保存路径
+ `'-o','--output'`：输出文件保存路径

模型加载可能花费30s左右的时间

#### 中文测试

同英文测试，具体如下

进入`test/`目录，使用如下命令进行测试：

~~~shell
python ./test_cn.py -i <input_file_path> -o <output_file_path>
~~~

`test_cn.py `文件接收三个参数：

+ `'-m','--model_path'`：待测试的模型路径，默认为`"../model/cn/model.pth"`，故未改变目录结构的情况下无需指定
+ `'-i','--input'`：输入文件保存路径
+ `'-o','--output'`：输出文件保存路径

模型加载可能花费30s左右的时间



### 补充

`train/`目录中的两个文件是训练所用文件，在此一并附上



### 联系我们

mail：郑航 azure123@sjtu.edu.cn，zhenghang707@gmail.com

