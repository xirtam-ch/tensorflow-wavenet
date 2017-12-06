# DeepMind WaveNet论文的TensorFlow实现

[![Build Status](https://travis-ci.org/ibab/tensorflow-wavenet.svg?branch=master)](https://travis-ci.org/ibab/tensorflow-wavenet)

这是用于音频生成的 [WaveNet生成神经网络](https://deepmind.com/blog/wavenet-generative-model-raw-audio/) 的TensorFlow实现。

<table style="border-collapse: collapse">
<tr>
<td>
<p>
WaveNet神经网络可以直接生成原始音频波形，在文本转换成语音和一般音频生成方面表现出色（详细信息见DeepMind的博客文章）。
</p>
<p>
给定初始的样本和附加参数，网络对条件概率进行建模以生成音频波形中的下一样本。
</p>
<p>
在音频预处理步骤之后，输入的波形被量化为一个优化过的整数范围，然后通过一次热编码以产生形状的张量 <code>(num_samples, num_channels)</code> 。
</p>
<p>
仅访问当前和之前输入的卷积层然后减小信道维度。
</p>
<p>
网络的核心是由一系列因果扩张层构成的 ，每层都是一个扩大的卷积（与孔的卷积），它只访问当前和过去的音频样本。</p>
<p>
所有层的输出结合起来，并通过一系列密集的后处理层扩展回原来的通道数，然后用softmax函数将输出转换成分类分布。</p>
<p>
损失函数是每个时步的输出和下一个时步的输入之间的交叉熵。
</p>
<p>
在这个项目中，网络实现可以在 <a href="./wavenet/model.py">model.py</a>中找到。
</p>
</td>
<td width="300">
<img src="images/network.png" width="300"></img>
</td>
</tr>
</table>

## 需要的环境

在运行训练脚本之前需要安装TensorFlow。 代码在Python 2.7和Python 3.5的TensorFlow版本1.0.1上进行测试。

另外，必须安装[librosa](https://github.com/librosa/librosa) 才能读取和写入音频。

要安装所需的Python包，运行
```bash
pip install -r requirements.txt
```

需要GPU支持，请使用
```bash
pip install -r requirements_gpu.txt
```

## 训练网络

您可以使用任何包含 `.wav` 文件的音频样本库。 
到目前为止，我们主要使用 [VCTK语音样本库](http://homepages.inf.ed.ac.uk/jyamagis/page3/page58/page58.html) (大约10.4GB，  [备用地址](http://www.udialogue.org/download/cstr-vctk-corpus.html)) so far.

训练网络，执行
```bash
python train.py --data_dir=corpus
```
训练网络，其中 `corpus` 是一个包含 `.wav` 文件的目录。
该脚本将递归地收集目录中的所有 `.wav` 文件。

你可以通过运行查看每个培训设置的文档
```bash
python train.py --help
```

你可以在 [`wavenet_params.json`](./wavenet_params.json)中找到模型参数的配置。
培训和生成之间需要保持一致。

### 全局条件
全局条件修改是指修改模型，使得在训练和生成.wav文件时指定一组相互排斥的类别的ID。 在VCTK的情况下，这个id是说话者的整数ID，有一百个以上。 这允许（确实需要）在生成时指定讲话者ID以选择它应该模仿哪个讲话者。 欲了解更多信息，请参阅论文或源代码。

### 全局条件训练
以上关于训练的说明是指没有全局条件的培训。 要使用全局条件训练，请指定命令行参数，如下所示：
```
python train.py --data_dir=corpus --gc_channels=32
```
--gc_channels参数做了两件事：
* 它告诉train.py脚本，它应该建立一个包含全局条件的模型。
* 它指定根据说话者的id查找的嵌入向量的大小。

train.py和audio_reader.py中的全局条件逻辑此时与VCTK语料库“硬关联”，因为它希望能够从VCTK中使用的文件命名模式中确定演讲者ID，但是可以很容易被修改。

## 生成音频

基于VCTK语料库中扬声器280的@jyegerlehner生成的[输出示例](https://soundcloud.com/user-731806733/tensorflow-wavenet-500-msec-88k-train-steps)

你可以使用之前训练过的模型使用 `generate.py` 脚本生成音频。

### 不使用全局条件生成
运行
```
python generate.py --samples 16000 logdir/train/2017-02-13T16-45-34/model.ckpt-80000
```
其中 `logdir/train/2017-02-13T16-45-34/model.ckpt-80000` 需要成为先前保存模型的路径（没有扩展名）。
`--samples` 参数指定要生成多少个音频采样（默认情况下16000对应于1秒）。

生成的波形可以使用TensorBoard播放，或通过使用`--wav_out_path`参数存储为`.wav` 文件：
```
python generate.py --wav_out_path=generated.wav --samples 16000 logdir/train/2017-02-13T16-45-34/model.ckpt-80000
```

除了 `--wav_out_path`  之外，使用参数 `--save_every`可以将每n个样本保存正在进行的wav文件。
```
python generate.py --wav_out_path=generated.wav --save_every 2000 --samples 16000 logdir/train/2017-02-13T16-45-34/model.ckpt-80000
```

快速生成是默认启用的。
它使用[Fast Wavenet](https://github.com/tomlepaine/fast-wavenet) 库中的实现。
您可以按照链接了解它是如何工作的。 这样可以将样品生成时间缩短到几分钟。

禁用快速生成：
```
python generate.py --samples 16000 logdir/train/2017-02-13T16-45-34/model.ckpt-80000 --fast_generation=false
```

### 使用全局条件生成
从包含全局条件的模型生成如下：
```
python generate.py --samples 16000  --wav_out_path speaker311.wav --gc_channels=32 --gc_cardinality=377 --gc_id=311 logdir/train/2017-02-13T16-45-34/model.ckpt-80000
```
其中：

`--gc_channels=32` 指定32是嵌入向量的大小，并且必须与训练时指定的值相匹配。

`--gc_cardinality=377` 是必需的，因为376是VCTK语料库中发言者的最大ID。 如果使用其他语料库，那么这个数字应该与训练时间由train.py脚本自动确定并打印出来的数字相匹配。

`--gc_id=311` 指定要为其生成样本的说话者的说话者311的id。

## 运行测试

安装测试环境
```
pip install -r requirements_test.txt
```

运行测试
```
./ci/test.sh
```

##  缺少功能

目前没有额外的信息的本地条件，这将允许上下文堆栈或控制什么语音产生。


## 相关项目

- [tex-wavenet](https://github.com/Zeta36/tensorflow-tex-wavenet)，用于文本生成的WaveNet。
- [image-wavenet](https://github.com/Zeta36/tensorflow-image-wavenet)，用于图像生成的WaveNet。
