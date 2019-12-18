## Intro to Decision Tree

### background

[Decision Tree](https://en.wikipedia.org/wiki/Decision_tree)可以说是非常早期的机器学习的方法了，到现在仍然有其使用的价值。设想你观察了你的暗恋对象的日常生活习惯，类似，在一个阳关明媚的周末，她会去逛商场，在雨天有时候选择宅在家里看剧，有时候去国图待上一整天，不知不觉你收集了她十年的数据，然后某个阳光明媚的下午，你发现她带着孩子在逛街......言归正传，看一下Decision Tree怎么处理这个分类问题。


### Decision Tree Representation

以下是一棵Decision Tree

<p align="center">
<img width=500 src="https://github.com/thelostpeace/thelosepeace/blob/master/image/decision_tree.png?raw=true">
</p>

其中方形里面的`Outlook`、`Humidity`、`Wind`表示属性，也就是特征，边的值`Sunny`、`High`、`Normal`表示特征值，叶节点`Yes`、`No`表示分类标签，当然，可以有很多种分类。所以Decision Tree的问题就是当前节点怎么做特征选取的问题。


### Which Attribute is the Best Classifier

information theory里的熵在机器学习里的很多算loss的场景里都会用到，大道至简嘛，怎么用最小的信息量对信息做表示。

<p align="center">
<img height=30 src="https://github.com/thelostpeace/thelosepeace/blob/master/image/decision_tree_entropy.png?raw=true">
</p>

其中`P+`表示正例，`P-`表示负例。

<p align="center">
<img height=60 src="https://github.com/thelostpeace/thelosepeace/blob/master/image/decision_tree_entropy1.png?raw=true">
</p>

对于多个分类的entropy计算：

<p align="center">
<img height=60 src="https://github.com/thelostpeace/thelosepeace/blob/master/image/decision_tree_entropy2.png?raw=true">
</p>

最原始的ID3算法如下，其实后续的C4.5只是对ID3的一些改进，以避免overfitting的问题。

<p align="center">
<img width=540 src="https://github.com/thelostpeace/thelosepeace/blob/master/image/ID3.png?raw=true">
</p>

每一步会对当前的所有特征做information gain计算，每次选取最大的information gain，这样对于ID3算法来说，是无法做backtracking的，也就是hillclimbing得到的可能只是一个局部最优解。

<p align="center">
<img height=60 src="https://github.com/thelostpeace/thelosepeace/blob/master/image/information_gain.png?raw=true">
</p>

<p align="center">
<img width=480 src="https://github.com/thelostpeace/thelosepeace/blob/master/image/information_gain2.png?raw=true">
</p>

<p align="center">
<img width=570 src="https://github.com/thelostpeace/thelosepeace/blob/master/image/information_gain3.png?raw=true">
</p>

对于以上的例子来说，`Humidity`算得的information gain比`Wind`要高，所以当前节点会选择`Humidity`。

### How to Represent Bias

对于Decision Tree或者其他机器学习算法来说，有一个统一的规则，`more simple more general`，也就是大道至简。1320年，哲学上有人提出了一个观点叫[Occams's Razor](https://en.wikipedia.org/wiki/Occam%27s_razor)。

<p align="center">
<img height=30 src="https://github.com/thelostpeace/thelosepeace/blob/master/image/Occam's_Razor.png?raw=true">
</p>

对于Decision Tree来说，矮树优于高树。

<p align="center">
<img = width=570 src="https://github.com/thelostpeace/thelosepeace/blob/master/image/decision_tree_bias.png?raw=true">
</p>


### How to Avoid Overfitting

对于overfitting的定义如下，在tensorboard里看初版模型训练记录，可以有直观的感受。

<p align="center">
<img width=570 src="https://github.com/thelostpeace/thelosepeace/blob/master/image/overfitting.png?raw=true">
</p>

如果数据没有噪声，ID3得到的数是可以完全fitting训练集的，怎么防止overfitting呢？

 - 在ID3算法走完之前就停止训练，什么时候停止呢？没有一个明确的标准，有时候可以指定树高，有时候可以指定information gain小于一个阈值的时候停止。
 - 对树做剪枝操作，用validation set去验证剪枝后的准确率。

对于后者来说，使用的较多，对于C4.5来说，最后并没有树的概念了，而是对于所有的路径生成规则，`if (A && B && C && ...)`这种，然后对规则做剪枝，然后在validation set上做验证。

### How to Handle Continuous Attribute Values

对于温度或者一些其他连续值来说，怎么确定特征值分界？确定了特征值分界就相当于变为了离散的特征值。

<p align="center">
<img width=540 src="https://github.com/thelostpeace/thelosepeace/blob/master/image/decision_tree4.png?raw=true">
</p>

去分类变化的分界点的均值作为临界点，即`(48 + 60) / 2 = 54`和`(80 + 90) / 2 = 85`做为临界点，但是对于日期来说，每天都是一个明确的分类，这样information gain很高，但是这样会引起严重的overfitting的问题，所以后面提出了另一个思想。

<p align="center">
<img height=75 src="https://github.com/thelostpeace/thelosepeace/blob/master/image/split_information.png?raw=true">
</p>

<p align="center">
<img height=75 src="https://github.com/thelostpeace/thelosepeace/blob/master/image/gain_ratio.png?raw=true">
</p>

其中`Si`表示分类`i`的数量，`S`表示总样本数，则分类越细，完全分开的话`SplitInformation`的值变为`log2(N)`，其中N为样本总量。所以说`SplitInformation`算是对`InformationGain`的一个调整。

还有一起其他的计算方式，但是思想不变。

<p align="center">
<img height=75 src="https://github.com/thelostpeace/thelosepeace/blob/master/image/gain_ratio1.png?raw=true">
</p>

<p align="center">
<img height=75 src="https://github.com/thelostpeace/thelosepeace/blob/master/image/gain_ratio2.png?raw=true">
</p>

### Tree Hole

Decision Tree算是非常老的机器学习方法了，但是其优势很明显，就是对数据的直观解析性很强。拿着PPT给老板做演示的时候，有理有据，效果挺好。