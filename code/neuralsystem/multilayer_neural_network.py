import numpy as np
import matplotlib.pyplot as plt
import neurolab as nl

min_val = -15
max_val = 15
num_points = 130
x = np.linspace(min_val, max_val, num_points)
y = 3 * np.square(x) + 5
y /= np.linalg.norm(y)

data = x.reshape(num_points, 1)
labels = y.reshape(num_points, 1)

plt.figure()
plt.scatter(data, labels)
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.title('Input data')

nn = nl.net.newff([[min_val, max_val]], [10, 6, 1])

nn.trainf = nl.train.train_gd

error_progress = nn.train(data, labels, epochs=2000, show=100, goal=0.01)

output = nn.sim(data)
y_pred = output.reshape(num_points)

plt.figure()
plt.plot(error_progress)
plt.xlabel('Number of epochs')
plt.ylabel('Error')
plt.title('Training error progress')

x_dense = np.linspace(min_val, max_val, num_points * 2)
y_dense_pred = nn.sim(x_dense.reshape(x_dense.size,1)).reshape(x_dense.size)

plt.figure()
plt.plot(x_dense, y_dense_pred, '-', x, y, '.', x, y_pred, 'p')
plt.title('Actual vs predicted')

plt.show()

# coding: utf-8
import os
import sys


def format_file(filename, str1, str2):
    """
    :return:
    """
    with open(filename, 'r') as f:
        var_object = f.read()
        if "gitalk" not in var_object:
            var_object = var_object.replace(str1, str2)
        # print(var_object)

    f = open(filename, "w")
    f.write(var_object)


if __name__ == "__main__":
    if len(sys.argv) == 3:
        version, u_type = sys.argv[1], sys.argv[2]
    else:
        print("Usage: Python3" % len(sys.argv))
        sys.exit(-1)

    tag = True
    if u_type == "index":
        tag = False
        # if version == "home":
        #     filename = "_book/index.html"
        # else:
        #     filename = "_book/docs/%s/index.html" % version
        # str1 = """
        # </head>
        # <body>
        # """

        # str2 = """
        # <script type="text/javascript">
        #     function hidden_left(){
        #         document.getElementsByClassName("btn pull-left js-toolbar-action")[0].click()
        #     }
        #     // window.onload = hidden_left();
        # </script>
        # </head>
        # <body onload="hidden_left()">
        # """
    elif u_type == "book":
        if version == "home":
            filename = "book.json"
            tag = False
        else:
            filename = "docs/%s/book.json" % version
            str1 = "https://github.com/apachecn/AiLearning/blob/master"
            str2 = "https://github.com/apachecn/AiLearning/blob/master/docs/%s" % version

    elif u_type == "powered":
        if version == "home":
            filename = "node_modules/gitbook-plugin-tbfed-pagefooter/index.js"
        else:
            filename = "docs/%s/node_modules/gitbook-plugin-tbfed-pagefooter/index.js" % version
        str1 = "powered by Gitbook"
        str2 = "ApacheCN"

    elif u_type == "gitalk":
        if version == "home":
            filename = "node_modules/gitbook-plugin-tbfed-pagefooter/index.js"
        else:
            filename = "docs/%s/node_modules/gitbook-plugin-tbfed-pagefooter/index.js" % version
        str1 = """      var str = ' \\n\\n<footer class="page-footer">' + _copy +
        '<span class="footer-modification">' +
        _label +
        '\\n{{file.mtime | date("' + _format +
        '")}}\\n</span></footer>'"""

        str2 = """
      var str = '\\n\\n'+
      '\\n<hr/>'+
      '\\n<div align="center">'+
      '\\n    <p><a href="http://www.apachecn.org" target="_blank"><font face="KaiTi" size="6" color="red">我们一直在努力</font></a></p>'+
      '\\n    <p><a href="https://github.com/apachecn/AiLearning/" target="_blank">apachecn/AiLearning</a></p>'+
      '\\n    <p><iframe align="middle" src="https://ghbtns.com/github-btn.html?user=apachecn&repo=AiLearning&type=watch&count=true&v=2" 
      '\\n    <frameborder="0" scrolling="0" width="100px" height="25px"></iframe>'+
      '\\n    <iframe align="middle" src="https://ghbtns.com/github-btn.html?user=apachecn&repo=AiLearning&type=star&count=true" 
      '\\n    <frameborder="0" scrolling="0" width="100px" height="25px"></iframe>'+
      '\\n    <iframe align="middle" src="https://ghbtns.com/github-btn.html?user=apachecn&repo=AiLearning&type=fork&count=true" 
      '\\n    <frameborder="0" scrolling="0" width="100px" height="25px"></iframe>'+
      '\\n    <a target="_blank" href="//shang.qq.com/wpa/qunwpa?idkey=bcee938030cc9e1552deb3bd9617bbbf62d3ec1647e4b60d9cd6b6e8f78ddc03"><img border="0" 
      '\\n    <src="http://data.apachecn.org/img/logo/ApacheCN-group.png" alt="ML | ApacheCN" title="ML | ApacheCN"></a></p>'+
      '\\n</div>'+
      '\\n <div style="text-align:center;margin:0 0 10.5px;">'+
      '\\n     <script async src="//pagead2.googlesyndication.com/pagead/js/adsbygoogle.js"></script>'+
      '\\n     <ins class="adsbygoogle"'+
      '\\n         style="display:inline-block;width:728px;height:90px"'+
      '\\n         data-ad-client="ca-pub-3565452474788507"'+
      '\\n         data-ad-slot="2543897000">'+
      '\\n     </ins>'+
      '\\n     <script>(adsbygoogle = window.adsbygoogle || []).push({});</script>'+
      '\\n'+
      '\\n    <script>'+
      '\\n      var _hmt = _hmt || [];'+
      '\\n      (function() {'+
      '\\n        var hm = document.createElement("script");'+
      '\\n        hm.src = "https://hm.baidu.com/hm.js?33149aa6702022b1203201da06e23d81";'+
      '\\n        var s = document.getElementsByTagName("script")[0]; '+
      '\\n        s.parentNode.insertBefore(hm, s);'+
      '\\n      })();'+
      '\\n    </script>'+
      '\\n'+
      '\\n    <script async src="https://www.googletagmanager.com/gtag/js?id=UA-127082511-1"></script>'+
      '\\n    <script>'+
      '\\n      window.dataLayer = window.dataLayer || [];'+
      '\\n      function gtag(){dataLayer.push(arguments);}'+
      '\\n      gtag(\\'js\\', new Date());'+
      '\\n'+
      '\\n      gtag(\\'config\\', \\'UA-127082511-1\\');'+
      '\\n    </script>'+
     '\\n</div>'+
      '\\n'+
      '\\n<meta name="google-site-verification" content="pyo9N70ZWyh8JB43bIu633mhxesJ1IcwWCZlM3jUfFo" />'+
      '\\n<iframe src="https://www.bilibili.com/read/cv2710377" style="display:none"></iframe>'+ 
      '\\n<img src="http://t.cn/AiCoDHwb" hidden="hidden" />'
      str += '\\n\\n'+
      '\\n<div>'+
      '\\n    <link rel="stylesheet" href="https://unpkg.com/gitalk/dist/gitalk.css">'+
      '\\n    <script src="https://unpkg.com/gitalk/dist/gitalk.min.js"></script>'+
      '\\n    <script src="https://cdn.bootcss.com/blueimp-md5/2.10.0/js/md5.min.js"></script>'+
      '\\n    <div id="gitalk-container"></div>'+
      '\\n    <script type="text/javascript">'+
      '\\n        const gitalk = new Gitalk({'+
      '\\n        clientID: \\'2e62dee5b9896e2eede6\\','+
      '\\n        clientSecret: \\'ca6819a54656af0d87960af15315320f8a628a53\\','+
      '\\n        repo: \\'AiLearning\\','+
      '\\n        owner: \\'apachecn\\','+
      '\\n        admin: [\\'jiangzhonglian\\', \\'wizardforcel\\'],'+
      '\\n        id: md5(location.pathname),'+
      '\\n        distractionFreeMode: false'+
      '\\n        })'+
      '\\n        gitalk.render(\\'gitalk-container\\')'+
      '\\n    </script>'+
      '\\n</div>'
      str += '\\n\\n<footer class="page-footer">' + _copy + '<span class="footer-modification">' 
      + _label + '\\n{{file.mtime | date("' + _format + '")}}\\n</span></footer>'
        """

    if tag: format_file(filename, str1, str2)

        #!/usr/bin/env python
# -*- coding: UTF-8 -*-


import numpy as np


class ReluActivator(object):
    def forward(self, weighted_input):
        #return weighted_input
        return max(0, weighted_input)

    def backward(self, output):
        return 1 if output > 0 else 0


class IdentityActivator(object):
    def forward(self, weighted_input):
        return weighted_input

    def backward(self, output):
        return 1


class SigmoidActivator(object):
    def forward(self, weighted_input):
        return np.longfloat(1.0 / (1.0 + np.exp(-weighted_input)))

    def backward(self, output):
        return output * (1 - output)


class TanhActivator(object):
    def forward(self, weighted_input):
        return 2.0 / (1.0 + np.exp(-2 * weighted_input)) - 1.0

    def backward(self, output):
        return 1 - output * output
    
#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import random
from functools import reduce
from numpy import *

# sigmoid
def sigmoid(inX):
    '''
    Desc:
        sigmoid
    Args:
        inX --- 
    Returns:
        sigmoid
    '''
    return 1.0 / (1 + exp(-inX))


class Node(object):
    '''
    Desc:
        unk
    '''
    def __init__(self, layer_index, node_index):
        '''
        Desc:
            unk
        Args:
            layer_index --- 
            node_index --- 
        Returns:
            None
        '''
        self.layer_index = layer_index
        self.node_index = node_index
        self.downstream = []
        self.upstream = []
        self.output = 0
        self.delta = 0

    def set_output(self, output):
        '''
        Desc:
            output
        Args:
            output --- output
        Returns:
            None
        '''
        self.output = output

    def append_downstream_connection(self, conn):
        '''
        Desc:
           unk
        Args:
            conn --- list
        Returns:
            None
        '''
        # list append conn downstream
        self.downstream.append(conn)

    def append_upstream_connection(self, conn):
        '''
        Desc:
            unk
        Args:
            conn ---- list
        Returns:
            None
        '''
        # list append conn upstream
        self.upstream.append(conn)

    def calc_output(self):
        '''
        Desc:
            output = sigmoid(wTx)
        Args:
            None
        Returns:
            None
        '''
        # reduce()
        output = reduce(lambda ret, conn: ret + conn.upstream_node.output * conn.weight, self.upstream, 0)
        # output weights sigmoid output
        self.output = sigmoid(output)

    def calc_hidden_layer_delta(self):
        '''
        Desc:
            delta
        Args:
            output --- input
        Returns:
            None
        '''
        downstream_delta = reduce(lambda ret, conn: ret + conn.downstream_node.delta * conn.weight, self.downstream, 0.0)
        # delta
        self.delta = self.output * (1 - self.output) * downstream_delta

    def calc_output_layer_delta(self, label):
        '''
        Desc:
            delta
        Args:
            label ---
        Returns:
            None
        '''
        # delta
        self.delta = self.output * (1 - self.output) * (label - self.output)

    def __str__(self):
        '''
        Desc:
            unk
        Args:
            None
        Returns:
            None
        '''
        node_str = '%u-%u: output: %f delta: %f' % (self.layer_index, self.node_index, self.output, self.delta)
        downstream_str = reduce(lambda ret, conn: ret + '\n\t' + str(conn), self.downstream, '')
        upstream_str = reduce(lambda ret, conn: ret + '\n\t' + str(conn), self.upstream, '')
        return node_str + '\n\tdownstream:' + downstream_str + '\n\tupstream:' + upstream_str


# ConstNode
class ConstNode(object):
    '''
    Desc:
        unk
    '''
    def __init__(self, layer_index, node_index):
        '''
        Desc:
            unk
        Args:
            layer_index ---
            node_index ---
        Returns:
            None
        '''    
        self.layer_index = layer_index
        self.node_index = node_index
        self.downstream = []
        self.output = 1


    def append_downstream_connection(self, conn):
        '''
        Desc:
            unk
        Args:
            conn ---                                         
        Returns:
            None
        '''       
        self.downstream.append(conn)


    def calc_hidden_layer_delta(self):
        '''
        Desc:
            delta
        Args:
            None
        Returns:
            None
        '''
        downstream_delta = reduce(lambda ret, conn: ret + conn.downstream_node.delta * conn.weight, self.downstream, 0.0)
        self.delta = self.output * (1 - self.output) * downstream_delta


    def __str__(self):
        '''
        Desc:
           unk
        Args:
            None
        Returns:
            None
        '''
        node_str = '%u-%u: output: 1' % (self.layer_index, self.node_index)
        downstream_str = reduce(lambda ret, conn: ret + '\n\t' + str(conn), self.downstream, '')
        return node_str + '\n\tdownstream:' + downstream_str

class Layer(object):
    '''
    Desc:
        Layer
    '''

    def __init__(self, layer_index, node_count):
        '''
        Args:
            layer_index --- 
            node_count --- 
        Returns:
            None
        '''
        self.layer_index = layer_index
        self.nodes = []
        for i in range(node_count):
            self.nodes.append(Node(layer_index, i))
        # 将 ConstNode 节点也添加到 nodes 中
        self.nodes.append(ConstNode(layer_index, node_count))

    def set_output(self, data):
        '''
        Desc:
            设置层的输出，当层是输入层时会用到
        Args:
            data --- 输出的值的 list
        Returns:
            None
        '''
        # 设置输入层中各个节点的 output
        for i in range(len(data)):
            self.nodes[i].set_output(data[i])

    def calc_output(self):
        '''
        Desc:
            计算层的输出向量
        Args:
            None
        Returns:
            None
        '''
        # 遍历本层的所有节点（除去最后一个节点，因为它是恒为常数的偏置项b）
        # 调用节点的 calc_output 方法来计算输出向量
        for node in self.nodes[:-1]:
            node.calc_output()

    def dump(self):
        '''
        Desc:
            将层信息打印出来
        Args:
            None
        Returns:
            None
        '''
        # 遍历层的所有的节点 nodes，将节点信息打印出来
        for node in self.nodes:
            print(node)


# Connection 对象类，主要负责记录连接的权重，以及这个连接所关联的上下游的节点
class Connection(object):
    '''
    Desc:
        Connection 对象，记录连接权重和连接所关联的上下游节点，注意，这里的 connection 没有 s ，不是复数
    '''
    def __init__(self, upstream_node, downstream_node):
        '''
        Desc:
            初始化 Connection 对象
        Args:
            upstream_node --- 上游节点
            downstream_node --- 下游节点
        Returns:
            None
        '''
        # 设置上游节点
        self.upstream_node = upstream_node
        # 设置下游节点
        self.downstream_node = downstream_node
        # 设置权重，这里设置的权重是 -0.1 到 0.1 之间的任何数
        self.weight = random.uniform(-0.1, 0.1)
        # 设置梯度 为 0.0
        self.gradient = 0.0

    def calc_gradient(self):
        '''
        Desc:
            计算梯度
        Args:
            None
        Returns:
            None
        '''
        # 下游节点的 delta * 上游节点的 output 计算得到梯度
        self.gradient = self.downstream_node.delta * self.upstream_node.output

    def update_weight(self, rate):
        '''
        Desc:
            根据梯度下降算法更新权重
        Args:
            rate --- 学习率 / 或者成为步长
        Returns:
            None
        '''
        # 调用计算梯度的函数来将梯度计算出来
        self.calc_gradient()
        # 使用梯度下降算法来更新权重
        self.weight += rate * self.gradient

    def get_gradient(self):
        '''
        Desc:
            获取当前的梯度
        Args:
            None
        Returns:
            当前的梯度 gradient 
        '''
        return self.gradient

    def __str__(self):
        '''
        Desc:
            将连接信息打印出来
        Args:
            None
        Returns:
            连接信息进行返回
        '''
        # 格式为: 上游节点的层的索引+上游节点的节点索引 ---> 下游节点的层的索引+下游节点的节点索引，最后一个数是权重
        return '(%u-%u) -> (%u-%u) = %f' % (
            self.upstream_node.layer_index, 
            self.upstream_node.node_index,
            self.downstream_node.layer_index, 
            self.downstream_node.node_index, 
            self.weight)



# Connections 对象，提供 Connection 集合操作。
class Connections(object):
    '''
    Desc:
        Connections 对象，提供 Connection 集合的操作，看清楚后面有没有 s ，不要看错
    '''
    def __init__(self):
        '''
        Desc:
            初始化 Connections 对象
        Args:
            None
        Returns:
            None
        '''
        # 初始化一个列表 list 
        self.connections = []

    def add_connection(self, connection):
        '''
        Desc:
            将 connection 中的节点信息 append 到 connections 中
        Args:
            None
        Returns:
            None
        '''
        self.connections.append(connection)

    def dump(self):
        '''
        Desc:
            将 Connections 的节点信息打印出来
        Args:
            None
        Returns:
            None
        '''
        for conn in self.connections:
            print(conn)


# Network 对象，提供相应 API
class Network(object):
    '''
    Desc:
        Network 类
    '''
    def __init__(self, layers):
        '''
        Desc:
            初始化一个全连接神经网络
        Args:
            layers --- 二维数组，描述神经网络的每层节点数
        Returns:
            None
        '''
        # 初始化 connections，使用的是 Connections 对象
        self.connections = Connections()
        # 初始化 layers
        self.layers = []
        # 我们的神经网络的层数
        layer_count = len(layers)
        # 节点数
        node_count = 0
        # 遍历所有的层，将每层信息添加到 layers 中去
        for i in range(layer_count):
            self.layers.append(Layer(i, layers[i]))
        # 遍历除去输出层之外的所有层，将连接信息添加到 connections 对象中
        for layer in range(layer_count - 1):
            connections = [Connection(upstream_node, downstream_node) 
            for upstream_node in self.layers[layer].nodes for downstream_node in self.layers[layer + 1].nodes[:-1]]
            # 遍历 connections，将 conn 添加到 connections 中
            for conn in connections:
                self.connections.add_connection(conn)
                # 为下游节点添加上游节点为 conn
                conn.downstream_node.append_upstream_connection(conn)
                # 为上游节点添加下游节点为 conn
                conn.upstream_node.append_downstream_connection(conn)


    def train(self, labels, data_set, rate, epoch):
        '''
        Desc:
            训练神经网络
        Args:
            labels --- 数组，训练样本标签，每个元素是一个样本的标签
            data_set --- 二维数组，训练样本的特征数据。每行数据是一个样本的特征
            rate --- 学习率
            epoch --- 迭代次数
        Returns:
            None
        '''
        # 循环迭代 epoch 次
        for i in range(epoch):
            # 遍历每个训练样本
            for d in range(len(data_set)):
                # 使用此样本进行训练（一条样本进行训练）
                self.train_one_sample(labels[d], data_set[d], rate)
                # print 'sample %d training finished' % d

    def train_one_sample(self, label, sample, rate):
        '''
        Desc:
            内部函数，使用一个样本对网络进行训练
        Args:
            label --- 样本的标签
            sample --- 样本的特征
            rate --- 学习率
        Returns:
            None
        '''
        # 调用 Network 的 predict 方法，对这个样本进行预测
        self.predict(sample)
        # 计算根据此样本得到的结果的 delta
        self.calc_delta(label)
        # 更新权重
        self.update_weight(rate)

    def calc_delta(self, label):
        '''
        Desc:
            计算每个节点的 delta
        Args:
            label --- 样本的真实值，也就是样本的标签
        Returns:
            None
        '''
        # 获取输出层的所有节点
        output_nodes = self.layers[-1].nodes
        # 遍历所有的 label
        for i in range(len(label)):
            # 计算输出层节点的 delta
            output_nodes[i].calc_output_layer_delta(label[i])
        # 这个用法就是切片的用法， [-2::-1] 就是将 layers: aaa = [1,2,3,4,5,6,7,8,9],bbb = aaa[-2::-1] ==> bbb = [8, 7, 6, 5, 4, 3, 2, 1]
        # 实际上就是除掉输出层之外的所有层按照相反的顺序进行遍历
        for layer in self.layers[-2::-1]:
            # 遍历每层的所有节点
            for node in layer.nodes:
                # 计算隐藏层的 delta
                node.calc_hidden_layer_delta()

    def update_weight(self, rate):
        '''
        Desc:
            更新每个连接的权重
        Args:
            rate --- 学习率
        Returns:
            None
        '''
        # 按照正常顺序遍历除了输出层的层
        for layer in self.layers[:-1]:
            # 遍历每层的所有节点
            for node in layer.nodes:
                # 遍历节点的下游节点
                for conn in node.downstream:
                    # 根据下游节点来更新连接的权重
                    conn.update_weight(rate)

    def calc_gradient(self):
        '''
        Desc:
            计算每个连接的梯度
        Args:
            None
        Returns:
            None
        '''
        # 按照正常顺序遍历除了输出层之外的层
        for layer in self.layers[:-1]:
            # 遍历层中的所有节点
            for node in layer.nodes:
                # 遍历节点的下游节点
                for conn in node.downstream:
                    # 计算梯度
                    conn.calc_gradient()

    def get_gradient(self, label, sample):
        '''
        Desc:
            获得网络在一个样本下，每个连接上的梯度
        Args:
            label --- 样本标签
            sample --- 样本特征
        Returns:
            None
        '''
        # 调用 predict() 方法，利用样本的特征数据对样本进行预测
        self.predict(sample)
        # 计算 delta
        self.calc_delta(label)
        # 计算梯度
        self.calc_gradient()

    def predict(self, sample):
        '''
        Desc:
            根据输入的样本预测输出值
        Args:
            sample --- 数组，样本的特征，也就是网络的输入向量
        Returns:
            使用我们的感知器规则计算网络的输出
        '''
        # 首先为输入层设置输出值output为样本的输入向量，即不发生任何变化
        self.layers[0].set_output(sample)
        # 遍历除去输入层开始到最后一层
        for i in range(1, len(self.layers)):
            # 计算 output
            self.layers[i].calc_output()
        # 将计算得到的输出，也就是我们的预测值返回
        return list(map(lambda node: node.output, self.layers[-1].nodes[:-1]))

    def dump(self):
        '''
        Desc:
            打印出我们的网络信息
        Args:
            None
        Returns:
            None
        '''
        # 遍历所有的 layers
        for layer in self.layers:
            # 将所有的层的信息打印出来
            layer.dump()

class Normalizer(object):
    '''
    Desc:
        归一化工具类
    Args:
        object --- 对象
    Returns:
        None
    '''
    def __init__(self):
        '''
        Desc:
            初始化
        Args:
            None
        Returns:
            None
        '''
        # 初始化 16 进制的数，用来判断位的，分别是
        # 0x1 ---- 00000001
        # 0x2 ---- 00000010
        # 0x4 ---- 00000100
        # 0x8 ---- 00001000
        # 0x10 --- 00010000
        # 0x20 --- 00100000
        # 0x40 --- 01000000
        # 0x80 --- 10000000
        self.mask = [0x1, 0x2, 0x4, 0x8, 0x10, 0x20, 0x40, 0x80]

    def norm(self, number):
        '''
        Desc:
            对 number 进行规范化
        Args:
            number --- 要规范化的数据
        Returns:
            规范化之后的数据
        '''
        # 此方法就相当于判断一个 8 位的向量，哪一位上有数字，如果有就将这个数设置为  0.9 ，否则，设置为 0.1，通俗比较来说，就是我们这里用 0.9 表示 1，用 0.1 表示 0
        return list(map(lambda m: 0.9 if number & m else 0.1, self.mask))

    def denorm(self, vec):
        '''
        Desc:
            对我们得到的向量进行反规范化
        Args:
            vec --- 得到的向量
        Returns:
            最终的预测结果
        '''
        # 进行二分类，大于 0.5 就设置为 1，小于 0.5 就设置为 0
        binary = list(map(lambda i: 1 if i > 0.5 else 0, vec))
        # 遍历 mask
        for i in range(len(self.mask)):
            binary[i] = binary[i] * self.mask[i]
        # 将结果相加得到最终的预测结果
        return reduce(lambda x,y: x + y, binary)


def mean_square_error(vec1, vec2):
    '''
    Desc:
        计算平均平方误差
    Args:
        vec1 --- 第一个数
        vec2 --- 第二个数
    Returns:
        返回 1/2 * (x-y)^2 计算得到的值
    '''
    return 0.5 * reduce(lambda a, b: a + b, map(lambda v: (v[0] - v[1]) * (v[0] - v[1]), zip(vec1, vec2)))



def gradient_check(network, sample_feature, sample_label):
    '''
    Desc:
        梯度检查
    Args:
        network --- 神经网络对象
        sample_feature --- 样本的特征
        sample_label --- 样本的标签   
    Returns:
        None
    '''
    # 计算网络误差
    network_error = lambda vec1, vec2: 0.5 * reduce(lambda a, b: a + b, map(lambda v: (v[0] - v[1]) * (v[0] - v[1]), zip(vec1, vec2)))

    # 获取网络在当前样本下每个连接的梯度
    network.get_gradient(sample_feature, sample_label)

    # 对每个权重做梯度检查    
    for conn in network.connections.connections: 
        # 获取指定连接的梯度
        actual_gradient = conn.get_gradient()
    
        # 增加一个很小的值，计算网络的误差
        epsilon = 0.0001
        conn.weight += epsilon
        error1 = network_error(network.predict(sample_feature), sample_label)
    
        conn.weight -= 2 * epsilon
        error2 = network_error(network.predict(sample_feature), sample_label)
    
        expected_gradient = (error2 - error1) / (2 * epsilon)
    
        print('expected gradient: \t%f\nactual gradient: \t%f' % (expected_gradient, actual_gradient))


def train_data_set():
    '''
    Desc:
        unk
    Args:
        None
    Returns:
        labels --- 
    '''
    normalizer = Normalizer()
    data_set = []
    labels = []
    for i in range(0, 256, 8):
        # normalizer norm
        n = normalizer.norm(int(random.uniform(0, 256)))
        # data_set append n
        data_set.append(n)
        # labels append n
        labels.append(n)
    return labels, data_set


def train(network):
    '''
    Desc:
        unk
    Args:
        network --- 
    Returns:
        None
    '''
    labels, data_set = train_data_set()
    labels = list(labels)
    data_set = list(labels)
    # network train
    network.train(labels, data_set, 0.3, 50)


def test(net,data):

    '''
    Desc:
        unk
    Args:
        network --- 
        data ------ 
    Returns:
        None
    '''
    # 调用 Normalizer()

    normalizer = Normalizer()
    norm_data = normalizer.norm(data)
    norm_data = list(norm_data)
    predict_data = net.predict(norm_data)
    print('\ttestdata(%u)\tpredict(%u)' % (data, normalizer.denorm(predict_data)))


def correct_ratio(network):
    '''
    Desc:
        unk
    Args:
        network --- 
    Returns:
        None
    '''
    normalizer = Normalizer()
    correct = 0.0
    for i in range(256):
        if normalizer.denorm(network.predict(normalizer.norm(i))) == i:
            correct += 1.0
    print('correct_ratio: %.2f%%' % (correct / 256 * 100))


def gradient_check_test():
    '''
    Desc:
        None
    Args:
        None
    Returns:
        None
    '''
    net = Network([2, 2, 2])
    sample_feature = [0.9, 0.1]
    sample_label = [0.9, 0.1]
    gradient_check(net, sample_feature, sample_label)


if __name__ == '__main__':
    '''
    Desc:
        None
    Args:
        None
    Returns:
        None
    '''
    net = Network([8, 3, 8])
    train(net)
    net.dump()
    correct_ratio(net)
    
#!/usr/bin/env python
# -*- coding: UTF-8 -*-


import numpy as np
from activators import ReluActivator, IdentityActivator


# 获取卷积区域
def get_patch(input_array, i, j, filter_width,
              filter_height, stride):
    '''
    从输入数组中获取本次卷积的区域，
    自动适配输入为2D和3D的情况
    '''
    start_i = i * stride
    start_j = j * stride
    if input_array.ndim == 2:
        return input_array[
               start_i: start_i + filter_height,
               start_j: start_j + filter_width]
    elif input_array.ndim == 3:
        return input_array[:,
               start_i: start_i + filter_height,
               start_j: start_j + filter_width]


# 获取一个2D区域的最大值所在的索引
def get_max_index(array):
    max_i = 0
    max_j = 0
    max_value = array[0, 0]
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            if array[i, j] > max_value:
                max_value = array[i, j]
                max_i, max_j = i, j
    return max_i, max_j


# 计算卷积
def conv(input_array,
         kernel_array,
         output_array,
         stride, bias):
    '''
    计算卷积，自动适配输入为2D和3D的情况
    '''
    channel_number = input_array.ndim
    output_width = output_array.shape[1]
    output_height = output_array.shape[0]
    kernel_width = kernel_array.shape[-1]
    kernel_height = kernel_array.shape[-2]
    for i in range(output_height):
        for j in range(output_width):
            output_array[i][j] = (
                                         get_patch(input_array, i, j, kernel_width,
                                                   kernel_height, stride) * kernel_array
                                 ).sum() + bias


# 为数组增加Zero padding
def padding(input_array, zp):
    '''
    为数组增加Zero padding，自动适配输入为2D和3D的情况
    '''
    if zp == 0:
        return input_array
    else:
        if input_array.ndim == 3:
            input_width = input_array.shape[2]
            input_height = input_array.shape[1]
            input_depth = input_array.shape[0]
            padded_array = np.zeros((
                input_depth,
                input_height + 2 * zp,
                input_width + 2 * zp))
            padded_array[:,
            zp: zp + input_height,
            zp: zp + input_width] = input_array
            return padded_array
        elif input_array.ndim == 2:
            input_width = input_array.shape[1]
            input_height = input_array.shape[0]
            padded_array = np.zeros((
                input_height + 2 * zp,
                input_width + 2 * zp))
            padded_array[zp: zp + input_height,
            zp: zp + input_width] = input_array
            return padded_array


# 对numpy数组进行element wise操作
def element_wise_op(array, op):
    for i in np.nditer(array,
                       op_flags=['readwrite']):
        i[...] = op(i)


class Filter(object):
    def __init__(self, width, height, depth):
        self.weights = np.random.uniform(-1e-4, 1e-4,
                                         (depth, height, width))
        self.bias = 0
        self.weights_grad = np.zeros(
            self.weights.shape)
        self.bias_grad = 0

    def __repr__(self):
        return 'filter weights:\n%s\nbias:\n%s' % (
            repr(self.weights), repr(self.bias))

    def get_weights(self):
        return self.weights

    def get_bias(self):
        return self.bias

    def update(self, learning_rate):
        self.weights -= learning_rate * self.weights_grad
        self.bias -= learning_rate * self.bias_grad


class ConvLayer(object):
    def __init__(self, input_width, input_height,
                 channel_number, filter_width,
                 filter_height, filter_number,
                 zero_padding, stride, activator,
                 learning_rate):
        self.input_width = input_width
        self.input_height = input_height
        self.channel_number = channel_number
        self.filter_width = filter_width
        self.filter_height = filter_height
        self.filter_number = filter_number
        self.zero_padding = zero_padding
        self.stride = stride
        self.output_width = \
            ConvLayer.calculate_output_size(
                self.input_width, filter_width, zero_padding,
                stride)
        self.output_height = \
            ConvLayer.calculate_output_size(
                self.input_height, filter_height, zero_padding,
                stride)
        self.output_array = np.zeros((self.filter_number,
                                      self.output_height, self.output_width))
        self.filters = []
        for i in range(filter_number):
            self.filters.append(Filter(filter_width,
                                       filter_height, self.channel_number))
        self.activator = activator
        self.learning_rate = learning_rate

    def forward(self, input_array):
        '''
        计算卷积层的输出
        输出结果保存在self.output_array
        '''
        self.input_array = input_array
        self.padded_input_array = padding(input_array,
                                          self.zero_padding)
        for f in range(self.filter_number):
            filter = self.filters[f]
            conv(self.padded_input_array,
                 filter.get_weights(), self.output_array[f],
                 self.stride, filter.get_bias())
        element_wise_op(self.output_array,
                        self.activator.forward)

    def backward(self, input_array, sensitivity_array,
                 activator):
        '''
        计算传递给前一层的误差项，以及计算每个权重的梯度
        前一层的误差项保存在self.delta_array
        梯度保存在Filter对象的weights_grad
        '''
        self.forward(input_array)
        self.bp_sensitivity_map(sensitivity_array,
                                activator)
        self.bp_gradient(sensitivity_array)

    def update(self):
        '''
        按照梯度下降，更新权重
        '''
        for filter in self.filters:
            filter.update(self.learning_rate)

    def bp_sensitivity_map(self, sensitivity_array,
                           activator):
        '''
        计算传递到上一层的sensitivity map
        sensitivity_array: 本层的sensitivity map
        activator: 上一层的激活函数
        '''
        # 处理卷积步长，对原始sensitivity map进行扩展
        expanded_array = self.expand_sensitivity_map(
            sensitivity_array)
        # full卷积，对sensitivitiy map进行zero padding
        # 虽然原始输入的zero padding单元也会获得残差
        # 但这个残差不需要继续向上传递，因此就不计算了
        expanded_width = expanded_array.shape[2]
        zp = (self.input_width +
              self.filter_width - 1 - expanded_width) // 2
        padded_array = padding(expanded_array, zp)
        # 初始化delta_array，用于保存传递到上一层的
        # sensitivity map
        self.delta_array = self.create_delta_array()
        # 对于具有多个filter的卷积层来说，最终传递到上一层的
        # sensitivity map相当于所有的filter的
        # sensitivity map之和
        for f in range(self.filter_number):
            filter = self.filters[f]
            # 将filter权重翻转180度
            flipped_weights = np.array(list(map(lambda i: np.rot90(i, 2), filter.get_weights())))
            # 计算与一个filter对应的delta_array
            delta_array = self.create_delta_array()
            for d in range(delta_array.shape[0]):
                conv(padded_array[f], flipped_weights[d],
                     delta_array[d], 1, 0)
            self.delta_array += delta_array
        # 将计算结果与激活函数的偏导数做element-wise乘法操作
        derivative_array = np.array(self.input_array)
        element_wise_op(derivative_array,
                        activator.backward)
        self.delta_array *= derivative_array

    def bp_gradient(self, sensitivity_array):
        # 处理卷积步长，对原始sensitivity map进行扩展
        expanded_array = self.expand_sensitivity_map(
            sensitivity_array)
        for f in range(self.filter_number):
            # 计算每个权重的梯度
            filter = self.filters[f]
            for d in range(filter.weights.shape[0]):
                conv(self.padded_input_array[d],
                     expanded_array[f],
                     filter.weights_grad[d], 1, 0)
            # 计算偏置项的梯度
            filter.bias_grad = expanded_array[f].sum()

    def expand_sensitivity_map(self, sensitivity_array):
        depth = sensitivity_array.shape[0]
        # 确定扩展后sensitivity map的大小
        # 计算stride为1时sensitivity map的大小
        expanded_width = (self.input_width -
                          self.filter_width + 2 * self.zero_padding + 1)
        expanded_height = (self.input_height -
                           self.filter_height + 2 * self.zero_padding + 1)
        # 构建新的sensitivity_map
        expand_array = np.zeros((depth, expanded_height,
                                 expanded_width))
        # 从原始sensitivity map拷贝误差值
        for i in range(self.output_height):
            for j in range(self.output_width):
                i_pos = i * self.stride
                j_pos = j * self.stride
                expand_array[:, i_pos, j_pos] = \
                    sensitivity_array[:, i, j]
        return expand_array

    def create_delta_array(self):
        return np.zeros((self.channel_number,
                         self.input_height, self.input_width))

    @staticmethod
    def calculate_output_size(input_size,
                              filter_size, zero_padding, stride):
        return (input_size - filter_size +
                2 * zero_padding) // stride + 1


class MaxPoolingLayer(object):
    def __init__(self, input_width, input_height,
                 channel_number, filter_width,
                 filter_height, stride):
        self.input_width = input_width
        self.input_height = input_height
        self.channel_number = channel_number
        self.filter_width = filter_width
        self.filter_height = filter_height
        self.stride = stride
        self.output_width = (input_width -
                             filter_width) // self.stride + 1
        self.output_height = (input_height -
                              filter_height) // self.stride + 1
        self.output_array = np.zeros((self.channel_number,
                                      self.output_height, self.output_width))

    def forward(self, input_array):
        for d in range(self.channel_number):
            for i in range(self.output_height):
                for j in range(self.output_width):
                    self.output_array[d, i, j] = (
                        get_patch(input_array[d], i, j,
                                  self.filter_width,
                                  self.filter_height,
                                  self.stride).max())

    def backward(self, input_array, sensitivity_array):
        self.delta_array = np.zeros(input_array.shape)
        for d in range(self.channel_number):
            for i in range(self.output_height):
                for j in range(self.output_width):
                    patch_array = get_patch(
                        input_array[d], i, j,
                        self.filter_width,
                        self.filter_height,
                        self.stride)
                    k, l = get_max_index(patch_array)
                    self.delta_array[d,
                                     i * self.stride + k,
                                     j * self.stride + l] = \
                        sensitivity_array[d, i, j]


def init_test():
    a = np.array(
        [[[0, 1, 1, 0, 2],
          [2, 2, 2, 2, 1],
          [1, 0, 0, 2, 0],
          [0, 1, 1, 0, 0],
          [1, 2, 0, 0, 2]],
         [[1, 0, 2, 2, 0],
          [0, 0, 0, 2, 0],
          [1, 2, 1, 2, 1],
          [1, 0, 0, 0, 0],
          [1, 2, 1, 1, 1]],
         [[2, 1, 2, 0, 0],
          [1, 0, 0, 1, 0],
          [0, 2, 1, 0, 1],
          [0, 1, 2, 2, 2],
          [2, 1, 0, 0, 1]]])
    b = np.array(
        [[[0, 1, 1],
          [2, 2, 2],
          [1, 0, 0]],
         [[1, 0, 2],
          [0, 0, 0],
          [1, 2, 1]]])
    cl = ConvLayer(5, 5, 3, 3, 3, 2, 1, 2, IdentityActivator(), 0.001)
    cl.filters[0].weights = np.array(
        [[[-1, 1, 0],
          [0, 1, 0],
          [0, 1, 1]],
         [[-1, -1, 0],
          [0, 0, 0],
          [0, -1, 0]],
         [[0, 0, -1],
          [0, 1, 0],
          [1, -1, -1]]], dtype=np.float64)
    cl.filters[0].bias = 1
    cl.filters[1].weights = np.array(
        [[[1, 1, -1],
          [-1, -1, 1],
          [0, -1, 1]],
         [[0, 1, 0],
          [-1, 0, -1],
          [-1, 1, 0]],
         [[-1, 0, 0],
          [-1, 0, 1],
          [-1, 0, 0]]], dtype=np.float64)
    return a, b, cl


def test():
    a, b, cl = init_test()
    cl.forward(a)
    print(
    cl.output_array)


def test_bp():
    a, b, cl = init_test()
    cl.backward(a, b, IdentityActivator())
    cl.update()
    print(
    cl.filters[0])
    print(
    cl.filters[1])

def gradient_check():
    '''
    梯度检查
    '''
    # 设计一个误差函数，取所有节点输出项之和
    error_function = lambda o: o.sum()

    # 计算forward值
    a, b, cl = init_test()
    cl.forward(a)

    # 求取sensitivity map
    sensitivity_array = np.ones(cl.output_array.shape,
                                dtype=np.float64)
    # 计算梯度
    cl.backward(a, sensitivity_array,
                IdentityActivator())
    # 检查梯度
    epsilon = 10e-4
    for d in range(cl.filters[0].weights_grad.shape[0]):
        for i in range(cl.filters[0].weights_grad.shape[1]):
            for j in range(cl.filters[0].weights_grad.shape[2]):
                cl.filters[0].weights[d, i, j] += epsilon
                cl.forward(a)
                err1 = error_function(cl.output_array)
                cl.filters[0].weights[d, i, j] -= 2 * epsilon
                cl.forward(a)
                err2 = error_function(cl.output_array)
                expect_grad = (err1 - err2) / (2 * epsilon)
                cl.filters[0].weights[d, i, j] += epsilon
                print(
                'weights(%d,%d,%d): expected - actural %f - %f' % (
                    d, i, j, expect_grad, cl.filters[0].weights_grad[d, i, j]))


def init_pool_test():
    a = np.array(
        [[[1, 1, 2, 4],
          [5, 6, 7, 8],
          [3, 2, 1, 0],
          [1, 2, 3, 4]],
         [[0, 1, 2, 3],
          [4, 5, 6, 7],
          [8, 9, 0, 1],
          [3, 4, 5, 6]]], dtype=np.float64)

    b = np.array(
        [[[1, 2],
          [2, 4]],
         [[3, 5],
          [8, 2]]], dtype=np.float64)

    mpl = MaxPoolingLayer(4, 4, 2, 2, 2, 2)

    return a, b, mpl


def test_pool():
    a, b, mpl = init_pool_test()
    mpl.forward(a)
    print(
    'input array:\n%s\noutput array:\n%s' % (a,
                                             mpl.output_array))


def test_pool_bp():
    a, b, mpl = init_pool_test()
    mpl.backward(a, b)
    print(
    'input array:\n%s\nsensitivity array:\n%s\ndelta array:\n%s' % (
        a, b, mpl.delta_array))


if __name__=='__main__':
    gradient_check()
    
#!/usr/bin/env python
# -*- coding: UTF-8 -*-


import random
import numpy as np
from functools import reduce
from activators import SigmoidActivator, IdentityActivator


# 全连接层实现类
class FullConnectedLayer(object):
    def __init__(self, input_size, output_size, 
                 activator):
        '''
        构造函数
        input_size: 本层输入向量的维度
        output_size: 本层输出向量的维度
        activator: 激活函数
        '''
        self.input_size = input_size
        self.output_size = output_size
        self.activator = activator
        # 权重数组W
        self.W = np.random.uniform(-0.1, 0.1,
            (output_size, input_size))
        # 偏置项b
        self.b = np.zeros((output_size, 1))
        # 输出向量
        self.output = np.zeros((output_size, 1))

    def forward(self, input_array):
        '''
        前向计算
        input_array: 输入向量，维度必须等于input_size
        '''
        # 式2
        self.input = input_array
        self.output = self.activator.forward(
            np.dot(self.W, input_array) + self.b)

    def backward(self, delta_array):
        '''
        反向计算W和b的梯度
        delta_array: 从上一层传递过来的误差项
        '''
        # 式8
        self.delta = self.activator.backward(self.input) * np.dot(
            self.W.T, delta_array)
        self.W_grad = np.dot(delta_array, self.input.T)
        self.b_grad = delta_array

    def update(self, learning_rate):
        '''
        使用梯度下降算法更新权重
        '''
        self.W += learning_rate * self.W_grad
        self.b += learning_rate * self.b_grad

    def dump(self):
        print('W: %s\nb:%s' % (self.W, self.b))


# 神经网络类
class Network(object):
    def __init__(self, layers):
        '''
        构造函数
        '''
        self.layers = []
        for i in range(len(layers) - 1):
            self.layers.append(
                FullConnectedLayer(
                    layers[i], layers[i+1],
                    SigmoidActivator()
                )
            )

    def predict(self, sample):
        '''
        使用神经网络实现预测
        sample: 输入样本
        '''
        output = sample
        for layer in self.layers:
            layer.forward(output)
            output = layer.output
        return output

    def train(self, labels, data_set, rate, epoch):
        '''
        训练函数
        labels: 样本标签
        data_set: 输入样本
        rate: 学习速率
        epoch: 训练轮数
        '''
        for i in range(epoch):
            for d in range(len(list(data_set))):
                self.train_one_sample(labels[d], 
                    data_set[d], rate)

    def train_one_sample(self, label, sample, rate):
        self.predict(sample)
        self.calc_gradient(label)
        self.update_weight(rate)

    def calc_gradient(self, label):
        delta = self.layers[-1].activator.backward(
            self.layers[-1].output
        ) * (label - self.layers[-1].output)
        for layer in self.layers[::-1]:
            layer.backward(delta)
            delta = layer.delta
        return delta

    def update_weight(self, rate):
        for layer in self.layers:
            layer.update(rate)

    def dump(self):
        for layer in self.layers:
            layer.dump()

    def loss(self, output, label):
        return 0.5 * ((label - output) * (label - output)).sum()

    def gradient_check(self, sample_feature, sample_label):
        '''
        梯度检查
        network: 神经网络对象
        sample_feature: 样本的特征
        sample_label: 样本的标签
        '''

        # 获取网络在当前样本下每个连接的梯度
        self.predict(sample_feature)
        self.calc_gradient(sample_label)

        # 检查梯度
        epsilon = 10e-4
        for fc in self.layers:
            for i in range(fc.W.shape[0]):
                for j in range(fc.W.shape[1]):
                    fc.W[i,j] += epsilon
                    output = self.predict(sample_feature)
                    err1 = self.loss(sample_label, output)
                    fc.W[i,j] -= 2*epsilon
                    output = self.predict(sample_feature)
                    err2 = self.loss(sample_label, output)
                    expect_grad = (err1 - err2) / (2 * epsilon)
                    fc.W[i,j] += epsilon
                    print('weights(%d,%d): expected - actural %.4e - %.4e' % (
                        i, j, expect_grad, fc.W_grad[i,j]))


from bp import train_data_set


def transpose(args):
    return map(
        lambda arg: map(
            lambda line: np.array(line).reshape(len(line), 1)
            , arg)
        , args
    )


class Normalizer(object):
    def __init__(self):
        self.mask = [
            0x1, 0x2, 0x4, 0x8, 0x10, 0x20, 0x40, 0x80
        ]

    def norm(self, number):
        data = list(map(lambda m: 0.9 if number & m else 0.1, self.mask))
        return np.array(data).reshape(8, 1)

    def denorm(self, vec):
        binary = list(map(lambda i: 1 if i > 0.5 else 0, vec[:,0]))
        for i in range(len(self.mask)):
            binary[i] = binary[i] * self.mask[i]
        return reduce(lambda x,y: x + y, binary)

def train_data_set():
    normalizer = Normalizer()
    data_set = []
    labels = []
    for i in range(0, 256):
        n = normalizer.norm(i)
        data_set.append(n)
        labels.append(n)
    return labels, data_set

def correct_ratio(network):
    normalizer = Normalizer()
    correct = 0.0;
    for i in range(256):
        if normalizer.denorm(network.predict(normalizer.norm(i))) == i:
            correct += 1.0
    print('correct_ratio: %.2f%%' % (correct / 256 * 100))


def test():
    labels, data_set = list(transpose(train_data_set()))
    labels=list(labels)
    data_set=list(data_set)
    net = Network([8, 3, 8])
    rate = 0.5
    mini_batch = 20
    epoch = 10
    for i in range(epoch):
        net.train(labels, list(data_set), rate, mini_batch)
        print('after epoch %d loss: %f' % (
            (i + 1),
            net.loss(labels[-1], net.predict(data_set[-1]))
        ))
        rate /= 2
    correct_ratio(net)


def gradient_check():
    '''
    梯度检查
    '''
    labels, data_set = transpose(train_data_set())
    net = Network([8, 3, 8])
    net.gradient_check(data_set[0], labels[0])
    return net

#!/usr/bin/env python
# -*- coding: UTF-8 -*-


import matplotlib.pyplot as plt
import numpy as np
from cnn import element_wise_op
from activators import SigmoidActivator, TanhActivator, IdentityActivator


class LstmLayer(object):
    def __init__(self, input_width, state_width, 
                 learning_rate):
        self.input_width = input_width
        self.state_width = state_width
        self.learning_rate = learning_rate
        # 门的激活函数
        self.gate_activator = SigmoidActivator()
        # 输出的激活函数
        self.output_activator = TanhActivator()
        # 当前时刻初始化为t0
        self.times = 0       
        # 各个时刻的单元状态向量c
        self.c_list = self.init_state_vec()
        # 各个时刻的输出向量h
        self.h_list = self.init_state_vec()
        # 各个时刻的遗忘门f
        self.f_list = self.init_state_vec()
        # 各个时刻的输入门i
        self.i_list = self.init_state_vec()
        # 各个时刻的输出门o
        self.o_list = self.init_state_vec()
        # 各个时刻的即时状态c~
        self.ct_list = self.init_state_vec()
        # 遗忘门权重矩阵Wfh, Wfx, 偏置项bf
        self.Wfh, self.Wfx, self.bf = (
            self.init_weight_mat())
        # 输入门权重矩阵Wfh, Wfx, 偏置项bf
        self.Wih, self.Wix, self.bi = (
            self.init_weight_mat())
        # 输出门权重矩阵Wfh, Wfx, 偏置项bf
        self.Woh, self.Wox, self.bo = (
            self.init_weight_mat())
        # 单元状态权重矩阵Wfh, Wfx, 偏置项bf
        self.Wch, self.Wcx, self.bc = (
            self.init_weight_mat())

    def init_state_vec(self):
        '''
        初始化保存状态的向量
        '''
        state_vec_list = []
        state_vec_list.append(np.zeros(
            (self.state_width, 1)))
        return state_vec_list

    def init_weight_mat(self):
        '''
        初始化权重矩阵
        '''
        Wh = np.random.uniform(-1e-4, 1e-4,
            (self.state_width, self.state_width))
        Wx = np.random.uniform(-1e-4, 1e-4,
            (self.state_width, self.input_width))
        b = np.zeros((self.state_width, 1))
        return Wh, Wx, b

    def forward(self, x):
        '''
        根据式1-式6进行前向计算
        '''
        self.times += 1
        # 遗忘门
        fg = self.calc_gate(x, self.Wfx, self.Wfh, 
            self.bf, self.gate_activator)
        self.f_list.append(fg)
        # 输入门
        ig = self.calc_gate(x, self.Wix, self.Wih,
            self.bi, self.gate_activator)
        self.i_list.append(ig)
        # 输出门
        og = self.calc_gate(x, self.Wox, self.Woh,
            self.bo, self.gate_activator)
        self.o_list.append(og)
        # 即时状态
        ct = self.calc_gate(x, self.Wcx, self.Wch,
            self.bc, self.output_activator)
        self.ct_list.append(ct)
        # 单元状态
        c = fg * self.c_list[self.times - 1] + ig * ct
        self.c_list.append(c)
        # 输出
        h = og * self.output_activator.forward(c)
        self.h_list.append(h)

    def calc_gate(self, x, Wx, Wh, b, activator):
        '''
        计算门
        '''
        h = self.h_list[self.times - 1] # 上次的LSTM输出
        net = np.dot(Wh, h) + np.dot(Wx, x) + b
        gate = activator.forward(net)
        return gate


    def backward(self, x, delta_h, activator):
        '''
        实现LSTM训练算法
        '''
        self.calc_delta(delta_h, activator)
        self.calc_gradient(x)

    def update(self):
        '''
        按照梯度下降，更新权重
        '''
        self.Wfh -= self.learning_rate * self.Whf_grad
        self.Wfx -= self.learning_rate * self.Whx_grad
        self.bf -= self.learning_rate * self.bf_grad
        self.Wih -= self.learning_rate * self.Whi_grad
        self.Wix -= self.learning_rate * self.Whi_grad
        self.bi -= self.learning_rate * self.bi_grad
        self.Woh -= self.learning_rate * self.Wof_grad
        self.Wox -= self.learning_rate * self.Wox_grad
        self.bo -= self.learning_rate * self.bo_grad
        self.Wch -= self.learning_rate * self.Wcf_grad
        self.Wcx -= self.learning_rate * self.Wcx_grad
        self.bc -= self.learning_rate * self.bc_grad

    def calc_delta(self, delta_h, activator):
        # 初始化各个时刻的误差项
        self.delta_h_list = self.init_delta()  # 输出误差项
        self.delta_o_list = self.init_delta()  # 输出门误差项
        self.delta_i_list = self.init_delta()  # 输入门误差项
        self.delta_f_list = self.init_delta()  # 遗忘门误差项
        self.delta_ct_list = self.init_delta() # 即时输出误差项

        # 保存从上一层传递下来的当前时刻的误差项
        self.delta_h_list[-1] = delta_h
        
        # 迭代计算每个时刻的误差项
        for k in range(self.times, 0, -1):
            self.calc_delta_k(k)

    def init_delta(self):
        '''
        初始化误差项
        '''
        delta_list = []
        for i in range(self.times + 1):
            delta_list.append(np.zeros(
                (self.state_width, 1)))
        return delta_list

    def calc_delta_k(self, k):
        '''
        根据k时刻的delta_h，计算k时刻的delta_f、
        delta_i、delta_o、delta_ct，以及k-1时刻的delta_h
        '''
        # 获得k时刻前向计算的值
        ig = self.i_list[k]
        og = self.o_list[k]
        fg = self.f_list[k]
        ct = self.ct_list[k]
        c = self.c_list[k]
        c_prev = self.c_list[k-1]
        tanh_c = self.output_activator.forward(c)
        delta_k = self.delta_h_list[k]

        # 根据式9计算delta_o
        delta_o = (delta_k * tanh_c * 
            self.gate_activator.backward(og))
        delta_f = (delta_k * og * 
            (1 - tanh_c * tanh_c) * c_prev *
            self.gate_activator.backward(fg))
        delta_i = (delta_k * og * 
            (1 - tanh_c * tanh_c) * ct *
            self.gate_activator.backward(ig))
        delta_ct = (delta_k * og * 
            (1 - tanh_c * tanh_c) * ig *
            self.output_activator.backward(ct))
        delta_h_prev = (
                np.dot(delta_o.transpose(), self.Woh) +
                np.dot(delta_i.transpose(), self.Wih) +
                np.dot(delta_f.transpose(), self.Wfh) +
                np.dot(delta_ct.transpose(), self.Wch)
            ).transpose()

        # 保存全部delta值
        self.delta_h_list[k-1] = delta_h_prev
        self.delta_f_list[k] = delta_f
        self.delta_i_list[k] = delta_i
        self.delta_o_list[k] = delta_o
        self.delta_ct_list[k] = delta_ct

    def calc_gradient(self, x):
        # 初始化遗忘门权重梯度矩阵和偏置项
        self.Wfh_grad, self.Wfx_grad, self.bf_grad = (
            self.init_weight_gradient_mat())
        # 初始化输入门权重梯度矩阵和偏置项
        self.Wih_grad, self.Wix_grad, self.bi_grad = (
            self.init_weight_gradient_mat())
        # 初始化输出门权重梯度矩阵和偏置项
        self.Woh_grad, self.Wox_grad, self.bo_grad = (
            self.init_weight_gradient_mat())
        # 初始化单元状态权重梯度矩阵和偏置项
        self.Wch_grad, self.Wcx_grad, self.bc_grad = (
            self.init_weight_gradient_mat())

       # 计算对上一次输出h的权重梯度
        for t in range(self.times, 0, -1):
            # 计算各个时刻的梯度
            (Wfh_grad, bf_grad,
            Wih_grad, bi_grad,
            Woh_grad, bo_grad,
            Wch_grad, bc_grad) = (
                self.calc_gradient_t(t))
            # 实际梯度是各时刻梯度之和
            self.Wfh_grad += Wfh_grad
            self.bf_grad += bf_grad
            self.Wih_grad += Wih_grad
            self.bi_grad += bi_grad
            self.Woh_grad += Woh_grad
            self.bo_grad += bo_grad
            self.Wch_grad += Wch_grad
            self.bc_grad += bc_grad

        # 计算对本次输入x的权重梯度
        xt = x.transpose()
        self.Wfx_grad = np.dot(self.delta_f_list[-1], xt)
        self.Wix_grad = np.dot(self.delta_i_list[-1], xt)
        self.Wox_grad = np.dot(self.delta_o_list[-1], xt)
        self.Wcx_grad = np.dot(self.delta_ct_list[-1], xt)

    def init_weight_gradient_mat(self):
        '''
        初始化权重矩阵
        '''
        Wh_grad = np.zeros((self.state_width,
            self.state_width))
        Wx_grad = np.zeros((self.state_width,
            self.input_width))
        b_grad = np.zeros((self.state_width, 1))
        return Wh_grad, Wx_grad, b_grad

    def calc_gradient_t(self, t):
        '''
        计算每个时刻t权重的梯度
        '''
        h_prev = self.h_list[t-1].transpose()
        Wfh_grad = np.dot(self.delta_f_list[t], h_prev)
        bf_grad = self.delta_f_list[t]
        Wih_grad = np.dot(self.delta_i_list[t], h_prev)
        bi_grad = self.delta_f_list[t]
        Woh_grad = np.dot(self.delta_o_list[t], h_prev)
        bo_grad = self.delta_f_list[t]
        Wch_grad = np.dot(self.delta_ct_list[t], h_prev)
        bc_grad = self.delta_ct_list[t]
        return Wfh_grad, bf_grad, Wih_grad, bi_grad, \
               Woh_grad, bo_grad, Wch_grad, bc_grad

    def reset_state(self):
        # 当前时刻初始化为t0
        self.times = 0       
        # 各个时刻的单元状态向量c
        self.c_list = self.init_state_vec()
        # 各个时刻的输出向量h
        self.h_list = self.init_state_vec()
        # 各个时刻的遗忘门f
        self.f_list = self.init_state_vec()
        # 各个时刻的输入门i
        self.i_list = self.init_state_vec()
        # 各个时刻的输出门o
        self.o_list = self.init_state_vec()
        # 各个时刻的即时状态c~
        self.ct_list = self.init_state_vec()


def data_set():
    x = [np.array([[1], [2], [3]]),
         np.array([[2], [3], [4]])]
    d = np.array([[1], [2]])
    return x, d


def gradient_check():
    '''
    梯度检查
    '''
    # 设计一个误差函数，取所有节点输出项之和
    error_function = lambda o: o.sum()
    
    lstm = LstmLayer(3, 2, 1e-3)

    # 计算forward值
    x, d = data_set()
    lstm.forward(x[0])
    lstm.forward(x[1])
    
    # 求取sensitivity map
    sensitivity_array = np.ones(lstm.h_list[-1].shape,
                                dtype=np.float64)
    # 计算梯度
    lstm.backward(x[1], sensitivity_array, IdentityActivator())
    
    # 检查梯度
    epsilon = 10e-4
    for i in range(lstm.Wfh.shape[0]):
        for j in range(lstm.Wfh.shape[1]):
            lstm.Wfh[i,j] += epsilon
            lstm.reset_state()
            lstm.forward(x[0])
            lstm.forward(x[1])
            err1 = error_function(lstm.h_list[-1])
            lstm.Wfh[i,j] -= 2*epsilon
            lstm.reset_state()
            lstm.forward(x[0])
            lstm.forward(x[1])
            err2 = error_function(lstm.h_list[-1])
            expect_grad = (err1 - err2) / (2 * epsilon)
            lstm.Wfh[i,j] += epsilon
            print('weights(%d,%d): expected - actural %.4e - %.4e' % (
                i, j, expect_grad, lstm.Wfh_grad[i,j]))
    return lstm


def test():
    l = LstmLayer(3, 2, 1e-3)
    x, d = data_set()
    l.forward(x[0])
    l.forward(x[1])
    l.backward(x[1], d, IdentityActivator())
    return l

def test_gradient_check():
    gradient_check()
    
