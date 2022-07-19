# 8 节课带你入门深度学习——深度学习从理论到实践

## 项目介绍

深度学习作为机器学习的一个热门分支领域，被广泛应用在多个领域。在深度入门过程中，从理论到实践是个不小的挑战。
本项目旨在用概览、精炼的方式，帮助读者，尤其是深度学习的初学者，快速、高效地掌握深度学习领域的**核心知识**以及深度学习系统的**核心组件**，帮助初学者更好地从理论过渡到实践。
## 项目说明

本项目从全局入手，结合深度学习框架，带大家一步步自己动手搭建简单的深度学习模型。项目是由八个可交互式学习的Jupyter Notebook教程文档组成，共分为以下八个章节：  
1.快速上手：本章带领读者概览深度学习的过程，从加载数据，搭建网络，训练模型到模型保存。    
2.Tensor：本章介绍了Tensor的多种创建方法、属性以及OneFlow对Tensor灵活的操作方式。  
3.Dataset 与 DataLoader：本章介绍了用于数据集管理与模型训练解耦的Dataset与DataLoader。  
4.搭建神经网络：本章通过使用oneflow.nn 名称空间下的 API，来构建神经网络各层结构。  
5.Autograd：本章从计算图的基本概念引入，介绍了OneFlow 中与自动求导有关的常见接口。  
6.反向传播与 optimizer：本章从反向传播的基本概念着手，介绍了oneflow.optim 如何简化实现反向传播的代码。    
7.模型的加载与保存：本章主要介绍如何使用 save 和 load API 来完成对模型的保存与加载，同时也会展示如何加载预训练模型，完成预测任务。  
8.静态图模块 nn.Graph：本章介绍了OneFlow的nn.Graph模块的使用，该模块帮助用户构建静态图并训练模型。
### 目录结构

```
|-- README.md
|-- 01_quickstart.ipynb  # 第一章，概览深度学习
|-- 02_tensor.ipynb    # 第二章，Tensor介绍
|-- 03_dataset_dataloader.ipynb  # 第三章，Dataset与DataLoader
|-- 04_build_network   # 第四章，构建神经网络各层结构
|-- 05_autograd.ipynb    # 第五章，OneFlow的自动求导机制
|-- 06_optimization.ipynb  # 第六章，oneflow.optim 类的使用
|-- 07_model_load_save.ipynb   # 第七章，模型的加载与保存
|-- 08_nn_graph.ipynb  # 第八章，nn.Graph模块的使用
|-- data    # 数据集
|-- img    # 存放文本中用于展示的图片
|-- checkpoints    # 保存实验状态，可以在该点恢复训练
|-- GraphMobileNetV2 # 保存模型最新的状态（第八章）
|-- model  #保存Module对象的参数（第七章）
 
```
###更多内容

关于OneFlow 的更多信息：[详情请点击](docs.oneflow.org)。  