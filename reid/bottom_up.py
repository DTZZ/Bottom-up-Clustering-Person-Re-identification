import torch
from torch import nn
from reid import models
from reid.trainers import Trainer
from reid.evaluators import extract_features, Evaluator
from reid.dist_metric import DistanceMetric
import numpy as np
from collections import OrderedDict
import os.path as osp
import pickle
import copy, sys
from reid.utils.serialization import load_checkpoint
from reid.utils.data import transforms as T
from torch.utils.data import DataLoader
from reid.utils.data.preprocessor import Preprocessor
import random
import pickle as pkl
from reid.exclusive_loss import ExLoss


class Bottom_up():
    # 初始化参数
    def __init__(self, model_name, batch_size, num_classes, dataset, u_data, save_path, embeding_fea_size=1024,
                 dropout=0.5, max_frames=900, initial_steps=20, step_size=16):

        self.model_name = model_name            # model_name = 'avg_pool'
        self.num_classes = num_classes          # 训练集数量
        self.data_dir = dataset.images_dir      # data_dir = dataset_all.image_dir
        self.is_video = dataset.is_video        # is_video = dataset_all.is_video
        self.save_path = save_path              # save_path = os.path.join(working_dir,'logs') = D:/GitHub/BUC/logs

        self.dataset = dataset                  # 'market1501'
        self.u_data = u_data                    # u_data = change_to_unlabel(dataset_all)返回值
        self.u_label = np.array([label for _, label, _, _ in u_data])
                                                # _为不重要的变量的占位符  取出u_data中的label作为u_label
        self.dataloader_params = {}
        self.dataloader_params['height'] = 256
        self.dataloader_params['width'] = 128
        self.dataloader_params['batch_size'] = batch_size
        self.dataloader_params['workers'] = 6

        self.batch_size = batch_size            # batch_size = 16
        self.data_height = 256
        self.data_width = 128
        self.data_workers = 6

        self.initial_steps = initial_steps      # initial_steps = 20
        self.step_size = step_size              # step_size = 16

        # batch size for eval mode. Default is 1.
        self.dropout = dropout                  # dropout = 0.5
        self.max_frames = max_frames            # max_frames = 900
        self.embeding_fea_size = embeding_fea_size      # embedding_fea_size = 1024

        if self.is_video:
            self.eval_bs = 1
            self.fixed_layer = True
            self.frames_per_video = 16
            self.later_steps = 5
        else:
            self.eval_bs = 64                   # eval_bs:evaluators_batchsize
            self.fixed_layer = False
            self.frames_per_video = 1           # 图片
            self.later_steps = 2                # 后面的步骤数为2

        model = models.create(self.model_name, dropout=self.dropout,    # model_name = 'avg_pool', dropout = 0.5
                              embeding_fea_size=self.embeding_fea_size, fixed_layer=self.fixed_layer)       # embedding_fea_size = 1024  # fixed_layer = False
        self.model = nn.DataParallel(model).cuda()

        self.criterion = ExLoss(self.embeding_fea_size, self.num_classes, t=10).cuda()
        # 调用ExLoss()    embeding_fea_size = 1024, num_classes = 训练集数量
    # 数据读取
    def get_dataloader(self, dataset, training=False):
        normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        if training:                                            # import transforms as T
            transformer = T.Compose([
                T.RandomSizedRectCrop(self.data_height, self.data_width),   # data_height = 256 data_width = 128
                T.RandomHorizontalFlip(),                       # 随机水平翻转
                T.ToTensor(),                                   # Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.
                normalizer,                                     # normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                                                                #                          std=[0.229, 0.224, 0.225])
            ])
            batch_size = self.batch_size                        # batch_size = 16
        else:
            transformer = T.Compose([
                T.RectScale(self.data_height, self.data_width), # RectScale():三角缩放(?)
                T.ToTensor(),                                   # Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.
                normalizer,                                     # normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
            ])                                                  #                           std=[0.229, 0.224, 0.225])
            batch_size = self.eval_bs                           # batch_size = 64
        data_dir = self.data_dir                                # data_dir = dataset_all.image_dir

        data_loader = DataLoader(                               # DataLoader()
            Preprocessor(dataset, root=data_dir, num_samples=self.frames_per_video,                 # root = dataset_all.image_dir  num_samples = 1
                         transform=transformer, is_training=training, max_frames=self.max_frames),  # transform = T.compose()返回值    is_training = False max_frames = 900
            batch_size=batch_size, num_workers=self.data_workers,                                   # batch_size = 16   data_workers = 6
            shuffle=training, pin_memory=True, drop_last=training)                                  # shuffle = True   drop_last = True

        current_status = "Training" if training else "Testing"                                      # current_status = 'Training'
        print("Create dataloader for {} with batch_size {}".format(current_status, batch_size))     # Create dataloader for Training with batch_size 16
        return data_loader
    # 训练
    def train(self, train_data, step, loss, dropout=0.5):               # train_data = BuMain.get_new_train_data()返回值
        # adjust training epochs and learning rate
        epochs = self.initial_steps if step==0 else self.later_steps    # step为0时:epochs = 20   否则epochs = 2
        init_lr = 0.1 if step==0 else 0.01                              # step为0时:lr = 0.1 否则lr = 0.01
        step_size = self.step_size if step==0 else sys.maxsize          # step为0时:step_size = 16 否则step_size = maxsize(系统最大值)

        """ create model and dataloader """
        dataloader = self.get_dataloader(train_data, training=True)     # 调用get_dataloader()获得训练数据

        # the base parameters for the backbone (e.g. ResNet50)            返回参数标签
        base_param_ids = set(map(id, self.model.module.CNN.base.parameters()))

        # we fixed the first three blocks to save GPU memory              过滤参数 得到需要的参数
        base_params_need_for_grad = filter(lambda p: p.requires_grad, self.model.module.CNN.base.parameters())

        # params of the new layers                                        新参数
        new_params = [p for p in self.model.parameters() if id(p) not in base_param_ids]

        # set the learning rate for backbone to be 0.1 times              对于梯度计算需要的参数 lr = 0.1
        param_groups = [                                                # 新参数 lr = 1.0
            {'params': base_params_need_for_grad, 'lr_mult': 0.1},
            {'params': new_params, 'lr_mult': 1.0}]
        # 优化器为SGD   lr = 0.1 or 0.01    momentum = 0.9  weight_decay(权重衰减) = 0.0005 nesterov:momentum的变种
        optimizer = torch.optim.SGD(param_groups, lr=init_lr, momentum=0.9, weight_decay=5e-4, nesterov=True)

        # change the learning rate by step  根据轮次不同更新lr
        def adjust_lr(epoch, step_size):
            lr = init_lr / (10 ** (epoch // step_size))                 # lr = 0.1/(10^(epoch//16)）
            for g in optimizer.param_groups:                            #
                g['lr'] = lr * g.get('lr_mult', 1)

        """ main training process """
        trainer = Trainer(self.model, self.criterion, fixed_layer=self.fixed_layer)     # fixed_layer = False
        for epoch in range(epochs):                                     # epochs = 20 or 2
            adjust_lr(epoch, step_size)                                 # epoch:0~19    step_size = 16
            trainer.train(epoch, dataloader, optimizer, print_freq=max(5, len(dataloader) // 30 * 10))      # 调用trainer.train()
    # 特征提取
    def get_feature(self, dataset):                                         # 返回features和fcs
        dataloader = self.get_dataloader(dataset, training=False)           # 获取数据 training = False
        features, _, fcs = extract_features(self.model, dataloader)         # 调用extract_features()得到features和fcs
        features = np.array([logit.numpy() for logit in features.values()]) # 赋值features
        fcs = np.array([logit.numpy() for logit in fcs.values()])           # 赋值fcs
        return features, fcs                                                # 返回features和fcs
    # 更新内存
    def update_memory(self, weight):
        self.criterion.weight = torch.from_numpy(weight).cuda() # 从numpy到tensor
    # 评估训练结果
    def evaluate(self, query, gallery):                         # 得到测试数据
        test_loader = self.get_dataloader(list(set(query) | set(gallery)), training=False)
        evaluator = Evaluator(self.model)                       # 评估器
        rank1, mAP = evaluator.evaluate(test_loader, query, gallery)    # 得到rank1准确率和mAP
        return rank1, mAP
    # 计算距离
    def calculate_distance(self,u_feas):
        # calculate distance between features   计算特征之间的距离
        x = torch.from_numpy(u_feas)                            # 从numpy转为tensor
        y = x
        m = len(u_feas)                                         # 特征长度
        dists = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, m) + \
                torch.pow(y, 2).sum(dim=1, keepdim=True).expand(m, m).t()
        dists.addmm_(1, -2, x, y.t())                           # 距离
        return dists

    # def select_merge_data(self, u_feas, nums_to_merge, label, label_to_images,  ratio_n,  dists):
    #     #calculate final distance (feature distance + diversity regularization)
    #     tri = np.tri(len(u_feas), dtype=np.float32)
    #     tri = tri * np.iinfo(np.int32).max
    #     tri = tri.astype('float32')
    #     tri = torch.from_numpy(tri)
    #     dists = dists + tri
    #     for idx in range(len(u_feas)):
    #         for j in range(idx + 1, len(u_feas)):
    #             if label[idx] == label[j]:
    #                 dists[idx, j] = np.iinfo(np.int32).max
    #             else:
    #                 dists[idx][j] =  dists[idx][j] + \
    #                                 + ratio_n * ((len(label_to_images[label[idx]])) + (len(label_to_images[label[j]])))
    #     dists = dists.numpy()
    #     ind = np.unravel_index(np.argsort(dists, axis=None), dists.shape)
    #     idx1 = ind[0]
    #     idx2 = ind[1]
    #     return idx1, idx2

    #
    # 选择合并数据
    def select_merge_data(self, u_feas, label, label_to_images,  ratio_n,  dists):   # ratio_n = 0.005   dists:计算得到的距离
        dists.add_(torch.tril(100000 * torch.ones(len(u_feas), len(u_feas))))        # 距离加上100000

        cnt = torch.FloatTensor([len(label_to_images[label[idx]]) for idx in range(len(u_feas))])   # (参数有问题)
        dists += ratio_n * (cnt.view(1, len(cnt)) + cnt.view(len(cnt), 1))           # view():变成1行n列
        
        for idx in range(len(u_feas)):                          # 如果两个标签为同一类 则距离为100000
            for j in range(idx + 1, len(u_feas)):
                if label[idx] == label[j]:
                    dists[idx, j] = 100000

        dists = dists.numpy()                                   # dists转化为numpy
        ind = np.unravel_index(np.argsort(dists, axis=None), dists.shape)   # 将平面索引或平面索引数组转换为坐标数组的元组。
        idx1 = ind[0]                                           # 返回idx1, idx2
        idx2 = ind[1]
        return idx1, idx2


    # 产生新的训练数据
    def generate_new_train_data(self, idx1, idx2, label, num_to_merge):
        correct = 0
        num_before_merge = len(np.unique(np.array(label)))
        # merge clusters with minimum dissimilarity 将不相似度最小的类汇聚为一类
        for i in range(len(idx1)):
            label1 = label[idx1[i]]
            label2 = label[idx2[i]]
            if label1 < label2:
                label = [label1 if x == label2 else x for x in label]
            else:
                label = [label2 if x == label1 else x for x in label]
            if self.u_label[idx1[i]] == self.u_label[idx2[i]]:
                correct += 1
            num_merged =  num_before_merge - len(np.sort(np.unique(np.array(label))))
            if num_merged == num_to_merge:
                break

        # set new label to the new training data 为新的训练数据设置新的标签
        unique_label = np.sort(np.unique(np.array(label)))
        for i in range(len(unique_label)):
            label_now = unique_label[i]
            label = [i if x == label_now else x for x in label]
        new_train_data = []
        for idx, data in enumerate(self.u_data):                #
            new_data = copy.deepcopy(data)                      # copy数据
            new_data[3] = label[idx]                            #
            new_train_data.append(new_data)                     # 得到新的训练数据

        num_after_merge = len(np.unique(np.array(label)))       # 汇聚后的数量
        print("num of label before merge: ", num_before_merge, " after_merge: ", num_after_merge, " sub: ",
              num_before_merge - num_after_merge)               # 汇聚前的数量、汇聚后的数量、差值
        return new_train_data, label                            # 返回新的数据、标签
    # 产生平均特征
    def generate_average_feature(self, labels):
        #extract feature/classifier 提取特征/分类器
        u_feas, fcs = self.get_feature(self.u_data)             # u_feas = features fcs = fcs

        #images of the same cluster
        label_to_images = {}                                    #
        print('bottom_up.py 247行')
        for idx, l in enumerate(labels):                        # idx:数据下标 l:数据
            print(idx, l)
            label_to_images[l] = label_to_images.get(l, []) + [idx]     # (?)

        #calculate average feature/classifier of a cluster  计算一个聚类的平均特征/分类器
        feature_avg = np.zeros((len(label_to_images), len(u_feas[0])))  # 全0的矩阵
        fc_avg = np.zeros((len(label_to_images), len(fcs[0])))          # 全0的矩阵
        for l in label_to_images:
            feas = u_feas[label_to_images[l]]
            feature_avg[l] = np.mean(feas, axis=0)                      # 压缩行，求各列平均值
            fc_avg[l] = np.mean(fcs[label_to_images[l]], axis=0)        # 压缩列，求各行平均值
        return u_feas, feature_avg, label_to_images, fc_avg             # 返回
    # 得到新的训练数据
    def get_new_train_data(self, labels, nums_to_merge, size_penalty):
        u_feas, feature_avg, label_to_images, fc_avg = self.generate_average_feature(labels)
        
        dists = self.calculate_distance(u_feas)
        
        idx1, idx2 = self.select_merge_data(u_feas, labels, label_to_images, size_penalty,dists)
        
        new_train_data, labels = self.generate_new_train_data(idx1, idx2, labels,nums_to_merge)
        
        num_train_ids = len(np.unique(np.array(labels)))

        # change the criterion classifer
        self.criterion = ExLoss(self.embeding_fea_size, num_train_ids, t=10).cuda()
        #new_classifier = fc_avg.astype(np.float32)
        #self.criterion.V = torch.from_numpy(new_classifier).cuda()

        return labels, new_train_data

# 转为无标签数据
def change_to_unlabel(dataset):
    # generate unlabeled set 产生无标签集合
    trimmed_dataset = []                                    # 修剪的数据集
    init_videoid = int(dataset.train[0][3])                 # (?)
    print('bottom_up.py 第284行', dataset.train[0][3])
    for (imgs, pid, camid, videoid) in dataset.train:
        videoid = int(videoid) - init_videoid
        if videoid < 0:
            print(videoid, 'RANGE ERROR')
        assert videoid >= 0
        trimmed_dataset.append([imgs, pid, camid, videoid]) # 修改后的数据集

    index_labels = []
    for idx, data in enumerate(trimmed_dataset):
        data[3] = idx                                       # data[3] is the label of the data array data[3]为数据标签
        index_labels.append(data[3])                        # index加上data[3]
    
    return trimmed_dataset, index_labels
