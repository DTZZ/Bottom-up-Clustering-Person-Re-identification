from __future__ import print_function, absolute_import
from reid.bottom_up import *
from reid import datasets
from reid import models
import numpy as np
import argparse
import os, sys, time
from reid.utils.logging import Logger
import os.path as osp
from torch.backends import cudnn
import warnings
from warnings import simplefilter

simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore")

def main(args):
    simplefilter(action='ignore', category=FutureWarning)
    warnings.filterwarnings("ignore")
    cudnn.benchmark = True      # 让内置的 cuDNN 的 auto-tuner 自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题
    cudnn.enabled = True
    
    save_path = args.logs_dir   # 日志存储路径
    sys.stdout = Logger(osp.join(args.logs_dir, 'log'+ str(args.merge_percent) + '.txt'))       # 调用logger()

    # get all unlabeled data for training   取出所有数据
    dataset_all = datasets.create(args.dataset, osp.join(args.data_dir, args.dataset))          # 调用datasets.create()
    new_train_data, cluster_id_labels = change_to_unlabel(dataset_all)                          # 调用change_to_unlabel()

    num_train_ids = len(np.unique(np.array(cluster_id_labels)))             # 训练集数量
    nums_to_merge = int(num_train_ids * args.merge_percent)                 # 合并的训练集数量

    BuMain = Bottom_up(model_name=args.arch, batch_size=args.batch_size,    # 调用Bottom_up()
            num_classes=num_train_ids,
            dataset=dataset_all,
            u_data=new_train_data, save_path=args.logs_dir, max_frames=args.max_frames,
            embeding_fea_size=args.fea)
    # num_classes = 训练集数量   dataset = 调用datasets.create()返回值    u_data = change_to_unlabel(dataset_all)返回值
    # save_path = os.path.join(working_dir,'logs')  max_frames = 900    embeding_fea_size = 2048
    for step in range(int(1/args.merge_percent)-1):     # merge_percent = 0.05  range(19)
        print('step: ',step)

        BuMain.train(new_train_data, step, loss=args.loss)      # 调用BuMain.train() change_to_unlabel(dataset_all)返回new_train_data
                                                                # step为循环次数 loss = ExLoss(exclusive loss 排他性损失)
        BuMain.evaluate(dataset_all.query, dataset_all.gallery) # 调用Bumain.evaluate() datasets.create()返回datasets_all
                                                                # dataset_all.query dataset_all.gallery均为[]
        # get new train data for the next iteration 获取下一次迭代的训练数据
        print('----------------------------------------bottom-up clustering------------------------------------------------')
        cluster_id_labels, new_train_data = BuMain.get_new_train_data(cluster_id_labels, nums_to_merge, size_penalty=args.size_penalty)
        print('\n\n')
        # 赋予 cluster_id_labels, new_train_data新的值


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='bottom-up clustering')
    parser.add_argument('-d', '--dataset', type=str, default='market1501',  # 数据集
                        choices=datasets.names())                           # 'mars', 'DukeMTMC-VideoReID', 'market1501', 'duke'
    parser.add_argument('-b', '--batch-size', type=int, default=16)         # batch size
    parser.add_argument('-f', '--fea', type=int, default=2048)              #
    parser.add_argument('-a', '--arch', type=str, default='avg_pool',choices=models.names())        #
    working_dir = os.path.dirname(os.path.abspath(__file__))                # 当前文件的绝对路径
    parser.add_argument('--data_dir', type=str, metavar='PATH',             # 数据目录
                        default=os.path.join(working_dir,'data'))
    parser.add_argument('--logs_dir', type=str, metavar='PATH',             # 日志目录
                        default=os.path.join(working_dir,'logs'))
    parser.add_argument('--max_frames', type=int, default=900)              # 最大框架
    parser.add_argument('--loss', type=str, default='ExLoss')               # 损失
    parser.add_argument('-m', '--momentum', type=float, default=0.5)        # 动量
    parser.add_argument('-s', '--step_size', type=int, default=55)          # 步长
    parser.add_argument('--size_penalty',type=float, default=0.005)         # （步长）惩罚
    parser.add_argument('-mp', '--merge_percent',type=float, default=0.05)  # 合并比例
    main(parser.parse_args())

