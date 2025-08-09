import torch
from torch.utils import data
from torchvision import transforms
import os
import sys
import time
import pickle
import glob
import csv
import pandas as pd
import numpy as np
import cv2

sys.path.append('../Utils')
from Utils.augmentation import *
from tqdm import tqdm
from joblib import Parallel, delayed


def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


class UCF101_3d(data.Dataset):
    def __init__(self,
                 mode='train',
                 transform=None,
                 num_seq=8,
                 downsample=3,
                 epsilon=5,
                 which_split=1,
                 return_label=False):
        self.mode = mode
        self.transform = transform
        self.num_seq = num_seq
        self.downsample = downsample
        self.epsilon = epsilon
        self.which_split = which_split
        self.return_label = return_label
        print('1')

        # 根据模式加载分割文件
        if mode == 'train':
            split = '../ProcessData/data/ucf101/train_split%02d.csv' % self.which_split
            video_info = pd.read_csv(split, header=None)
            video_info.dropna(axis=0, how='any', inplace=True)
            # print(video_info)
            # print(len(video_info))
        elif mode in ['val', 'test']:  # 对于 val 和 test 使用同一个分割文件
            split = '../ProcessData/data/ucf101/test_split%02d.csv' % self.which_split
            video_info = pd.read_csv(split, header=None)
            video_info.dropna(axis=0, how='any', inplace=True)
        else:
            raise ValueError('wrong mode')

        # 读取动作类别列表
        self.action_dict_encode = {}
        self.action_dict_decode = {}
        action_file = os.path.join('../ProcessData/data/ucf101', 'classInd.txt')
        action_df = pd.read_csv(action_file, sep=' ', header=None)
        for _, row in action_df.iterrows():
            act_id, act_name = row
            self.action_dict_decode[act_id] = act_name
            self.action_dict_encode[act_name] = act_id

        # 筛选掉过短视频
        drop_idx = []
        for idx, row in video_info.iterrows():
            vpath, vlen = row
            # print("vlen",vlen)
            # print('row',row)
            # print('self.num_seq * self.downsample',self.num_seq * self.downsample)
            if vlen - self.num_seq * self.downsample <= 0:
                drop_idx.append(idx)
        video_info = video_info.drop(drop_idx, axis=0)

        # 若为验证模式，随机采样部分数据
        if mode == 'val':
            video_info = video_info.sample(frac=0.3)

        # 如果需要返回标签，则在初始化时计算 label
        if self.return_label:
            labels = []
            for idx, row in video_info.iterrows():
                vpath, _ = row
                try:
                    # 尝试从倒数第三个路径分量获取动作名称
                    vname = vpath.split('/')[-3]
                except Exception as e:
                    # 若失败则从倒数第二个分量获取
                    vname = vpath.split('/')[-2]
                label = self.encode_action(vname)
                labels.append(label)
            video_info['label'] = labels

        self.video_info = video_info.reset_index(drop=True)

    def idx_sampler(self, vlen):
        '''Sample frame indices from a video'''
        if vlen - self.num_seq * self.downsample <= 0:
            return [None]
        n = 1
        start_idx = np.random.choice(range(int(vlen) - self.num_seq * self.downsample), n)
        # 生成形状为 (num_seq, 1) 的索引数组
        seq_idx = np.expand_dims(np.arange(self.num_seq), -1) * self.downsample + start_idx
        return seq_idx

    def __getitem__(self, index):
        # 根据是否返回 label 选择读取的内容
        if self.return_label:
            vpath, vlen, label = self.video_info.iloc[index]
        else:
            vpath, vlen = self.video_info.iloc[index]
        seq_idx = self.idx_sampler(vlen)
        if seq_idx is None:
            print(vpath)
        # 保证索引形状为 (num_seq, 1)
        assert seq_idx.shape == (self.num_seq, 1)
        seq_idx = seq_idx.reshape(self.num_seq)

        # 载入序列帧图像
        seq = [pil_loader(os.path.join(vpath, 'image_%05d.jpg' % (i + 1))) for i in seq_idx]
        t_seq = self.transform(seq)  # 对序列应用统一变换

        (C, H, W) = t_seq[0].size()
        t_seq = torch.stack(t_seq, 0)
        t_seq = t_seq.view(self.num_seq, C, H, W)

        if self.return_label:
            label_tensor = torch.LongTensor([label])
            return t_seq, label_tensor
        return t_seq

    def __len__(self):
        return len(self.video_info)

    def encode_action(self, action_name):
        '''Given an action name, return its action code'''
        return self.action_dict_encode[action_name]

    def decode_action(self, action_code):
        '''Given an action code, return the corresponding action name'''
        return self.action_dict_decode[action_code]

