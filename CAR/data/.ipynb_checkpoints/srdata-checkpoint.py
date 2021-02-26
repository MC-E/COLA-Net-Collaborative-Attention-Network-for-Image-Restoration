import os
import glob
import random
import pickle

from data import common

import numpy as np
import imageio
import torch
import torch.utils.data as data
import cv2


class SRData(data.Dataset):
    def __init__(self, args, name='', train=True, benchmark=False):
        self.args = args
        self.name = name
        self.train = train
        self.split = 'train' if train else 'test'
        self.do_eval = True
        self.benchmark = benchmark
        self.input_large = (args.model == 'VDSR')
        self.scale = args.scale
        self.idx_scale = 0

        self._set_filesystem(args.dir_data)
        if args.ext.find('img') < 0:
            path_bin = os.path.join(self.apath, 'bin')
            os.makedirs(path_bin, exist_ok=True)

        list_hr, list_lr = self._scan()
        if args.ext.find('img') >= 0 or benchmark:
            self.images_hr, self.images_lr = list_hr, list_lr
        elif args.ext.find('sep') >= 0:
            # os.makedirs(
            #     self.dir_hr.replace(self.apath, path_bin),
            #     exist_ok=True
            # )
            # for s in self.scale:
            #     os.makedirs(
            #         os.path.join(
            #             self.dir_lr.replace(self.apath, path_bin),
            #             'X{}'.format(s)
            #         ),
            #         exist_ok=True
            #     )
            #
            # self.images_hr, self.images_lr = [], [[] for _ in self.scale]
            # for h in list_hr:
            #     b = h.replace(self.apath, path_bin)
            #     b = b.replace(self.ext[0], '.pt')
            #     self.images_hr.append(b)
            #     self._check_and_load(args.ext, h, b, verbose=True)
            # print(len(list_lr))
            # for i, ll in enumerate(list_lr):
            #     for l in ll:
            #         print(len(ll))
            #         print(ll)
            #         exit(0)
            #         b = l.replace(self.apath, path_bin)
            #         b = b.replace(self.ext[1], '.pt')
            #         self.images_lr[i].append(b)
            #         self._check_and_load(args.ext, l, b, verbose=True)
            # print(self.ext)
            # exit(0)
            self.images_hr, self.images_lr = self._scan()
            if args.ext.find('reset') >= 0:
                print('Preparing seperated binary files')
                for v in self.images_hr:
                    hr = cv2.imread(v).astype(np.float)
                    name_sep = v.replace(self.ext, '.npy')
                    np.save(name_sep, hr)
                    print(name_sep)
                # for si, s in enumerate(self.noise_level):
                for v in self.images_lr:
                    lr = cv2.imread(v).astype(np.float)
                    name_sep = v.replace(self.ext, '.npy')
                    np.save(name_sep, lr)
                    print(name_sep)

            self.images_hr = [
                v.replace(self.ext, '.npy') for v in self.images_hr
            ]

            self.images_lr = [
                v.replace(self.ext, '.npy') for v in self.images_lr
            ]
        if train:
            n_patches = args.batch_size * args.test_every
            n_images = len(args.data_train) * len(self.images_hr)
            if n_images == 0:
                self.repeat = 0
            else:
                self.repeat = max(n_patches // n_images, 1)#//2

    # Below functions as used to prepare images
    def _scan(self):
        # print(self.dir_lr)
        # print(self.dir_hr)
        name_hr = []
        name_lr = []
        list_hr = os.listdir(self.dir_hr)
        list_lr = os.listdir(self.dir_lr)
        #print(len(list_hr),len(list_lr))

        for i in list_hr:
            if i.endswith('.png') or i.endswith('jpeg'):
                name_hr.append(os.path.join(self.dir_hr,i))
        for j in list_lr:
            if j.endswith('.png') or j.endswith('jpeg'):
                name_lr.append(os.path.join(self.dir_lr,j.replace('.jpeg','.png')))
        name_hr.sort()
        name_lr.sort()
        #print(len(name_lr),len(name_hr))
        return name_hr, name_lr

    def _set_filesystem(self, dir_data):
        self.apath = os.path.join(dir_data, self.name)
        self.dir_hr = os.path.join(self.apath, 'HR')
        self.dir_lr = os.path.join(self.apath, 'LR_bicubic')
        if self.input_large: self.dir_lr += 'L'
        self.ext = '.png'

    def _check_and_load(self, ext, img, f, verbose=True):
        if not os.path.isfile(f) or ext.find('reset') >= 0:
            if verbose:
                print('Making a binary: {}'.format(f))
            with open(f, 'wb') as _f:
                pickle.dump(imageio.imread(img), _f)

    def __getitem__(self, idx):
        # lr, hr, filename = self._load_file(idx)
        # pair = self.get_patch(lr, hr)
        # pair = common.set_channel(*pair, n_channels=self.args.n_colors)
        # pair_t = common.np2Tensor(*pair, rgb_range=self.args.rgb_range)
        #
        # return pair_t[0], pair_t[1], filename
        lr, hr, filename = self._load_file(idx)
        # print('1',lr.shape, np.max(lr))
        # print(lr,hr)
        lr = lr[:,:,0]
        lr, hr = self.get_patch(lr, hr)
        lr = np.expand_dims(lr,2)
        # print('2',lr.shape, np.max(lr))

        lr, hr = common.set_channel([lr, hr], self.args.n_colors)
        #lr, hr = self.get_patch(lr,hr)
        # print(lr,hr)
        # print('3',lr.shape, np.max(lr))
        #print(lr.shape,hr.shape)
        #exit(0)
        lr_tensor, hr_tensor = common.np2Tensor([lr, hr], self.args.rgb_range)
        # print(lr_tensor,hr_tensor)
        # exit(0)
        return lr_tensor, hr_tensor, filename

    def __len__(self):
        if self.train:                                 # method of repeat
            return len(self.images_hr) * self.repeat
        else:
            return len(self.images_hr)
        # return len(self.images_hr)

    def _get_index(self, idx):
        if self.train:                               # method of repeat
            return idx % len(self.images_hr)
        else:
            return idx
        # return idx

    def _load_file(self, idx):
        # print(idx)
        idx = self._get_index(idx)
        # print(idx)
        f_hr = self.images_hr[idx]#.sort()
        f_lr = self.images_lr[idx].replace('.png','.jpeg')#.sort()#[self.idx_scale][idx]
        #print(f_lr,f_hr)
        #exit(0)

        # f_hr = f_lr.replace('/Val','').replace('jpeg','png').replace('LQ','HQ').replace('/25','')
        # print(f_lr,f_hr)

        filename, _ = os.path.splitext(os.path.basename(f_hr))
        if self.args.ext == 'img' or self.benchmark:
            hr = imageio.imread(f_hr)#[:,:,0]
            if len(hr.shape)==3:
                hr = hr[:,:,0]
            lr = imageio.imread(f_lr)
            # print(hr.shape,lr.shape)
            # exit(0)
        elif self.args.ext.find('sep') >= 0:
            lr = np.load(f_lr)
            hr = np.load(f_hr)
            # with open(f_hr, 'rb') as _f:
            #     hr = pickle.load(_f)
            # with open(f_lr, 'rb') as _f:
            #     lr = pickle.load(_f)

        return lr, hr, filename

    def get_patch(self, lr, hr):
        patch_size = self.args.patch_size
        scale = self.scale[self.idx_scale]
        multi_scale = len(self.scale) > 1
        if self.train:
            lr, hr = common.get_patch(
                lr, hr, patch_size, scale, multi_scale=multi_scale
            )
            lr, hr = common.augment([lr, hr])
            # print('Yes')
            lr = common.add_noise(lr, self.args.noise)
        else:
            ih, iw = lr.shape[0:2]
            hr = hr[0:ih * scale, 0:iw * scale]

        return lr, hr

    def set_scale(self, idx_scale):
        if not self.input_large:
            self.idx_scale = idx_scale
        else:
            self.idx_scale = random.randint(0, len(self.scale) - 1)
