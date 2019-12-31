from __future__ import print_function

import os
import numpy as np
import random
import math
from skimage import io
import cv2

import torch
import torch.utils.data as data
#from torch.utils.serialization import load_lua
import torchfile

# from utils.utils import *
from utils.imutils import *
from utils.transforms import *
from datasets.W300LP import W300LP
from .pts_loader import load

class VW300(W300LP):

    def __init__(self, args, split):
        self.nParts = 68
        self.pointType = args.pointType
        # self.anno = anno
        self.img_folder = args.data
        self.base_dir = os.path.join(self.img_folder[:4],'300W')
        self.split = split
        self.is_train = True if self.split == 'train' else False
        self.anno = self._getDataFaces(self.is_train)
        self.total = len(self.anno)
        self.scale_factor = args.scale_factor
        self.rot_factor = args.rot_factor
        self.mean, self.std = self._comput_mean()

    def _getDataFaces(self, is_train):
        base_dir = os.path.join(self.img_folder[:4],'300W')
        dirs = os.listdir(base_dir)
        total = []
        for d in dirs:
            if d == '300W':
                total1 = []
                total1_path = []
                _dirs=os.listdir(os.path.join(base_dir, d))
                for _d in _dirs:
                    files1 = [f for f in os.listdir(os.path.join(base_dir+'/'+d, _d)) if f.endswith('.png')]
                    for i in range(len(files1)):
                        total1.append(_d +'/'+files1[i])

                for i in range(len(total1)):
                    A = base_dir+'/'+d+'/'+total1[i]
                    total1_path.extend(A.split(' '))


            elif d == 'afw':
                total2 =[]
                files2 = [f for f in os.listdir(os.path.join(base_dir, d)) if f.endswith('.jpg')]
                for i in range(len(files2)):
                    A = base_dir+'/'+d+'/'+files2[i]
                    total2.extend(A.split(' '))

                total_0 = total2

            elif d == 'helen':
                _dirs=os.listdir(os.path.join(base_dir, d))
                total3 = []
                test_total3 = []
                total3_path = []
                for _d in _dirs:
                    if _d == 'trainset':
                        files3 = [f for f in os.listdir(os.path.join(base_dir+'/'+d, _d)) if f.endswith('.jpg')]
                        for i in range(len(files3)):
                            total3.append('data/300W/helen/'+_d +'/'+files3[i])

                    elif _d == 'testset':
                        test_files3 = [f for f in os.listdir(os.path.join(base_dir+'/'+d, _d)) if f.endswith('.jpg')]
                        for i in range(len(test_files3)):
                            test_total3.append('data/300W/helen/'+_d +'/'+test_files3[i])


                for i in range(len(total3)):
                    A = base_dir+'/'+d+'/'+total3[i]
                    total3_path.extend(A.split(' '))

            elif d == 'ibug':
                total4 = []
                files4 = [f for f in os.listdir(os.path.join(base_dir, d)) if f.endswith('.jpg')]
                for i in range(len(files4)):
                    A = base_dir+'/'+d+'/'+files4[i]
                    total4.extend(A.split(' '))
                

            elif d == 'lfpw':
                _dirs=os.listdir(os.path.join(base_dir, d))
                total5 = []
                test_total5 = []
                total5_path = []
                for _d in _dirs:
                    if _d == 'trainset':
                        files5 = [f for f in os.listdir(os.path.join(base_dir+'/'+d, _d)) if f.endswith('.png')]
                        for i in range(len(files5)):
                            total5.append('data/300W/lfpw/'+_d +'/'+files5[i])

                    elif _d == 'testset':
                        test_files5 = [f for f in os.listdir(os.path.join(base_dir+'/'+d, _d)) if f.endswith('.png')]
                        for i in range(len(test_files5)):
                            test_total5.append('data/300W/lfpw/'+_d +'/'+test_files5[i])


                for i in range(len(total5)):
                    A = base_dir+'/'+d+'/'+total5[i]
                    total5_path.extend(A.split(' '))

            
        _train = total3 + total5 + total_0 + total4 + test_total5 + test_total3
        _total = len(_train)
        num_training = _total * 0.9

        lines = _train[:int(num_training)]
        #vallines = _train[int(num_training):]
        vallines = test_total3
        #print(vallines)
        #print('1')

        if is_train:
            print('=> loaded train set, {} images were found'.format(len(lines)))
            return lines
        else:
            print('=> loaded validation set, {} images were found'.format(len(vallines)))
            return vallines

    def __len__(self):
        return self.total

    def __getitem__(self, index):
        inp, out, pts, c, s, reference_scale = self.generateSampleFace(index)
        self.pts, self.c, self.s, self.reference_scale = pts, c, s, reference_scale
        if self.is_train:
            return inp, out
        else:
            meta = {'index': index, 'center': c, 'scale': s, 'pts': pts, 'reference_scale': reference_scale}
            return inp, out, meta

    def generateSampleFace(self, idx):

        sf = self.scale_factor
        rf = self.rot_factor

        imagepath = os.path.join(self.anno[idx])
        img = load_image(imagepath)

        main_pts = load(self.anno[idx][:-3] + 'pts')
        #pts = main_pts[0] if self.pointType == '2D' else main_pts[1]
        #pts = np.array(main_pts)

        height = img.size(1)
        width = img.size(2)
        hw = max(width, height)
        #consider the landmarks are mainly distributed in the lower half face, so it needs move the center to some lower along height direction
        c = torch.FloatTensor(( float(width*1.0/2), float(height*1.0/2 + height*0.12) ))
        reference_scale = torch.tensor(200.0)
        #print('1')
        #we hope face for train to be larger than in raw image, so edge will be short by 0.8 ration
        scale_x = hw*0.71 / reference_scale #0.8 common set = 3,6. #0.5 challenge set 8.69, #0.6 for fullset 3.9
        scale_y = hw*0.71 / reference_scale #0.71 for LFPW,
        s = torch.FloatTensor(( scale_x, scale_y ))
        r = 0
        pts = np.asarray(main_pts)
        #print('3')

        if self.is_train:
            s[0] = s[0] * torch.randn(1).mul_(sf).add_(1).clamp(1 - sf, 1 + sf)[0]
            s[1] = s[0]
            r = torch.randn(1).mul_(rf).clamp(-2 * rf, 2 * rf)[0] if random.random() <= 0.6 else 0

            if random.random() <= 0.5:
                img = torch.from_numpy(fliplr(img.numpy())).float()
                pts = shufflelr(pts, width=img.size(2), dataset='w300lp')
                c[0] = img.size(2) - c[0]

            img[0, :, :].mul_(random.uniform(0.7, 1.3)).clamp_(0, 1)
            img[1, :, :].mul_(random.uniform(0.7, 1.3)).clamp_(0, 1)
            img[2, :, :].mul_(random.uniform(0.7, 1.3)).clamp_(0, 1)
        
        inp = crop(img, c, s, reference_scale, [256, 256], rot=r) 
        pts = to_torch(pts)
        tpts = pts.clone()
        
        S = 64
        out = torch.zeros(self.nParts, S, S)
        for i in range(self.nParts):
            if tpts[i, 0] > 0:
                tpts[i, 0:2] = to_torch(transform(tpts[i, 0:2], c, s, reference_scale, [S, S], rot=r))
                out[i] = draw_labelmap(out[i], tpts[i], sigma=1)

        return inp, out, pts, c, s, reference_scale

    def _comput_mean(self):
        meanstd_file = './data/300W_LP/mean.pth.tar'
        if os.path.isfile(meanstd_file):
            ms = torch.load(meanstd_file)
        else:
            print("\tcomputing mean and std for the first time, it may takes a while, drink a cup of coffe...")
            mean = torch.zeros(3)
            std = torch.zeros(3)
            print(self.img_folder)
            if self.is_train:
                for i in range(self.total):
                    a = self.anno[i]
                    img_path = os.path.join(self.img_folder, self.anno[i].split('_')[0],
                                            self.anno[i][:-8] + '.jpg')
                    img = load_image(img_path)
                    mean += img.view(img.size(0), -1).mean(1)
                    std += img.view(img.size(0), -1).std(1)

            mean /= self.total
            std /= self.total
            ms = {
                'mean': mean,
                'std': std,
            }
            torch.save(ms, meanstd_file)
        if self.is_train:
            print('\tMean: %.4f, %.4f, %.4f' % (ms['mean'][0], ms['mean'][1], ms['mean'][2]))
            print('\tStd:  %.4f, %.4f, %.4f' % (ms['std'][0], ms['std'][1], ms['std'][2]))
        return ms['mean'], ms['std']
