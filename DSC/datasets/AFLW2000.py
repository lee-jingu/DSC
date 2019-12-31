from __future__ import print_function

import os
import numpy as np
import os.path as osp
import random
import math
from skimage import io
from scipy import io as sio

import torch
import torch.utils.data as data
#from torch.utils.serialization import load_lua
import torchfile

from utils.imutils import *
from utils.transforms1 import *

from datasets.W300LP import W300LP
#from datasets.SW300W import SW300W

def _load(fp):
    suffix = _get_suffix(fp)
    if suffix == 'npy':
        return np.load(fp)
    elif suffix == 'pkl':
        return pickle.load(open(fp, 'rb'))


def _get_suffix(filename):
    """a.jpg -> jpg"""
    pos = filename.rfind('.')
    if pos == -1:
        return ''
    return filename[pos + 1:]

class AFLW2000(W300LP):
    def __init__(self, args, split):
        super(AFLW2000, self).__init__(args, split)
        #self.is_train = False
        assert self.pointType == '3D', "AFLW2000 provided only 68 3D points"

    def _getDataFaces(self, is_train):
        base_dir = self.img_folder
        lines = []
        files = [f for f in os.listdir(base_dir) if f.endswith('.mat')]
        for f in files:
            lines.append(os.path.join(base_dir, f))
        #print('=> loaded AFLW2000 set, {} images were found'.format(len(lines)))
        #return sorted(lines)
        if is_train:
            print('=> loaded train set, {} images were found'.format(len(lines)))
            return sorted(lines)
        else:
            print('=> loaded validation set, {} images were found'.format(len(lines)))
            return sorted(lines)

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
        reference_scale = torch.tensor(200.0)
        pts68_all_re = _load(osp.join('data', 'AFLW2000-3D-Reannotated.pts68.npy'))
        pts = pts68_all_re[idx]
        pts = pts[0:2, :].transpose()
        pts = torch.from_numpy(pts)
        '''
        main_pts = sio.loadmat(self.anno[idx])
        pts = main_pts['pt3d_68'][0:2, :].transpose()
        pts = torch.from_numpy(pts)
        '''
        '''
        mins_ = torch.min(pts, 0)[0].view(2) # min vals
        maxs_ = torch.max(pts, 0)[0].view(2) # max vals
        c = torch.FloatTensor((maxs_[0]-(maxs_[0]-mins_[0])/2, maxs_[1]-(maxs_[1]-mins_[1])/2))
        print(c[1], type(c[1]))
        c[1] -= ((maxs_[1]-mins_[1]) * 0.12)

        s = (maxs_[0]-mins_[0]+maxs_[1]-mins_[1])/195

        img = load_image(self.anno[idx][:-4] + '.jpg')
        s = torch.FloatTensor(( s, s))
        #print('s', s[0])
        r = 0
        '''
        r = 0
        img = load_image(self.anno[idx][:-4] + '.jpg')
        height = img.size(1)
        width = img.size(2)
        hw = max(width, height)
        c = torch.FloatTensor(( float(width*1.0/2), float(height*1.0/2 + height*0.12) ))
        reference_scale = torch.tensor(200.0)
        #we hope face for train to be larger than in raw image, so edge will be short by 0.8 ration
        scale_x = hw*0.47 / reference_scale
        scale_y = hw*0.47 / reference_scale
        s = torch.FloatTensor(( scale_x, scale_y ))
        

        if self.is_train:
            s[0] = s[0] * torch.randn(1).mul_(sf).add_(1).clamp(1 - sf, 1 + sf)[0]
            s[1] = s[0]
            r = torch.randn(1).mul_(rf).clamp(-2 * rf, 2 * rf)[0] if random.random() <= 0.6 else 0

            if random.random() <= 0.5:
                #img = torch.from_numpy(fliplr(img.numpy())).float()
                #pts = shufflelr(pts, width=img.size(2), dataset='aflw2000')
                c[0] = img.size(2) - c[0]

            img[0, :, :].mul_(random.uniform(0.7, 1.3)).clamp_(0, 1)
            img[1, :, :].mul_(random.uniform(0.7, 1.3)).clamp_(0, 1)
            img[2, :, :].mul_(random.uniform(0.7, 1.3)).clamp_(0, 1)


        inp = crop(img, c, s, reference_scale, [256, 256], rot=r) 
        #inp = color_normalize(inp, self.mean, self.std)
        S = 64
        tpts = pts.clone()
        out = torch.zeros(self.nParts, 64, 64)
        for i in range(self.nParts):
            if tpts[i, 0] > 0:
                tpts[i, 0:2] = to_torch(transform(tpts[i, 0:2], c, s, reference_scale, [S, S], rot=r))
                out[i] = draw_labelmap(out[i], tpts[i] - 1, sigma=1)

        return inp, out, pts, c, s, reference_scale