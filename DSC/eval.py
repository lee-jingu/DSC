import os
import sys
import time
import matplotlib
matplotlib.use('Agg')
from progress.bar import Bar
import matplotlib.pyplot as plt

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torchvision.datasets as datasets

import datetime
import models
from utils import *
from datasets import W300LP, VW300, AFLW2000, LS3DW
from utils.logger import Logger, savefig
from utils.imutils import batch_with_heatmap
from utils.evaluation import accuracy, AverageMeter, final_preds, calc_metrics, calc_dists
from utils.misc import adjust_learning_rate, save_checkpoint, save_pred
import opts
from models.fcDensenet import FCDenseNet, load_kernel
from models.fcn8s import FCN8s 
import warnings

#from models.FCDensenet import FCDenseNet
#from models.tiramisu import FCDenseNet57
from models.DCN import FCDenseNet57, FCDenseNet67, FCDenseNet103

warnings.filterwarnings('ignore')
args = opts.argparser()
model_names = sorted(
    name for name in models.__dict__
    if not name.startswith("__") and callable(models.__dict__[name]))
# torch.setdefaulttensortype('torch.FloatTensor')

best_acc = 0.
best_auc = 0.
idx = range(1, 69, 1)


def validate(loader, model, criterion, netType, save_path,flip):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acces = AverageMeter()
    end = time.time()

    # predictions
    predictions = torch.Tensor(loader.dataset.__len__(), 68, 2)

    model.eval()
    gt_win, pred_win = None, None
    bar = Bar('Validating', max=len(loader))
    all_dists = torch.zeros((68, loader.dataset.__len__()))
    for i, (inputs, target, meta) in enumerate(loader):
        data_time.update(time.time() - end)

        input_var = torch.autograd.Variable(inputs.cuda())
        target_var = torch.autograd.Variable(target.cuda(async=True))

        output = model(input_var)
        score_map = output[-1].data.cpu()

        if flip:
            flip_input_var = torch.autograd.Variable(
                torch.from_numpy(shufflelr(inputs.clone().numpy())).float().cuda())
            flip_output_var = model(flip_input_var)
            flip_output = flip_back(flip_output_var[-1].data.cpu())
            score_map += flip_output

        # intermediate supervision
        loss = 0
        for o in output:
            loss += criterion(o, target_var)
        acc, batch_dists = accuracy(score_map, target.cpu(), idx, thr=0.07)
        all_dists[:, i * args.val_batch:(i + 1) * args.val_batch] = batch_dists

        #preds = final_preds(score_map, meta['center'], meta['scale'], meta['reference_scale'], [64, 64])
        pts, pts_img = get_preds_fromhm(score_map, meta['center'], meta['scale'], meta['reference_scale'])
        preds = pts_img
        for n in range(score_map.size(0)):
            predictions[meta['index'][n], :, :] = preds[n, :, :]

        losses.update(loss.item(), inputs.size(0))
        acces.update(acc[0], inputs.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        bar.suffix = '({batch}/{size}) Data: {data:.6f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | Acc: {acc: .4f}'.format(
            batch=i + 1,
            size=len(loader),
            data=data_time.val,
            bt=batch_time.val,
            total=bar.elapsed_td,
            eta=bar.eta_td,
            loss=losses.avg,
            acc=acces.avg)
        bar.next()

    bar.finish()
    mean_error = torch.mean(all_dists)
    auc = calc_metrics(all_dists,save_path) # this is auc of predicted maps and target.
    print("=> Mean Error: {:.2f}, AUC@0.07: {} based on maps".format(mean_error*100., auc))
    sys.stdout.flush()
    return losses.avg, acces.avg, predictions, auc


def get_loader(data):
    return {
        '300W_LP': W300LP,
        'LS3D-W/300VW-3D': VW300,
        'AFLW2000': AFLW2000,
        'LS3D-W': LS3DW,
    }[data[5:]]

if __name__ == '__main__':

	args = opts.argparser()
	device ='cuda'

	modelfilename = 'checkpoint/Test_image/Test/model_best.pth.tar'
	save_path = 'checkpoint/300W/challenge'


	if 'cuda' in device:
		torch.backends.cudnn.benchmark = True


	
	face_alignment_net = nn.DataParallel(FCDenseNet103(n_classes=68))
	#modelfilename = 'checkpoint/300W_LP/SDC-Regress/checkpoint_40.pth.tar'
	model_path = modelfilename
	fan_weights = torch.load(model_path,map_location=lambda storage,loc: storage)
		
	face_alignment_net.load_state_dict(fan_weights['state_dict']) #Structure of the model
	face_alignment_net = face_alignment_net.module

	
	model = face_alignment_net
	criterion = torch.nn.MSELoss(size_average=True).cuda()
	Loader = get_loader(args.data)
	val_loader = torch.utils.data.DataLoader(
		Loader(args,'A'),
		batch_size=args.val_batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True)

	validate(val_loader, model, criterion, args.netType, save_path ,flip = False)