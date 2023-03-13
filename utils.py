from datetime import datetime
import os
import math
import glob
import shutil
from PIL import Image

import numpy as np
import torch

from pytorch_msssim import ssim_matlab as ssim_pth


def load_dataset(dataset_name, data_root, batch_size, test_batch_size, num_workers, test_mode='extreme'):
    """
        Load dataset for training/evalution for TAIN
        
        Args:
            dataset_name - name of the dataset - [vimeo90k, ucf, snufilm, middlebury]
            data_root - path to the dataset
            batch_size - batch size of the training set
            test_batch_size - batch size of the testing set
            num_workers - number of workers
            test_mode - test mode for SNU-FILM dataset ('snufilm') - [easy, medium, hard, extreme]
            
        Returns:
            train_loader - training data loader
            test_loader - testing data loader
    """
    
    if dataset_name == 'vimeo90k':
        from data.vimeo90k import get_loader
        train_loader = get_loader('train', data_root, batch_size, shuffle=True, num_workers=num_workers)
        test_loader = get_loader('test', data_root, test_batch_size, shuffle=False, num_workers=num_workers)
        
        return train_loader, test_loader
        
    elif dataset_name == 'snufilm':
        from data.snufilm import get_loader
    elif dataset_name == 'middlebury':
        from data.middlebury import get_loader
    elif dataset_name == 'ucf':
        from data.ucf101 import get_loader
    else:
        raise NotImplementedError('Training / Testing for this dataset is not implemented.')
                    
    test_loader = get_loader('test', data_root, test_batch_size, shuffle=False, num_workers=num_workers, test_mode=test_mode)
    return None, test_loader
    


def build_input(images, imgpaths, is_training=True, device=torch.device('cuda')):
    """
        Build input for TAIN
        
        Args:
            images - three consecutive image frames
            imgpaths - corresponding image paths
            is_training - indicator for training or testing
            device - torch.device
            
        Returns:
            im1 - first input image frame
            im2 - second input image frame
            gt - true intermediate frame to predict
    """
    
    if isinstance(images[0], list):
        images_gathered = [None, None, None]
        for j in range(3):
            _images = [images[k][j] for k in range(len(images))]
            images_gathered[j] = torch.cat(_images, 0)
        imgpaths = [p for _ in images for p in imgpaths]
        images = images_gathered
            
    im1, im2 = images[0].to(device), images[2].to(device)
    gt = images[1].to(device)
    
    return im1, im2, gt


def load_checkpoint(args, model, optimizer):
    """Load checkpoint"""
    if args.resume_exp is None:
        args.resume_exp = args.exp_name
    load_name = os.path.join('checkpoint', args.resume_exp, 'model_best.pth')
    print("Loading checkpoint %s..." % load_name)
    
    checkpoint = torch.load(load_name)
    args.start_epoch = checkpoint['epoch'] + 1
    if args.resume_exp != args.exp_name:
        args.start_epoch = 0

    # filter out different keys or those with size mismatch
    model_dict = model.state_dict()
    ckpt_dict = {}
    mismatch = False
    for k, v in checkpoint['state_dict'].items():
        if model_dict[k].size() == v.size():
            ckpt_dict[k] = v
        else:
            print('Size mismatch while loading!   %s != %s   Skipping %s...'
                    % (str(model_dict[k].size()), str(v.size()), k))
            mismatch = True
            
    if len(model.state_dict().keys()) > len(ckpt_dict.keys()):
        mismatch = True
        
    # overwrite parameters to model_dict
    model_dict.update(ckpt_dict)
    
    # load to model
    model.load_state_dict(model_dict)
    
    # if sizes match and resuming experiment, load optimizer
    if (not mismatch) and (optimizer is not None) and (args.resume_exp is not None):
        optimizer.load_state_dict(checkpoint['optimizer'])
        update_lr(optimizer, args.lr)
        
    del checkpoint, ckpt_dict, model_dict


def save_checkpoint(state, is_best, exp_name, filename='checkpoint.pth'):
    """Save checkpoint to disk"""
    directory = "checkpoint/%s/" % (exp_name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'checkpoint/%s/' % (exp_name) + 'model_best.pth')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def init_losses(loss_str):
    loss_specifics = {}
    loss_list = loss_str.split('+')
    for l in loss_list:
        _, loss_type = l.split('*')
        loss_specifics[loss_type] = AverageMeter()
    loss_specifics['total'] = AverageMeter()
    return loss_specifics


def init_meters(loss_str):
    losses = init_losses(loss_str)
    psnrs = AverageMeter()
    ssims = AverageMeter()
    return losses, psnrs, ssims


def init_meters_masked():
    psnrs_masked = AverageMeter()
    ssims_masked = AverageMeter()
    return psnrs_masked, ssims_masked


def quantize(img, rgb_range=255):
    return img.mul(255 / rgb_range).clamp(0, 255).round()


def calc_psnr(pred, gt, mask=None):
    """
        Here we assume quantized(0-255) arguments.
    """
    diff = (pred - gt).div(255)

    if mask is not None:
        mse = diff.pow(2).sum() / (3 * mask.sum())
    else:
        mse = diff.pow(2).mean() + 1e-8

    return -10 * math.log10(mse)


def calc_ssim(img1, img2, datarange=255.):
    im1 = img1.numpy().transpose(1, 2, 0).astype(np.uint8)
    im2 = img2.numpy().transpose(1, 2, 0).astype(np.uint8)
    return compare_ssim(im1, im2, datarange=datarange, multichannel=True, gaussian_weights=True)


def calc_metrics(im_pred, im_gt, mask=None):
    q_im_pred = quantize(im_pred.data, rgb_range=1.)
    q_im_gt = quantize(im_gt.data, rgb_range=1.)
    if mask is not None:
        q_im_pred = q_im_pred * mask
        q_im_gt = q_im_gt * mask
    psnr = calc_psnr(q_im_pred, q_im_gt, mask=mask)
    # ssim = calc_ssim(q_im_pred.cpu(), q_im_gt.cpu())
    ssim = ssim_pth(q_im_pred.unsqueeze(0), q_im_gt.unsqueeze(0), val_range=255)
    return psnr, ssim


def calc_IE(im_pred, im_gt):
                
    im_pred = np.round(np.squeeze(im_pred.detach().cpu().numpy()).transpose(1, 2, 0) * 255)
    im_gt = (np.squeeze(im_gt.detach().cpu().numpy()).transpose(1, 2, 0) * 255.0).astype('uint8')*1.0
    IE = np.abs((im_pred - im_gt)).mean()
    return IE


def eval_metrics(output, gt, psnrs, ssims):
    for b in range(gt.size(0)):
        psnr, ssim = calc_metrics(output[b], gt[b], None)
        psnrs.update(psnr)
        ssims.update(ssim)
        

def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# tensorboard
def log_tensorboard(writer, losses, psnr, ssim, lr, timestep, mode='train'):
    for k, v in losses.items():
        writer.add_scalar('Loss/%s/%s' % (mode, k), v.avg, timestep)
    writer.add_scalar('PSNR/%s' % mode, psnr, timestep)
    writer.add_scalar('SSIM/%s' % mode, ssim, timestep)
    
    if mode == 'train':
        writer.add_scalar('lr', lr, timestep)

# save image to file
def save_image(img, path):
    # img : torch Tensor of size (C, H, W)
    q_im = quantize(img.data.mul(255))
    if len(img.size()) == 2:    # grayscale image
        im = Image.fromarray(q_im.cpu().numpy().astype(np.uint8), 'L')
    elif len(img.size()) == 3:
        im = Image.fromarray(q_im.permute(1, 2, 0).cpu().numpy().astype(np.uint8), 'RGB')
    else:
        pass
    im.save(path)
