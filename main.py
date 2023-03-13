import os
import time
import random
import numpy as np
from PIL import Image

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam

import config
import utils
from loss import Loss
from model.tain import TAIN


# argument parser
args, unparsed = config.get_args()
cwd = os.getcwd()


# tensorboard
if args.mode != 'test':
    writer = SummaryWriter('logs/%s' % args.exp_name)

device = torch.device('cuda' if args.cuda else 'cpu')
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

torch.manual_seed(args.random_seed)
if args.cuda:
    torch.cuda.manual_seed(args.random_seed)

# load dataset
train_loader, test_loader = utils.load_dataset(
        args.dataset_name, args.data_root, args.batch_size, args.test_batch_size,
        args.num_workers, test_mode=args.test_mode)

# build model
model = TAIN(depth=args.depth, n_resgroups=args.n_resgroups)
model = torch.nn.DataParallel(model).to(device)

# loss
criterion = Loss(args)

# optimizer
optimizer = Adam(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))

# resume
if args.resume:
    utils.load_checkpoint(args, model, optimizer)

# learning rate scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5, verbose=True)

LOSS_0 = 0

def train(args, epoch):
    global LOSS_0
    losses, psnrs, ssims = utils.init_meters(args.loss)
    model.train()
    criterion.train()

    t = time.time()
    for i, (images, imgpaths) in enumerate(train_loader):
                
        # build input batch
        im1, im2, gt = utils.build_input(images, imgpaths)

        optimizer.zero_grad()
        out = model(im1, im2)
        
        # loss
        loss, loss_specific = criterion(out, gt)

        for k, v in losses.items():
            if k != 'total':
                v.update(loss_specific[k].item())
        if LOSS_0 == 0:
            LOSS_0 = loss.data.item()
        losses['total'].update(loss.item())

        loss.backward()
        if loss.data.item() > 10.0 * LOSS_0:
            continue
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()

        # compute evaluation metrics
        if i % args.log_iter == 0:
            utils.eval_metrics(out, gt, psnrs, ssims)

            print('Train Epoch: {} [{}/{}]\tLoss: {:.6f}\tPSNR: {:.4f}\tTime({:.2f})'.format(
                epoch, i, len(train_loader), losses['total'].avg, psnrs.avg, time.time() - t))
            
            # log to tensorboard
            utils.log_tensorboard(writer, losses, psnrs.avg, ssims.avg,
                optimizer.param_groups[-1]['lr'], epoch * len(train_loader) + i)

            # reset metrics
            losses, psnrs, ssims = utils.init_meters(args.loss)
            t = time.time()


def test(args, epoch, eval_alpha=0.5):

    print('Evaluating for epoch = %d' % epoch)
    losses, psnrs, ssims = utils.init_meters(args.loss)
    
    if args.dataset_name=='middlebury':
        IEs = []
    model.eval()
    criterion.eval()
                
    save_folder = 'test%03d' % epoch
    if args.dataset_name == 'snufilm':
        save_folder = os.path.join(save_folder, args.dataset_name, args.test_mode)
    else:
        save_folder = os.path.join(save_folder, args.dataset_name)
    save_dir = os.path.join('checkpoint', args.exp_name)
    
    if not os.path.exists(os.path.join(save_dir, save_folder)):
        # make directory to save validation images
        os.makedirs(os.path.join(save_dir, save_folder))
        
    save_fn = os.path.join(save_dir, 'results.txt')
    
    with open(save_fn, 'a') as f:
        f.write('For epoch=%d\n' % epoch)

    t = time.time()
                       
    with torch.no_grad():
        for i, (images, imgpaths) in enumerate(test_loader):
        
            # build input batch
            im1, im2, gt = utils.build_input(images, imgpaths, is_training=False)
                  
            out = model(im1, im2)

            # loss
            loss, loss_specific = criterion(out, gt)
            for k, v in losses.items():
                if k != 'total':
                    v.update(loss_specific[k].item())
            losses['total'].update(loss.item())

            # compute evaluation metrics
            utils.eval_metrics(out, gt, psnrs, ssims)
            
            if args.dataset_name=='middlebury':
                    
                IEs.append(utils.calc_IE(out, gt))
                    
                diff_rgb = 128.0 + (out.cpu().numpy() - gt.cpu().numpy())*255
                avg_interp_error_abs = np.mean(np.abs(diff_rgb - 128.0))
                
            # save prediction
            if (epoch + 1) % 1 == 0:
                savepath = os.path.join('checkpoint', args.exp_name, save_folder)

                for b in range(0,images[0].size(0),100):
                    paths = imgpaths[1][b].split('/')
                    fp = os.path.join(savepath, paths[-3], paths[-2])
                    if not os.path.exists(fp):
                        os.makedirs(fp)
                        
                    fp = os.path.join(fp, paths[-1][:-4])
                    utils.save_image(out[b], "%s.png" % fp)

    # print progress
    print('   Images processed: {:d}/{:d} {:.3f}s   \r'.format(i + 1, len(test_loader), time.time() - t))
                                
    if args.dataset_name=='middlebury':
        print("   Loss: %f, PSNR: %f, SSIM: %f, IE: %f\n" %
            (losses['total'].avg, psnrs.avg, ssims.avg, np.mean(IEs)))
    else:
        print("   Loss: %f, PSNR: %f, SSIM: %f\n" %
            (losses['total'].avg, psnrs.avg, ssims.avg))

    # save psnr & ssim
    save_fn = os.path.join('checkpoint', args.exp_name, 'results.txt')
    with open(save_fn, 'a') as f:
        f.write("PSNR: %f, SSIM: %f\n" %
                (psnrs.avg, ssims.avg))

    # tensorboard
    if args.mode != 'test':
        utils.log_tensorboard(writer, losses, psnrs.avg, ssims.avg,
            optimizer.param_groups[-1]['lr'], epoch * len(train_loader) + i, mode='test')

    return losses['total'].avg, psnrs.avg, ssims.avg


def main(args):
    if args.mode == 'test':
        _, _, _ = test(args, args.start_epoch)
        return

    best_psnr = 0
    for epoch in range(args.start_epoch, args.max_epoch):
        
        # train
        train(args, epoch)

        # test
        test_loss, psnr, _ = test(args, epoch)

        # save checkpoint
        is_best = psnr > best_psnr
        best_psnr = max(psnr, best_psnr)
        utils.save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_psnr': best_psnr
        }, is_best, args.exp_name)

        # update optimizer policy
        scheduler.step(test_loss)

if __name__ == "__main__":
    main(args)
