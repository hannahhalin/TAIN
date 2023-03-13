import torch
import torch.nn as nn
import torch.nn.functional as F


class gradL1(nn.Module):
    def __init__(self, alpha=1):
        super(gradL1, self).__init__()

        self.alpha = alpha

    def gradient(self,x):
        # x: (b,c,h,w), float32 or float64
        # dx, dy: (b,c,h,w)

        h_x = x.size()[-2]
        w_x = x.size()[-1]
        
        left = x
        right = F.pad(x, [0, 1, 0, 0])[:, :, :, 1:]
        top = x
        bottom = F.pad(x, [0, 0, 0, 1])[:, :, 1:, :]

        dx, dy = right - left, bottom - top
        
        dx[:, :, :, -1] = 0
        dy[:, :, -1, :] = 0

        return dx, dy

    def forward(self, sr, hr):
    
        # gradient
        gen_dx, gen_dy = self.gradient(sr)
        gt_dx, gt_dy = self.gradient(hr)
        
        grad_diff_x = torch.abs(gt_dx - gen_dx)
        grad_diff_y = torch.abs(gt_dy - gen_dy)

        # average
        return torch.mean(grad_diff_x ** self.alpha + grad_diff_y ** self.alpha)


# wrapper of loss functions
class Loss(nn.modules.loss._Loss):
    def __init__(self, args):
        super(Loss, self).__init__()

        self.loss = []
        self.loss_module = nn.ModuleList()
        for loss in args.loss.split('+'):
            weight, loss_type = loss.split('*')
            if loss_type == 'L1':
                loss_function = nn.L1Loss()
            elif loss_type.find('gradL1') >= 0:
                loss_function = gradL1()

            self.loss.append({
                'type': loss_type,
                'weight': float(weight),
                'function': loss_function}
            )

        if len(self.loss) > 1:
            self.loss.append({'type': 'Total', 'weight': 0, 'function': None})

        for l in self.loss:
            if l['function'] is not None:
                self.loss_module.append(l['function'])

        device = torch.device('cuda' if args.cuda else 'cpu')
        self.loss_module.to(device)
        
        if args.cuda:
            self.loss_module = nn.DataParallel(self.loss_module)

    def forward(self, sr, hr):
        loss = 0
        losses = {}
        for i, l in enumerate(self.loss):
            if l['function'] is not None:
                _loss = l['function'](sr, hr)
                effective_loss = l['weight'] * _loss
                losses[l['type']] = effective_loss
                loss += effective_loss
        
        return loss, losses
