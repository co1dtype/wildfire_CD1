"""
Lovasz-Softmax and Jaccard hinge loss in PyTorch
Maxim Berman 2018 ESAT-PSI KU Leuven (MIT License)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule


def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    # 결정론적 설정 비활성화
    was_deterministic = torch.are_deterministic_algorithms_enabled()
    if was_deterministic:
        torch.use_deterministic_algorithms(False)
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    if was_deterministic:
        torch.use_deterministic_algorithms(True)
    jaccard = 1 - intersection / union
    if p > 1:  # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard

def hinge(pred, label):
    signs = 2 * label - 1
    errors = 1 - pred * signs
    return errors

def lovasz_hinge_flat(logits, labels, ignore_index):
    """
    Binary Lovasz hinge loss
      logits: [P] Variable, logits at each prediction (between -\infty and +\infty)
      labels: [P] Tensor, binary ground truth labels (0 or 1)
      ignore_index: label to ignore
    """
    logits = logits.contiguous().view(-1)
    labels = labels.contiguous().view(-1)
    if ignore_index is not None:
        mask = labels != ignore_index
        logits = logits[mask]
        labels = labels[mask]
    errors = hinge(logits, labels)
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    perm = perm.data
    gt_sorted = labels[perm]
    grad = lovasz_grad(gt_sorted)
    loss = torch.dot(F.elu(errors_sorted) + 1, grad)
    return loss
    
def lovasz_hinge(logits, labels, per_image=True, ignore=None):
    """
    Binary Lovasz hinge loss
      logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
      labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
      per_image: compute the loss per image instead of per batch
      ignore: void class id
    """
    if per_image:
        loss = torch.mean(torch.stack([lovasz_hinge_flat(*flatten_binary_scores(log.unsqueeze(0), lab.unsqueeze(0), ignore))
                          for log, lab in zip(logits, labels)]))
    else:
        loss = lovasz_hinge_flat(*flatten_binary_scores(logits, labels, ignore))
    return loss

class LovaszLoss(LightningModule):
    """
    Binary Lovasz hinge loss
      logits: [P] Variable, logits at each prediction (between -\infty and +\infty)
      labels: [P] Tensor, binary ground truth labels (0 or 1)
      ignore_index: label to ignore
    """
    def __init__(self, ignore_index=None):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, logits, labels):
        return lovasz_hinge_flat(logits, labels, self.ignore_index)



class TverskyLoss(LightningModule):
    def __init__(self, alpha=0.7, beta=0.3, smooth=1e-6):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, y_pred, y_true):
        y_true = y_true.float()
        y_pred = y_pred.float()
        y_pred = F.sigmoid(y_pred) 
        # Flatten label and prediction tensors
        y_true = y_true.view(-1)
        y_pred = y_pred.view(-1)
        
        # True Positives, False Positives & False Negatives
        tp = (y_true * y_pred).sum()
        fp = ((1 - y_true) * y_pred).sum()
        fn = (y_true * (1 - y_pred)).sum()

        # Tversky index
        tversky_index = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        
        # Tversky loss
        return 1 - tversky_index
    

class DiceLoss(LightningModule):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        #inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        #print(inputs, targets)
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice
class DiceFocalLoss(LightningModule):
    def __init__(self, gamma=2., alpha=0.25, weight=None, size_average=True):
        super(DiceFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs, targets, smooth=1):
        # Focal Loss
        if not (targets.size() == inputs.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})"
                             .format(targets.size(), inputs.size()))

        max_val = (-inputs).clamp(min=0)
        loss = inputs - inputs * targets + max_val + ((-max_val).exp() + (-inputs - max_val).exp()).log()

        invprobs = F.logsigmoid(-inputs * (targets * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss
        
        focal_loss = self.alpha * (1 - targets) * loss + (1 - self.alpha) * targets * loss
        focal_loss = focal_loss.sum()

        # Dice Loss
        inputs = torch.sigmoid(inputs) 
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2.*intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        
        return 0.75* dice_loss + 0.25 * focal_loss
    

def get_loss(loss_name: str):
    
    if loss_name == 'LovaszLoss':
        return LovaszLoss()
    
    elif loss_name == 'DiceLoss':
        return DiceLoss()
    
    elif loss_name == 'DiceBCELoss':
        return DiceBCELoss()
    
    elif loss_name == 'DiceFocalLoss':
        return DiceFocalLoss()
    
    elif loss_name == 'TverskyLoss':
        return TverskyLoss()
    
    else:
        print(f'{loss_name}: invalid loss name')
        return
