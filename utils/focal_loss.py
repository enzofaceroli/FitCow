import torch
import torch.nn as nn
import torch.nn.functional as F 

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, reduction='mean', num_classes=5):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.num_classes = num_classes
        
    def forward(self, inputs, targets):
        log_probs = F.log_softmax(inputs, dim=1)
        probs = torch.exp(log_probs)
        
        num_classes = inputs.size(1)
        targets_one_hot = F.one_hot(targets, num_classes=num_classes).float()
        
        ce_loss = -targets_one_hot * log_probs
        
        p_t = torch.sum(probs * targets_one_hot, dim=1)
        focal_weight = (1 - p_t) ** self.gamma
        
        if self.alpha is not None:
            alpha_device = self.alpha.to(inputs.device) 
            alpha_t = alpha_device.gather(0, targets)
            ce_loss = alpha_t.unsqueeze(1) * ce_loss
        
        loss = focal_weight.unsqueeze(1) * ce_loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss