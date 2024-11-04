import torch
import torch.nn as nn
import torch.nn.functional as F

class CombinedLoss(nn.Module):
    def __init__(self, ce_weight=1.0, dice_weight=1.0, class_weights=None):
        super(CombinedLoss, self).__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        
        # Initialize class weights
        if class_weights is not None:
            self.class_weights = torch.tensor(class_weights, dtype=torch.float32)
        else:
            self.class_weights = None
        
        # Placeholder for CrossEntropyLoss; will set weight in forward()
        self.ce = nn.CrossEntropyLoss(weight=self.class_weights)

    def forward(self, inputs, targets):
        # Move class_weights to the same device as inputs
        if self.class_weights is not None:
            self.ce.weight = self.class_weights.to(inputs.device)
        
        # Compute Cross-Entropy Loss
        loss_ce = self.ce(inputs, targets)
        
        # Compute Dice Loss
        loss_dice = self.dice_loss(inputs, targets)
        
        # Combine losses
        return self.ce_weight * loss_ce + self.dice_weight * loss_dice

    def dice_loss(self, inputs, targets, smooth=1.):
        # Apply softmax to get probabilities
        inputs_softmax = torch.softmax(inputs, dim=1)
        
        # Convert targets to one-hot encoding and move to the same device
        targets_one_hot = F.one_hot(targets, num_classes=inputs.shape[1]).permute(0, 3, 1, 2).float().to(inputs.device)
        
        # Compute intersection and union
        intersection = (inputs_softmax * targets_one_hot).sum(dim=(2, 3))
        union = inputs_softmax.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3))
        
        # Compute Dice coefficient
        dice_coeff = (2. * intersection + smooth) / (union + smooth)
        
        # Compute Dice loss
        loss = 1 - dice_coeff.mean()
        return loss
