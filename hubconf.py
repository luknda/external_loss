dependencies = ['torch']

def combined_loss(ce_weight=1.0, dice_weight=1.0, class_weights=None):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class CombinedLoss(nn.Module):
        def __init__(self, ce_weight, dice_weight, class_weights):
            super(CombinedLoss, self).__init__()
            self.ce_weight = ce_weight
            self.dice_weight = dice_weight
            if class_weights is not None:
                self.class_weights = torch.tensor(class_weights)
            else:
                self.class_weights = None
            self.ce = nn.CrossEntropyLoss(weight=self.class_weights)

        def forward(self, inputs, targets):
            loss_ce = self.ce(inputs, targets)
            loss_dice = self.dice_loss(inputs, targets)
            return self.ce_weight * loss_ce + self.dice_weight * loss_dice

        def dice_loss(self, inputs, targets, smooth=1.):
            inputs = torch.softmax(inputs, dim=1)
            targets_one_hot = F.one_hot(targets, num_classes=inputs.shape[1]).permute(0, 3, 1, 2).float()
            intersection = (inputs * targets_one_hot).sum(dim=(2, 3))
            union = inputs.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3))
            dice = (2. * intersection + smooth) / (union + smooth)
            loss = 1 - dice.mean()
            return loss

    return CombinedLoss(ce_weight, dice_weight, class_weights)