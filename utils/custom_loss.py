import torch
import torch.nn as nn

class CLoss(nn.Module):
    def __init__(self, sigma=0.5):
        super(CLoss, self).__init__()
        self.sigma = sigma
    
    def forward(self, output, target):
        loss = 1 - torch.exp(-(target - output) ** 2 / 2 * (self.sigma) ** 2)
        return torch.mean(loss)
    
class HybridLoss(nn.Module):
    def __init__(self):
        super(HybridLoss, self).__init__()
        self.mse = torch.nn.MSELoss()
        self.mae = torch.nn.L1Loss()
        self.c = CLoss()
        
    def forward(self, output, target):
        MSE = self.mse(output, target)
        MAE = self.mae(output, target)
        C = self.c(output, target)
        # loss =  MSE * (MSE / (MSE + MAE + C)) + MAE * (MAE / (MSE + MAE + C)) + C * (C / (MSE + MAE + C))
        # loss = MSE
        loss = MSE + MAE + 8*C
        return loss
        
  