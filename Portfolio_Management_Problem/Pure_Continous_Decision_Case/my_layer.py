import torch.nn as nn
import torch.nn.functional as F
import torch

N=1

class RegressionLayer(nn.Module):
    def __init__(self,n_z,n_y):
        super(RegressionLayer, self).__init__()

        """
        Input:
        n_z: Int. Dimension of the covariate z
        n_y: Int. Number of assets
        """
        self.output_dim=n_y
        self.input_dim=n_z
        self.fc11 = nn.Linear(self.input_dim, 22,dtype=torch.float32)
        self.fc12 = nn.Linear(22,27,dtype=torch.float32)
        self.fc13 = nn.Linear(27, self.output_dim,dtype=torch.float32)

        self.fc21 = nn.Linear(self.input_dim, 22,dtype=torch.float32)
        self.fc22 = nn.Linear(22,27,dtype=torch.float32)
        self.fc23 = nn.Linear(27, self.output_dim,dtype=torch.float32)


    def forward(self, x):

        y1 = F.relu(self.fc11(x))
        y1 = F.relu(self.fc12(y1))
        y1 = self.fc13(y1).reshape(-1,self.output_dim,1)


        y2 = F.relu(self.fc21(x))
        y2 = F.relu(self.fc22(y2))
        y2 = self.fc23(y2).reshape(-1,self.output_dim,1)

        return y1,y2