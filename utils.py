
import torch.nn as nn


class BinaryClassificationModelSAFSAR(nn.Module):
    def __init__(self, input_dim):
        super(BinaryClassificationModelSAFSAR, self).__init__()
        self.fc1 = nn.Linear(input_dim, input_dim*2)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(input_dim*2, input_dim)
        self.act2 = nn.ReLU()
        self.fc3 = nn.Linear(input_dim, 64)
        self.act3 = nn.ReLU()
        self.fc4 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):  # Shape is 40, 28, 1152
        x = self.act1(self.fc1(x))  # Shape is 40X28, 512
        x = self.act2(self.fc2(x))  # Shape is 40X28, 128
        x = self.act3(self.fc3(x))  # Shape is 40X28, 64
        x = self.sigmoid(self.fc4(x))
        return x



def split_first_dim_linear(x, first_two_dims):
    """
    Undo the stacking operation
    """
    x_shape = x.size()
    new_shape = first_two_dims
    if len(x_shape) > 1:
        new_shape += [x_shape[-1]]
    return x.view(new_shape)
