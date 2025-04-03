
import torch.nn as nn
import os
import cv2
from PIL import Image

class FakeCap:
    # This simulates a camera, but is a video
    def __init__(self, images_path):
        self.images_path = images_path
        self.images = os.listdir(images_path)
        self.images = sorted(self.images, key=lambda x: int(x.split(".")[0]))
        self.counter = 0

    def read(self):
        if self.counter == len(self.images):
            self.counter = 0
        image = cv2.imread(os.path.join(self.images_path, self.images[self.counter]))
        self.counter += 1
        return True, image

    def is_over(self):
        return self.counter == len(self.images)


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

def load_ss(ss_path, n_frames):
    
    # Load ss
    ss = {}
    ss_gifs = {}
    for action in ss_path.iterdir():
        ss[action.name] = []
        ss_gifs[action.name] = []
        for example in (ss_path / action.name).iterdir():
            example_imgs = []
            for i in range(n_frames):
                example_imgs.append(Image.open(ss_path / action.name / example.name / f"{i}.jpg"))
            ss[action.name].append(example_imgs)
            ss_gifs[action.name].append(ss_path / action.name / example.name / f"{action.name}.gif")
    # sort dict alfabetically
    ss = dict(sorted(ss.items()))
    ss_gifs = dict(sorted(ss_gifs.items()))
    return ss, ss_gifs