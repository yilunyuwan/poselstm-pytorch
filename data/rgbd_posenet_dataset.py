import os.path
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset, get_posenet_transform
from data.image_folder import make_dataset
from PIL import Image
import PIL
import random
import numpy

class RGBDPoseNetDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot

        split_file = os.path.join(self.root , 'dataset_'+opt.phase+'.txt')
        self.A_paths = numpy.loadtxt(split_file, dtype=str, delimiter=' ', skiprows=3, usecols=(0))
        self.A_paths = [os.path.join(self.root, path) for path in self.A_paths]
        self.A_poses = numpy.loadtxt(split_file, dtype=float, delimiter=' ', skiprows=3, usecols=(1,2,3,4,5,6,7))
        self.mean_image_depth = numpy.load(os.path.join(self.root , 'mean_image_depth.npy'))

        if opt.model == "poselstm":
            self.mean_image = None
            print("mean image subtraction is deactivated")

        self.A_size = len(self.A_paths)
        self.transform = get_posenet_transform(opt, self.mean_image_depth)

    def __getitem__(self, index):
        A_path = self.A_paths[index % self.A_size]
        index_A = index % self.A_size
        # print('(A, B) = (%d, %d)' % (index_A, index_B))
        image = numpy.array(Image.open(A_path).convert('RGB'))
        depth = numpy.array(Image.open(A_path[: -10] + '.depth.png').convert('I'))
        A_img = numpy.concatenate((image, depth[:, :, numpy.newaxis]), axis=2)
        A_pose = self.A_poses[index % self.A_size]

        A = self.transform(A_img)                       

        return {'A': A, 'B': A_pose,
                'A_paths': A_path}

    def __len__(self):
        return self.A_size

    def name(self):
        return 'RGBDPoseNetDataset'
