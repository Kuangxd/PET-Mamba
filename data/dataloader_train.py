from torch.utils.data import Dataset
import torch
import numpy as np
import pdb
import os

class Dataset(Dataset):
    def __init__(self, args, root_dirs):
        self.root_dirs = root_dirs
        self.frame_num = args.frame_num
        self.img_noise_path = []
        self.img_clean_path = []
        self.img_hw = args.img_hw

        for root, _, files in os.walk(os.path.join(root_dirs, 'noise')):
            for file in files:
                self.img_noise_path.append(os.path.join(root, file))
                self.img_clean_path.append(os.path.join(root, file).replace('noise', 'clean'))

    def __len__(self):
        return len(self.img_noise_path)

    def __getitem__(self, idx):
        img_noise_path = self.img_noise_path[idx]
        img_clean_path = self.img_clean_path[idx]
        
        image_clean = np.fromfile(img_clean_path, np.float32).reshape(self.img_hw, self.img_hw, self.frame_num).transpose(2, 0, 1)
        image_noise = np.fromfile(img_noise_path, np.float32).reshape(self.img_hw, self.img_hw, self.frame_num).transpose(2, 0, 1)

        image_clean = torch.from_numpy(image_clean)
        image_noise = torch.from_numpy(image_noise)

        return image_clean, image_noise

def get_dataloader(args):
    dataloader = torch.utils.data.DataLoader(Dataset(args, args.train_path), batch_size=args.train_batch, shuffle=True)

    return dataloader
