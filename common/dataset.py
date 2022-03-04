import os

from torch.utils.data import DataLoader, Dataset

def make_dataset(img_list):
    images = [val.strip() for val in img_list]
    return images

class ImageList(Dataset):
    def __init__(self, data_path, root_path, H=224, W=224):
        img_list = open(data_path).readlines()
        self.n_samples = len(img_list)
        self.imgs = make_dataset(img_list)
        self.original_im_root_folder = root_path
        if self.n_samples == 0:
            raise (RuntimeError("Found 0 images in subfolders"))
        self.H = H
        self.W = W

    def __getitem__(self, index):
        im_name = self.imgs[index]
        im_path = os.path.join(self.original_im_root_folder, im_name)

        f = open(im_path, "rb")
        f.read()
        f.close()

        return  0, 0

    def __len__(self):
        return len(self.imgs)
