"""Dataset Classes"""

import glob
import PIL
from torch.utils.data import Dataset
from tqdm.auto import tqdm


class MP3DL_dataset(Dataset):
    """
    Dataset for MP3D-layout dataset
    
    path: The path to the folder with images to load
    transform: Composed trasnformation to apply
    load_all: load all the elements when creating the dataset [TRUE] or just when called [FALSE]
    """
    def __init__(self, path, transform=None, load_all=True) -> None:
        self.path = path
        self.x_path = self.path+"/image/*"
        self.y_path = self.path+"/layout_depth/*"
        self.x_data_path = sorted(glob.glob(self.x_path))
        self.y_data_path = sorted(glob.glob(self.y_path))
        self.transform = transform
        self.load_all = load_all
        self.x = []
        self.y = []

        if self.load_all:
            for self.path in tqdm(self.x_data_path):
                self.x.append(
                    self.transform(PIL.Image.open(path))
                )
            for self.path in tqdm(self.y_data_path):
                self.y.append(
                    self.transform(PIL.Image.open(path))
                )


    def __len__(self):
        return len(self.x_data_path) if self.load_all is False else len(self.x)


    def __getitem__(self, idx):
        if self.load_all:
            return self.x[idx]/self.x[idx].max(), self.y[idx]/self.y[idx].max()

        x = self.transform(PIL.Image.open(self.x_data_path[idx]))
        y = self.transform(PIL.Image.open(self.y_data_path[idx]))

        return x/x.max(), y/y.max()
