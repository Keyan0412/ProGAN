import torch.utils.data as data
import os
from PIL import Image

class Mydata(data.Dataset):
    def __init__(self, dir: str, transform = None):
        self.dir = dir
        self.images = []
        self.transform = transform
        for subdir, dirs, files in os.walk(dir):  
            for file in files:  
                img_path = os.path.join(subdir, file)  
                self.images.append(img_path)   

    def __getitem__(self, idx: int):
        img_path = self.images[idx]  
        image = Image.open(img_path)

        if self.transform:  
            image = self.transform(image)  
  
        return image, True

    def __len__(self):
        return len(self.images) 
