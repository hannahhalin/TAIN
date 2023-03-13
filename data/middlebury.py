import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class Middlebury(Dataset):
    def __init__(self, data_root):
        self.image_root =os.path.join(data_root,'other-data')
        self.gt_root =os.path.join(data_root,'other-gt-interp')
        
        self.folder_list = sorted(os.listdir(data_root+'other-gt-interp/'))
        
                    
    def __getitem__(self, index):
        imgpathX = os.path.join(self.image_root, self.folder_list[index])
        imgpathY = os.path.join(self.gt_root, self.folder_list[index])
        imgpaths = [imgpathX + '/frame10.png', imgpathY + '/frame10i11.png', imgpathX + '/frame11.png']

        # Load images
        img1 = Image.open(imgpaths[0])
        img2 = Image.open(imgpaths[1])
        img3 = Image.open(imgpaths[2])


        T = transforms.ToTensor()
        img1 = T(img1)
        img2 = T(img2)
        img3 = T(img3)
        
        imgs = [img1, img2, img3]
        
        return imgs, imgpaths

    def __len__(self):
        return len(self.folder_list)
 

def get_loader(mode, data_root, batch_size, shuffle, num_workers, test_mode=None):
    dataset = Middlebury(data_root)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)
