import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class UCF101(Dataset):
    def __init__(self, data_root, is_training):
        self.data_root = data_root
        self.training = is_training
        
        self.folder_list = sorted(os.listdir(data_root))
        print('Found '+str(len(self.folder_list)) +' folders/samples in UCF-101 dataset...')

    def __getitem__(self, index):
        imgpath = os.path.join(self.data_root, self.folder_list[index])
        imgpaths = [imgpath + '/frame_00.png', imgpath + '/frame_01_gt.png', imgpath + '/frame_02.png']

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
    if mode == 'train':
        is_training = True
    else:
        is_training = False
    dataset = UCF101(data_root, is_training=is_training)
    
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)
