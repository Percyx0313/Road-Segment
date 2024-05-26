import torch
from PIL import Image
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from sklearn.model_selection import train_test_split

num_class = 4 # Road / Box shaped obstacle / Destination point / Background (Others) 
new_h, new_w = 224, 224 # cropped image size
flip_rate = 0.4
img_dir = "./data/data_dataset_voc/JPEGImages"
seg_img_dir = "./data/data_dataset_voc/SegmentationClassNpy"
labels_file = "./data/data_dataset_voc/class_names.txt"

def create_train_val_set(img_dir=img_dir, seg_dir=seg_img_dir):
    X = list(sorted([os.path.join(img_dir, file) for file in os.listdir(img_dir)]))
    y = list(sorted([os.path.join(seg_dir, file) for file in os.listdir(seg_dir)]))

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, shuffle=True)
    return X_train, X_val, y_train, y_val

class JetBotDataset(Dataset):
    def __init__(self,
                 image_files, 
                 seg_img_files,
                 split="train"):
        self.X = image_files
        self.Y = seg_img_files
        self.new_h = new_h
        self.new_w = new_w
        self.split = split
        self.n_class = num_class

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # open image data
        img = Image.open(self.X[idx]).convert('RGB')
        w, h = img.size
        img = np.asarray(img)

        # open segement image .npy data 
        seg_img = np.load(self.Y[idx])

        # crop image and seg_img
        if self.split == "train":
            A_x_offset = np.int32(np.random.randint(0, w - self.new_w + 1, 1))[0]
            A_y_offset = np.int32(np.random.randint(0, h - self.new_h + 1, 1))[0]
        else:
            A_x_offset = int((w - self.new_w)/2)
            A_y_offset = int((h - self.new_h)/2)
            
        img = img[A_y_offset: A_y_offset + self.new_h, A_x_offset: A_x_offset + self.new_w] 
        seg_img = seg_img[A_y_offset: A_y_offset + self.new_h, A_x_offset: A_x_offset + self.new_w]

        # flip images and seg_img 
        img = np.transpose(img, (2, 0, 1)) / 255.
        if np.random.sample() < flip_rate and self.split == "train":
            img = np.fliplr(img)
            seg_img = np.fliplr(seg_img)

        # create tensor
        img = torch.from_numpy(img.copy()).float()
        seg_img = torch.from_numpy(seg_img.copy()).long()

        # create one-hot encoding tensor
        h, w = seg_img.size()
        target = torch.zeros(self.n_class, h, w)
        for c in range(self.n_class):
            target[c][seg_img == c] = 1

        return {"img": img, "label": target}
    
def read_labels(labels_file=labels_file):
    label_map = {}
    with open(labels_file, 'r') as fp:
        for idx, label_name in enumerate(fp.readlines()):
            label_map[idx] = label_name
    # {0: '_background_\n', 1: 'road\n', 2: 'obstacle', 3: 'destination'}
    return label_map
    
def get_train_val_dataloader(batch_size, num_workers=0):
    X_train, X_val, y_train, y_val = create_train_val_set()

    # train set enable data augumentation
    train_set = JetBotDataset(X_train, y_train, 'train')
    # val set no data augumentation
    val_set = JetBotDataset(X_val, y_val, 'val')
    
    dataset_size = {'train': train_set.__len__(), 'val': val_set.__len__()}

    datasets = {'train': train_set, 'val': val_set}
    dataloaders = {x: DataLoader(dataset=datasets[x],
                                 shuffle=True if x == 'train' else False,
                                 batch_size=batch_size,
                                 num_workers=num_workers)
                   for x in ['train', 'val']}
    return dataloaders, dataset_size
