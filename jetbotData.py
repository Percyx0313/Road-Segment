import torch
from PIL import Image
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from sklearn.model_selection import train_test_split

num_class = 4 # Road / Box shaped obstacle / Destination point / Background (Others) 
new_h, new_w = 128, 128 # cropped image size
flip_rate = 0.4
data_dirs = {
    "labelme": {
        "img_dir": "./data/labelme/fine_tune/images",
        "seg_img_dir": "./data/labelme/fine_tune/masks"
    },
    "roadlane": {
        "img_dir": "./data/roadlane"
    }
}
labels_file = "./data/data_dataset_voc/class_names.txt"

def create_train_val_set(data_dir, type="roadlane"):
    if type == "labelme":
        X = list(sorted([os.path.join(data_dir["img_dir"], file) for file in os.listdir(data_dir["img_dir"])]))
        y = list(sorted([os.path.join(data_dir["seg_img_dir"], file) for file in os.listdir(data_dir["seg_img_dir"])]))

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, shuffle=True)  
    elif type == "roadlane":
        train_image_dir = os.path.join(data_dir["img_dir"], "train", "images")
        train_seg_image_dir = os.path.join(data_dir["img_dir"], "train", "masks")
        val_image_dir = os.path.join(data_dir["img_dir"], "val", "images")
        val_seg_image_dir = os.path.join(data_dir["img_dir"], "val", "masks")

        X_train = list(sorted([os.path.join(train_image_dir, file) for file in os.listdir(train_image_dir)]))  
        y_train = list(sorted([os.path.join(train_seg_image_dir, file) for file in os.listdir(train_seg_image_dir)]))  
        X_val = list(sorted([os.path.join(val_image_dir, file) for file in os.listdir(val_image_dir)])) 
        y_val = list(sorted([os.path.join(val_seg_image_dir, file) for file in os.listdir(val_seg_image_dir)])) 
    return X_train, X_val, y_train, y_val

class LabelmeDataset(Dataset):
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
        self.label_colors = {
            0: [0,0,0],   # black -> background
            1: [128,0,0],   # dark red -> road
            2: [0,128,0],   # light green -> obstacle
            3: [128,128,0]  # green -> destination
        }

    def __len__(self):
        return len(self.X)

    def rgb_to_label_image(self, rgb_image):
        # Create an empty label image with the same height and width as the input image
        height, width, _ = rgb_image.shape
        label_image = np.zeros((height, width), dtype=np.int32)
        
        # Loop through each color and assign the corresponding label
        for label, color in self.label_colors.items():
            # Create a mask for the current color
            mask = np.all(rgb_image == color, axis=-1)
            # Assign the label to the mask
            label_image[mask] = label
    
        return label_image
    
    def __getitem__(self, idx):
        # open image data
        img = Image.open(self.X[idx]).convert('RGB')
        if self.split == "train":
            img = img.resize((2*self.new_w, 2*self.new_h), resample=Image.Resampling.NEAREST)
        else:
            img = img.resize((self.new_w, self.new_h), resample=Image.Resampling.NEAREST)
        w, h = img.size
        img = np.asarray(img)

        # open segement image data 
        seg_rgb_img = Image.open(self.Y[idx]).convert('RGB')
        if self.split == "train":
            seg_rgb_img = seg_rgb_img.resize((2*self.new_w, 2*self.new_h), resample=Image.Resampling.NEAREST)
        else:
            seg_rgb_img = seg_rgb_img.resize((self.new_w, self.new_h), resample=Image.Resampling.NEAREST)
        seg_rgb_img = np.asarray(seg_rgb_img)
        seg_img = self.rgb_to_label_image(seg_rgb_img)

        # crop image and seg_img
        if self.split == "train":
            A_x_offset = np.int32(np.random.randint(0, w - self.new_w + 1, 1))[0]
            A_y_offset = np.int32(np.random.randint(0, h - self.new_h + 1, 1))[0]
            img = img[A_y_offset: A_y_offset + self.new_h, A_x_offset: A_x_offset + self.new_w] 
            seg_rgb_img = seg_rgb_img[A_y_offset: A_y_offset + self.new_h, A_x_offset: A_x_offset + self.new_w]
            seg_img = seg_img[A_y_offset: A_y_offset + self.new_h, A_x_offset: A_x_offset + self.new_w]

        # flip images and seg_img 
        if np.random.sample() < flip_rate and self.split == "train":
            img = np.fliplr(img)
            seg_rgb_img = np.fliplr(seg_rgb_img)
            seg_img = np.fliplr(seg_img)
        img = np.transpose(img, (2, 0, 1)) / 255.
        seg_rgb_img = np.transpose(seg_rgb_img, (2, 0, 1)) / 255.

        # create tensor
        img = torch.from_numpy(img.copy()).float()
        seg_img = torch.from_numpy(seg_img.copy()).long()

        # create one-hot encoding tensor
        h, w = seg_img.size()
        target = torch.zeros(self.n_class, h, w)
        for c in range(self.n_class):
            target[c][seg_img == c] = 1

        return {"img": img, "label": target, "seg_rgb": seg_rgb_img}
    

class RoadLaneDataset(Dataset):
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
        self.label_colors = {
            0: [0,0,0],   # black -> background
            1: [128,0,0],   # dark red -> road
            2: [0,128,0],   # light green -> landmark solid
            3: [128,128,0]  # green -> landmark dashed
        }
        self.colors = [[0,0,0],[128,0,0],[0,128,0],[128,128,0]]

    def __len__(self):
        return len(self.X)

    def rgb_to_label_image(self, rgb_image):
        # Create an empty label image with the same height and width as the input image
        height, width, _ = rgb_image.shape
        label_image = np.zeros((height, width), dtype=np.int32)
        
        # Loop through each color and assign the corresponding label
        for label, color in self.label_colors.items():
            # Create a mask for the current color
            mask = np.all(rgb_image == color, axis=-1)
            # Assign the label to the mask
            label_image[mask] = label
    
        return label_image

    def __getitem__(self, idx):
        # open image data
        img = Image.open(self.X[idx]).convert('RGB')
        img = img.resize((self.new_w, self.new_h), resample=Image.Resampling.NEAREST)
        img = np.asarray(img)

        # open segement image data 
        seg_rgb_img = Image.open(self.Y[idx]).convert('RGB')
        seg_rgb_img = seg_rgb_img.resize((self.new_w, self.new_h), resample=Image.Resampling.NEAREST)
        seg_rgb_img = np.asarray(seg_rgb_img)
        seg_img = self.rgb_to_label_image(seg_rgb_img)

        # flip images and seg_img 
        if np.random.sample() < flip_rate and self.split == "train":
            img = np.fliplr(img)
            seg_rgb_img = np.fliplr(seg_rgb_img)
            seg_img = np.fliplr(seg_img)
        img = np.transpose(img, (2, 0, 1)) / 255.
        seg_rgb_img = np.transpose(seg_rgb_img, (2, 0, 1)) / 255.

        # create tensor
        img = torch.from_numpy(img.copy()).float()
        seg_img = torch.from_numpy(seg_img.copy()).long()

        # create one-hot encoding tensor
        h, w = seg_img.size()
        target = torch.zeros(self.n_class, h, w)
        for c in range(self.n_class):
            target[c][seg_img == c] = 1

        return {"img": img, "label": target, "seg_rgb": seg_rgb_img}
    
def read_labels(labels_file=labels_file):
    label_map = {}
    with open(labels_file, 'r') as fp:
        for idx, label_name in enumerate(fp.readlines()):
            label_map[idx] = label_name
    # {0: '_background_\n', 1: 'road\n', 2: 'obstacle', 3: 'destination'}
    return label_map
    
def get_train_val_dataloader(batch_size, num_workers=0, type='roadlane'):
    train_set = None
    val_set = None
    data_dir = data_dirs[type]
    X_train, X_val, y_train, y_val = create_train_val_set(data_dir, type)

    # train set enable data augumentation
    # val set no data augumentation
    if type == "labelme":
        train_set = LabelmeDataset(X_train, y_train, 'train')
        val_set = LabelmeDataset(X_val, y_val, 'val')
    elif type == "roadlane":
        train_set = RoadLaneDataset(X_train, y_train, 'train')
        val_set = RoadLaneDataset(X_val, y_val, 'val')
    
    dataset_size = {'train': train_set.__len__(), 'val': val_set.__len__()}

    datasets = {'train': train_set, 'val': val_set}
    dataloaders = {x: DataLoader(dataset=datasets[x],
                                 shuffle=True if x == 'train' else False,
                                 batch_size=batch_size,
                                 num_workers=num_workers)
                   for x in ['train', 'val']}
    return dataloaders, dataset_size
