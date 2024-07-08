import torch
import albumentations as A
import cv2
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

class OEMDataset(torch.utils.data.Dataset):
    """DeepGlobe Road Extraction Challenge Dataset. Read images, apply augmentation and preprocessing transformations.
    
    Args:
        df (pd.DataFrame): DataFrame containing images / labels paths
        split (str): Dataset split ('train', 'val', 'test')
        augmentation (albumentations.Compose): Data transformation pipeline (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): Data preprocessing (e.g. normalization, shape manipulation, etc.)
        resize_dimensions (int): Height and Width of transformed images
    """
    def __init__(
            self, 
            df: pd.DataFrame,
            split: str = 'train', 
            augmentation=None, 
            preprocessing=None,
            resize_dimensions: int = 256
    ):
        self.image_paths = df.loc[df['group'] == split, 'image'].tolist()
        self.mask_paths = df.loc[df['group'] == split, 'mask'].tolist()
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.resize_dimensions = resize_dimensions
        
    @staticmethod
    def normalise_band(band):
        return (band - band.min()) / (band.max() - band.min())
        
    def __getitem__(self, i):
        # read images and masks
        image = cv2.cvtColor(cv2.imread(self.image_paths[i]), cv2.COLOR_BGR2RGB)
        
        # Normalise image
        image = self.normalise_band(image)
        
        # Convert mask to grayscale
        mask = np.expand_dims(cv2.cvtColor(cv2.imread(self.mask_paths[i]), cv2.COLOR_BGR2GRAY), axis=-1)  # Convert mask to grayscale
                
        # Resize transform
        target_size = (self.resize_dimensions, self.resize_dimensions)
        resize_transform = A.Resize(*target_size, interpolation=cv2.INTER_LINEAR)
        transformed = resize_transform(image=image, mask=mask)
        image, mask = transformed['image'], transformed['mask']
        
        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
            
        # Convert mask to a tensor with the appropriate shape
        image = torch.tensor(image, dtype=torch.float).permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
        mask = torch.tensor(mask, dtype=torch.long).permute(2, 0, 1)  # (H, W) -> (1, H, W)
        
        return image, mask
        
    def __len__(self):
        return len(self.image_paths)

# PyTorch Lightning Data Module
import pytorch_lightning as L

class OEMDataModule(L.LightningDataModule):
    def __init__(self, metadata_csv, 
                 augmentation, preprocessing, batch_size, num_workers, resize_dimensions):
        super().__init__()
        self.metadata = pd.read_csv(metadata_csv)
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.resize_dimensions = resize_dimensions
        
    def setup(self, stage: str = None) -> None:
        if stage == 'fit' or stage is None:
            self.train_dataset = OEMDataset(
                df=self.metadata, split='train',
                augmentation=self.augmentation,
                preprocessing=self.preprocessing,
                resize_dimensions=self.resize_dimensions
            )
            self.val_dataset = OEMDataset(
                df=self.metadata, split='val',
                preprocessing=self.preprocessing,
                resize_dimensions=self.resize_dimensions
            )
        if stage == 'test':
            self.test_dataset = OEMDataset(
                df=self.metadata, split='test',
                preprocessing=self.preprocessing,
                resize_dimensions=self.resize_dimensions
            )
            
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True, prefetch_factor=self.batch_size,pin_memory=False, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, prefetch_factor=self.batch_size, drop_last=False, num_workers=self.num_workers)
        
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, prefetch_factor=self.batch_size, shuffle=False, drop_last=False, num_workers=self.num_workers)
    
##### Data Augmentation #####
import numpy as np
def get_training_augmentation():
    train_transform = [
        A.RandomGamma(p=0.2),  # random gamma
        A.RandomBrightnessContrast(p=0.2),  # random brightness and contrast,
        A.OneOf([
            A.HorizontalFlip(p=0.5),  # horizontal flip
            A.VerticalFlip(p=0.5),  # vertical flip
            A.Rotate(limit=90),  # 90 degree rotation
            A.Flip(p=0.5), # flip
            A.Transpose(p=0.5), # transpose
            A.RandomRotate90(p=0.5) # random 90 degree rotation
        ], p=1.0),
        A.OneOf([
            A.ChannelShuffle(),
            A.CoarseDropout(max_holes=np.random.randint(1, 20), max_height=np.random.randint(5, 25), max_width=np.random.randint(5, 25), mask_fill_value=0),
            A.PixelDropout(per_channel=True),
        ], p=1.0),
        A.OneOf([
            # A.Equalize(),
            # A.CLAHE(), 
            A.InvertImg(),  
        ], p=0.3),
        A.ElasticTransform(p=0.2),
        A.RandomResizedCrop(size=(512, 512), scale=(0.5, 1.0), ratio=(0.75, 1.3333333333333333), p=0.3), # Size should equal the resize dimension parameter
    ]
    return A.Compose(train_transform)


import segmentation_models_pytorch as sm_torch

def to_tensor(x, **kwargs):
    return x.transpose(0, 1, 2).astype('float32')
    
def get_preprocessing_function(encoder, weights):
    preprocessing_function = sm_torch.encoders.get_preprocessing_fn(encoder, weights)
    return preprocessing_function

def get_preprocessing(preprocessing_fn=None):  
    """Construct preprocessing transform    
    Args:
        preprocessing_fn (callable): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    """
    _transform = []
    if preprocessing_fn:
        _transform.append(A.Lambda(image=preprocessing_fn))
    _transform.append(A.Lambda(image=to_tensor, mask=to_tensor))
        
    return A.Compose(_transform)