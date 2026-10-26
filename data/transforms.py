import random
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

class Random90Rotation:
    """Randomly rotate the image by 0, 90, 180, or 270 degrees."""
    def __call__(self, img):
        angle = random.choice([0, 90, 180, 270])
        if angle == 0:
            return img
        return TF.rotate(img, angle)

def get_transforms():
    """
    Returns train and validation transforms.
    Augmentations are spatial only (rotation, flips) to preserve the color anchors
    required for FixedViridisDequantizer and yield proxy calculations.
    """
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        Random90Rotation(),
        transforms.ToTensor(),
    ])
    
    val_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    return train_transform, val_transform
