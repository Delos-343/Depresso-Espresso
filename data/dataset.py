import os
from PIL import Image
from torch.utils.data import Dataset

# Define the three classes (order matters: indices 0, 1, 2)
CLASSES = ["depression", "stress", "anxiety"]

class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory containing this file along with 'images/'.
            transform (callable, optional): Optional transform to be applied on an image.
        """
        self.transform = transform
        self.image_paths = []
        self.labels = []

        # Images are located in root_dir/images/ with sub-folders for each class
        images_dir = os.path.join(root_dir, "images")
        for idx, cls in enumerate(CLASSES):
            class_dir = os.path.join(images_dir, cls)
            if os.path.isdir(class_dir):
                for file_name in os.listdir(class_dir):
                    if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                        self.image_paths.append(os.path.join(class_dir, file_name))
                        self.labels.append(idx)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label
