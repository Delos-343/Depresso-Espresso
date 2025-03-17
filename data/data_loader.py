import os
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset

# Define the three classes (order matters: indices 0, 1, 2)
CLASSES = ["depression", "stress", "anxiety"]

class CustomImageDataset(Dataset):
    
    def __init__(self, root_dir, transform=None):
        
        """
        Args:
            root_dir (string): Directory containing 'images/' subfolder.
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
                        
                        full_path = os.path.join(class_dir, file_name)
                        
                        try:
                            # Attempt to open and verify the image file
                            with Image.open(full_path) as img:
                                img.verify()  # Verify that it is a valid image
                            
                            # If verification succeeds, add the image to the dataset
                            self.image_paths.append(full_path)
                            self.labels.append(idx)
                        except (UnidentifiedImageError, IOError, SyntaxError) as e:
                            print(f"Skipping file {full_path}: {e}")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        label = self.labels[idx]
        
        return image, label
