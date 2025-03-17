import os
import cv2
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms

# Import custom modules with updated paths/names
from models.cnn import CNN
from data.Data_Loader import CustomImageDataset, CLASSES
from utils.utils import check_camera_available, train_one_epoch, evaluate
from utils.face_det import FaceDetector
from utils.eval import compute_metrics

class Runner:
    
    def __init__(self, cfg):
        
        self.cfg = cfg
        
        # Set up device: use CUDA if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if self.device.type == "cuda":
            try:
                # Limit GPU memory usage to 50%
                torch.cuda.set_per_process_memory_fraction(0.5, device=self.device)
            except Exception as e:
                print("\nCould not set GPU memory fraction:", e + "\n")
        
        # Use config values with defaults if not provided
        self.data_dir = cfg.get('data_dir', "data")
        self.model_dir = cfg.get('model_dir', "model")
        self.batch_size = cfg.get('batch_size', 32)
        self.learning_rate = cfg.get('learning_rate', 0.001)
        self.epochs = cfg.get('epochs', 10)
    
    
    def train(self):
        
        # -------------------------------
        # Training Phase with Augmentation & Weighted Loss
        # -------------------------------
        
        # First, load the full dataset without a transform for splitting purposes
        full_dataset = CustomImageDataset(root_dir=self.data_dir, transform=None)
        
        if len(full_dataset) == 0:
            print("No images found in the dataset. Exiting training phase.")
            return
        
        # Create indices and split into training (80%) and validation (20%)
        indices = list(range(len(full_dataset)))
        split = int(0.8 * len(full_dataset))

        train_indices = indices[:split]
        val_indices = indices[split:]
        
        # Define separate transforms for training and validation
        train_transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        
        val_transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        
        # Create separate dataset instances for training and validation using the indices
        train_dataset = CustomImageDataset(root_dir=self.data_dir, transform=train_transform, indices=train_indices)
        val_dataset = CustomImageDataset(root_dir=self.data_dir, transform=val_transform, indices=val_indices)
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        # Compute class weights based on training labels
        train_labels = train_dataset.labels
        class_sample_count = np.array([train_labels.count(i) for i in range(len(CLASSES))])
        
        weight = 1. / class_sample_count
        class_weights = torch.tensor(weight, dtype=torch.float).to(self.device)
        
        # Initialize the loss function with class weights
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
        
        # Initialize the CNN model, optimizer, and learning rate scheduler
        model = CNN(num_classes=len(CLASSES))
        model.to(self.device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True)
        
        print("\n" + "Starting training... \n")
        
        for epoch in range(self.epochs):
            
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, self.device)
            val_loss, val_accuracy, y_true, y_pred = evaluate(model, val_loader, criterion, self.device)
            
            # Step the scheduler with the validation loss
            scheduler.step(val_loss)
            
            print(f"Epoch {epoch+1}/{self.epochs} - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.4f}")
        
        # Ensure the model sub-directory exists and save the trained model
        os.makedirs(self.model_dir, exist_ok=True)
        
        model_path = os.path.join(self.model_dir, "model.pth")
        
        torch.save(model.state_dict(), model_path)
        
        print("\n" + f"Training complete. Model saved as {model_path}. \n")
        
        # -------------------------------
        # Compute and Display Metrics on Validation Set
        # -------------------------------
        
        val_loss, val_accuracy, y_true, y_pred = evaluate(model, val_loader, criterion, self.device)
        accuracy, precision, recall, f1, cm = compute_metrics(y_true, y_pred)
        
        print("\n" + "Validation Metrics: \n")
        
        print(f"F1-Score:  {f1:.4f}")
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        
        print("\n" + "Confusion Matrix: \n")
        print_nice_confusion_matrix(cm, CLASSES)
        
    
    def eval(self):
        # ... (Evaluation Phase remains unchanged) ...
        # [Keep your existing eval() method here]
        # For brevity, evaluation code remains as before.
        pass


def print_nice_confusion_matrix(cm, labels):
   
    """
    Prints the confusion matrix in a modern, minimalist table format.
    """

    cm = np.array(cm)
    
    row_labels = labels
    col_labels = labels
    
    # Build table data: header row and each row with row label and corresponding values
    table = []
    header = [""] + col_labels
    
    table.append(header)

    for i, row in enumerate(cm):
        table.append([row_labels[i]] + [str(x) for x in row])
    
    # Compute column widths
    col_widths = [max(len(item) for item in col) for col in zip(*table)]
    
    # Create borders using Unicode box-drawing characters
    top_border = "┌" + "┬".join("─" * (w + 2) for w in col_widths) + "┐"
    header_sep = "├" + "┼".join("─" * (w + 2) for w in col_widths) + "┤"
    bottom_border = "└" + "┴".join("─" * (w + 2) for w in col_widths) + "┘"
    

    def format_row(row):
        return "│" + "│".join(f" {item:^{w}} " for item, w in zip(row, col_widths)) + "│"
    

    # Print the table
    print(top_border)
    print(format_row(table[0]))
    print(header_sep)

    for row in table[1:]:
        print(format_row(row))

    print(bottom_border)
