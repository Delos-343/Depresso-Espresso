import os
import cv2
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision import transforms
from torchvision.transforms import RandomResizedCrop, RandomHorizontalFlip, RandomRotation, ColorJitter, Normalize, ToTensor

# Import custom modules with updated paths/names
from data.Data_Loader import CustomImageDataset, CLASSES
from utils.utils import check_camera_available, train_one_epoch, evaluate
from utils.face_det import FaceDetector
from utils.eval import compute_metrics
# CHANGE: Import focal loss option
from utils.foc_loss import FocalLoss

class Runner:
    
    def __init__(self, cfg):

        self.cfg = cfg
        
        # Set up device: use CUDA if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.device.type == "cuda":
            try:
                torch.cuda.set_per_process_memory_fraction(0.5, device=self.device)
            except Exception as e:
                print("Could not set GPU memory fraction:", e)
        
        # Configuration parameters
        self.data_dir = cfg.get('data_dir', "data")
        self.model_dir = cfg.get('model_dir', "model")
        self.batch_size = cfg.get('batch_size', 16)
        self.learning_rate = cfg.get('learning_rate', 1e-4)
        self.epochs = cfg.get('epochs', 20)
        self.patience = cfg.get('patience', 5)
    
    
    def train(self):

        # -------------------------------
        # Training Phase with Augmentation, Weighted Sampling & Fine-Tuning
        # -------------------------------
        
        # Load full dataset for splitting
        full_dataset = CustomImageDataset(root_dir=self.data_dir, transform=None)

        if len(full_dataset) == 0:
            print("No images found in the dataset. Exiting training phase.")
            return
        
        # DEBUG: Print full dataset class distribution
        distribution = {cls: full_dataset.labels.count(idx) for idx, cls in enumerate(CLASSES)}

        print("Full dataset class distribution:", distribution)
        
        indices = list(range(len(full_dataset)))
        split = int(0.8 * len(full_dataset))

        train_indices = indices[:split]
        val_indices = indices[split:]
        
        # CHANGE: Define robust training transform with augmentation
        train_transform = transforms.Compose([
            RandomResizedCrop(size=64, scale=(0.8, 1.0)),  # Random crop then resize
            RandomHorizontalFlip(),
            RandomRotation(10),
            ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406],
                      std=[0.229, 0.224, 0.225])
        ])
        # CHANGE: Define validation transform (simple resize + normalization)
        val_transform = transforms.Compose([
            transforms.Resize((64, 64)),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406],
                      std=[0.229, 0.224, 0.225])
        ])
        
        train_dataset = CustomImageDataset(root_dir=self.data_dir, transform=train_transform, indices=train_indices)
        val_dataset = CustomImageDataset(root_dir=self.data_dir, transform=val_transform, indices=val_indices)
        
        # Compute sample weights for WeightedRandomSampler
        train_labels = np.array(train_dataset.labels)

        class_counts = np.bincount(train_labels, minlength=len(CLASSES))
        class_counts = np.where(class_counts == 0, 1, class_counts)

        sample_weights = [1.0 / class_counts[label] for label in train_labels]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=sampler)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        # Choose model architecture based on configuration.
        if self.cfg.get('use_pretrained', False):

            from models.resnet import ResNetTransfer

            # Use unfreezed model for full fine-tuning.
            model = ResNetTransfer(num_classes=len(CLASSES), freeze_layers=self.cfg.get('freeze_layers', False))

        else:

            from models.cnn import CNN

            model = CNN(num_classes=len(CLASSES))

        model.to(self.device)
        
        # Initialize loss function based on configuration:
        if self.cfg.get('use_focal_loss', False):
            loss_fn = FocalLoss(alpha=self.cfg.get('focal_alpha', 0.25), gamma=self.cfg.get('focal_gamma', 2.0))
        else:
            loss_fn = torch.nn.CrossEntropyLoss()
        
        # Use weight decay in optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)
        
        print("\nStarting training...\n")

        best_val_loss = float('inf')
        best_model_state = None
        
        for epoch in range(self.epochs):

            train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, self.device)

            val_loss, val_accuracy, y_true, y_pred = evaluate(model, val_loader, loss_fn, self.device)

            scheduler.step(val_loss)

            print(f"Epoch {epoch+1}/{self.epochs} - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.4f}")
            
            # Save best model based on validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict()
        
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        os.makedirs(self.model_dir, exist_ok=True)

        model_path = os.path.join(self.model_dir, "dep_esp.pth")

        torch.save(model.state_dict(), model_path)

        print(f"\nTraining complete. Model saved as {model_path}.\n")
        
        # -------------------------------
        # Compute and Display Metrics on Validation Set
        # -------------------------------

        val_loss, val_accuracy, y_true, y_pred = evaluate(model, val_loader, loss_fn, self.device)

        accuracy, precision, recall, f1, cm = compute_metrics(y_true, y_pred)

        print("\nValidation Metrics:\n")

        print(f"F1-Score:  {f1:.4f}")
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")

        print("\nConfusion Matrix:\n")
        print_nice_confusion_matrix(cm, CLASSES)
    

    def eval(self):

        # -------------------------------
        # Evaluation Phase
        # -------------------------------

        if self.cfg.get('use_pretrained', False):
            from models.resnet import ResNetTransfer
            model = ResNetTransfer(num_classes=len(CLASSES), freeze_layers=self.cfg.get('freeze_layers', False))
        else:
            from models.cnn import CNN
            model = CNN(num_classes=len(CLASSES))

        model_path = os.path.join(self.model_dir, "dep_esp.pth")

        if not os.path.exists(model_path):
            print(f"\nModel file {model_path} does not exist. Exiting evaluation phase.\n")
            return
        
        model.load_state_dict(torch.load(model_path, map_location=self.device))

        model.to(self.device)
        
        model.eval()
        
        if check_camera_available():

            print("\nCamera detected. Running live evaluation.")

            face_detector = FaceDetector()
            cap = cv2.VideoCapture(0)

            while True:

                ret, frame = cap.read()

                if not ret:
                    break

                faces = face_detector.detect_faces(frame)

                for (x, y, w, h) in faces:

                    face_img = frame[y:y+h, x:x+w]
                    face_img = cv2.resize(face_img, (64, 64))
                    face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

                    face_tensor = self.transform(torch.from_numpy(face_rgb)).unsqueeze(0).to(self.device)
                    output = model(face_tensor)

                    _, predicted = torch.max(output, 1)
                    label = CLASSES[predicted.item()]

                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

                    cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                
                cv2.imshow("Depresso-Espresso - Live Evaluation (Press 'q' to quit)", frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cap.release()

            cv2.destroyAllWindows()

        else:
            
            print("\nNo camera detected. Running evaluation on local images.")
            
            test_images_dir = os.path.join(self.data_dir, "test")
            
            if not os.path.isdir(test_images_dir):
                print(f"Test folder '{test_images_dir}' not found. Exiting evaluation phase.")
                return
            
            for image_name in os.listdir(test_images_dir):
                
                image_path = os.path.join(self.data_dir, "test", image_name)
                image = cv2.imread(image_path)
                
                if image is None:
                    continue
                
                faces = FaceDetector().detect_faces(image)
                
                for (x, y, w, h) in faces:
                    
                    face_img = image[y:y+h, x:x+w]
                    face_img = cv2.resize(face_img, (64, 64))
                    face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                    
                    face_tensor = self.transform(torch.from_numpy(face_rgb)).unsqueeze(0).to(self.device)
                    output = model(face_tensor)
                    
                    _, predicted = torch.max(output, 1)
                    label = CLASSES[predicted.item()]
                    
                    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    
                    cv2.putText(image, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                
                cv2.imshow("Depresso-Espresso - Image Evaluation (Press any key for next)", image)
                
                cv2.waitKey(0)

            cv2.destroyAllWindows()


def print_nice_confusion_matrix(cm, labels):

    """
    Prints the confusion matrix in a modern, minimalist table format.
    """

    cm = np.array(cm)

    row_labels = labels
    col_labels = labels

    table = []
    header = [""] + col_labels

    table.append(header)

    for i, row in enumerate(cm):
        table.append([row_labels[i]] + [str(x) for x in row])

    col_widths = [max(len(item) for item in col) for col in zip(*table)]

    top_border = "┌" + "┬".join("─" * (w + 2) for w in col_widths) + "┐"

    header_sep = "├" + "┼".join("─" * (w + 2) for w in col_widths) + "┤"

    bottom_border = "└" + "┴".join("─" * (w + 2) for w in col_widths) + "┘"
    

    def format_row(row):
        return "│" + "│".join(f" {item:^{w}} " for item, w in zip(row, col_widths)) + "│"
    

    print(top_border)

    print(format_row(table[0]))

    print(header_sep)

    for row in table[1:]:
        print(format_row(row))

    print(bottom_border)
