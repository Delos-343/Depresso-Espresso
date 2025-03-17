import os
import cv2
import torch
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
                print("Could not set GPU memory fraction:", e)
        
        # Define image transformations: resize to 64x64, convert to tensor, normalize
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        
        # Use config values with defaults if not provided
        self.data_dir = cfg.get('data_dir', "data")
        self.model_dir = cfg.get('model_dir', "model")
        self.batch_size = cfg.get('batch_size', 32)
        self.learning_rate = cfg.get('learning_rate', 0.001)
        self.epochs = cfg.get('epochs', 10)


    def train(self):
        
        # -------------------------------
        # Training Phase
        # -------------------------------
        
        dataset = CustomImageDataset(root_dir=self.data_dir, transform=self.transform)
        
        if len(dataset) == 0:
            print("No images found in the dataset. Exiting training phase.")
            return
        
        # Split dataset into training (80%) and validation (20%)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        # Initialize the CNN model, loss function, and optimizer
        model = CNN(num_classes=len(CLASSES))
        model.to(self.device)
        
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        
        print("\n" + "Starting training... \n")
        
        for epoch in range(self.epochs):
            
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, self.device)
            
            val_loss, val_accuracy, y_true, y_pred = evaluate(model, val_loader, criterion, self.device)
            
            print(f"Epoch {epoch+1}/{self.epochs} - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.4f}")
        
        # Ensure the model sub-directory exists and save the trained model
        os.makedirs(self.model_dir, exist_ok=True)
        model_path = os.path.join(self.model_dir, "model.pth")
        torch.save(model.state_dict(), model_path)
        
        print(f"Training complete. Model saved as {model_path}.")
        
        # -------------------------------
        # Compute and Display Metrics on Validation Set
        # -------------------------------
        
        val_loss, val_accuracy, y_true, y_pred = evaluate(model, val_loader, criterion, self.device)
        accuracy, precision, recall, f1, cm = compute_metrics(y_true, y_pred)
        
        print("\nValidation Metrics:")
        
        print(f"F1-Score:  {f1:.4f}")
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")

        print("Confusion Matrix:")
        print(cm)
    

    def eval(self):
        
        # -------------------------------
        # Evaluation Phase
        # -------------------------------
        
        # Initialize model
        model = CNN(num_classes=len(CLASSES))
        model_path = os.path.join(self.model_dir, "model.pth")
        
        if not os.path.exists(model_path):
            print("\n" + f"Model file {model_path} does not exist. Exiting evaluation phase. \n")
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
                    
                    # Convert BGR to RGB, then to tensor using the same transformation as training
                    face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                    face_tensor = self.transform(torch.from_numpy(face_rgb)).unsqueeze(0).to(self.device)
                    
                    output = model(face_tensor)
                    _, predicted = torch.max(output, 1)
                    
                    label = CLASSES[predicted.item()]
                    
                    # Draw bounding box and label on the frame
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
                
                image_path = os.path.join(test_images_dir, image_name)
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
