import torch
import cv2

def check_camera_available():
    
    cap = cv2.VideoCapture(0)
    
    if cap is None or not cap.isOpened():
        return False
    
    cap.release()
    
    return True


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    
    model.train()
    running_loss = 0.0
    
    for images, labels in dataloader:
        
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        
        loss = criterion(outputs, labels)
        loss.backward()
        
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    
    epoch_loss = running_loss / len(dataloader.dataset)
    
    return epoch_loss


def evaluate(model, dataloader, criterion, device):
    
    model.eval()
    running_loss = 0.0
    correct = 0
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        
        for images, labels in dataloader:
            images = images.to(device)
            
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            
            correct += torch.sum(preds == labels.data)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(dataloader.dataset)
    
    accuracy = correct.double() / len(dataloader.dataset)
    
    return epoch_loss, accuracy.item(), all_labels, all_preds
