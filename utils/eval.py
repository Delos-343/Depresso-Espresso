from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix

def compute_metrics(y_true, y_pred, average='macro'):
    
    accuracy = accuracy_score(y_true, y_pred)
    
    precision = precision_score(y_true, y_pred, average=average, zero_division=0)
    
    recall = recall_score(y_true, y_pred, average=average, zero_division=0)
    
    f1 = f1_score(y_true, y_pred, average=average, zero_division=0)
    
    cm = confusion_matrix(y_true, y_pred)
    
    return accuracy, precision, recall, f1, cm
