import cv2

def check_camera_available():
    
    cap = cv2.VideoCapture(0)
    
    if cap is None or not cap.isOpened():
        return False
    
    cap.release()
    
    return True
