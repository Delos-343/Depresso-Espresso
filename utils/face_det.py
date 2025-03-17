import cv2

class FaceDetector:
   
    def __init__(self, cascade_path=None):
        
        # Use OpenCV's default frontal face cascade if none provided
        if cascade_path is None:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
    
    def detect_faces(self, image):
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        
        return faces
