import cv2

face_ref = cv2.CascadeClassifier("face_ref.xml")
camera = cv2.VideoCapture(0)

def face_detection():
    # Implementasi deteksi wajah
    while True:
        ret, frame = camera.read()
        if not ret:
            break
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_ref.detectMultiScale(gray, 1.3, 5)
        
        # Draw rectangles around faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # Display the frame
        cv2.imshow('Face Detection', frame)
        
        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release camera and close windows
    camera.release()
    cv2.destroyAllWindows()

def drawer_box():
    # Implementasi drawer box
    print("Drawer box function called")
    pass

def main():
    print("Starting face detection system..."   )
    face_detection()

if __name__ == "__main__":
    main()

