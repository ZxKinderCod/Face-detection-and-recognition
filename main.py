import cv2
import os
import numpy as np
import pickle
from collections import defaultdict
import time

# Original face detection setup
face_ref = cv2.CascadeClassifier("face_ref.xml")
camera = cv2.VideoCapture(0)

class ImprovedFaceRecognizer:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.face_encodings = []
        self.face_labels = []
        self.label_names = {}
        self.is_trained = False
        
        # Enhanced accuracy parameters
        self.recognition_history = {}
        self.history_size = 20  # Increased for better stability
        self.confidence_threshold = 0.35  # More strict threshold
        self.min_confidence_for_display = 75  # Higher minimum confidence
        self.min_consistent_predictions = 12  # Minimum consistent predictions needed
        
    def extract_face_features(self, face_gray):
        """Extract improved features from face with better preprocessing"""
        # Resize face to standard size
        face_resized = cv2.resize(face_gray, (100, 100))
        
        # Better preprocessing
        face_equalized = cv2.equalizeHist(face_resized)
        
        # Apply Gaussian blur to reduce noise
        face_smooth = cv2.GaussianBlur(face_equalized, (3, 3), 0)
        
        # Multiple feature extraction methods
        # 1. Histogram features with more bins
        hist = cv2.calcHist([face_smooth], [0], None, [128], [0, 256])  # Increased bins
        hist = hist.flatten()
        hist = hist / (hist.sum() + 1e-7)
        
        # 2. LBP-like features
        lbp_features = self.extract_lbp_features(face_smooth)
        
        # 3. Gradient features (simple edge detection)
        gradient_features = self.extract_gradient_features(face_smooth)
        
        # Combine all features
        features = np.concatenate([hist, lbp_features, gradient_features])
        
        return features
    
    def extract_gradient_features(self, image):
        """Extract gradient-based features"""
        # Calculate gradients
        grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        
        # Calculate gradient magnitude
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Create histogram of gradient magnitudes
        grad_hist, _ = np.histogram(magnitude.flatten(), bins=32, range=(0, 255))
        grad_hist = grad_hist / (grad_hist.sum() + 1e-7)
        
        return grad_hist
    
    def extract_lbp_features(self, image):
        """Extract improved Local Binary Pattern features"""
        rows, cols = image.shape
        lbp_features = []
        
        # More dense sampling for better features
        for i in range(1, rows-1, 8):  # Reduced step size from 10 to 8
            for j in range(1, cols-1, 8):
                center = image[i, j]
                pattern = 0
                
                neighbors = [(i-1,j-1), (i-1,j), (i-1,j+1), (i,j+1),
                           (i+1,j+1), (i+1,j), (i+1,j-1), (i,j-1)]
                
                for idx, (ni, nj) in enumerate(neighbors):
                    if ni >= 0 and ni < rows and nj >= 0 and nj < cols:
                        if image[ni, nj] >= center:
                            pattern |= (1 << idx)
                
                lbp_features.append(pattern)
        
        # Create histogram with more bins for better discrimination
        lbp_hist, _ = np.histogram(lbp_features, bins=64, range=(0, 256))  # Increased from 32
        lbp_hist = lbp_hist / (lbp_hist.sum() + 1e-7)
        
        return lbp_hist
    
    def prepare_training_data(self, data_folder="data/data_wajah"):
        """Prepare training data from folders"""
        print("🚀 Preparing training data...")
        
        if not os.path.exists(data_folder):
            print(f"❌ Folder {data_folder} tidak ditemukan!")
            return False
        
        label = 0
        total_faces = 0
        
        for person_name in os.listdir(data_folder):
            person_folder = os.path.join(data_folder, person_name)
            
            if not os.path.isdir(person_folder):
                continue
                
            print(f"👤 Processing {person_name}...")
            self.label_names[label] = person_name
            person_faces = 0
            
            for image_name in os.listdir(person_folder):
                if image_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_path = os.path.join(person_folder, image_name)
                    image = cv2.imread(image_path)
                    
                    if image is None:
                        continue
                    
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    faces = self.face_cascade.detectMultiScale(
                        gray, 
                        scaleFactor=1.1,
                        minNeighbors=5,
                        minSize=(50, 50)
                    )
                    
                    for (x, y, w, h) in faces:
                        face = gray[y:y+h, x:x+w]
                        features = self.extract_face_features(face)
                        
                        self.face_encodings.append(features)
                        self.face_labels.append(label)
                        person_faces += 1
                        total_faces += 1
                        print(f"   ✅ {image_name}: Face processed")
                        break
            
            print(f"   📊 {person_name}: {person_faces} faces processed")
            label += 1
        
        print(f"📋 Total: {total_faces} faces from {len(self.label_names)} persons")
        return total_faces > 0
    
    def train_model(self):
        """Train improved face recognition model"""
        if not self.prepare_training_data():
            return False
        
        print("🧠 Training model...")
        
        self.face_encodings = np.array(self.face_encodings)
        self.face_labels = np.array(self.face_labels)
        
        os.makedirs("models", exist_ok=True)
        model_data = {
            'encodings': self.face_encodings,
            'labels': self.face_labels,
            'label_names': self.label_names
        }
        
        with open("models/improved_face_model.pkl", "wb") as f:
            pickle.dump(model_data, f)
        
        self.is_trained = True
        print("✅ Training completed and model saved!")
        return True
    
    def load_model(self):
        """Load trained model"""
        try:
            with open("models/improved_face_model.pkl", "rb") as f:
                model_data = pickle.load(f)
            
            self.face_encodings = model_data['encodings']
            self.face_labels = model_data['labels']
            self.label_names = model_data['label_names']
            self.is_trained = True
            
            print(f"✅ Model loaded with {len(self.label_names)} persons")
            return True
        except FileNotFoundError:
            print("❌ Model tidak ditemukan. Jalankan training dulu!")
            return False
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def predict_face_stable(self, face_gray, face_id):
        """Predict face with improved stability and accuracy"""
        if not self.is_trained:
            return "Unknown", 0
        
        features = self.extract_face_features(face_gray)
        distances = []
        
        # Calculate distances with improved similarity measure
        for encoding in self.face_encodings:
            # Use both cosine and euclidean distance
            dot_product = np.dot(features, encoding)
            norm_a = np.linalg.norm(features)
            norm_b = np.linalg.norm(encoding)
            
            if norm_a == 0 or norm_b == 0:
                cosine_distance = 1
            else:
                cosine_sim = dot_product / (norm_a * norm_b)
                cosine_distance = 1 - cosine_sim
            
            # Also calculate euclidean distance (normalized)
            euclidean_distance = np.linalg.norm(features - encoding)
            euclidean_distance = euclidean_distance / np.sqrt(len(features))
            
            # Combine both distances (weighted average)
            combined_distance = 0.7 * cosine_distance + 0.3 * euclidean_distance
            distances.append(combined_distance)
        
        distances = np.array(distances)
        min_distance_idx = np.argmin(distances)
        min_distance = distances[min_distance_idx]
        
        if min_distance < self.confidence_threshold:
            label = self.face_labels[min_distance_idx]
            name = self.label_names[label]
            confidence = max(0, 100 - (min_distance * 150))  # Adjusted scaling
        else:
            name = "Unknown"
            confidence = 0
        
        # Enhanced temporal smoothing
        if face_id not in self.recognition_history:
            self.recognition_history[face_id] = []
        
        self.recognition_history[face_id].append((name, confidence, min_distance))
        
        if len(self.recognition_history[face_id]) > self.history_size:
            self.recognition_history[face_id].pop(0)
        
        # Improved voting system with confidence weighting
        name_scores = defaultdict(float)
        total_weight = 0
        
        for hist_name, hist_conf, hist_dist in self.recognition_history[face_id]:
            # Weight recent predictions more heavily
            weight = 1.0 + (len(self.recognition_history[face_id]) - 
                           self.recognition_history[face_id].index((hist_name, hist_conf, hist_dist))) * 0.1
            
            # Also weight by confidence
            confidence_weight = hist_conf / 100.0 if hist_conf > 0 else 0.1
            final_weight = weight * confidence_weight
            
            name_scores[hist_name] += final_weight
            total_weight += final_weight
        
        # Get best prediction
        if total_weight > 0:
            best_name = max(name_scores, key=name_scores.get)
            best_score = name_scores[best_name] / total_weight
            
            # Calculate average confidence for the best name
            best_confidences = [conf for n, conf, _ in self.recognition_history[face_id] if n == best_name]
            avg_confidence = np.mean(best_confidences) if best_confidences else 0
            
            # Only return recognized name if confidence is high enough
            if best_name != "Unknown" and avg_confidence >= self.min_confidence_for_display:
                return best_name, avg_confidence
        
        return "Unknown", 0

# Global recognizer instance
recognizer = ImprovedFaceRecognizer()

def remove_overlapping_faces(faces, overlap_threshold=0.3):
    """Remove overlapping face detections"""
    if len(faces) == 0:
        return faces
    
    # Convert to list for easier manipulation
    faces_list = list(faces)
    keep = [True] * len(faces_list)
    
    for i in range(len(faces_list)):
        if not keep[i]:
            continue
            
        x1, y1, w1, h1 = faces_list[i]
        area1 = w1 * h1
        
        for j in range(i + 1, len(faces_list)):
            if not keep[j]:
                continue
                
            x2, y2, w2, h2 = faces_list[j]
            area2 = w2 * h2
            
            # Calculate intersection
            x_left = max(x1, x2)
            y_top = max(y1, y2)
            x_right = min(x1 + w1, x2 + w2)
            y_bottom = min(y1 + h1, y2 + h2)
            
            if x_right > x_left and y_bottom > y_top:
                intersection = (x_right - x_left) * (y_bottom - y_top)
                union = area1 + area2 - intersection
                overlap = intersection / union if union > 0 else 0
                
                # If overlap is significant, keep the larger face
                if overlap > overlap_threshold:
                    if area1 >= area2:
                        keep[j] = False
                    else:
                        keep[i] = False
                        break
    
    # Return filtered faces
    return np.array([faces_list[i] for i in range(len(faces_list)) if keep[i]])

def face_detection():
    """Combined face detection and recognition function"""
    # Try to load recognition model (optional)
    has_recognition = recognizer.load_model()
    
    # Set camera properties dengan resolusi lebih tinggi
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    camera.set(cv2.CAP_PROP_FPS, 30)
    
    # Set window properties untuk full screen
    cv2.namedWindow('Face Detection & Recognition' if has_recognition else 'Face Detection', cv2.WINDOW_AUTOSIZE)
    
    # Face tracking variables for recognition
    face_id_counter = 0
    active_faces = {}
    frame_counter = 0
    stable_faces = {}  # For stable face tracking
    
    print("🎥 Starting face detection" + (" with recognition" if has_recognition else "") + "...")
    print("💡 Press 'q' to quit")
    
    while True:
        ret, frame = camera.read()
        if not ret:
            break
        
        frame_counter += 1
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces with improved parameters
        faces = face_ref.detectMultiScale(
            gray, 
            scaleFactor=1.1,  # More gradual scaling
            minNeighbors=8,   # Higher threshold to reduce false positives
            minSize=(80, 80), # Larger minimum size
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        # If no faces detected with original, try with built-in cascade
        if len(faces) == 0 and has_recognition:
            faces = recognizer.face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.1,
                minNeighbors=8,
                minSize=(80, 80),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
        
        # Remove overlapping detections
        faces = remove_overlapping_faces(faces)
        
        # Process faces only every 3rd frame for stability
        if frame_counter % 3 == 0:
            new_stable_faces = {}
            
            for i, (x, y, w, h) in enumerate(faces):
                face_center = (x + w//2, y + h//2)
                face_id = None
                
                # Find closest existing stable face
                min_dist = float('inf')
                for fid, (prev_center, prev_rect, _) in stable_faces.items():
                    dist = np.sqrt((face_center[0] - prev_center[0])**2 + 
                                 (face_center[1] - prev_center[1])**2)
                    if dist < min_dist and dist < 150:  # Increased threshold
                        min_dist = dist
                        face_id = fid
                
                if face_id is None:
                    face_id = face_id_counter
                    face_id_counter += 1
                
                # Store stable face info
                face_rect = (x, y, w, h)
                
                if has_recognition:
                    face_gray = gray[y:y+h, x:x+w]
                    name, confidence = recognizer.predict_face_stable(face_gray, face_id)
                    new_stable_faces[face_id] = (face_center, face_rect, (name, confidence))
                else:
                    new_stable_faces[face_id] = (face_center, face_rect, None)
            
            # Update stable faces
            stable_faces = new_stable_faces
        
        # Draw stable faces
        for face_id, (center, (x, y, w, h), recognition_data) in stable_faces.items():
            if has_recognition and recognition_data:
                name, confidence = recognition_data
                
                # Choose color and label based on recognition
                if name == "Unknown":
                    color = (0, 0, 255)  # Red
                    label = f"{name}"
                else:
                    color = (0, 255, 0)  # Green
                    label = f"{name}"
                
                # Draw rectangle and label
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
                
                # Draw label background
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                cv2.rectangle(frame, (x, y-35), (x + label_size[0] + 15, y), color, -1)
                cv2.putText(frame, label, (x+7, y-12), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            else:
                # Simple detection mode
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 3)
        
        # Display the frame dengan window yang sudah dibuat
        window_title = 'Face Detection & Recognition' if has_recognition else 'Face Detection'
        cv2.imshow(window_title, frame)
        
        # Resize window jika diperlukan (optional - untuk full screen)
        # cv2.setWindowProperty(window_title, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    # Release camera and close windows
    camera.release()
    cv2.destroyAllWindows()
    print("👋 Detection stopped")

def train_model():
    """Train face recognition model"""
    success = recognizer.train_model()
    if success:
        print("✅ Training berhasil! Model siap digunakan untuk recognition.")
    else:
        print("❌ Training gagal! Periksa folder data/data_wajah/")

def main():
    """Main function - keeping original structure"""
    print("Starting face detection system...")
    print("=" * 50)
    print("🎯 Combined Face Detection & Recognition System")
    print("=" * 50)
    
    while True:
        print("\nMenu:")
        print("1. Face Detection & Recognition")
        print("2. Train Recognition Model")
        print("3. Exit")
        
        choice = input("\nPilih menu (1-3): ").strip()
        
        if choice == "1":
            print("🔍 Starting combined face detection & recognition...")
            face_detection()
        
        elif choice == "2":
            print("🧠 Training recognition model...")
            train_model()
        
        elif choice == "3":
            print("👋 Goodbye!")
            break
        
        else:
            print("❌ Pilihan tidak valid!")

if __name__ == "__main__":
    main()