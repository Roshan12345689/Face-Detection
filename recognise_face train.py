import os
import face_recognition
import numpy as np
from sklearn.svm import SVC
import pickle

# ✅ Set correct dataset path (Use raw string to avoid Unicode errors)
TRAIN_DIR = TRAIN_DIR = r"D:\FACE EMOTION BOTH LIVE AND IMAGE(DP-3)\Ai image organizer\face recognition\dataset_train\train"


# ✅ Function to load images and extract encodings
def load_images_from_directory(directory):
    X = []
    y = []

    for person_name in os.listdir(directory):
        person_path = os.path.join(directory, person_name)
        if not os.path.isdir(person_path):  # ✅ Skip if it's not a folder
            continue
        
        for img_name in os.listdir(person_path):
            img_path = os.path.join(person_path, img_name)
            if not os.path.isfile(img_path):  # ✅ Skip if it's not a file
                continue
            
            try:
                image = face_recognition.load_image_file(img_path)
                encoding = face_recognition.face_encodings(image)

                if encoding:  # ✅ Ensure at least one face is found
                    X.append(encoding[0])
                    y.append(person_name)
                else:
                    print(f"[WARNING] No face found in {img_path}. Skipping...")

            except Exception as e:
                print(f"[ERROR] Failed to process {img_path}: {e}")

    return np.array(X), np.array(y)

# ✅ Load training data
print("[INFO] Loading training images...")
X_train, y_train = load_images_from_directory(TRAIN_DIR)

# ✅ Debugging: Check dataset
print("Labels in y_train:", set(y_train))
print("Total classes:", len(set(y_train)))
print("Number of samples:", len(y_train))

# ✅ Ensure we have more than 1 class
if len(set(y_train)) < 2:
    raise ValueError("[ERROR] Not enough classes to train. Add more labeled images.")

# ✅ Initialize and train SVM classifier
print("[INFO] Training SVM classifier...")
clf = SVC(kernel='linear', probability=True)
clf.fit(X_train, y_train)

# ✅ Save trained model
with open("face_recognition_model.pkl", "wb") as f:
    pickle.dump(clf, f)

print("[INFO] Training complete. Model saved as 'face_recognition_model.pkl'.")
