
from PIL import Image
from skimage.feature import hog
from skimage.color import rgb2gray
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import os
import numpy as np
import joblib
folder_path = "C:\\Users\\PC\\Desktop\\flower classification\\archive\\flowers\\"

# Get the five flower classes
classes = os.listdir(folder_path)
print("Classes :", classes)
# Dataset
dataset = []

# Process every flower class
for class_name in classes:

    class_path = folder_path + class_name + "\\"
    images = os.listdir(class_path)

    print("Processing:", class_name)

    # Process every image
    for image_name in images:

        img = Image.open(class_path + image_name).convert("RGB")

        # Resize the image
        img = img.resize((32,32))
        pixels = np.array(img)

        avgr = np.mean(pixels[:, :, 0])
        avgg = np.mean(pixels[:, :, 1])
        avgb = np.mean(pixels[:, :, 2])             
        # Convert image to NumPy array
        img_array = np.array(img)

        # Convert RGB to grayscale
        gray_image = rgb2gray(pixels)

# Extract HOG features
        features = hog(gray_image)

# Combine RGB and HOG
        features = np.concatenate((
        [avgr, avgg, avgb],features
          ))
        # Store features and label
        flower = {
            "features": features,
            "label": class_name
        }

        dataset.append(flower)

# Prepare X and Y AFTER processing all images
X = []
Y = []

for flower in dataset:
    X.append(flower["features"])
    Y.append(flower["label"])

# Split the dataset
trainX, testX, trainY, testY = train_test_split(
    X, Y,
    test_size=0.2,
    random_state=42,
    stratify=Y
)

# Train KNN
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(trainX, trainY)
joblib.dump(model, "flower_model.pkl")
# Test the model
predictions = model.predict(testX)

# Calculate accuracy
accuracy = accuracy_score(testY, predictions)
print("Accuracy:", accuracy)

# Load a new image
img = Image.open("C:\\Users\\PC\\Desktop\\ML project\\test_flower.jpg").convert("RGB")

# Resize to the same size used during training
img = img.resize((32, 32))

# Extract RGB features
pixels = np.array(img)

avgr = np.mean(pixels[:, :, 0])
avgg = np.mean(pixels[:, :, 1])
avgb = np.mean(pixels[:, :, 2])

# Extract HOG features
gray_image = rgb2gray(pixels)
features = hog(gray_image)

# Combine RGB and HOG
features = np.concatenate(([avgr, avgg, avgb], features))

# Predict
prediction = model.predict([features])

print("Predicted flower:", prediction[0])