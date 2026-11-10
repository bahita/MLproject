
from PIL import Image
from skimage.feature import hog
from skimage.color import rgb2gray
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import os
import numpy as np

folder_path = "C:\\Users\\PC\\Desktop\\flower classification\\archive\\flowers\\"

# Get the five flower classes
classes = os.listdir(folder_path)
print("Classes:", classes)

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
        img = img.resize((64, 64))

        # Convert image to NumPy array
        img_array = np.array(img)

        # Convert RGB to grayscale
        gray_image = rgb2gray(img_array)

        # Extract HOG features
        features = hog(gray_image)

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
model = KNeighborsClassifier(n_neighbors=3)
model.fit(trainX, trainY)

# Test the model
predictions = model.predict(testX)

# Calculate accuracy
accuracy = accuracy_score(testY, predictions)
print("Accuracy:", accuracy)

