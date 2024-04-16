import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import numpy as np
import cv2
import os

# Define constants
IMAGE_SIZE = (128, 128)  # Input image size
BATCH_SIZE = 32
EPOCHS = 400

# Function to preprocess image
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, IMAGE_SIZE)  # Resize image
    image = image / 255.0  # Normalize pixel values
    return image

# Load images and labels
def load_data(data_dir):
    images = []
    labels = []
    for foldername in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, foldername)
        label = 1 if foldername == 'stego' else 0  # Assuming stego images are in 'stego' folder
        for filename in os.listdir(folder_path):
            image_path = os.path.join(folder_path, filename)
            images.append(preprocess_image(image_path))
            labels.append(label)
    return np.array(images), np.array(labels)

# Load data
data_dir = 'dataset/train'
images, labels = load_data(data_dir)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Define CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(X_test, y_test))

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Test accuracy: {test_acc}')

# Save the model
model.save('steganalysis_model.h5')
