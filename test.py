import os
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

model = tf.keras.models.load_model('steganalysis_model.h5')

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (128, 128)) 
    image = image / 255.0 
    return image

def detect_message(image_path):
    preprocessed_image = preprocess_image(image_path)
    preprocessed_image = np.expand_dims(preprocessed_image, axis=0)
    
    prediction = model.predict(preprocessed_image)
    if prediction[0][0] > 0.5:
        print(f"Steganography detected in {image_path}")
        # Additional decoding logic here if required
        decode(image_path)
    else:
        print(f"No message detected in {image_path}")

def iterate_files(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            detect_message(file_path)
        
def decode(input_path):
    with Image.open(input_path) as img:
        array = np.array(list(img.getdata()))

        if img.mode == 'RGB':
            n = 3
        elif img.mode == 'RGBA':
            n = 4
        total_pixels = array.size//n

        hidden_bits = ""
        for p in range(total_pixels):
            for q in range(0, 3):
                hidden_bits += (bin(array[p][q])[2:][-1])
                
        hidden_bytes = [hidden_bits[i:i+8] for i in range(0, len(hidden_bits), 8)]

        message = ""
        
        for i in range(len(hidden_bytes)):
            if message[-8:] == "pr34mb13":
                break
            else:
                message += chr(int(hidden_bytes[i], 2))
        if "pr34mb13" in message:
            print("Hidden Message:", message[:-8])
        else:
            print("No Hidden Message Found")   
            

def main():
    stego_folder = "dataset/test/stego"
    non_stego_folder = "dataset/test/non-stego"

    print("Detecting steganography in stego images:")
    iterate_files(stego_folder)

    print("\nDetecting steganography in non-stego images:")
    iterate_files(non_stego_folder)

    print("Completed")

if __name__ == "__main__":
    main()
