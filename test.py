import os
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

total_pos, total_neg, tp, fp, tn, fn = 0, 0, 0, 0, 0, 0

model = tf.keras.models.load_model('best_model.h5')

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    return image

def log(message):
    with open("output_log.txt", "a") as file:
        file.write(message + "\n")

def detect_message(image_path):
    global total_pos, total_neg, tp, fp, tn, fn
    
    preprocessed_image = preprocess_image(image_path)
    preprocessed_image = np.expand_dims(preprocessed_image, axis=0)
    
    prediction = model.predict(preprocessed_image)
    

    if prediction[0][0] > 0.5:
        log(f"Steganography detected in {image_path}")
        total_pos += 1
        if 'non-stego' in image_path:
            fp += 1 
            log(f"False positive: {fp} / {total_pos}")
        else:
            tp += 1
            log(f"True positive: {tp} / {total_pos}")
        decode(image_path)
        
    else:
        log(f"No message detected in {image_path}")
        total_neg += 1
        if 'non-stego' in image_path:
            tn += 1
            log(f"True negative: {tn} / {total_neg}")
        else:
            fn += 1
            log(f"False negative: {fn} / {total_neg}")
            decode(image_path) 
            

def iterate_files(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            detect_message(file_path)
        
def decode(input_path):
    with Image.open(input_path) as img:
        array = np.array(list(img.getdata()))

        total_pixels = array.size//3     # assuming img.mode == 'RGB' and ignoring 'RGBA'

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
            log(f"Hidden Message: {message[:-8]}")
        else:  
            log(f"No Hidden Message Found")  
            

def main():
    stego_folder = "dataset/test/stego"
    non_stego_folder = "dataset/test/non-stego"
    
    if os.path.exists("output_log.txt"):
        os.remove("output_log.txt")

    print("Detecting steganography in stego images:")
    log("Detecting steganography in stego images:")
    iterate_files(stego_folder)

    print("\nDetecting steganography in non-stego images:")
    log("\nDetecting steganography in non-stego images:")
    iterate_files(non_stego_folder)

    accuracy = (tp+tn)/(total_pos+total_neg)
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    
    print("==========================================================")
    print("True positive:", tp/total_pos)
    print("False positive:", fp/total_pos)
    print("True negative:", tn/total_neg)
    print("False negative:", fn/total_neg)
        
    print("----------------------------------------------------------")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")

if __name__ == "__main__":
    main()
