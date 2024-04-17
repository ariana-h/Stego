import os
import numpy as np
from PIL import Image
import random
import nltk

nltk.download('words')
from nltk.corpus import words

def encode(path, message, convert_to_png=True):
    with Image.open(path) as img:
        if convert_to_png:
            if path.lower().endswith(('.jpg', '.jpeg')):    # convert JPG ang JPEG to PNG to ensure lossless encoding
                path = path.rsplit('.', 1)[0] + '.png'
                
        width, height = img.size
        # get pixel data
        array = np.array(list(img.getdata()))

        total_pixels = array.size//3     # assuming img.mode == 'RGB' and ignoring 'RGBA'

        binary = ''.join([format(ord(i), "08b") for i in message])
        # print("Message in binary:", binary)

        # ADD PREAMBLE TO DETERMINE MESSAGE LENGTH
        message += "pr34mb13"
        b_message = ''.join([format(ord(i), "08b") for i in message])
        req_pixels = len(b_message)
        # print("Message after adding preamble:", b_message)


        if req_pixels > total_pixels:
            print("ERROR: Need larger file size")
        else:
            
            index=0
            for p in range(total_pixels):
                for q in range(0, 3):
                    if index < req_pixels:
                        pixel_value = array[p][q]
                        modified_pixel = (pixel_value & 0xFE) | int(b_message[index])
                        array[p][q] = modified_pixel
                        index += 1

            array=array.reshape(height, width, 3)     
            enc_img = Image.fromarray(array.astype('uint8'), img.mode)
            enc_img.save(path)
            print("Message hidden in:", path)


def generate_random_message(word_count=150):
    '''
    words = ["hello", "world", "encode", "message", "image", "random", "text", "secret", "hidden", 
             "encryption", "watermarking", "steganography", "digital signature", "authentication", 
             "integrity", "confidentiality", "access", "control", "biometrics", "forensic", "analysis", 
             "tamper-proof", "secure", "transmission", "data", "hiding", "privacy", "storage", "threat", 
             "detection", "malware", "intrusion", "anti-counterfeiting", "hashing", "protocols", 
             "two-factor", "deletion", "recognition", "facial", "backups", "audit", "vulnerability", 
             "assessment", "testing", "firewalls", "cybersecurity", "network", "endpoint", "identity", 
             "verification", "sockets", "public", "key", "infrastructure", "sharing", "incident", "response", 
             "intelligence", "tokens", "cryptography", "boot", "printing", "digital", "watermark"] # out of 65
    message = ' '.join(random.choices(words, k=word_count))
    '''
    word_list = words.words()
    message = ' '.join(random.choices(word_list, k=word_count))
    return message


def main():
    src = "dataset/stego"
    total = 0
    
    for file in os.listdir(src):
        if file.lower().endswith(('.png', '.jpg', '.jpeg')): 
            path = os.path.join(src, file) 
            message = generate_random_message()
            # print(f"Encoding message: \"{message}\" in {file}")
            total += 1
            encode(path, message)
            if file.lower().endswith(('.jpg', '.jpeg')):
                os.remove(path)
    
    print("Completed encoding ", total, " images.")
    

if __name__ == "__main__":
    main()