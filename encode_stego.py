import os
import numpy as np
from PIL import Image
import random

def encode(path, message, convert_to_png=True):
    with Image.open(path) as img:
        if convert_to_png:
            # convert JPG ang JPEG to PNG to ensure lossless encoding
            if path.lower().endswith(('.jpg', '.jpeg')):
                path = path.rsplit('.', 1)[0] + '.png'
                
        width, height = img.size
        # get pixel data
        array = np.array(list(img.getdata()))

        if img.mode == 'RGB':
            n = 3
        elif img.mode == 'RGBA':
            n = 4
        total_pixels = array.size//n

        # print binary message
        binary = ''.join([format(ord(i), "08b") for i in message])
        # print("Message in binary:", binary)

        # ADD PREAMBLE TO DETERMINE MESSAGE LENGTH
        message += "pr34mb13"
        b_message = ''.join([format(ord(i), "08b") for i in message])
        req_pixels = len(b_message)
        print("Message after adding preamble:", b_message)


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

            array=array.reshape(height, width, n)     
            enc_img = Image.fromarray(array.astype('uint8'), img.mode)
            enc_img.save(path)
            print("Message hidden in:", path)
            


def generate_random_message(word_count=5):
    words = ["hello", "world", "encode", "message", "image", "random", "text", "secret", "hidden"]
    message = ' '.join(random.choices(words, k=word_count))
    return message


def main():
    src = "dataset/stego"
    
    for file in os.listdir(src):
        if file.lower().endswith(('.png', '.jpg', '.jpeg')): 
            path = os.path.join(src, file) 
            message = generate_random_message()
            # print(f"Encoding message: \"{message}\" in {file}")
            encode(path, message)
            if file.lower().endswith(('.jpg', '.jpeg')):
                os.remove(path)
    
    print("Completed")
    

if __name__ == "__main__":
    main()