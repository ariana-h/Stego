import os
import numpy as np
from PIL import Image

def encode(input_path, message, output_path):
    with Image.open(input_path) as img:
        img = Image.open(input_path, 'r')
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
        print("Messgage in binary:", binary)

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
                        array[p][q] = int(bin(array[p][q])[2:-1] + b_message[index], 2)
                        index += 1

            array=array.reshape(height, width, n)
            enc_img = Image.fromarray(array.astype('uint8'), img.mode)
            enc_img.save(output_path)
            print("Message hidden in:", output_path)


def main():
    src = "resized"

    if not os.path.exists('encoded'):
        os.makedirs('encoded') 
    dest = "encoded"

    print("Enter a message to encode:")
    message = input()


    for file in os.listdir(src):
        if file.lower().endswith(('.png', '.jpg', '.jpeg')): 
            input_path = os.path.join(src, file) 
            output_path = os.path.join(dest, file) 
            encode(input_path, message, output_path)

if __name__ == "__main__":
    main()