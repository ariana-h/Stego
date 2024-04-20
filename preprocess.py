import os
from PIL import Image
import shutil


def convert_to_png(source_folder, output_size=(128, 128)):
    for filename in os.listdir(source_folder):
        if filename.lower().endswith(('.jpg', '.jpeg')):
            img_path = os.path.join(source_folder, filename)
            img = Image.open(img_path)
            png_path = os.path.splitext(img_path)[0] + '.png'
            img.save(png_path, 'PNG')
            os.remove(img_path)
            resized_img = img.resize(output_size)
            resized_img.save(png_path)
            

def copy_dataset(source_folder, dest_folder):
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    files = [f for f in os.listdir(source_folder) if os.path.isfile(os.path.join(source_folder, f))]
       
    for image in files:
        shutil.copy(os.path.join(source_folder, image), os.path.join(dest_folder, image))


def remove_images(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        
        if os.path.isfile(file_path):
            os.remove(file_path)
            
            
def main():
    if os.path.exists("data"):
        os.rename("data", "dataset")
    
    print("Preprocessing data...")
    source_folder = "dataset"
    normal_destination = "dataset/non-stego"
    stego_destination = "dataset/stego"
    
    convert_to_png(source_folder)
    copy_dataset(source_folder, normal_destination)
    copy_dataset(source_folder, stego_destination)
    
    remove_images(source_folder)

    print("Completed")


if __name__ == "__main__":
    main()