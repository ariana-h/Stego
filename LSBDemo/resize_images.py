from PIL import Image
import os

def resize_image(input_folder, output_folder, size=(256, 256)):
    for file in os.listdir(input_folder):
        if file.lower().endswith(('.png', '.jpg', '.jpeg')): 
            input_path = os.path.join(input_folder, file) 
            output_path = os.path.join(output_folder, file) 
                
            with Image.open(input_path) as img:
                resized_img = img.resize(size)
                resized_img.save(output_path)


def main():
    input_folder = 'LSBDemo/images' 
    
    if not os.path.exists('LSBDemo/resized'):
        os.makedirs('LSBDemo/resized') 
        
    output_folder = 'LSBDemo/resized'  
    
    print("Resizing images...")
    resize_image(input_folder, output_folder)
    
    print("Completed")
    
    
if __name__ == "__main__":
    main()