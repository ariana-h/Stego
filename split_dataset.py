import os
import random
import shutil

def split_dataset(input_folder, output_folder, split_ratio=0.8):
    train_stego = os.path.join(output_folder, 'train', 'stego')
    test_stego = os.path.join(output_folder, 'test', 'stego')  
    train_non_stego = os.path.join(output_folder, 'train', 'non-stego')
    test_non_stego = os.path.join(output_folder, 'test', 'non-stego') 
    
    for folder in [train_stego, test_stego, train_non_stego, test_non_stego]:
        if not os.path.exists(folder):
            os.makedirs(folder)


    stego_images = [f for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f))]
    random.shuffle(stego_images)

    split_index = int(len(stego_images) * split_ratio)

    train_images_stego = stego_images[:split_index]
    test_images_stego = stego_images[split_index:]


    for image in train_images_stego:
        shutil.move(os.path.join(input_folder, image), os.path.join(train_stego, image))
    for image in test_images_stego:
        shutil.move(os.path.join(input_folder, image), os.path.join(test_stego, image))


    input_non_stego = input_folder.replace("stego", "non-stego")
    non_stego_images = [f for f in os.listdir(input_non_stego) if os.path.isfile(os.path.join(input_non_stego, f))]
    random.shuffle(non_stego_images)

    split_index_non_stego = int(len(non_stego_images) * split_ratio)

    train_images_non_stego = non_stego_images[:split_index_non_stego]
    test_images_non_stego = non_stego_images[split_index_non_stego:]


    for image in train_images_non_stego:
        shutil.move(os.path.join(input_non_stego, image), os.path.join(train_non_stego, image))
    for image in test_images_non_stego:
        shutil.move(os.path.join(input_non_stego, image), os.path.join(test_non_stego, image))

    print("Dataset split completed.")
    
    # Remove the input folders if they are empty
    if not os.listdir(input_folder):
        os.rmdir(input_folder)
    if not os.listdir(input_non_stego):
        os.rmdir(input_non_stego)
    

def main():
    input_stego = 'dataset/stego'
    input_non_stego = 'dataset/non-stego'
    output_folder = 'dataset/'
    split_ratio = 0.8  # 80% training, 20% testing

    split_dataset(input_stego, output_folder, split_ratio)
    split_dataset(input_non_stego, output_folder, split_ratio)
    
if __name__ == "__main__":
    main()
