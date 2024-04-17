import os
import random
import shutil

def split_dataset(input_stego, input_non_stego, output_folder, split_ratio):
    train_stego = os.path.join(output_folder, 'train', 'stego')
    test_stego = os.path.join(output_folder, 'test', 'stego')  
    train_non_stego = os.path.join(output_folder, 'train', 'non-stego')
    test_non_stego = os.path.join(output_folder, 'test', 'non-stego') 
    
    for folder in [train_stego, test_stego, train_non_stego, test_non_stego]:
        if not os.path.exists(folder):
            os.makedirs(folder)

    # Check if input stego folder exists and is not empty
    if os.path.exists(input_stego) and os.listdir(input_stego):
        stego_images = [f for f in os.listdir(input_stego) if os.path.isfile(os.path.join(input_stego, f))]
        random.shuffle(stego_images)

        split_index = int(len(stego_images) * split_ratio)

        train_images_stego = stego_images[:split_index]
        test_images_stego = stego_images[split_index:]

        for image in train_images_stego:
            shutil.move(os.path.join(input_stego, image), os.path.join(train_stego, image))
        for image in test_images_stego:
            shutil.move(os.path.join(input_stego, image), os.path.join(test_stego, image))

        # print("Stego dataset split completed.")
    else:
        print(f"Error: Input folder '{input_stego}' does not exist or is empty.")

    # Check if input non-stego folder exists and is not empty
    if os.path.exists(input_non_stego) and os.listdir(input_non_stego):
        non_stego_images = [f for f in os.listdir(input_non_stego) if os.path.isfile(os.path.join(input_non_stego, f))]
        random.shuffle(non_stego_images)

        split_index_non_stego = int(len(non_stego_images) * split_ratio)

        train_images_non_stego = non_stego_images[:split_index_non_stego]
        test_images_non_stego = non_stego_images[split_index_non_stego:]

        for image in train_images_non_stego:
            shutil.move(os.path.join(input_non_stego, image), os.path.join(train_non_stego, image))
        for image in test_images_non_stego:
            shutil.move(os.path.join(input_non_stego, image), os.path.join(test_non_stego, image))

        # print("Non-Stego dataset split completed.")
    else:
        print(f"Error: Input folder '{input_non_stego}' does not exist or is empty.")
    
    print("Dataset split completed.")
    

def main():
    input_stego = 'dataset/stego'
    input_non_stego = 'dataset/non-stego'
    output_folder = 'dataset/'
    split_ratio = 0.8  # 80% training, 20% testing
    
    print("Rearranging dataset...")

    split_dataset(input_stego, input_non_stego, output_folder, split_ratio)
    
    # Remove empty input folders
    if os.path.exists(input_stego) and not os.listdir(input_stego):
        os.rmdir(input_stego)
    if os.path.exists(input_non_stego) and not os.listdir(input_non_stego):
        os.rmdir(input_non_stego)
    
if __name__ == "__main__":
    main()
