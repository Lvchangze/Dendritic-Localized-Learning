import os
from shutil import copyfile, rmtree

def process_tinyimagenet_val_set(val_dir):
    """
    Process the TinyImageNet validation set by moving images to their corresponding class subdirectories.
    """
    # Paths for the validation images and annotations file
    images_dir = os.path.join(val_dir, 'images')
    annotations_file = os.path.join(val_dir, 'val_annotations.txt')

    if not os.path.exists(images_dir) or not os.path.exists(annotations_file):
        raise FileNotFoundError("Validation set path or annotations file does not exist. Please check the dataset integrity!")
    
    with open(annotations_file, 'r') as f:
        for line in f.readlines():
            img_name, class_name = line.strip().split()[:2]
            
            class_dir = os.path.join(val_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)
            
            # Move image from images/ to its respective class directory
            src_path = os.path.join(images_dir, img_name)
            dst_path = os.path.join(class_dir, img_name)
            if os.path.exists(src_path):
                copyfile(src_path, dst_path)
            else:
                print(f"Image {src_path} does not exist, skipping.")

    # Remove the images directory and its contents after processing
    rmtree(images_dir)

val_dir = './tiny-imagenet-200/val'
process_tinyimagenet_val_set(val_dir)