import os
import shutil
import random


DATASET_BASE = os.path.join("fruits-360_100x100", "fruits-360")
OUTPUT_DIR = os.path.join("..", "fruit_dataset_split")  # Placed outside or alongside fruit_recognition


TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
SEED = 42

def split_data():
    random.seed(SEED)
    

    train_src = os.path.join(DATASET_BASE, "Training")
    test_src = os.path.join(DATASET_BASE, "Test")

    if not os.path.exists(train_src):
        print(f"Error: Path '{train_src}' does not exist. Make sure you are running this script inside the 'fruit_recognition' folder.")
        return


    classes = sorted(list(set(os.listdir(train_src) + (os.listdir(test_src) if os.path.exists(test_src) else []))))
    print(f"Found {len(classes)} fruit classes.")

    total_processed = 0

    for idx, class_name in enumerate(classes, 1):
        all_images = []


        class_train_path = os.path.join(train_src, class_name)
        if os.path.exists(class_train_path):
            for img in os.listdir(class_train_path):
                if img.lower().endswith(('.jpg', '.jpeg', '.png')):
                    all_images.append(os.path.join(class_train_path, img))


        class_test_path = os.path.join(test_src, class_name)
        if os.path.exists(class_test_path):
            for img in os.listdir(class_test_path):
                if img.lower().endswith(('.jpg', '.jpeg', '.png')):
                    all_images.append(os.path.join(class_test_path, img))

        if not all_images:
            continue


        random.shuffle(all_images)

        total_imgs = len(all_images)
        train_count = int(total_imgs * TRAIN_RATIO)
        val_count = int(total_imgs * VAL_RATIO)

        splits = {
            'train': all_images[:train_count],
            'val': all_images[train_count:train_count + val_count],
            'test': all_images[train_count + val_count:]
        }


        for split_name, img_paths in splits.items():
            dest_dir = os.path.join(OUTPUT_DIR, split_name, class_name)
            os.makedirs(dest_dir, exist_ok=True)
            for img_path in img_paths:
                shutil.copy(img_path, os.path.join(dest_dir, os.path.basename(img_path)))

        total_processed += total_imgs
        if idx % 20 == 0 or idx == len(classes):
            print(f"Processed {idx}/{len(classes)} classes...")

    print(f"\nSuccessfully split {total_processed} images into '{os.path.abspath(OUTPUT_DIR)}'!")

if __name__ == '__main__':
    split_data()
