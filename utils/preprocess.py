import os
import shutil
import random
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# Define paths
DATASET_DIR = 'path_to_dataset'
OUTPUT_DIR = 'processed_dataset'
TRAIN_DIR = os.path.join(OUTPUT_DIR, 'train')
VAL_DIR = os.path.join(OUTPUT_DIR, 'val')
TEST_DIR = os.path.join(OUTPUT_DIR, 'test')

# Create output directories
for dir_path in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
    for label in ['drowsy', 'non_drowsy']:
        os.makedirs(os.path.join(dir_path, label), exist_ok=True)

# Identify subjects
subjects = os.listdir(DATASET_DIR)
random.shuffle(subjects)

# Split subjects
val_subjects = subjects[:3]
test_subjects = subjects[3:6]
train_subjects = subjects[6:]

# Function to copy images
def copy_images(subjects_list, target_dir):
    for subject in subjects_list:
        subject_path = os.path.join(DATASET_DIR, subject)
        label = 'drowsy' if subject.isupper() else 'non_drowsy'
        for img_name in os.listdir(subject_path):
            src = os.path.join(subject_path, img_name)
            dst = os.path.join(target_dir, label, f"{subject}_{img_name}")
            shutil.copy(src, dst)

# Copy images to respective directories
copy_images(train_subjects, TRAIN_DIR)
copy_images(val_subjects, VAL_DIR)
copy_images(test_subjects, TEST_DIR)

# Compute mean and std
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def compute_mean_std(image_dir):
    mean = 0.
    std = 0.
    nb_samples = 0.
    for label in ['drowsy', 'non_drowsy']:
        path = os.path.join(image_dir, label)
        for img_name in tqdm(os.listdir(path)):
            img_path = os.path.join(path, img_name)
            img = Image.open(img_path).convert('RGB')
            img = transform(img)
            nb_samples += 1
            mean += img.mean([1, 2])
            std += img.std([1, 2])
    mean /= nb_samples
    std /= nb_samples
    return mean, std

mean, std = compute_mean_std(TRAIN_DIR)
print(f"Mean: {mean}")
print(f"Std: {std}")
