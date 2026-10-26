import os
import argparse
from PIL import Image
import imagehash
from tqdm import tqdm

def get_image_files(directory):
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    image_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(image_extensions):
                image_files.append(os.path.join(root, file))
    return image_files

def compute_hashes(image_paths, hash_size=16):
    hashes = {}
    print(f"Computing hashes for {len(image_paths)} images...")
    for path in tqdm(image_paths, desc="Hashing"):
        try:
            with Image.open(path) as img:
                h = imagehash.phash(img, hash_size=hash_size)
                hashes[path] = h
        except Exception as e:
            print(f"Error reading {path}: {e}")
    return hashes

def find_duplicates(train_hashes, target_hashes, threshold=5):
    duplicates = []
    for target_path, target_hash in tqdm(target_hashes.items(), desc="Checking duplicates"):
        min_dist = float('inf')
        best_match = None
        for train_path, train_hash in train_hashes.items():
            dist = target_hash - train_hash  # Hamming distance
            if dist < min_dist:
                min_dist = dist
                best_match = train_path
        
        if min_dist < threshold:
            duplicates.append({
                'target_path': target_path,
                'train_path': best_match,
                'distance': min_dist
            })
    return duplicates

def main():
    parser = argparse.ArgumentParser(description="GD-CoRPA Data Leakage / Deduplication Check")
    parser.add_argument("--train-dir", type=str, required=True, help="Path to training set directory")
    parser.add_argument("--val-dir", type=str, required=True, help="Path to validation set directory")
    parser.add_argument("--test-dir", type=str, required=True, help="Path to test set directory")
    parser.add_argument("--hash-size", type=int, default=16, help="Perceptual hash size")
    parser.add_argument("--hamming-threshold", type=int, default=5, help="Hamming distance threshold for duplicates")
    
    args = parser.parse_args()
    
    # 1. Collect image files
    train_files = get_image_files(args.train_dir)
    val_files = get_image_files(args.val_dir)
    test_files = get_image_files(args.test_dir)
    
    print(f"Found {len(train_files)} training, {len(val_files)} validation, and {len(test_files)} test images.")
    
    # 2. Compute hashes
    train_hashes = compute_hashes(train_files, hash_size=args.hash_size)
    val_hashes = compute_hashes(val_files, hash_size=args.hash_size)
    test_hashes = compute_hashes(test_files, hash_size=args.hash_size)
    
    # 3. Find duplicates in val
    print("\nChecking validation set for leakage...")
    val_duplicates = find_duplicates(train_hashes, val_hashes, threshold=args.hamming_threshold)
    
    # 4. Find duplicates in test
    print("\nChecking test set for leakage...")
    test_duplicates = find_duplicates(train_hashes, test_hashes, threshold=args.hamming_threshold)
    
    # 5. Report results
    print("\n" + "="*50)
    print("DEDUPLICATION & DATA LEAKAGE REPORT")
    print("="*50)
    
    print(f"Validation set duplicates found: {len(val_duplicates)} / {len(val_files)} ({len(val_duplicates)/len(val_files)*100:.2f}%)")
    if val_duplicates:
        print("Sample validation duplicates:")
        for dup in val_duplicates[:5]:
            print(f"  Val:   {os.path.basename(dup['target_path'])}")
            print(f"  Train: {os.path.basename(dup['train_path'])} (distance: {dup['distance']})")
            
    print(f"\nTest set duplicates found: {len(test_duplicates)} / {len(test_files)} ({len(test_duplicates)/len(test_files)*100:.2f}%)")
    if test_duplicates:
        print("Sample test duplicates:")
        for dup in test_duplicates[:5]:
            print(f"  Test:  {os.path.basename(dup['target_path'])}")
            print(f"  Train: {os.path.basename(dup['train_path'])} (distance: {dup['distance']})")
            
    total_leaked = len(val_duplicates) + len(test_duplicates)
    total_target = len(val_files) + len(test_files)
    print("\n" + "="*50)
    print(f"Total leakage: {total_leaked} / {total_target} ({total_leaked/total_target*100:.2f}%)")
    print("="*50)
    
    if total_leaked > 0:
        print("\n[WARNING] Significant data leakage detected. Pre-balanced split contains near-duplicate augmented images.")
    else:
        print("\n[INFO] No data leakage detected. The splits are clean.")

if __name__ == "__main__":
    main()
