import os
from pathlib import Path
from cloud_cluster_analyzer import CloudClusterAnalyzer
from PIL import Image
import numpy as np

# Paths
input_dir = "./dataset/images"  # adjust to where your TIFs are
output_img_dir = "./dataset/images"
output_ann_dir = "./dataset/annotations"

# Create output folders if not there
os.makedirs(output_img_dir, exist_ok=True)
os.makedirs(output_ann_dir, exist_ok=True)

# List of all .tif files
tif_files = [f for f in os.listdir(input_dir) if f.endswith(".tif")]

# Initialize analyzer
analyzer = CloudClusterAnalyzer(estimated_km_per_pixel=4.0)

for tif_file in tif_files:
    filepath = os.path.join(input_dir, tif_file)
    print(f"[INFO] Processing {filepath}")
    
    # run the analyzer to get the final TCC mask
    final_mask = analyzer.analyze_clusters(filepath)
    
    # store original scaled brightness temperature image
    data = analyzer.data
    tb_img = np.clip((data - 180) * 4, 0, 255).astype(np.uint8)
    
    # Save brightness temperature as a PNG
    out_img_path = os.path.join(output_img_dir, tif_file.replace(".tif", ".png"))
    Image.fromarray(tb_img).save(out_img_path)
    
    # Save mask as PNG (convert to 0/255)
    mask_uint8 = (final_mask.astype(np.uint8) * 255)
    out_ann_path = os.path.join(output_ann_dir, tif_file.replace(".tif", ".png"))
    Image.fromarray(mask_uint8).save(out_ann_path)
    
    print(f"[INFO] Saved image to {out_img_path}")
    print(f"[INFO] Saved annotation to {out_ann_path}")

print("[DONE] DeepLabV3+ style dataset prepared.")
