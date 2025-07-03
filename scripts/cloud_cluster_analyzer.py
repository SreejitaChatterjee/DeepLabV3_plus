# scripts/cloud_cluster_analyzer.py
import numpy as np
import rasterio
from skimage.measure import label, regionprops
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import os
from pathlib import Path

class CloudClusterAnalyzer:
    def __init__(self, estimated_km_per_pixel=4.0):
        self.estimated_km_per_pixel = estimated_km_per_pixel
        self.data = None
        self.metadata = {}
        self.tb_threshold = 220
        self.min_radius_km = 111
        self.min_area_km2 = 34800
        self.tcc_labels = None
        self.tcc_properties = None
        
    def load_tiff(self, filepath):
        with rasterio.open(filepath) as src:
            self.data = src.read(1)
            self.metadata = {
                'transform': src.transform,
                'crs': src.crs,
                'width': src.width,
                'height': src.height
            }
        return self.data
    
    def apply_tb_mask(self, data=None, threshold=None):
        if data is None:
            data = self.data
        if threshold is None:
            threshold = self.tb_threshold
        return data < threshold
    
    def apply_size_filter(self, mask):
        min_area_pixels = np.pi * (self.min_radius_km / self.estimated_km_per_pixel) ** 2
        labeled = label(mask)
        valid_mask = np.zeros_like(mask, dtype=bool)
        for region in regionprops(labeled):
            if region.area >= min_area_pixels:
                valid_mask[labeled == region.label] = True
        return valid_mask
    
    def apply_area_filter(self, mask):
        min_area_pixels = self.min_area_km2 / (self.estimated_km_per_pixel ** 2)
        labeled = label(mask)
        valid_mask = np.zeros_like(mask, dtype=bool)
        for region in regionprops(labeled):
            if region.area >= min_area_pixels:
                valid_mask[labeled == region.label] = True
        return valid_mask
    
    def analyze_clusters(self, filepath):
        self.load_tiff(filepath)
        tb_mask = self.apply_tb_mask()
        radius_mask = self.apply_size_filter(tb_mask)
        final_mask = self.apply_area_filter(radius_mask)
        return final_mask


    
    def save_mask(self, mask, out_path):
        with rasterio.open(
            out_path,
            'w',
            driver='GTiff',
            height=mask.shape[0],
            width=mask.shape[1],
            count=1,
            dtype='uint8',
            crs=self.metadata['crs'],
            transform=self.metadata['transform']
        ) as dst:
            dst.write(mask.astype(np.uint8), 1)
        print(f"Saved: {out_path}")
