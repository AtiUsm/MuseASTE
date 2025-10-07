# -*- coding: utf-8 -*-
"""
Created on Tue Oct  7 05:05:10 2025

@author: Atiya
"""

import pandas as pd
from icrawler.builtin import GoogleImageCrawler
import os
import random



df = pd.read_csv('objective_ground_truth_complete_perfectum.csv', encoding='latin-1')

# Folder to save images
output_dir = 'car_images'
os.makedirs(output_dir, exist_ok=True)

# Function to create search query
def create_query(row):
    # If multiple colors, pick one randomly
    colors = [c.strip() for c in row['Color'].split(',')]
    chosen_color = random.choice(colors)
    query = f"{row['Make']} {row['Model']} {chosen_color} {row['Body Type']}"
    return query

# Loop through each row and download one image
for index, row in df.iterrows():
    query = create_query(row)
    crawler = GoogleImageCrawler(storage={'root_dir': output_dir})
    crawler.crawl(keyword=query, max_num=1, file_idx_offset=row['id'])

    # Rename the downloaded image to match the id
    for f in os.listdir(output_dir):
        if f.startswith(str(row['id'])):
            os.rename(os.path.join(output_dir, f),os.path.join(output_dir, f"{str(row['id']).lstrip('0')}.jpg"))
            break

print("Download completed!")


