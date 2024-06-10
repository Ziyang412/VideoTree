from PIL import Image
import requests
import torch
import transformers
from pathlib import Path
import json
import pickle
from tqdm import tqdm
import os
from pprint import pprint
import pdb

import numpy as np
from kmeans_pytorch import kmeans
from sklearn.cluster import KMeans
from transformers import AutoModel, AutoConfig
from transformers import CLIPImageProcessor, pipeline, CLIPTokenizer
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode

### VLMs

# blip2
from transformers import Blip2Processor, Blip2ForConditionalGeneration



### LLMs

# llama2
from transformers import AutoTokenizer
# flanT5
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def load_image_features(name_ids, save_folder):
    """
    Load image features from a .pt file.

    Args:
    - filename (str): Name of the .pt file to load

    Returns:
    - img_feats (torch.Tensor): Loaded image features
    """
    filename = f"{name_ids}.pt"  # Construct filename with name_ids
    filepath = os.path.join(save_folder, filename)
    img_feats = torch.load(filepath)
    return img_feats

def find_closest_points_per_cluster(x, cluster_ids, cluster_centers):
    # Dictionary to store the indices of the closest points for each cluster
    closest_points_idx_per_cluster = {cluster_id: [] for cluster_id in range(len(cluster_centers))}
    
    # Iterate over each cluster
    for cluster_id in range(len(cluster_centers)):
        # Filter points belonging to the current cluster
        indices_in_cluster = torch.where(cluster_ids == cluster_id)[0]
        points_in_cluster = x[indices_in_cluster]
        
        # Calculate distances from points in the cluster to the cluster center
        distances = torch.norm(points_in_cluster - cluster_centers[cluster_id], dim=1)

        if distances.numel() > 0:    
            
            # Find the index (within the cluster) of the point closest to the cluster center
            closest_idx_in_cluster = torch.argmin(distances).item()
            
            # Map back to the original index in x
            closest_global_idx = indices_in_cluster[closest_idx_in_cluster].item()
            
            # Store the global index
            closest_points_idx_per_cluster[cluster_id].append(closest_global_idx)

    return closest_points_idx_per_cluster


def load_pkl(fn):
    with open(fn, 'rb') as f:
        data = pickle.load(f)
    return data

def save_pkl(data, fn):
    with open(fn, 'wb') as f:
        pickle.dump(data, f)

def load_json(fn):
    with open(fn, 'r') as f:
        data = json.load(f)
    return data

def save_json(data, fn, indent=4):
    with open(fn, 'w') as f:
        json.dump(data, f, indent=indent)


def clip_es():
    device = "cuda" if torch.cuda.is_available() else "cpu" 

    resume = True
    output_base_path = Path('./clip_es')
    output_base_path.mkdir(parents=True, exist_ok=True)
    base_path = Path('/Path/to/Egoschema/dataset/images')
    save_folder = Path('path/to/data/egoschema_features')

    all_data = []

    with open('/data/path/EgoSchema/subset_answers.json', 'r') as file:
        json_data = json.load(file)    
    subset_names_list = list(json_data.keys())

    example_path_list = list(base_path.iterdir())

    pbar = tqdm(total=len(example_path_list))

    i = 0 
    max = 50

    for example_path in example_path_list:

        # for subset videos
        if example_path.name not in subset_names_list:
            continue

        name_ids = example_path.name
        img_feats = load_image_features(name_ids, save_folder)
        
        cluster_ids_x, cluster_centers = kmeans(X=img_feats, num_clusters=32, distance='cosine', device=torch.device('cuda:0'))

        # send cluster_ids_x to GPU 
        cluster_ids_x = cluster_ids_x.to('cuda')
        cluster_centers = cluster_centers.to('cuda')
        closest_points_idx_per_cluster = find_closest_points_per_cluster(img_feats, cluster_ids_x, cluster_centers)


        if closest_points_idx_per_cluster is None:
            print("closest_points_idx_per_cluster is None")
            continue
        sorted_values = sorted([value for sublist in closest_points_idx_per_cluster.values() for value in sublist])
        cluster_ids_x = cluster_ids_x.tolist()
        all_data.append({"name": example_path.name, "sorted_values": sorted_values, "cluster_ids_x": cluster_ids_x})
       
        pbar.update(1)
    save_json(all_data, 'first/level/node/output.json')

    pbar.close()



if __name__ == '__main__':
    clip_es()
