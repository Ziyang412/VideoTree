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



# image_path = "CLIP.png"
model_name_or_path = "BAAI/EVA-CLIP-8B" # or /path/to/local/EVA-CLIP-18B

image_size = 448
processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")


model = AutoModel.from_pretrained(
    model_name_or_path, 
    torch_dtype=torch.float16,
    trust_remote_code=True).to('cuda').eval()



def clip_es():
    device = "cuda" if torch.cuda.is_available() else "cpu" 
    processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")

    model = AutoModel.from_pretrained(
    model_name_or_path, 
    torch_dtype=torch.float16,
    trust_remote_code=True).to('cuda').eval()
    resume = True
    output_base_path = Path('./clip_es')
    output_base_path.mkdir(parents=True, exist_ok=True)
    base_path = Path('/Path/to/Egoschema/dataset/images')
    # res = {}  # uid --> [narr1, narr2, ...]

    all_data = []

    with open('/data/path/EgoSchema/subset_answers.json', 'r') as file:
        json_data = json.load(file)    
    subset_names_list = list(json_data.keys())

    example_path_list = list(base_path.iterdir())

    pbar = tqdm(total=len(example_path_list))

    i = 0 
    max = 50

    for example_path in example_path_list:

        if example_path.name not in subset_names_list:
            continue
        # else:
        #     print("example_path in subset")
        example_output_path = output_base_path / f'{example_path.name}.json'
        if resume and example_output_path.exists():
            pbar.update(1)
            continue
        narr_list = []
        image_paths = list(example_path.iterdir())
        image_paths.sort(key=lambda x: int(x.stem))
        img_feature_list = []
        for image_path in image_paths:
            image = Image.open(str(image_path))

            input_pixels = processor(images=image, return_tensors="pt", padding=True).pixel_values.to('cuda')

            with torch.no_grad(), torch.cuda.amp.autocast():
                image_features = model.encode_image(input_pixels)
                img_feature_list.append(image_features)
                # print("image_features shape",image_features.shape)
            # print("input pixel shape",input_pixels.shape)
        # convert img_feature_list to tensor
        img_feature_tensor = torch.stack(img_feature_list)
        img_feats = img_feature_tensor.squeeze(1)

        # cluster_ids_x, cluster_centers = kmeans(X=img_feats, num_clusters=32, device=torch.device('cuda:0'))
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
        # save_json(narr_list, example_output_path)
    save_json(all_data, 'first/level/node/output.json')

    pbar.close()



if __name__ == '__main__':
    # blip2_next()
    clip_es()
