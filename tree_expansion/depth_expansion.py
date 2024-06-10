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

from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist

### LLMs

# llama2
from transformers import AutoTokenizer
# flanT5
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster

def hierarchical_clustering(video_features, relevance_scores, num_clusters=5, num_subclusters=5, num_subsubclusters=5):

    if len(relevance_scores) > num_clusters:
        relevance_scores = relevance_scores[:num_clusters]
    elif len(relevance_scores) < num_clusters:
        relevance_scores.extend([3] * (num_clusters - len(relevance_scores)))  # Append '3' to fill the list


    # Level 1 Clustering
    linked = linkage(video_features, method='ward')
    primary_cluster_labels = fcluster(linked, num_clusters, criterion='maxclust')

    # Data structure to hold clusters at each level
    clusters = {i: {} for i in range(1, num_clusters + 1)}

    # Processing each primary cluster based on relevance score
    for cluster_id, score in enumerate(relevance_scores, 1):
        primary_indices = np.where(primary_cluster_labels == cluster_id)[0]
        if len(primary_indices) < 2 or score == 1:
            clusters[cluster_id] = primary_indices  # Only store primary indices for score 1
            continue
        
        sub_features = video_features[primary_indices]
        linked_sub = linkage(sub_features, method='ward')
        sub_cluster_labels = fcluster(linked_sub, num_subclusters, criterion='maxclust')

        if score == 2:
            clusters[cluster_id] = {i: primary_indices[np.where(sub_cluster_labels == i)[0]] for i in range(1, num_subclusters + 1)}
            continue
        
        # Level 3 Clustering for score 3
        for subcluster_id in range(1, num_subclusters + 1):
            sub_indices = np.where(sub_cluster_labels == subcluster_id)[0]
            if len(sub_indices) < 2:
                continue

            subsub_features = sub_features[sub_indices]
            linked_subsub = linkage(subsub_features, method='ward')
            subsub_cluster_labels = fcluster(linked_subsub, num_subsubclusters, criterion='maxclust')

            clusters[cluster_id][subcluster_id] = {}
            for subsubcluster_id in range(1, num_subsubclusters + 1):
                final_indices = np.where(subsub_cluster_labels == subsubcluster_id)[0]
                original_indices = primary_indices[sub_indices[final_indices]]
                clusters[cluster_id][subcluster_id][subsubcluster_id] = original_indices

    return clusters

import torch
import torch.nn.functional as F
import numpy as np

def cosine_similarity(points, centroid):
    """
    Calculate cosine similarity between points and centroid.
    Returns the cosine distances (1 - similarity).
    """
    points_normalized = F.normalize(points, dim=1)
    centroid_normalized = F.normalize(centroid.unsqueeze(0), dim=1)
    return 1 - torch.mm(points_normalized, centroid_normalized.T).squeeze()

def find_closest_points_in_temporal_order_subsub(x, clusters, relevance_scores):
    closest_points_indices = []

    for cluster_id, cluster_data in clusters.items():
        relevance = relevance_scores[cluster_id - 1]

        if isinstance(cluster_data, np.ndarray):  # Primary cluster directly
            if cluster_data.size == 0:
                continue  # Skip empty clusters
            points_in_cluster = x[torch.tensor(cluster_data, dtype=torch.long)]
            cluster_centroid = points_in_cluster.mean(dim=0)
            distances = cosine_similarity(points_in_cluster, cluster_centroid)
            if distances.numel() > 0:
                closest_idx = torch.argmin(distances).item()
                closest_points_indices.append(int(cluster_data[closest_idx]))

        elif isinstance(cluster_data, dict):  # Handle subclusters and sub-subclusters
            if relevance == 1:
                # Only take the representative frame for the primary cluster
                primary_indices = []
                for subcluster_data in cluster_data.values():
                    if isinstance(subcluster_data, dict):
                        for sub_data in subcluster_data.values():
                            if sub_data.size > 0:
                                primary_indices.append(sub_data)
                    elif isinstance(subcluster_data, np.ndarray) and subcluster_data.size > 0:
                        primary_indices.append(subcluster_data)

                if primary_indices:
                    primary_indices = np.concatenate(primary_indices)
                    primary_points = x[torch.tensor(primary_indices, dtype=torch.long)]
                    primary_centroid = primary_points.mean(dim=0)
                    primary_distances = cosine_similarity(primary_points, primary_centroid)
                    if primary_distances.numel() > 0:
                        closest_primary_idx = torch.argmin(primary_distances).item()
                        closest_points_indices.append(int(primary_indices[closest_primary_idx]))
                continue

            elif relevance == 2 or relevance == 3:
                # Include primary cluster representative
                primary_indices = []
                for subcluster_data in cluster_data.values():
                    if isinstance(subcluster_data, dict):
                        for sub_data in subcluster_data.values():
                            if sub_data.size > 0:
                                primary_indices.append(sub_data)
                    elif isinstance(subcluster_data, np.ndarray) and subcluster_data.size > 0:
                        primary_indices.append(subcluster_data)

                if primary_indices:
                    primary_indices = np.concatenate(primary_indices)
                    primary_points = x[torch.tensor(primary_indices, dtype=torch.long)]
                    primary_centroid = primary_points.mean(dim=0)
                    primary_distances = cosine_similarity(primary_points, primary_centroid)
                    if primary_distances.numel() > 0:
                        closest_primary_idx = torch.argmin(primary_distances).item()
                        closest_points_indices.append(int(primary_indices[closest_primary_idx]))

                for subcluster_id, subclusters in cluster_data.items():
                    if isinstance(subclusters, dict):  # Sub-subclusters
                        for subsubcluster_id, indices in subclusters.items():
                            if indices.size == 0:
                                continue  # Skip empty sub-subclusters
                            indices_tensor = torch.tensor(indices, dtype=torch.long)
                            points_in_subsubcluster = x[indices_tensor]
                            subsubcluster_centroid = points_in_subsubcluster.mean(dim=0)
                            distances = cosine_similarity(points_in_subsubcluster, subsubcluster_centroid)
                            if distances.numel() > 0:
                                closest_idx_in_subsubcluster = torch.argmin(distances).item()
                                closest_global_idx = indices[closest_idx_in_subsubcluster]
                                closest_points_indices.append(int(closest_global_idx))

                    elif isinstance(subclusters, np.ndarray):
                        if subclusters.size == 0:
                            continue  # Skip empty subclusters
                        points_in_subcluster = x[torch.tensor(subclusters, dtype=torch.long)]
                        subcluster_centroid = points_in_subcluster.mean(dim=0)
                        distances = cosine_similarity(points_in_subcluster, subcluster_centroid)
                        if distances.numel() > 0:
                            closest_idx = torch.argmin(distances).item()
                            closest_points_indices.append(int(subclusters[closest_idx]))

    closest_points_indices.sort()  # Ensure the points are in temporal order
    return closest_points_indices




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


# image_path = "CLIP.png"
model_name_or_path = "BAAI/EVA-CLIP-8B" # or /path/to/local/EVA-CLIP-18B

image_size = 448



def clip_es():
    device = "cuda" if torch.cuda.is_available() else "cpu" 

    output_base_path = Path('./clip_es')
    output_base_path.mkdir(parents=True, exist_ok=True)
    base_path = Path('/Path/to/Egoschema/dataset/images')
    save_folder = Path('path/to/data/egoschema_features')
    # res = {}  # uid --> [narr1, narr2, ...]

    rel_path = '/relevance/output/json/file'
    with open(rel_path, 'r') as file:
        cap_score_data = json.load(file)

    all_data = []

    with open('/data/path/subset_answers.json', 'r') as file:
        json_data = json.load(file)    
    subset_names_list = list(json_data.keys())
    # print("subset_names_list",subset_names_list)

    example_path_list = list(base_path.iterdir())

    pbar = tqdm(total=len(example_path_list))

    i = 0 
    max = 50

    for example_path in example_path_list:


        if example_path.name not in subset_names_list:
            continue

        name_ids = example_path.name
        img_feats = load_image_features(name_ids, save_folder)
        relevance_scores = cap_score_data['data'][name_ids]['relevance']
        print("relevance_scores",relevance_scores)



        img_feats = img_feats.cpu()
        clusters_info = hierarchical_clustering(img_feats,relevance_scores, num_clusters=32 ,num_subclusters=5, num_subsubclusters=5)
        closest_points_temporal_subsub = find_closest_points_in_temporal_order_subsub(img_feats, clusters_info,relevance_scores)
        print("closest_points_temporal_subsub",closest_points_temporal_subsub)



        all_data.append({"name": example_path.name, "sorted_values": closest_points_temporal_subsub, "relevance": relevance_scores})

        pbar.update(1)

    save_json(all_data, '/path/to/depth/expension/output.json')

    pbar.close()


if __name__ == '__main__':
    clip_es()
