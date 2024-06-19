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
import torch.nn.functional as F

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

from scipy.cluster.hierarchy import linkage, fcluster

def hierarchical_clustering_with_external_primary(video_features, cluster_ids, relevance_scores, num_subclusters=5, num_subsubclusters=5):
    clusters = {i: {} for i in range(0, max(cluster_ids)+1)}

    for cluster_id in set(cluster_ids):
        primary_indices = [i for i, x in enumerate(cluster_ids) if x == cluster_id]

        if cluster_id < len(relevance_scores):
            score = relevance_scores[cluster_id]
        else:
            score = 3

        if len(primary_indices) < 2:
            clusters[cluster_id] = primary_indices
            continue

        sub_features = video_features[primary_indices]

        if score == 1:
            clusters[cluster_id] = primary_indices
            continue

        linked_sub = linkage(sub_features, method='ward')
        sub_cluster_labels = fcluster(linked_sub, num_subclusters, criterion='maxclust')
        sub_cluster_labels = sub_cluster_labels - 1

        if score == 2:
            clusters[cluster_id] = {i: [primary_indices[j] for j in np.where(sub_cluster_labels == i)[0]] for i in range(0, num_subclusters)}
            continue

        for subcluster_id in range(0, num_subclusters):
            sub_indices = np.where(sub_cluster_labels == subcluster_id)[0]
            if len(sub_indices) < 2:
                continue

            subsub_features = sub_features[sub_indices]
            linked_subsub = linkage(subsub_features, method='ward')
            subsub_cluster_labels = fcluster(linked_subsub, num_subsubclusters, criterion='maxclust')
            subsub_cluster_labels = subsub_cluster_labels - 1

            clusters[cluster_id][subcluster_id] = {}
            for subsubcluster_id in range(0, num_subsubclusters):
                final_indices = sub_indices[np.where(subsub_cluster_labels == subsubcluster_id)[0]]  # Correctly index into sub_indices
                original_indices = [primary_indices[i] for i in final_indices]
                clusters[cluster_id][subcluster_id][subsubcluster_id] = original_indices

    return clusters


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
        # print("cluster_id",cluster_id)
        # print("cluster_data type",type(cluster_data))

        if cluster_id < len(relevance_scores):
            relevance = relevance_scores[cluster_id]
        else:
            relevance = 3

        if isinstance(cluster_data, list):  # Primary cluster directly
            cluster_data = np.array(cluster_data)
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
                print("line 207")
                primary_indices = []
                for subcluster_data in cluster_data.values():
                    if isinstance(subcluster_data, dict):
                        for sub_data in subcluster_data.values():
                            if sub_data.size > 0:
                                primary_indices.append(sub_data)
                    elif isinstance(subcluster_data, list) and len(subcluster_data) > 0:
                        subcluster_data = np.array(subcluster_data)
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
                            if len(sub_data) > 0:
                                primary_indices.append(sub_data)
                    elif isinstance(subcluster_data, list) and len(subcluster_data) > 0:
                        subcluster_data = np.array(subcluster_data)
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
                            if len(indices) == 0:
                                continue  # Skip empty sub-subclusters
                            indices_tensor = torch.tensor(indices, dtype=torch.long)
                            points_in_subsubcluster = x[indices_tensor]
                            subsubcluster_centroid = points_in_subsubcluster.mean(dim=0)
                            distances = cosine_similarity(points_in_subsubcluster, subsubcluster_centroid)
                            if distances.numel() > 0:
                                closest_idx_in_subsubcluster = torch.argmin(distances).item()
                                closest_global_idx = indices[closest_idx_in_subsubcluster]
                                closest_points_indices.append(int(closest_global_idx))

                    elif isinstance(subclusters, list):
                        subclusters = np.array(subclusters)
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

def load_json(fn):
    with open(fn, 'r') as f:
        data = json.load(f)
    return data

def save_json(data, fn, indent=4):
    with open(fn, 'w') as f:
        json.dump(data, f, indent=indent)


def depth_expansion():
    device = "cuda" if torch.cuda.is_available() else "cpu" 

    output_base_path = Path('./clip_es')
    output_base_path.mkdir(parents=True, exist_ok=True)
    base_path = Path('path/to/data/egoschema_frames')
    save_folder = '/path/to/egoschema/frame_features'

    rel_path = '/path/to/output/of/dynamic_width_expansion/relevance_score.json'
    with open(rel_path, 'r') as file:
        cap_score_data = json.load(file)

    width_res_path = '/path/to/output/of/dynamic_width_expansion/width_res.json'
    with open(width_res_path, 'r') as file:
        width_res_data = json.load(file)
    width_cluster_id_dict = {item['name']: item['cluster_ids_x'] for item in width_res_data}


    all_data = []

    with open('path/data/egoschema/subset_answers.json', 'r') as file:
        json_data = json.load(file)    
    subset_names_list = list(json_data.keys())
    # print("subset_names_list",subset_names_list)

    example_path_list = list(base_path.iterdir())

    pbar = tqdm(total=len(example_path_list))

    i = 0 
    max = 1

    for example_path in example_path_list:

        # comment out when testing full set
        if example_path.name not in subset_names_list:
            continue

        name_ids = example_path.name
        img_feats = load_image_features(name_ids, save_folder)
        relevance_scores = cap_score_data['data'][name_ids]['pred']
        primary_cluster_ids = width_cluster_id_dict.get(name_ids, None)

        img_feats = img_feats.cpu()
        
        clusters_info = hierarchical_clustering_with_external_primary(img_feats,primary_cluster_ids, relevance_scores ,num_subclusters=4, num_subsubclusters=4)

        closest_points_temporal_subsub = find_closest_points_in_temporal_order_subsub(img_feats, clusters_info,relevance_scores)
        # print("closest_points_temporal_subsub",closest_points_temporal_subsub)

        all_data.append({"name": example_path.name, "sorted_values": closest_points_temporal_subsub, "relevance": relevance_scores})

        pbar.update(1)

    save_json(all_data, '/path/to/save/output/depth_expansion_res.json')

    pbar.close()



if __name__ == '__main__':
    depth_expansion()
