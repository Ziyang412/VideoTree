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



def save_image_features(img_feats, name_ids, save_folder):
    """
    Save image features to a .pt file in a specified folder.

    Args:
    - img_feats (torch.Tensor): Tensor containing image features
    - name_ids (str): Identifier to include in the filename
    - save_folder (str): Path to the folder where the file should be saved

    Returns:
    - None
    """
    filename = f"{name_ids}.pt"  # Construct filename with name_ids
    filepath = os.path.join(save_folder, filename)
    torch.save(img_feats, filepath)


def load_json(fn):
    with open(fn, 'r') as f:
        data = json.load(f)
    return data

def save_json(data, fn, indent=4):
    with open(fn, 'w') as f:
        json.dump(data, f, indent=indent)



# image_path = "CLIP.png"
model_name_or_path = "BAAI/EVA-CLIP-8B" # or /path/to/local/EVA-CLIP-8B
image_size = 224

processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")


def clip_es():
    device = "cuda" if torch.cuda.is_available() else "cpu" 

    model = AutoModel.from_pretrained(
    model_name_or_path, 
    torch_dtype=torch.float16,
    trust_remote_code=True).to('cuda').eval()
    resume = True

    base_path = Path('path/to/data/egoschema_frames')
    save_folder = '/path/to/egoschema/frame_features'
    all_data = []

    with open('./data/subset_answers.json', 'r') as file:
        json_data = json.load(file)    
    subset_names_list = list(json_data.keys())

    example_path_list = list(base_path.iterdir())

    pbar = tqdm(total=len(example_path_list))

    i = 0 
    max = 50

    for example_path in example_path_list:

        # for subset videos, comment out for fullset
        if example_path.name not in subset_names_list:
            continue
        # else:
        #     print("example_path in subset")

        image_paths = list(example_path.iterdir())
        image_paths.sort(key=lambda x: int(x.stem))
        img_feature_list = []
        for image_path in image_paths:
            image = Image.open(str(image_path))

            input_pixels = processor(images=image, return_tensors="pt", padding=True).pixel_values.to('cuda')

            with torch.no_grad(), torch.cuda.amp.autocast():
                image_features = model.encode_image(input_pixels)
                img_feature_list.append(image_features)
        img_feature_tensor = torch.stack(img_feature_list)
        img_feats = img_feature_tensor.squeeze(1)

        name_ids = example_path.name


        save_image_features(img_feats, name_ids, save_folder)
        pbar.update(1)


    pbar.close()

if __name__ == '__main__':
    clip_es()
