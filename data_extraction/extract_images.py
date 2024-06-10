import cv2
from pathlib import Path
from tqdm import tqdm
import json


def load_json(fn):
    with open(fn, 'r') as f:
        data = json.load(f)
    return data

def save_json(data, fn, indent=4):
    with open(fn, 'w') as f:
        json.dump(data, f, indent=indent)


def extract_es():
    input_base_path = Path('/path/to/egoschema/videos')
    output_base_path = Path('/path/to/data/egoschema_frames)')
    fps = 1
    pbar = tqdm(total=len(list(input_base_path.iterdir())))
    for video_fp in input_base_path.iterdir():
        output_path = output_base_path / video_fp.stem
        output_path.mkdir(parents=True, exist_ok=True)
        vidcap = cv2.VideoCapture(str(video_fp))
        count = 0
        success = True
        fps_ori = int(vidcap.get(cv2.CAP_PROP_FPS))   
        frame_interval = int(1 / fps * fps_ori)
        while success:
            success, image = vidcap.read()
            if not success:
                break
            if count % (frame_interval) == 0 :
                cv2.imwrite(f'{output_path}/{count}.jpg', image)
            count+=1
        pbar.update(1)
    pbar.close()


if __name__ == '__main__':
    extract_es()