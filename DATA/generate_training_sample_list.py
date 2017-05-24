import os
import json

data_path = "test_clip_2017_03_1"
with open(os.path.join(data_path, "config.json"), 'r') as f:
    conf = json.load(f)

roi_fullname_pattern = os.path.join(data_path, conf['roi_file_pattern'])
roi_path, _ = os.path.split(roi_fullname_pattern)
roi_fullname = lambda fn: os.path.join(roi_path, fn)
roi_to_label = [[roi_fullname(f), [-1, -1]] for f in os.listdir(roi_path) if os.path.isfile(roi_fullname(f))]


json_fname = 'labelled_samples.json'
with open(os.path.join(data_path, json_fname), 'w') as f:
    json.dump({'labelled_samples': roi_to_label}, f, indent=2)