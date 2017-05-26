import os
import json

LABEL_POLICY = 2

config_fname = 'config.json' if LABEL_POLICY == 1 else 'config2.json'
data_path = "test_clip_2017_03_1"
with open(os.path.join(data_path, config_fname), 'r') as f:
    conf = json.load(f)

if LABEL_POLICY == 1:

    roi_fullname_pattern = os.path.join(data_path, conf['roi_file_pattern'])
    roi_path, _ = os.path.split(roi_fullname_pattern)
    roi_fullname = lambda fn: os.path.join(roi_path, fn)
    roi_to_label = [[roi_fullname(f), [-1, -1]] for f in os.listdir(roi_path) if os.path.isfile(roi_fullname(f))]


    json_fname = 'labelled_samples.json'
    with open(os.path.join(data_path, json_fname), 'w') as f:
        json.dump({'labelled_samples': roi_to_label}, f, indent=2)

elif LABEL_POLICY == 2:
    frames_to_label = range(370, 430, 10) + range(500, 1301, 100) + range(1330, 1650, 30)
    samples = []
    for f_id in frames_to_label:
        frame_fname = os.path.join(data_path, 'original_frames', 'f{:06d}.png'.format(f_id))
        samples.append([frame_fname, [], []])

    json_fname = 'labelled_samples2.json'
    with open(os.path.join(data_path, json_fname), 'w') as f:
        json.dump({'roi': conf['total_roi'], 'samples':samples}, f, indent=2)
