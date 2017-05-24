import json
import cv2
import os
import numpy as np

data_path = "test_clip_2017_03_1"
with open(os.path.join(data_path, "config.json"), 'r') as f:
    conf = json.load(f)
for frame_id in range(conf['frame_id']['start'], conf['frame_id']['end']+1, 10):

    in_fname = conf['frame_file_pattern'].format(frame_id)
    full_fname = os.path.join(data_path, in_fname)
    # Load in-frame
    frame = cv2.imread(full_fname)

    # Crop ROIs from frame
    for roi_id, r in enumerate(conf['rois']):
        x0 = r['left']
        x1 = x0 + r['width']
        y0 = r['top']
        y1 = y0 + r['height']
        roi = frame[y0:y1, x0:x1, :]

        # Process the ROI
        for c in range(3):
            roi[:,:,c] = cv2.equalizeHist(roi[:,:,c])

        # Save ROIs
        out_roi_fname = os.path.join(data_path, conf['roi_file_pattern'].format(frame_id, roi_id))
        cv2.imwrite(out_roi_fname, roi)

    cv2.imshow("test", frame)
    cv2.waitKey(20);

cv2.destroyAllWindows()
