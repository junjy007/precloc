import os
import json
import numpy as np
import torch
import torch.nn as nn
import cv2
from torch.autograd import Variable
from sklearn.cross_validation import train_test_split


USE_CUDA = torch.cuda.is_available()
DATA_PATH = "../../DATA"
DATA_SET = "test_clip_2017_03_1"
LIST_FILE = "labelled_samples2_done2.json"
# Model Hyper-parameters
INPUT_CHANNELS = 3
KSIZE1 = 3
KNUM1 = 32
KSIZE2 = 3
KNUM2 = 64
HIDDEN_UNITS_NUM1 = 256
LOCATION_NUM = 120
# Data
TOTAL_SAMPLES = 1000
TRAIN_SPLIT_RATIO = 0.9
DETECTION_AREA_WIDTH = 500
DETECTION_AREA_HEIGHT = 20
# Training
LEARNING_RATE = 1e-4
SAVE_EVERY_N_STEP = 10
SAVE_PATH = '../../RUNS/detector_snapshots'

def get_value(v):
    """
    torch Variable -> np array
    :type v: Variable
    :return:
    """
    va = v.data
    return (va.cpu().numpy() if USE_CUDA else va.numpy())

def tensor_to_var(t, rg):
    if USE_CUDA:
        t=t.cuda()
    return Variable(t, requires_grad=rg)

def get_variable_32F(a, rg=True):
    t = torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))
    return tensor_to_var(t, rg)

def get_variable_LongInt(a, rg=True):
    t = torch.LongTensor(a)
    return tensor_to_var(t, rg)

######################################
# DATA
######################################
def preprocessor(im, m):
    im2 = np.float32(im) - np.float32(m)
    im2 = np.maximum(im2, -128.0)
    im2 = np.minimum(im2, 127.0)
    im2 /= 128.0
    return im2

class Data:
    def __init__(self):
        self.rng = np.random.RandomState(0)
        with open(os.path.join(DATA_PATH, DATA_SET, LIST_FILE), 'r') as f:
            d = json.load(f)
        self.roi = d['roi']
        samples = d['samples']
        fullname_fn = lambda x: os.path.join(DATA_PATH, x)
        self.train_images, self.label_images, self.mean_bgr = \
            self._get_labelled_images(samples, fullname_fn)

        self.images, self.labels = self.setup(TOTAL_SAMPLES)
        self.train_images, self.valid_images, \
        self.train_labels, self.valid_labels = train_test_split(
            self.images, self.labels,
            test_size = 1.0 - TRAIN_SPLIT_RATIO, random_state = 42)

    def data_source(self, data_split, minibatch):
        i = 0
        ep = 0
        b = 0
        image_set, label_set = (self.train_images, self.train_labels) \
            if data_split == 'train' \
            else (self.valid_images, self.valid_labels)
        while True:
            yield image_set[i:i+minibatch], label_set[i:i+minibatch], ep, b
            i += minibatch
            b += 1
            if i > len(image_set) - minibatch:
                i = 0
                b = 0
                ep += 1

    def _get_labelled_images(self, samples, fullname_fn):
        """
        :param samples: a list, samples[i] is
          [filename, [x-coordinates of control points], [y-...]]
        :param fullname_fn: converts file name to full.
        """
        train_images = []
        label_images = []
        mean_bgr_ = np.zeros(3)
        for fname, xs, ys in samples:
            im = cv2.imread(fullname_fn(fname))
            label_im = np.zeros_like(im[:, :, 0])
            for x0, x1, y0, y1 in zip(xs[:-1], xs[1:], ys[:-1], ys[1:]):
                cv2.line(label_im, (int(x0), int(y0)), (int(x1), int(y1)), 255, 2)

            x0_, x1_ = self.roi['left'], self.roi['left'] + self.roi['width']
            y0_, y1_ = self.roi['top'], self.roi['top'] + self.roi['height']

            train_images.append(im[y0_:y1_, x0_:x1_, :])
            label_images.append(label_im[y0_:y1_, x0_:x1_])

            mean_bgr_ += train_images[-1].mean(axis=0).mean(axis=0)

            # im2 = cv2.addWeighted(im, 0.5, cv2.cvtColor(label_im, cv2.COLOR_GRAY2BGR), 0.5, 0)
            # cv2.imshow("a", im2)
            # cv2.waitKey()
        # cv2.destroyAllWindows()
        return train_images, label_images, mean_bgr_ / len(train_images)

    def _get_label_code_h(self, label_im, num_classes):
        """
        Encode the horizontal position of the labels -- the left most pixel
        """
        onehot_code = np.zeros(num_classes + 1, dtype=np.uint8)
        w = label_im.shape[1]
        _, xs = np.nonzero(label_im)
        if xs.size > 0:
            xmin = np.min(xs)
            pos_w = float(w) / num_classes
            cls_id = int(float(xmin) / pos_w)
            onehot_code[cls_id] = 1
        else:
            xmin = -1
            onehot_code[-1] = 1
            cls_id = num_classes
        return onehot_code, cls_id, xmin

    def setup(self, sample_num=1000):
        debug_display = False

        x_offset_max = self.train_images[0].shape[1] - DETECTION_AREA_WIDTH
        y_offset_max = self.train_images[0].shape[0] - DETECTION_AREA_HEIGHT
        rng = np.random.RandomState(0)

        input_x = []
        gnd_y = []
        for i in range(sample_num):
            image_id = rng.randint(len(self.train_images))

            x0_, y0_ = rng.randint(x_offset_max), rng.randint(y_offset_max)
            x1_, y1_ = x0_ + DETECTION_AREA_WIDTH, y0_ + DETECTION_AREA_HEIGHT

            train_area = self.train_images[image_id][y0_:y1_, x0_:x1_, :]
            label_area = self.label_images[image_id][y0_:y1_, x0_:x1_]
            cls_code, cls_id, _ = self._get_label_code_h(label_area, LOCATION_NUM)
            input_x.append(np.rollaxis(preprocessor(train_area, self.mean_bgr), 2))
            gnd_y.append(cls_id)

            if debug_display:
                im2 = self.train_images[image_id].copy()
                im2l = cv2.cvtColor(label_area, cv2.COLOR_GRAY2BGR)
                if cls_id < LOCATION_NUM:
                    bx0_ = int(cls_id * float(DETECTION_AREA_WIDTH) / LOCATION_NUM)
                    bx1_ = int((cls_id + 1) * float(DETECTION_AREA_WIDTH) / LOCATION_NUM)
                    im2l[:, bx0_:bx1_, 1] = 255
                    print cls_id, cls_code
                else:
                    im2l[..., 2] += 128

                im2[y0_:y1_, x0_:x1_, :] = \
                    cv2.addWeighted(im2[y0_:y1_, x0_:x1_, :], 0.5, im2l, 0.5, 0)
                cv2.imshow("a", im2)
                cv2.waitKey()

        if debug_display:
            cv2.destroyAllWindows()
        return input_x, gnd_y

class LaneCloseDetectNet(nn.Module):
    def __init__(self, opts):
        super(LaneCloseDetectNet, self).__init__()

        im_width  = opts['image_width' ]
        im_height = opts['image_height']

        self._conv_layer1 = nn.Sequential(
            nn.Conv2d(in_channels=INPUT_CHANNELS, out_channels=KNUM1, kernel_size=KSIZE1, padding=(KSIZE1 - 1) / 2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self._conv_layer2 = nn.Sequential(
            nn.Conv2d(in_channels=KNUM1, out_channels=KNUM2, kernel_size=KSIZE2, padding=(KSIZE2 - 1) / 2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self._feature = nn.Sequential(self._conv_layer1, self._conv_layer2)
        dummy_input = Variable(torch.rand(1, INPUT_CHANNELS, im_height, im_width))
        dummy_feature = self._feature(dummy_input)
        nfeat = np.prod(dummy_feature.size()[1:])

        self._fc1 = nn.Linear(in_features=nfeat, out_features=HIDDEN_UNITS_NUM1)
        self._fc2 = nn.Linear(in_features=HIDDEN_UNITS_NUM1, out_features=LOCATION_NUM + 1)
        self._fullconn = nn.Sequential(self._fc1, self._fc2, nn.LogSoftmax())

        self._num_features = nfeat

        if USE_CUDA:
            with torch.cuda.device(0):
                self.cuda()

    def forward(self, x):
        y = self._feature(x)
        y = y.view(-1, self._num_features)
        y = self._fullconn(y)
        return y

def main():
    opts = {'image_width': DETECTION_AREA_WIDTH, 'image_height': DETECTION_AREA_HEIGHT}
    ldet = LaneCloseDetectNet(opts)
    optimiser = torch.optim.Adagrad(ldet.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.NLLLoss()
    data = Data()
    train_set = data.data_source('train', minibatch=20)
    valid_set = data.data_source('valid', minibatch=20)

    fname2 = os.path.join(SAVE_PATH, 'current_status')
    if os.path.exists(fname2):
        with open(fname2, 'r') as f:
            stat_ = json.load(f)
            fname = stat_['latest_model_checkpoint']
            ldet.load_state_dict(torch.load(fname))
            total_batches = stat_['total_batches']
            loss_history = stat_['loss_history']
    else:
        total_batches = 0
        loss_history = []
    for im_batch, label_batch, episode, batch_id in train_set:
        X = get_variable_32F(im_batch, False)
        Y = get_variable_LongInt(label_batch, False)
        pred = ldet(X)
        loss = loss_fn(pred, Y)
        loss_value = get_value(loss)[0]  # single-element array
        print "Total Batch-{}, Episode-{}, Batch-{}, Loss {}".format(
            total_batches, episode, batch_id, loss_value)
        loss_history.append(float(loss_value))

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        total_batches += 1

        if total_batches % SAVE_EVERY_N_STEP == 0:
            fname = os.path.join(SAVE_PATH, 'checkpoint-{}'.format(total_batches))
            torch.save(ldet.state_dict(), fname)
            fname2 = os.path.join(SAVE_PATH, 'current_status')
            with open(fname2, 'w') as f:
                json.dump({
                    'total_batches': total_batches,
                    'loss_history': loss_history,
                    'latest_model_checkpoint': fname
                }, f)
            print "Model Saved to {}".format(fname)



if __name__ == '__main__':
    main()
