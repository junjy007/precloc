import cv2
# import cv2.cv as cv
import numpy as np

class LocatablePattern:
    """
    This represents an object, which has an anchor point. The anchor can be
    precisely determined in an input image (given that the input image contains
    this pattern) by comparing to the reference image of the pattern.

    NOTE: in general, all images are "locatable", e.g. we can optimise over
    affine transforms to align one image to another. This object is focused
    on locating one *anchor*, which is usually of some significance for higher
    level processing, e.g. the world coordinate of the anchor can be known.


    TODO: use SIFT, if ORB doesn't do a proper job

    """
    def __init__(self, ref_im, anchor_point=None):
        """
        :param ref_im:
        :type ref_im: np.ndarray
        """
        if ref_im.ndim == 3:
            self.ref_im = cv2.cvtColor(ref_im, cv2.COLOR_BGR2GRAY)
        else:
            self.ref_im = ref_im
        im_h, im_w = ref_im.shape[:2]
        if anchor_point is None:
            self.ref_anchor = np.asarray([im_w, im_h], dtype=np.float32) / 2.0
        else:
            self.ref_anchor = np.float32(anchor_point)

        self.feature_detector = cv2.xfeatures2d.SIFT_create()
        self.ref_keypts, self.ref_descs = self.feature_detector.detectAndCompute(ref_im, None)
        self.matcher = cv2.BFMatcher()

        self.opts = {
            'debug': {'raw_match': False,
                      'homography': {'draw': True, 'with_matches': False}}
        }

    def _get_match_plot(self, query_im, query_im_keypoints, matches, good_criteria=0.75, draw=True):
        """
        Given two images and the keypoint matches between them, generate an image
        showing correspondence between the matches.
        :return:
        """
        # - ratio test to remove obvious outliers -- for drawing
        good = []
        for m, n in matches:
            if m.distance < good_criteria * n.distance:
                good.append([m])

        # cv2.drawMatchesKnn expects list of lists as matches.
        match_im = cv2.drawMatchesKnn(self.ref_im, self.ref_keypts,
                                      query_im,
                                      query_im_keypoints, good, None, flags=2)

        if match_im.shape[0]>800:
            ratio = 800.0 / match_im.shape[0]
            match_im = cv2.resize(match_im, dsize=None, fx=ratio, fy=ratio)

        if draw:
            cv2.imshow("matches", match_im)
            cv2.waitKey()
            cv2.destroyWindow("matches")
        return match_im

    def _get_homography_plot(self, query_im, M, opts):

        if opts['with_matches']:
            assert 0, "draw matches not implemented yet"
        h, w = self.ref_im.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)

        im2 = cv2.cvtColor(query_im, cv2.COLOR_GRAY2BGR)
        print [np.int32(dst)]
        print np.int32(dst).shape
        im2 = cv2.polylines(im2, [np.int32(dst.squeeze())], True, 255, 3, cv2.LINE_AA)
        query_anchor = cv2.perspectiveTransform(self.ref_anchor.reshape(1, 1, 2), M)
        print query_anchor
        print query_anchor.shape

        im2 = cv2.circle(im2, tuple( np.int32(query_anchor.squeeze()) ),
                         radius=2, color=(0, 0, 255), thickness=2)

        if opts['draw']:
            im1 = cv2.cvtColor(self.ref_im, cv2.COLOR_GRAY2BGR)
            im1 = cv2.circle(im1, tuple(np.int32(self.ref_anchor)),
                             radius=2, color=(0, 0, 255), thickness=2)
            cv2.imshow("f", im2)
            cv2.imshow("f1", im1)
            cv2.waitKey()
            cv2.destroyWindow("f")
            cv2.destroyWindow("f1")

        return im2

    def locate(self, input_im):
        """
        Locate the anchor point in an input image. It first detects feature points (in-fps)
        in the input image then use OpenCV matcher to correspond in-fps and ref-fps.

        See the [tutorial](http://docs.opencv.org/trunk/dc/dc3/tutorial_py_matcher.html)

        Using the correspondence, an affine map is computed, and the location of the
        anchor point in the input image is computed.

        TODO: use RANSAC, the above procedure as one iteration.
        :param input_im:
        :return: an image coordinate corresponding the *anchor point* in the input image
        """
        if input_im.ndim == 3:
            input_im = cv2.cvtColor(input_im, cv2.COLOR_BGR2GRAY)
        # detect feature points in the input image
        query_keypts, query_descs = self.feature_detector.detectAndCompute(input_im, None)

        # find matching of the feature points in (input, ref) image
        matches = self.matcher.knnMatch(self.ref_descs, query_descs, k=2)
        good = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:  # TODO: 0.7 goes to parameter settings
                good.append(m)

        if self.opts['debug']['raw_match']:
            self._get_match_plot(input_im, query_keypts, matches)

        # compute homography (2D affine transform) using keypoint detections
        # Follow [note](https://goo.gl/AC5Bm0)
        if len(good) > 10:  # TODO: param -- MIN_MATCH_COUNT:
            # Don't try to understand who is "query" and who's "train" in the naming of (one-side)
            # matches. If it is not working, exchange query and train -- in my mind, $m is
            # the first piece of a @match, which was computed using
            # (first-arg=self.ref_image, second-arg=query_image)
            # so if $match == ($m, $n), from $m's point of view, "train" should refer to self.ref_image
            # and "query" to query_image. But the example seems go the other way.
            src_pts = np.float32([self.ref_keypts[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([query_keypts[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            matchesMask = mask.ravel().tolist()

            if self.opts['debug']['homography']:
                self._get_homography_plot(input_im, M, self.opts['debug']['homography'])

            query_anchor = cv2.perspectiveTransform(self.ref_anchor.reshape(1,1,2), M)
            return query_anchor.squeeze()
        else:
            print "Not enough matches are found - %d/%d" % (len(good), MIN_MATCH)
            matchesMask = None
            return None


class CircleMarker:
    def __init__(self, char_im, point_WCS):
        """
        :param char_im: Characteristic pattern of this marker. Including one point
          recognisable whenever the marker is visible. Later when a candidate
          is detected, the candi_im and char_im can be compared to locate an anchor point
          P on the input candi_im, corresponding to the point_WCS given in this
          initialiser.
        :param point_WCS: the marker's character point in the world coordinate
        :type point_WCS: np.ndarray
        """
        self.coord_WCS = point_WCS

    def confirm_detection(self, im):
        """

        :param im:
        :type im: numpy.ndarray
        :return: True/False whether the im contains a detection
                 real number: probability of the image contains detection
                 (image coordinate) the image coord of a detection
        """
        return self


class Camera:
    def __init__(self):
        self.K = np.zeros((3, 3))  # internal parameter
        self.distort_coeff = np.zeros(5)


class CircleMarkerCalibrater:
    def __init__(self, markers, camera):
        """

        :param markers:
        :type markers: list[CircleMarker]
        :param camera:
        :type camera: Camera
        """
        self.markers = markers
        self.camera = camera
        self.rvec = np.zeros(3)
        self.tvec = np.zeros(3)
        self.opts = {
            'circle_det_resolution_factor': 1.0,
            'min_dist_between_dets': 20,
            'canny_high_threshold': 500,
            'accumulator_threshold': 15,
            'min_circle_radius': 10,
            'max_circle_radius': 80,
            'DEBUG_draw_circle_dets': True
        }

    def calibrate_from_image(self, im, warm_start=None):
        """
        :param im:
        :type im: numpy.ndarray
        :param warm_start: if not None, another ext calibration object
        """
        circle_candidates = self._detect_candidates(im)  # type: np.ndarray
        assert (circle_candidates.shape[1] == 3)

        # marker_dets = self._confirm_candidates(im, circle_candidates)

        # calib = self._calibrate_cameraCS_worldCS(marker_dets, warm_start)
        return None

    def _detect_candidates(self, im):
        """
        :param im:
        :type im: np.ndarray
        :return:
        """
        GRAY_NORMAL = 0
        GRAY_SATU   = 1
        GRAY_METHOD = GRAY_SATU
        if GRAY_METHOD == GRAY_NORMAL:
            im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        elif GRAY_METHOD == GRAY_SATU:
            im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
            im_gray = im_gray[:,:,1]  # saturation
            im_gray = np.abs(im_gray.astype(np.float32) - 128.0) * 2.0
            im_gray = im_gray.astype(np.uint8)

        im_gray = cv2.blur(im_gray, (3, 3))
        cv2.imshow("tmp2", im_gray)
        cv2.waitKey()


        circle_dets = cv2.HoughCircles(
            im_gray,  method=cv.CV_HOUGH_GRADIENT,
            dp=self.opts['circle_det_resolution_factor'],
            minDist=self.opts['min_dist_between_dets'],
            param1=self.opts['canny_high_threshold'],
            param2=self.opts['accumulator_threshold'],
            minRadius=self.opts['min_circle_radius'],
            maxRadius=self.opts['max_circle_radius'])[0]
        # The result is enclosed in a 3D array

        if self.opts['DEBUG_draw_circle_dets']:
            tmpim = im.copy()
            #tmpim = cv2.Canny(im_gray, self.opts['canny_high_threshold'], self.opts['canny_high_threshold']/2)
            #tmpim = cv2.cvtColor(tmpim, cv2.COLOR_GRAY2BGR)
            if circle_dets is None:
                circle_dets = []
            for i, c in enumerate(circle_dets):
                print c
                cv2.circle(tmpim, tuple(c[:2]), c[2], (0, 0, 255), thickness=2)
                cv2.circle(tmpim, tuple(c[:2]), 2, (0, 255, 0), thickness=2)
                msg = "{}".format(i)
                cv2.putText(tmpim, msg, tuple(c[:2]), cv2.FONT_HERSHEY_PLAIN, 3.0, (0,25,0), 2)

            cv2.imshow("det", tmpim)
            cv2.waitKey()
            cv2.destroyAllWindows()
        return circle_dets

    def _confirm_candidates(self, im, circle_candidates):
        num_candidates = circle_candidates.shape[0]
        num_markers = len(self.markers)
        confirm = np.zeros((num_candidates, num_markers), dtype=np.uint8)
        for c in circle_candidates:
            for m in self.markers:
                confirm[c, m] = m.confirm_detection(self._extract_circle_marker_area(im, c))

                # deal with confirm[c]:
                # - one confirm
                # - none confirm
                # - multiple confirm

        pass

    # noinspection PyPep8Naming
    def _calibrate_cameraCS_worldCS(self, detections, warm_start):
        """

        :param detections: Image detection of the markers. (marker, circle-detection)
        :type detections: list[(CircleMarker, np.ndarray)]
        :param warm_start: if not None, another ext calibration object
        :type warm_start: CircleMarkerCalibrater
        :return:
        """
        ipts, wpts = [], []
        for marker, cirdet in detections:
            ipts.append(cirdet[:2])
            wpts.append(marker.coord_WCS)

        if warm_start:
            rvec = warm_start.rvec
            tvec = warm_start.tvec
            use_guess = True
        else:
            rvec = np.zeros(3)
            tvec = np.zeros(3)
            use_guess = False

        cv2.solvePnP(
            objectPoints=wpts, imagePoints=ipts,
            cameraMatrix=self.camera.K, distCoeffs=self.camera.distort_coeff,
            rvec=rvec, tvec=tvec,
            useExtrinsicGuess=use_guess,
            flags=None
        )

        self.rvec[...] = rvec
        self.tvec[...] = tvec

        # The simplest way is to take the centre of the circle detection and calibrate using
        # camera internal
        # cam_iparam, img_coord, wld_coord):

    @staticmethod
    def _extract_circle_marker_area(im, circle):
        return im


if __name__ == '__main__':
    ref_im = cv2.imread('DATA/ref.png')
    #ref_im = cv2.resize(ref_im, dsize=None, fx=0.4, fy=0.4)
    test_im = cv2.imread('DATA/test2.png')
    # test_im = cv2.resize(test_im, dsize=None, fx=0.4, fy=0.4)
    pattern = LocatablePattern(ref_im)
    pattern.locate(test_im)

    #im = cv2.imread('DATA/tmp002.jpg')
    #camera = Camera()
    #markers = [CircleMarker(None,None), CircleMarker]
    #c = CircleMarkerCalibrater(markers, camera)
    #c.calibrate_from_image(im)