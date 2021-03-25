import numpy as np
from .retinaface import RetinaFace
import cv2

class FaceDetector:
    def __init__(self, thresh, scales=[1024, 1980], gpuid=0) -> None:
        self.thresh = thresh
        self.scales = scales
        self.gpuid = gpuid

        self.detector = RetinaFace('./model/R50', 0, self.gpuid, 'net3')

    def predict(self, img, flip=False, blur: bool=False):
        im_shape = img.shape
        target_size = self.scales[0]
        max_size = self.scales[1]
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])
        im_scale = float(target_size) / float(im_size_min)
        # prevent bigger axis from being more than max_size:
        if np.round(im_scale * im_size_max) > max_size:
            im_scale = float(max_size) / float(im_size_max)

        scales = [im_scale]
        
        faces, _ = self.detector.detect(img, threshold=self.thresh, scales=scales, do_flip=flip)

        if faces is not None:
            n_faces = faces.shape[0]
            bboxes = [face.astype(np.int) for face in faces]

            if blur:
                for bbox in bboxes:
                    sub_face = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                    sub_face = cv2.GaussianBlur(sub_face,(45, 45), 30)
                    img[bbox[1]:bbox[1]+sub_face.shape[0], bbox[0]:bbox[0]+sub_face.shape[1]] = sub_face

            return n_faces, bboxes
        else:
            return 0, []
