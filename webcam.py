import os

import cv2
import dlib
import numpy as np
from PIL import Image

import util.util as util
from data.custom_dataset_data_loader import CustomDatasetDataLoaderInference
from models.models import create_model
from options.test_options import TestOptions
import pdb


class LandmarkDetector:
    def __init__(self):
        face_dataset_path = './datasets/face'
        predictor_path = os.path.join(face_dataset_path, 'shape_predictor_68_face_landmarks.dat')
        self._detector = dlib.get_frontal_face_detector()
        self._predictor = dlib.shape_predictor(predictor_path)

    def get_keypoints(self, image):
        # io.imread
        # PIL to numpy array
        image = np.array(image)
        points = np.array([])
        dets = self._detector(image, 1)
        if len(dets) > 0:
            shape = self._predictor(image, dets[0])
            points = np.empty([68, 2], dtype=int)
            for b in range(68):
                points[b, 0] = shape.part(b).x
                points[b, 1] = shape.part(b).y

        # make sure to return an ndarray
        return points.astype(np.float64)


class Synthesizer(LandmarkDetector):
    def __init__(self, opt):
        super().__init__()
        # TODO - below line should be in main function
        opt = TestOptions().parse()

        ### setup dataset
        _data_loader = self.CreateDataLoaderInference(opt)
        self._dataset = _data_loader.load_data()

        ### setup models
        self._model = create_model(opt)
        self._model.eval()

    @staticmethod
    def CreateDataLoaderInference(opt):
        data_loader = CustomDatasetDataLoaderInference()
        print(data_loader.name())
        data_loader.initialize(opt)
        return data_loader

    def synthesize(self, image):
        keypoints = self.get_keypoints(image)
        synthesized_image = np.array([])
        tgt_image = None
        ref_image = None
        data_list = None

        # process only if keypoints are found
        if keypoints.any():
            data = self._dataset.process(img_seq=image, keypoints_seq=keypoints)
            data_list = [data['tgt_label'], data['tgt_image'], None, None, data['ref_label'], data['ref_image'], None,
                         None,
                         None]
            synthesized_image, _, _, _, _, _ = self._model(data_list)
            # pdb.set_trace()

            synthesized_image = util.tensor2im(synthesized_image)
            tgt_image = util.tensor2im(data['tgt_image'])
            ref_image = util.tensor2im(data['ref_image'], tile=True)

        return {
            'synthesized_image': synthesized_image,
            'tgt_image': tgt_image,
            'ref_image': ref_image,
            'data_list': data_list
        }


def get_frame_rate_of_web_cam():
    # Start default camera
    video = cv2.VideoCapture(0)
    # Find OpenCV version
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

    # getting total FPS of the web cam
    if int(major_ver) < 3:
        fps = video.get(cv2.cv.CV_CAP_PROP_FPS)
        print("Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps))
    else:
        fps = video.get(cv2.CAP_PROP_FPS)
        print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))

    video.release()
    return fps


def capture_frame(req_fps, web_cam_fps, prediction_object):
    # Start default camera
    video = cv2.VideoCapture(0)
    cv2.namedWindow("test")
    capture_at = int(web_cam_fps / req_fps)
    frame_counter = 0

    video.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    video.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    while True:
        ret, frame = video.read()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if not ret:
            break

        synthesized_frame = np.array([])
        if frame_counter % capture_at == 0:
            frame_PIL = Image.fromarray(np.uint8(frame))
            synthesized_frame = prediction_object.synthesize(image=frame_PIL)['synthesized_image']
        frame_counter += 1

        cv2.imshow("test", synthesized_frame if synthesized_frame.any() else frame)

    video.release()


def main():
    opt = TestOptions().parse()
    synthesizer = Synthesizer(opt)
    image_dir = './datasets/face/test_images/0001'
    image_paths = sorted(os.listdir(image_dir))
    outputs = []
    for image_path in image_paths:
        image = Image.open(os.path.join(image_dir, image_path))
        op = synthesizer.synthesize(image=image)
        op_PIL = Image.fromarray(op['synthesized_image'])
        op_PIL.save(os.path.join('./synthesized', image_path))
        op['path'] = image_path
        outputs.append(op)

    return outputs


def run():
    # process frame at the cam_fps/fps_arg per second
    # Eg: if cam_fps = 30 and fps_arg = 3; every 10th frame is processed
    fps_arg = 20
    total_fps = get_frame_rate_of_web_cam()

    opt = TestOptions().parse()
    synthesizer = Synthesizer(opt)
    capture_frame(fps_arg, total_fps, synthesizer)


if __name__ == '__main__':
    run()
# Go on... i was just looking...
