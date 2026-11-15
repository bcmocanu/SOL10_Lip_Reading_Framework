# V01_01 - 18.08.2025
##########################################################################
##########################################################################
import warnings
import cv2

from .my_retina_face_detector import MyRetinaFaceDetector
from .my_landmark_detector import MyLandmarkDetector
from .my_simple_face_tracker import MySimpleFaceTracker
from .my_head_pose_estimator import MyHeadPoseEstimator

warnings.filterwarnings("ignore")
##########################################################################
##########################################################################


##########################################################################
##########################################################################
class My_Face_Landmark_Track_Pose_Detector:
    def __init__(self):

        modelFace_Path = "./Models/Resnet50_Final.pth"
        modelLandmark_Path = "./Models/2dfan2.pth"
        mean_shape_path = "./Models/bfm_lms.npy"

        self.face_detector = MyRetinaFaceDetector(threshold=0.8, model=MyRetinaFaceDetector.get_model(modelFace_Path))
        self.landmark_detector = MyLandmarkDetector(model=MyLandmarkDetector.get_model(modelLandmark_Path))
        self.face_tracker = MySimpleFaceTracker(iou_threshold=0.4)
        self.head_pose_estimator = MyHeadPoseEstimator(mean_shape_path)

        self.infoFaces = {}
        self.frame_number = 0
    ##########################################################################

    ##########################################################################
    def processAFrame(self, frame):

        faces = self.face_detector(frame, rgb=False)
        landmarks, scores = self.landmark_detector(frame, faces, rgb=False)
        track_ids = self.face_tracker(faces)
        head_poses = [self.head_pose_estimator(face[5:15].reshape((-1, 2)), *frame.shape[1::-1]) for face in faces]

        aInfoFace = {
            "faces": faces,
            "landmarks": landmarks,
            "scores": scores,
            "head_poses": head_poses,
            "track_ids": track_ids,
        }

        return aInfoFace
    ##########################################################################

    ##########################################################################
    def processVideo(self, videoPath, start_frame = None, end_frame = None):

        cap = cv2.VideoCapture(videoPath)

        if start_frame is not None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            self.frame_number = start_frame

        while True:
            _, frame = cap.read()
            if frame is None:
                break
            else:
                aInfoFace = self.processAFrame(frame)
                self.infoFaces[self.frame_number] = aInfoFace
                self.frame_number = self.frame_number + 1

            if end_frame is not None and self.frame_number > end_frame:
                break

        cap.release()
        return self.infoFaces
    ##########################################################################

    ##########################################################################
    def reset(self):
        self.infoFaces = {}
        self.frame_number = 0
        self.face_tracker.reset(reset_tracklet_counter=False)
    ##########################################################################


##########################################################################
##########################################################################

##########################################################################
##########################################################################