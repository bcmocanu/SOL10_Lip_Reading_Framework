# V01_01 - 18.08.2025
##########################################################################
##########################################################################
import warnings
import cv2
import os
import numpy as np

warnings.filterwarnings("ignore")
##########################################################################
##########################################################################


##########################################################################
##########################################################################
def linear_interpolate(landmarks, start_idx, stop_idx):
    start_landmarks = landmarks[start_idx]
    stop_landmarks = landmarks[stop_idx]
    delta = stop_landmarks - start_landmarks
    for idx in range(1, stop_idx - start_idx):
        landmarks[start_idx + idx] = (
            start_landmarks + idx / float(stop_idx - start_idx) * delta
        )
    return landmarks
##########################################################################
##########################################################################


##########################################################################
##########################################################################
class My_Face_Aligner:
    def __init__(
        self,
        window_margin=12,
    ):

        mean_face_path = "./Models/20words_mean_face.npy"

        self.reference = np.load(os.path.realpath(mean_face_path))
        self.window_margin = window_margin
    ##########################################################################

    ##########################################################################
    def align_landmarks(self, landmarks):

        preprocessed_landmarks = self.interpolate_landmarks(landmarks)
        if (not preprocessed_landmarks or len(preprocessed_landmarks) < self.window_margin):
            return None

        # Affine transformation
        transformed_landmarks_All = self.alignAll(preprocessed_landmarks)
        return transformed_landmarks_All
    ##########################################################################

    ##########################################################################
    def interpolate_landmarks(self, landmarks):
        valid_frames_idx = [idx for idx, lm in enumerate(landmarks) if lm is not None]

        if not valid_frames_idx:
            return None

        for idx in range(1, len(valid_frames_idx)):
            if valid_frames_idx[idx] - valid_frames_idx[idx - 1] > 1:
                landmarks = linear_interpolate(landmarks, valid_frames_idx[idx - 1], valid_frames_idx[idx])

        valid_frames_idx = [idx for idx, lm in enumerate(landmarks) if lm is not None]

        if valid_frames_idx:
            landmarks[: valid_frames_idx[0]] = [
                landmarks[valid_frames_idx[0]]
            ] * valid_frames_idx[0]
            landmarks[valid_frames_idx[-1] :] = [landmarks[valid_frames_idx[-1]]] * (
                len(landmarks) - valid_frames_idx[-1]
            )

        assert all(lm is not None for lm in landmarks), "not every frame has landmark"

        return landmarks
    ##########################################################################

    ##########################################################################
    def alignAll(self, landmarks):

        transformed_landmarks_All = []
        transform_matrix_All = []

        for idLM, landM in enumerate(landmarks):
            window_margin = min(self.window_margin // 2, idLM, len(landmarks) - 1 - idLM)

            smoothed_landmarks = np.mean([landmarks[x] for x in range(idLM - window_margin, idLM + window_margin + 1)], axis=0)
            smoothed_landmarks += landmarks[idLM].mean(axis=0) - smoothed_landmarks.mean(axis=0)

            transformed_landmarks, transform_matrix = self.affine_transform(smoothed_landmarks, self.reference)

            transformed_landmarks_All.append(transformed_landmarks)
            transform_matrix_All.append(transform_matrix)

        return transformed_landmarks_All, transform_matrix_All
    ##########################################################################

    ##########################################################################
    def affine_transform(
        self,
        landmarks,
        reference,
        target_size=(256, 256),
        reference_size=(256, 256),
        stable_points=(28, 33, 36, 39, 42, 45, 48, 54)
    ):
        stable_reference = self.get_stable_reference(reference, stable_points, reference_size, target_size)
        transform_matrix = self.estimate_affine_transform(landmarks, stable_points, stable_reference)

        transformed_landmarks = (np.matmul(landmarks, transform_matrix[:, :2].transpose()) + transform_matrix[:, 2].transpose())

        return transformed_landmarks, transform_matrix
    ##########################################################################


    ##########################################################################
    def get_stable_reference(self, reference, stable_points, reference_size, target_size):
        stable_reference = np.vstack([reference[x] for x in stable_points])
        stable_reference[:, 0] -= (reference_size[0] - target_size[0]) / 2.0
        stable_reference[:, 1] -= (reference_size[1] - target_size[1]) / 2.0
        return stable_reference
    ##########################################################################


    ##########################################################################
    def estimate_affine_transform(self, landmarks, stable_points, stable_reference):
        return cv2.estimateAffinePartial2D(
            np.vstack([landmarks[x] for x in stable_points]),
            stable_reference,
            method=cv2.LMEDS,
        )[0]
    ##########################################################################

##########################################################################
##########################################################################

##########################################################################
##########################################################################