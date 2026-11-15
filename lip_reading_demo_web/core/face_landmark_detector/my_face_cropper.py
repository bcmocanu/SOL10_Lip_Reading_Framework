import warnings
import cv2
import os
import numpy as np

warnings.filterwarnings("ignore")

def linear_interpolate(landmarks, start_idx, stop_idx):
    start_landmarks = landmarks[start_idx]
    stop_landmarks = landmarks[stop_idx]
    delta = stop_landmarks - start_landmarks
    for idx in range(1, stop_idx - start_idx):
        landmarks[start_idx + idx] = (
            start_landmarks + idx / float(stop_idx - start_idx) * delta
        )
    return landmarks

class My_Face_Cropper:
    def __init__(
            self,
            crop_width=96,
            crop_height=96,
            start_idx=48,
            stop_idx=68,
            window_margin=12,
            convert_gray=False,
    ):
        self.crop_width = crop_width
        self.crop_height = crop_height
        self.start_idx = start_idx
        self.stop_idx = stop_idx
        self.window_margin = window_margin
        self.convert_gray = convert_gray

        self.target_size = (256, 256)
        self.interpolation = cv2.INTER_LINEAR
        self.border_mode = cv2.BORDER_CONSTANT
        self.border_value = 0

    def crop_faces(self, transform_matrix, transformed_landmarks, video_frames):

        transformed_frames = []
        for idx, frame in enumerate(video_frames):
            if self.convert_gray:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

            transformed_frame = cv2.warpAffine(
                frame,
                transform_matrix[idx],
                dsize = (self.target_size[0], self.target_size[1]),
                flags = self.interpolation,
                borderMode = self.border_mode,
                borderValue = self.border_value,
            )

            patch = self.cut_patch(
                transformed_frame,
                transformed_landmarks[idx][self.start_idx: self.stop_idx],
                self.crop_height // 2,
                self.crop_width // 2,
            )
            transformed_frames.append(patch)

        return transformed_frames

    def cut_patch(self, img, landmarks, height, width, threshold=5):
        center_x, center_y = np.mean(landmarks, axis=0)
        if abs(center_y - img.shape[0] / 2) > height + threshold:
            
            return None
        if abs(center_x - img.shape[1] / 2) > width + threshold:
            
            return None
        y_min = int(round(np.clip(center_y - height, 0, img.shape[0])))
        y_max = int(round(np.clip(center_y + height, 0, img.shape[0])))
        x_min = int(round(np.clip(center_x - width, 0, img.shape[1])))
        x_max = int(round(np.clip(center_x + width, 0, img.shape[1])))
        cutted_img = np.copy(img[y_min:y_max, x_min:x_max])
        return cutted_img

