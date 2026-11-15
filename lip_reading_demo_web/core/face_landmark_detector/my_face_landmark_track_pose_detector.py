import warnings
import cv2
import numpy as np

from .my_retina_face_detector import MyRetinaFaceDetector
from .my_landmark_detector import MyLandmarkDetector
from .my_simple_face_tracker import MySimpleFaceTracker
from .my_head_pose_estimator import MyHeadPoseEstimator
from .. import pathFunctions

warnings.filterwarnings("ignore")

class My_Face_Landmark_Track_Pose_Detector:
    def __init__(self, model_dir=None, device='cuda:0', face_threshold=0.8, tracker_iou_threshold=0.4):
        """Create the face detection, tracking, landmark and pose pipeline.

        model_dir should contain:
        - Resnet50_Final.pth
        - 2dfan2.pth
        - bfm_lms.npy
        """
        from pathlib import Path

        if model_dir is None:
            model_dir = Path(__file__).resolve().parents[1] / "models"
        model_dir = Path(model_dir)

        modelFace_Path = str(model_dir / "Resnet50_Final.pth")
        modelLandmark_Path = str(model_dir / "2dfan2.pth")
        mean_shape_path = str(model_dir / "bfm_lms.npy")

        self.device = device
        self.model_dir = model_dir
        self.face_detector = MyRetinaFaceDetector(
            threshold=face_threshold,
            device=device,
            model=MyRetinaFaceDetector.get_model(modelFace_Path)
        )
        self.landmark_detector = MyLandmarkDetector(
            device=device,
            model=MyLandmarkDetector.get_model(modelLandmark_Path)
        )
        self.face_tracker = MySimpleFaceTracker(iou_threshold=tracker_iou_threshold)
        self.head_pose_estimator = MyHeadPoseEstimator(mean_shape_path)

        self.infoFaces = {}
        self.frame_number = 0

    def processAFrame_detect_track(self, frame):
        faces = self.face_detector(frame, rgb=False)
        track_ids = self.face_tracker(faces)
        return {
            "faces": faces,
            "track_ids": track_ids,
        }

    def processAFrame_landmarks_pose(self, frame, faces):
        if faces is None or len(faces) == 0:
            return {
                "landmarks": np.empty((0, 10)),
                "scores": np.empty((0,)),
                "head_poses": [],
            }

        landmarks, scores = self.landmark_detector(frame, faces, rgb=False)
        head_poses = [
            self.head_pose_estimator(face[5:15].reshape((-1, 2)), *frame.shape[1::-1])
            for face in faces
        ]

        return {
            "landmarks": landmarks,
            "scores": scores,
            "head_poses": head_poses,
        }

    def _filter_tracks(
        self,
        infoSegment,
        frame_width,
        frame_height,
        start_frame,
        end_frame,
        size_ratio_face2face=None,
        min_percent_face2img=None,
        require_full_length=True,
        fps=None,
        show_stats=False
    ):
        size_per_track = {}
        frames_per_track = {}

        for frNr, frame_data in infoSegment.items():
            faces = frame_data.get("faces", [])
            track_ids = frame_data.get("track_ids", [])

            for i, tid in enumerate(track_ids):
                
                if i >= len(faces):
                    continue
                x1, y1, x2, y2 = faces[i][:4]
                area = float((x2 - x1) * (y2 - y1))

                size_per_track.setdefault(tid, []).append(area)
                frames_per_track.setdefault(tid, []).append(frNr)

        if not size_per_track:
            return None

        avg_size = {tid: float(np.mean(areas)) for tid, areas in size_per_track.items()}
        max_avg = max(avg_size.values()) if avg_size else 0.0
        threshold_rel = max_avg * float(size_ratio_face2face)

        image_area = float(frame_width * frame_height)
        min_area_abs = (float(min_percent_face2img) / 100.0) * image_area

        valid_by_size = {
            tid for tid, a in avg_size.items()
            if a >= threshold_rel and a >= min_area_abs
        }

        if require_full_length:
            segment_set = set(range(start_frame, end_frame + 1))
            valid_by_length = {
                tid for tid, frs in frames_per_track.items()
                if set(frs) == segment_set
            }
        else:
            valid_by_length = set(frames_per_track.keys())

        valid_track_ids = valid_by_size.intersection(valid_by_length)

        if show_stats:
            print("\n=== Statistici dimensiune fețe ===")
            for tid in sorted(frames_per_track.keys()):
                a = avg_size.get(tid, 0.0)
                pct = (a / image_area) * 100.0 if image_area > 0 else 0.0
                status = "VALID" if tid in valid_track_ids else "REMOVE"
                if fps is not None:
                    t0 = pathFunctions.frames_to_timecode(frames_per_track[tid][0], fps)
                    t1 = pathFunctions.frames_to_timecode(frames_per_track[tid][-1], fps)
                    print(f"Track {tid} {t0} - {t1} : {pct:.2f}% din imagine (media) -> {status}")
                else:
                    print(f"Track {tid}: {pct:.2f}% din imagine (media) -> {status}")

        if not valid_track_ids:
            return None

        for frNr, frame_data in infoSegment.items():
            faces = frame_data.get("faces", [])
            track_ids = frame_data.get("track_ids", [])

            new_faces = []
            new_tids = []

            for i, tid in enumerate(track_ids):
                if tid in valid_track_ids and i < len(faces):
                    new_faces.append(faces[i])
                    new_tids.append(tid)

            frame_data["faces"] = np.array(new_faces) if new_faces else np.empty((0, 15))
            frame_data["track_ids"] = new_tids

        return infoSegment

    def _filter_head_poses_faces(self, infoSegment, fps, show_stats=False):

        min_PYR_TH = 50  
        mean_PYR_TH = 25  

        head_pose_data = {}
        frames_per_person = {}

        for frNr, frame_data in infoSegment.items():
            head_poses = frame_data.get('head_poses', [])
            track_ids = frame_data.get('track_ids', [])

            for i, track_id in enumerate(track_ids):
                if i >= len(head_poses):
                    continue
                pitch, yaw, roll = head_poses[i]

                if track_id not in head_pose_data:
                    head_pose_data[track_id] = {'pitch': [], 'yaw': [], 'roll': []}
                    frames_per_person[track_id] = []

                frames_per_person[track_id].append(frNr)
                head_pose_data[track_id]['pitch'].append(float(pitch))
                head_pose_data[track_id]['yaw'].append(float(yaw))
                head_pose_data[track_id]['roll'].append(float(roll))

        if not head_pose_data:
            return None

        valid_track_ids = set()

        if show_stats:
            print("\n=== Statistici Head Pose ===")

        for track_id, values in head_pose_data.items():
            P = values['pitch']
            Y = values['yaw']
            R = values['roll']

            pitch_mean = float(np.mean(P)) if P else 0.0
            yaw_mean   = float(np.mean(Y)) if Y else 0.0
            roll_mean  = float(np.mean(R)) if R else 0.0

            pitch_min = float(np.min(P)) if P else 0.0
            yaw_min   = float(np.min(Y)) if Y else 0.0
            roll_min  = float(np.min(R)) if R else 0.0

            pitch_max = float(np.max(P)) if P else 0.0
            yaw_max   = float(np.max(Y)) if Y else 0.0
            roll_max  = float(np.max(R)) if R else 0.0

            if (
                abs(pitch_min) < min_PYR_TH and abs(yaw_min) < min_PYR_TH and abs(roll_min) < min_PYR_TH and
                abs(pitch_max) < min_PYR_TH and abs(yaw_max) < min_PYR_TH and abs(roll_max) < min_PYR_TH and
                abs(pitch_mean) < mean_PYR_TH and abs(yaw_mean) < mean_PYR_TH and abs(roll_mean) < mean_PYR_TH
            ):
                status = "VALID"
                valid_track_ids.add(track_id)
            else:
                status = "REMOVE"

            if show_stats:
                t0f = frames_per_person[track_id][0]
                t1f = frames_per_person[track_id][-1]
                t0 = pathFunctions.frames_to_timecode(t0f, fps) if fps is not None else str(t0f)
                t1 = pathFunctions.frames_to_timecode(t1f, fps) if fps is not None else str(t1f)

                print(f"Track {track_id} {t0} - {t1} -> "
                      f"PitchMIN:{pitch_min:.2f} YawMIN:{yaw_min:.2f} RollMIN:{roll_min:.2f}")
                print(f"Track {track_id} {t0} - {t1} -> "
                      f"PitchMAX:{pitch_max:.2f} YawMAX:{yaw_max:.2f} RollMAX:{roll_max:.2f}")
                print(f"Track {track_id} {t0} - {t1} -> "
                      f"Pitch:{pitch_mean:.2f} Yaw:{yaw_mean:.2f} Roll:{roll_mean:.2f}")
                print(f"Track {track_id} {t0} - {t1}: {status}")

        if not valid_track_ids:
            return None

        for _, frame_data in infoSegment.items():
            faces     = frame_data.get('faces', [])
            landmarks = frame_data.get('landmarks', [])
            scores    = frame_data.get('scores', [])
            head_poses= frame_data.get('head_poses', [])
            track_ids = frame_data.get('track_ids', [])

            new_faces, new_landmarks, new_scores, new_head_poses, new_track_ids = [], [], [], [], []

            for i, tid in enumerate(track_ids):
                if tid in valid_track_ids:
                    if i < len(faces):
                        new_faces.append(faces[i])
                    if len(landmarks) > 0 and i < len(landmarks):
                        new_landmarks.append(landmarks[i])
                    if len(scores) > 0 and i < len(scores):
                        new_scores.append(scores[i])
                    if len(head_poses) > 0 and i < len(head_poses):
                        new_head_poses.append(head_poses[i])
                    new_track_ids.append(tid)

            frame_data['faces']     = np.array(new_faces) if new_faces else np.empty((0, 15))
            frame_data['landmarks'] = np.array(new_landmarks) if new_landmarks else np.empty((0, 10))
            frame_data['scores']    = np.array(new_scores) if new_scores else np.empty((0,))
            frame_data['head_poses']= new_head_poses
            frame_data['track_ids'] = new_track_ids

        return infoSegment

    def processVideo(
        self,
        cap,
        start_frame=None,
        end_frame=None,
        size_ratio_face2face=None,
        min_percent_face2img=None,
        require_full_length=True,
        frame_width=None,
        frame_height=None,
        fps=None,
        show_stats=False
    ):
        if start_frame is None or end_frame is None:
            raise ValueError("start_frame și end_frame trebuie să fie setate (nu None).")
        if frame_width is None or frame_height is None:
            raise ValueError("frame_width și frame_height trebuie să fie setate (nu None).")

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        self.frame_number = start_frame
        self.infoFaces = {}

        possible_full_tracks = None  
        
        min_area_abs = None
        if min_percent_face2img is not None:
            min_area_abs = (float(min_percent_face2img) / 100.0) * float(frame_width * frame_height)

            while True:
                ok, frame = cap.read()
                if not ok or frame is None:
                    break

                frame_info = self.processAFrame_detect_track(frame)
                self.infoFaces[self.frame_number] = frame_info

                faces = frame_info.get("faces", [])
                tids_list = frame_info.get("track_ids", [])
                current_track_ids = set(tids_list)

                if self.frame_number == start_frame:
                    possible_full_tracks = set(current_track_ids)
                else:
                    if possible_full_tracks is not None:
                        possible_full_tracks &= current_track_ids

                if min_area_abs is not None and possible_full_tracks:
                    bad = set()
                    for i, tid in enumerate(tids_list):
                        if tid not in possible_full_tracks:
                            continue
                        if i >= len(faces):
                            continue

                        x1, y1, x2, y2 = faces[i][:4]
                        area = float((x2 - x1) * (y2 - y1))

                        if area < min_area_abs:
                            bad.add(tid)

                    possible_full_tracks -= bad

                if possible_full_tracks is not None and len(possible_full_tracks) == 0:
                    if show_stats:
                        print(
                            f"\n[EARLY STOP] Niciun track full-length pentru segment {pathFunctions.frames_to_timecode(start_frame, fps)} - {pathFunctions.frames_to_timecode(end_frame, fps)}")
                    break

                self.frame_number += 1
                if self.frame_number > end_frame:
                    break

        filtered = self._filter_tracks(
            self.infoFaces,
            frame_width=frame_width,
            frame_height=frame_height,
            start_frame=start_frame,
            end_frame=end_frame,
            size_ratio_face2face=size_ratio_face2face,
            min_percent_face2img=min_percent_face2img,
            require_full_length=require_full_length,
            fps=fps,
            show_stats=show_stats
        )

        if filtered is None:
            self.infoFaces = {}
            return None

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frNr = start_frame

        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            if frNr > end_frame:
                break

            frame_data = self.infoFaces.get(frNr, None)
            if frame_data is None:
                frNr += 1
                continue

            faces = frame_data.get("faces", None)
            extra = self.processAFrame_landmarks_pose(frame, faces)

            frame_data["landmarks"] = extra["landmarks"]
            frame_data["scores"] = extra["scores"]
            frame_data["head_poses"] = extra["head_poses"]

            frNr += 1

        filtered_pose = self._filter_head_poses_faces(self.infoFaces, fps, show_stats=show_stats)
        if filtered_pose is None:
            self.infoFaces = {}
            return None

        self.infoFaces = filtered_pose
        return self.infoFaces

    def reset(self):
        self.infoFaces = {}
        self.frame_number = 0
        self.face_tracker.reset(reset_tracklet_counter=False)

