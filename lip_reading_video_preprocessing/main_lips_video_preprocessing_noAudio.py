# V01_01 - 18.08.2025
##########################################################################
##########################################################################
import subprocess
import cv2
import imageio
import argparse

import pathFunctions
from face_landmark_detector.my_face_cropper import My_Face_Cropper
from face_landmark_detector.my_face_landmark_track_pose_detector import My_Face_Landmark_Track_Pose_Detector
from face_landmark_detector.my_face_aligner import My_Face_Aligner
##########################################################################
##########################################################################


##########################################################################
##########################################################################
FPS = 25
MINIMUM_VIDEO_LENGTH = 1*FPS
MAXIMUM_VIDEO_LENGTH = 15*FPS
##########################################################################
##########################################################################


##########################################################################
##########################################################################
def splitVideoIntoTemporalSegments(videoPath):

    # THIS FUNCTION SHOULD BE UPDATED TO SPLIT THE VIDEO (Based on Shot boundaries) - if necessary
    infoVideoSegments = []

    cap = cv2.VideoCapture(videoPath)

    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    aSeg = {
        "start_frame": 0,
        "end_frame": total_frames - 1
    }
    infoVideoSegments.append(aSeg)

    cap.release()
    return infoVideoSegments
##########################################################################
##########################################################################


##########################################################################
##########################################################################
def extractFrames(videoPath, start_frame = None, end_frame = None):

    frames = []
    cap = cv2.VideoCapture(videoPath)

    if start_frame is not None:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frame_number = start_frame
    else:
        frame_number = 0

    while True:
        _, frame = cap.read()
        if frame is None:
            break
        else:
            frames.append(frame)
            frame_number = frame_number + 1

        if end_frame is not None and frame_number > end_frame:
            break

    cap.release()

    return frames
##########################################################################
##########################################################################


##########################################################################
##########################################################################
def detect_Faces_Landmarks_Tracks_Poses(videoPath, infoVideoSegments):

    face_landmark_track_pose_detector = My_Face_Landmark_Track_Pose_Detector()

    for seg in infoVideoSegments:
        infoSegment = face_landmark_track_pose_detector.processVideo(videoPath, start_frame = seg["start_frame"], end_frame=seg["end_frame"])
        seg["info_faces"] = infoSegment
        face_landmark_track_pose_detector.reset()

    return infoVideoSegments
##########################################################################
##########################################################################


##########################################################################
##########################################################################
def align_faces(peopleData):
    my_face_aligner = My_Face_Aligner()

    for plID, plData in peopleData.items():
        for segID, plDataSeg in enumerate(plData):
            transformed_landmarks, transform_matrix = my_face_aligner.align_landmarks(plDataSeg["landmarks"])
            plDataSeg["transformed_landmarks"] = transformed_landmarks
            plDataSeg["transform_matrix"] = transform_matrix

    return peopleData
##########################################################################
##########################################################################

##########################################################################
##########################################################################
def convert_fps_ffmpeg(video_original_path, video_final_path):
    command = [
        "ffmpeg",
        "-y",  # suprascrie fără confirmare
        "-i", video_original_path,
        "-r",
        str(FPS),
        video_final_path
    ]
    try:
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as e:
        print("Eroare la conversie fps cu ffmpeg:", e)

    return None
##########################################################################
##########################################################################

##########################################################################
##########################################################################
def convert_to_peopleBased(infoVideoSegments):

    allPeopleData = {}
    for seg in infoVideoSegments:
        peopleData = {}
        for frNr, frameData in seg["info_faces"].items():
            for idx, trackID in enumerate(frameData["track_ids"]):
                if trackID not in peopleData:
                    peopleData[trackID] = {
                        "faces": [],
                        "landmarks": [],
                        "scores": [],
                        "head_poses": [],
                        "frames":[]
                    }

                peopleData[trackID]["faces"].append(frameData["faces"][idx])
                peopleData[trackID]["landmarks"].append(frameData["landmarks"][idx])
                peopleData[trackID]["scores"].append(frameData["scores"][idx])
                peopleData[trackID]["head_poses"].append(frameData["head_poses"][idx])
                peopleData[trackID]["frames"].append(frNr)

        seg["people_data"] = peopleData

        for plId, plData in peopleData.items():
            if plData["frames"][-1] - plData["frames"][0] > MINIMUM_VIDEO_LENGTH: ##Delete if too short 3 sec
                allPeopleData[plId] = [plData]

    return allPeopleData
##########################################################################
##########################################################################


##########################################################################
##########################################################################
def splitPeopleDetectionsIfTooLong(peopleData):

    for plID, plData in peopleData.items():
        if plData[0]["frames"][-1] - plData[0]["frames"][0] + 1 > MAXIMUM_VIDEO_LENGTH:
            print("A person - {} - appear for more than 15 sec - TO DO - split in multiple segments".format(plID))
            #TO DO - Maybe analyse the mouth to see if it speaks! Use aligned landmarks

    return peopleData
##########################################################################
##########################################################################


##########################################################################
##########################################################################
def cropFaces(peopleData, videoPath_25fps):

    my_face_cropper = My_Face_Cropper()
    for plID, plData in peopleData.items():
        for segID, plDataSeg in enumerate(plData):
            if plDataSeg["frames"][-1] - plDataSeg["frames"][0] < MAXIMUM_VIDEO_LENGTH and plDataSeg["frames"][-1] - plDataSeg["frames"][0] > MINIMUM_VIDEO_LENGTH:

                videoFrames = extractFrames(videoPath_25fps, start_frame = plDataSeg["frames"][0], end_frame = plDataSeg["frames"][-1])
                plDataSeg["transformed_frames"] = my_face_cropper.crop_faces(plDataSeg['transform_matrix'], plDataSeg["transformed_landmarks"],videoFrames)

    pathFunctions.delete_file(videoPath_25fps)

    return peopleData
##########################################################################
##########################################################################

##########################################################################
##########################################################################
def saveVideoWithDetectedPeople_opencv(peopleData, video_dir):

    for plID, plData in peopleData.items():
        for segID, plDataSeg in enumerate(plData):

            output_file = video_dir + pathFunctions.get_last_folder(video_dir) + "_ID" + str(plID) + "_SEG" + str(segID) + "_NoAudio.avi"
            frame_height, frame_width, _ = plDataSeg["transformed_frames"][0].shape

            #fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fourcc = cv2.VideoWriter_fourcc(*'H264')
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(output_file, fourcc, FPS, (frame_width, frame_height))

            for idx, frame in enumerate(plDataSeg["transformed_frames"]):
                out.write(frame)

            out.release()
            cv2.destroyAllWindows()

    return
##########################################################################
##########################################################################


##########################################################################
##########################################################################
def saveVideosWithDetectedPeople_IMAGEIO(peopleData, video_dir):

    for plID, plData in peopleData.items():
        for segID, plDataSeg in enumerate(plData):

            output_file = video_dir + pathFunctions.get_last_folder(video_dir) + "_ID" + str(plID) + "_SEG" + str(segID) + "_NoAudio.mp4"
            frame_height, frame_width, _ = plDataSeg["transformed_frames"][0].shape

            with imageio.get_writer(output_file, fps=FPS, codec='libx264', pixelformat='yuv420p', ffmpeg_params=['-crf', '16', '-preset', 'slow']) as writer:
                for idx, frame in enumerate(plDataSeg["transformed_frames"]):
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    writer.append_data(frame_rgb)

    return
##########################################################################
##########################################################################


##########################################################################
##########################################################################
def processOneVideos(video_path):

    video_dir = pathFunctions.get_path_and_file_name_without_extension(video_path) + "/"
    pathFunctions.delete_and_recreate_folder(video_dir)

    #Step 1: Convert to 25 fps
    print(">>> Converteste video la 25 fps, fara audio!")
    videoPath_25fps = pathFunctions.replace_extension(video_path, newExtension="_25fps.mp4")
    convert_fps_ffmpeg(video_path, videoPath_25fps)

    #Step 2: Split video into video segments - TO DO
    infoVideoSegments = splitVideoIntoTemporalSegments(videoPath_25fps)

    #Step 3: Detect faces, landmarks, tracks, poses
    print(">>> Detecteaza fetele, tracking, puncte de interes!")
    infoVideoSegments = detect_Faces_Landmarks_Tracks_Poses(videoPath_25fps, infoVideoSegments)

    #Step 4: Convert info from frame based TO people based
    peopleData = convert_to_peopleBased(infoVideoSegments)

    #Step 5: Align the detected faces
    print(">>> Aliniaza fetele!")
    peopleData = align_faces(peopleData)

    #Step 6: Split the segments where a person is detected if the sequence is too long > 15 sec
    peopleData = splitPeopleDetectionsIfTooLong(peopleData)

    #Step 7: Crop faces to 96x96
    print(">>> Decupeaza fetele din fluxul video!")
    peopleData = cropFaces(peopleData, videoPath_25fps)

    #Step 8: Save videos
    print(">>> Salveaza videourile cu zona gurii!")
    saveVideosWithDetectedPeople_IMAGEIO(peopleData, video_dir)

    return
##########################################################################
##########################################################################

##########################################################################
##########################################################################
def main():
    parser = argparse.ArgumentParser(description="Preprocesare video pentru lip reading!")
    parser.add_argument(
        "video_path",
        type=str,
        help="Calea catre fisierul video (ex: /path/to/video.mp4)"
    )
    args = parser.parse_args()
    print("RUN lip_reading_video_preprocessing - V01_01 18.08.2025")
    processOneVideos(args.video_path)
    return
##########################################################################
##########################################################################

##########################################################################
##########################################################################
if __name__ == "__main__":
    main()
##########################################################################
##########################################################################