# V01_01 - 18.08.2025
##########################################################################
##########################################################################
import logging
import os
import shutil
import datetime
from glob import glob
import time
import sys
import json
import subprocess
##########################################################################
##########################################################################


##########################################################################
##########################################################################
my_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, my_path + "/../")

COLB = "\x1b[1;32m"
COLB_ERR = "\x1b[1;31m"
COLE = "\x1b[0m"
##########################################################################
##########################################################################


##########################################################################
##########################################################################
def get_path_and_file_name_without_extension(path):
    filename, file_extension = os.path.splitext(path)
    return filename
##########################################################################
##########################################################################


##########################################################################
##########################################################################
def get_filename_without_path_without_extension(path):
    filename, file_extension = os.path.splitext(os.path.basename(path))
    return filename
##########################################################################
##########################################################################


##########################################################################
##########################################################################
def get_filename_without_path_with_extension(path):
    filename_w_ext = os.path.basename(path)
    return filename_w_ext
##########################################################################
##########################################################################


##########################################################################
##########################################################################
def get_extension(path):
    filename, file_extension = os.path.splitext(path)
    return file_extension
##########################################################################
##########################################################################


##########################################################################
##########################################################################
def get_base_path(path):
    basepath, filename = os.path.split(path)
    return basepath
##########################################################################
##########################################################################


##########################################################################
##########################################################################
def get_last_folder(path):
    last_folder = os.path.basename(os.path.dirname(path))
    return last_folder
##########################################################################
##########################################################################


##########################################################################
##########################################################################
def get_path_without_last_folder(path):
    result_path = os.path.dirname(get_base_path(path))
    return result_path
##########################################################################
##########################################################################


##########################################################################
##########################################################################
def rreplace(s, old, new, occurrence):
    li = s.rsplit(old, occurrence)
    return new.join(li)
##########################################################################
##########################################################################


##########################################################################
##########################################################################
def replace_last_folder(path, newLastFolder):
    lastFolder = get_last_folder(path)
    result_path = rreplace(path, f"/{lastFolder}/", f"/{newLastFolder}/", 1)
    return result_path
##########################################################################
##########################################################################


##########################################################################
##########################################################################
def replace_extension(path, newExtension):
    filePath = get_path_and_file_name_without_extension(path)
    newName = filePath + newExtension
    return newName
##########################################################################
##########################################################################


##########################################################################
##########################################################################
def replace_last_but_one_folder(path, newLastButOneFolder):
    pathWithoutLastFolder = get_path_without_last_folder(path)
    lastButOneFolder = get_last_folder(pathWithoutLastFolder + "/")
    result_path = rreplace(path, f"/{lastButOneFolder}/", f"/{newLastButOneFolder}/", 1)
    return result_path
##########################################################################
##########################################################################


##########################################################################
##########################################################################
def get_last_but_one_folder(path):
    pathWithoutLastFolder = get_path_without_last_folder(path)
    lastButOneFolder = get_last_folder(pathWithoutLastFolder + "/")
    return lastButOneFolder
##########################################################################
##########################################################################


##########################################################################
##########################################################################
def delete_folder(path):
    if os.path.exists(path):
        logging.info(f"Deleting folder: {path}")
        shutil.rmtree(path)
##########################################################################
##########################################################################


##########################################################################
##########################################################################
def delete_and_recreate_folder(path):
    delete_folder(path)
    os.makedirs(path)
##########################################################################
##########################################################################


##########################################################################
##########################################################################
def delete_and_recreate_folder_fastLinux(path):
    subprocess.run(['rm', '-rf', path], check=True)
    os.makedirs(path)
##########################################################################
##########################################################################


##########################################################################
##########################################################################
def verifyIf_folder_file_exist(path):
    exists = os.path.exists(path)
    return exists
##########################################################################
##########################################################################


##########################################################################
##########################################################################
def create_folder_onlyIf_not_exist(path):
    if os.path.exists(path):
        return False
    else:
        logging.info(f"Creating folder: {path}")
        os.makedirs(path)
        return True
##########################################################################
##########################################################################


##########################################################################
##########################################################################
def copy_all_from_folder(pathSrc, pathDest):
    shutil.copytree(pathSrc, pathDest)
##########################################################################
##########################################################################


##########################################################################
##########################################################################
def delete_file(file):
    if os.path.exists(file):
        os.remove(file)
##########################################################################
##########################################################################


##########################################################################
##########################################################################
def delete_files_by_pattern(pattern):

    for file_path in glob(pattern):
        if os.path.isfile(file_path):
            try:
                os.remove(file_path)
                print(f"Deleted: {file_path}")
            except PermissionError:
                print(f"Permission denied: {file_path}")
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")
##########################################################################
##########################################################################


##########################################################################
##########################################################################
def print_Nice_Time_Delta(stime, etime):
    delay = datetime.timedelta(seconds=(etime - stime))
    if delay.days > 0:
        out = str(delay).replace(" days, ", ":")
    else:
        out = str(delay)
    outAr = out.split(":")
    outms = out.split(".")
    outAr = ["%02d" % (int(float(x))) for x in outAr]
    out = ":".join(outAr)
    if len(outms) > 1:
        if len(outms[1]) > 3:
            out = out + "." + outms[1][0:3]
    else:
        out = out + ".000"
    return out
##########################################################################
##########################################################################


##########################################################################
##########################################################################
def timestamp_TO_seconds(value, fps):
    if isinstance(value, str):  # value seems to be a timestamp
        _zip_ft = zip((3600, 60, 1, 1 / fps), value.split(":"))
        seconds = sum(f * float(t) for f, t in _zip_ft)
        return seconds
    elif isinstance(value, (int, float)):  # frames
        return value / fps
    else:
        return 0
##########################################################################
##########################################################################


##########################################################################
##########################################################################
def seconds_TO_timestamp(seconds, fps):
    timestamp = "{h:02d}:{m:02d}:{s:02d}:{f:02d}".format(
        h=int(seconds / 3600),
        m=int(seconds / 60 % 60),
        s=int(seconds % 60),
        f=round((seconds - int(seconds)) * fps),
    )
    return timestamp
##########################################################################
##########################################################################


##########################################################################
##########################################################################
def seconds_TO_frames(seconds, fps):
    frames = seconds * fps
    return frames
##########################################################################
##########################################################################


##########################################################################
##########################################################################
def timestamp_TO_frames(timecode, fps, start=None):
    frames = seconds_TO_frames(
        timestamp_TO_seconds(timecode, fps) - timestamp_TO_seconds(start, fps), fps
    )
    return frames
##########################################################################
##########################################################################


##########################################################################
##########################################################################
def frames_to_timecode(frames, fps, start=None):
    timecode = seconds_TO_timestamp(
        timestamp_TO_seconds(frames, fps) + timestamp_TO_seconds(start, fps), fps
    )
    return timecode
##########################################################################
##########################################################################


##########################################################################
##########################################################################
def get_all_files_from_all_sub_directories(path, extensionPattern, sort=False):
    files = []
    for dir, _, _ in os.walk(path):
        files.extend(glob(os.path.join(dir, extensionPattern)))

    if sort:
        files.sort()

    return files
##########################################################################
##########################################################################


##########################################################################
##########################################################################
def convert_Time_FF_to_MS(timestampFF, fps=25):
    timeParts = timestampFF.split(":")
    timeInMS = (
        int(timeParts[0]) * 60 * 60 * 1000
        + int(timeParts[1]) * 60 * 1000
        + int(timeParts[2]) * 1000
        + int(timeParts[3]) * 1000 / fps
    )
    return timeInMS
##########################################################################
##########################################################################


##########################################################################
##########################################################################
def convert_Time_FF_to_FRAMES(time_str, fps=25):
    time_parts = time_str.split(":")
    hours = int(time_parts[0])
    minutes = int(time_parts[1])
    seconds = int(time_parts[2])
    frames = int(time_parts[3])

    total_seconds = hours * 3600 + minutes * 60 + seconds
    total_frames = total_seconds * fps + frames
    return total_frames
##########################################################################
##########################################################################


##########################################################################
##########################################################################
def convert_MS_to_FF(timestamp, fps=25):
    delay = datetime.timedelta(milliseconds=(timestamp))
    if delay.days > 0:
        out = str(delay).replace(" days, ", ":")
    else:
        out = str(delay)
    outAr = out.split(":")
    outms = out.split(".")
    outAr = ["%02d" % (int(float(x))) for x in outAr]
    out = ":".join(outAr)
    if len(outms) == 2:
        ms = int(fps * int(outms[1][0:2]) / 100)
        out = out + "." + str(ms).zfill(2)
    else:
        out = out + ".00"
    return out
##########################################################################
##########################################################################


##########################################################################
##########################################################################
def get_Atr_Val(elem, atrmod):
    for atrN, atrV in elem.attrib.items():
        if atrmod in atrN:
            return atrV
##########################################################################
##########################################################################


##########################################################################
##########################################################################
def indent_TTML(elem, level=0):
    i = "\n" + level * "  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            indent_TTML(elem, level + 1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i
##########################################################################
##########################################################################


##########################################################################
##########################################################################
class MyProgress:
    def __init__(self, progressName="", maxVal=None, printStep=10):
        self.startTime = time.time()
        self.crtTime = None
        self.maxValue = maxVal
        self.crtValue = 0
        self.progressName = progressName
        self.crtProcentProgress = 0
        self.lastPrint = -1
        self.printStep = printStep
    ##########################################################################

    ##########################################################################
    def update_progress(self, crtV, print=True, addInfo=""):
        self.crtValue = crtV
        self.crtProcentProgress = int(crtV * 100 / self.maxValue)
        if print:
            self.print_progress(addInfo)
    ##########################################################################

    ##########################################################################
    def print_progress(self, addInfo=""):
        if (
            self.crtValue
            > int(self.maxValue * (self.lastPrint + self.printStep) / 100.0)
            or self.lastPrint < 0
        ):
            if self.lastPrint < 0:
                self.lastPrint = 0
                print(COLB + self.progressName + COLE, end="", flush=True)
                print(
                    COLB
                    + ">{}%  |  {} / {}   | Time Left: 0 / Time Remaining: ?".format(
                        self.lastPrint, self.crtValue, self.maxValue
                    )
                    + COLE,
                    flush=True,
                )
            else:
                self.lastPrint = int(100 * self.crtValue / self.maxValue)
                print(COLB + self.progressName + COLE, end="", flush=True)
                for idd in range(0, int(self.crtProcentProgress / self.printStep)):
                    print(COLB + "." + COLE, end="")

                self.crtTime = time.time()
                difTime = self.crtTime - self.startTime
                leftTime = (
                    (100 - self.crtProcentProgress) * difTime / self.crtProcentProgress
                )
                leftTimeString = seconds_TO_timestamp(leftTime, 25)

                difTimeString = print_Nice_Time_Delta(self.startTime, self.crtTime)
                if addInfo != "":
                    addInfo = " --> " + addInfo
                print(
                    COLB
                    + ">{}%  |  {} / {}   | Time Left: {} / Time Remaining: {} {}".format(
                        self.lastPrint,
                        self.crtValue,
                        self.maxValue,
                        difTimeString,
                        leftTimeString,
                        addInfo,
                    )
                    + COLE,
                    flush=True,
                )
    ##########################################################################

    ##########################################################################
    def end_process(self):
        self.lastPrint = self.lastPrint + self.printStep
        print(COLB + self.progressName + COLE, end="", flush=True)
        for idd in range(0, int(100 / self.printStep)):
            print(COLB + "." + COLE, end="")
        print(COLB + "." + COLE, end="")
        self.crtTime = time.time()
        difTimeString = print_Nice_Time_Delta(self.startTime, self.crtTime)

        print(
            COLB + ">Done   | Time Left: {} ".format(difTimeString) + COLE, flush=True
        )
##########################################################################
##########################################################################


##########################################################################
##########################################################################
def saveResultsToJsonFile(data, file):
    with open(file, "w", encoding="utf-8") as outfile:
        json.dump(data, outfile, indent=4, ensure_ascii=False)
##########################################################################
##########################################################################

##########################################################################
##########################################################################
def get_folders_in_folder(path):
    return [os.path.join(path, name) for name in os.listdir(path)
            if os.path.isdir(os.path.join(path, name))]

##########################################################################
##########################################################################

##########################################################################
##########################################################################
def loadDataFromJsonFile(file):
    with open(file, "r", encoding="utf-8") as infile:
        data = json.load(infile)
    return data
##########################################################################
##########################################################################


##########################################################################
##########################################################################
def frame_to_ms(frame_number, fps=30):
    return int((frame_number / fps) * 1000)
##########################################################################
##########################################################################












