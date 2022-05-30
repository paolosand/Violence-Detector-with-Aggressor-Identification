"""
violence_detection_v3-single_frames.py

- Creates predictions after analyzing single frames
"""


# Running in build folder to speed up prototyping and testing

# From Python
# It requires OpenCV installed for Python
import sys
import cv2
import os
from sys import platform
import argparse
import time
import math
import numpy as np

# Import Scikit-learn for SVM model
import sklearn as skl

# Import pandas for csv management
import pandas as pd

# Import joblib to use exported models in our code
import joblib

# helper functions

# get angles given array of keypoints
def get_angles(keypoints):
    """
        1. Extract Relevant Keypoints (use index)
            Right Side
                shoulder angle - 1, 2, 3
                elbow - 2, 3, 4
                knee - 9, 10, 11
            
            Left Side
                shoulder angle - 1, 5, 6
                elbow - 5, 6, 7
                knee - 12, 13, 14

        2. Get angles using specified formula
    """
    keypoint_sets = [[1,2,3], [2,3,4], [9,10,11], [1,5,6], [5,6,7], [12,13,14]]
    angles = []
    for person in keypoints:
        p_angles = []
        for joint_set in keypoint_sets:
            a = joint_set[0]
            b = joint_set[1]
            c = joint_set[2]

            abx = abs(person[a][0] - person[b][0])
            aby = abs(person[a][1] - person[b][1])
            cbx = abs(person[c][0] - person[b][0])
            cby = abs(person[c][1] - person[b][1])

            angle = ((math.atan2(aby, abx) - math.atan2(cby, cbx)) * 180) / math.pi     

            p_angles.append(angle)
        angles.append(p_angles)

    # if there's less than two people then fill with zeros
    if len(keypoints) < 2:
        for i in range(2-len(keypoints)):
            # fill with zeros where keypoint angle values would've gone
            empty_angles = [0.0 for i in range(len(keypoint_sets))]
            angles.append(empty_angles)

    return angles

# get angles given array of keypoints
def get_velocity(curr_keypoints, prev_keypoints):
    """
        1. Extract Relevant Keypoints (use index)
            wrists, elbows, hips, shoulders, head

        2. Get velocities using specified formula
    """
    tracked_parts = [0, 2, 3, 4, 9, 5, 6, 7, 12]
    velocities = []
    for curr_position, prev_position in zip(curr_keypoints, prev_keypoints):
        # for person detected
        p_velocity = []

        for i in range(len(tracked_parts)):
            # for part we want to track
            dx = abs(curr_position[i][0] - prev_position[i][0])
            dy = abs(curr_position[i][1] - prev_position[i][1])

            dd = math.sqrt((dx**2) + (dy**2))

            v = dd / (1/30)

            p_velocity.append(v)

        velocities.append(p_velocity)
    
    # if there's less than two people, then fill with zeros
    if len(curr_keypoints) < 2:
        for i in range(2-len(curr_keypoints)):
            # fill with zeros where keypoint velocity values would've gone
            empty_velocities = [0.0 for i in range(len(tracked_parts))]
            velocities.append(empty_velocities)

    return velocities

# check for aggressive contact between two people
def get_contact(curr_keypoints, angles):
    fists = [4, 7]
    feet = [11, 14]
    imp_angles = [1, 4] # based on index from get_angles output for elbow angle
    imp_angles2 = [2, 5] # based on index from get_angles output for knee angle
    head = 0
    torso = 8
    contact_pairs = []
    if len(curr_keypoints) < 2:
        return []

    for i, person in enumerate(curr_keypoints):
        for j, other_person in enumerate(curr_keypoints):
            if i != j:
                # since people are different, we can check if person is attacking other_person
                # check for punches
                for k, fist in enumerate(fists):
                    if person[fist][1] >= other_person[head][1] - 100 and person[fist][1] <= other_person[torso][1] + 100:
                        # fist is at the correct height to make contact with the lethal section
                        if min(abs(person[fist][0] - other_person[head][0]), abs(person[fist][0] - other_person[torso][0])) <= 50:
                            if abs(angles[i][imp_angles[k]]) <= 30:
                                # arm is extended
                                contact_pairs.append([i, j])
                                break

                # check for kicks
                for k, foot in enumerate(feet):
                    # check for kicks
                    lowest_point = max(other_person[11][1], other_person[14][1]) # get y value of foot that is on the ground

                    if person[foot][1] >= other_person[head][1] - 100 and person[foot][1] <= lowest_point + 100:
                        # foot is at the correct height to make contact with the person
                        if min(abs(person[foot][0] - other_person[head][0]), min(abs(person[foot][0] - other_person[11][0]), abs(person[foot][0] - other_person[11][0]))) <= 50:
                            if abs(angles[i][imp_angles[k]]) <= 30:
                                # arm is extended
                                contact_pairs.append([i, j])
                                break
    return contact_pairs

# show results of angles
def output_angles(img, angles):
    y0, dy = 30, 20
    for i in range(len(angles[0])):
        y = y0 + i*dy
        cv2.putText(img, "{}: {}".format(i, angles[0][i]), 
            (30,y), 
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            2)
    return img

# show results of velocities
def output_velocities(img, velocities):
    y0, dy = 200, 20
    for i in range(len(velocities[0])):
        y = y0 + i*dy
        cv2.putText(img, "{}: {}".format(i, velocities[0][i]), 
            (30,y), 
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            2)
    return img

# show results of contact_pairs
def output_contact(img, contact_pairs):
    y0, dy = 400, 20
    for i, pair in enumerate(contact_pairs):
        y = y0 + i*dy
        cv2.putText(img, "{} -> {}".format(pair[0], pair[1]), 
            (30,y), 
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            2)
        # print("{} -> {}".format(pair[0], pair[1]))
    return img

# show results of violence detection
def output_prediction(img, prediction):
    cv2.putText(img, "is violent? -> {}".format(prediction), 
        (400, 30), 
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
        2)
        # print("{} -> {}".format(pair[0], pair[1]))
    return img

# process results to feed into the model
def process_results(angles, velocities, contact_pairs):
    new_angles = angles[0] + angles[1]
    new_velocities = velocities[0] + velocities[1]
    new_contact = 1 if len(contact_pairs) > 0 else 0

    results = new_angles + new_velocities + [new_contact]
    return results

# import trained violence detector model
violence_detector = joblib.load('violence_detector.pkl')

try:
    # Import Openpose (Windows/Ubuntu/OSX)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    try:
        # Windows Import
        if platform == "win32":
            # Change these variables to point to the correct folder (Release/x64 etc.)
            sys.path.append(dir_path + '/../../python/openpose/Release');
            os.environ['PATH']  = os.environ['PATH'] + ';' + dir_path + '/../../x64/Release;' +  dir_path + '/../../bin;'
            import pyopenpose as op
        else:
            # Change these variables to point to the correct folder (Release/x64 etc.)
            sys.path.append('../../python');
            # If you run `make install` (default path is `/usr/local/python` for Ubuntu), you can also access the OpenPose/python module from there. This will install OpenPose and the python library at your desired installation path. Ensure that this is in your python path in order to use it.
            # sys.path.append('/usr/local/python')
            from openpose import pyopenpose as op
    except ImportError as e:
        print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
        raise e
        

    # Flags
    parser = argparse.ArgumentParser()

    # get images from research dataset | change default path to location of dataset
    parser.add_argument("--image_dir", default="../../../research_dataset/social_behaviour_labeled/0_act_01-1", help="Process a directory of images. Read all standard formats (jpg, png, bmp, etc.).")

    # display results
    parser.add_argument("--no_display", default=False, help="Enable to disable the visual display.")
    # parser.add_argument("--camera", default=0, help="Connect Webcam")
    args = parser.parse_known_args()

    # Custom Params (refer to include/openpose/flags.hpp for more parameters)
    params = dict()
    params["model_folder"] = "../../../models/"

    # Add others in path?
    for i in range(0, len(args[1])):
        curr_item = args[1][i]
        if i != len(args[1])-1: next_item = args[1][i+1]
        else: next_item = "1"
        if "--" in curr_item and "--" in next_item:
            key = curr_item.replace('-','')
            if key not in params:  params[key] = "1"
        elif "--" in curr_item and "--" not in next_item:
            key = curr_item.replace('-','')
            if key not in params: params[key] = next_item

    # Construct it from system arguments
    # op.init_argv(args[1])
    # oppython = op.OpenposePython()

    # Starting OpenPose
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    mode = input("0 -> webcam\n1 -> folder\nSelect mode (0 or 1): ")

    if mode == '0':
        print("\nrunning violence detector on webcam...")
        # Read frames from webcam
        vc = cv2.VideoCapture(0)
        if vc.isOpened(): # try to get the first frame
            rval, frame = vc.read()
        else:
            rval = False
        start = time.time()

        # Process and display images

        first_frame = True

        while rval:
            datum = op.Datum()
            imageToProcess = frame # get frame
            datum.cvInputData = imageToProcess
            opWrapper.emplaceAndPop(op.VectorDatum([datum]))

            if first_frame:
                curr_datum = datum
                prev_datum = datum
                first_frame = False
            else:
                curr_datum = datum

            # angle output
            angles = get_angles(curr_datum.poseKeypoints)

            # velocity output
            velocities = get_velocity(curr_datum.poseKeypoints, prev_datum.poseKeypoints)

            # contact output
            contact_pairs = get_contact(curr_datum.poseKeypoints, angles)
            
            # combine results before appending to csv
            results = process_results(angles, velocities, contact_pairs)

            # feed results to model
            print(violence_detector.predict_proba([results]))

            # print("Body keypoints: \n" + str(curr_datum.poseKeypoints))

            if not args[0].no_display:
                img = curr_datum.cvOutputData

                # uncomment to show angle output
                img = output_angles(img, angles)

                # uncomment to show velocity output
                img = output_velocities(img, velocities)

                # uncomment to show contact output
                img = output_contact(img, contact_pairs)

                img = output_prediction(img, violence_detector.predict([results]))

                cv2.imshow("OpenPose 1.7.0 - Tutorial Python API", img)
                rval, frame = vc.read() # get next frame
                key = cv2.waitKey(15)
                if key == 27: break

            prev_datum = curr_datum

    elif mode == '1':
        print("\nrunning violence detector on folder...")
        # Read frames on directory
        imagePaths = op.get_images_on_directory(args[0].image_dir);
        start = time.time()

        first_frame = True
        # Process and display images
        for imagePath in imagePaths:
            datum = op.Datum()
            imageToProcess = cv2.imread(imagePath)
            datum.cvInputData = imageToProcess
            opWrapper.emplaceAndPop(op.VectorDatum([datum]))

            if first_frame:
                curr_datum = datum
                prev_datum = datum
                first_frame = False
            else:
                curr_datum = datum

            # angle output
            angles = get_angles(curr_datum.poseKeypoints)

            # velocity output
            velocities = get_velocity(curr_datum.poseKeypoints, prev_datum.poseKeypoints)

            # contact output
            contact_pairs = get_contact(curr_datum.poseKeypoints, angles)
            
            # combine results before appending to csv
            results = process_results(angles, velocities, contact_pairs)

            # feed results to model
            print(violence_detector.predict_proba([results]))

            # print("Body keypoints: \n" + str(curr_datum.poseKeypoints))

            if not args[0].no_display:
                img = curr_datum.cvOutputData

                # uncomment to show angle output
                img = output_angles(img, angles)

                # uncomment to show velocity output
                img = output_velocities(img, velocities)

                # uncomment to show contact output
                img = output_contact(img, contact_pairs)

                img = output_prediction(img, violence_detector.predict([results]))

                cv2.imshow("OpenPose 1.7.0 - Tutorial Python API", img)
                key = cv2.waitKey(15)
                if key == 27: break
            prev_datum = curr_datum

    end = time.time()
    print("OpenPose demo successfully finished. Total time: " + str(end - start) + " seconds")
except Exception as e:
    print(e)
    sys.exit(-1)
