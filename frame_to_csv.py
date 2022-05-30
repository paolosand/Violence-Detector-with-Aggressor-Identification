"""
frame_to_csv.py

this code was used to create the frame_data dataset

PLEASE DONT RUN THIS UNLESS YOU NEED TO BUILD THE CSV DATASET AGAIN

- Updated to extract features from new annotated dataset into frame_data3.csv
- Updated get_contact function to detect kicks
- Updated so that people are labeled left to right

CSV file is saved as frame_data3.csv

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
                            if abs(angles[i][imp_angles2[k]]) <= 30:
                                # knee is extended
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

# get all folder paths in dataset
def get_folder_paths():
    # change default path to the location of the dataset
    default_path = "../../../research_dataset/social_behaviour_labeled-ANNOTATED/"

    all_folders = [x[0] for x in os.walk(default_path)] # get all image folders from dataset
    all_folders.pop(0) # remove first element

    return all_folders

# get label given folder
def get_label(folder_path):
    i = folder_path.split('/')
    j = i[-1].split('_')
    label = j[0]

    return label

# get exact folder name
def get_folder_name(folder_path):
    i = folder_path.split('/')
    return i[-1] # last element

# process results to save in csv
def process_results(angles, velocities, contact_pairs, folder_path, frame_number):
    new_angles = angles[0] + angles[1]
    new_velocities = velocities[0] + velocities[1]
    new_contact = 1 if len(contact_pairs) > 0 else 0
    label = get_label(folder_path)

    results = [get_folder_name(folder_path), frame_number] + new_angles + new_velocities + [new_contact, int(label)]
    return results

# each folder in the dataset is a set of videos that has to be run through openpose.
all_folders = get_folder_paths()

# create lists which contain the column names of the extracted features
angles_features = [
    "person0_rs_angle", 
    "person0_re_angle", 
    "person0_rk_angle", 
    "person0_ls_angle", 
    "person0_le_angle", 
    "person0_lk_angle", 
    "person1_rs_angle",
    "person1_re_angle",
    "person1_rk_angle",
    "person1_ls_angle",
    "person1_le_angle",
    "person1_lk_angle"
]

velocity_features = [
    "person0_h_velocity",
    "person0_rs_velocity",
    "person0_re_velocity",
    "person0_rw_velocity",
    "person0_rh_velocity",
    "person0_ls_velocity",
    "person0_le_velocity",
    "person0_lw_velocity",
    "person0_lh_velocity",
    "person1_h_velocity",
    "person1_rs_velocity",
    "person1_re_velocity",
    "person1_rw_velocity",
    "person1_rh_velocity",
    "person1_ls_velocity",
    "person1_le_velocity",
    "person1_lw_velocity",
    "person1_lh_velocity"
]

contact_features = ["has_contact"]

# combine columns and add file_name, frame_number, and label column
all_columns = ['file_name', 'frame_number'] + angles_features + velocity_features + contact_features + ["label"]

# Create dataframe for frame_data
frame_data = pd.DataFrame(columns=all_columns)

# Create csv file
frame_data.to_csv('frame_data3.csv', index= False)


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
        
    # run openpose on each image folder
    for folder_path in all_folders:
        # read current frame_data3.csv
        frame_data = pd.read_csv('frame_data3.csv')

        # Flags
        parser = argparse.ArgumentParser()

        # get images from research dataset
        parser.add_argument("--image_dir", default=folder_path, help="Process a directory of images. Read all standard formats (jpg, png, bmp, etc.).")

        # print current folder
        print("\ncurrently in {}...\n".format(folder_path))

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

        # Read frames on directory
        imagePaths = op.get_images_on_directory(args[0].image_dir);
        start = time.time()

        # Print running open pose
        print('running open pose on {}'.format(folder_path))

        first_frame = True
        frame_number = 0

        # Process and display images
        for imagePath in imagePaths:
            datum = op.Datum()
            imageToProcess = cv2.imread(imagePath)
            datum.cvInputData = imageToProcess
            opWrapper.emplaceAndPop(op.VectorDatum([datum]))

            datum.poseKeypoints = np.asarray(sorted(datum.poseKeypoints, key=lambda x: x[0][0])) # sort people from left to right

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
            results = process_results(angles, velocities, contact_pairs, folder_path, frame_number)

            # create dictionary with results and column names
            new_row = {all_columns[i]: results[i] for i in range(len(results))}

            # append new row to csv
            frame_data = frame_data.append(new_row, ignore_index= True)

            # print("Body keypoints: \n" + str(curr_datum.poseKeypoints))

            if not args[0].no_display:
                img = curr_datum.cvOutputData

                # uncomment to show angle output
                img = output_angles(img, angles)

                # uncomment to show velocity output
                img = output_velocities(img, velocities)

                # uncomment to show contact output
                img = output_contact(img, contact_pairs)

                cv2.imshow("OpenPose 1.7.0 - Tutorial Python API", img)
                key = cv2.waitKey(15)
                if key == 27: break

            prev_datum = curr_datum

            # increment frame_number
            frame_number+=1

        end = time.time()
        print("OpenPose demo successfully finished. Total time: " + str(end - start) + " seconds")
        print("exiting {}...\n".format(folder_path))

        # Save current updated frame_data to frame_data3.csv
        frame_data.to_csv('frame_data3.csv', index= False)
    print("FINISHED SUCCESSFULLY! :)\n")
except Exception as e:
    print(e)
    sys.exit(-1)
