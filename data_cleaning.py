from tqdm import tqdm

import numpy as np
import os

"""
This python script transform the .skeleton files into
numpy arrays saved in .npy files in two different datasets.

In the first dataset, we consider the sequence (of frames) of
the 25 joints in a 2D images (with a resolution of 1920 x 1080)
The resulting arrays in this dataset have a dimension equals to
nb_frames x nb_joints x nb_image_dim = nb_frames x 25 x 2.

In the second dataset, we consider the sequence of the 25 joints in
unormalized 3D images. The resulting arrays in this dataset have a
dimention equals to nb_frames x nb_joints x 3.

For a very few number of elements in both datasets some values
are set to None and will be replace by -100 when encountered.

The datasets of .skeleton files is available here
(to download and to unzip the /data folder of this project):
https://drive.google.com/file/d/1U1veKcEC2B5Xn_o3StN3U8qNFHRhxqLu/view?usp=sharing
"""

# data directory absolute path (useful to be able to launch the script from anywhere)
root_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = root_dir + "/data/"

# set of all the raw skeletons file names
raw_files_set = set(os.listdir(data_dir + "nturgbd60_skeletons/"))

# set of unwanted files (they have missing skeletons or frames)
with open(data_dir + "missing_skeletons.txt", 'r') as missing_skeletons_file:
    missing_files_set = set([file_name.replace('\n','') + ".skeleton" for file_name in missing_skeletons_file.readlines()])
    missing_skeletons_file.close()

# we discard the unwanted files from the complete set of files 
raw_files_set = raw_files_set - missing_files_set

# wz now turn our sets into lists as it is more convenient to manipulate
raw_files_set = list(raw_files_set)
missing_files_set = list(missing_files_set)
nb_files = len(raw_files_set)

# creation of the final folders for the datasets
if "nturgbd60_skeletons_2D" not in os.listdir(data_dir):
    os.mkdir(data_dir + "nturgbd60_skeletons_2D/")
final2D_set = os.listdir(data_dir + "nturgbd60_skeletons_2D/")
if "nturgbd60_skeletons_3D" not in os.listdir(data_dir):
    os.mkdir(data_dir + "nturgbd60_skeletons_3D/")
final3D_set = os.listdir(data_dir + "nturgbd60_skeletons_3D/")

# we now process to the transformation of the .skeleton files into .npy files
for i in tqdm(range(nb_files)):

    # verifying this condition is interesting when the transformation
    # process has been set once and restarted for some reasons
    if raw_files_set[i].replace(".skeleton", ".npy") not in final3D_set:

        with open(data_dir + "nturgbd60_skeletons/" + raw_files_set[i], 'r') as raw_file:
            file_lines = raw_file.readlines()
            nb_frames = int(file_lines[0][:-1])
            raw_file.close()
        
        array2D = - np.zeros((nb_frames,25,2))
        array3D = - np.zeros((nb_frames,25,3))
        
        j = 1
        frame_count = 0
        
        while j < len(file_lines):
        
            nb_skeletons = int(file_lines[j][:-1])
            if nb_skeletons > 0:
                for k in range(25):
                    values = file_lines[j+k+3].split(" ")
                    x2D, y2D, x3D, y3D, z3D = float(values[5]), float(values[6]), float(values[0]), float(values[1]), float(values[2])
                    if x2D is None or np.isnan(x2D): x2D, y2D = -100, -100
                    if y2D is None or np.isnan(y2D): x2D, y2D = -100, -100
                    if x3D is None or np.isnan(x3D): x3D, y3D, z3D = -100, -100, -100
                    if y3D is None or np.isnan(y3D): x3D, y3D, z3D = -100, -100, -100
                    if z3D is None or np.isnan(z3D): x3D, y3D, z3D = -100, -100, -100
                    array2D[frame_count][k] = [float(x2D), float(y2D)]
                    array3D[frame_count][k] = [float(x3D), float(y3D), float(z3D)]
            j += 27 * nb_skeletons + 1
            frame_count += 1
        
        with open(data_dir + "nturgbd60_skeletons_2D/" + raw_files_set[i].replace(".skeleton", ".npy"), 'wb') as array2D_file:
                    np.save(array2D_file, array2D)
                    array2D_file.close()
        
        with open(data_dir + "nturgbd60_skeletons_3D/" + raw_files_set[i].replace(".skeleton", ".npy"), 'wb') as array3D_file:
                    np.save(array3D_file, array3D)
                    array3D_file.close()