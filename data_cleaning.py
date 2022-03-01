from tqdm import tqdm

import numpy as np
import os



"""
The following python script transform the .skeleton files into
arrays saved in .npy files.txt in two different datasets.txt.

In the first dataset, we consider the sequence (frame by frame)
of the 25 joints in a 2D images (with a resolution of 1920 x 1080)
plus the estimated depth in meter.
The resulting arrays have a dimension equals to
nb_frames x nb_joints x nb_image_dim = nb_frames x 25 x 2.

In the second dataset, we consider the sequence of the 25 joints in
unormalized 3D images. The resulting arrays have a dimention equals
to nb_frames x nb_joints x 3.
"""



# data directory absolute path (needed to be able to launch the script from anywhere)
root_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = root_dir + "/data/"

# all the raw skeletons file names
raw_files_set = set(os.listdir(data_dir + "nturgbd60_skeletons/"))


# some files have missing skeletons
with open(data_dir + "missing_skeletons.txt", 'r') as missing_skeletons_file:
    missing_files_set = set([file_name.replace('\n','') + ".skeleton" for file_name in missing_skeletons_file.readlines()])
    missing_skeletons_file.close()


# we discard the previous files from the complete set of files 
raw_files_set = raw_files_set - missing_files_set


# it is more convenient to manipulate lists for what follows
raw_files_set = list(raw_files_set)
missing_files_set = list(missing_files_set)
nb_files = len(raw_files_set)


# creation of the final folders for the data
if "nturgbd60_skeletons_2D" not in os.listdir(data_dir):
    os.mkdir(data_dir + "nturgbd60_skeletons_2D/")
final2D_set = os.listdir(data_dir + "nturgbd60_skeletons_2D/")
if "nturgbd60_skeletons_3D" not in os.listdir(data_dir):
    os.mkdir(data_dir + "nturgbd60_skeletons_3D/")
final3D_set = os.listdir(data_dir + "nturgbd60_skeletons_3D/")


# we now process to the cleaning/transformation of the files in the filtered set
for i in tqdm(range(nb_files)):

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
                    array2D[frame_count][k] = [float(values[5]), float(values[6])]
                    array3D[frame_count][k] = [float(values[0]), float(values[1]), float(values[2])]
            j += 27 * nb_skeletons + 1
            frame_count += 1
        
        with open(data_dir + "nturgbd60_skeletons_2D/" + raw_files_set[i].replace(".skeleton", ".npy"), 'wb') as array2D_file:
                    np.save(array2D_file, array2D)
                    array2D_file.close()
        
        with open(data_dir + "nturgbd60_skeletons_3D/" + raw_files_set[i].replace(".skeleton", ".npy"), 'wb') as array3D_file:
                    np.save(array3D_file, array3D)
                    array3D_file.close()