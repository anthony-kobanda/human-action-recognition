from tqdm import tqdm

import numpy as np
import os



# data directory path (needed to be able to launch the script from anywhere)
root_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = root_dir + "/data/"


# all the skeletons file names
complete_set = set(os.listdir(data_dir + "nturgbd_skeletons_s001_to_s017/"))


# some files have missing skeletons or lead to issues
with open(data_dir + "missing_skeletons.txt", 'r') as missing_skeletons_file:
    missing_set = set([file_name.replace('\n','') + ".skeleton" for file_name in missing_skeletons_file.readlines()])
    missing_skeletons_file.close()
with open(data_dir + "issue_skeletons.txt", 'r') as issue_skeletons_file:
    issue_set = set([file_name.replace('\n','') + ".skeleton" for file_name in issue_skeletons_file.readlines()])
    issue_skeletons_file.close()


# we discard the previous files from the complete set of files 
filtered_set = complete_set - missing_set - issue_set
nb_files = len(filtered_set)

# it is more convenient to manipulate lists for what follows
complete_set = list(complete_set)
missing_set = list(missing_set)
issue_set = list(issue_set)
filtered_set = list(filtered_set)


# creation of the final folder for the data
if "nturgbd_skeletons_cleaned" not in os.listdir(data_dir):
    os.mkdir(data_dir + "nturgbd_skeletons_cleaned/")
cleaned_set = os.listdir(data_dir + "nturgbd_skeletons_cleaned/")


# we now process to the cleaning of the files in the filtered set
for i in tqdm(range(nb_files)):

    if filtered_set[i].replace(".skeleton", ".npy") not in cleaned_set:

        with open(data_dir + "nturgbd_skeletons_s001_to_s017/" + filtered_set[i], 'r') as raw_file:
            file_lines = raw_file.readlines()
            nb_frames = int(file_lines[0][:-1])
            raw_file.close()
        
        if nb_frames == (len(file_lines)-1)/28:
            
            try:

                # we only keep the spatial coordinate for each of the 25 joints at each frame 
                cleaned_file_data = np.array([[file_lines[j].split(' ')[:3] for j in range(4+28*i,1+28*(i+1))] for i in range(nb_frames)], dtype=float)

                # we save the data as a numpy array of shape: (nb_frames, nb_joints, nb_spatial_coordinates) = (nb_frames, 25, 3) 
                with open(data_dir + "nturgbd_skeletons_cleaned/" + filtered_set[i].replace(".skeleton", ".npy"), 'wb') as cleaned_file:
                    np.save(cleaned_file, cleaned_file_data)
                    cleaned_file.close()
            
            # the file leads to an issue so we add its name in the corresponding file
            except:

                if filtered_set[i].replace(".skeleton", "") not in issue_set:
                    with open(data_dir + "issue_skeletons.txt", 'a') as issue_skeletons_file:
                        line_content = filtered_set[i].replace(".skeleton", "")
                        if i>0:
                            line_content = '\n' + line_content
                        issue_skeletons_file.write(line_content)
                        issue_skeletons_file.close()
        
        # the file leads to an issue (nb_frames doesn't match the number of lines) so we add its name in the corresponding file
        else:

            if filtered_set[i].replace(".skeleton", "") not in issue_set:
                with open(data_dir + "issue_skeletons.txt", 'a') as issue_skeletons_file:
                    line_content = filtered_set[i].replace(".skeleton", "")
                    if i>0:
                        line_content = '\n' + line_content
                    issue_skeletons_file.write(line_content)
                    issue_skeletons_file.close()