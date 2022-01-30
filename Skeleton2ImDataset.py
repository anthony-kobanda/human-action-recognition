import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from PIL import Image, ImageDraw
import os
from tqdm import tqdm

os.chdir("data/")
path = os.getcwd()
# print(path)

bust_joints = [0, 1, 20, 2, 3]
arm_joints = [23, 24, 11, 10, 9, 8, 20, 4, 5, 6, 7, 22, 21]
leg_joints = [19, 18, 17, 16, 0, 12, 13, 14, 15]

body_joints = [bust_joints, arm_joints, leg_joints]

def get_skeleton_info(file):
    """
    Read the .skeleton files and process the data
    
    Example of output:
    {'numb_of_frames': 103, 
     'frame_info': [{'numb_of_skeletons': 1, 
                     'skeleton_info': [{'numb_of_joints': 25, 
                                        'joint_info': [{'x': 0.2181153, 'y': 0.1725972, 'z': 3.785547, 
                                                        'depthX': 277.419, 'depthY': 191.8218, 
                                                        'colorX': 1036.233, 'colorY': 519.1677, 
                                                        'orientationW': -0.2059419, 'orientationX': 0.05349901, 
                                                        'orientationY': 0.9692109, 'orientationZ': -0.1239193, 
                                                        'trackingState': 2.0}, 
                                                       {'x': 0.2323292, 'y': 0.4326636, 'z': 3.714767, ...} ...]
    """

    with open(file, 'r') as f:
        skeleton_sequence = {}
        skeleton_sequence['numb_of_frames'] = int(f.readline())
        skeleton_sequence['frame_info'] = []
        # typically: skeleton_sequence = {'numb_of_frames': 103, 'frame_info': []}

        for frame in range(skeleton_sequence['numb_of_frames']):
            frame_info = {}
            frame_info['numb_of_skeletons'] = int(f.readline())
            frame_info['skeleton_info'] = []
            # typically: frame_info = {'numb_of_skeletons': 1, 'skeleton_info': []}

            for m in range(frame_info['numb_of_skeletons']):
                #print(f"    \u21FE m = {m}")
                skeleton_info = {}
                skeleton_info_key = ['bodyID', 'clipedEdges', 'handLeftConfidence','handLeftState', 'handRightConfidence', 'handRightState','isResticted', 'leanX', 'leanY', 'trackingState']
                skeleton_info_info = {
                    k: float(v)
                    for k, v in zip(skeleton_info_key, f.readline().split())
                }
                skeleton_info['numb_of_joints'] = int(f.readline())
                skeleton_info['joint_info'] = []
                #print(f"    \u25AA skeleton_info = {skeleton_info}")

                for v in range(skeleton_info['numb_of_joints']):
                    joint_info_key = ['x', 'y', 'z', 'depthX', 'depthY', 'colorX', 'colorY','orientationW', 'orientationX', 'orientationY','orientationZ', 'trackingState']
                    joint_info = {
                        k: float(v)
                        for k, v in zip(joint_info_key, f.readline().split())
                    }
                    skeleton_info['joint_info'].append(joint_info)
                frame_info['skeleton_info'].append(skeleton_info)
                #print(f"skeleton_info = {skeleton_info}")

            skeleton_sequence['frame_info'].append(frame_info)
            #print(f"skeleton_sequence = {skeleton_sequence}")
    return skeleton_sequence

def get_pixels(skeleton):
    data = np.zeros((skeleton['numb_of_frames'], 25, 3))  # (3,frame_nums,25 2)
    for n, f in enumerate(skeleton['frame_info']):
        for m, b in enumerate(f['skeleton_info']):
            for j, v in enumerate(b['joint_info']):
                if j < 25:
                    data[n, j] = [int(v['colorX']), int(v['colorY']), v['z']]
                else:
                    pass
    #print(f"\u25A0 data = {data}\ndata.shape = {data.shape}")
#     np.save("./data_to_get_rid_of", data)
    return data


def normalize_z(skeleton):
    # get pixels but also normalize the depth between 0 and 1
    sklt_pixels = get_pixels(skeleton)
    mini = sklt_pixels[:,:,2].min()
    maxi = sklt_pixels[:,:,2].max()
#     print(maxi)
    sklt_pixels[:,:,2] = (sklt_pixels[:,:,2]-mini)/(maxi-mini)
    return sklt_pixels

# all the skeletons file names
complete_set = set(os.listdir("nturgbd_skeletons_s001_to_s017/"))

# some files have missing skeletons or lead to issues
with open("missing_skeletons.txt", 'r') as missing_skeletons_file:
    missing_set = set([file_name.replace('\n','') + ".skeleton" for file_name in missing_skeletons_file.readlines()])
    missing_skeletons_file.close()
with open("issue_skeletons.txt", 'r') as issue_skeletons_file:
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
if "nturgbd_pixel+depth" not in os.listdir():
    os.mkdir("nturgbd_pixel+depth/")
cleaned_set = os.listdir("nturgbd_pixel+depth/")

# we now process to the cleaning of the files in the filtered set
for i in tqdm(range(nb_files)):

    if filtered_set[i].replace(".skeleton", ".npy") not in cleaned_set:
        skeleton = get_skeleton_info("nturgbd_skeletons_s001_to_s017/"+filtered_set[i])
#         sklt_pixels = get_pixels(skeleton)
        try:
            sklt_pixels = normalize_z(skeleton)
            with open("nturgbd_pixel+depth/" + filtered_set[i].replace(".skeleton", ".npy"), 'wb') as cleaned_file:
                np.save(cleaned_file, sklt_pixels)
                cleaned_file.close()
        except:
            continue