import copy
import cv2
import numpy as np
import pickle as pkl
import os

from torch.utils.data import Dataset

import torch
import torch.nn as nn



# setting device on GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device: {}".format(device))

mode = 0



####################
### DATA LOADING ###
####################

print("\nloading data...")

# data directory absolute path (needed to be able to launch the script from anywhere)
root_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = root_dir + "/data/"

data2D_dir = data_dir + "nturgbd60_skeletons_2D/"
data3D_dir = data_dir + "nturgbd60_skeletons_3D/"

data2D_files = os.listdir(data2D_dir)
data3D_files = os.listdir(data3D_dir)


with open("./data/actions.txt", 'r') as actions_file:
    actions = [line.replace('\n', '') for line in actions_file.readlines()]
    actions_file.close()

classes = [5, 6, 7, 8, 14, 24, 30, 32, 42]
# class 0  (6) : pickup
# class 1  (7) : throw
# class 2  (8) : sitting down
# class 3  (9) : standing up (from sitting position)
# class 4 (15) : take off jacket
# class 5 (25) : reach into pocket
# class 6 (31) : pointing to something with finger
# class 7 (33) : check time (from watch)
# class 8 (43) : falling

bust_joints = [0, 1, 20, 2, 3]
arm_joints = [23, 24, 11, 10, 9, 8, 20, 4, 5, 6, 7, 22, 21]
leg_joints = [19, 18, 17, 16, 0, 12, 13, 14, 15]
body_parts = [bust_joints, arm_joints, leg_joints]


class HumanActionDataset2D(Dataset):

    def __init__(self, data_dir, data_files, classes):
        self.data_dir = data_dir
        self.data_files = [data_file for data_file in data_files if int(data_file[17:-4])-1 in classes]
        self.classes = classes

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        tensor = torch.Tensor(np.load(self.data_dir + self.data_files[idx]))
        tensor = tensor.reshape((tensor.shape[0], 50))
        label = self.classes.index(int(self.data_files[idx][17:-4])-1)
        return (tensor, label)

class HumanActionDataset3D(Dataset):

    def __init__(self, data_dir, data_files, classes, with_depth=True):
        self.data_dir = data_dir
        self.data_files = [data_file for data_file in data_files if int(data_file[17:-4])-1 in classes]
        self.classes = classes
        self.with_depth = with_depth

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        tensor = torch.Tensor(np.load(self.data_dir + self.data_files[idx]))
        if self.with_depth:
            tensor = tensor.reshape((tensor.shape[0], 75))
        else:
            tensor = torch.tensor([[tensor[i,k//2,k%2] for k in range(50)] for i in range(tensor.shape[0])])
        label = self.classes.index(int(self.data_files[idx][17:-4])-1)
        return (tensor, label)

HAD2D = HumanActionDataset2D(data2D_dir, data2D_files, classes)
HAD3D = HumanActionDataset3D(data3D_dir, data3D_files, classes)

modelX = pkl.load(open("./models_saved/regX.sav", "rb"))
modelY = pkl.load(open("./models_saved/regY.sav", "rb"))

def tensor2D23D(tensor2D,z):
    X2D = np.array([tensor2D[2*k] for k in range(25)]).reshape(-1,1)
    Y2D = np.array([tensor2D[2*k+1] for k in range(25)]).reshape(-1,1)
    X3D = modelX.predict(X2D).reshape(25)
    Y3D = modelY.predict(Y2D).reshape(25)
    Z3D = np.array([z for k in range(25)]).reshape(25) # depthmap
    return torch.tensor([(float(X3D[k]), float(Y3D[k]), float(Z3D[k])) for k in range(25)]).reshape((1,1,75))


#####################
### MODEL LOADING ###
#####################

print("loading model...")

sm = nn.Softmax(dim=1).to(device)
h_n, c_n = None, None

class LSTM03D(nn.Module):

    def __init__(self, nb_classes, input_size, hidden_size_lstm, hidden_size_classifier, num_layers, device):

        super(LSTM03D, self).__init__()

        self.num_classes = nb_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size_lstm
        self.device = device

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size_lstm, num_layers=num_layers, batch_first=True) # lstm
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size_lstm, hidden_size_classifier),
            nn.ReLU(),
            nn.Linear(hidden_size_classifier, nb_classes)
        )

    def forward(self,x,h_0=None,c_0=None):
        if h_0 is None:
            h_0 = torch.rand(self.num_layers, x.size(0), self.hidden_size).to(self.device) # hidden state (short memory)
            c_0 = torch.rand(self.num_layers, x.size(0), self.hidden_size).to(self.device) # internal state (long memory)
        _, (h_n, c_n) = self.lstm(x, (h_0, c_0))
        h_n_reshape = h_n[-1].reshape(1, h_n.shape[1], h_n.shape[2]).view(-1, self.hidden_size)
        results = self.classifier(h_n_reshape) # reshaping the data for clasifier
        return results, h_n, c_n

model_LSTM3DF = LSTM03D(nb_classes=len(classes), input_size=75, hidden_size_lstm=256, hidden_size_classifier=128, num_layers=1, device=device)
model_LSTM3DF.to(device)
model_LSTM3DF.load_state_dict(torch.load("./models_saved/LSTM3DF_{}.pt".format(mode)))
model_LSTM3DF.eval()



##################
### PREDICTION ###
##################

print("tracking ...")


# we will put random sequence of actions one after the other and see how the prediction evolves

state_list = []

current_sequence_index = np.random.randint(low=0, high=len(HAD3D))
current_sequence_2D,label_2D = HAD2D[current_sequence_index]
current_sequence_3D,label_3D = HAD3D[current_sequence_index]
assert label_2D == label_3D

acc0, acc1, curr, delay, found, n, N1, N2 = 0, 0, 1, 0, False, 0, 1, 0

frame_base = np.ones((1080//2,1920//2)) * 125

cv2.imshow("Demo from 3D data", frame_base)

while cv2.getWindowProperty("Demo from 3D data", 0) >= 0:

    frame_base = np.ones((1080//2,1920//2,3)) * 0.75

    frame_base = cv2.putText(
        img=frame_base,
        text="true label :",
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale = 0.5,
        org=(10,25), color=(0,0,0),
        thickness=2)
    
    frame_base = cv2.putText(
        img=frame_base,
        text="{}".format(actions[classes[label_2D]]),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale = 0.5,
        org=(105,25), color=(0,0,0),
        thickness=2)
    
    if mode == 0:
        t = tensor2D23D(current_sequence_2D[0].reshape(50),z=3).reshape(1,75).to(device)
        state_list.append(t)
        state_list = state_list[-100:]
        state = torch.reshape(torch.concat(state_list), (1, len(state_list), 75))
        h_n, c_n = None, None
        output, h_n, c_n = model_LSTM3DF(state, h_n, c_n)
    else:
        t = tensor2D23D(current_sequence_2D[0].reshape(50),z=3).to(device)
        output, h_n, c_n = model_LSTM3DF(t, h_n, c_n)
    h_n, c_n = copy.copy(h_n).to(device), copy.copy(c_n).to(device)
    probs = sm(output).reshape(len(classes)).detach().cpu().numpy()

    if np.argmax(probs) == label_2D:
        acc0 += 1
        if not found:
            found = True
            acc1 += 1
            delay += curr
            N2 += 1
    n += 1
    curr += 1

    v1, v2, v3 = acc0/n, acc1/N1, delay/max(1,N2)

    frame_base = cv2.putText(
        img=frame_base,
        text="Acc (abs) :",
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale = 0.5,
        org=(10,290), color=(0,0,0),
        thickness=2)
    
    frame_base = cv2.putText(
        img=frame_base,
        text="Acc (rel)  :",
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale = 0.5,
        org=(10,310), color=(0,0,0),
        thickness=2)
    
    frame_base = cv2.putText(
        img=frame_base,
        text="Delay (abs) : ",
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale = 0.5,
        org=(10,330), color=(0,0,0),
        thickness=2)

    frame_base = cv2.putText(
        img=frame_base,
        text="{}".format(np.round(100*v1, 2)),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale = 0.5,
        org=(120,290), color=(0,v1,1-v1),
        thickness=2)
    
    frame_base = cv2.putText(
        img=frame_base,
        text="{}".format(np.round(100*v2, 2)),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale = 0.5,
        org=(120,310), color=(0,v2,1-v2),
        thickness=2)
    
    frame_base = cv2.putText(
        img=frame_base,
        text="{}".format(np.round(v3, 2)),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale = 0.5,
        org=(120,330), color=(0,1-v3/100,v3/100),
        thickness=2)

    for i in range(len(classes)):
        
        p = float(probs[i].round(2))
        color = (0,0,255)
        if i == label_2D:
            color = (0,255,0)
        frame_base = cv2.rectangle(img=frame_base, pt1=(5,38+i*25), pt2=(int(5+p*245),63+i*25), color=color, thickness=-1)
        frame_base = cv2.rectangle(img=frame_base, pt1=(5,38+i*25), pt2=(250,63+i*25), color=(0,0,0), thickness=2)

        frame_base = cv2.putText(
            img=frame_base,
            text="{}".format(actions[classes[i]]),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale = 0.33,
            org=(10,55+i*25), color=(0,0,0),
            thickness=1)


    if True:

        for i,body_part in enumerate(body_parts):

            for j in range(len(body_part)-1):

                xa = current_sequence_2D[0,2*body_part[j]].item()
                ya = current_sequence_2D[0,2*body_part[j]+1].item()
                za = current_sequence_3D[0,3*body_part[j]+2].item()
                xb = current_sequence_2D[0,2*body_part[j+1]].item()
                yb = current_sequence_2D[0,2*body_part[j+1]+1].item()
                zb = current_sequence_3D[0,3*body_part[j+1]+2].item()

                if np.isnan(xa) or np.isnan(ya) or np.isnan(za) or np.isnan(xb) or np.isnan(yb) or np.isnan(zb):
                    pass
                else:
                    xa = int(np.clip(xa, 1, 1920)//2-1)
                    ya = int(np.clip(ya, 1, 1080)//2-1)
                    xb = int(np.clip(xb, 1, 1920)//2-1)
                    yb = int(np.clip(yb, 1, 1080)//2-1)
                    frame_base = cv2.line(img=frame_base, pt1=(xa,ya), pt2=(xb,yb), color=(255,0,0), thickness=3)
                    if body_part[j+1] == 3:
                        frame_base = cv2.circle(img=frame_base, center=(xb,yb), radius=15, color=(255,0,0), thickness=-1)

    cv2.imshow("Demo from 3D data", frame_base)

    current_sequence_2D = current_sequence_2D[1:,:]
    current_sequence_3D = current_sequence_3D[1:,:]

    # end of the current sequence => we randomly select an other one
    if current_sequence_2D.shape[0] == 0:
        current_sequence_index = np.random.randint(low=0, high=len(HAD3D))
        current_sequence_2D,label_2D = HAD2D[current_sequence_index]
        current_sequence_3D,label_3D = HAD3D[current_sequence_index]
        found = False
        curr = 1
        N1 += 1
        assert label_2D == label_3D
    
    c = cv2.waitKey(50)
    if c == 27:
        break

    

print("done")