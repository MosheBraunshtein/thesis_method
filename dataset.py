import numpy as np
import os
from config import label_to_number
from pathlib import Path
from typing import List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
JIGSAWS = PROJECT_ROOT / "thesis_ML_bio" / "datasets" / "JIGSAWS"

class Dataset:
    def __init__(self) -> None:
        self.data = []
        self.target = []
        self.index = 0
        self.total_timestamps = 0

    def __len__(self):
        # number of trials
        return len(self.target)

    
    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index < len(self.target):
            data_val, targets_val = self.data[self.index], self.target[self.index]
            self.index += 1
            return data_val, targets_val
        else:
            raise StopIteration
        
    def types(self): 
        print(f"data type:{type(self.data[0].dtype)} \nlabels type: {type(self.target[0].dtype)}")

    def add_rep(self,lines,kinematics):
        trial_start_t =  int(lines[0].split()[0])
        trial_end_t =  int(lines[-1].split()[1]) 

        kinematics = kinematics[trial_start_t:trial_end_t+1,:]

        labels = np.zeros(kinematics.shape[0],dtype=int)
        self.total_timestamps += kinematics.shape[0]

        for line in lines:
            s_t, e_t, label = line.strip().split()
            s_t = int(s_t)-trial_start_t
            e_t = (int(e_t)+1)-trial_start_t 
            labels[s_t:e_t] = label_to_number(label)
        
        self.target.append(labels)
        self.data.append(kinematics)

def dataset(task,usrOut=None) -> Tuple[List[np.ndarray], List[np.ndarray]] :
    train = Dataset()
    test = Dataset()

    print(f"Extract {task} dataset ...\n")
    transcriptions_path = JIGSAWS / task / "transcriptions"
    kinematics_path = JIGSAWS / task / "kinematics"
    
    for rep in os.listdir(transcriptions_path):
        lines = None
        kinematics = None
        print(rep)
        path = transcriptions_path / rep
        with open(path, 'r') as file:
            lines = file.readlines()

        path = kinematics_path / rep
        kinematics = np.loadtxt(path)

        if usrOut is None:
            test.add_rep(lines,kinematics)
        else:
            if usrOut in rep:
                test.add_rep(lines,kinematics)
            else: 
                train.add_rep(lines,kinematics)
    
    return train,test


    

