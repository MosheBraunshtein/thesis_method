
import numpy as np

def norm(data):
    print("norm data ...")
    for trial in data:
        feature_mean = np.mean(trial,axis=0) 
        feature_std = np.std(trial,axis=0)
        # avoid => / 0
        feature_std[feature_std==0] = 1
        trial = (trial - feature_mean) / feature_std 
