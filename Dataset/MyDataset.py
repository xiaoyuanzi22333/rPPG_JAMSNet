import torch
from torch.utils.data import Dataset
import os
import numpy as np
import cv2

class MyDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """
    def __init__(self, data_folder, split):
        """
        :param data_folder: folder where data files are stored  # Folder where data files are stored
        :param split: split, one of 'TRAIN' or 'TEST'
        """
        self.split = split
        self.data_folder = data_folder  # ./DATASET/train/
        
        # def the data list need to use
        self.rep_0 = []
        self.rep_1 = []
        self.rep_2 = []
        self.gtppg = []
        

        # Read data files
        for subject in os.listdir(self.data_folder):
            
            subject_pth = os.path.join(self.data_folder, subject)    # ./dataset/train/subject1/
            for item in os.listdir(subject_pth):
                video_pth = os.path.join(subject_pth, item)  # ./dataset/train/subject1/video_0
                self.rep_0.append(video_pth + '/rep_0.npy')
                self.rep_1.append(video_pth + '/rep_1.npy')
                self.rep_2.append(video_pth + '/rep_2.npy')
                self.gtppg.append(video_pth + '/gtTrace.npy')

    def __len__(self):
        return len(self.gtppg)

    def __getitem__(self, idx):
        # get video address from idx
        rep0 = self.rep_0[idx]
        rep1 = self.rep_1[idx]
        rep2 = self.rep_2[idx]
        
        # transform video address to tensor
        stRep_0 = np.array(np.load(rep0))
        stRep_1 = np.array(np.load(rep1))
        stRep_2 = np.array(np.load(rep2))
        
        stRep_0 = stRep_0 / 255
        stRep_1 = stRep_1 / 255
        stRep_2 = stRep_2 / 255
        stRep_0 = torch.FloatTensor(stRep_0)
        stRep_1 = torch.FloatTensor(stRep_1)
        stRep_2 = torch.FloatTensor(stRep_2)
        
        # print(stRep_0.shape)
        # print(stRep_1.shape)
        # print(stRep_2.shape)
        
        # Read label
        gtPPGFile = self.gtppg[idx]
        gtPPG = np.load(gtPPGFile)
        gtPPG = np.array(gtPPG)
        # print(gtPPG.shape)
        
        gtPPG = gtPPG - np.mean(gtPPG)
        gtPPG = gtPPG / np.std(gtPPG)
        gtPPG = torch.FloatTensor(gtPPG)

        return stRep_0, stRep_1, stRep_2, gtPPG