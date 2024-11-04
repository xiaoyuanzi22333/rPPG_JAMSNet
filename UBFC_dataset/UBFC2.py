import numpy as np
import pandas as pd
import os
import cv2
import argparse


class UBFC2_data():
    
    def __init__(self,path:str):
        self.folder_path = path
        self.video_path = path + '/vid.avi'
        self.gt_path = path + '/ground_truth.txt'

    def getVideo(self):
        return self.video_path
    
    def getGT(self):
        with open(self.gt_path) as file:
            gtTrace = list(np.float_(file.readline().split()))
            gtHR = list(np.float_(file.readline().split()))
            gtTime = list(np.float_(file.readline().split()))
            
        return gtTrace, gtHR



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-num", "--number", type=int)
    args = parser.parse_args()    

    path_test = './DATASET_2/subject' + str(args.number)
    data = UBFC2_data(path_test)
    
    videoPath = data.getVideo()
    gtTrace, gtHR = data.getGT()
    
    videoCapture=cv2.VideoCapture(videoPath)
    
    fps = videoCapture.get(cv2.CAP_PROP_FPS)
    frames = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)
    print("fps: "+str(fps))
    print("total: "+str(frames))
    
    print(len(gtTrace))
    print(len(gtHR))
    
    
        