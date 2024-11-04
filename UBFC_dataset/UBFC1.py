import numpy as np
import pandas as pd
import os
import cv2
import argparse

class UBFC1_data():
    
    def __init__(self,path):
        self.video_path = path + '/vid.avi'
        self.gt_path = path + '/gtdump.xmp'
        
    def getVideo(self):
        return self.video_path
    
    # Trace =>Blood Volume Pressure
    # HR => Heart Rate
    # Time => ms
    def getGT(self):        
        pdFrame = pd.read_csv(self.gt_path, header = None)
        gtTrace = pdFrame[3].values.tolist()
        gtTime = pdFrame[0].values.tolist()
        gtHR = pdFrame[1].values.tolist()
        
        return gtTime, gtTrace, gtHR
        
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-num", type=int)
    args = parser.parse_args()
    
    data = UBFC1_data('./DATASET_1/' + str(args.num) + '-gt')
    videoPath = data.getVideo()
    time, trace, hr = data.getGT()
    
    videoCapture=cv2.VideoCapture(videoPath)
    print(videoPath)

    fps = videoCapture.get(cv2.CAP_PROP_FPS)

    frames = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)
    print("fps: "+str(fps))
    print("total: " + str(frames))
    
    print(len(time))
    print(len(trace))
    # print(hr)