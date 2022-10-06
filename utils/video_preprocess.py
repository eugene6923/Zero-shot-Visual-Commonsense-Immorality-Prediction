import cv2
import numpy as np
from skimage.transform import resize
from IPython.display import clear_output
import os
from PIL import Image
import pathlib


def read_frames2(arr):
    vs = []

    for i,file_video in enumerate(arr):
        if i%1==0:
            clear_output()
            # print(np.round(i*100/len(arr),3))
            
        iml = []
        Video_File_Path = file_video
        # print(file_video)
        # input()
        Video_Caption = cv2.VideoCapture(str(Video_File_Path))
        # print(Video_Caption)
        # input()
        Frame_Rate = Video_Caption.get(5) #프레임 넓이
        # print(Frame_Rate)
        # input()
    
        while Video_Caption.isOpened():
        
            Current_Frame_ID = Video_Caption.get(1)
        
            ret,frame = Video_Caption.read()
        
            if ret != True:
                break
            
            if Current_Frame_ID % np.floor(Frame_Rate) == 0:
                Frame_Resize = cv2.resize(frame,(64,64))
                iml.append(Frame_Resize)
            
        vs.append(iml)
        Video_Caption.release()
    
    return vs

def read_frames(arr):
    vs=[] 
    for j  in range(len(arr)):
        if j%1==0:
            clear_output()
            print(j)
        vcap=cv2.VideoCapture(arr[j])
        success=True
  
        iml=[]
        c=0
        while success:
            try:
              success,image=vcap.read()
              c+=1
              if c%5==0:
                im=resize(image,(64,64))
                iml.append(im)
            except Exception as e:
                print(e)
        vs.append(iml)
        if len(iml) < 5:
            print(len(iml))
            return None
    
    return vs


def select_frames(frames_arr , n=5,a=3): 
    videos=[]
    for i in range(len(frames_arr)):
        frames=[]
        for t in np.linspace(0, len(frames_arr[i])-1, num=n):
            
            frames.append(frames_arr[i][int(t)])     

        videos.append(frames[a])
        
    vl = np.array(videos)
    return vl


def onesec(filepath,savepath):
    save_path=os.path.join(savepath,pathlib.Path(filepath).stem)
    # print(save_path)
    video = cv2.VideoCapture(filepath) 

    if not video.isOpened():
        # exit(1)
        print(pathlib.Path(filepath).stem)

    length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video.get(cv2.CAP_PROP_FPS)

    print("length :", length)
    print("width :", width)
    print("height :", height)
    print("fps :", fps)

    count = 0

    while (video.isOpened()):
        ret, image = video.read()
        if ret == True:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            if(int(video.get(1)) % (fps) == 0) :
                if int(video.get(1))>=length:
                    break
                PIL_image = Image.fromarray(np.uint8(image))
                PIL_image.save(save_path + "/frame%d.jpg" % count)
                print('Saved frame number :', str(int(video.get(1))))
                count += 1
                
        else:
            break
    video.release()
