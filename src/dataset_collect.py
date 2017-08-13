import pythoncom, pyHook
import os
import sys
import time
import datetime
from PIL import ImageGrab
import glob
ctrleft = 0
ctrright = 0
ctrup = 0
ctrdown = 0
takeshot = 1
imgs1 = []
imgs2 = []
labels = []
def OnKeyboardEvent(event):
    im = ImageGrab.grab()
    timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")
    print(event.Key)
    if event.Key == 'Q' and takeshot==1:
        global takeshot
        takeshot = 0
    elif event.Key == 'Q' and takeshot==0:
        global takeshot
        takeshot = 1
    elif takeshot==1:
        imgs1.append(im)
        #time.sleep(0.1)
        im = ImageGrab.grab()
        imgs2.append(im)
        labels.append(event.Key)
    if event.Key == 'C':
        imgs1 = []
        imgs2 = []
        labels = []
    if event.Key =='X':
        print(ctrleft,ctrright,ctrup,ctrdown)
        global imgs1,imgs2,labels
        for i in range(len(imgs1)):
            t = 0
            ctr = 0
            im1 = imgs1[i]
            im2 = imgs2[i]
            Key = labels[i]
            if Key == 'D' or Key=='Right':
                t = 1
                ctr = ctrright
                global ctrright
                ctrright += 1
            elif Key == 'A' or Key=='Left':
                t = 2
                ctr = ctrleft
                global ctrleft
                ctrleft += 1
            elif Key == 'W' or Key=='Up' or Key=='Space':
                t = 3
                ctr = ctrup
                global ctrup
                ctrup += 1
            elif Key == 'S' or Key=='Down':
                t = 4
                ctr = ctrdown
                global ctrup
                ctrup += 1
            else:
                continue
            im1.save('Data_2\\'+Key+'_screenshot'+str(ctr)+'_1.png')
            im2.save('Data_2\\'+Key+'_screenshot'+str(ctr)+'_2.png')
        imgs1=[]
        imgs2 = []
        labels=[]
    elif event.Key == 'E':
        exit()
    return True

global ctrleft,ctrright,ctrup,ctrdown
DataFolder = 'C:/Users/arna/Desktop/Data_2/'
A=glob.glob(DataFolder+'Left*')
ctrleft = len(A)/2;
A=glob.glob(DataFolder+'Right*')
ctrright = len(A)/2;
A=glob.glob(DataFolder+'Up*')
ctrup = len(A)/2;
A=glob.glob(DataFolder+'Down*')
ctrdown = len(A)/2;
# create a hook manager
hm = pyHook.HookManager()
# watch for all mouse events
hm.KeyDown = OnKeyboardEvent
# set the hook
hm.HookKeyboard()
# wait forever
pythoncom.PumpMessages()
