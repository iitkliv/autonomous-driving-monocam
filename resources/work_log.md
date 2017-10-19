Read papers related to GAN

Read rahul's thesis

Read papers related to future frame prediction.
- deep multi-scale
- comma ai

Explored different autonomous driving techniques presently available
- Nvidia
- Comma ai
- Apple S + U
- lane detection - bsnake, caltech, CHEVP

Saw various implementations on udacity simulator.

Downloaded comma ai dataset
Installed Torcs

Udacity simulator running on  local and agv server
Installed Anaconda
Created udacity specific environment from .yaml
downloaded official udacity sim dataset

[25th aug]Written keras code for training steer only model (output 1 variable)
The model trained well (validation loss drops with training loss), and also makes the car run on the track.
The car doesn't run well on another track. So need to make model to better generalize on other tracks too.
However, the inference code gives a constant throttle. So need to design constant velocity controller

Written function for PI controller in inference code.
Car moves with constant speed on the track. Used previous trained model.

[26th Aug] Trained network for steer and speed (output 2 variable), while keeping the model same
The validation loss continuously increases even when the training loss drops. So need to change model or rectify data
Still tried to infer the model. Car looses off the track quite early. Moreover, the speed doesn't vary as it should.


[25th Sept] Steer variable is only trained. Rnn part was trained after extracting feature vector of size 64 from each image, and later was cascaded with CNN to train RCNN. The training curves were good, but testing on simulator is not performning very well. I can see it gives negative values even when its turning too left. Also, its not able to better recover from when completely deviated. So I think the training data should have more samples of how to recover and also turning positive turns. Also, its not able to do anything on the second track. That is, its not able to generalize.

I have trained rnn part of steer speed. I don't know how much loss to expect, but the loss is decreasing. One doubt I have to clear is given the output variable range, what loss to expect to judge that loss is close.


[10th Oct] I used 43 param network on my dataset, and it works well. This imples some problem with my model. Moreover, there is no augmentation for increasing non-zero steering value images in 43 param network. This network was able to generalize to new track even when no images from new track were used for training. My model needs to first generalize like this.



#ToDo
Apply rnn to working 160*320 model of steering only
Stick to it till it runs well, on rnn. This will be used to prove that it performs better
Make use of generator. Make working models run with same accuracy using generator data.
 
1. Clean training data. negative values in between many positive values should be truncated. Use some kernal, to smoothen the variable.
done2. Add training data pertaining to recovery of vehicle when deviated too much. 
done3. Train CNN RNN of steer only till best results are obtained
4. Cant train for speed, use acceleration between images for training. That is usefull. Get acceleration only after doing 1, 2 above. 
5. [Optional] Separate tracks dataset, because there might be training sequences involving transisiton between tracks. 

