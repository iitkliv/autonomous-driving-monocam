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


#ToDo
Train tf version of video prediction
