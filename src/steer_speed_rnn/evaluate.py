import csv
import os
import numpy as np
import cv2
import keras

from keras.models import load_model
from keras.models import model_from_json
import h5py
from keras import __version__ as keras_version

from time import sleep
from tqdm import tqdm


seq_images = []
seq_len = 10

VAL_PATH = "../../data/udacity_sim_data/"


def load_dataset(file_path):
    '''
    Loads dataset in memory
    '''
    dataset = []
    with open(file_path) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            try:
                dataset.append({'center':line[0], 'left':line[1], 'right':line[2], 'steering':float(line[3]), 
                            'throttle':float(line[4]), 'brake':float(line[5]), 'speed':float(line[6])})
            except:
                continue
    return dataset

val_dataset = load_dataset(os.path.join(VAL_PATH, "driving_log.csv"))

print("Loaded {} samples from file {}".format(len(val_dataset),VAL_PATH))


def img_for_model(data):
    img = cv2.imread(VAL_PATH + data["center"].strip())
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)[:, :, 1]
    img = np.asarray(img).reshape(160, 320, 1)
    return img

def predict_speed(model, data):
    global seq_images, seq_len
    
    pred_speed = 0.0
    gt_speed = data["speed"]
    
    image_array = img_for_model(data)
    
    if len(seq_images) < seq_len:
        seq_images.append(image_array)

    else:
        seq_images.pop(0)
        seq_images.append(image_array)

        transformed_image_array = np.array(seq_images)
        transformed_image_array = transformed_image_array[None, :, :, :, :]

        pred_speed = float(model.predict(transformed_image_array, batch_size=1))*15 + 15
    
    return pred_speed, gt_speed


# compile and load weights
model_path = "../../data/weights/speed_cnn_rnn/model.json"
with open(model_path, 'r') as jfile:
    model = model_from_json(jfile.read())

model.compile("rmsprop", "mse")
weights_file = model_path.replace('json', 'h5')
model.load_weights(weights_file)

font = cv2.FONT_HERSHEY_SIMPLEX
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter('output.avi',fourcc, 10.0, (320,160))

for i in tqdm(range(100), total=100):
    pred_speed, gt_speed = predict_speed(model, val_dataset[i])
    pred_speed = round(pred_speed, 2)
    gt_speed = round(gt_speed, 2)
    image = cv2.imread(VAL_PATH + val_dataset[i]["center"].strip())
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.putText(image,str(pred_speed), (10,130), font, 0.5,(0,255,0),1,cv2.LINE_AA)
    cv2.putText(image,str(gt_speed), (10,150), font, 0.5,(0,0,255),1,cv2.LINE_AA)
    out.write(image)
    #cv2.imshow("Front view", image)
    #if cv2.waitKey(1) & 0xFF == ord('q'):
    #    break

out.release()
#cv2.destroyAllWindows()