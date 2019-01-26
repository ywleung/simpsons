import numpy as np
import pandas as pd
import os
import cv2
import math
import urllib
import matplotlib.pyplot as plt
from model import *
from keras.models import Model, load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras import activations
from vis.utils import utils
from vis.visualization import visualize_saliency


def load_dataset(path, height, width, least_num):
    image_list, label_list = [], []
    for root, dirs, files in os.walk(path):
        temp_images, temp_labels = [], []
        for file in files:
            try:
                image_path = os.path.join(root, file)
                image = cv2.imread(image_path)
                image = cv2.resize(image, (height, width))
                temp_images.append(image)
                temp_labels.append(root.split('\\')[1])
            except:
                continue
        if len(temp_images) >= least_num:
            image_list.extend(temp_images)
            label_list.extend(temp_labels)
            
    num_classes = len(np.unique(label_list))
    
    return image_list, label_list, num_classes

def preprocess_image(image, target_height, target_width):
    height, width, depth = image.shape
            
    if width < height:
        crop_len = math.floor((height-width) / 2)
        image_new = image[crop_len:crop_len+width, :, :]
    else:
        crop_len = math.floor((width-height) / 2)
        image_new = image[:, crop_len:crop_len+height, :]
                
        # resize to target_size
    image_new = image_new / 255.
    image_new = cv2.resize(image_new, (target_height, target_width))

    return image_new

def url_to_image(url, height, width):
    req = urllib.request.urlopen(url)
    image = np.asarray(bytearray(req.read()), dtype='uint8')
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    
    return image

def plot_and_predict(url, model, class_names, height, width):
    image = url_to_image(url, height, width)
    plt.imshow(image[:, :, (2, 1, 0)])
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    image = preprocess_image(image, height, width)
    pred = model.predict(np.expand_dims(image, axis=0))[0]
    plt.xlabel('{}: {:.2%}'.format(class_names[np.argmax(pred)].replace('_', ' '), np.max(pred)))
    plt.show()
    
def plot_attention(model, class_name, image):
    layer_idx = -1
    model.layers[layer_idx].activations = activations.linear
    temp_model = utils.apply_modifications(model)
    pred_class = np.argmax(model.predict(image), axis=1)
    grads= []
    for i in range(len(image)):
        grad = visualize_saliency(temp_model, layer_idx, pred_class, image[i])
        grads.append(grad)
    
    num_cols = 4
    num_row = np.int(np.ceil(len(image) / num_cols))
    plt.figure(figsize=(2*num_col, 2*num_row))
    
    for i in range(len(image)):
        plt.subplot(num_row, num_col, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(image[i])
        plt.imshow(grads[i], alpha=0.6)
        plt.xlabel(class_name[pred_class[i]])

def get_predict_video(video_path, output_path, height, width, model, char_icon_list, class_name):
    video_capture = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = int(video_capture.get(cv2.CAP_PROP_FPS))
    video_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    out = cv2.VideoWriter(output_path, fourcc, fps, (video_width, video_height))
    
    while video_capture.isOpened():
        ret, frame = video_capture.read()
    
        if ret == True:
            frame_reshaped = cv2.resize(frame / 255., (height, width))

            y_pred = model.predict(np.array([frame_reshaped]))
            index_sorted = np.argsort(y_pred, axis= 1)[0]     
    
            top1_pred = y_pred[0][index_sorted][-1]
            top2_pred = y_pred[0][index_sorted][-2]
            top3_pred = y_pred[0][index_sorted][-3]                                     
    
            if top1_pred >= 0.7:
                character_name_top1 = class_name[index_sorted[-1]]
                character_name_top2 = class_name[index_sorted[-2]]
                character_name_top3 = class_name[index_sorted[-3]]
        
                Text1 = "%s : %.2f%%"%(character_name_top1.split('_')[0],top1_pred*100)
                Text2 = "%s : %.2f%%"%(character_name_top2.split('_')[0],top2_pred*100)
                Text3 = "%s : %.2f%%"%(character_name_top3.split('_')[0],top3_pred*100)
        
                frame[2:50, 2:50] = char_icon_list[index_sorted[-1]]
                cv2.putText(frame,Text1,(2,65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        
                frame[80:128,2:50] = char_icon_list[index_sorted[-2]]
                cv2.putText(frame,Text2,(2,143), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        
                frame[158:206,2:50] = char_icon_list[index_sorted[-3]]
                cv2.putText(frame,Text3,(2,221), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        
                out.write(frame)
        
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
        else:
            break
        
    video_capture.release()
    out.release()
    
def train_model(model, epochs, save_path, X_train, y_train, X_dev, y_dev, data_augmentation=True):
    
    batch_size = 32
    early_stopping = EarlyStopping(patience=20, verbose=1)
    model_checkpoint = ModelCheckpoint(save_path, save_best_only=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(factor=0.5, patience=5, verbose=1)
    callbacks = [early_stopping, model_checkpoint, reduce_lr]
    
    if data_augmentation:
        datagen = ImageDataGenerator(
            rescale=1/255.,
            featurewise_center=False, # Set input mean to 0 over the dataset, feature-wise.
            samplewise_center=False, # Set each sample mean to 0.
            featurewise_std_normalization=False, # Divide inputs by std of the dataset, feature-wise.
            samplewise_std_normalization=False, # Divide each input by its std.
            zca_whitening=False, # Apply ZCA whitening.
            rotation_range=15, # Degree range for random rotations.
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            vertical_flip=False)

    else:
        datagen = ImageDataGenerator(
            rescale=1/255.)
        
    history = model.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size), 
                                  steps_per_epoch=len(X_train) / batch_size,
                                  validation_data=datagen.flow(X_dev, y_dev, batch_size=batch_size),
                                  validation_steps=len(X_dev) / batch_size,
                                  epochs=epochs,
                                  callbacks=callbacks)
    
    return model, history