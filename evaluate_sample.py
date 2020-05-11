'''
Loads pretrained model of I3d Inception architecture for the paper: 'https://arxiv.org/abs/1705.07750'
Evaluates a RGB and Flow sample similar to the paper's github repo: 'https://github.com/deepmind/kinetics-i3d'
'''

import numpy as np
import argparse
import tensorflow as tf
from tensorflow.python.keras.models import Model
from i3d_inception import Inception_Inflated3d

from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.layers import Conv3DTranspose,Conv3D,Input
from tensorflow.keras.backend import concatenate
from tensorflow.keras.utils import plot_model
import numpy as np

NUM_FRAMES = 10
FRAME_HEIGHT = 224
FRAME_WIDTH = 224
NUM_RGB_CHANNELS = 3
NUM_FLOW_CHANNELS = 2

NUM_CLASSES = 400

SAMPLE_DATA_PATH = {
    'rgb' : 'data/v_CricketShot_g04_c01_rgb.gif',
    'flow' : 'data/v_CricketShot_g04_c01_flow.gif'
}

LABEL_MAP_PATH = 'data/label_map.txt'

def main(args):
    
    model = ResNet50(weights='imagenet')
    model_res = Model(inputs=model.input, outputs=[model.get_layer('conv1_relu').output,model.get_layer('conv2_block1_out').output,model.get_layer('conv3_block1_out').output])
    
    
    # load the kinetics classes
    kinetics_classes = [x.strip() for x in open(LABEL_MAP_PATH, 'r')]


    if args.eval_type in ['rgb', 'joint']:
        if args.no_imagenet_pretrained:
            # build model for RGB data
            # and load pretrained weights (trained on kinetics dataset only) 
            rgb_model = Inception_Inflated3d(
                include_top=True,
                weights='rgb_kinetics_only',
                input_shape=(NUM_FRAMES, FRAME_HEIGHT, FRAME_WIDTH, NUM_RGB_CHANNELS),
                classes=NUM_CLASSES)
        else:
            # build model for RGB data
            # and load pretrained weights (trained on imagenet and kinetics dataset)
            rgb_model = Inception_Inflated3d(
                include_top=True,
                weights='rgb_imagenet_and_kinetics',
                input_shape=(NUM_FRAMES, FRAME_HEIGHT, FRAME_WIDTH, NUM_RGB_CHANNELS),
                classes=NUM_CLASSES)
        model_rgb=Model(inputs=rgb_model.input, outputs=rgb_model.get_layer('Conv3d_3c_3b_1x1').output)
#         print(model_rgb.summary())  
        # load RGB sample (just one example)
#         rgb_sample = np.load(SAMPLE_DATA_PATH['rgb'])
        
        # make prediction
#         rgb_logits = rgb_model.predict(rgb_sample)


    if args.eval_type in ['flow', 'joint']:
        if args.no_imagenet_pretrained:
            # build model for optical flow data
            # and load pretrained weights (trained on kinetics dataset only)
            flow_model = Inception_Inflated3d(
                include_top=True,
                weights='flow_kinetics_only',
                input_shape=(NUM_FRAMES, FRAME_HEIGHT, FRAME_WIDTH, NUM_FLOW_CHANNELS),
                classes=NUM_CLASSES)
        else:
            # build model for optical flow data
            # and load pretrained weights (trained on imagenet and kinetics dataset)
            flow_model = Inception_Inflated3d(
                include_top=True,
                weights='flow_imagenet_and_kinetics',
                input_shape=(NUM_FRAMES, FRAME_HEIGHT, FRAME_WIDTH, NUM_FLOW_CHANNELS),
                classes=NUM_CLASSES)
            
        # load flow sample (just one example)
#         flow_sample = np.load(SAMPLE_DATA_PATH['flow'])
        
        # make prediction
#         flow_logits = flow_model.predict(flow_sample)
        model_flow=Model(inputs=flow_model.input, outputs=flow_model.get_layer('Conv3d_3c_3b_1x1').output)
#         print(model_flow.summary())  
    # produce final model logits
#     if args.eval_type == 'rgb':
#         sample_logits = rgb_logits
#     elif args.eval_type == 'flow':
#         sample_logits = flow_logits
#     else: # joint
#         sample_logits = rgb_logits + flow_logits

#     # produce softmax output from model logit for class probabilities
#     sample_logits = sample_logits[0] # we are dealing with just one example
#     sample_predictions = np.exp(sample_logits) / np.sum(np.exp(sample_logits))

#     sorted_indices = np.argsort(sample_predictions)[::-1]

#     print('\nNorm of logits: %f' % np.linalg.norm(sample_logits))
#     print('\nTop classes and probabilities')
#     for index in sorted_indices[:20]:
#         print(sample_predictions[index], sample_logits[index], kinetics_classes[index])
    import STLSTM
    NUM_CELL = 1
    FILTERS0 = 128
    FILTERS1 = 64
    FILTERS2 = 64
    FILTERS3 = 64
    KERNEL_SIZE = 3
#     for layer in model_rgb.layers:
#         layer.trainable = False
#     for layer in model_res.layers:
#         layer.trainable=False
    #Model 1
#     m1,m2,m3=model_res(input3)
    #MODEL 2
#     rgb=model_rgb(input1)
#     flow=model_flow(input2)x_rgb=Input(shape=(10,224,224,3))

    cells0 = STLSTM.StackedSTLSTMCells([STLSTM.STLSTMCell(filters=FILTERS0, kernel_size=KERNEL_SIZE,padding="same",data_format="channels_last") for i in range(NUM_CELL)])
    cells1 = STLSTM.StackedSTLSTMCells([STLSTM.STLSTMCell(filters=FILTERS1, kernel_size=KERNEL_SIZE,padding="same",data_format="channels_last") for i in range(NUM_CELL)])
    cells2 = STLSTM.StackedSTLSTMCells([STLSTM.STLSTMCell(filters=FILTERS2, kernel_size=KERNEL_SIZE,padding="same",data_format="channels_last") for i in range(NUM_CELL)])
    cells3 = STLSTM.StackedSTLSTMCells([STLSTM.STLSTMCell(filters=FILTERS3, kernel_size=KERNEL_SIZE,padding="same",data_format="channels_last") for i in range(NUM_CELL)])
    
    
    x_rgb=Input(shape=(10,224,224,3))
#     x_flow=Input(shape=(10,224,224,2))
    x=model_rgb(x_rgb)
#     x_flow1=model_flow(x_flow)
    l1=[]
    l2=[]
    l3=[]
    for i in range(10):
        [m1,m2,m3]=model_res(x_rgb[:,i,:,:,:])
        l1.append(m1)
        l2.append(m2)
        l3.append(m3)
#     [merge1,merge2,merge3]=model_res(x_res)
    skip_conn1=tf.stack(l1,axis=1)
    skip_conn2=tf.stack(l2,axis=1)
    skip_conn3=tf.stack(l3,axis=1)
    print(skip_conn1.shape)
    print(skip_conn2.shape)
    print(skip_conn3.shape)
    x=STLSTM.STLSTM2D(cells0, return_sequences=True)(x)
    
    x=STLSTM.STLSTM2D(cells1, return_sequences=True)(x)
    x=STLSTM.STLSTM2D(cells2, return_sequences=True)(x)
    x=STLSTM.STLSTM2D(cells3, return_sequences=True)(x)
    x=Conv3DTranspose(64,(3,3,3),strides=(2, 1, 1), output_padding=(1,0,0),padding='valid', data_format="channels_last")(x)
    x=Conv3D(64,(3,3,3),strides=(1, 1, 1), padding='valid',data_format="channels_last")(x)
    x=tf.concat([x,skip_conn3],axis=4)
    print(x.shape)
    x=Conv3DTranspose(64,(3,3,3),strides=(1, 2, 2),output_padding=(0,1,1),padding='valid', data_format="channels_last")(x)
    x=Conv3D(64,(3,3,3),strides=(1, 1, 1), padding='valid',data_format="channels_last")(x)
    print(x.shape)
    x=tf.concat([x,skip_conn2],axis=4)
    x=Conv3DTranspose(64,(3,3,3),strides=(1, 2, 2),output_padding=(0,1,1),padding='valid', data_format="channels_last")(x)
    x=Conv3D(64,(3,3,3),strides=(1, 1, 1), padding='valid',data_format="channels_last")(x)
    print(x.shape)
    x=tf.concat([x,skip_conn1],axis=4)
    x=Conv3DTranspose(64,(3,3,3),strides=(1, 2, 2),output_padding=(0,1,1),padding='valid', data_format="channels_last")(x)
    x=Conv3D(64,(3,3,3),strides=(1, 1, 1), padding='valid',data_format="channels_last")(x)
    print(x.shape)
    x=Conv3D(3,(3,3,3),strides=(1, 1, 1), padding='same',data_format="channels_last")(x)
    model_final=Model(inputs=x_rgb,outputs=x)
    print(x.shape)
#     print(model_final.summary())
    print(model_final.summary())
    plot_model(model_final,to_file='feature_extract.png')
#     x=STLSTM(rgb+flow)
#     x=STLSTM(x)
#     x=STLSTM(x)
#     x=DCONV(x)
#     x=CONV(x)
#     #Combine
#     x=CONV(m1+x)
#     x=DCONV(x)
#     x=CONV(m2+x)
#     x=DCONV(x)
#     x=CONV(m3+x)
#     output=DCONV(x)
    
#     model_final=Model(inputs=[input1,input2,input3],outputs=output)
    
    
    
    
    
    return 


if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval-type', 
        help='specify model type. 1 stream (rgb or flow) or 2 stream (joint = rgb and flow).', 
        type=str, choices=['rgb', 'flow', 'joint'], default='joint')

    parser.add_argument('--no-imagenet-pretrained',
        help='If set, load model weights trained only on kinetics dataset. Otherwise, load model weights trained on imagenet and kinetics dataset.',
        action='store_true')


    args = parser.parse_args()
    main(args)
