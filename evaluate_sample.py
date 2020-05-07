'''
Loads pretrained model of I3d Inception architecture for the paper: 'https://arxiv.org/abs/1705.07750'
Evaluates a RGB and Flow sample similar to the paper's github repo: 'https://github.com/deepmind/kinetics-i3d'
'''

import numpy as np
import argparse
from tensorflow.python.keras.models import Model
from i3d_inception import Inception_Inflated3d

from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
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
#     model_res = Model(inputs=model.input, outputs=[model.get_layer('activation_1').output,model.get_layer('activation_7').output,model.get_layer('activation_7').output])
    
    
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
    FILTERS = 256
    KERNEL_SIZE = 3
    
    #Model 1
#     m1,m2,m3=model_res(input3)
    #MODEL 2
#     rgb=model_rgb(input1)
#     flow=model_flow(input2)
    cells = STLSTM.StackedSTLSTMCells([STLSTM.STLSTMCell(filters=FILTERS, kernel_size=KERNEL_SIZE,padding="same",data_format="channels_last") for i in range(NUM_CELL)])
    x=STLSTM.STLSTM2D(cells, return_sequences=True)(model_rgb.output)
    model_final=Model(inputs=model_rgb.input,outputs=x)
    print(model_final.summary())
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
