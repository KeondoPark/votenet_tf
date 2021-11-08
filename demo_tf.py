# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

""" Demo of using VoteNet 3D object detector to detect objects from a point cloud.
"""

import os
import sys
import numpy as np
import argparse
import importlib
import time

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='sunrgbd', help='Dataset: sunrgbd or scannet [default: sunrgbd]')
parser.add_argument('--num_point', type=int, default=20000, help='Point Number [default: 20000]')
parser.add_argument('--checkpoint_path', default=None, help='Model checkpoint path [default: None]')
parser.add_argument('--gpu_mem_limit', type=int, default=0, help='GPU memory usage')
parser.add_argument('--use_tflite', action='store_true', help='Use tflite')
FLAGS = parser.parse_args()

import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
DATA_DIR = 'sunrgbd'
#DATA_DIR = '/home/aiot/data'
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
from pc_util import random_sampling, read_ply
from ap_helper_tf import parse_predictions

import votenet_tf
from votenet_tf import dump_results

def preprocess_point_cloud(point_cloud):
    ''' Prepare the numpy point cloud (N,3) for forward pass '''
    #point_cloud = point_cloud[:,0:3] # do not use color for now
    floor_height = np.percentile(point_cloud[:,2],0.99)
    height = point_cloud[:,2] - floor_height
    point_cloud = np.concatenate([point_cloud, np.expand_dims(height, 1)],1) # (N,4) or (N,7)
    point_cloud = random_sampling(point_cloud, FLAGS.num_point)
    pc = np.expand_dims(point_cloud.astype(np.float32), 0) # (1,40000,4)
    return pc

def create_pascal_label_colormap():
  """Creates a label colormap used in PASCAL VOC segmentation benchmark.

  Returns:
    A Colormap for visualizing segmentation results.
  """
  colormap = np.zeros((256, 3), dtype=int)
  indices = np.arange(256, dtype=int)

  for shift in reversed(range(8)):
    for channel in range(3):
      colormap[:, channel] |= ((indices >> channel) & 1) << shift
    indices >>= 3

  return colormap

def label_to_color_image(label):
  """Adds color defined by the dataset colormap to the label.

  Args:
    label: A 2D array with integer type, storing the segmentation label.

  Returns:
    result: A 2D array with floating type. The element of the array
      is the color indexed by the corresponding element in the input label
      to the PASCAL color map.

  Raises:
    ValueError: If label is not of rank 2 or its value is larger than color
      map maximum entry.
  """
  if label.ndim != 2:
    raise ValueError('Expect 2-D input label')

  colormap = create_pascal_label_colormap()

  if np.max(label) >= len(colormap):
    raise ValueError('label value too large.')

  return colormap[label]

if __name__=='__main__':
   
    # Limit GPU Memory usage, 256MB suffices in jetson nano
    if FLAGS.gpu_mem_limit:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                tf.config.experimental.set_virtual_device_configuration(gpus[0], 
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=FLAGS.gpu_mem_limit)])
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs", len(logical_gpus), "Logical GPUs")
            except RuntimeError as e:
                print(e)

    # Set file paths and dataset config
    demo_dir = os.path.join(BASE_DIR, 'demo_files') 
    if FLAGS.dataset == 'sunrgbd':
        sys.path.append(os.path.join(ROOT_DIR, 'sunrgbd'))
        from sunrgbd_detection_dataset_tf import DC # dataset config
        from sunrgbd_data import sunrgbd_object, run_semantic_segmentation  
        checkpoint_path = FLAGS.checkpoint_path # os.path.join(demo_dir, 'tf_ckpt_210812')        
        pc_path = os.path.join(demo_dir, 'input_pc_sunrgbd.ply')
        #pc_path = os.path.join(demo_dir, 'pc_person2.ply')
    elif FLAGS.dataset == 'scannet':
        sys.path.append(os.path.join(ROOT_DIR, 'scannet'))
        from scannet_detection_dataset import DC # dataset config
        checkpoint_path = os.path.join(demo_dir, 'pretrained_votenet_on_scannet.tar')
        pc_path = os.path.join(demo_dir, 'input_pc_scannet.ply')
    else:
        print('Unkown dataset. Exiting.')
        exit(-1)

    


    eval_config_dict = {'remove_empty_box': True, 'use_3d_nms': True, 'nms_iou': 0.25,
        'use_old_type_nms': False, 'cls_nms': False, 'per_class_proposal': False,
        'conf_thresh': 0.5, 'dataset_config': DC}

    # Init the model and optimzier    
    net = votenet_tf.VoteNet(num_proposal=256, input_feature_dim=1, vote_factor=1,
        #sampling='seed_fps', num_class=DC.num_class,
        sampling='vote_fps', num_class=DC.num_class,
        num_heading_bin=DC.num_heading_bin,
        num_size_cluster=DC.num_size_cluster,
        mean_size_arr=DC.mean_size_arr,
        use_tflite=FLAGS.use_tflite)
    print('Constructed model.')
    
    # Load checkpoint
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    #inside_checkpoint=tf.train.list_variables(tf.train.latest_checkpoint(checkpoint_path))
    #for node in inside_checkpoint:
    #    print(node)

    if FLAGS.use_tflite:
        #fp1 = tf.train.Checkpoint(fp1=net.backbone_net.fp1)
        #fp2 = tf.train.Checkpoint(fp2=net.backbone_net.fp2)  
        restore_list = []  
        #restore_list.append(tf.train.Checkpoint(backbone_net=fp1))
        #restore_list.append(tf.train.Checkpoint(backbone_net=fp2))
        #restore_list.append(tf.train.Checkpoint(pnet=net.pnet))
        #restore_list.append(tf.train.Checkpoint(vgen=net.vgen))
        
        for layer in restore_list:
            new_root = tf.train.Checkpoint(net=layer)
            new_root.restore(tf.train.latest_checkpoint(checkpoint_path)).expect_partial()

        #new_root = tf.train.Checkpoint(net=pnet)
        #new_root.restore(tf.train.latest_checkpoint(checkpoint_path))

        #new_root = tf.train.Checkpoint(net=backbone_net1)
        #new_root.restore(tf.train.latest_checkpoint(checkpoint_path))

        #new_root = tf.train.Checkpoint(net=backbone_net2)
        #new_root.restore(tf.train.latest_checkpoint(checkpoint_path))
    else:    
        ckpt = tf.train.Checkpoint(epoch=tf.Variable(1), optimizer=optimizer, net=net)
        manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=3)
        ckpt.restore(manager.latest_checkpoint)
        epoch = ckpt.epoch.numpy()

        print("Loaded checkpoint %s (epoch: %d)"%(checkpoint_path, epoch))  
   
    # Load and preprocess input point cloud     
    #point_cloud = read_ply(pc_path)
    #pc = preprocess_point_cloud(point_cloud)
    #print('Loaded point cloud data: %s'%(pc_path))
    data_idx = 5051
    dataset = sunrgbd_object(os.path.join(DATA_DIR,'sunrgbd_trainval'), 'training', use_v1=True)
    point_cloud = dataset.get_depth(data_idx)    

    pointpainting = True
    if pointpainting:
        INPUT_SIZE = 513
        
        from pycoral.utils.edgetpu import make_interpreter
        from pycoral.adapters import common
        from pycoral.adapters import segment
        from PIL import Image
        interpreter = make_interpreter(os.path.join(ROOT_DIR,os.path.join("tflite_models",'sunrgbd_ade20k_11_quant_edgetpu.tflite')))
        interpreter.allocate_tensors()
        width, height = common.input_size(interpreter)

        img = dataset.get_image2(data_idx) 
        print(img.size)  
        orig_w, orig_h = img.size      

        resized_img, (scale, scale) = common.set_resized_input(
            interpreter, img.size, lambda size: img.resize(size, Image.ANTIALIAS))

        interpreter.invoke()
        result = segment.get_output(interpreter)        
        
        new_width, new_height = resized_img.size
        pred_prob = result[:new_height, :new_width, :]
        pred_class = np.argmax(pred_prob, axis=-1) 


        mask_img = Image.fromarray(label_to_color_image(pred_class).astype(np.uint8))

        # Concat resized input image and processed segmentation results.
        output_img = Image.new('RGB', (2 * new_width, new_height))
        output_img.paste(resized_img, (0, 0))
        output_img.paste(mask_img, (width, 0))
        output_img.save('semantic_result.jpg')

        x = (np.array(range(orig_h)) * scale).astype(np.int)
        y = (np.array(range(orig_w)) * scale).astype(np.int)
        xv, yv = np.meshgrid(x, y, indexing='ij')

        pred_prob = pred_prob[xv, yv]
        pred_class = pred_class[xv, yv]


        #result = result[:new_height, :new_width]
        
        """
        with tf.compat.v1.gfile.GFile('test/saved_model/sunrgbd_ade20k_9.pb', "rb") as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())
        
        myGraph = tf.compat.v1.Graph()
        with myGraph.as_default():
            tf.compat.v1.import_graph_def(graph_def, name='')

        sess = tf.compat.v1.Session(graph=myGraph)
        """

        calib = dataset.get_calibration(data_idx)
        uv,d = calib.project_upright_depth_to_image(point_cloud[:,0:3]) #uv: (N, 2)

        # Run image segmentation result and get result
        img = dataset.get_image2(data_idx)        
        #pred_prob, pred_class = run_semantic_segmentation(img, sess, INPUT_SIZE) # (w, h, num_class)   
        pred_prob = pred_prob[:,:,1:(DC.num_class+1)] # 0 is background class
        uv[:,0] = np.rint(uv[:,0] - 1)
        uv[:,1] = np.rint(uv[:,1] - 1)
        projected_class = pred_class[uv[:,1].astype(np.int), uv[:,0].astype(np.int)]
        isObj = np.where((projected_class > 0) & (projected_class < 11), 1, 0) # Point belongs to background?                    
        isObj = np.expand_dims(isObj, axis=-1)
        painted = np.concatenate([point_cloud[:,:3],\
                                isObj,
                                pred_prob[uv[:,1].astype(np.int), uv[:,0].astype(np.int)]
                                ], axis=-1)
    
    pc = preprocess_point_cloud(painted)
   
    # Model inference
    inputs = {'point_clouds': tf.convert_to_tensor(pc)}

    tic = time.time()
    end_points = net(inputs['point_clouds'], training=False)
    toc = time.time()
    print('Inference time: %f'%(toc-tic))

    end_points['point_clouds'] = inputs['point_clouds']
    pred_map_cls = parse_predictions(end_points, eval_config_dict)
    
    type2class={'bed':0, 'table':1, 'sofa':2, 'chair':3, 'toilet':4, 'desk':5, 'dresser':6, 'night_stand':7, 'bookshelf':8, 'bathtub':9, 'person':10}
    class2type = {type2class[t]:t for t in type2class}

    print(pred_map_cls[0])
    #for pred in pred_map_cls[0]:
    #    print('-'*20)        
    #    print('class:', class2type[pred[0].numpy()])
    #    print('conf:', pred[2])
        #print('coords', pred[1])

    """
    from prettytable import PrettyTable

    def count_parameters(model):
        table = PrettyTable(["Modules", "Parameters"])
        total_params = 0
        for layer in model.layers:
            #if not parameter.requires_grad: continue
            param = layer.count_params()            
            table.add_row([layer.name, param])
            total_params+=param
        print(table)
        print(f"Total Trainable Params: {total_params}")
        return total_params
        
    count_parameters(net)
    """
    print('Finished detection. %d object detected.'%(len(pred_map_cls[0][0])))
  
    dump_dir = os.path.join(demo_dir, '%s_results'%(FLAGS.dataset))
    if not os.path.exists(dump_dir): os.mkdir(dump_dir) 
    dump_results(end_points, dump_dir, DC, True)
    print('Dumped detection results to folder %s'%(dump_dir))

