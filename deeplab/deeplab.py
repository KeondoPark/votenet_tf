import numpy as np
from PIL import Image
import os
import tensorflow as tf
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

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

def save_semantic_result(img, pred_class, save_name='semantic_result'):
    orig_w, orig_h = img.size
    mask_img = Image.fromarray(label_to_color_image(pred_class).astype(np.uint8))

    # Concat resized input image and processed segmentation results.
    output_img = Image.new('RGB', (2 * orig_w, orig_h))
    output_img.paste(img, (0, 0))
    output_img.paste(mask_img, (orig_w, 0))
    output_img.save('semantic_results/%s.jpg'%(save_name))


def run_semantic_segmentation_graph(image, sess, input_size):        
    width, height = image.size
    #resize_ratio = 1.0 * input_size / max(width, height)
    resize_ratio = min(input_size[0] / width, input_size[1] / height)
    target_size = (int(resize_ratio * width), int(resize_ratio * height))
    #target_size = input_size
    resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)
    batch_seg_map = sess.run(
        ['SemanticProbabilities:0',
        'SemanticPredictions:0'],
        feed_dict={'ImageTensor:0': [np.asarray(resized_image)]})
    resized_seg_prob = batch_seg_map[0][0] # (height * resize_ratio, width * resize_ratio, num_class)
    resized_seg_class = batch_seg_map[1][0] # (height * resize_ratio, width * resize_ratio)

    # Map segmentation result to original image size
    x = (np.array(range(height)) * resize_ratio)
    y = (np.array(range(width)) * resize_ratio)

    x_int0 = x.astype(np.int)
    y_int0 = y.astype(np.int)
    '''
    x_int1 = np.minimum(x_int + 1, height - 1)
    y_int1 = np.minimum(y_int + 1, width - 1)

    x_f = x - x_int0
    y_f = y - y_int0
    '''
    xv0, yv0 = np.meshgrid(x_int0, y_int0, indexing='ij') # xv, yv has shape (height, width)
    #xv1, yv1 = np.meshgrid(x_int1, y_int1, indexing='ij') # xv, yv has shape (height, width)

    seg_prob = resized_seg_prob[xv0, yv0]
    seg_class = resized_seg_class[xv0, yv0]

    return seg_prob, seg_class

def run_semantic_seg(imgs, save_result=False, save_name='semantic_result', tflite_file='sunrgbd_COCO_15.pb', input_size=513):    
    
    with tf.compat.v1.gfile.GFile(os.path.join(BASE_DIR,'saved_model',tflite_file), "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
    
    myGraph = tf.compat.v1.Graph()
    with myGraph.as_default():
        tf.compat.v1.import_graph_def(graph_def, name='')

    sess = tf.compat.v1.Session(graph=myGraph)    

    pred_prob_list, pred_class_list = [], []
    for img in imgs:
      pred_prob, pred_class = run_semantic_segmentation_graph(img, sess, input_size) # (w, h, num_class)
      pred_prob_list.append(pred_prob)
      pred_class_list.append(pred_class_list)


    # Save semantic segmentation result as image file(Original vs Semantic result)
    if save_result:
        save_semantic_result(img, pred_class, save_name)
    
    return pred_prob, pred_class


def run_semantic_seg_tflite(img_list, save_result=False, tflite_file='sunrgbd_ade20k_12_quant_edgetpu.tflite'):
    
    from pycoral.utils.edgetpu import make_interpreter
    from pycoral.adapters import common
    from pycoral.adapters import segment
    
    interpreter = make_interpreter(os.path.join('saved_model', tflite_file))
    interpreter.allocate_tensors()
    width, height = common.input_size(interpreter)         
    
    
    orig_w, orig_h = img_list[0].size  

    pred_prob_list = []
    for img in img_list:   

      resized_img, (scale1, scale2) = common.set_resized_input(
          interpreter, img.size, lambda size: img.resize(size, Image.ANTIALIAS))    
      
      interpreter.invoke()    

      result = segment.get_output(interpreter)        
      result = result/255 # Output is int8, not dequantized output
      
      new_width, new_height = resized_img.size
      pred_prob = result[:new_height, :new_width, :]    
      
      # Return to original image size
      x = (np.array(range(orig_h)) * scale1).astype(np.int)
      y = (np.array(range(orig_w)) * scale2).astype(np.int)
      xv, yv = np.meshgrid(x, y, indexing='ij')

      pred_prob_list.append(pred_prob[xv, yv]) # height, width

    # Save semantic segmentation result as image file(Original vs Semantic result)
    if save_result:
        print("Saving deeplab results...")
        pred_class = np.argmax(pred_prob, axis=-1) 
        save_semantic_result(img, pred_class)

    return pred_prob_list

if __name__ == '__main__':
  import os, sys
  sys.path.append(os.path.join(ROOT_DIR, 'sunrgbd'))
  from sunrgbd_data import sunrgbd_object
  sys.path.append(os.path.join(ROOT_DIR, 'scannet'))
  from model_util_scannet import scannet_object
  
  import json
  environ_file = os.path.join(ROOT_DIR,'configs','environ.json')
  environ = json.load(open(environ_file))['environ']

  if environ == 'server':    
      DATA_DIR = '/home/aiot/data'
  elif environ == 'jetson':
      DATA_DIR= 'sunrgbd'
  elif environ == 'server2':    
      DATA_DIR = '/data'

  '''
  data_idx = 5051
  #dataset = sunrgbd_object(os.path.join(DATA_DIR,'sunrgbd_trainval'), 'training', use_v1=True)
  dataset = sunrgbd_object(os.path.join(ROOT_DIR,'sunrgbd','sunrgbd_trainval'), 'training', use_v1=True)
  img = dataset.get_image2(data_idx)
  start = time.time()  
  pred_prob = run_semantic_seg_tflite(img, tflite_file='sunrgbd_COCO_15_quant_edgetpu.tflite')
  print("Deeplab inference time", time.time() - start)
  #pred_prob, pred_class = run_semantic_seg(img, save_result=True, save_name=str(data_idx))  
  #print(np.unique(pred_class))
  
  '''
  dataobj = scannet_object()
  img, pose = dataobj.get_image_and_pose(0)
  start = time.time()  
  pred_prob = run_semantic_seg_tflite(img, tflite_file='scannet_2_quant_edgetpu.tflite')
  pred_prob = run_semantic_seg_tflite(img, tflite_file='scannet_2_quant_edgetpu.tflite')
  pred_prob = run_semantic_seg_tflite(img, tflite_file='scannet_2_quant_edgetpu.tflite')
  print("Deeplab inference time", time.time() - start)
  






