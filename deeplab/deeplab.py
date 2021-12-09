import numpy as np
from PIL import Image
import os
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR

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

def save_semantic_result(img, pred_class):
    orig_w, orig_h = img.size
    mask_img = Image.fromarray(label_to_color_image(pred_class).astype(np.uint8))

    # Concat resized input image and processed segmentation results.
    output_img = Image.new('RGB', (2 * orig_w, orig_h))
    output_img.paste(img, (0, 0))
    output_img.paste(mask_img, (orig_w, 0))
    output_img.save('semantic_result.jpg')


def run_semantic_segmentation_graph(image, sess, input_size):        
    width, height = image.size
    resize_ratio = 1.0 * input_size / max(width, height)
    target_size = (int(resize_ratio * width), int(resize_ratio * height))
    resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)
    batch_seg_map = sess.run(
        ['SemanticProbabilities:0',
        'SemanticPredictions:0'],
        feed_dict={'ImageTensor:0': [np.asarray(resized_image)]})
    resized_seg_prob = batch_seg_map[0][0] # (height * resize_ratio, width * resize_ratio, num_class)
    resized_seg_class = batch_seg_map[1][0] # (height * resize_ratio, width * resize_ratio)

    # Map segmentation result to original image size
    x = (np.array(range(height)) * resize_ratio).astype(np.int)
    y = (np.array(range(width)) * resize_ratio).astype(np.int)

    xv, yv = np.meshgrid(x, y, indexing='ij') # xv, yv has shape (height, width)

    seg_prob = resized_seg_prob[xv, yv]
    seg_class = resized_seg_class[xv, yv]

    return seg_prob, seg_class

def run_semantic_seg(img, save_result=False):
    INPUT_SIZE = 513
    with tf.compat.v1.gfile.GFile('saved_model/sunrgbd_ade20k_12.pb', "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
    
    myGraph = tf.compat.v1.Graph()
    with myGraph.as_default():
        tf.compat.v1.import_graph_def(graph_def, name='')

    sess = tf.compat.v1.Session(graph=myGraph)    
    pred_prob, pred_class = run_semantic_segmentation_graph(img, sess, INPUT_SIZE) # (w, h, num_class)       

    # Save semantic segmentation result as image file(Original vs Semantic result)
    if save_result:
        save_semantic_result(img, pred_class)
    
    return pred_prob, pred_class


def run_semantic_seg_tflite(img, save_result=False):
    
    from pycoral.utils.edgetpu import make_interpreter
    from pycoral.adapters import common
    from pycoral.adapters import segment
    
    interpreter = make_interpreter(os.path.join(ROOT_DIR, os.path.join('sunrgbd_ade20k_12_quant_edgetpu.tflite')))
    interpreter.allocate_tensors()
    width, height = common.input_size(interpreter)         
    
    orig_w, orig_h = img.size      

    resized_img, (scale, scale) = common.set_resized_input(
        interpreter, img.size, lambda size: img.resize(size, Image.ANTIALIAS))

    interpreter.invoke()
    result = segment.get_output(interpreter)        
    
    new_width, new_height = resized_img.size
    pred_prob = result[:new_height, :new_width, :]    
    
    # Return to original image size
    x = (np.array(range(orig_h)) * scale).astype(np.int)
    y = (np.array(range(orig_w)) * scale).astype(np.int)
    xv, yv = np.meshgrid(x, y, indexing='ij')

    pred_prob = pred_prob[xv, yv] # height, width
    #pred_class = pred_class[xv, yv]

    # Save semantic segmentation result as image file(Original vs Semantic result)
    if save_result:
        pred_class = np.argmax(pred_prob, axis=-1) 
        save_semantic_result(img, pred_class)

    return pred_prob