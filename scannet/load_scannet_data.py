# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

""" Load Scannet scenes with vertices and ground truth labels
for semantic and instance segmentations
"""

# python imports
import math
import os, sys, argparse
import inspect
import json
import pdb
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../deeplab'))
sys.path.append(BASE_DIR)
from deeplab import run_semantic_segmentation_graph
import tensorflow as tf


try:
    import numpy as np
except:
    print("Failed to import numpy package.")
    sys.exit(-1)

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
import scannet_utils

def read_aggregation(filename):
    assert os.path.isfile(filename)
    object_id_to_segs = {}
    label_to_segs = {}
    with open(filename) as f:
        data = json.load(f)
        num_objects = len(data['segGroups'])
        for i in range(num_objects):
            object_id = data['segGroups'][i]['objectId'] + 1 # instance ids should be 1-indexed
            label = data['segGroups'][i]['label']
            segs = data['segGroups'][i]['segments']
            object_id_to_segs[object_id] = segs
            if label in label_to_segs:
                label_to_segs[label].extend(segs)
            else:
                label_to_segs[label] = segs
    return object_id_to_segs, label_to_segs


def read_segmentation(filename):
    assert os.path.isfile(filename)
    seg_to_verts = {}
    with open(filename) as f:
        data = json.load(f)
        num_verts = len(data['segIndices'])
        for i in range(num_verts):
            seg_id = data['segIndices'][i]
            if seg_id in seg_to_verts:
                seg_to_verts[seg_id].append(i)
            else:
                seg_to_verts[seg_id] = [i]
    return seg_to_verts, num_verts


def export(mesh_file, agg_file, seg_file, meta_file, label_map_file, output_file=None):
    """ points are XYZ RGB (RGB in 0-255),
    semantic label as nyu40 ids,
    instance label as 1-#instance,
    box as (cx,cy,cz,dx,dy,dz,semantic_label)
    """
    label_map = scannet_utils.read_label_mapping(label_map_file,
        label_from='raw_category', label_to='nyu40id')    
    mesh_vertices = scannet_utils.read_mesh_vertices_rgb(mesh_file)

    # Load scene axis alignment matrix
    lines = open(meta_file).readlines()
    for line in lines:
        if 'axisAlignment' in line:
            axis_align_matrix = [float(x) \
                for x in line.rstrip().strip('axisAlignment = ').split(' ')]
            break
    axis_align_matrix = np.array(axis_align_matrix).reshape((4,4))
    pts = np.ones((mesh_vertices.shape[0], 4))
    pts[:,0:3] = mesh_vertices[:,0:3]
    pts = np.dot(pts, axis_align_matrix.transpose()) # Nx4
    mesh_vertices[:,0:3] = pts[:,0:3]

    # Load semantic and instance labels
    object_id_to_segs, label_to_segs = read_aggregation(agg_file)
    seg_to_verts, num_verts = read_segmentation(seg_file)
    label_ids = np.zeros(shape=(num_verts), dtype=np.uint32) # 0: unannotated
    object_id_to_label_id = {}
    for label, segs in label_to_segs.items():
        label_id = label_map[label]
        for seg in segs:
            verts = seg_to_verts[seg]
            label_ids[verts] = label_id
    instance_ids = np.zeros(shape=(num_verts), dtype=np.uint32) # 0: unannotated
    num_instances = len(np.unique(list(object_id_to_segs.keys())))
    for object_id, segs in object_id_to_segs.items():
        for seg in segs:
            verts = seg_to_verts[seg]
            instance_ids[verts] = object_id
            if object_id not in object_id_to_label_id:
                object_id_to_label_id[object_id] = label_ids[verts][0]
    instance_bboxes = np.zeros((num_instances,7))
    for obj_id in object_id_to_segs:
        label_id = object_id_to_label_id[obj_id]
        obj_pc = mesh_vertices[instance_ids==obj_id, 0:3]
        if len(obj_pc) == 0: continue
        # Compute axis aligned box
        # An axis aligned bounding box is parameterized by
        # (cx,cy,cz) and (dx,dy,dz) and label id
        # where (cx,cy,cz) is the center point of the box,
        # dx is the x-axis length of the box.
        xmin = np.min(obj_pc[:,0])
        ymin = np.min(obj_pc[:,1])
        zmin = np.min(obj_pc[:,2])
        xmax = np.max(obj_pc[:,0])
        ymax = np.max(obj_pc[:,1])
        zmax = np.max(obj_pc[:,2])
        bbox = np.array([(xmin+xmax)/2, (ymin+ymax)/2, (zmin+zmax)/2,
            xmax-xmin, ymax-ymin, zmax-zmin, label_id])
        # NOTE: this assumes obj_id is in 1,2,3,.,,,.NUM_INSTANCES
        instance_bboxes[obj_id-1,:] = bbox 

    if output_file is not None:
        np.save(output_file+'_vert.npy', mesh_vertices)
        np.save(output_file+'_sem_label.npy', label_ids)
        np.save(output_file+'_ins_label.npy', instance_ids)
        np.save(output_file+'_bbox.npy', instance_bboxes)

    return mesh_vertices, label_ids, instance_ids,\
        instance_bboxes, object_id_to_label_id

def read_matrix(filepath):
    out_matrix = np.zeros((4,4))

    with open(filepath, 'r') as f:
        lines = f.readlines()
        i = 0
        for line in lines:
            values = line.strip().split(' ')
            for j in range(4):
                out_matrix[i,j] = values[j]
            i += 1
    return out_matrix

def export_2dseg_results(sess, exported_scan_dir=None, output_dir=None):
    ## Preparation for pointpainting    
    n_classes = 18

    # deeplabv3+ tf session
    INPUT_SIZE = (321, 321)    

    color_files = os.listdir(os.path.join(exported_scan_dir, 'color'))   
    frame_nums = [int(f[:-4]) for f in color_files]

    if len(frame_nums) > 2 * 3:
        N = 2
    else:
        N = 1

    for frame in frame_nums[0::N]:
        # Get sementationr esult
        img_file = os.path.join(exported_scan_dir, 'color', str(frame) + '.jpg')
        img = Image.open(img_file)
        pred_prob, pred_class = run_semantic_segmentation_graph(img, sess, INPUT_SIZE)
        pred_prob = pred_prob[:,:,:(n_classes+1)].astype(np.float32) # 0 is background class
        np.save(os.path.join(output_dir, 'prob_' + str(frame) + '.npy'), pred_prob)
        #np.save(os.path.join(output_dir, 'class_' + str(frame) + '.npy'), pred_class)



def export_with_2dseg(mesh_file, agg_file, seg_file, meta_file, label_map_file, output_file=None, exported_scan_dir=None):
    """ points are XYZ RGB (RGB in 0-255),
    semantic label as nyu40 ids,
    instance label as 1-#instance,
    box as (cx,cy,cz,dx,dy,dz,semantic_label)
    """
    ## Preparation for pointpainting    
    n_classes = 18

    # deeplabv3+ tf session
    INPUT_SIZE = (321, 321)    
    with tf.compat.v1.gfile.GFile('../deeplab/saved_model/scannet_2.pb', "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
    
    myGraph = tf.compat.v1.Graph()
    with myGraph.as_default():
        tf.compat.v1.import_graph_def(graph_def, name='')

    sess = tf.compat.v1.Session(graph=myGraph)


    label_map = scannet_utils.read_label_mapping(label_map_file,
        label_from='raw_category', label_to='nyu40id')    
    mesh_vertices = scannet_utils.read_mesh_vertices_rgb(mesh_file) #(N,6)

    # Load scene axis alignment matrix
    lines = open(meta_file).readlines()
    for line in lines:
        if 'axisAlignment' in line:
            axis_align_matrix = [float(x) \
                for x in line.rstrip().strip('axisAlignment = ').split(' ')]
        if 'colorWidth' in line:
            colorW = int(line.strip('colorWidth = ').split(' ')[0])
        if 'colorHeight' in line:
            colorH = int(line.strip('colorHeight = ').split(' ')[0])

    axis_align_matrix = np.array(axis_align_matrix).reshape((4,4))


    ## Point painting
    # Camera instrinsic
    color_intrinsic_file = os.path.join(exported_scan_dir, 'intrinsic', 'intrinsic_color.txt')   
    color_intrinsic = read_matrix(color_intrinsic_file)


    color_files = os.listdir(os.path.join(exported_scan_dir, 'color'))    

    frame_nums = [int(f[:-4]) for f in color_files]
    
    #frames_selected = []

    #num_img = 5
    #sliced = len(frame_nums) // num_img
    #for i in range(num_img):
    #    if i == num_img - 1:
    #        frames_selected.append(np.random.choice(frame_nums[sliced*i:], 1, replace=False)[0])
    #    else:
    #        frames_selected.append(np.random.choice(frame_nums[sliced*i:sliced*(i+1)], 1, replace=False)[0])
    #frames_selected.append(0)

    #vertices_recon = np.zeros((len(mesh_vertices), 3 + (1 + 1 + n_classes))) # number of class + background class + isPainted 
    #vertices_recon[:,:3] = mesh_vertices[:,:3]

    for frame in frame_nums:
        '''
        pose_file = os.path.join(exported_scan_dir, 'pose', str(frame) + '.txt')        
        
        # read pose matrix(Rotation and translation)
        pose_matrix = read_matrix(pose_file)   

        sampled_h = np.ones((len(mesh_vertices), 4))
        sampled_h[:,:3] = mesh_vertices[:,:3]

        camera_coord = np.matmul(np.linalg.inv(pose_matrix), np.transpose(sampled_h))
        camera_proj = np.matmul(color_intrinsic, camera_coord)

        # Get valid points for the image
        x = camera_proj[0,:]
        y = camera_proj[1,:]
        z = camera_proj[2,:]
        filter_idx = np.where((x/z >= 0) & (x/z < colorW) & (y/z >= 0) & (y/z < colorH) & (z > 0))[0]

        # Normalize by 4th coords(Homogeneous -> 3 coords system)
        camera_proj_normalized = camera_proj / camera_proj[2,:]

        #Get 3d -> 2d mapping
        projected = camera_proj_normalized[:2, filter_idx]

        #Reduce to 320,240 size
        camera_proj_sm = np.zeros((5, projected.shape[-1]))

        camera_proj_sm[0,:] = projected[0,:] * 320/colorW
        camera_proj_sm[1,:] = projected[1,:] * 240/colorH

        # Get pixel index
        x = camera_proj_sm[0,:].astype(np.uint8)
        y = camera_proj_sm[1,:].astype(np.uint8)
        ''' 
        
        # Get sementationr esult
        img_file = os.path.join(exported_scan_dir, 'color', str(frame) + '.jpg')
        img = Image.open(img_file)
        pred_prob, pred_class = run_semantic_segmentation_graph(img, sess, INPUT_SIZE)
        pred_prob = pred_prob[:,:,:(n_classes+1)] # 0 is background class
        

        #pred_prob = pred_prob[y, x]
        #projected_class = pred_class[y, x]    

        #isPainted = np.where((projected_class > 0) & (projected_class < n_classes+1), 1, 0) # Point belongs to foreground?        
        
        #vertices_recon[filter_idx, 3] = isPainted
        #vertices_recon[filter_idx, 4:] = pred_prob

    
    pts = np.ones((mesh_vertices.shape[0], 4))
    pts[:,0:3] = mesh_vertices[:,0:3]
    pts = np.dot(pts, axis_align_matrix.transpose()) # Nx4
    mesh_vertices[:,0:3] = pts[:,0:3]
    '''

    pts = np.ones((vertices_recon.shape[0], 4))
    pts[:,0:3] = vertices_recon[:,0:3]
    pts = np.dot(pts, axis_align_matrix.transpose()) # Nx4
    vertices_recon[:,0:3] = pts[:,0:3]
    '''
 

    # Load semantic and instance labels
    object_id_to_segs, label_to_segs = read_aggregation(agg_file)
    seg_to_verts, num_verts = read_segmentation(seg_file)
    label_ids = np.zeros(shape=(num_verts), dtype=np.uint32) # 0: unannotated
    object_id_to_label_id = {}
    for label, segs in label_to_segs.items():
        label_id = label_map[label]
        for seg in segs:
            verts = seg_to_verts[seg]
            label_ids[verts] = label_id
    instance_ids = np.zeros(shape=(num_verts), dtype=np.uint32) # 0: unannotated
    num_instances = len(np.unique(list(object_id_to_segs.keys())))
    for object_id, segs in object_id_to_segs.items():
        for seg in segs:
            verts = seg_to_verts[seg]
            instance_ids[verts] = object_id
            if object_id not in object_id_to_label_id:
                object_id_to_label_id[object_id] = label_ids[verts][0]
    instance_bboxes = np.zeros((num_instances,7))
    for obj_id in object_id_to_segs:
        label_id = object_id_to_label_id[obj_id]
        obj_pc = mesh_vertices[instance_ids==obj_id, 0:3]
        if len(obj_pc) == 0: continue
        # Compute axis aligned box
        # An axis aligned bounding box is parameterized by
        # (cx,cy,cz) and (dx,dy,dz) and label id
        # where (cx,cy,cz) is the center point of the box,
        # dx is the x-axis length of the box.
        xmin = np.min(obj_pc[:,0])
        ymin = np.min(obj_pc[:,1])
        zmin = np.min(obj_pc[:,2])
        xmax = np.max(obj_pc[:,0])
        ymax = np.max(obj_pc[:,1])
        zmax = np.max(obj_pc[:,2])
        bbox = np.array([(xmin+xmax)/2, (ymin+ymax)/2, (zmin+zmax)/2,
            xmax-xmin, ymax-ymin, zmax-zmin, label_id])
        # NOTE: this assumes obj_id is in 1,2,3,.,,,.NUM_INSTANCES
        instance_bboxes[obj_id-1,:] = bbox 

    if output_file is not None:
        np.save(output_file+'_vert.npy', mesh_vertices)
        np.save(output_file+'_sem_label.npy', label_ids)
        np.save(output_file+'_ins_label.npy', instance_ids)
        np.save(output_file+'_bbox.npy', instance_bboxes)

    return mesh_vertices, label_ids, instance_ids,\
        instance_bboxes, object_id_to_label_id

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scan_path', required=True, help='path to scannet scene (e.g., data/ScanNet/v2/scene0000_00')
    parser.add_argument('--output_file', required=True, help='output file')
    parser.add_argument('--label_map_file', required=True, help='path to scannetv2-labels.combined.tsv')
    opt = parser.parse_args()

    scan_name = os.path.split(opt.scan_path)[-1]
    mesh_file = os.path.join(opt.scan_path, scan_name + '_vh_clean_2.ply')
    agg_file = os.path.join(opt.scan_path, scan_name + '.aggregation.json')
    seg_file = os.path.join(opt.scan_path, scan_name + '_vh_clean_2.0.010000.segs.json')
    meta_file = os.path.join(opt.scan_path, scan_name + '.txt') # includes axisAlignment info for the train set scans.

    #export(mesh_file, agg_file, seg_file, meta_file, opt.label_map_file, opt.output_file)

    export_with_2dseg(mesh_file, agg_file, seg_file, meta_file, opt.label_map_file, opt.output_file)

if __name__ == '__main__':
    main()