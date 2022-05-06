 
# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

""" Batch mode in loading Scannet scenes with vertices and ground truth labels
for semantic and instance segmentations
Usage example: python ./batch_load_scannet_data.py
"""
import os
import sys
import datetime
import numpy as np
from load_scannet_data import export, export_with_2dseg, export_2dseg_results
import pdb
import tensorflow as tf

SCANNET_DIR = 'scans'
EXPORTED_2D_DIR = 'frames_square'
TRAIN_SCAN_NAMES = [line.rstrip() for line in open('meta_data/scannet_train.txt')]
LABEL_MAP_FILE = 'meta_data/scannetv2-labels.combined.tsv'
DONOTCARE_CLASS_IDS = np.array([])
OBJ_CLASS_IDS = np.array([3,4,5,6,7,8,9,10,11,12,14,16,24,28,33,34,36,39])
MAX_NUM_POINT = 50000
OUTPUT_FOLDER = './scannet_train_detection_data'
#OUTPUT_FOLDER = './scannet_train_detection_data_painted'
OUTPUT_2DSEG_FOLDER = './semantic_2d_results'

def export_one_scan(scan_name, output_filename_prefix):    
    mesh_file = os.path.join(SCANNET_DIR, scan_name, scan_name + '_vh_clean_2.ply')
    agg_file = os.path.join(SCANNET_DIR, scan_name, scan_name + '.aggregation.json')
    seg_file = os.path.join(SCANNET_DIR, scan_name, scan_name + '_vh_clean_2.0.010000.segs.json')
    meta_file = os.path.join(SCANNET_DIR, scan_name, scan_name + '.txt') # includes axisAlignment info for the train set scans.   

    mesh_vertices, semantic_labels, instance_labels, instance_bboxes, instance2semantic = \
        export(mesh_file, agg_file, seg_file, meta_file, LABEL_MAP_FILE, None)

    mask = np.logical_not(np.in1d(semantic_labels, DONOTCARE_CLASS_IDS))
    mesh_vertices = mesh_vertices[mask,:]
    semantic_labels = semantic_labels[mask]
    instance_labels = instance_labels[mask]

    num_instances = len(np.unique(instance_labels))
    print('Num of instances: ', num_instances)

    bbox_mask = np.in1d(instance_bboxes[:,-1], OBJ_CLASS_IDS)
    instance_bboxes = instance_bboxes[bbox_mask,:]
    print('Num of care instances: ', instance_bboxes.shape[0])

    N = mesh_vertices.shape[0]
    if N > MAX_NUM_POINT:
        choices = np.random.choice(N, MAX_NUM_POINT, replace=False)
        mesh_vertices = mesh_vertices[choices, :]
        semantic_labels = semantic_labels[choices]
        instance_labels = instance_labels[choices]

    np.save(output_filename_prefix+'_vert.npy', mesh_vertices)
    np.save(output_filename_prefix+'_sem_label.npy', semantic_labels)
    np.save(output_filename_prefix+'_ins_label.npy', instance_labels)
    np.save(output_filename_prefix+'_bbox.npy', instance_bboxes)

def export_one_scan_with_2dseg(scan_name, output_filename_prefix):    
    mesh_file = os.path.join(SCANNET_DIR, scan_name, scan_name + '_vh_clean_2.ply')
    agg_file = os.path.join(SCANNET_DIR, scan_name, scan_name + '.aggregation.json')
    seg_file = os.path.join(SCANNET_DIR, scan_name, scan_name + '_vh_clean_2.0.010000.segs.json')
    meta_file = os.path.join(SCANNET_DIR, scan_name, scan_name + '.txt') # includes axisAlignment info for the train set scans. 

    exported_scan_dir = os.path.join(EXPORTED_2D_DIR, scan_name)

    mesh_vertices, semantic_labels, instance_labels, instance_bboxes, instance2semantic = \
        export_with_2dseg(mesh_file, agg_file, seg_file, meta_file, LABEL_MAP_FILE, None, exported_scan_dir)

    mask = np.logical_not(np.in1d(semantic_labels, DONOTCARE_CLASS_IDS))
    mesh_vertices = mesh_vertices[mask,:]
    semantic_labels = semantic_labels[mask]
    instance_labels = instance_labels[mask]

    num_instances = len(np.unique(instance_labels))
    print('Num of instances: ', num_instances)

    bbox_mask = np.in1d(instance_bboxes[:,-1], OBJ_CLASS_IDS)
    instance_bboxes = instance_bboxes[bbox_mask,:]
    print('Num of care instances: ', instance_bboxes.shape[0])

    N = mesh_vertices.shape[0]
    if N > MAX_NUM_POINT:
        choices = np.random.choice(N, MAX_NUM_POINT, replace=False)
        mesh_vertices = mesh_vertices[choices, :]
        semantic_labels = semantic_labels[choices]
        instance_labels = instance_labels[choices]

    np.save(output_filename_prefix+'_vert.npy', mesh_vertices)
    np.save(output_filename_prefix+'_sem_label.npy', semantic_labels)
    np.save(output_filename_prefix+'_ins_label.npy', instance_labels)
    np.save(output_filename_prefix+'_bbox.npy', instance_bboxes)



def batch_export():
    if not os.path.exists(OUTPUT_FOLDER):
        print('Creating new data folder: {}'.format(OUTPUT_FOLDER))                
        os.mkdir(OUTPUT_FOLDER)        

    exported_scan_dir = os.path.join(EXPORTED_2D_DIR, scan_name)
        
    for scan_name in TRAIN_SCAN_NAMES:
        print('-'*20+'begin')
        print(datetime.datetime.now())
        print(scan_name)
        output_filename_prefix = os.path.join(OUTPUT_FOLDER, scan_name) 
        if os.path.isfile(output_filename_prefix+'_vert.npy'):
            print('File already exists. skipping.')
            print('-'*20+'done')
            continue
        try:            
            export_one_scan(scan_name, output_filename_prefix)
        except:
            print('Failed export scan: %s'%(scan_name))                    
        print('-'*20+'done')

def save_2dsemantic_results():
    if not os.path.exists(OUTPUT_2DSEG_FOLDER):
        print('Creating new data folder: {}'.format(OUTPUT_2DSEG_FOLDER))                
        os.mkdir(OUTPUT_2DSEG_FOLDER) 
    
    with tf.compat.v1.gfile.GFile('../deeplab/saved_model/scannet_3.pb', "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
    
    myGraph = tf.compat.v1.Graph()
    with myGraph.as_default():
        tf.compat.v1.import_graph_def(graph_def, name='')

    sess = tf.compat.v1.Session(graph=myGraph)

        
    for scan_name in TRAIN_SCAN_NAMES:
        print('-'*20+'begin')
        print(datetime.datetime.now())
        print(scan_name)
        os.makedirs(os.path.join(OUTPUT_2DSEG_FOLDER, scan_name), exist_ok=True)
        output_filename_prefix = os.path.join(OUTPUT_2DSEG_FOLDER, scan_name) 
        exported_scan_dir = os.path.join(EXPORTED_2D_DIR, scan_name)
        
        try:            
            export_2dseg_results(sess, exported_scan_dir, output_filename_prefix)
        except:
            print('Failed export scan: %s'%(scan_name))                    
        print('-'*20+'done')                

if __name__=='__main__':    
    #batch_export()
    save_2dsemantic_results()

