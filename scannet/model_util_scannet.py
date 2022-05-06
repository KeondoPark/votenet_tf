# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
from box_util import get_3d_box
from PIL import Image

class ScannetDatasetConfig(object):
    def __init__(self):
        self.num_class = 18
        self.num_heading_bin = 1
        self.num_size_cluster = 18

        self.type2class = {'cabinet':0, 'bed':1, 'chair':2, 'sofa':3, 'table':4, 'door':5,
            'window':6,'bookshelf':7,'picture':8, 'counter':9, 'desk':10, 'curtain':11,
            'refrigerator':12, 'showercurtrain':13, 'toilet':14, 'sink':15, 'bathtub':16, 'garbagebin':17}  
        self.class2type = {self.type2class[t]:t for t in self.type2class}
        self.nyu40ids = np.array([3,4,5,6,7,8,9,10,11,12,14,16,24,28,33,34,36,39])
        self.nyu40id2class = {nyu40id: i for i,nyu40id in enumerate(list(self.nyu40ids))}
        self.mean_size_arr = np.load(os.path.join(ROOT_DIR,'scannet/meta_data/scannet_means.npz'))['arr_0']
        self.type_mean_size = {}
        for i in range(self.num_size_cluster):
            self.type_mean_size[self.class2type[i]] = self.mean_size_arr[i,:]

        # The 2dimage size of exported scan
        self.exported_scan_size = (320,240)

    def angle2class(self, angle):
        ''' Convert continuous angle to discrete class
            [optinal] also small regression number from  
            class center angle to current angle.
           
            angle is from 0-2pi (or -pi~pi), class center at 0, 1*(2pi/N), 2*(2pi/N) ...  (N-1)*(2pi/N)
            return is class of int32 of 0,1,...,N-1 and a number such that
                class*(2pi/N) + number = angle

            NOT USED.
        '''
        assert(False)
    
    def class2angle(self, pred_cls, residual, to_label_format=True):
        ''' Inverse function to angle2class.
        
        As ScanNet only has axis-alined boxes so angles are always 0. '''
        return 0

    def size2class(self, size, type_name):
        ''' Convert 3D box size (l,w,h) to size class and size residual '''
        size_class = self.type2class[type_name]
        size_residual = size - self.type_mean_size[type_name]
        return size_class, size_residual
    
    def class2size(self, pred_cls, residual):
        ''' Inverse function to size2class '''        
        return self.mean_size_arr[pred_cls, :] + residual

    def param2obb(self, center, heading_class, heading_residual, size_class, size_residual):
        heading_angle = self.class2angle(heading_class, heading_residual)
        box_size = self.class2size(int(size_class), size_residual)
        obb = np.zeros((7,))
        obb[0:3] = center
        obb[3:6] = box_size
        obb[6] = heading_angle*-1
        return obb

def rotate_aligned_boxes(input_boxes, rot_mat):    
    centers, lengths = input_boxes[:,0:3], input_boxes[:,3:6]    
    new_centers = np.dot(centers, np.transpose(rot_mat))
           
    dx, dy = lengths[:,0]/2.0, lengths[:,1]/2.0
    new_x = np.zeros((dx.shape[0], 4))
    new_y = np.zeros((dx.shape[0], 4))
    
    for i, crnr in enumerate([(-1,-1), (1, -1), (1, 1), (-1, 1)]):        
        crnrs = np.zeros((dx.shape[0], 3))
        crnrs[:,0] = crnr[0]*dx
        crnrs[:,1] = crnr[1]*dy
        crnrs = np.dot(crnrs, np.transpose(rot_mat))
        new_x[:,i] = crnrs[:,0]
        new_y[:,i] = crnrs[:,1]
    
    
    new_dx = 2.0*np.max(new_x, 1)
    new_dy = 2.0*np.max(new_y, 1)    
    new_lengths = np.stack((new_dx, new_dy, lengths[:,2]), axis=1)
                  
    return np.concatenate([new_centers, new_lengths], axis=1)


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

class scannet_object(object):
    ''' Load and parse object data '''
    def __init__(self, split_set='train'):
        self.raw_data_path = os.path.join(BASE_DIR, 'scans')
        self.data_path = os.path.join(BASE_DIR, 'scannet_train_detection_data')
        self.exported_scan_dir = os.path.join(BASE_DIR, 'frames_square')
        all_scan_names = list(set([os.path.basename(x)[0:12] \
            for x in os.listdir(self.data_path) if x.startswith('scene')]))

        self.split_set = split_set        

        split_filenames = os.path.join(ROOT_DIR, 'scannet/meta_data',
                'scannetv2_{}.txt'.format(split_set))
        with open(split_filenames, 'r') as f:
            self.scan_names = f.read().splitlines()   
        # remove unavailiable scans
        self.num_scans = len(self.scan_names)
        self.scan_names = [sname for sname in self.scan_names \
            if sname in all_scan_names]
        print('kept {} scans out of {}'.format(len(self.scan_names), self.num_scans))        
       

    def __len__(self):
        return self.num_scans

    def get_image_and_pose(self, idx, num_img=1):
        # Randomly select image and pose from the frames
        scan_name = self.scan_names[idx]
        color_files = os.listdir(os.path.join(self.exported_scan_dir, scan_name, 'color'))

        #Randomly choose color file
        color_file_idx = np.random.choice(len(color_files),num_img)
        color_files_selected = [color_files[idx] for idx in color_file_idx]
        pose_files = [f[:-4] + '.txt' for f in color_files_selected] #Same name but different extension
        
        imgs, pose_matrices = [], []
        for i in range(num_img):
            img = Image.open(os.path.join(self.exported_scan_dir, scan_name, 'color', color_files_selected[i]))
            pose_file_path = os.path.join(self.exported_scan_dir, scan_name, 'pose', pose_files[i])
            pose_matrix = read_matrix(pose_file_path) 
            imgs.append(img)
            pose_matrices.append(pose_matrix)
        
        return imgs, pose_matrices

    def get_pointcloud(self, idx): 
        scan_name = self.scan_names[idx]
        pointcloud = np.load(os.path.join(self.data_path, scan_name)+'_vert.npy')
        return pointcloud

    def get_color_intrinsic(self, idx):
        # Return Color intrinsic of selected scan
        scan_name = self.scan_names[idx]
        color_intrinsic_file = os.path.join(self.exported_scan_dir, scan_name, 'intrinsic', 'intrinsic_color.txt')  
        color_intrinsic = read_matrix(color_intrinsic_file)
        return color_intrinsic        

    def get_axisAlignment(self, idx):
        scan_name = self.scan_names[idx]
        meta_file = os.path.join(self.raw_data_path, scan_name, scan_name + '.txt') # includes axisAlignment info for the train set scans.
        lines = open(meta_file).readlines()
        for line in lines:
            if 'axisAlignment' in line:
                axis_align_matrix = [float(x) \
                    for x in line.rstrip().strip('axisAlignment = ').split(' ')]

        axis_align_matrix = np.array(axis_align_matrix).reshape((4,4))
        return axis_align_matrix
    def get_colorSize(self, idx):
        scan_name = self.scan_names[idx]
        meta_file = os.path.join(self.raw_data_path, scan_name, scan_name + '.txt') # includes axisAlignment info for the train set scans.
        lines = open(meta_file).readlines()
        for line in lines:
            if 'colorWidth' in line:
                colorW = int(line.strip('colorWidth = ').split(' ')[0])
            if 'colorHeight' in line:
                colorH = int(line.strip('colorHeight = ').split(' ')[0])
        return colorW, colorH

    def get_calibration(self, idx, pose):
        K = self.get_color_intrinsic(idx)
        axis_align_matrix = self.get_axisAlignment(idx)
        colorW, colorH = self.get_colorSize(idx)
        return scannet_calibration(K, axis_align_matrix, pose, colorW, colorH)

DC = ScannetDatasetConfig()

class scannet_calibration(object):
    def __init__(self, K, axis_align_matrix, pose, colorW, colorH):
        self.K = K
        self.axis_align_matrix = axis_align_matrix
        self.pose = pose
        self.colorW = colorW
        self.colorH = colorH

    def project_upright_depth_to_image(self, point_cloud):

        alinged_verts = np.ones((point_cloud.shape[0],4))
        alinged_verts[:,:3] = point_cloud[:,:3]
        
        unalign_verts = np.dot(alinged_verts, np.linalg.inv(self.axis_align_matrix.transpose()))

        sampled_h = np.ones((len(unalign_verts), 4))
        sampled_h[:,:3] = unalign_verts[:,:3]

        camera_coord = np.matmul(np.linalg.inv(self.pose), np.transpose(sampled_h))
        camera_proj = np.matmul(self.K, camera_coord)

        # Get valid points for the image
        x = camera_proj[0,:]
        y = camera_proj[1,:]
        z = camera_proj[2,:]
        filter_idx = np.where((x/z >= 0) & (x/z < self.colorW) & (y/z >= 0) & (y/z < self.colorH) & (z > 0))[0]

        # Normalize by 4th coords(Homogeneous -> 3 coords system)
        camera_proj_normalized = camera_proj / camera_proj[2,:]

        #Get 3d -> 2d mapping
        projected = camera_proj_normalized[:2, filter_idx]

        #Reduce to 320,240 size
        exportW, exportH = DC.exported_scan_size
        camera_proj_sm = np.zeros((5, projected.shape[-1]))        

        camera_proj_sm[0,:] = projected[0,:] * exportW/self.colorW
        camera_proj_sm[1,:] = projected[1,:] * exportH/self.colorH

        return camera_proj_sm[0:2].transpose(), camera_proj[2,:], filter_idx





