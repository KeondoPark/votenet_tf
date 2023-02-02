# coding: utf-8
# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

""" Dataset for object bounding box regression.
An axis aligned bounding box is parameterized by (cx,cy,cz) and (dx,dy,dz)
where (cx,cy,cz) is the center point of the box, dx is the x-axis length of the box.
"""
import os
import sys
import numpy as np
from torch.utils.data import Dataset
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import pc_util
from model_util_scannet import rotate_aligned_boxes

from model_util_scannet import ScannetDatasetConfig, read_matrix
from PIL import Image

DC = ScannetDatasetConfig()
MAX_NUM_OBJ = 64
MEAN_COLOR_RGB = np.array([109.8, 97.2, 83.8])

SCANNET_DIR =os.path.join(BASE_DIR,'scans')



class ScannetDetectionDataset(Dataset):
       
    def __init__(self, split_set='train', num_points=20000,
        use_color=False, use_height=False, augment=False,
        use_painted=False):

        self.use_painted = use_painted
        #if self.use_painted:
        #    self.data_path = os.path.join(BASE_DIR, 'scannet_train_detection_data_painted')
        #else:
        self.data_path = os.path.join(BASE_DIR, 'scannet_train_detection_data')
        all_scan_names = list(set([os.path.basename(x)[0:12] \
            for x in os.listdir(self.data_path) if x.startswith('scene')]))
        if split_set=='all':            
            self.scan_names = all_scan_names
        elif split_set in ['train', 'val', 'test']:
            split_filenames = os.path.join(ROOT_DIR, 'scannet/meta_data',
                'scannetv2_{}.txt'.format(split_set))
            with open(split_filenames, 'r') as f:
                self.scan_names = f.read().splitlines()   
            # remove unavailiable scans
            num_scans = len(self.scan_names)
            self.scan_names = [sname for sname in self.scan_names \
                if sname in all_scan_names]
            print('kept {} scans out of {}'.format(len(self.scan_names), num_scans))
            num_scans = len(self.scan_names)
        else:
            print('illegal split name')
            return

        mesh_vertices_list = []
        instance_labels_list = []
        semantic_labels_list = []
        instance_bboxes_list = []
        for scan_name in self.scan_names:
            mesh_vertices = np.load(os.path.join(self.data_path, scan_name) + '_vert.npy')
            instance_labels = np.load(os.path.join(self.data_path, scan_name) + '_ins_label.npy')
            semantic_labels = np.load(os.path.join(self.data_path, scan_name) + '_sem_label.npy')
            instance_bboxes = np.load(os.path.join(self.data_path, scan_name) + '_bbox.npy')
            mesh_vertices_list.append(mesh_vertices)
            instance_labels_list.append(instance_labels)
            semantic_labels_list.append(semantic_labels)
            instance_bboxes_list.append(instance_bboxes)

        self.mesh_vertices_list = mesh_vertices_list
        self.instance_labels_list = instance_labels_list
        self.semantic_labels_list = semantic_labels_list
        self.instance_bboxes_list = instance_bboxes_list
        
        self.seed = 1111
        
        
        self.num_points = num_points
        self.use_color = use_color        
        self.use_height = use_height
        self.augment = augment
       
    def __len__(self):
        return len(self.scan_names)

    def __getitem__(self, idx):
        """
        Returns a dict with following keys:
            point_clouds: (N,3+C)
            center_label: (MAX_NUM_OBJ,3) for GT box center XYZ
            sem_cls_label: (MAX_NUM_OBJ,) semantic class index
            angle_class_label: (MAX_NUM_OBJ,) with int values in 0,...,NUM_HEADING_BIN-1
            angle_residual_label: (MAX_NUM_OBJ,)
            size_classe_label: (MAX_NUM_OBJ,) with int values in 0,...,NUM_SIZE_CLUSTER
            size_residual_label: (MAX_NUM_OBJ,3)
            box_label_mask: (MAX_NUM_OBJ) as 0/1 with 1 indicating a unique box
            point_votes: (N,3) with votes XYZ
            point_votes_mask: (N,) with 0/1 with 1 indicating the point is in one of the object's OBB.
            scan_idx: int scan index in scan_names list
            pcl_color: unused
        """
        # np.random.seed(self.seed)

        mesh_vertices = self.mesh_vertices_list[idx]
        instance_labels = self.instance_labels_list[idx]
        semantic_labels = self.semantic_labels_list[idx]
        instance_bboxes = self.instance_bboxes_list[idx]   
        scan_name = self.scan_names[idx]

        if not self.use_color:
            point_cloud = mesh_vertices[:,0:3]  #[:,0:3] # do not use color for now
            #pcl_color = mesh_vertices[:,3:6]
        else:
            point_cloud = mesh_vertices[:,0:6] 
            point_cloud[:,3:] = (point_cloud[:,3:]-MEAN_COLOR_RGB)/256.0        

        # ------------------------------- USE HEIGHT ------------------------------ 
        if self.use_height:
            floor_height = np.percentile(point_cloud[:,2],0.99)
            height = point_cloud[:,2] - floor_height
            point_cloud = np.concatenate([point_cloud, np.expand_dims(height, 1)],1)
        

        # ------------------------------- LABELS ------------------------------        
        target_bboxes = np.zeros((MAX_NUM_OBJ, 6))
        target_bboxes_mask = np.zeros((MAX_NUM_OBJ))    
        angle_classes = np.zeros((MAX_NUM_OBJ,))
        angle_residuals = np.zeros((MAX_NUM_OBJ,))
        size_classes = np.zeros((MAX_NUM_OBJ,))
        size_residuals = np.zeros((MAX_NUM_OBJ, 3))
        size_gts = np.zeros((MAX_NUM_OBJ, 3))
        
        point_cloud, choices = pc_util.random_sampling(point_cloud,
            self.num_points, return_choices=True, seed=self.seed)        
        instance_labels = instance_labels[choices]
        semantic_labels = semantic_labels[choices]

        # ------------------------------- POINT PAINTING ------------------------------        
        # Load scene axis alignment matrix
        if self.use_painted:
            meta_file = os.path.join(SCANNET_DIR, scan_name, scan_name + '.txt') # includes axisAlignment info for the train set scans. 
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
            

            alinged_verts = np.ones((point_cloud.shape[0],4))
            alinged_verts[:,:3] = point_cloud[:,:3]
            
            unalign_verts = np.dot(alinged_verts, np.linalg.inv(axis_align_matrix.transpose()))

            exported_scan_dir = os.path.join(BASE_DIR, 'frames_square', scan_name)
            color_intrinsic_file = os.path.join(exported_scan_dir, 'intrinsic', 'intrinsic_color.txt')   
            color_intrinsic = read_matrix(color_intrinsic_file)

            seg_files = os.listdir(os.path.join(BASE_DIR, 'semantic_2d_results', scan_name))
            #color_files = os.listdir(os.path.join(exported_scan_dir, 'color'))    
            
            frame_nums = [int(f[5:-4]) for f in seg_files]
            frame_nums.sort()
            

            point_cloud_recon = np.zeros((point_cloud.shape[0], 3 + 1 + 1 + DC.num_class + 1))
            point_cloud_recon[:,:3] = point_cloud[:,:3]
            point_cloud_recon[:,-1] = point_cloud[:,-1]

            num_img = 3
            sliced = len(frame_nums) // num_img
            frames_selected = []
            
            if len(frame_nums) > num_img:
                for i in range(num_img):
                    if i == num_img - 1:
                        frames_selected.append(np.random.choice(frame_nums[sliced*i:], 1, replace=False)[0])
                    else:
                        frames_selected.append(np.random.choice(frame_nums[sliced*i:sliced*(i+1)], 1, replace=False)[0])                
            else:
                frames_selected = np.random.choice(frame_nums, num_img, replace=True)          
            
            for frame in frames_selected:
                pose_file = os.path.join(exported_scan_dir, 'pose', str(frame) + '.txt')
                depth_file = os.path.join(exported_scan_dir, 'depth-erode', str(frame) + '.png')

                depth_img = Image.open(depth_file)
                depth_np = np.array(depth_img)                          

                # read pose matrix(Rotation and translation)
                pose_matrix = read_matrix(pose_file)   

                sampled_h = np.ones((len(unalign_verts), 4))
                sampled_h[:,:3] = unalign_verts[:,:3]

                # Transform to Camera coords
                camera_coord = np.matmul(np.linalg.inv(pose_matrix), np.transpose(sampled_h))
                camera_proj = np.matmul(color_intrinsic, camera_coord)

                # Get valid points for the image
                x = camera_proj[0,:]
                y = camera_proj[1,:]
                z = camera_proj[2,:]
                filter_idx = np.where((x/z >= 0) & (x/z < colorW) & (y/z >= 0) & (y/z < colorH) & (z > 0))[0]                

                # Normalize by 3rd coords(Homogeneous -> 2 coords system)
                camera_proj_normalized = camera_proj[:,filter_idx] / camera_proj[2, filter_idx]

                # Get 3d -> 2d mapping
                projected = camera_proj_normalized[:2]

                # Reduce to 320,240 size
                camera_proj_sm = np.zeros((2, projected.shape[-1]))

                exportW, exportH = DC.exported_scan_size

                camera_proj_sm[0,:] = projected[0,:] * exportW/colorW
                camera_proj_sm[1,:] = projected[1,:] * exportH/colorH
                
                camera_proj_sm = np.rint(camera_proj_sm)                

                # Get pixel index
                x = np.clip(camera_proj_sm[0,:].astype(np.uint8), 0, exportW-1)
                y = np.clip(camera_proj_sm[1,:].astype(np.uint8), 0, exportH-1)

                depth_filter = np.where(depth_np[y,x]/1000 * 1.1 > camera_proj[2,filter_idx])[0]
                filter_idx2 = np.take(filter_idx, depth_filter)

                x = x[depth_filter]
                y = y[depth_filter]                

                #pred_class = np.load(os.path.join(BASE_DIR, 'semantic_2d_results', scan_name, 'class_' + str(frame) + '.npy'))
                pred_prob = np.load(os.path.join(BASE_DIR, 'semantic_2d_results', scan_name, 'prob_' + str(frame) + '.npy'))

                pred_prob = pred_prob[y, x]
                projected_class = np.argmax(pred_prob, axis=-1)
                
                #projected_class = pred_class[y, x]    

                isPainted = np.where(projected_class == 0, -1, projected_class) # Point belongs to foreground?                    
                
                point_cloud_recon[filter_idx2, 3] = isPainted
                point_cloud_recon[filter_idx2, 4:-1] = pred_prob
            
            point_cloud = point_cloud_recon
        
        #pcl_color = pcl_color[choices]
        
        target_bboxes_mask[0:instance_bboxes.shape[0]] = 1
        target_bboxes[0:instance_bboxes.shape[0],:] = instance_bboxes[:,0:6]
        
        # ------------------------------- DATA AUGMENTATION ------------------------------        
        if self.augment:
            if np.random.random() > 0.5:
                # Flipping along the YZ plane
                point_cloud[:,0] = -1 * point_cloud[:,0]
                target_bboxes[:,0] = -1 * target_bboxes[:,0]                
                
            if np.random.random() > 0.5:
                # Flipping along the XZ plane
                point_cloud[:,1] = -1 * point_cloud[:,1]
                target_bboxes[:,1] = -1 * target_bboxes[:,1]                                
            
            # Rotation along up-axis/Z-axis
            rot_angle = (np.random.random()*np.pi/18) - np.pi/36 # -5 ~ +5 degree
            rot_mat = pc_util.rotz(rot_angle)
            point_cloud[:,0:3] = np.dot(point_cloud[:,0:3], np.transpose(rot_mat))
            target_bboxes = rotate_aligned_boxes(target_bboxes, rot_mat)

        gt_centers = target_bboxes[:, 0:3]
        gt_centers[instance_bboxes.shape[0]:, :] += 1000.0  # padding centers with a large number

        # compute votes *AFTER* augmentation
        # generate votes
        # Note: since there's no map between bbox instance labels and
        # pc instance_labels (it had been filtered 
        # in the data preparation step) we'll compute the instance bbox
        # from the points sharing the same instance label. 
        point_obj_mask = np.zeros(self.num_points)
        point_instance_label = np.zeros(self.num_points) - 1
        for i_instance in np.unique(instance_labels):            
            # find all points belong to that instance
            ind = np.where(instance_labels == i_instance)[0]
            # find the semantic label            
            if semantic_labels[ind[0]] in DC.nyu40ids:
                x = point_cloud[ind,:3]
                center = 0.5*(x.min(0) + x.max(0))
                ilabel = np.argmin(((center - gt_centers) ** 2).sum(-1))
                point_instance_label[ind] = ilabel
                point_obj_mask[ind] = 1.0
         
        
        class_ind = [np.where(DC.nyu40ids == x)[0][0] for x in instance_bboxes[:,-1]]   
        # NOTE: set size class as semantic class. Consider use size2class.
        size_classes[0:instance_bboxes.shape[0]] = class_ind
        size_residuals[0:instance_bboxes.shape[0], :] = \
            target_bboxes[0:instance_bboxes.shape[0], 3:6] - DC.mean_size_arr[class_ind,:]
        size_gts[0:instance_bboxes.shape[0], :] = target_bboxes[0:instance_bboxes.shape[0], 3:6]
            
        ret_dict = {}
        ret_dict['point_clouds'] = point_cloud.astype(np.float32)
        ret_dict['center_label'] = gt_centers.astype(np.float32)
        ret_dict['heading_class_label'] = angle_classes.astype(np.int64)
        ret_dict['heading_residual_label'] = angle_residuals.astype(np.float32)
        ret_dict['size_class_label'] = size_classes.astype(np.int64)
        ret_dict['size_residual_label'] = size_residuals.astype(np.float32)        
        target_bboxes_semcls = np.zeros((MAX_NUM_OBJ))                                
        target_bboxes_semcls[0:instance_bboxes.shape[0]] = \
            [DC.nyu40id2class[x] for x in instance_bboxes[:,-1][0:instance_bboxes.shape[0]]]                
        ret_dict['sem_cls_label'] = target_bboxes_semcls.astype(np.int64)
        ret_dict['box_label_mask'] = target_bboxes_mask.astype(np.int64)
        ret_dict['point_obj_mask'] = point_obj_mask.astype(np.int64)
        ret_dict['point_instance_label'] = point_instance_label.astype(np.int64)
        ret_dict['size_gts'] = size_gts.astype(np.float32)
        ret_dict['scan_idx'] = np.array(idx).astype(np.int64)
        #ret_dict['pcl_color'] = pcl_color
        return ret_dict
        
############# Visualizaion ########

def viz_votes(pc, point_votes, point_votes_mask, name=''):
    """ Visualize point votes and point votes mask labels
    pc: (N,3 or 6), point_votes: (N,9), point_votes_mask: (N,)
    """
    inds = (point_votes_mask==1)
    pc_obj = pc[inds,0:3]
    pc_obj_voted1 = pc_obj + point_votes[inds,0:3]    
    pc_util.write_ply(pc_obj, 'pc_obj{}.ply'.format(name))
    pc_util.write_ply(pc_obj_voted1, 'pc_obj_voted1{}.ply'.format(name))
    
def viz_obb(pc, label, mask, angle_classes, angle_residuals,
    size_classes, size_residuals, name=''):
    """ Visualize oriented bounding box ground truth
    pc: (N,3)
    label: (K,3)  K == MAX_NUM_OBJ
    mask: (K,)
    angle_classes: (K,)
    angle_residuals: (K,)
    size_classes: (K,)
    size_residuals: (K,3)
    """
    oriented_boxes = []
    K = label.shape[0]
    for i in range(K):
        if mask[i] == 0: continue
        obb = np.zeros(7)
        obb[0:3] = label[i,0:3]
        heading_angle = 0 # hard code to 0
        box_size = DC.mean_size_arr[size_classes[i], :] + size_residuals[i, :]
        obb[3:6] = box_size
        obb[6] = -1 * heading_angle
        print(obb)        
        oriented_boxes.append(obb)
    pc_util.write_oriented_bbox(oriented_boxes, 'gt_obbs{}.ply'.format(name))
    pc_util.write_ply(label[mask==1,:], 'gt_centroids{}.ply'.format(name))

    
if __name__=='__main__': 
    dset = ScannetDetectionDataset(use_height=True, num_points=40000, use_painted=True)
    for i_example in range(1):
        example = dset.__getitem__(20)
        pc = example['point_clouds']
        print(pc.shape)
        new_pc = pc[:,:3]
        labels = np.argmax(pc[:,4:-1], axis=-1)

        #pc_util.write_ply(example['point_clouds'], 'pc_{}.ply'.format(i_example))    
        pc_util.write_ply_color(pc, labels, 'pc_{}.ply'.format(i_example))
        '''
        viz_votes(example['point_clouds'], example['vote_label'],
            example['vote_label_mask'],name=i_example)    
        viz_obb(pc=example['point_clouds'], label=example['center_label'],
            mask=example['box_label_mask'],
            angle_classes=None, angle_residuals=None,
            size_classes=example['size_class_label'], size_residuals=example['size_residual_label'],
            name=i_example)
        '''