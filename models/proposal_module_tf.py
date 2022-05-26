# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import tensorflow as tf
from tensorflow.keras import layers

import numpy as np
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'pointnet2'))
from pointnet2_modules_tf import PointnetSAModuleVotes, SamplingAndGrouping
import pointnet2_utils_tf
from tf_ops.sampling import tf_sampling
import tf_utils

def decode_scores(net, end_points, num_class, num_heading_bin, num_size_cluster, mean_size_arr, num_proposal):
    """
    Args:
        net: (B, num_proposal, 2+num_heading_bin*2+num_size_cluster*4)
        end_points: dictionary of 'aggregated_vote_xyz', 'aggregated_vote_inds', 
                                  'fp2_xyz', 'fp2_features', 'seed_inds', 'seed_xyz', 'seed_features',
                                  'vote_xyz', 'vote_features'
    return:
        Add 'objectness_scores', 'center', 'heading_scores', 'heading_residuals_normalized', 'heading_residuals', 
            'size_scores', 'size_residuals_normalized', 'size_residuals', 'sem_cls_scores' to end_points
                
    """


    objectness_scores = net[:,:,0:2]
    end_points['objectness_scores'] = objectness_scores    

    NH = num_heading_bin
    NC = num_size_cluster
    
    heading_scores = net[:,:, 2:2+NH]
    size_scores = net[:,:, 2 + NH:2 + NH + NC]

    heading_residuals_normalized = net[:,:,2 + NH + NC:2 + NH + NC + NH]
    pi = tf.constant(3.14159265359, dtype=tf.float32)
    end_points['heading_scores'] = heading_scores # B x num_proposal x num_heading_bin    

    end_points['heading_residuals_normalized'] = heading_residuals_normalized # Bxnum_proposalxnum_heading_bin (should be -1 to 1)
    end_points['heading_residuals'] = heading_residuals_normalized * (pi/tf.cast(NH,tf.float32)) # B x num_proposal x num_heading_bin

    size_residuals = net[:,:,2 + NH*2 + NC:2 + NH*2 + NC*4]
    size_residuals_normalized = layers.Reshape((num_proposal, NC, 3))(size_residuals) # B x num_proposal x num_size_cluster x 3    
    end_points['size_scores'] = size_scores
    end_points['size_residuals_normalized'] = size_residuals_normalized
    end_points['size_residuals'] = size_residuals_normalized * tf.expand_dims(tf.expand_dims(tf.cast(mean_size_arr,dtype=tf.float32), axis=0), axis=0)

    sem_cls_scores = net[:,:,2+num_heading_bin*2+num_size_cluster*4:] # B x num_proposal x 10
    end_points['sem_cls_scores'] = sem_cls_scores    
    
    return end_points


class ProposalModule(layers.Layer):
    def __init__(self, num_class, num_heading_bin, num_size_cluster, mean_size_arr, num_proposal, sampling, seed_feat_dim, model_config):
        super().__init__() 
        """
        num_class: Number of classes
        num_heading_bin: Number of heading bin, 12 (30 degree per each bin)
        num_size_cluster: Number of size cluster, 10
        mean_size_arr: Average size of objects in each class
        num_proposal: Number of proposals
        sampling: sampling type
        seed_feat_dim: Number of feature dimensions of votes                
        model_config:
            use_tflite: boolean, If True, use tflite
            tflite_name: string, The name of tflite file
            q_gran: Quantization granularity of the last layer in proposal module
            activation: relu or relu6

        """
        self.num_class = num_class
        self.num_heading_bin = num_heading_bin
        self.num_size_cluster = num_size_cluster
        self.mean_size_arr = mean_size_arr
        self.num_proposal = num_proposal
        self.sampling = sampling
        self.seed_feat_dim = seed_feat_dim
        self.npoint = self.num_proposal
        self.nsample = 16
        self.vote_aggregation = SamplingAndGrouping( 
                npoint=self.num_proposal,
                radius=0.3,
                nsample=self.nsample,                
                use_xyz=True,
                normalize_xyz=True
            )
        self.use_tflite = model_config['use_tflite']
        #self.sep_coords = model_config['sep_coords']
        self.q_gran = model_config['q_gran']
        
        mlp_spec = [self.seed_feat_dim, 128, 128, 128]
        mlp_spec[0] += 3  

        if self.use_tflite:
            self.use_edgetpu = model_config['use_edgetpu']
            tflite_folder = model_config['tflite_folder']
            tflite_file = 'va_quant'
            if self.q_gran != 'semantic':
                tflite_file += '_' + self.q_gran

            if self.use_edgetpu: 
                tflite_file += '_edgetpu'           
                from pycoral.utils.edgetpu import make_interpreter            
                self.interpreter = make_interpreter(os.path.join(ROOT_DIR,os.path.join(tflite_folder, tflite_file + '.tflite')))
            else:                
                self.interpreter = tf.lite.Interpreter(model_path=os.path.join(ROOT_DIR,os.path.join(tflite_folder, tflite_file + '.tflite')))                             
            self.interpreter.allocate_tensors()

            # Get input and output tensors.
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
        else:
            self.mlp_module = tf_utils.SharedMLP(mlp_spec, bn=True, input_shape=[self.npoint, self.nsample, 3+mlp_spec[0]])        
            self.max_pool = layers.MaxPooling2D(pool_size=(1, self.nsample), strides=1, data_format="channels_last")
            
            # Object proposal/detection
            # Objectness scores (2), center residual (3),
            # heading class+residual (num_heading_bin*2), size class+residual(num_size_cluster*4)            
            #### Changed to Conv2D to be compatible with EdgeTPU compiler
            self.conv1 = layers.Conv2D(filters=128, kernel_size=1) 
            self.conv2 = layers.Conv2D(filters=128, kernel_size=1) 
            
            # 2: objectness_scores, 3: offset(From vote to center), score/residuals for num_heading_bin(12),
            # score/(H,W,C) for size, Class score            
            self.conv3 = layers.Conv2D(filters=2+3+num_heading_bin*2+num_size_cluster*4+self.num_class, kernel_size=1)
            
            self.bn1 = layers.BatchNormalization(axis=-1)
            self.bn2 = layers.BatchNormalization(axis=-1)

            act = model_config['activation'] if 'activation' in model_config['activation'] else 'relu6'
            if act == 'relu6':
                maxval = 6
            else:
                maxval = None

            self.relu1 = layers.ReLU(maxval)
            self.relu2 = layers.ReLU(maxval)

    def call(self, xyz, features, end_points):
        """
        Args:
            xyz: (B,K,3)
            features: (B,K,C)
        Returns:
            scores: (B,num_proposal,2+3+NH*2+NS*4) 
        """

        if self.sampling == 'vote_fps':
            # Farthest point sampling (FPS) on votes            
            xyz, fps_inds, va_grouped_features, _ = self.vote_aggregation(xyz, isPainted=None, features=features) #NoMLP version            
            end_points['va_grouped_features'] = va_grouped_features            
            sample_inds = fps_inds
        elif self.sampling == 'seed_fps': 
            # FPS on seed and choose the votes corresponding to the seeds
            # This gets us a slightly better coverage of *object* votes than vote_fps (which tends to get more cluster votes)
            sample_inds = tf_sampling.farthest_point_sample(self.num_proposal, seed_xyz)
            xyz, features, fps_inds, _, _ = self.vote_aggregation(xyz, features, sample_inds)
        elif self.sampling == 'random':
            # Random sampling from the votes
            num_seed = seed_xyz.shape[1]
            batch_size = seed_xyz.shape[0]
            seed_pts = np.random.randint(low=0, high=num_seed, size=(batch_size, self.num_proposal)).astype('float32')
            sample_inds = tf.constant(seed_pts)
            xyz, features, fps_inds, _, _ = self.vote_aggregation(xyz, features, sample_inds)
        else:
            log_string('Unknown sampling strategy: %s. Exiting!'%(self.sampling))
            exit()        
        
        if self.use_tflite:
            self.interpreter.set_tensor(self.input_details[0]['index'], va_grouped_features)            
            if len(self.input_details) > 1:
                self.interpreter.set_tensor(self.input_details[1]['index'], xyz)
            self.interpreter.invoke()
            if self.q_gran == 'semantic':
                offset = self.interpreter.get_tensor(self.output_details[0]['index'])
                offset = tf.convert_to_tensor(offset)
                center = xyz + offset
                net2 = self.interpreter.get_tensor(self.output_details[1]['index']) 
                net3 = self.interpreter.get_tensor(self.output_details[2]['index'])                 

                obj_score = net2[:,:,:2]
                head_score = net2[:,:,2:2+self.num_heading_bin]
                cluster_score = net2[:,:,2+self.num_heading_bin:2+self.num_heading_bin+self.num_size_cluster]
                class_score = net2[:,:,2+self.num_heading_bin+self.num_size_cluster:]

                head_residual = net3[:,:,:self.num_heading_bin]
                cluster_residual = net3[:,:,self.num_heading_bin:]

                net = np.concatenate([obj_score, head_score, cluster_score, head_residual, cluster_residual, class_score], axis=-1)

                net = tf.convert_to_tensor(net)                  

            elif self.q_gran == 'channel':
                out = []
                for i in range(2+3+self.num_heading_bin*2+self.num_size_cluster*4+self.num_class):
                    out.append(self.interpreter.get_tensor(self.output_details[i]['index']))

                offset = layers.Concatenate(axis=-1)(out[:3])
                offset = layers.Reshape((self.npoint, 3))(offset)
                net = layers.Concatenate(axis=-1)(out[3:])
                net = layers.Reshape((self.npoint, net.shape[-1]))(net)                

                center = xyz + offset

            elif self.q_gran == 'group':
                out = []
                for i in range(3):
                    out.append(self.interpreter.get_tensor(self.output_details[i]['index']))

                net = layers.Concatenate(axis=-1)(out)
                net = layers.Reshape((self.npoint, net.shape[-1]))(net)
                offset = net[:,:,0:3]
                net = net[:,:,3:]                
                center = xyz + offset  

            else:
                net = self.interpreter.get_tensor(self.output_details[0]['index']) 
                net = tf.convert_to_tensor(net)
                offset = net[:,:,0:3]
                net = net[:,:,3:]                               
                center = xyz + offset            
        else:
            new_features = self.mlp_module(va_grouped_features)
            features = self.max_pool(new_features)
            
            # --------- PROPOSAL GENERATION ---------
            net = self.relu1(self.bn1(self.conv1(features)))
            net = self.relu2(self.bn2(self.conv2(net)))
            net = self.conv3(net)
            offset = net[:,:,:,0:3]
            net = net[:,:,:,3:]

            offset = layers.Reshape((self.npoint, 3))(offset)                
            center = xyz + offset            
            net = layers.Reshape((self.npoint, net.shape[-1]))(net)            
            

        # Return from expanded shape
        end_points['aggregated_vote_xyz'] = xyz # (batch_size, num_proposal, 3)
        end_points['aggregated_vote_inds'] = sample_inds # (batch_size, num_proposal,) # should be 0,1,2,...,num_proposal
        end_points['center'] = center # (batch_size, num_proposal, 3)

        end_points = decode_scores(net, end_points, self.num_class, self.num_heading_bin, self.num_size_cluster, self.mean_size_arr, self.num_proposal)        


        return end_points

if __name__=='__main__':
    sys.path.append(os.path.join(ROOT_DIR, 'sunrgbd'))
    from sunrgbd_detection_dataset_tf import SunrgbdDetectionVotesDataset, DC
    net = ProposalModule(DC.num_class, DC.num_heading_bin,
        DC.num_size_cluster, DC.mean_size_arr,
        128, 'seed_fps')
    end_points = {'seed_xyz': tf.constant(np.random.random((8,1024,3)).astype('float32'))}
    xyz = tf.constant(np.random.random((8,1024,3)).astype('float32'))
    features = tf.constant(np.random.random((8,256,1024)).astype('float32'))

    out = net(xyz, features, end_points)
    for key in out:
        print(key, out[key].shape)

