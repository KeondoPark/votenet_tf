import os
import sys
import numpy as np
import sys
import cv2
import argparse
from PIL import Image
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils/'))
import pc_util
import sunrgbd_utils
import sunrgbd_data

def get_voxel_stats(idx_filename):
  print("Starting get_voxel_stats")
  dataset = sunrgbd_data.sunrgbd_object('/home/aiot/sunrgbd_trainval', split='training', use_v1=True)
  print("Loading sunrgbd object")
  data_idx_list = [int(line.rstrip()) for line in open(idx_filename)]

  print("Before for loop")

  NUM_SLICE = 64
  cnt = 0
  adj = [[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]]

  for data_idx in data_idx_list:
    
    grid = np.zeros((NUM_SLICE,NUM_SLICE,NUM_SLICE))

    pc_upright_depth = dataset.get_depth(data_idx)
    print("Shape:", pc_upright_depth.shape)

    num_pts = pc_upright_depth.shape[0]   
    
    max_x = max(pc_upright_depth[:,0])
    max_y = max(pc_upright_depth[:,1])
    max_z = max(pc_upright_depth[:,2])
    
    min_x = min(pc_upright_depth[:,0])
    min_y = min(pc_upright_depth[:,1])
    min_z = min(pc_upright_depth[:,2])

    len_x = (max_x - min_x) / NUM_SLICE
    len_y = (max_y - min_y) / NUM_SLICE
    len_z = (max_z - min_z) / NUM_SLICE

    print(max_x, max_y, max_z)
    print(min_x, min_y, min_z)
    print(len_x, len_y, len_z)


    if cnt == 0:
      with open('sample.ply','w') as file:     
        pt_cnt = 0   
        for pt in pc_upright_depth:
          pt_cnt += 1
          x = pt[0]
          y = pt[1]
          z = pt[2]

          file.write(str(x) + ' ' + str(y) + ' ' + str(z) + '\n')

          idx_x = int((x - min_x) / len_x) if int((x - min_x) / len_x) < NUM_SLICE else NUM_SLICE - 1
          idx_y = int((y - min_y) / len_y) if int((y - min_y) / len_y) < NUM_SLICE else NUM_SLICE - 1
          idx_z = int((z - min_x) / len_z) if int((z - min_z) / len_z) < NUM_SLICE else NUM_SLICE - 1

          grid[idx_x, idx_y, idx_z] = 1        

          if idx_x > 10 and idx_y == 0 and idx_z == 0:
            print(x, y, z)
            print(idx_x, idx_y, idx_z)
        
        
    else:
      for pt in pc_upright_depth:
        x = pt[0]
        y = pt[1]
        z = pt[2]

        idx_x = int((x - min_x) / len_x) if int((x - min_x) / len_x) < NUM_SLICE else NUM_SLICE - 1
        idx_y = int((y - min_y) / len_y) if int((y - min_y) / len_y) < NUM_SLICE else NUM_SLICE - 1
        idx_z = int((z - min_x) / len_z) if int((z - min_z) / len_z) < NUM_SLICE else NUM_SLICE - 1

        grid[idx_x, idx_y, idx_z] = 1
        

    
    adj_stat = [0,0,0,0,0,0,0]

    for i in range(NUM_SLICE):
      for j in range(NUM_SLICE):
        for k in range(NUM_SLICE):          
          if grid[i,j,k] == 1:
            adj_cnt = 0
            for item in adj:
              if i + item[0] < NUM_SLICE and j + item[1] < NUM_SLICE and k + item[2] < NUM_SLICE and grid[i+item[0], j+item[1], k+item[2]] == 1:
                adj_cnt += 1

            adj_stat[adj_cnt] += 1

    if cnt == 0:
      with open('sample_filtered.ply','w') as file:        
        for pt in pc_upright_depth:
          x = pt[0]
          y = pt[1]
          z = pt[2]

          idx_x = int((x - min_x) / len_x) if int((x - min_x) / len_x) < NUM_SLICE else NUM_SLICE - 1
          idx_y = int((y - min_y) / len_y) if int((y - min_y) / len_y) < NUM_SLICE else NUM_SLICE - 1
          idx_z = int((z - min_x) / len_z) if int((z - min_z) / len_z) < NUM_SLICE else NUM_SLICE - 1
          
          adj_cnt = 0
          
          for item in adj:
              if idx_x + item[0] < NUM_SLICE and idx_y + item[1] < NUM_SLICE and idx_z + item[2] < NUM_SLICE and grid[idx_x+item[0], idx_y+item[1], idx_z+item[2]] == 1:
                adj_cnt += 1
          
          if adj_cnt < 5 and adj_cnt > 1:
            file.write(str(x) + ' ' + str(y) + ' ' + str(z) + '\n')

    print("How many neighbor voxels are non-empty:", adj_stat)
    print(np.sum(grid))
    print("Non empty ratio:", np.sum(grid) / (NUM_SLICE ** 3))
    cnt += 1

    if cnt > 0:
      break

if __name__ == '__main__':
  print("Starting")
  
  get_voxel_stats('/home/aiot/sunrgbd_trainval/train_data_idx.txt')