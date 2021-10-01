// Copyright (c) Facebook, Inc. and its affiliates.
// 
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include "ball_query.h"
#include "utils.h"

void query_ball_point_kernel_wrapper(int b, int n, int m, float radius,
                                     int nsample, const float *new_xyz,
                                     const float *xyz, int *idx);

int binarySearch(const float arr[], int l, int r, float x, int lu_opt);

at::Tensor ball_query(at::Tensor new_xyz, at::Tensor xyz, const float radius,
                      const int nsample) {
  CHECK_CONTIGUOUS(new_xyz);
  CHECK_CONTIGUOUS(xyz);
  CHECK_IS_FLOAT(new_xyz);
  CHECK_IS_FLOAT(xyz);

  if (new_xyz.type().is_cuda()) {
    CHECK_CUDA(xyz);
  }

  at::Tensor idx =
      torch::zeros({new_xyz.size(0), new_xyz.size(1), nsample},
                   at::device(new_xyz.device()).dtype(at::ScalarType::Int));

  if (new_xyz.type().is_cuda()) {
    query_ball_point_kernel_wrapper(xyz.size(0), xyz.size(1), new_xyz.size(1),
                                    radius, nsample, new_xyz.data<float>(),
                                    xyz.data<float>(), idx.data<int>());
  } else {
    AT_CHECK(false, "CPU not supported");
  }

  return idx;
}




/*
  Assume batch_distances is sorted
  arg_sort is sorted index of batch_distances
*/

at::Tensor ball_query_nocuda(at::Tensor new_xyz, at::Tensor xyz, float radius,
                      const int nsample, at::Tensor batch_distances, at::Tensor inds,
                      at::Tensor arg_sort) {

  //printf("Strting ball_query_nocuda...\n");
  
  at::Tensor idx = torch::zeros({new_xyz.size(0), nsample},
                   at::device(new_xyz.device()).dtype(at::ScalarType::Int));
  
  int n = xyz.size(0);
  int m = new_xyz.size(0);

  float r2 = radius * radius;

  const float* center_points = (float *) new_xyz.data<float>();
  const float* points = (float *) xyz.data<float>();
  const float* batch_d = (float *) batch_distances.data<float>();
  //const float* batch_d2 = (float *) batch_distances2.data<float>();
  const int* inds_centers = (int *) inds.data<int>();
  const int* inds_sort = (int *) arg_sort.data<int>();
  //const int* inds_sort2 = (int *) arg_sort2.data<int>();

  int* output = idx.data<int>();

  for (int m_idx = 0; m_idx < m; ++m_idx){
    float center_x = center_points[m_idx * 3 + 0];
    float center_y = center_points[m_idx * 3 + 1];
    float center_z = center_points[m_idx * 3 + 2];

    int i = inds_centers[m_idx];
    int j = -1;
    int j2 = -1;
    
    for (int n_idx = 0; n_idx < n; ++n_idx){
      if (inds_sort[n_idx] == i)
        j = n_idx;        
      
      //if (inds_sort2[n_idx] == i)
      //  j2 = n_idx;        
      
      //if (j >= 0 && j2 >= 0)
      if (j >= 0)
        break;
    }

        
    float d = batch_d[j];
    //float d2 = batch_d2[j2];

    //printf("Before binary search...\n");

    int jmin = -1;
    int jmax = -1;

    if (batch_d[0] >= d - radius){
      jmin = 0;
    } else {
      jmin = binarySearch(batch_d, 0, j, d-radius, 0);    
    }

    if (batch_d[n-1] <= d + radius){
      jmax = n -1;
    } else {   
      jmax = binarySearch(batch_d, j, n-1, d+radius, 1);
    }
    //int jmin2 = binarySearch(batch_d2, 0, j2, d2-radius, 0);
    //int jmax2 = binarySearch(batch_d2, j2, n-1, d2+radius, 1);

    //printf("jmin: %d, jmax: %d\n", jmin, jmax);

    if (jmin < 0) jmin = 0;
    if (jmax < 0) jmax = n-1;
    //if (jmin2 < 0) jmin2 = 0;
    //if (jmax2 < 0) jmax2 = n-1;
    
    int cnt = 0;

    for (int i1 = jmin; i1 <= jmax; ++i1){
      int n_idx = inds_sort[i1];
      //for (int i2 = jmin2; i2 <= jmax2; ++i2){
        //if (n_idx == inds_sort2[i2]){
          float x = points[n_idx*3 + 0];
          float y = points[n_idx*3 + 1];
          float z = points[n_idx*3 + 2];

          float dist = (x - center_x) * (x - center_x) + (y - center_y) * (y - center_y) + (z - center_z) * (z - center_z); 
          
          if (dist <= r2){
            if (cnt == 0){
              for (int k = 0; k < nsample; ++k){
                output[m_idx * nsample + k] = n_idx;  
              }            
            }
            output[m_idx * nsample + cnt] = n_idx;
            cnt += 1;                    
          }
          //break;
        //}
      //}
      if (cnt >= nsample) break;      
    }

    if (cnt == 0){
      //printf("No points in the ball. m_idx: %d, radius: %f\n", m_idx, radius);
    }    
  } 

  return idx;
}

int binarySearch(const float arr[], int l, int r, float x, int lu_opt) 
{ 
  //lu_opt = 0 to find the largest of smaller than x
  //lu_opt = 1 to find the smallest of larger than x
    if (r >= l) { 
        int mid = l + (r - l) / 2; 
  
        // If the element is present at the middle 
        // itself 
        if (arr[mid] == x) 
            return mid; 
  
        // If element is smaller than mid, then 
        // it can only be present in left subarray 
        if (arr[mid] > x){
          if (arr[mid - 1] < x){
            return mid - 1 + lu_opt;
          } else {
            return binarySearch(arr, l, mid - 1, x, lu_opt); 
          }
        }
  
        // Else the element can only be present 
        // in right subarray
        if (arr[mid + 1] > x){
          return mid + lu_opt;
        }

        return binarySearch(arr, mid + 1, r, x, lu_opt); 
    } 
  
    // We reach here when element is not 
    // present in array 
    return -1; 
} 