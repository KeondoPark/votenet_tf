// Copyright (c) Facebook, Inc. and its affiliates.
// 
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include "sampling.h"
#include "utils.h"

using namespace std;

void gather_points_kernel_wrapper(int b, int c, int n, int npoints,
                                  const float *points, const int *idx,
                                  float *out);
void gather_points_grad_kernel_wrapper(int b, int c, int n, int npoints,
                                       const float *grad_out, const int *idx,
                                       float *grad_points);

void furthest_point_sampling_kernel_wrapper(int b, int n, int m,
                                            const float *dataset, float *temp,
                                            int *idxs);



template <typename T>
std::vector<size_t> sort_indexes(const std::vector<T> &v) {

  // initialize original index locations
  vector<size_t> idx(v.size());
  iota(idx.begin(), idx.end(), 0);

  // sort indexes based on comparing values in v
  // using std::stable_sort instead of std::sort
  // to avoid unnecessary index re-orderings
  // when v contains elements of equal values 
  stable_sort(idx.begin(), idx.end(),
       [&v](size_t i1, size_t i2) {return v[i1] < v[i2];});

  return idx;
}

at::Tensor gather_points(at::Tensor points, at::Tensor idx) {
  CHECK_CONTIGUOUS(points);
  CHECK_CONTIGUOUS(idx);
  CHECK_IS_FLOAT(points);
  CHECK_IS_INT(idx);

  if (points.type().is_cuda()) {
    CHECK_CUDA(idx);
  }

  at::Tensor output =
      torch::zeros({points.size(0), points.size(1), idx.size(1)},
                   at::device(points.device()).dtype(at::ScalarType::Float));

  if (points.type().is_cuda()) {
    gather_points_kernel_wrapper(points.size(0), points.size(1), points.size(2),
                                 idx.size(1), points.data<float>(),
                                 idx.data<int>(), output.data<float>());
  } else {
    AT_CHECK(false, "CPU not supported");
  }

  return output;
}

at::Tensor gather_points_grad(at::Tensor grad_out, at::Tensor idx,
                              const int n) {
  CHECK_CONTIGUOUS(grad_out);
  CHECK_CONTIGUOUS(idx);
  CHECK_IS_FLOAT(grad_out);
  CHECK_IS_INT(idx);

  if (grad_out.type().is_cuda()) {
    CHECK_CUDA(idx);
  }

  at::Tensor output =
      torch::zeros({grad_out.size(0), grad_out.size(1), n},
                   at::device(grad_out.device()).dtype(at::ScalarType::Float));

  if (grad_out.type().is_cuda()) {
    gather_points_grad_kernel_wrapper(grad_out.size(0), grad_out.size(1), n,
                                      idx.size(1), grad_out.data<float>(),
                                      idx.data<int>(), output.data<float>());
  } else {
    AT_CHECK(false, "CPU not supported");
  }

  return output;
}

at::Tensor furthest_point_sampling(at::Tensor points, const int nsamples) {
  CHECK_CONTIGUOUS(points);
  CHECK_IS_FLOAT(points);

  at::Tensor output =
      torch::zeros({points.size(0), nsamples},
                   at::device(points.device()).dtype(at::ScalarType::Int));

  at::Tensor tmp =
      torch::full({points.size(0), points.size(1)}, 1e10,
                  at::device(points.device()).dtype(at::ScalarType::Float));

  if (points.type().is_cuda()) {
    furthest_point_sampling_kernel_wrapper(
        points.size(0), points.size(1), nsamples, points.data<float>(),
        tmp.data<float>(), output.data<int>());
  } else {
    AT_CHECK(false, "CPU not supported");
  }

  return output;
}


std::vector<at::Tensor> sampling_from_avg(at::Tensor points, const int m) {

  //printf("Strting distance_from_avg...\n");

  int B = points.size(0);
  int n = points.size(1);

  at::Tensor dist =
      torch::zeros({B, n}, at::device(points.device()).dtype(at::ScalarType::Float));

  at::Tensor idx =
      torch::zeros({B, m}, at::device(points.device()).dtype(at::ScalarType::Int));

  const float* dataset = (float *) points.data<float>();
  float* distances = dist.data<float>();
  int* inds = (int *) idx.data<int>();

  for (int b_idx = 0; b_idx < B; ++b_idx){
    float avgx = 0;
    float avgy = 0;
    float avgz = 0;    

    for (int n_idx = 0; n_idx < n; ++n_idx){
      float x = dataset[b_idx * n * 3 + n_idx*3 + 0];
      float y = dataset[b_idx * n * 3 + n_idx*3 + 1];
      float z = dataset[b_idx * n * 3 + n_idx*3 + 2];

      avgx += x;
      avgy += y;
      avgz += z;
    }

    avgx = avgx / n;
    avgy = avgy / n;
    avgz = avgz / n;

    //Count the number of points in each quadrant
    int cnt_1q = 0;
    int cnt_2q = 0;
    int cnt_3q = 0;
    int cnt_4q = 0;
    int cnt_5q = 0;
    int cnt_6q = 0;
    int cnt_7q = 0;
    int cnt_8q = 0;

    int pts_1q[m];
    int pts_2q[m];
    int pts_3q[m];
    int pts_4q[m];
    int pts_5q[m];
    int pts_6q[m];
    int pts_7q[m];
    int pts_8q[m];

    int pts_whichq[n];
    
    int m_idx = 0;
    int step = (int) (n / m);

    //Calculate distance from average point and quadrants based on the average point.
    for (int n_idx = 0; n_idx < n; ++n_idx){
      float x = dataset[b_idx * n * 3 + n_idx*3 + 0];
      float y = dataset[b_idx * n * 3 + n_idx*3 + 1];
      float z = dataset[b_idx * n * 3 + n_idx*3 + 2];
      
      distances[b_idx * n + n_idx] = (x - avgx) * (x - avgx) + (y - avgy) * (y - avgy) + (z - avgz) * (z - avgz);    

      if (x > avgx && y > avgy && z > avgz){
        pts_whichq[n_idx] = 1;
      } else if (x < avgx && y > avgy && z > avgz){
        pts_whichq[n_idx] = 2;
      } else if (x < avgx && y < avgy && z > avgz){
        pts_whichq[n_idx] = 3;
      } else if (x > avgx && y < avgy && z > avgz){
        pts_whichq[n_idx] = 4;
      } else if (x > avgx && y > avgy && z < avgz){
        pts_whichq[n_idx] = 5;
      } else if (x < avgx && y > avgy && z < avgz){
        pts_whichq[n_idx] = 6;
      } else if (x < avgx && y < avgy && z < avgz){
        pts_whichq[n_idx] = 7;
      } else if (x > avgx && y < avgy && z < avgz){
        pts_whichq[n_idx] = 8;
      }  
    }

    std::vector<float> v(distances + b_idx * n, distances + (b_idx+1) * n);
    std::vector<long unsigned int> w = sort_indexes(v);

    for (auto i: w) {
      if (pts_whichq[w[i]] == 1){
        if (cnt_1q % step == 0 && m_idx < m){
          inds[b_idx * m + m_idx] = w[i];
          m_idx++;
        }
        cnt_1q++;
      } else if (pts_whichq[w[i]] == 2){
        if (cnt_2q % step == 0 && m_idx < m){
          inds[b_idx * m + m_idx] = w[i];
          m_idx++;
        }
        cnt_2q++;
      } else if (pts_whichq[w[i]] == 3){
        if (cnt_3q % step == 0 && m_idx < m){
          inds[b_idx * m + m_idx] = w[i];
          m_idx++;
        }
        cnt_3q++;
      } else if (pts_whichq[w[i]] == 4){
        if (cnt_4q % step == 0 && m_idx < m){
          inds[b_idx * m + m_idx] = w[i];
          m_idx++;
        }
        cnt_4q++;
      }  else if (pts_whichq[w[i]] == 5){
        if (cnt_5q % step == 0 && m_idx < m){
          inds[b_idx * m + m_idx] = w[i];
          m_idx++;
        }
        cnt_5q++;
      } else if (pts_whichq[w[i]] == 6){
        if (cnt_6q % step == 0 && m_idx < m){
          inds[b_idx * m + m_idx] = w[i];
          m_idx++;
        }
        cnt_6q++;
      } else if (pts_whichq[w[i]] == 7){
        if (cnt_7q % step == 0 && m_idx < m){
          inds[b_idx * m + m_idx] = w[i];
          m_idx++;
        }
        cnt_7q++;
      } else if (pts_whichq[w[i]] == 8){
        if (cnt_8q % step == 0 && m_idx < m){
          inds[b_idx * m + m_idx] = w[i];
          m_idx++;
        }
        cnt_8q++;
      }
      
    }
  }
  
  return {idx, dist};
}

std::vector<at::Tensor> sampling_from_ending_points(at::Tensor points, const int m) {

  //printf("Strting distance_from_avg...\n");

  int B = points.size(0);
  int n = points.size(1);

  at::Tensor dist =
      torch::zeros({B, n}, at::device(points.device()).dtype(at::ScalarType::Float));

  at::Tensor idx =
      torch::zeros({B, m}, at::device(points.device()).dtype(at::ScalarType::Int));

  const float* dataset = (float *) points.data<float>();
  float* distance_avg = dist.data<float>();
  int* inds = (int *) idx.data<int>();

  for (int b_idx = 0; b_idx < B; ++b_idx){

    //find the farthest point from the point
    float far1_x = dataset[b_idx * n * 3 + 0];
    float far1_y = dataset[b_idx * n * 3 + 1];
    float far1_z = dataset[b_idx * n * 3 + 2];

    float farthest = 0;
    int farthest_idx1 = 1;

    //Find the average point
    float avgx = far1_x;
    float avgy = far1_y;
    float avgz = far1_z;    

    for (int n_idx = 1; n_idx < n; ++n_idx){
      float x = dataset[b_idx * n * 3 + n_idx*3 + 0];
      float y = dataset[b_idx * n * 3 + n_idx*3 + 1];
      float z = dataset[b_idx * n * 3 + n_idx*3 + 2];

      float dist_far1 = (x - far1_x) * (x - far1_x) + (y - far1_y) * (y - far1_y) + (z - far1_z) * (z - far1_z);

      avgx += x;
      avgy += y;
      avgz += z;
      
      if (dist_far1 > farthest){
        farthest = dist_far1;
        farthest_idx1 = n_idx;
      }
    }

    far1_x = dataset[b_idx * n * 3 + farthest_idx1*3 + 0];
    far1_y = dataset[b_idx * n * 3 + farthest_idx1*3 + 1];
    far1_z = dataset[b_idx * n * 3 + farthest_idx1*3 + 2];

    
    avgx = avgx / n;
    avgy = avgy / n;
    avgz = avgz / n;

    float distance1[n];
    farthest = 0;
    int farthest_idx2 = 0;

    // FInd the farthest point from the first farthest point
    // Also calculate distance from average point and the first farthest point
    for (int n_idx = 0; n_idx < n; ++n_idx){
      float x = dataset[b_idx * n * 3 + n_idx*3 + 0];
      float y = dataset[b_idx * n * 3 + n_idx*3 + 1];
      float z = dataset[b_idx * n * 3 + n_idx*3 + 2];

      distance1[n_idx] = (x - far1_x) * (x - far1_x) + (y - far1_y) * (y - far1_y) + (z - far1_z) * (z - far1_z);
      distance_avg[b_idx * n + n_idx] = (x - avgx) * (x - avgx) + (y - avgy) * (y - avgy) + (z - avgz) * (z - avgz);  
      
      if (distance1[n_idx] > farthest){
        farthest = distance1[n_idx];
        farthest_idx2 = n_idx;       
      }
    }

    float far2_x = dataset[b_idx * n * 3 + farthest_idx2*3 + 0];
    float far2_y = dataset[b_idx * n * 3 + farthest_idx2*3 + 1];
    float far2_z = dataset[b_idx * n * 3 + farthest_idx2*3 + 2];

    float distance2[n];

    // Calculate the distance from the second farthest point
    for (int n_idx = 0; n_idx < n; ++n_idx){
      float x = dataset[b_idx * n * 3 + n_idx*3 + 0];
      float y = dataset[b_idx * n * 3 + n_idx*3 + 1];
      float z = dataset[b_idx * n * 3 + n_idx*3 + 2];

      distance2[n_idx] = (x - far2_x) * (x - far2_x) + (y - far2_y) * (y - far2_y) + (z - far2_z) * (z - far2_z);     
    }


    //Sort distance 1 and distance 2
    std::vector<float> v1(distance1, distance1 + n);
    std::vector<long unsigned int> w1 = sort_indexes(v1);

    std::vector<float> v2(distance2, distance2 + n);
    std::vector<long unsigned int> w2 = sort_indexes(v2);

    int m_idx = 0;    
    int step = (int) (n / m);
    if (n % m != 0){
      step++;
    }

    // Sample from w1 and w2, in reverse order
    while (m_idx < m){
      if (m_idx % 2 == 0){
        inds[b_idx * m + m_idx] = w1[n - 1 - step * m_idx/2];        
      } else {
        inds[b_idx * m + m_idx] = w2[n - 1 - step * (m_idx-1)/2];        
      }
      m_idx++;
    }    
  }
  
  return {idx, dist};
}


std::vector<at::Tensor> voxel_sampling(at::Tensor points, const int m) {

  //printf("Strting distance_from_avg...\n");

  int B = points.size(0);
  int n = points.size(1);

  at::Tensor dist =
      torch::zeros({B, n}, at::device(points.device()).dtype(at::ScalarType::Float));

  at::Tensor idx =
      torch::zeros({B, m}, at::device(points.device()).dtype(at::ScalarType::Int));

  const float* dataset = (float *) points.data<float>();
  float* distance_avg = dist.data<float>();
  int* inds = (int *) idx.data<int>();

  int n_slice = 16;

  if (m > 1638){
    n_slice=32;    
  }

  for (int b_idx = 0; b_idx < B; ++b_idx){

    float avgx = 0;
    float avgy = 0;
    float avgz = 0;    

    float x_max, y_max, z_max;
    float x_min, y_min, z_min;

    x_max = dataset[b_idx * n * 3 + 0];
    y_max = dataset[b_idx * n * 3 + 1];
    z_max = dataset[b_idx * n * 3 + 2];

    x_min = x_max;
    y_min = y_max;
    z_min = z_max;

    for (int n_idx = 0; n_idx < n; ++n_idx){
      float x = dataset[b_idx * n * 3 + n_idx*3 + 0];
      float y = dataset[b_idx * n * 3 + n_idx*3 + 1];
      float z = dataset[b_idx * n * 3 + n_idx*3 + 2];

      if (x > x_max){
        x_max = x;        
      } else if (x < x_min){
        x_min = x;
      }

      if (y > y_max){
        y_max = y;        
      } else if (y < y_min){
        y_min = y;
      }

      if (z > z_max){
        z_max = z;        
      } else if (z < z_min){
        z_min = z;
      }

      avgx += x;
      avgy += y;
      avgz += z;
    }

    avgx = avgx / n;
    avgy = avgy / n;
    avgz = avgz / n;

    float edge_x = (x_max - x_min) / n_slice;
    float edge_y = (y_max - y_min) / n_slice;
    float edge_z = (z_max - z_min) / n_slice;


    int n_voxel = n_slice * n_slice * n_slice;
    float *voxel =  (float *) malloc(sizeof(float) * n_voxel);
    int included[n];
    
    int m_idx = 0;
    int loop_cnt = 0;

    while (m_idx < m){
      for (int n_idx = 0; n_idx < n; ++n_idx){
        if (included[n_idx] == 1) continue;
        float x = dataset[b_idx * n * 3 + n_idx*3 + 0];
        float y = dataset[b_idx * n * 3 + n_idx*3 + 1];
        float z = dataset[b_idx * n * 3 + n_idx*3 + 2];

        int voxel_no_x = (int) ((x - x_min)/ edge_x);
        int voxel_no_y = (int) ((y - y_min)/ edge_y);
        int voxel_no_z = (int) ((z - z_min)/ edge_z);

        if (voxel_no_x >= n_slice){
          voxel_no_x = n_slice - 1;
        }

        if (voxel_no_y >= n_slice){
          voxel_no_y = n_slice - 1;
        }

        if (voxel_no_z >= n_slice){
          voxel_no_z = n_slice - 1;
        }

        int point_per_voxel = voxel[voxel_no_x * n_slice * n_slice + voxel_no_y * n_slice + voxel_no_z];        

        if (point_per_voxel == loop_cnt){
          voxel[voxel_no_x * n_slice * n_slice + voxel_no_y * n_slice + voxel_no_z] += 1;
          inds[b_idx * m + m_idx] = n_idx;
          included[n_idx] == 1;
          m_idx++;
        }

        if (m_idx >= m) break;
      }
      loop_cnt++;
    }

    for (int n_idx = 0; n_idx < n; ++n_idx){
      float x = dataset[b_idx * n * 3 + n_idx*3 + 0];
      float y = dataset[b_idx * n * 3 + n_idx*3 + 1];
      float z = dataset[b_idx * n * 3 + n_idx*3 + 2];

      distance_avg[b_idx * n + n_idx] = (x - avgx) * (x - avgx) + (y - avgy) * (y - avgy) + (z - avgz) * (z - avgz);  
    }
  }
  
  return {idx, dist};
}



//Added
at::Tensor distance_from_avg(at::Tensor points, const int n) {

  //printf("Strting distance_from_avg...\n");


  at::Tensor output =
      torch::zeros({n}, at::device(points.device()).dtype(at::ScalarType::Float));

  const float* dataset = (float *) points.data<float>();
  float* distances = output.data<float>();

  float avgx = 0;
  float avgy = 0;
  float avgz = 0;    

  for (int n_idx = 0; n_idx < n; ++n_idx){
    float x = dataset[n_idx*3 + 0];
    float y = dataset[n_idx*3 + 1];
    float z = dataset[n_idx*3 + 2];

    avgx += x;
    avgy += y;
    avgz += z;
  }

  avgx = avgx / n;
  avgy = avgy / n;
  avgz = avgz / n;   


  for (int n_idx = 0; n_idx < n; ++n_idx){
    float x = dataset[n_idx*3 + 0];
    float y = dataset[n_idx*3 + 1];
    float z = dataset[n_idx*3 + 2];

    distances[n_idx] = (x - avgx) * (x - avgx) + (y - avgy) * (y - avgy) + (z - avgz) * (z - avgz);    
  }
  
  return output;
}


at::Tensor gather_points_nocuda(at::Tensor points, at::Tensor idx) {
  CHECK_CONTIGUOUS(points);
  CHECK_CONTIGUOUS(idx);
  CHECK_IS_FLOAT(points);
  CHECK_IS_INT(idx);
  
  int B = points.size(0);
  int c = points.size(1);
  int n = points.size(2);
  int m = idx.size(1);

  at::Tensor output =
      torch::zeros({points.size(0), points.size(1), idx.size(1)},
                   at::device(points.device()).dtype(at::ScalarType::Float));

  const float* dataset = (float *) points.data<float>();
  const int* inds = (int *) idx.data<int>();

  float* gather = output.data<float>();

  for (int b_idx = 0; b_idx < B; ++b_idx){
    for (int m_idx = 0; m_idx < m; ++m_idx){
      int center = inds[b_idx * m + m_idx];
      for (int c_idx = 0; c_idx < c; ++c_idx){
        gather[b_idx * c * m + c_idx * m + m_idx] = dataset[b_idx * c * n + c_idx * n + center];
      }      
    }
  }

  return output;
}

at::Tensor gather_points_grad_nocuda(at::Tensor grad_out, at::Tensor idx,
                              const int n) {
  CHECK_CONTIGUOUS(grad_out);
  CHECK_CONTIGUOUS(idx);
  CHECK_IS_FLOAT(grad_out);
  CHECK_IS_INT(idx);

  int B = grad_out.size(0);
  int c = grad_out.size(1);
  int m = grad_out.size(2);


  at::Tensor output =
      torch::zeros({grad_out.size(0), grad_out.size(1), n},
                   at::device(grad_out.device()).dtype(at::ScalarType::Float));
  
  const float* dataset = (float *) grad_out.data<float>(); 
  const int* inds = (int *) idx.data<int>();   

  float* grad_features = output.data<float>();

  for (int b_idx = 0; b_idx < B; ++b_idx){
    for (int m_idx = 0; m_idx < m; ++m_idx){
      int center = inds[b_idx * m + m_idx];
      for (int c_idx = 0; c_idx < c; ++c_idx){
        grad_features[b_idx * c * n + c_idx * n + center] = dataset[b_idx * c * m + c_idx * m + m_idx];
      }      
    }
  }

  return output;
}

