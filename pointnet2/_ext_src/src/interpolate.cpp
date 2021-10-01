// Copyright (c) Facebook, Inc. and its affiliates.
// 
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include "interpolate.h"
#include "utils.h"

void three_nn_kernel_wrapper(int b, int n, int m, const float *unknown,
                             const float *known, float *dist2, int *idx);
void three_interpolate_kernel_wrapper(int b, int c, int m, int n,
                                      const float *points, const int *idx,
                                      const float *weight, float *out);
void three_interpolate_grad_kernel_wrapper(int b, int c, int n, int m,
                                           const float *grad_out,
                                           const int *idx, const float *weight,
                                           float *grad_points);

std::vector<at::Tensor> three_nn(at::Tensor unknowns, at::Tensor knows) {
  CHECK_CONTIGUOUS(unknowns);
  CHECK_CONTIGUOUS(knows);
  CHECK_IS_FLOAT(unknowns);
  CHECK_IS_FLOAT(knows);

  if (unknowns.type().is_cuda()) {
    CHECK_CUDA(knows);
  }

  at::Tensor idx =
      torch::zeros({unknowns.size(0), unknowns.size(1), 3},
                   at::device(unknowns.device()).dtype(at::ScalarType::Int));
  at::Tensor dist2 =
      torch::zeros({unknowns.size(0), unknowns.size(1), 3},
                   at::device(unknowns.device()).dtype(at::ScalarType::Float));

  if (unknowns.type().is_cuda()) {
    three_nn_kernel_wrapper(unknowns.size(0), unknowns.size(1), knows.size(1),
                            unknowns.data<float>(), knows.data<float>(),
                            dist2.data<float>(), idx.data<int>());
  } else {
    AT_CHECK(false, "CPU not supported");
  }

  return {dist2, idx};
}

at::Tensor three_interpolate(at::Tensor points, at::Tensor idx,
                             at::Tensor weight) {
  CHECK_CONTIGUOUS(points);
  CHECK_CONTIGUOUS(idx);
  CHECK_CONTIGUOUS(weight);
  CHECK_IS_FLOAT(points);
  CHECK_IS_INT(idx);
  CHECK_IS_FLOAT(weight);

  if (points.type().is_cuda()) {
    CHECK_CUDA(idx);
    CHECK_CUDA(weight);
  }

  at::Tensor output =
      torch::zeros({points.size(0), points.size(1), idx.size(1)},
                   at::device(points.device()).dtype(at::ScalarType::Float));

  if (points.type().is_cuda()) {
    three_interpolate_kernel_wrapper(
        points.size(0), points.size(1), points.size(2), idx.size(1),
        points.data<float>(), idx.data<int>(), weight.data<float>(),
        output.data<float>());
  } else {
    AT_CHECK(false, "CPU not supported");
  }

  return output;
}
at::Tensor three_interpolate_grad(at::Tensor grad_out, at::Tensor idx,
                                  at::Tensor weight, const int m) {
  CHECK_CONTIGUOUS(grad_out);
  CHECK_CONTIGUOUS(idx);
  CHECK_CONTIGUOUS(weight);
  CHECK_IS_FLOAT(grad_out);
  CHECK_IS_INT(idx);
  CHECK_IS_FLOAT(weight);

  if (grad_out.type().is_cuda()) {
    CHECK_CUDA(idx);
    CHECK_CUDA(weight);
  }

  at::Tensor output =
      torch::zeros({grad_out.size(0), grad_out.size(1), m},
                   at::device(grad_out.device()).dtype(at::ScalarType::Float));

  if (grad_out.type().is_cuda()) {
    three_interpolate_grad_kernel_wrapper(
        grad_out.size(0), grad_out.size(1), grad_out.size(2), m,
        grad_out.data<float>(), idx.data<int>(), weight.data<float>(),
        output.data<float>());
  } else {
    AT_CHECK(false, "CPU not supported");
  }

  return output;
}


//Input and output is transposed for increasing cache utility
// (B, c, m) -> (B, m, c)
// (B, c, n) -> (B, c, n)
at::Tensor inv_interpolate_nocuda(at::Tensor points, at::Tensor idx) {

  //printf("Strting inv_interpolate_nocuda...\n");
  
  int B = points.size(0);  
  int m = points.size(1);
  int c = points.size(2);
  int n = idx.size(1);
  int cp = idx.size(2);
  

  at::Tensor output =
      torch::zeros({B, n, c},
                   at::device(points.device()).dtype(at::ScalarType::Float));

  const float* features = (float *) points.data<float>();
  const int* inds = (int *) idx.data<int>();

  float* interpolated = output.data<float>();
  float weight = 1 / cp;

  for(int b_idx = 0; b_idx < B; ++b_idx){
    for (int n_idx = 0; n_idx < n; ++n_idx){
      for (int cp_idx = 0; cp_idx < cp; ++cp_idx){
        int center = inds[b_idx * n * cp + n_idx * cp + cp_idx];        
        for (int c_idx = 0; c_idx < c; ++c_idx){
          interpolated[b_idx * c * n + n_idx * c + c_idx] += features[b_idx * c * m + center * c + c_idx] * weight;
        }         
      }
    }
  }

  //Devide by the number of center points(Averaging)
  for (int i = 0; i < B * c  * n; ++i){
    interpolated[i] = interpolated[i] / cp;
  }
 
  return output;
}

//Input and output is transposed
//Input: (B, n, c)
//Output: (B, m, c)
//Index(Inverse ball query result): (B, n, cp)
at::Tensor inv_interpolate_nocuda_grad(at::Tensor grad_out, at::Tensor idx, int m) {
  
  //printf("Strting inv_interpolate_nocuda_grad...\n");
  
  int B = grad_out.size(0);  
  int n = grad_out.size(1);
  int c = grad_out.size(2);  
  int cp = idx.size(2);
  

  at::Tensor output =
      torch::zeros({B, m, c},
                   at::device(grad_out.device()).dtype(at::ScalarType::Float));

  const float* grad = (float *) grad_out.data<float>();
  const int* inds = (int *) idx.data<int>();

  float* inv_intp = output.data<float>();
  float weight = 1 / cp;

  for (int b_idx = 0; b_idx < B; ++b_idx){
    for (int n_idx = 0; n_idx < n; ++n_idx){
      for (int cp_idx = 0; cp_idx < cp; ++cp_idx){
        int center = inds[b_idx * n * cp + n_idx * cp + cp_idx];
        for (int c_idx = 0; c_idx < c; ++c_idx){
          inv_intp[b_idx * c * m + center * c + c_idx] += grad[b_idx * c * n + n_idx * c + c_idx] * weight;
        }
      }
    }
  }
 
  return output;
}



//Added
at::Tensor inv_ball_query_nocuda(at::Tensor unknowns, at::Tensor knows, at::Tensor grouped_xyz, at::Tensor idx) {

  //printf("Strting inv_ball_query_nocuda...\n");

  int b = unknowns.size(0);
  int n = unknowns.size(1);
  int m = knows.size(1);
  int k = grouped_xyz.size(2);
  int cp = (n / m) > 4 ? (int) (n / m): 4;

  at::Tensor inv_idx =
      torch::zeros({b, n, cp},
                   at::device(unknowns.device()).dtype(at::ScalarType::Int));

  const float* unknown_points = (float *) unknowns.data<float>();
  const float* known_points = (float *) knows.data<float>();
  const int* balls = (int *) grouped_xyz.data<int>();
  const int* inds = (int  *) idx.data<int>();

  int* centers = inv_idx.data<int>();

  for (int b_idx = 0; b_idx < b; ++b_idx){
    for (int n_idx = 0; n_idx < n; ++n_idx){
      int j = 0;
      //Check if n_idx is center point
      for (int m_idx = 0; m_idx < m; ++m_idx){
        if (inds[b_idx * m + m_idx] == n_idx){
          while (j < cp){
            centers[b_idx * n * cp + n_idx * cp + j] = m_idx;
            j++;
          }
          break;
        }
      }

      if (j >= cp)
        continue;

      //if n_idx is not the center point
      for (int m_idx = 0; m_idx < m; ++m_idx){
        for (int k_idx = 0; k_idx < k; ++k_idx){
          if (balls[b_idx * m * k + m_idx * k + k_idx] == n_idx){
            centers[b_idx * n * cp + n_idx * cp + j] = m_idx;
            j++;
            break;
          }
        }
        if (j >= cp)
          break;
      }

      int uniq_cnt = j;

      //If no center point is found, randomly assign center point
      if (j == 0){
        //printf("No center point around. b_idx: %d, n_idx: %d\n", b_idx, n_idx);
        float min_dist = 1e+10;
        int min_m = -1;
        float x = unknown_points[b_idx * n * 3 + n_idx * 3 + 0];
        float y = unknown_points[b_idx * n * 3 + n_idx * 3 + 1];
        float z = unknown_points[b_idx * n * 3 + n_idx * 3 + 2];
        for (int m_idx = 0; m_idx < m; ++m_idx){
          float center_x = known_points[b_idx * m * 3 + m_idx * 3 + 0];
          float center_y = known_points[b_idx * m * 3 + m_idx * 3 + 1];
          float center_z = known_points[b_idx * m * 3 + m_idx * 3 + 2];

          float dist = (x - center_x) * (x - center_x) + (y - center_y) * (y - center_y) + (z - center_z) * (z - center_z);

          if (min_dist > dist){
            min_dist = dist;
            min_m = m_idx;
          } 
        }


        while (j < cp){
            //centers[b_idx * n * cp + n_idx * cp + j] = std::experimental::randint(0, m);
            centers[b_idx * n * cp + n_idx * cp + j] = min_m;
            j++;
          }
      } else {
        //if the number of centers found is less than cp, repeat the centers already found
        while (j < cp){
          centers[b_idx * n * cp + n_idx * cp + j] = centers[b_idx * n * cp + n_idx * cp + j - uniq_cnt];
          j++;
        }
      }
    }
  }

  return inv_idx;
}