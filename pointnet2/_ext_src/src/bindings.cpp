// Copyright (c) Facebook, Inc. and its affiliates.
// 
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include "ball_query.h"
#include "group_points.h"
#include "interpolate.h"
#include "sampling.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gather_points", &gather_points);
  m.def("gather_points_grad", &gather_points_grad);
  m.def("furthest_point_sampling", &furthest_point_sampling);

  m.def("three_nn", &three_nn);
  m.def("three_interpolate", &three_interpolate);
  m.def("three_interpolate_grad", &three_interpolate_grad);

  m.def("ball_query", &ball_query);

  m.def("group_points", &group_points);
  m.def("group_points_grad", &group_points_grad);


  m.def("sampling_from_avg", &sampling_from_avg); 
  m.def("sampling_from_ending_points", &sampling_from_ending_points); 
  m.def("voxel_sampling", &voxel_sampling); 
  m.def("distance_from_avg", &distance_from_avg);
  m.def("ball_query_nocuda", &ball_query_nocuda);
  m.def("inv_ball_query_nocuda", &inv_ball_query_nocuda);
  m.def("inv_interpolate_nocuda", &inv_interpolate_nocuda);
  m.def("inv_interpolate_nocuda_grad", &inv_interpolate_nocuda_grad);
  m.def("group_points_nocuda", &group_points_nocuda);
  m.def("group_points_grad_nocuda", &group_points_grad_nocuda);
  m.def("gather_points_nocuda", &gather_points_nocuda);
  m.def("gather_points_grad_nocuda", &gather_points_grad_nocuda);
  
}
