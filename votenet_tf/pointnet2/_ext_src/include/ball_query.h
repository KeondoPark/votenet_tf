// Copyright (c) Facebook, Inc. and its affiliates.
// 
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once
#include <torch/extension.h>

at::Tensor ball_query(at::Tensor new_xyz, at::Tensor xyz, const float radius,
                      const int nsample);
at::Tensor ball_query_nocuda(at::Tensor new_xyz, at::Tensor xyz, float radius,
                      const int nsample, at::Tensor batch_distances, at::Tensor inds,
                      at::Tensor arg_sort);
int binarySearch(const float arr[], int l, int r, float x, int lu_opt);