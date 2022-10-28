// Copyright (c) Facebook, Inc. and its affiliates.
// 
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <torch/extension.h>
#include <vector>

std::vector<at::Tensor> three_nn(at::Tensor unknowns, at::Tensor knows);
at::Tensor three_interpolate(at::Tensor points, at::Tensor idx,
                             at::Tensor weight);
at::Tensor three_interpolate_grad(at::Tensor grad_out, at::Tensor idx,
                                  at::Tensor weight, const int m);
at::Tensor inv_ball_query_nocuda(at::Tensor unknowns, at::Tensor knows, at::Tensor grouped_xyz, at::Tensor idx);
at::Tensor inv_interpolate_nocuda(at::Tensor points, at::Tensor idx);
at::Tensor inv_interpolate_nocuda_grad(at::Tensor grad_out, at::Tensor idx, int m);