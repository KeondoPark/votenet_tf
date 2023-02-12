#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include <cuda_runtime.h>

using namespace tensorflow;

REGISTER_OP("KnnSample")
  .Attr("k: int")
  .Input("xyz: float32")
  .Input("new_xyz: float32")
  .Input("offset: int32")
  .Input("new_offset: int32")
  .Output("idx: int32")
  .Output("dist: float32")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    ::tensorflow::shape_inference::ShapeHandle dims1; // B, M, 3
    c->WithRank(c->input(1), 3, &dims1); // shape of new_xyz
    int k;
    TF_RETURN_IF_ERROR(c->GetAttr("k", &k));
    ::tensorflow::shape_inference::ShapeHandle output1 = c->MakeShape({c->Dim(dims1, 0), c->Dim(dims1, 1), k});
    c->set_output(0, output1);
    ::tensorflow::shape_inference::ShapeHandle output2 = c->MakeShape({c->Dim(dims1, 0), c->Dim(dims1, 1), k});
    c->set_output(1, output2);
    return Status::OK();
  });

void knnquery_cuda_launcher(int b, int m, int n, int s, int nsample, const float *xyz, const float *new_xyz, const int *offset, const int *new_offset, int *idx, float *dist2);
class KnnSampleGpuOp: public OpKernel{
  public:
    explicit KnnSampleGpuOp(OpKernelConstruction* context):OpKernel(context) {
                    OP_REQUIRES_OK(context, context->GetAttr("k", &nsample_));
                    OP_REQUIRES(context, nsample_ > 0, errors::InvalidArgument("KnnSample expects positive k"));
                }
    void Compute(OpKernelContext * context)override{
      int nsample = nsample_;

      const Tensor& inp_tensor = context->input(0);
      const Tensor& new_inp_tensor = context->input(1);
      const Tensor& offset_tensor = context->input(2);
      const Tensor& new_offset_tensor = context->input(3);
      OP_REQUIRES(context, inp_tensor.dims()==3 && inp_tensor.shape().dim_size(2)==3, 
                errors::InvalidArgument("knnSample expects xyz has (B,N,3) shape"));
      OP_REQUIRES(context, new_inp_tensor.dims()==3 && new_inp_tensor.shape().dim_size(2)==3, 
                errors::InvalidArgument("knnSample expects new xyz has (B,M,3) shape"));

      int b = inp_tensor.shape().dim_size(0);
      int n = inp_tensor.shape().dim_size(1);
      int m = new_inp_tensor.shape().dim_size(1);
      int s = offset_tensor.shape().dim_size(1);

      auto inp_flat = inp_tensor.flat<float>();
      auto new_inp_flat = new_inp_tensor.flat<float>();
      auto offset_flat = offset_tensor.flat<int>();
      auto new_offset_flat = offset_tensor.flat<int>();
      const float * inp = &(inp_flat(0));
      const float * new_inp = &(new_inp_flat(0));
      const int * offset = &(offset_flat(0));
      const int * new_offset = &(new_offset_flat(0));

      Tensor * out_tensor;
      OP_REQUIRES_OK(context,context->allocate_output(0, TensorShape{b,m,nsample}, &out_tensor));      
      auto out_flat = out_tensor->flat<int>();
      int * out = &(out_flat(0));

      Tensor * dist_tensor;
      OP_REQUIRES_OK(context,context->allocate_output(1, TensorShape{b,m,nsample}, &dist_tensor));      
      auto dist_flat = dist_tensor->flat<float>();
      float * dist = &(dist_flat(0));

      knnquery_cuda_launcher(b, m, n, s, nsample, inp, new_inp, offset, new_offset, out, dist);
    }
    private:
        int nsample_;
};
REGISTER_KERNEL_BUILDER(Name("KnnSample").Device(DEVICE_GPU), KnnSampleGpuOp);

