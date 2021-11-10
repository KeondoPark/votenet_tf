/* Furthest point sampling
 * Original author: Haoqiang Fan
 * Modified by Charles R. Qi
 * All Rights Reserved. 2017. 
 */
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include <cuda_runtime.h>

using namespace tensorflow;

REGISTER_OP("ProbSample")
  .Input("inp: float32")
  .Input("inpr: float32")
  .Output("out: int32")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    ::tensorflow::shape_inference::ShapeHandle dims1; // batch_size * ncategory
    c->WithRank(c->input(0), 2, &dims1);
    ::tensorflow::shape_inference::ShapeHandle dims2; // batch_size * npoints
    c->WithRank(c->input(1), 2, &dims2);
    // batch_size * npoints
    ::tensorflow::shape_inference::ShapeHandle output = c->MakeShape({c->Dim(dims2, 0), c->Dim(dims2, 1)});
    c->set_output(0, output);
    return Status::OK();
  });


REGISTER_OP("FarthestPointSample")
  .Attr("npoint: int")
  .Input("inp: float32")
  .Output("out: int32")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    ::tensorflow::shape_inference::ShapeHandle dims1; // batch_size * npoint * 3
    c->WithRank(c->input(0), 3, &dims1);
    int npoint;
    TF_RETURN_IF_ERROR(c->GetAttr("npoint", &npoint));
    ::tensorflow::shape_inference::ShapeHandle output = c->MakeShape({c->Dim(dims1, 0), npoint});
    c->set_output(0, output);
    return Status::OK();
  });

REGISTER_OP("FarthestPointSampleBg")
  .Attr("npoint: int")
  .Attr("weight: float")
  .Attr("isFront: int")
  .Input("inp: float32")
  .Output("out: int32")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    ::tensorflow::shape_inference::ShapeHandle dims1; // batch_size * npoint * 3
    c->WithRank(c->input(0), 3, &dims1);
    int npoint;
    float weight;
    int isFront;
    TF_RETURN_IF_ERROR(c->GetAttr("npoint", &npoint));
    TF_RETURN_IF_ERROR(c->GetAttr("weight", &weight));
    TF_RETURN_IF_ERROR(c->GetAttr("isFront", &isFront));
    ::tensorflow::shape_inference::ShapeHandle output = c->MakeShape({c->Dim(dims1, 0), npoint});
    c->set_output(0, output);
    return Status::OK();
  });

REGISTER_OP("FarthestPointSampleBg2")
  .Attr("npoint: int")
  .Attr("weight1: float")
  .Attr("weight2: float")
  .Attr("isFront1: int")
  .Attr("isFront2: int")
  .Input("inp: float32")
  .Output("out: int32")  
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    ::tensorflow::shape_inference::ShapeHandle dims1; // batch_size * npoint * 3
    c->WithRank(c->input(0), 3, &dims1);
    int npoint;
    float weight1;
    int isFront1;
    float weight2;
    int isFront2;
    TF_RETURN_IF_ERROR(c->GetAttr("npoint", &npoint));
    TF_RETURN_IF_ERROR(c->GetAttr("weight1", &weight1));
    TF_RETURN_IF_ERROR(c->GetAttr("isFront1", &isFront1));
    TF_RETURN_IF_ERROR(c->GetAttr("weight2", &weight2));
    TF_RETURN_IF_ERROR(c->GetAttr("isFront2", &isFront2));
    ::tensorflow::shape_inference::ShapeHandle output = c->MakeShape({c->Dim(dims1, 0), 2*npoint});    
    c->set_output(0, output);    
    return Status::OK();
  });


REGISTER_OP("GatherPoint")
  .Input("inp: float32")
  .Input("idx: int32")
  .Output("out: float32")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    ::tensorflow::shape_inference::ShapeHandle dims1; // batch_size * ndataset * 3
    c->WithRank(c->input(0), 3, &dims1);
    ::tensorflow::shape_inference::ShapeHandle dims2; // batch_size * npoints
    c->WithRank(c->input(1), 2, &dims2);
    // batch_size * npoints * 3
    ::tensorflow::shape_inference::ShapeHandle output = c->MakeShape({c->Dim(dims1, 0), c->Dim(dims2, 1), c->Dim(dims1, 2)});
    c->set_output(0, output);
    return Status::OK();
  });
REGISTER_OP("GatherPointGrad")
  .Input("inp: float32")
  .Input("idx: int32")
  .Input("out_g: float32")
  .Output("inp_g: float32")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    c->set_output(0, c->input(0));
    return Status::OK();
  });

void probsampleLauncher(int b,int n,int m,const float * inp_p,const float * inp_r,float * temp,int * out);
class ProbSampleGpuOp: public OpKernel{
  public:
    explicit ProbSampleGpuOp(OpKernelConstruction* context):OpKernel(context){}
    void Compute(OpKernelContext * context)override{
      const Tensor& inp_tensor=context->input(0);
      const Tensor& inpr_tensor=context->input(1);
      auto inp_flat=inp_tensor.flat<float>();
      auto inpr_flat=inpr_tensor.flat<float>();
      const float * inp=&(inp_flat(0));
      const float * inpr=&(inpr_flat(0));
      OP_REQUIRES(context,inp_tensor.dims()==2,errors::InvalidArgument("ProbSample expects (batch_size,num_choices) inp shape"));
      int b=inp_tensor.shape().dim_size(0);
      int n=inp_tensor.shape().dim_size(1);
      OP_REQUIRES(context,inpr_tensor.dims()==2 && inpr_tensor.shape().dim_size(0)==b,errors::InvalidArgument("ProbSample expects (batch_size,num_points) inpr shape"));
      int m=inpr_tensor.shape().dim_size(1);
      Tensor * out_tensor=NULL;
      OP_REQUIRES_OK(context,context->allocate_output(0,TensorShape{b,m},&out_tensor));
      auto out_flat=out_tensor->flat<int>();
      int * out=&(out_flat(0));
      Tensor temp_tensor;
      OP_REQUIRES_OK(context,context->allocate_temp(DataTypeToEnum<float>::value,TensorShape{b,n},&temp_tensor));
      auto temp_flat=temp_tensor.flat<float>();
      float * temp=&(temp_flat(0));
      probsampleLauncher(b,n,m,inp,inpr,temp,out);
    }
};
REGISTER_KERNEL_BUILDER(Name("ProbSample").Device(DEVICE_GPU), ProbSampleGpuOp);

void farthestpointsamplingLauncher(int b,int n,int m,const float * inp,float * temp,int * out);
class FarthestPointSampleGpuOp: public OpKernel{
  public:
    explicit FarthestPointSampleGpuOp(OpKernelConstruction* context):OpKernel(context) {
                    OP_REQUIRES_OK(context, context->GetAttr("npoint", &npoint_));
                    OP_REQUIRES(context, npoint_ > 0, errors::InvalidArgument("FarthestPointSample expects positive npoint"));
                }
    void Compute(OpKernelContext * context)override{
      int m = npoint_;

      const Tensor& inp_tensor=context->input(0);
      OP_REQUIRES(context, inp_tensor.dims()==3 && inp_tensor.shape().dim_size(2)==3, errors::InvalidArgument("FarthestPointSample expects (batch_size,num_points,3) inp shape"));
      int b=inp_tensor.shape().dim_size(0);
      int n=inp_tensor.shape().dim_size(1);
      auto inp_flat=inp_tensor.flat<float>();
      const float * inp=&(inp_flat(0));
      Tensor * out_tensor;
      OP_REQUIRES_OK(context,context->allocate_output(0,TensorShape{b,m},&out_tensor));
      auto out_flat=out_tensor->flat<int>();
      int * out=&(out_flat(0));
      Tensor temp_tensor;
      OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<float>::value, TensorShape{32,n}, &temp_tensor));
      auto temp_flat=temp_tensor.flat<float>();
      float * temp=&(temp_flat(0));
      farthestpointsamplingLauncher(b,n,m,inp,temp,out);
    }
    private:
        int npoint_;
};
REGISTER_KERNEL_BUILDER(Name("FarthestPointSample").Device(DEVICE_GPU),FarthestPointSampleGpuOp);


void farthestpointsamplingBgLauncher(int b, int n, int m, const float * inp, float * temp, int * isObj, int * out, float wght, int isFront);
class FarthestPointSampleBgGpuOp: public OpKernel{
  public:
    explicit FarthestPointSampleBgGpuOp(OpKernelConstruction* context):OpKernel(context) {
                    OP_REQUIRES_OK(context, context->GetAttr("npoint", &npoint_));
                    OP_REQUIRES_OK(context, context->GetAttr("weight", &weight_));
                    OP_REQUIRES_OK(context, context->GetAttr("isFront", &isFront_));
                    OP_REQUIRES(context, npoint_ > 0, errors::InvalidArgument("FarthestPointSample expects positive npoint"));
                }
    void Compute(OpKernelContext * context)override{
      int m = npoint_;
      float wght = weight_;
      int isF = isFront_;

      const Tensor& inp_tensor=context->input(0);
      OP_REQUIRES(context, inp_tensor.dims()==3 && inp_tensor.shape().dim_size(2)==4, errors::InvalidArgument("FarthestPointSampleBg expects (batch_size,num_points,4) inp shape"));
      int b=inp_tensor.shape().dim_size(0);
      int n=inp_tensor.shape().dim_size(1);
      auto inp_flat=inp_tensor.flat<float>();
      const float * inp=&(inp_flat(0));
      Tensor * out_tensor;
      OP_REQUIRES_OK(context,context->allocate_output(0,TensorShape{b,m},&out_tensor));
      auto out_flat=out_tensor->flat<int>();
      int * out=&(out_flat(0));
      Tensor temp_tensor;
      OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<float>::value, TensorShape{32,n}, &temp_tensor));
      auto temp_flat = temp_tensor.flat<float>();
      float * temp = &(temp_flat(0));

      Tensor isObj_tensor;
      OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<int>::value, TensorShape{32,n}, &isObj_tensor));
      auto isObj_flat = isObj_tensor.flat<int>();
      int * isObj = &(isObj_flat(0));

      farthestpointsamplingBgLauncher(b,n,m,inp,temp,isObj,out,wght,isF);
    }
    private:
        int npoint_;
        float weight_;
        int isFront_;
};
REGISTER_KERNEL_BUILDER(Name("FarthestPointSampleBg").Device(DEVICE_GPU),FarthestPointSampleBgGpuOp);

void farthestpointsamplingBgLauncher2(int b, int n, int m, const float * inp, float * temp1, float * temp2, int * isObj, int * out, float wght1, float wght2, int isFront1, int isFront2);
class FarthestPointSampleBgGpuOp2: public OpKernel{
  public:
    explicit FarthestPointSampleBgGpuOp2(OpKernelConstruction* context):OpKernel(context) {
                    OP_REQUIRES_OK(context, context->GetAttr("npoint", &npoint_));
                    OP_REQUIRES_OK(context, context->GetAttr("weight1", &weight1_));
                    OP_REQUIRES_OK(context, context->GetAttr("weight2", &weight2_));
                    OP_REQUIRES_OK(context, context->GetAttr("isFront1", &isFront1_));
                    OP_REQUIRES_OK(context, context->GetAttr("isFront2", &isFront2_));
                    OP_REQUIRES(context, npoint_ > 0, errors::InvalidArgument("FarthestPointSample expects positive npoint"));
                }
    void Compute(OpKernelContext * context)override{
      int m = npoint_;
      float wght1 = weight1_;      
      float wght2 = weight2_;
      int isF1 = isFront1_;
      int isF2 = isFront2_;

      const Tensor& inp_tensor=context->input(0);
      OP_REQUIRES(context, inp_tensor.dims()==3 && inp_tensor.shape().dim_size(2)==4, errors::InvalidArgument("FarthestPointSampleBg expects (batch_size,num_points,4) inp shape"));
      int b=inp_tensor.shape().dim_size(0);
      int n=inp_tensor.shape().dim_size(1);
      auto inp_flat=inp_tensor.flat<float>();
      const float * inp=&(inp_flat(0));
      
      Tensor * out_tensor;
      OP_REQUIRES_OK(context,context->allocate_output(0,TensorShape{b,2*m},&out_tensor));
      auto out_flat = out_tensor->flat<int>();
      int * out = &(out_flat(0));
      
      Tensor temp_tensor1;
      OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<float>::value, TensorShape{32,n}, &temp_tensor1));
      auto temp_flat1 = temp_tensor1.flat<float>();
      float * temp1 = &(temp_flat1(0));

      Tensor temp_tensor2;
      OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<float>::value, TensorShape{32,n}, &temp_tensor2));
      auto temp_flat2 = temp_tensor2.flat<float>();
      float * temp2 = &(temp_flat2(0));

      Tensor isObj_tensor;
      OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<int>::value, TensorShape{32,n}, &isObj_tensor));
      auto isObj_flat = isObj_tensor.flat<int>();
      int * isObj = &(isObj_flat(0));

      farthestpointsamplingBgLauncher2(b, n, m, inp, temp1, temp2, isObj, out, wght1, wght2, isF1, isF2);
    }
    private:
        int npoint_;
        float weight1_;
        float weight2_;
        int isFront1_;
        int isFront2_;
};
REGISTER_KERNEL_BUILDER(Name("FarthestPointSampleBg2").Device(DEVICE_GPU),FarthestPointSampleBgGpuOp2);


void gatherpointLauncher(int b,int n,int m,const float * inp,const int * idx,float * out);
class GatherPointGpuOp: public OpKernel{
  public:
    explicit GatherPointGpuOp(OpKernelConstruction * context):OpKernel(context){}
    void Compute(OpKernelContext * context)override{
      const Tensor& inp_tensor=context->input(0);
      OP_REQUIRES(context,inp_tensor.dims()==3 && inp_tensor.shape().dim_size(2)==3,errors::InvalidArgument("GatherPoint expects (batch_size,num_points,3) inp shape"));
      int b=inp_tensor.shape().dim_size(0);
      int n=inp_tensor.shape().dim_size(1);
      const Tensor& idx_tensor=context->input(1);
      OP_REQUIRES(context,idx_tensor.dims()==2 && idx_tensor.shape().dim_size(0)==b,errors::InvalidArgument("GatherPoint expects (batch_size,num_result) idx shape"));
      int m=idx_tensor.shape().dim_size(1);
      auto inp_flat=inp_tensor.flat<float>();
      const float * inp=&(inp_flat(0));
      auto idx_flat=idx_tensor.flat<int>();
      const int * idx=&(idx_flat(0));
      Tensor * out_tensor=NULL;
      OP_REQUIRES_OK(context,context->allocate_output(0,TensorShape{b,m,3},&out_tensor));
      auto out_flat=out_tensor->flat<float>();
      float * out=&(out_flat(0));
      gatherpointLauncher(b,n,m,inp,idx,out);
    }
};
REGISTER_KERNEL_BUILDER(Name("GatherPoint").Device(DEVICE_GPU),GatherPointGpuOp);

void scatteraddpointLauncher(int b,int n,int m,const float * out_g,const int * idx,float * inp_g);
class GatherPointGradGpuOp: public OpKernel{
  public:
    explicit GatherPointGradGpuOp(OpKernelConstruction * context):OpKernel(context){}
    void Compute(OpKernelContext * context)override{
      const Tensor& inp_tensor=context->input(0);
      OP_REQUIRES(context,inp_tensor.dims()==3 && inp_tensor.shape().dim_size(2)==3,errors::InvalidArgument("GatherPointGradGpuOp expects (batch_size,num_points,3) inp"));
      int b=inp_tensor.shape().dim_size(0);
      int n=inp_tensor.shape().dim_size(1);
      const Tensor& idx_tensor=context->input(1);
      OP_REQUIRES(context,idx_tensor.dims()==2 && idx_tensor.shape().dim_size(0)==b,errors::InvalidArgument("GatherPointGradGpuOp expects (batch_size,num_result) idx shape"));
      int m=idx_tensor.shape().dim_size(1);
      auto inp_flat=inp_tensor.flat<float>();
      const float * inp=&(inp_flat(0));
      auto idx_flat=idx_tensor.flat<int>();
      const int * idx=&(idx_flat(0));
      const Tensor& out_g_tensor=context->input(2);
      OP_REQUIRES(context,out_g_tensor.dims()==3 && out_g_tensor.shape().dim_size(0)==b && out_g_tensor.shape().dim_size(1)==m && out_g_tensor.shape().dim_size(2)==3,errors::InvalidArgument("GatherPointGradGpuOp expects (batch_size,num_result,3) out_g shape"));
      auto out_g_flat=out_g_tensor.flat<float>();
      const float * out_g=&(out_g_flat(0));
      Tensor * inp_g_tensor=NULL;
      OP_REQUIRES_OK(context,context->allocate_output(0,TensorShape{b,n,3},&inp_g_tensor));
      auto inp_g_flat=inp_g_tensor->flat<float>();
      float * inp_g=&(inp_g_flat(0));
      cudaMemset(inp_g,0,b*n*3*4);
      scatteraddpointLauncher(b,n,m,out_g,idx,inp_g);
    }
};
REGISTER_KERNEL_BUILDER(Name("GatherPointGrad").Device(DEVICE_GPU),GatherPointGradGpuOp);