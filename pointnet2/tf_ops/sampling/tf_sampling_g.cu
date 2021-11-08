/* Furthest point sampling GPU implementation
 * Original author: Haoqiang Fan
 * Modified by Charles R. Qi
 * All Rights Reserved. 2017. 
 */

 __global__ void cumsumKernel(int b,int n,const float * __restrict__ inp,float * __restrict__ out){
  const int BlockSize=2048;
  const int paddingLevel=5;
  __shared__ float buffer4[BlockSize*4];
  __shared__ float buffer[BlockSize+(BlockSize>>paddingLevel)];
  for (int i=blockIdx.x;i<b;i+=gridDim.x){
    float runningsum=0,runningsum2=0;
    for (int j=0;j<n;j+=BlockSize*4){
      int n24_i=min(n-j,BlockSize*4);
      int n24=(n24_i+3)&~3;
      int n2=n24>>2;
      for (int k=threadIdx.x*4;k<n24_i;k+=blockDim.x*4){
        if (k+3<n24_i){
          float v1=inp[i*n+j+k];
          float v2=inp[i*n+j+k+1];
          v2+=v1;
          float v3=inp[i*n+j+k+2];
          float v4=inp[i*n+j+k+3];
          v4+=v3;
          v3+=v2;
          v4+=v2;
          buffer4[k]=v1;
          buffer4[k+1]=v2;
          buffer4[k+2]=v3;
          buffer4[k+3]=v4;
          buffer[(k>>2)+(k>>(2+paddingLevel))]=v4;
        }else{
          float v=0;
          for (int k2=k;k2<n24_i;k2++){
            v+=inp[i*n+j+k2];
            buffer4[k2]=v;
          }
          for (int k2=n24_i;k2<n24;k2++){
            buffer4[k2]=v;
          }
          buffer[(k>>2)+(k>>(2+paddingLevel))]=v;
        }
      }
      int u=0;
      for (;(2<<u)<=n2;u++){
        __syncthreads();
        for (int k=threadIdx.x;k<int(n2>>(u+1));k+=blockDim.x){
          int i1=(((k<<1)+2)<<u)-1;
          int i2=(((k<<1)+1)<<u)-1;
          i1+=i1>>paddingLevel;
          i2+=i2>>paddingLevel;
          buffer[i1]+=buffer[i2];
        }
      }
      u--;
      for (;u>=0;u--){
        __syncthreads();
        for (int k=threadIdx.x;k<int((n2-(1<<u))>>(u+1));k+=blockDim.x){
          int i1=(((k<<1)+3)<<u)-1;
          int i2=(((k<<1)+2)<<u)-1;
          i1+=i1>>paddingLevel;
          i2+=i2>>paddingLevel;
          buffer[i1]+=buffer[i2];
        }
      }
      __syncthreads();
      for (int k=threadIdx.x*4;k<n24;k+=blockDim.x*4){
        if (k!=0){
          int k2=((k>>2)-1)+(((k>>2)-1)>>paddingLevel);
          buffer4[k]+=buffer[k2];
          buffer4[k+1]+=buffer[k2];
          buffer4[k+2]+=buffer[k2];
          buffer4[k+3]+=buffer[k2];
        }
      }
      __syncthreads();
      for (int k=threadIdx.x;k<n24_i;k+=blockDim.x){
        out[i*n+j+k]=buffer4[k]+runningsum;
      }
      float t=buffer[(n2-1)+((n2-1)>>paddingLevel)]+runningsum2;
      float r2=runningsum+t;
      runningsum2=t-(r2-runningsum);
      runningsum=r2;
      __syncthreads();
    }
  }
}

__global__ void binarysearchKernel(int b,int n,int m,const float * __restrict__ dataset,const float * __restrict__ query, int * __restrict__ result){
  int base=1;
  while (base<n)
    base<<=1;
  for (int i=blockIdx.x;i<b;i+=gridDim.x){
    for (int j=blockIdx.y*blockDim.x+threadIdx.x;j<m;j+=blockDim.x*gridDim.y){
      float q=query[i*m+j]*dataset[i*n+n-1];
      int r=n-1;
      for (int k=base;k>=1;k>>=1)
        if (r>=k && dataset[i*n+r-k]>=q)
          r-=k;
      result[i*m+j]=r;
    }
  }
}
__global__ void farthestpointsamplingKernel(int b,int n,int m,const float * __restrict__ dataset,float * __restrict__ temp,int * __restrict__ idxs){
  if (m<=0)
    return;
  const int BlockSize=512;
  __shared__ float dists[BlockSize];
  __shared__ int dists_i[BlockSize];
  const int BufferSize=3072;
  __shared__ float buf[BufferSize*3];
  
  // b: Batch size, n: Number of input points, m: Number of output(Sampled) points
  // gridDim: This variable contains the dimensions of the grid.
  // blockIdx: This variable contains the block index within the grid.
  // blockDim: This variable and contains the dimensions of the block.
  // threadIdx: This variable contains the thread index within the block.

  for (int i=blockIdx.x; i<b; i+=gridDim.x){
    int old=0;
    if (threadIdx.x==0)
      idxs[i*m+0] = old; // First output is set to 0th point

    // Initialize temp array
    // For each point, the closest distance to sampled points
    for (int j = threadIdx.x; j < n; j += blockDim.x){
      temp[blockIdx.x*n+j] = 1e38;
    }

    for (int j = threadIdx.x; j < min(BufferSize,n)*3; j += blockDim.x){
      buf[j] = dataset[i*n*3 + j];
    }

    __syncthreads();

    for (int j = 1; j < m; j++){
      int besti = 0;
      float best = -1;
      // Sampled point... old is the current point index      
      float x1 = dataset[i*n*3 + old*3 + 0];
      float y1 = dataset[i*n*3 + old*3 + 1];
      float z1 = dataset[i*n*3 + old*3 + 2];

      // Calculate distance to every blockDim.x point in the point set
      // Get the maximum of minimum distance(to sampled point)
      for (int k = threadIdx.x; k < n; k += blockDim.x){
        float td = temp[blockIdx.x*n + k];
        float x2,y2,z2;
        if (k < BufferSize){
          x2 = buf[k*3 + 0];
          y2 = buf[k*3 + 1];
          z2 = buf[k*3 + 2];
        }else{
          x2 = dataset[i*n*3 + k*3 + 0];
          y2 = dataset[i*n*3 + k*3 + 1];
          z2 = dataset[i*n*3 + k*3 + 2];
        }
        float d = (x2-x1)*(x2-x1) + (y2-y1)*(y2-y1) + (z2-z1)*(z2-z1);
        float d2 = min(d,td);
        if (d2!=td)
          temp[blockIdx.x*n + k] = d2;
        if (d2 > best){
          best = d2;
          besti = k;
        }
      }
      dists[threadIdx.x] = best;
      dists_i[threadIdx.x] = besti;

      //Reduce to the farthest one(512, 256, 128, ... 1)
      for (int u=0; (1<<u) < blockDim.x; u++){
        __syncthreads();
        if (threadIdx.x < (blockDim.x>>(u+1))){
          int i1 = (threadIdx.x*2)<<u;
          int i2 = (threadIdx.x*2+1)<<u;
          if (dists[i1] < dists[i2]){
            dists[i1] = dists[i2];
            dists_i[i1] = dists_i[i2];
          }
        }
      }
      __syncthreads();
      old = dists_i[0];
      if (threadIdx.x == 0)
        idxs[i*m+j] = old;
    }
  }
}

__global__ void farthestpointsamplingBgKernel(int b,int n,int m,const float * __restrict__ dataset,float * __restrict__ temp, int * __restrict__ isObj, int * __restrict__ idxs, float wght, int isFront){
  if (m<=0)
    return;
  const int BlockSize=512;
  __shared__ float dists[BlockSize];
  __shared__ int dists_i[BlockSize];
  const int BufferSize=3072;
  __shared__ float buf[BufferSize*3];  
  //__shared__ int thr;
  __shared__ float maxy[8];
  __shared__ float miny[8];
  __shared__ float maxz[8];
  __shared__ float minz[8];
  
  
  // b: Batch size, n: Number of input points, m: Number of output(Sampled) points
  // gridDim: This variable contains the dimensions of the grid.
  // blockIdx: This variable contains the block index within the grid.
  // blockDim: This variable and contains the dimensions of the block.
  // threadIdx: This variable contains the thread index within the block.

  for (int i=blockIdx.x; i<b; i+=gridDim.x){
    int old = 0;
    int cntObj = 0;
    maxy[i] = -10000.0;
    miny[i] = 10000.0;
    maxz[i] = -10000.0;
    minz[i] = 10000.0;
    
    if (threadIdx.x==0)
      idxs[i*m+0] = old; // First output is set to 0th point

    // Initialize temp array
    // For each point, the closest distance to sampled points
    for (int j = threadIdx.x; j < n; j += blockDim.x){
      temp[blockIdx.x*n+j] = 1e38;      
      isObj[blockIdx.x*n+j] = dataset[i*n*4 + 4*j + 3];
    }

    for (int j = threadIdx.x; j < min(BufferSize*3, n*4); j += blockDim.x){
      int newj = (j/3)*4 + j % 3;
      buf[j] = dataset[i*n*4 + newj];
      //buf[j] = dataset[i*n*4 + j];
    }
    
    __syncthreads();

    if (threadIdx.x == 0){
      for (int j = 0; j < n; j++){
        if (isObj[j] > 0)
          cntObj++;
        maxy[i] = max(dataset[i*n*4 + 4*j + 1], maxy[i]);
        miny[i] = min(dataset[i*n*4 + 4*j + 1], miny[i]);
        //maxz[i] = max(dataset[i*n*4 + 4*j + 2], maxz[i]);
        //minz[i] = min(dataset[i*n*4 + 4*j + 2], minz[i]);

      }
      
      // If there is no painted point, find max and min of y coords.
      if (cntObj < 100) {
        if (isFront == 0){          
          /*
          maxy[i] = maxy[i] - (maxy[i] - miny[i]) * 1/10;
          miny[i] = miny[i] + (maxy[i] - miny[i]) * 1/10;
          maxz[i] = maxz[i] - (maxz[i] - minz[i]) * 1/10;
          minz[i] = minz[i] + (maxz[i] - minz[i]) * 1/10;
          */
          miny[i] = miny[i] + (maxy[i] - miny[i]) * 0.5;          
        } else if (isFront == 1)  {
          maxy[i] = maxy[i] - (maxy[i] - miny[i]) * 0.5;
        }        
      }
      

      /* 
      // Favor painted pointset when there are more painted points
      if (cntObj > 2000){
       thr = (cntObj - 2000) / 5 + 200;
        thr = min(thr, 1536);
      }

      
      //Favor painted pointset 
      if (cntObj < 640){
        thr = min(cntObj, 128);
      } else if (cntObj < thr * 5){
        thr = cntObj / 5;        
      }
      */
      
    }
    
    __syncthreads();

    for (int j = 1; j < m; j++){
      int besti = 0;
      float best = -1;

      // Sampled point... old is the current point index      
      float x1 = dataset[i*n*4 + old*4 + 0];
      float y1 = dataset[i*n*4 + old*4 + 1];
      float z1 = dataset[i*n*4 + old*4 + 2];

      // Calculate distance to every blockDim.x point in the point set
      // Get the maximum of minimum distance(to sampled point)
      for (int k = threadIdx.x; k < n; k += blockDim.x){                
        float td = temp[blockIdx.x*n + k];
        float x2,y2,z2;
        if (k < BufferSize){
          x2 = buf[k*3 + 0];
          y2 = buf[k*3 + 1];
          z2 = buf[k*3 + 2];
        }else{
          x2 = dataset[i*n*4 + k*4 + 0];
          y2 = dataset[i*n*4 + k*4 + 1];
          z2 = dataset[i*n*4 + k*4 + 2];
        }

        float d = (x2-x1)*(x2-x1) + (y2-y1)*(y2-y1) + (z2-z1)*(z2-z1);
        /*
        if (cntObj == 0){        
          if (isFront == 0){
            d = wght * (x2-x1)*(x2-x1) + (y2-y1)*(y2-y1) + (z2-z1)*(z2-z1);
          } else if (isFront == 1){
            d = (x2-x1)*(x2-x1) + wght * (y2-y1)*(y2-y1) + (z2-z1)*(z2-z1);
          } else if (isFront == 2){
            d = (x2-x1)*(x2-x1) + (y2-y1)*(y2-y1) + wght * (z2-z1)*(z2-z1);
          }          
        }
        */
        
        if (isObj[k] > 0){
          d = wght * d;
        }        

        if (cntObj < 100 && isFront >= 0){          
          //if (isFront > 0 && y2 < miny[i]){
          //  continue;
          //} 

          //if (isFront == 0 && y2 > maxy[i]){
          //  continue;
          //} 

          //Gives bigger weight to Focus area
          //if (isFront == 0 && y2 < maxy[i] && y2 > miny[i] && z2 < maxz[i] && z2 > minz[i]){
          if (y2 < maxy[i] && y2 > miny[i]){
            d = 2 * d;
          }          
        } 



        float d2 = min(d,td);
        if (d2!=td)
          temp[blockIdx.x*n + k] = d2;
        
        //if (noBg && isObj[k] == 0)
        //  continue;

        if (d2 > best){
          best = d2;
          besti = k;
        }
      }
      dists[threadIdx.x] = best;
      dists_i[threadIdx.x] = besti;

      //Reduce to the farthest one(512, 256, 128, ... 1)
      for (int u=0; (1<<u) < blockDim.x; u++){
        __syncthreads();
        if (threadIdx.x < (blockDim.x>>(u+1))){
          int i1 = (threadIdx.x*2)<<u;
          int i2 = (threadIdx.x*2+1)<<u;
          if (dists[i1] < dists[i2]){
            dists[i1] = dists[i2];
            dists_i[i1] = dists_i[i2];
          }
        }
      }
      __syncthreads();
      old = dists_i[0];
      if (threadIdx.x == 0)
        idxs[i*m+j] = old;
    }
  }
}


__global__ void gatherpointKernel(int b,int n,int m,const float * __restrict__ inp,const int * __restrict__ idx,float * __restrict__ out){
  for (int i=blockIdx.x;i<b;i+=gridDim.x){
    for (int j=blockIdx.y*blockDim.x+threadIdx.x;j<m;j+=blockDim.x*gridDim.y){
      int a=idx[i*m+j];
      out[(i*m+j)*3+0]=inp[(i*n+a)*3+0];
      out[(i*m+j)*3+1]=inp[(i*n+a)*3+1];
      out[(i*m+j)*3+2]=inp[(i*n+a)*3+2];
    }
  }
}

__global__ void scatteraddpointKernel(int b,int n,int m,const float * __restrict__ out_g,const int * __restrict__ idx,float * __restrict__ inp_g){
  for (int i=blockIdx.x;i<b;i+=gridDim.x){
    for (int j=blockIdx.y*blockDim.x+threadIdx.x;j<m;j+=blockDim.x*gridDim.y){
      int a=idx[i*m+j];
      atomicAdd(&inp_g[(i*n+a)*3+0],out_g[(i*m+j)*3+0]);
      atomicAdd(&inp_g[(i*n+a)*3+1],out_g[(i*m+j)*3+1]);
      atomicAdd(&inp_g[(i*n+a)*3+2],out_g[(i*m+j)*3+2]);
    }
  }
}

void cumsumLauncher(int b,int n,const float * inp,float * out){
  cumsumKernel<<<32,512>>>(b,n,inp,out);
}
//require b*n working space
void probsampleLauncher(int b,int n,int m,const float * inp_p,const float * inp_r,float * temp,int * out){
  cumsumKernel<<<32,512>>>(b,n,inp_p,temp);
  binarysearchKernel<<<dim3(32,8,1),512>>>(b,n,m,temp,inp_r,out);
}
//require 32*n working space
void farthestpointsamplingLauncher(int b,int n,int m,const float * inp,float * temp,int * out){
  farthestpointsamplingKernel<<<32,512>>>(b,n,m,inp,temp,out);
}

void farthestpointsamplingBgLauncher(int b, int n, int m, const float * inp, float * temp, int * isObj, int * out, float wght, int isFront){
  farthestpointsamplingBgKernel<<<32,512>>>(b,n,m,inp,temp,isObj,out,wght, isFront);
}

void gatherpointLauncher(int b,int n,int m,const float * inp,const int * idx,float * out){
  gatherpointKernel<<<dim3(2,8,1),512>>>(b,n,m,inp,idx,out);
}
void scatteraddpointLauncher(int b,int n,int m,const float * out_g,const int * idx,float * inp_g){
  scatteraddpointKernel<<<dim3(2,8,1),512>>>(b,n,m,out_g,idx,inp_g);
}