
#include <iostream>
#include <fstream>

// grid size
int const NX = 256;
int const NY = 256;

int const MaxIter = 2000; // number of Jacobi iterations

void checkError(cudaError_t e)
{
   if (e != cudaSuccess)
   {
      std::cerr << "CUDA error: " << int(e) << " : " << cudaGetErrorString(e) << '\n';
      abort();
   }
}

// CUDA kernel to update the Tnew array based on the Told
// using Jacobi iterate of the Laplace operator

__global__
void Laplace(float* Told, float* Tnew)
{
   // compute the x,y location of the node point
   // handled by this thread

   int x = blockIdx.x*blockDim.x + threadIdx.x ;
   int y = blockIdx.y*blockDim.y + threadIdx.y ;

   // get the natural index values of node (i,j) and its neighboring nodes
   //                         N
   int P = y*NX + x;           // node (x,j)              |
   int E = y*NX + (x+1);       // node (x+1,y)            |
   int W = y*NX + (x-1);       // node (x-1,y)     W ---- P ---- E
   int N = (y+1)*NX + x;       // node (x,y+1)            |
   int S = (y-1)*NX + x;       // node (x,y-1)            |
   //                         S

   // only update the interior points - fixed boundary conditions
   if(x > 0 && x < (NX-1) && y > 0 && y < (NY-1))
   {
      Tnew[P] = 0.25*(Told[E] + Told[W] + Told[N] + Told[S]);
   }
}

// Initial condition

void Initialize(float* T)
{
   for(int x=0; x < NX; ++x)
   {
      for(int y=0; y < NY; ++y)
      {
         int index = y*NX + x;
         T[index] = 0.0;
      }
   }

    // set left wall to 1
    for(int y=0; y < NY; ++y)
    {
       int index = y*NX;
       T[index] = 1.0;
    }
}

int main()
{
   // The temperature matrix on the CPU
   float* T = new float[NX*NY];


   // Storage on the device (GPU)
   float* T1_device;
   float* T2_device;

    // initialize array on the host
    Initialize(T);

    // allocate storage space on the GPU
    checkError(cudaMalloc(&T1_device, NX*NY*sizeof(float)));
    checkError(cudaMalloc(&T2_device, NX*NY*sizeof(float)));

    // copy (initialized) host arrays to the GPU memory from CPU memory
    checkError(cudaMemcpy(T1_device, T, NX*NY*sizeof(float), cudaMemcpyHostToDevice));
    checkError(cudaMemcpy(T2_device, T, NX*NY*sizeof(float), cudaMemcpyHostToDevice));

    // assign a 2D distribution of CUDA "threads" within each CUDA "block"
    int ThreadsPerBlock=16;
    dim3 dimBlock(ThreadsPerBlock, ThreadsPerBlock);

    // calculate number of blocks along X and Y in a 2D CUDA "grid"
    dim3 dimGrid((NX+dimBlock.x-1) / dimBlock.x, (NY+dimBlock.y-1) / dimBlock.y);

    // begin Jacobi iteration
    for (int k = 0; k < MaxIter; k += 2)
    {
        Laplace<<<dimGrid, dimBlock>>>(T1_device, T2_device);   // update T1 using data stored in T2
        Laplace<<<dimGrid, dimBlock>>>(T2_device, T1_device);   // update T2 using data stored in T1
    }

    // copy final array to the CPU from the GPU
    checkError(cudaMemcpy(T, T2_device, NX*NY*sizeof(float),cudaMemcpyDeviceToHost));
    cudaDeviceSynchronize();

    // print the results to a file
    std::ofstream Out("temperature.dat");
    for (int y = 0; y < NY; ++y)
    {
       for (int x = 0; x < NX; ++x)
       {
          int index = y*NX + x;
          Out << x << ' ' << y << ' ' << T[index] << '\n';
       }
       Out << '\n';
    }
    Out.close();

    // release memory
    delete[] T;
    cudaFree(T1_device);
    cudaFree(T2_device);
}

