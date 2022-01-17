#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>


void MatrixInit(float *M, int n, int p){
    printf("n = %d  \n", n);
    printf("p = %d  \n", p);
    for(int i=0; i<n; i++){
        for(int j=0; j<p; j++){
            M[i*p+j] = ((float)rand()/RAND_MAX*2-1);
            
        }
    }
}

void MatrixPrint(float *M, int n, int p){
    float * m = M;
    for(int i=0; i<n; i++){
        for(int j=0; j<p; j++){
            printf("%f \n", M[i*p+j]);
        }     
    }
}

void MatrixAdd(float *M1, float *M2, float *Mout, int n, int p){
    float *m1 = M1,*m2 = M2, *mout = Mout;
    for(int i=0; i<n; i++){
        for(int j=0; j<p; j++){
            mout[i*p+j] = m1[i*p+j] + m2[i*p+j];
        }    
    }
}

__global__ void cudaMatrixAdd(float *M1, float *M2, float *Mout, int n, int p){
    int ix = threadIdx.x + blockDim.x * blockIdx.x;
    int iy = threadIdx.y + blockDim.y * blockIdx.y;
    unsigned int idx = iy * n + ix;
    
    if(ix < n && iy < p){
        Mout[idx] = M1[idx] + M2[idx];
    }
}


void MatrixMult(float *M1, float *M2, float *Mout, int n){
    float *m1 = M1, *m2 = M2, *mout = Mout, *sum;
    for(int i=0; i<n; i++){
        for(int j=0; j<n; j++){
            float sum=0.0;
            for(int k=0; k<n; k++){
                sum+=m1[i*n+k]*m2[k*n+j];
            }    
            mout[i*n+j] = sum;
        }
    }
}

__global__ void cudaMatrixMult(float *M1, float *M2, float *Mout, int n){
    int x = threadIdx.x;
    int y = threadIdx.y;
    float sum = 0.0;
    float elem1 = 0.0, elem2=0.0,value=0.0;
    for (int i=0; i < n;i++){
        elem1 = M1[y * n + i];
        elem2 = M2[i * n + x];
        value += elem1 * elem2;
    }
    Mout[y*n + x] = value;
}



    
int main(int argc, char **argv){
    int dev = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    printf("using device %d: %s\n\n", dev, deviceProp.name);
    
        
    // set dimension
    int nx = 500;
    int ny = 500;
    int nxy = nx * ny;
    int nBytes = nxy * sizeof(float);
    
    // malloc host dimension
    float *hA, *hB, *hC, *hCC, *hD, *hDD;
    hA = (float *)malloc(nBytes);
    hB = (float *)malloc(nBytes);
    hC = (float *)malloc(nBytes);
    hCC = (float *)malloc(nBytes);
    hD = (float *)malloc(nBytes);
    hDD = (float *)malloc(nBytes);
        
    //initialize matrice
    MatrixInit(hA, nx, ny);
    MatrixInit(hB, nx, ny);
    printf("the first matrix:\n");
    MatrixPrint(hA, nx, ny);
    printf("the second matrix:\n");
    MatrixPrint(hB, nx, ny);
    
    //Addition de deux matrices sur CPU
    clock_t cpuStart = clock();
    MatrixAdd(hA,hB,hC, nx, ny);
    clock_t cpuEnd = clock();
    float cpuTime = (float)(cpuEnd - cpuStart) / CLOCKS_PER_SEC;
    printf("the addition of two matrix in cpu:\n");
    //MatrixPrint(hC, nx, ny);
    printf("cpu time:%f\n", cpuTime);
    //Addition de deux matrices sur GPU
    
    // malloc device memory
    float *d_MatA, *d_MatB, *d_MatC,*d_MatD;
    cudaMalloc((void **)&d_MatA, nBytes);
    cudaMalloc((void **)&d_MatB, nBytes);
    cudaMalloc((void **)&d_MatC, nBytes);
    cudaMalloc((void **)&d_MatD, nBytes);
    
    clock_t gpuStart = clock();
    cudaMemcpy(d_MatA, hA, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_MatB, hB, nBytes, cudaMemcpyHostToDevice);
    
    int dimx = 32;
    int dimy = 32;
    dim3 block(dimx, dimy);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);
    
    
    cudaMatrixAdd << <grid, block >> > (d_MatA, d_MatB, d_MatC, nx, ny);
    cudaDeviceSynchronize();
    cudaMemcpy(hCC, d_MatC, nBytes, cudaMemcpyDeviceToHost);
    printf("the addition of two matrix in gpu:\n");
    //MatrixPrint(hCC, nx, ny);
    clock_t gpuEnd = clock();
    float gpuTime = (float)(gpuEnd - gpuStart) / CLOCKS_PER_SEC;
    printf("gpu time:%f\n", gpuTime);
    
    
    clock_t cpuStart1 = clock();
    MatrixMult(hA,hB,hD,nx);
    clock_t cpuEnd1 = clock();
    float cpuTime1 = (float)(cpuEnd1 - cpuStart1) / CLOCKS_PER_SEC;
    printf("the multiplication of two matrix in cpu:\n");
    //MatrixPrint(hD, nx, ny);
    printf("cpu time:%f\n", cpuTime1);
    
    clock_t gpuStart1 = clock();
    cudaMemcpy(d_MatA, hA, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_MatB, hB, nBytes, cudaMemcpyHostToDevice);
    
    cudaMatrixMult<<<grid, block >>>(d_MatA,d_MatB,d_MatD,nx);
    cudaThreadSynchronize();
    cudaMemcpy(hDD, d_MatD, nBytes, cudaMemcpyDeviceToHost);
    printf("the multiplication of two matrix in gpu:\n");
    clock_t gpuEnd1 = clock();
    float gpuTime1 = (float)(gpuEnd1 - gpuStart1) / CLOCKS_PER_SEC;
    //MatrixPrint(hDD, nx, ny);
    printf("gpu time:%f\n", gpuTime1);
    return 0;
}