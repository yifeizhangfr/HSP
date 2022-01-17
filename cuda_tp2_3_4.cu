#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <iostream>

# define GRID 128
#define IMAGE_SIZE_IN 1*28*28
#define IMAGE_SIZE_PAD 1*32*32
#define CONV1_W_SIZE 6*1*5*5
#define CONV1_OUT_SIZE 6*28*28

#define POOL1_OUT_SIZE 6*14*14

#define CONV2_W_SIZE 16*6*5*5
#define CONV2_OUT_SIZE 16*10*10

#define POOL2_OUT_SIZE 16*5*5

#define FC1_W_SIZE 120*400
#define FC1_B_SIZE 120
#define FC1_OUT_SIZE 120

#define FC2_W_SIZE 120*84
#define FC2_B_SIZE 84
#define FC2_OUT_SIZE 84

#define FC3_W_SIZE 10*84
#define FC3_B_SIZE 10
#define FC3_OUT_SIZE 10


void MatrixInit(float *M, int n, int p){
    //printf("n = %d  \n", n);
    //printf("p = %d  \n", p);
    for(int i=0; i<n; i++){
        for(int j=0; j<p; j++){
            M[i*p+j] = ((float)rand()/RAND_MAX);
            
        }
    }
}

void MatrixInitMinMax(float *M, int n, int p, float min, float max){
    //printf("n = %d  \n", n);
    //printf("p = %d  \n", p);
    for(int i=0; i<n; i++){
        for(int j=0; j<p; j++){
            M[i*p+j] = ((float)rand()/RAND_MAX)*(max-min)+min;
            
        }
    }
}

void MatrixInitOnes(float *M, int n, int p){
    //printf("n = %d  \n", n);
    //printf("p = %d  \n", p);
    for(int i=0; i<n; i++){
        for(int j=0; j<p; j++){
            M[i*p+j] = 1.0;
            
        }
    }
}
void MatrixInitDiag(float *M, int n, int p){
    //printf("n = %d  \n", n);
    //printf("p = %d  \n", p);
    for(int i=0; i<n; i++){
        for(int j=0; j<p; j++){
            M[i*p+j] = i+j;
            
        }
    }
}


void MatrixInit3d(float *M, int n, int m, int p){
    //printf("n = %d  \n", n);
    //printf("m = %d  \n", m);
    //printf("p = %d  \n", p);
    for(int i=0; i<n; i++){
        for(int j=0; j<m; j++){
            for(int k=0; k<p; k++){
                M[i*m*p+j*p+k] = ((float)0);
            } 
        }
    }
}

void MatrixInitKernel(float *M, int n, int m, int p){
    //printf("n = %d  \n", n);
    //printf("m = %d  \n", m);
    //printf("p = %d  \n", p);
    for(int i=0; i<n; i++){
        for(int j=0; j<m; j++){
            for(int k=0; k<p; k++){
                //M[i*m*p+j*p+k] = ((float)rand()/RAND_MAX);
                M[i*m*p+j*p+k] = ((float)0);
                if (j == 2&&k==2){
                     M[i*m*p+j*p+k] = ((float)1);
                }
            } 
        }
    }
}

void MatrixInitpooling(float *M, int n, int m, int p){
    //printf("n = %d  \n", n);
    //printf("m = %d  \n", m);
    //printf("p = %d  \n", p);
    for(int i=0; i<n; i++){
        for(int j=0; j<m; j++){
            for(int k=0; k<p; k++){
                //M[i*m*p+j*p+k] = ((float)rand()/RAND_MAX);
                M[i*m*p+j*p+k] = ((float)0.25);
            } 
        }
    }
}


void MatrixPrint(float *M, int n, int p){
    float * m = M;
    for(int i=0; i<n; i++){
        for(int j=0; j<p; j++){
            printf("%1.1f ", M[i*p+j]);
        }
        printf("\n");
    }
}

void MatrixPrint3d(float *F, int n, int m, int p){
    float * f = F;
    for(int i=0; i<n; i++){
        for(int j=0; j<m; j++){
            for(int k=0; k<p; k++){
                printf("%1.1f ",  F[i*m*p+j*p+k]);
            }
            printf("\n");
        }
        printf("----------------------------------------------------------------------------------------------\n");
    }
}


void padding(float *input, int isize, int ichan, float *output, int pad) {
  int ocol, orow, och;
  int osize=isize+pad+pad;
  for (och = 0; och < ichan; och++) {
    for (orow = 0; orow < osize; orow++) {
      for (ocol = 0; ocol < osize; ocol++) {
        *(output+och*osize*osize+orow*osize+ocol) = (float)0.0;
      }
    }
  }
  
  for (och = 0; och < ichan; och++) {
    for (orow = 0; orow < isize; orow++) {
      for (ocol = 0; ocol < isize; ocol++) {
        *(output+och*osize*osize+(orow+pad)*osize+(ocol+pad)) = *(input+och*isize*isize+orow*isize+ocol);
      }
    }
  }
}


__global__ void convolution_2D(float *in, float *out, float *mask, int maskwidth, int w, int h,int channel) {
    int K = blockIdx.x;
    int Col = threadIdx.x;
    int Row = threadIdx.y;
    if (Row < h&&Col < w&&K < channel) {
        float pixVal = 0;
        //start
        int startCol = Col - maskwidth / 2;
        int startRow = Row - maskwidth / 2;
        //caculate the res
        for (int i = 0; i < maskwidth; i++)
        {
            for (int j = 0; j < maskwidth; j++)
            {
                int curRow = startRow + i;
                int curCol = startCol + j;
                if (curRow > -1 && curRow<h&&curCol>-1 && curCol < w)
                {
                    pixVal += mask[i*maskwidth + j] * in[curRow*w + curCol];
                }
            }
        }
        out[K*(w-4)*(h-4) + Row*(w-4) + Col] = pixVal;
    }
}



__global__ void Average_pooling(float* bottom_data, const int height, const int width, 
    const int pooled_height,const int out_height,float* top_data)
{
    int x = blockIdx.x;
    int y = blockIdx.y;
    int i,j,u,v,index;
    int index2=x*gridDim.y*out_height*out_height+y*out_height*out_height;
    float s;
    for (i = 0; i < out_height; ++i)
        for (j = 0; j < out_height; ++j)
        {
            index=x*gridDim.y*height*width+y*height*width+i*pooled_height*width+j*pooled_height;
            s=0.0;
            for (u = 0; u < pooled_height&&(u+pooled_height*i)<height; ++u)
                for (v = 0; v < pooled_height&&(v+pooled_height*j)<width; ++v)
                    s = s + *(bottom_data+index+u*width+v)/4;
            *(top_data+index2)=s;
            ++index2;
        }
}




__global__ void tanh_activate(float *in, float *out, int w, int h,int channel) {
    int K = blockIdx.x;
    int Col = threadIdx.x;
    int Row = threadIdx.y;
    if (Row < h&&Col < w&&K < channel) {
        float pixVal = 0;
        pixVal = (2.f/(1 + expf(-2*in[K*w*h + Row*w + Col])) - 1);
        out[K*w*h + Row*w + Col] = pixVal;
    }
}

__global__ void classifier(float *input, int isize, float *output, int osize,
                           float *weight, float *bias, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int j;

    if (idx < N) {
        *(output + idx) = 0.0;
        for (j = 0; j < isize; j++) {
            *(output + idx) += *(weight + idx * isize + j) * *(input + j);
        }
        *(output + idx) += *(bias + idx);
    }
}


void read_params(char *path, float *array, int size) {
  int i;
  FILE *fp;
  if((fp = fopen( path , "r" )) == NULL ) {
    printf( "fileopen error\n" ) ;
    exit(1);
  }
  for (i = 0; i < size; i++)
    if (fscanf(fp, "%f\n", &array[i])!='\0');
  fclose(fp);
}

void read_img(char *path, float *array, int size) {
  int i;
  FILE *fp;
  if((fp = fopen( path , "r" )) == NULL ) {
    printf( "fileopen error\n" ) ;
    exit(1);
  }
  for (i = 0; i < size; i++)
    if (fscanf(fp, "%f\n", &array[i])!='\0');
  fclose(fp);
}

void print_params(char *name, float *array, int size) {
  int i;
  printf("%s\n", name);
  for (i = 0; i < 3; i++)
    printf("%f, ",array[i]);
  printf("... ");
  for (i = 2; i >= 0; i--)
    printf(", %f",array[size-i-1]);
  printf("\n");
  fflush(stdout);
}


void norm_image(float *image, int size) {
    int i;
    for (i = 0; i < size; i++) {
        *(image+i) = *(image+i)/255.0;
    }
}


void print_all_params(float *array, int size) {
  int i;
  for (i = 0; i < size; i++) {
    printf("%6d : %f\n", i, array[i]);
  }
  fflush(stdout);
}


void softmax(float *input, int isize) {
  int i;
  float sum = 0.0;

  printf("Softmax:\n");
  printf("  isize=%d\n", isize);
  
  for (i = 0; i < isize; i++) {
    sum += expf(*(input + i));
  }
  for (i = 0; i < isize; i++) {
    *(input+i) = expf(*(input + i)) / sum;
  }
  printf("\n");
  fflush(stdout);
}

int main(int argc, char **argv){
// set dimension of image
    int nx = 32;
    int ny = 32;
    int nxy = nx * ny;
    int image_Bytes = nxy * sizeof(float);
// set dimension of output1
    int n_out1_x = 28;
    int n_out1_y = 28;
    int n_out1_c = 6;
    int n_out1_xyc = n_out1_x * n_out1_y * n_out1_c;
    int out1_Bytes = n_out1_xyc * sizeof(float);
// set dimension of output2
    int n_out2_x = 14;
    int n_out2_y = 14;
    int n_out2_c = 6;
    int n_out2_xyc = n_out2_x * n_out2_y * n_out2_c;
    int out2_Bytes = n_out2_xyc * sizeof(float);
// set dimension of kernel1 6*5*5
    int n_kernel1_x = 5;
    int n_kernel1_y = 5;
    int n_kernel1_c = 6;
    int n_kernel1_xyc = n_kernel1_x * n_kernel1_y * n_kernel1_c;
    int kernel1_Bytes = n_kernel1_xyc * sizeof(float);
// set dimension of kernel1 6*2*2
    int n_pooling_x = 2;
    int n_pooling_y = 2;
    int n_pooling_c = 6;
    int n_pooling_xyc = n_pooling_x * n_pooling_y * n_pooling_c;
    int pooling_Bytes = 14*14*6 * sizeof(float);
    
    
    
    
// malloc host dimension
    float *image;
    float *out1, *out2, *kernel1, *pooling;
    image = (float *)malloc(image_Bytes);
    out1 = (float *)malloc(out1_Bytes);
    out2 = (float *)malloc(out2_Bytes);
    kernel1 = (float *)malloc(kernel1_Bytes);
    float *res_1D = (float*)malloc(out1_Bytes);
    pooling = (float *)malloc(pooling_Bytes);
    float *res_1D_activation = (float*)malloc(out1_Bytes);
    
//init image data 32*32 and print
    MatrixInitDiag(image, nx, ny);
    printf("the image matrix:\n");
    MatrixPrint(image, nx, ny);
    
//init out1 data 6*28*28 and print
    MatrixInit3d(out1, n_out1_c, n_out1_x,n_out1_y);
    //printf("the out1 matrix:\n");
    //MatrixPrint3d(out1, n_out1_c, n_out1_x,n_out1_y);
    
//init out2 data 6*14*14 and print
    MatrixInit3d(out2, n_out2_c, n_out2_x,n_out2_y);
    //printf("the out2 matrix:\n");
    //MatrixPrint3d(out2, n_out2_c, n_out2_x,n_out2_y);
    
//init kernel1 data 6*5*5 and print
    MatrixInitKernel(kernel1, n_kernel1_c, n_kernel1_x, n_kernel1_y);
    //printf("the kernel1 matrix:\n");
    //MatrixPrint3d(kernel1, n_kernel1_c, n_kernel1_x, n_kernel1_y);

//init pooling data 6*2*2 and print
    MatrixInitpooling(pooling, n_pooling_c, n_pooling_x, n_pooling_y);
    //printf("the pooling matrix:\n");
    //MatrixPrint3d(pooling, n_pooling_c, n_pooling_x, n_pooling_y);
    
    
//cuda
    float *inD, *outD, *maskD, *act_out;
    cudaMalloc((void**)&inD, sizeof(float)*nxy);
    cudaMalloc((void**)&outD, sizeof(float)*n_out1_xyc);
    cudaMalloc((void**)&maskD, sizeof(float*)*n_kernel1_xyc);
    cudaMalloc((void**)&act_out, sizeof(float*)*28*28*6);
//copy
    cudaMemcpy(inD, image, sizeof(float)*nxy, cudaMemcpyHostToDevice);
    cudaMemcpy(outD, out1, sizeof(float)*n_out1_xyc, cudaMemcpyHostToDevice);
    cudaMemcpy(maskD, kernel1, sizeof(float)*n_kernel1_xyc, cudaMemcpyHostToDevice);

    dim3 dimGrid(6, 1, 1);
    dim3 dimBlock(28, 28, 1);
    convolution_2D << <dimGrid, dimBlock >> >(inD, outD, maskD, 5, 32, 32, 6);
    tanh_activate << <dimGrid, dimBlock >> >(outD, act_out, 28, 28, 6);
    cudaMemcpy(res_1D, outD, sizeof(float)*n_out1_xyc, cudaMemcpyDeviceToHost);
    cudaMemcpy(res_1D_activation, act_out, sizeof(float)*n_out1_xyc, cudaMemcpyDeviceToHost);
    printf("the res1D matrix:\n");
    MatrixPrint3d(res_1D, n_out1_c, n_out1_x,n_out1_y);
    printf("the res1D_activation matrix:\n");
    MatrixPrint3d(res_1D_activation, n_out1_c, n_out1_x,n_out1_y);
//activation function tanh
    
//pooling
    float *output_pooling, *input_pooling, *pooling_filter;
    cudaMalloc((void**)&output_pooling, sizeof(float*)*6*14*14);
    cudaMalloc((void**)&input_pooling, sizeof(float*)*6*28*28);
    cudaMalloc((void**)&pooling_filter, sizeof(float*)*n_pooling_xyc);
    cudaMemcpy(output_pooling, out2, sizeof(float)*6*14*14, cudaMemcpyHostToDevice);
    cudaMemcpy(input_pooling, res_1D_activation, sizeof(float)*n_out1_xyc, cudaMemcpyHostToDevice);
    cudaMemcpy(pooling_filter, pooling, sizeof(float)*n_pooling_xyc, cudaMemcpyHostToDevice);
    
    dim3 grid(1,6);
    Average_pooling<<<grid,1>>>(input_pooling,28,28,2,14,output_pooling);
    cudaMemcpy(pooling, output_pooling, sizeof(float)*6*14*14, cudaMemcpyDeviceToHost);
    
    printf("the pooling matrix:\n");
    MatrixPrint3d(pooling,6, 14, 14);

   
    
    
    //network
    float *image_in;
    float *image_pad;
    float *image_d;
    float *conv1_w;
    float *conv1_w_d, *conv1_out_d;
    float *act1_out_d;
    float *pool1_out_d;
  
    float *conv2_w, *conv2_b;
    float *conv2_w_d, *conv2_b_d, *conv2_out_d;
    float *act2_out_d;
    float *pool2_out_d;

    float *fc1_w, *fc1_b, *fc1_out;
    float *fc1_w_d, *fc1_b_d, *fc1_out_d;
    float *fc2_w, *fc2_b, *fc2_out;
    float *fc2_w_d, *fc2_b_d, *fc2_out_d;
    float *fc3_w, *fc3_b, *fc3_out;
    float *fc3_w_d, *fc3_b_d, *fc3_out_d;
    
    
    if ((image_in = (float *)malloc(sizeof(float)*IMAGE_SIZE_IN)) == NULL ||
        (image_pad = (float *)malloc(sizeof(float)*IMAGE_SIZE_PAD)) == NULL ||
        (conv1_w = (float *)malloc(sizeof(float)*CONV1_W_SIZE)) == NULL ||
        // (conv1_out = (float *)malloc(sizeof(float)*CONV1_OUT_SIZE)) == NULL ||
        // (pool1_out = (float *)malloc(sizeof(float)*POOL1_OUT_SIZE)) == NULL ||

        (conv2_w = (float *)malloc(sizeof(float)*CONV2_W_SIZE)) == NULL ||
        // (conv2_out = (float *)malloc(sizeof(float)*CONV2_OUT_SIZE)) == NULL ||
        // (pool2_out = (float *)malloc(sizeof(float)*POOL2_OUT_SIZE)) == NULL ||

        (fc1_w = (float *)malloc(sizeof(float)*FC1_W_SIZE)) == NULL ||
        (fc1_b = (float *)malloc(sizeof(float)*FC1_B_SIZE)) == NULL ||
        // (fc1_out = (float *)malloc(sizeof(float)*FC1_OUT_SIZE)) == NULL ||
        (fc2_w = (float *)malloc(sizeof(float)*FC2_W_SIZE)) == NULL ||
        (fc2_b = (float *)malloc(sizeof(float)*FC2_B_SIZE)) == NULL ||
        (fc3_w = (float *)malloc(sizeof(float)*FC3_W_SIZE)) == NULL ||
        (fc3_b = (float *)malloc(sizeof(float)*FC3_B_SIZE)) == NULL ||
        (fc3_out = (float *)malloc(sizeof(float)*FC3_OUT_SIZE)) == NULL ||
        0) {
        printf("MemError\n");
        exit(1);
    }
    
    cudaMalloc((void **)&image_d, sizeof(float)*IMAGE_SIZE_PAD);
    cudaMalloc((void **)&conv1_w_d, sizeof(float)*CONV1_W_SIZE);
    
    cudaMalloc((void **)&conv1_out_d, sizeof(float)*CONV1_OUT_SIZE);
    cudaMalloc((void **)&act1_out_d, sizeof(float)*CONV1_OUT_SIZE);
    cudaMalloc((void **)&pool1_out_d, sizeof(float)*POOL1_OUT_SIZE);
    cudaMalloc((void **)&conv2_w_d, sizeof(float)*CONV2_W_SIZE);

    cudaMalloc((void **)&conv2_out_d, sizeof(float)*CONV2_OUT_SIZE);
    cudaMalloc((void **)&act2_out_d, sizeof(float)*CONV2_OUT_SIZE);
    cudaMalloc((void **)&pool2_out_d, sizeof(float)*POOL2_OUT_SIZE);
    cudaMalloc((void **)&fc1_w_d, sizeof(float)*FC1_W_SIZE);
    cudaMalloc((void **)&fc1_b_d, sizeof(float)*FC1_B_SIZE);
    cudaMalloc((void **)&fc1_out_d, sizeof(float)*FC1_OUT_SIZE);
    cudaMalloc((void **)&fc2_w_d, sizeof(float)*FC2_W_SIZE);
    cudaMalloc((void **)&fc2_b_d, sizeof(float)*FC2_B_SIZE);
    cudaMalloc((void **)&fc2_out_d, sizeof(float)*FC2_OUT_SIZE);
    cudaMalloc((void **)&fc3_w_d, sizeof(float)*FC3_W_SIZE);
    cudaMalloc((void **)&fc3_b_d, sizeof(float)*FC3_B_SIZE);
    cudaMalloc((void **)&fc3_out_d, sizeof(float)*FC3_OUT_SIZE);
    printf("\n");

    
    
    read_params("conv1_w.txt", conv1_w, CONV1_W_SIZE);
    print_params("CONV1_W : ", conv1_w, CONV1_W_SIZE);
    //Read CONV2 params
    read_params("conv2_w.txt", conv2_w, CONV2_W_SIZE);
    print_params("CONV2_W : ", conv2_w, CONV2_W_SIZE);
    //Read FC1 params
    read_params("fc1_w.txt", fc1_w, FC1_W_SIZE);
    print_params("FC1_W : ", fc1_w, FC1_W_SIZE);
    read_params("fc1_b.txt", fc1_b, FC1_B_SIZE);
    print_params("FC1_B : ", fc1_b, FC1_B_SIZE);
    //Read FC2 params
    read_params("fc2_w.txt", fc2_w, FC2_W_SIZE);
    print_params("FC2_W : ", fc2_w, FC2_W_SIZE);
    read_params("fc2_b.txt", fc2_b, FC2_B_SIZE);
    print_params("FC2_B : ", fc2_b, FC2_B_SIZE);
    //Read FC3 params
    read_params("fc3_w.txt", fc3_w, FC3_W_SIZE);
    print_params("FC3_W : ", fc3_w, FC3_W_SIZE);
    read_params("fc3_b.txt", fc3_b, FC3_B_SIZE);
    print_params("FC3_B : ", fc3_b, FC3_B_SIZE);
    printf("\n");
    
    
    int i, j= 0;
    
    read_params("image1st.txt", image_in, IMAGE_SIZE_IN);
    norm_image(image_in, IMAGE_SIZE_IN);
    cudaMemcpy(image_d, image_in, sizeof(float)*IMAGE_SIZE_IN,
                    cudaMemcpyHostToDevice);
    padding(image_in, 28, 1, image_pad, 2);

//show iamge
    for (i = 0; i < 32; i++) {
        for (j = 0; j < 32; j++) {
                if (*(image_pad+i*32+j) > 0.5){
                printf ("* ");
                    
            } else {
                printf("  ");
            }
        }
        printf ("\n");
    }
    
    // Copy to GPU
    cudaMemcpy(conv1_w_d, conv1_w, sizeof(float)*CONV1_W_SIZE,
               cudaMemcpyHostToDevice);
    cudaMemcpy(conv2_w_d, conv2_w, sizeof(float)*CONV2_W_SIZE,
                    cudaMemcpyHostToDevice);
    cudaMemcpy(fc1_w_d, fc1_w, sizeof(float)*FC1_W_SIZE,
                    cudaMemcpyHostToDevice);
    cudaMemcpy(fc1_b_d, fc1_b, sizeof(float)*FC1_B_SIZE,
                    cudaMemcpyHostToDevice);
    cudaMemcpy(fc2_w_d, fc2_w, sizeof(float)*FC2_W_SIZE,
                    cudaMemcpyHostToDevice);
    cudaMemcpy(fc2_b_d, fc2_b, sizeof(float)*FC2_B_SIZE,
                    cudaMemcpyHostToDevice);
    cudaMemcpy(fc3_w_d, fc3_w, sizeof(float)*FC3_W_SIZE,
                    cudaMemcpyHostToDevice);
    cudaMemcpy(fc3_b_d, fc3_b, sizeof(float)*FC3_B_SIZE,
                    cudaMemcpyHostToDevice);
    
    cudaMemcpy(image_d, image_pad, sizeof(float)*IMAGE_SIZE_PAD,
                    cudaMemcpyHostToDevice);
    
    dim3 dimGrid1(6, 1, 1);
    dim3 dimBlock1(28, 28, 1);
    convolution_2D << <dimGrid1, dimBlock1 >> >(image_d, conv1_out_d, conv1_w_d, 5, 32, 32, 6);
    
    tanh_activate << <dimGrid1, dimBlock1 >> >(conv1_out_d, act1_out_d, 28, 28, 6);
    dim3 grid1(1,6);
    Average_pooling<<<grid1,1>>>(act1_out_d,28,28,2,14,pool1_out_d);
    
    dim3 dimGrid2(16, 1, 1);
    dim3 dimBlock2(10, 10, 1);
    convolution_2D << <dimGrid2, dimBlock2 >> >(pool1_out_d, conv2_out_d, conv2_w_d, 5, 14, 14, 16);
    tanh_activate << <dimGrid2, dimBlock2 >> >(conv2_out_d, act2_out_d, 10, 10, 16);
    dim3 grid2(1,16);
    Average_pooling<<<grid2,1>>>(act2_out_d,10,10,2,5,pool2_out_d);
    printf( "************************************************************************\n" );
    classifier <<< GRID, FC1_OUT_SIZE / GRID + 1 >>>
        (pool2_out_d, 400, fc1_out_d, 120, fc1_w_d, fc1_b_d, FC1_OUT_SIZE);//FC1
    
    
    classifier <<< GRID, FC1_OUT_SIZE / GRID + 1 >>>
        (fc1_out_d, 120, fc2_out_d, 84, fc2_w_d, fc2_b_d, FC2_OUT_SIZE);//FC1
    classifier <<< GRID, FC1_OUT_SIZE / GRID + 1 >>>
        (fc2_out_d, 84, fc3_out_d, 10, fc3_w_d, fc3_b_d, FC3_OUT_SIZE);//FC1
    printf( "************************************************************************\n" );
    cudaMemcpy(fc3_out, fc3_out_d, FC3_OUT_SIZE * sizeof(float),
        cudaMemcpyDeviceToHost);
    printf( "************************************************************************\n" );
    softmax(fc3_out, 10);
    print_all_params(fc3_out, 10);
}