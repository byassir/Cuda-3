#include "utils.h"
//#include "timer.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <string>
#include <cmath>
#include <iostream>

#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/reduce.h>
#include <thrust/extrema.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/sequence.h>

#include "loadSaveImage.h"
#include <stdio.h>

struct GpuTimer
{
  cudaEvent_t start;
  cudaEvent_t stop;

  GpuTimer()
  {
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
  }

  ~GpuTimer()
  {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
  }

  void Start()
  {
    cudaEventRecord(start, 0);
  }

  void Stop()
  {
    cudaEventRecord(stop, 0);
  }

  float Elapsed()
  {
    float elapsed;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    return elapsed;
  }
};

struct splitChannels : thrust::unary_function<uchar4, thrust::tuple<unsigned char, unsigned char, unsigned char> >{
  __host__ __device__
  thrust::tuple<unsigned char, unsigned char, unsigned char> operator()(uchar4 pixel) {
    return thrust::make_tuple(pixel.x, pixel.y, pixel.z);
  }
};

struct combineChannels : thrust::unary_function<thrust::tuple<unsigned char, unsigned char, unsigned char>, uchar4> {
  __host__ __device__
  uchar4 operator()(thrust::tuple<unsigned char, unsigned char, unsigned char> t) {
    return make_uchar4(thrust::get<0>(t), thrust::get<1>(t), thrust::get<2>(t), 255);
  }
};

struct combineResponses : thrust::unary_function<float, thrust::tuple<float, float, float> > {
  __host__ __device__
  float operator()(thrust::tuple<float, float, float> t) {
    return thrust::get<0>(t) * thrust::get<1>(t) * thrust::get<2>(t);
  }
};

//we need to save the input so we can remove the redeye for the output
static thrust::device_vector<unsigned char> d_red;
static thrust::device_vector<unsigned char> d_blue;
static thrust::device_vector<unsigned char> d_green;

static size_t numRowsImg;
static size_t numColsImg;
static size_t templateHalfWidth;
static size_t templateHalfHeight;

//simple cross correlation kernel copied from Mike's IPython Notebook
__global__ void naive_normalized_cross_correlation(
    float*         d_response,
    unsigned char* d_original,
    unsigned char* d_template,
    int            num_pixels_y,
    int            num_pixels_x,
    int            template_half_height,
    int            template_height,
    int            template_half_width,
    int            template_width,
    int            template_size,
    float          template_mean
    )
{
  int  ny             = num_pixels_y;
  int  nx             = num_pixels_x;
  int  knx            = template_width;
  int2 image_index_2d = make_int2( ( blockIdx.x * blockDim.x ) + threadIdx.x, ( blockIdx.y * blockDim.y ) + threadIdx.y );
  int  image_index_1d = ( nx * image_index_2d.y ) + image_index_2d.x;

  if ( image_index_2d.x < nx && image_index_2d.y < ny )
  {
    //
    // compute image mean
    //
    float image_sum = 0.0f;

    for ( int y = -template_half_height; y <= template_half_height; y++ )
    {
      for ( int x = -template_half_width; x <= template_half_width; x++ )
      {
        int2 image_offset_index_2d         = make_int2( image_index_2d.x + x, image_index_2d.y + y );
        int2 image_offset_index_2d_clamped = make_int2( min( nx - 1, max( 0, image_offset_index_2d.x ) ), min( ny - 1, max( 0, image_offset_index_2d.y ) ) );
        int  image_offset_index_1d_clamped = ( nx * image_offset_index_2d_clamped.y ) + image_offset_index_2d_clamped.x;

        unsigned char image_offset_value = d_original[ image_offset_index_1d_clamped ];

        image_sum += (float)image_offset_value;
      }
    }

    float image_mean = image_sum / (float)template_size;

    //
    // compute sums
    //
    float sum_of_image_template_diff_products = 0.0f;
    float sum_of_squared_image_diffs          = 0.0f;
    float sum_of_squared_template_diffs       = 0.0f;

    for ( int y = -template_half_height; y <= template_half_height; y++ )
    {
      for ( int x = -template_half_width; x <= template_half_width; x++ )
      {
        int2 image_offset_index_2d         = make_int2( image_index_2d.x + x, image_index_2d.y + y );
        int2 image_offset_index_2d_clamped = make_int2( min( nx - 1, max( 0, image_offset_index_2d.x ) ), min( ny - 1, max( 0, image_offset_index_2d.y ) ) );
        int  image_offset_index_1d_clamped = ( nx * image_offset_index_2d_clamped.y ) + image_offset_index_2d_clamped.x;

        unsigned char image_offset_value = d_original[ image_offset_index_1d_clamped ];
        float         image_diff         = (float)image_offset_value - image_mean;

        int2 template_index_2d = make_int2( x + template_half_width, y + template_half_height );
        int  template_index_1d = ( knx * template_index_2d.y ) + template_index_2d.x;

        unsigned char template_value = d_template[ template_index_1d ];
        float         template_diff  = template_value - template_mean;

        float image_template_diff_product = image_offset_value   * template_diff;
        float squared_image_diff          = image_diff           * image_diff;
        float squared_template_diff       = template_diff        * template_diff;

        sum_of_image_template_diff_products += image_template_diff_product;
        sum_of_squared_image_diffs          += squared_image_diff;
        sum_of_squared_template_diffs       += squared_template_diff;
      }
    }


    //
    // compute final result
    //
    float result_value = 0.0f;

    if ( sum_of_squared_image_diffs != 0 && sum_of_squared_template_diffs != 0 )
    {
      result_value = sum_of_image_template_diff_products / sqrt( sum_of_squared_image_diffs * sum_of_squared_template_diffs );
    }

    d_response[ image_index_1d ] = result_value;
  }
}


__global__ void remove_redness_from_coordinates(
    const unsigned int*  d_coordinates,
    unsigned char* d_r,
    unsigned char* d_b,
    unsigned char* d_g,
    unsigned char* d_r_output,
    int    num_coordinates,
    int    num_pixels_y,
    int    num_pixels_x,
    int    template_half_height,
    int    template_half_width
    )
{
  int ny              = num_pixels_y;
  int nx              = num_pixels_x;
  int global_index_1d = ( blockIdx.x * blockDim.x ) + threadIdx.x;

  int imgSize = num_pixels_x * num_pixels_y;

  if ( global_index_1d < num_coordinates )
  {
    unsigned int image_index_1d = d_coordinates[ imgSize - global_index_1d - 1 ];
    ushort2 image_index_2d = make_ushort2(image_index_1d % num_pixels_x, image_index_1d / num_pixels_x);

    for ( int y = image_index_2d.y - template_half_height; y <= image_index_2d.y + template_half_height; y++ )
    {
      for ( int x = image_index_2d.x - template_half_width; x <= image_index_2d.x + template_half_width; x++ )
      {
        int2 image_offset_index_2d         = make_int2( x, y );
        int2 image_offset_index_2d_clamped = make_int2( min( nx - 1, max( 0, image_offset_index_2d.x ) ), min( ny - 1, max( 0, image_offset_index_2d.y ) ) );
        int  image_offset_index_1d_clamped = ( nx * image_offset_index_2d_clamped.y ) + image_offset_index_2d_clamped.x;

        unsigned char g_value = d_g[ image_offset_index_1d_clamped ];
        unsigned char b_value = d_b[ image_offset_index_1d_clamped ];

        unsigned int gb_average = ( g_value + b_value ) / 2;

        d_r_output[ image_offset_index_1d_clamped ] = (unsigned char)gb_average;
      }
    }

  }
}

//return types are void since any internal error will be handled by quitting
//no point in returning error codes...
void preProcess(unsigned int **inputVals,
                unsigned int **inputPos,
                unsigned int **outputVals,
                unsigned int **outputPos,
                size_t &numElem,
                const std::string& filename) {
  //make sure the context initializes ok
  checkCudaErrors(cudaFree(0));

  uchar4 *inImg;
  uchar4 *eyeTemplate;

  size_t numRowsTemplate, numColsTemplate;

  std::string templateFilename("red_eye_effect_template_5.jpg");

  loadImageRGBA(filename, &inImg, &numRowsImg, &numColsImg);
  loadImageRGBA(templateFilename, &eyeTemplate, &numRowsTemplate, &numColsTemplate);

  templateHalfWidth = (numColsTemplate - 1) / 2;
  templateHalfHeight = (numRowsTemplate - 1) / 2;

  //we need to split each image into its separate channels
  //use thrust to demonstrate basic uses

  numElem = numRowsImg * numColsImg;
  size_t templateSize = numRowsTemplate * numColsTemplate;

  thrust::device_vector<uchar4> d_Img(inImg, inImg + numRowsImg * numColsImg);
  thrust::device_vector<uchar4> d_Template(eyeTemplate, eyeTemplate + numRowsTemplate * numColsTemplate);

  d_red.  resize(numElem);
  d_blue. resize(numElem);
  d_green.resize(numElem);

  thrust::device_vector<unsigned char> d_red_template(templateSize);
  thrust::device_vector<unsigned char> d_blue_template(templateSize);
  thrust::device_vector<unsigned char> d_green_template(templateSize);

  //split the image
  thrust::transform(d_Img.begin(), d_Img.end(), thrust::make_zip_iterator(
                                                  thrust::make_tuple(d_red.begin(),
                                                                     d_blue.begin(),
                                                                     d_green.begin())),
                                                splitChannels());

  //split the template
  thrust::transform(d_Template.begin(), d_Template.end(),
                    thrust::make_zip_iterator(thrust::make_tuple(d_red_template.begin(),
                                                                 d_blue_template.begin(),
                                                                 d_green_template.begin())),
                                                splitChannels());


  thrust::device_vector<float> d_red_response(numElem);
  thrust::device_vector<float> d_blue_response(numElem);
  thrust::device_vector<float> d_green_response(numElem);

  //need to compute the mean for each template channel
  unsigned int r_sum = thrust::reduce(d_red_template.begin(), d_red_template.end(), 0);
  unsigned int b_sum = thrust::reduce(d_blue_template.begin(), d_blue_template.end(), 0);
  unsigned int g_sum = thrust::reduce(d_green_template.begin(), d_green_template.end(), 0);

  float r_mean = (double)r_sum / templateSize;
  float b_mean = (double)b_sum / templateSize;
  float g_mean = (double)g_sum / templateSize;

  const dim3 blockSize(32, 8, 1);
  const dim3 gridSize( (numColsImg + blockSize.x - 1) / blockSize.x, (numRowsImg + blockSize.y - 1) / blockSize.y, 1);

  //now compute the cross-correlations for each channel

  naive_normalized_cross_correlation<<<gridSize, blockSize>>>(thrust::raw_pointer_cast(d_red_response.data()),
                                                              thrust::raw_pointer_cast(d_red.data()),
                                                              thrust::raw_pointer_cast(d_red_template.data()),
                                                              numRowsImg, numColsImg,
                                                              templateHalfHeight, numRowsTemplate,
                                                              templateHalfWidth, numColsTemplate,
                                                              numRowsTemplate * numColsTemplate, r_mean);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  naive_normalized_cross_correlation<<<gridSize, blockSize>>>(thrust::raw_pointer_cast(d_blue_response.data()),
                                                              thrust::raw_pointer_cast(d_blue.data()),
                                                              thrust::raw_pointer_cast(d_blue_template.data()),
                                                              numRowsImg, numColsImg,
                                                              templateHalfHeight, numRowsTemplate,
                                                              templateHalfWidth, numColsTemplate,
                                                              numRowsTemplate * numColsTemplate, b_mean);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  naive_normalized_cross_correlation<<<gridSize, blockSize>>>(thrust::raw_pointer_cast(d_green_response.data()),
                                                              thrust::raw_pointer_cast(d_green.data()),
                                                              thrust::raw_pointer_cast(d_green_template.data()),
                                                              numRowsImg, numColsImg,
                                                              templateHalfHeight, numRowsTemplate,
                                                              templateHalfWidth, numColsTemplate,
                                                              numRowsTemplate * numColsTemplate, g_mean);

  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  //generate combined response - multiply all channels together


  thrust::device_vector<float> d_combined_response(numElem);

  thrust::transform(thrust::make_zip_iterator(thrust::make_tuple(
                        d_red_response.begin(),
                        d_blue_response.begin(),
                        d_green_response.begin())),
                    thrust::make_zip_iterator(thrust::make_tuple(
                        d_red_response.end(),
                        d_blue_response.end(),
                        d_green_response.end())),
                    d_combined_response.begin(),
                    combineResponses());

  //find max/min of response

  typedef thrust::device_vector<float>::iterator floatIt;
  thrust::pair<floatIt, floatIt> minmax = thrust::minmax_element(d_combined_response.begin(), d_combined_response.end());

  float bias = *minmax.first;

  //we need to make all the numbers positive so that the students can sort them without any bit twiddling
  thrust::transform(d_combined_response.begin(), d_combined_response.end(), thrust::make_constant_iterator(-bias),
                    d_combined_response.begin(), thrust::plus<float>());

  //now we need to create the 1-D coordinates that will be attached to the keys
  thrust::device_vector<unsigned int> coords(numElem);
  thrust::sequence(coords.begin(), coords.end()); //[0, ..., numElem - 1]

  //allocate memory for output and copy since our device vectors will go out of scope
  //and be deleted
  checkCudaErrors(cudaMalloc(inputVals,  sizeof(unsigned int) * numElem));
  checkCudaErrors(cudaMalloc(inputPos,   sizeof(unsigned int) * numElem));
  checkCudaErrors(cudaMalloc(outputVals, sizeof(unsigned int) * numElem));
  checkCudaErrors(cudaMalloc(outputPos,  sizeof(unsigned int) * numElem));

  cudaMemcpy(*inputVals, thrust::raw_pointer_cast(d_combined_response.data()), sizeof(unsigned int) * numElem, cudaMemcpyDeviceToDevice);
  cudaMemcpy(*inputPos,  thrust::raw_pointer_cast(coords.data()), sizeof(unsigned int) * numElem, cudaMemcpyDeviceToDevice);
  checkCudaErrors(cudaMemset(*outputVals, 0, sizeof(unsigned int) * numElem));
  checkCudaErrors(cudaMemset(*outputPos, 0,  sizeof(unsigned int) * numElem));
}

void postProcess(const unsigned int* const outputVals,
                 const unsigned int* const outputPos,
                 const size_t numElems,
                 const std::string& output_file){

  thrust::device_vector<unsigned char> d_output_red = d_red;

  const dim3 blockSize(256, 1, 1);
  const dim3 gridSize( (40 + blockSize.x - 1) / blockSize.x, 1, 1);

  remove_redness_from_coordinates<<<gridSize, blockSize>>>(outputPos,
                                                           thrust::raw_pointer_cast(d_red.data()),
                                                           thrust::raw_pointer_cast(d_blue.data()),
                                                           thrust::raw_pointer_cast(d_green.data()),
                                                           thrust::raw_pointer_cast(d_output_red.data()),
                                                           40,
                                                           numRowsImg, numColsImg,
                                                           9, 9);


  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  //combine the new red channel with original blue and green for output
  thrust::device_vector<uchar4> d_outputImg(numElems);

  thrust::transform(thrust::make_zip_iterator(thrust::make_tuple(
                          d_output_red.begin(),
                          d_blue.begin(),
                          d_green.begin())),
                    thrust::make_zip_iterator(thrust::make_tuple(
                          d_output_red.end(),
                          d_blue.end(),
                          d_green.end())),
                    d_outputImg.begin(),
                    combineChannels());

  thrust::host_vector<uchar4> h_Img = d_outputImg;

  saveImageRGBA(&h_Img[0], numRowsImg, numColsImg, output_file);

  //Clear the global vectors otherwise something goes wrong trying to free them
  d_red.clear(); d_red.shrink_to_fit();
  d_blue.clear(); d_blue.shrink_to_fit();
  d_green.clear(); d_green.shrink_to_fit();
}

__global__
void gpuPrint(unsigned int* d_sum, unsigned char* d_predicate, unsigned int numElems){
	//printf("%d \n", d_cdf[0]);
	unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
	printf("%u \n", d_sum[gid]);

}

//calculates predicate and histogram
__global__
void predicate(unsigned int* const d_inputVals,
        	   unsigned char* d_predicate,
			   const size_t numElems,
        	   unsigned int mask,
        	   unsigned int bitPos){
	extern __shared__ unsigned int sdata[];
	unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int tid = threadIdx.x;

	if (gid < numElems){
		sdata[tid] = d_inputVals[gid];
		__syncthreads();
		d_predicate[gid] = (sdata[tid] & mask) >> bitPos;
	}

}

__global__
void blelloch_scan1(unsigned int* d_sum, unsigned int* d_auxSum, unsigned char* d_predicate, int blockSize, unsigned int numElems) {
	extern __shared__ unsigned int sdata2[];// allocated on invocation
	unsigned int tid = threadIdx.x;
	unsigned int sid = blockIdx.x * blockSize; //global id of first thread in block
	//printf("%i \n", tid);
	//printf("%i \n", d_predicate[2*tid]);
	//printf("%i \n", d_predicate[2*tid+1]);

	int offset = 1;

	if ((2*tid + sid) < numElems) {
		sdata2[2*tid] = d_predicate[sid + 2*tid]; // load input into shared memory
	}
	else {
		sdata2[2*tid] = 0; // load input into shared memory
	}
	if ((2*tid + 1 + sid) < numElems) {
			sdata2[2*tid+1] = d_predicate[sid + 2*tid+1];
		}
	else {
		sdata2[2*tid] = 0; // load input into shared memory
		sdata2[2*tid+1] = 0;
	}
	// build sum in place up the tree
	for (int d = blockSize>>1; d > 0; d >>= 1) {
		__syncthreads();
		if (tid < d) {
			int ai = offset*(2*tid+1)-1;
			int bi = offset*(2*tid+2)-1;
			sdata2[bi] += sdata2[ai];
		}
		offset *= 2;
	}
	if (tid == 0) { sdata2[blockSize - 1] = 0; } // clear the last element
	// traverse down tree & build scan
	for (int d = 1; d < blockSize; d *= 2) {
		offset >>= 1;
		__syncthreads();
		if (tid < d) {
			int ai = offset*(2*tid+1)-1;
			int bi = offset*(2*tid+2)-1;
			float t = sdata2[ai];
			sdata2[ai] = sdata2[bi];
			sdata2[bi] += t;
		}
	__syncthreads();
	d_sum[sid + 2*tid] = sdata2[2*tid]; // write results to device memory
	d_sum[sid + 2*tid + 1] = sdata2[2*tid + 1];
	}
	//__syncthreads();
	//extra block??
	if ((2*tid + 1 == blockSize - 1) && (sid + 2*tid + 1) < numElems ){
		//printf("%s %u %s %u \n", "global index:", sid + 2 * tid + 1, ",   block index:", blockIdx.x);
		//d_auxSum[blockIdx.x] = d_sum[sid + 2*tid + 1];
		d_auxSum[blockIdx.x] = sdata2[2*tid + 1];
		//printf("%u \n", d_sum[sid + 2*tid]);
	}
	//not sure if d_sum is right
	//printf("%u \n", d_sum[sid+2*tid]);
}

__global__
void blelloch_scan2(unsigned int* d_auxSum, int numBlocks) {
	extern __shared__ unsigned int sdata3[];// allocated on invocation
	int tid = threadIdx.x;
	//printf("%i \n", tid);
	//printf("%i \n", d_auxSum[tid]);

	int offset = 1;

	sdata3[2*tid] = d_auxSum[2*tid]; // load input into shared memory
	sdata3[2*tid+1] = d_auxSum[2*tid+1];

	// build sum in place up the tree
	for (int d = numBlocks>>1; d > 0; d >>= 1) {
		__syncthreads();
		if (tid < d) {
			int ai = offset*(2*tid+1)-1;
			int bi = offset*(2*tid+2)-1;
			sdata3[bi] += sdata3[ai];
		}
		if (tid==0){printf("%s %d %s %d  \n", "offset=", offset, ",   d=", d);};
		offset *= 2;

	}
	if (tid == 0) { sdata3[numBlocks - 1] = 0; } // clear the last element
	// traverse down tree & build scan
	for (int d = 1; d < numBlocks; d *= 2) {
		offset >>= 1;
		if (tid==0){printf("%s %d %s %d  \n", "offset=", offset, ",   d=", d);};

		__syncthreads();
		if (tid < d) {
			int ai = offset*(2*tid+1)-1;
			int bi = offset*(2*tid+2)-1;
			//printf("%s %d %s %d \n", "ai=",ai,",   bi=",bi);
			float t = sdata3[ai];
			sdata3[ai] = sdata3[bi];
			sdata3[bi] += t;
		}
	__syncthreads();
	d_auxSum[2*tid] = sdata3[2*tid]; // write results to device memory
	d_auxSum[2*tid+1] = sdata3[2*tid+1];
	}
	//printf("%u \n", d_auxSum[2*tid]);
	//printf("%u \n", d_auxSum[2*tid+1]);

}

unsigned int nextPow2(unsigned int x){
	unsigned int result=1;
	while (result < x) {result <<= 1;}
	return result;
}

void your_sort(unsigned int* const d_inputVals,
               unsigned int* const d_inputPos,
               unsigned int* const d_outputVals,
               unsigned int* const d_outputPos,
               const size_t numElems)
{
	const int numBits = 1;
	const int numBins = 1 << numBits;
	const size_t blockSize = 1024; //Make sure block size multiple of 2. Prefix sum scan only works for that. Otherwise you have to pad zeros, which gets annoying.
	const size_t numBlocks = (numElems/blockSize) + ((numElems%blockSize) ? 1 : 0);
	unsigned int numBlocksPad = nextPow2(numBlocks);
	printf("%u \n", numBlocksPad);


	unsigned char* d_predicate;
	unsigned int* d_sum;
	unsigned int* d_auxSum;
	unsigned int* h_sum;
	unsigned int* h_auxSum;
	h_sum = (unsigned int*) malloc(numElems * sizeof(unsigned int));
	h_auxSum = (unsigned int*) malloc(numBlocksPad * sizeof(unsigned int));
	checkCudaErrors(cudaMalloc(&d_predicate, numElems * sizeof(unsigned char)));
	//checkCudaErrors(cudaMalloc(&d_histogram, numBins * sizeof(unsigned int)));
	checkCudaErrors(cudaMalloc(&d_sum, numElems * sizeof(unsigned int)));
	checkCudaErrors(cudaMalloc(&d_auxSum, numBlocksPad * sizeof(unsigned int)));



	for (unsigned int bitPos = 0; bitPos < 1 ; bitPos++){
		checkCudaErrors(cudaMemset(d_predicate, 0, numElems * sizeof(unsigned char)));
		//checkCudaErrors(cudaMemset(d_histogram, 0, numBins * sizeof(unsigned int)));
		checkCudaErrors(cudaMemset(d_sum, 0, numElems * sizeof(unsigned int)));
		checkCudaErrors(cudaMemset(d_auxSum, 0, numBlocksPad * sizeof(unsigned int)));

		unsigned int mask = (numBins - 1) << bitPos;
		predicate<<<numBlocks, blockSize, blockSize * sizeof(unsigned int)>>>(d_inputVals, d_predicate, numElems, mask, bitPos);
		cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
		//gpuPrint<<<1,1>>>(d_histogram);
		//blelloch_scan<<<1, numBins/2, sizeof(unsigned int) * numBins>>>(d_cdf, d_histogram, numBins);
		//cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
		blelloch_scan1<<<numBlocks, blockSize/2, blockSize * sizeof(unsigned int)>>>(d_sum, d_auxSum, d_predicate, blockSize, numElems);
		cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());


		blelloch_scan2<<<1, numBlocksPad/2, numBlocksPad * sizeof(unsigned int)>>>(d_auxSum, numBlocksPad);
		cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

		checkCudaErrors(cudaMemcpy(h_auxSum, d_auxSum, numBlocksPad * sizeof(unsigned int), cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(h_sum, d_sum, numElems * sizeof(unsigned int), cudaMemcpyDeviceToHost));


		unsigned int s = 0;
		for (int i = 0 ; i < numBlocks; i++){
			 s+=h_sum[(i) * (blockSize) - 1];
			 printf("%u \n", s);
			 printf("%u \n", h_auxSum[i]);
		}










		//gpuPrint<<<numBlocks, blockSize>>>(d_sum, d_predicate, numElems);
	}
}



int main(int argc, char **argv) {
  unsigned int *inputVals;
  unsigned int *inputPos;
  unsigned int *outputVals;
  unsigned int *outputPos;

  size_t numElems;

  std::string input_file;
  std::string output_file;
  if (argc == 3) {
    input_file  = std::string(argv[1]);
    output_file = std::string(argv[2]);
  }
  else {
    std::cerr << "Usage: ./hw input_file output_file" << std::endl;
    exit(1);
  }
  //load the image and give us our input and output pointers
  preProcess(&inputVals, &inputPos, &outputVals, &outputPos, numElems, input_file);

  GpuTimer timer;
  timer.Start();

  //call the students' code
  your_sort(inputVals, inputPos, outputVals, outputPos, numElems);

  timer.Stop();
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
  int err = printf("%f msecs.\n", timer.Elapsed());

  if (err < 0) {
    //Couldn't print! Probably the student closed stdout - bad news
    std::cerr << "Couldn't print timing information! STDOUT Closed!" << std::endl;
    exit(1);
  }

  //check results and output the tone-mapped image
  postProcess(outputVals, outputPos, numElems, output_file);

  checkCudaErrors(cudaFree(inputVals));
  checkCudaErrors(cudaFree(inputPos));
  checkCudaErrors(cudaFree(outputVals));
  checkCudaErrors(cudaFree(outputPos));
  return 0;
}
