#include "coupling.cuh"

cusparseMatDescr_t CuSparseAdjacencyMatrix::default_descr;

AtomicCoupling::AtomicCoupling(int const N_) 
	: N(N_), is_firing(N_, 0), firing_neurons(N_, 0), downstream_copies(NBLOCKS * N_, 0)
{
}

#if USE_DOUBLE_PRECISION
inline __device__ double atomicAdd_double(double* address, double val)
{
	unsigned long long int* address_as_ull =
		(unsigned long long int*)address;
	unsigned long long int old = *address_as_ull, assumed;

	do {
		assumed = old;
		old = atomicCAS(address_as_ull, assumed,
			__double_as_longlong(val +
				__longlong_as_double(assumed)));

		// Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
	} while (assumed != old);

	return __longlong_as_double(old);
}
#define ATOMIC_ADD_SPECIALIZED atomicAdd_double
#else
#define ATOMIC_ADD_SPECIALIZED atomicAdd
#endif // USE_DOUBLE_PRECISION


inline __global__ void signal_spikes_kernel(int const num_firing,
	int const* connectivity,
	int const* connectivity_offsets,
	int const* firing_inds,
	Float* spikes,
	Float const* neuron_outputs)
{
	for (int tid = blockIdx.x * blockDim.x + threadIdx.x; tid < num_firing; tid += gridDim.x * blockDim.x) {
		int const pre_idx = firing_inds[tid];
		int const* post = connectivity + connectivity_offsets[pre_idx];
		Float const signal = neuron_outputs[pre_idx];
        for (; post != connectivity + connectivity_offsets[pre_idx + 1]; ++post)
			atomicAdd(spikes + *post, 1);
	}
}

inline __global__ void signal_spikes_kernel_chunked(
    int const num_firing,  
    int* const* conn_chunks,
    int* const* conn_off_chunks,
    int const* firing_inds,
    Float* downstream_copies,
    Float* downstream,
    Float const* neuron_outputs)
{
    __shared__ unsigned int shared_downstream[N_PER_CHUNK];
    for (int c = 0; c < NCHUNKS; ++c)
    {
		// Zero out shared memory
		for (int tid = threadIdx.x; tid < N_PER_CHUNK; tid += blockDim.x)
			shared_downstream[tid] = 0;

        // Get the relevant connectivity and downstream for our chunk.
        int const* conn = conn_chunks[c];
        int const* conn_off = conn_off_chunks[c];
        Float* downstream = downstream_copies + (N_PER_CHUNK * (blockIdx.x + c * gridDim.x));

        for (int tid = blockIdx.x * blockDim.x + threadIdx.x; tid < num_firing; tid += gridDim.x * blockDim.x) 
        {
            int const pre_idx = firing_inds[tid];
            int const* post = conn + conn_off[pre_idx];
            Float const signal = neuron_outputs[pre_idx];
            for (; post != conn + conn_off[pre_idx + 1]; ++post)
            {
                // Write to shared memory chunk.
                atomicInc(shared_downstream + *post, UINT_MAX);
            }
        }
        // Synchronize threads in block so that we can copy shared memory to global memory.
        __syncthreads();
        for (int tid = threadIdx.x; tid < N_PER_CHUNK; tid += blockDim.x)
            downstream[tid] = shared_downstream[tid];
    }
}


inline __global__ void merge_kernel(
    Float* const downstream_copies,
    Float* downstream)
{
    for (int c = 0; c < NCHUNKS; ++c)
    {
        Float* downstream_chunk = downstream + N_PER_CHUNK * c;
        for (int tid = blockIdx.x * blockDim.x + threadIdx.x; tid < N_PER_CHUNK; tid += gridDim.x * blockDim.x)
            for (int b = 0; b < NBLOCKS; ++b)
                downstream_chunk[tid] += downstream_copies[N_PER_CHUNK * (b + c * gridDim.x) + tid];
    }
}

struct CheckFiringOperator
{
	__host__ __device__ int operator()(Float const output) const
	{
		return output > 0.1 ? 1 : 0;
	}
};

int AtomicCoupling::operator()(AdjacencyList const& connectivity, Float* spikes, Float const* neuron_outputs)
{
	// Minor optimization: only need to zero the spike_counts if we fired previously. 
    if (fired_previously)
		thrust::fill_n(thrust::device_ptr<Float>(spikes), N, 0);

	// Determine which neurons are firing by checking against threshold and that they weren't previously firing.
	thrust::transform(thrust::device_ptr<Float const>(neuron_outputs), thrust::device_ptr<Float const>(neuron_outputs + N), is_firing.begin(), CheckFiringOperator());

	// See second example here: https://thrust.github.io/doc/classthrust_1_1counting__iterator.html
	auto const end_it = thrust::copy_if(thrust::make_counting_iterator(0),
										thrust::make_counting_iterator(N), 
										is_firing.begin(), 
										firing_neurons.begin(), 
										thrust::identity<int>());

	int const num_firing = (int)(end_it - firing_neurons.begin());
	if (num_firing == 0)
		return 0;

	// Transfer neurotransmitter from neurons to post-synaptic neurons.
	signal_spikes_kernel<<<NBLOCKS, NTHREADS>>>(
		num_firing, thrust::raw_pointer_cast(connectivity.indices.data()), thrust::raw_pointer_cast(connectivity.offset_in_indices.data()),
		thrust::raw_pointer_cast(firing_neurons.data()), spikes, neuron_outputs);
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaDeviceSynchronize() );

	// For next time:
	fired_previously = true;
	return num_firing;
}

int AtomicCoupling::operator()(ChunkedAdjacencyList const& connectivity, Float* spikes, Float const* neuron_outputs)
{
	// Minor optimization: only need to zero the spike_counts if we fired previously. 
    if (fired_previously)
    {
		thrust::fill_n(thrust::device_ptr<Float>(spikes), N, 0);
        thrust::fill_n(downstream_copies.begin(), NBLOCKS * N, 0);
    }

	// Determine which neurons are firing by checking against threshold and that they weren't previously firing.
	thrust::transform(thrust::device_ptr<Float const>(neuron_outputs), thrust::device_ptr<Float const>(neuron_outputs + N), is_firing.begin(), CheckFiringOperator());

	// See second example here: https://thrust.github.io/doc/classthrust_1_1counting__iterator.html
	auto const end_it = thrust::copy_if(thrust::make_counting_iterator(0),
										thrust::make_counting_iterator(N), 
										is_firing.begin(), 
										firing_neurons.begin(), 
										thrust::identity<int>());

	int const num_firing = (int)(end_it - firing_neurons.begin());
	if (num_firing == 0)
		return 0;

	// Transfer neurotransmitter from neurons to post-synaptic neurons.
	signal_spikes_kernel_chunked<<<NBLOCKS, NTHREADS>>>(
		num_firing, thrust::raw_pointer_cast(connectivity.indices_chunks.data()), thrust::raw_pointer_cast(connectivity.offset_in_indices_chunks.data()),
		thrust::raw_pointer_cast(firing_neurons.data()), thrust::raw_pointer_cast(downstream_copies.data()), spikes, neuron_outputs);
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaDeviceSynchronize() );

    merge_kernel << <NBLOCKS, NTHREADS >> > (thrust::raw_pointer_cast(downstream_copies.data()), spikes);

	// For next time:
	fired_previously = true;
	return num_firing;
}

MatrixCoupling::MatrixCoupling(int const N_) 
	: N(N_), is_firing(N_, 0) {}

//#if USE_DOUBLE_PRECISION
//#define CUSPARSE_GEMV cusparseDcsrmv
//#else
//#define CUSPARSE_GEMV cusparseScsrmv
//#endif //USE_DOUBLE_PRECISION

void CuSparseAdjacencyMatrix::gemv(Float const* x, Float* y, Float alpha, Float beta, bool transpose) const
{
	// TODO: UPDATE BECAUSE CUSPARSECSRMV IS NOT DEPRECATED.
//	// Note: Transpose is flipped since the matrix is in CSC format, not CSR
//	cusparseOperation_t trans = transpose ? CUSPARSE_OPERATION_NON_TRANSPOSE : CUSPARSE_OPERATION_TRANSPOSE;
//	int const N = (int)list.offset_in_indices.size() - 1;
//	cusparseCreateMatDescr(&default_descr);
//	CUSPARSE_GEMV(context,
//		trans,
//		N, N,
//		(int)vals.size(),
//		&alpha,
//		default_descr,
//		thrust::raw_pointer_cast(vals.data()),
//		thrust::raw_pointer_cast(list.offset_in_indices.data()),
//		thrust::raw_pointer_cast(list.indices.data()),
//		x,
//		&beta,
//		y);
}

int MatrixCoupling::operator()(CuSparseAdjacencyMatrix const& connectivity, Float* spikes, Float const* neuron_outputs)
{
	// Minor optimization: only need to zero the spike_counts if we fired previously. 
	if (fired_previously)
		thrust::fill_n(thrust::device_ptr<Float>(spikes), N, 0);

	// Determine which neurons are firing by checking against threshold and that they weren't previously firing.
	thrust::transform(thrust::device_ptr<Float const>(neuron_outputs), thrust::device_ptr<Float const>(neuron_outputs + N), is_firing.begin(), CheckFiringOperator());

	// Determine number of firing neurons 
	int num_firing = thrust::count(is_firing.begin(), is_firing.end(), 1);
	if (num_firing == 0)
		return 0;

	// Perform sparse-matrix dense-vectr mutliplication Tout = C Tin, where C is the connectivity matrix.
	connectivity.gemv(neuron_outputs, spikes);
	return num_firing;
}
