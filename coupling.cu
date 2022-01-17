#include "coupling.cuh"

AtomicCoupling::AtomicCoupling(int const N, AdjacencyList& list_, Float* downstream_inputs_, Float* neuron_outputs_)
    : list{}, downstream_inputs{ downstream_inputs_ }, neuron_outputs{ neuron_outputs_ }, is_firing(N, 0), firing_neurons(N, -1)
{
	// Move contents of list_ to list. This makes list_ now empty!
	std::swap(list.indices, list_.indices);
	std::swap(list.offset_in_indices, list_.offset_in_indices);
	std::swap(list.weights, list_.weights);
}

inline __global__ void signal_spikes_kernel(int const num_firing,
	int const* connectivity,
	int const* connectivity_offsets,
	int const* firing_inds,
	Float* downstream_inputs,
	Float const* neuron_outputs)
{
	for (int tid = blockIdx.x * blockDim.x + threadIdx.x; tid < num_firing; tid += gridDim.x * blockDim.x) {
		// Get index of upstream firing neuron.
		int const pre_idx = firing_inds[tid];

		// Output of firing neuron.
		Float const signal = neuron_outputs[pre_idx];

		// Pointer to first downstream neuron. 
		int const* post = connectivity + connectivity_offsets[pre_idx];

		// Iterate over all downstream neurons to this neuron and incrememnt their neurotransmitter input by signal.
        for (; post != connectivity + connectivity_offsets[pre_idx + 1]; ++post)
			atomicAdd(downstream_inputs + *post, signal);
	}
}

struct CheckFiringOperator
{
	__host__ __device__ int operator()(Float const output) const
	{
		return output > 0.1 ? 1 : 0;
	}
};

int AtomicCoupling::PerformCoupling()
{
	int const N = is_firing.size();
    thrust::fill_n(thrust::device_ptr<Float>(downstream_inputs), N, 0);

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
		num_firing, thrust::raw_pointer_cast(list.indices.data()), thrust::raw_pointer_cast(list.offset_in_indices.data()),
		thrust::raw_pointer_cast(firing_neurons.data()), downstream_inputs, neuron_outputs);
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaDeviceSynchronize() );

	return num_firing;
}


#if 0
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

void MatrixCoupling::operator()(CuSparseAdjacencyMatrix const& connectivity, Float* spikes, Float const* neuron_outputs)
{
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
#endif
