#pragma once
#include "defines.cuh"

struct AdjacencyList
{
	thrust::device_vector<int> indices;
	thrust::device_vector<int> offset_in_indices;
};

struct ChunkedAdjacencyList
{
	thrust::device_vector<int*> indices_chunks;
	thrust::device_vector<int*> offset_in_indices_chunks;
};

//!  Class that performs coupling using adjacency list telling us the downstream neurons per neuron. 
class AtomicCoupling
{
private:
	int N;
	thrust::device_vector<int> is_firing;
	thrust::device_vector<int> firing_neurons;
	bool fired_previously = false;
    thrust::device_vector<Float> downstream_copies;
public:
	AtomicCoupling(int const N_);

	//! Perform atomic coupling given connectivity provided as an AdjacenyList.
	/*!
	  \param [in] connectivity Network connectivity 
	  \param [out] spikes Output spikes for the neurons. For HH neurons, this is neurotramsmitter concentration for downstream neurons. 
	  \param [in] neuron_ouputs Neuron outputs, for HH neurons, this is the concentration of neurotransmitter outputted from the neuron.
	  \return Number of neurons firing 
	*/
	int operator()(AdjacencyList const& connectivity, Float* spikes, Float const* neuron_outputs);
	int operator()(ChunkedAdjacencyList const& connectivity, Float* spikes, Float const* neuron_outputs);
	thrust::device_vector<int> const& GetFiringNeurons() const { return firing_neurons; }
};

//! Stored in CSC format!
struct CuSparseAdjacencyMatrix
{
	cusparseHandle_t& context;
	AdjacencyList& list;
	thrust::device_vector<Float> vals;
	static cusparseMatDescr_t default_descr;

	//! gemv performs operator y = alpha * A * x + beta * y, where A is this matrix. 
	void gemv(Float const* x, Float* y, Float alpha = 1.0, Float beta = 0.0, bool transpose = false) const;
};

struct CuBlasAdjacencyMatrix
{
	cublasHandle_t& context;
	thrust::device_vector<Float> matrix;

	//! gemv performs operator y = alpha * A * x + beta * y, where A is this matrix. 
	void gemv(Float const* x, Float* y, Float alpha = 1.0, Float beta = 0.0, bool transpose = false) const;
};

//!  Class that performs coupling using a matrix that tells us the upstream neurons per neuron. Matrix can be sparse or dense. 
class MatrixCoupling
{
private:
	int N;
	thrust::device_vector<int> is_firing;
	bool fired_previously = false;

public:
	MatrixCoupling(int const N_);

	//! (SPARSE VERSION) Perform sparse coupling given connectivity provided as an AdjacenyMatrix using cuSparse.
	/*!
	  \param [in] connectivity Network connectivity 
	  \param [out] spikes Output spikes for the neurons. For HH neurons, this is neurotramsmitter concentration for downstream neurons. 
	  \param [in] neuron_ouputs Neuron outputs, for HH neurons, this is the concentration of neurotransmitter outputted from the neuron.
	  \return Number of neurons firing 
	*/
	int operator()(CuSparseAdjacencyMatrix const& connectivity, Float* spikes, Float const* neuron_outputs);

	//! (DENSE VERISON) Perform sparse coupling given connectivity provided as an AdjacenyMatrix using cuSparse.
	/*!
	  \param [in] connectivity Network connectivity 
	  \param [in] voltages Voltages of neurons at current timestep
	  \param [out] spikes Output spikes for the neurons. For HH neurons, this is neurotramsmitter concentration for downstream neurons. 
	  \param [in] neuron_ouputs Neuron outputs, for HH neurons, this is the concentration of neurotransmitter outputted from the neuron.
	  \return Number of neurons firing 
	*/
	int operator()(CuBlasAdjacencyMatrix const& connectivity, Float* spikes, Float const* neuron_outputs);
};
