#pragma once
#include "defines.cuh"
#include <cusparse.h>
#include <cublas.h>

//! Structure for storing sparse matrix connectivity in a compressed format. 
//! indices contains all the entries per row.
//! The ith element of offset_in_indices is the number of non-zero entries in the matrix up until row i.
//! Note that offset_in_indices has size ROWS + 1, where the last entry is the total number of non-zero elements (i.e., total number of connections).
//! WEIGHTS are used for learning stuff. The vector of weights is empty unless specifically constructed.
struct AdjacencyList
{
	thrust::device_vector<int> indices;
	thrust::device_vector<int> offset_in_indices;
	thrust::device_vector<Float> weights;
};

//! This is a struct that contains all the relevant stuff for coupling. 
//! It can be called which will perform coupling and store it. 
//! Description:
//! - list Adjacency list containing connectivity.
//! - neuron_outputs Output of neurons.
//! - downstream_inputs Input to downstream neurons after coupling.
//! - is_firing Vector of integers such that element i is 1 if the neuron i is firing and 0 otherwise.
//! - firing_neurons Vector of indices of firing neurons. The vector is of length N but only the first <NUMBER FIRING NEURONS> elements are valid.
class AtomicCoupling
{
private:
	Float* downstream_inputs;
	Float* neuron_outputs;

public:
	AdjacencyList list;
	thrust::device_vector<int> is_firing;
	thrust::device_vector<int> firing_neurons;

	//! Constructor. Note that list_ will be moved to internal AtomicCoupling member variable list, so list_ SHOULD NOT BE USED AFTER SENDING IT TO THE CTOR!!!
	AtomicCoupling(int const N, AdjacencyList& list_, Float* downstream_inputs_, Float* neuron_outputs_);
    
    //! Performs coupling and stores results in downstream_inputs. 
    //! \return Number of neurons firing.
    int PerformCoupling();
};

#if 0 // Old code for doing coupling with cusparse matrix. This is not being used ATM because we found that the cusparse matrix was always slower. Left here for possible future use.
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
	void operator()(CuSparseAdjacencyMatrix const& connectivity, Float* spikes, Float const* neuron_outputs);

	//! (DENSE VERISON) Perform sparse coupling given connectivity provided as an AdjacenyMatrix using cuSparse.
	/*!
	  \param [in] connectivity Network connectivity 
	  \param [in] voltages Voltages of neurons at current timestep
	  \param [out] spikes Output spikes for the neurons. For HH neurons, this is neurotramsmitter concentration for downstream neurons. 
	  \param [in] neuron_ouputs Neuron outputs, for HH neurons, this is the concentration of neurotransmitter outputted from the neuron.
	  \return Number of neurons firing 
	*/
	void operator()(CuBlasAdjacencyMatrix const& connectivity, Float* spikes, Float const* neuron_outputs);
};
#endif
