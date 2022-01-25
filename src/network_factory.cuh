#pragma once
#include "coupling.cuh"
#include "morphology.h"

//! Network is an abstract class that supports creation of connectivity on a set of neurons and operations as union.
class Network
{
protected:
	//! Connections indices for adjacency list format.
	thrust::host_vector<int> indices;

	//! Connections row offsets for adjacency list format.
	thrust::host_vector<int> offset_in_indices;

public:
	Network() = default;

	//! Generates a network. Pure virtual function.
	/*!
	  \param [in] N Number of neurons.
      \return If the function is successful for not.
	*/
    virtual bool Generate (int const N) = 0;


	//! Generates an AdjacencyList on the GPU and returns it.
	/*!
	  \param [out] Tcount Number of upstream connections per neuron. 
      \return The adjacency list.
	*/
	AdjacencyList ToAdjacencyList(Float*& Tcount) const;

	//! Generates a ChunkedAdjacencyList on the GPU and returns it.
	/*!
	  \param [out] Tcount Number of upstream connections per neuron. 
      \return The chunked adjacency list.
	*/
	ChunkedAdjacencyList ToChunkedAdjacencyList(Float*& Tcount) const;

	//! Generates a CuSparseAdjacencyMatrix on the GPU which gives upstream neuron connections (transpose of downstream).
	/*!
	  \param [in, out] handle The cusparse handle to use. 
	  \param [in, out] list Connectivity stored in AdjacencyList format. 
      \return The sparse matrix coupling.
	*/
	CuSparseAdjacencyMatrix ToUpstreamSparseMatrix(cusparseHandle_t& handle, AdjacencyList& list) const;

	//! Getter for indices on cpu.
	thrust::host_vector<int> const& GetCpuIndices() const { return indices; }

	//! Getter for offsets_in_indices on CPU.
	thrust::host_vector<int> const& GetCpuOffsetsIntoIndices() const { return offset_in_indices; }
};

//! Network where each neuron has a fixed number of downstream connections chosen uniformly randomly.
class UniformRandomNetwork : public Network
{
protected:
	int Ncouple;

public:
	/*!
	  \param [in] Ncouple_ Number of connections per neuron..
	*/
	UniformRandomNetwork(int Ncouple_) : Ncouple(Ncouple_) {}
	virtual bool Generate(int const N) override;
};

//! Localized coupling where a neuron is connected to the k-nearest neighbors.
class KNNNetwork : public Network
{
protected:
	int k;
	Morphology* morphology_handle;

public:
	/*!
	  \param [in] k_ Number of nearest neighbors to couple to per neuron.
	  \param [in] morphology_handle_ A pointer to the morphology which handles finding neighbors.
	*/
	KNNNetwork(int k_, Morphology* morphology_handle_) : k(k_), morphology_handle{ morphology_handle_ } {}
	virtual bool Generate(int const N) override;
};

//! Network read in from a file with the format inferred in constructor.
class FileNetwork : public Network
{
protected:	
	std::string fl_name;	
	std::string format;

public:
	/*!
	  \param [in] fl_name_ Name of file to read from.
	*/
	FileNetwork(std::string const& fl_name_);
	bool GenerateFromHDF(int const N);
	virtual bool Generate(int const N) override;
};
