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
	  \param [in] use_weights If true, weights are allocated for adjacency list and set to 1.
      \return The adjacency list.
	*/
	AdjacencyList ToAdjacencyList(Float*& Tcount, bool use_weights=false) const;

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

//! Localized coupling where a neuron is connected to a fixed number of neighbors within a certain radius with probability 
//! given by normal distribution. NOT the same as KNNNetwork! 
class NormalNetwork : public Network
{
protected:
	Morphology* morphology_handle;
	Float stddev;
    int n_stddevs; 

public:
	/*!
	  \param [in] morphology_handle_ A pointer to the morphology which handles finding neighbors.
	  \param [in] stddev_ Standard deviation in world coordinates for normal blobs.
	  \param [in] max_n_conns_ Number of neurons to couple to per neuron.
	  \param [in] n_stddevs_ Number of standard deviations to use for connections. Since 95% is within 2 stddevs, this defaults to 2.
	*/
	NormalNetwork(Morphology* morphology_handle_, Float stddev_, int n_stddevs_=2) 
		: morphology_handle{ morphology_handle_ }, stddev{ stddev_ }, n_stddevs{ n_stddevs_ } {}
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
