#pragma once
#include "coupling.cuh"

//! Implements a network of "electrodes" measuring averaged activity in regions. 
//! Regions are stored as an adjacency list.
class ElectrodeNet 
{
private:
	AdjacencyList d_regions;
	int N_regions;
	thrust::device_vector<Float> d_avg_voltages;
	thrust::host_vector<Float> h_avg_voltages;
	thrust::device_vector<int> d_firing_per_region;
	thrust::host_vector<int> h_firing_per_region;
public:
	ElectrodeNet (thrust::host_vector<int> const& regions_inds, 
	              thrust::host_vector<int> const& regions_offsets)
		: d_regions{regions_inds, regions_offsets}, 
		  N_regions{(int)regions_offsets.size() - 1},
		  d_avg_voltages(N_regions, 0),
		  h_avg_voltages(N_regions, 0),
		  d_firing_per_region(N_regions, 0),
		  h_firing_per_region(N_regions, 0)
	{
	}

	//! Computes the averaged activity in each region and then copies to CPU in h_avg_voltages.
	//! Also computes the number of neurons firing per region.
	void DoElectrodeMeasurements (Float const* V_ptr, thrust::device_vector<int> const& is_firing);

	//! Computes the average voltage over the ENTIRE network of neurons.
	static Float MeasureAverageEntireNetwork (Float const* V_ptr, int const N);
	
	Float const* ElectrodeOutputHandle () const
	{
		return h_avg_voltages.data();
	}

	int const* FiringPerRegionHandle() const
	{
		return h_firing_per_region.data();
	}
};


//! Implements a method of tracking variables related to synchrony in a region.
//! See http://www.scholarpedia.org/article/Neuronal_synchrony_measures for details.
class SynchronyNet
{
private:
	thrust::host_vector<int> h_region_inds;
	thrust::device_vector<int> d_region_inds;
	std::vector<int> sub_sample_counts;
	std::vector<double> Vavg2_sums, Vavg_sums; // <V(t)^2>_t, <V(t)>_t, for each sub population to average over.
	thrust::device_vector<double> d_Vi2_sum, d_Vi_sum; // <V_i(t)^2>_t, <V_i(t)>_t, for each i in h_region_inds.
	int N_temporal_steps; // Number of timesteps used for measurement.

public:
	//! Constructor.
	//! \param [in] h_region_inds_ Indices of neurons in region for measurement.
	//! \param [in] sub_sample_counts_ Vector of subsample counts to measure synchrony for. For example, 
	//!				if this is {1, 2, 4}, we get three variance measurements for variance of first 1, 2, and 4
	//!				from the region. As in the linked article, it is useful to have variance for subpopulations.
	SynchronyNet(thrust::host_vector<int>&& h_region_inds_, std::vector<int> sub_sample_rates_)
		: h_region_inds{ std::move(h_region_inds_) }, d_region_inds(h_region_inds),
		sub_sample_counts{ sub_sample_rates_ },
		Vavg2_sums(sub_sample_counts.size(), 0), Vavg_sums(sub_sample_counts.size(), 0),
		d_Vi2_sum(h_region_inds.size(), 0), d_Vi_sum(h_region_inds.size(), 0),
		N_temporal_steps{ 0 }
	{
	}

	//! Measure neuron variances for individual and for sub populations from regions.
	void MeasureVariances(Float const* V_ptr);

	//! Returns chi^2(N_sub) for each N_sub sub_sample_counts.
	std::vector<double> GetSynchronyMeasures();
};
