#define _USE_MATH_DEFINES
#include "network_factory.cuh"
#include <unordered_set>

AdjacencyList Network::ToAdjacencyList(Float*& Tcount, bool use_weights) const
{
    // Generate coupling
    AdjacencyList list;
	list.indices = indices;
	list.offset_in_indices = offset_in_indices;

	if (use_weights)
		list.weights = thrust::device_vector<Float>(list.indices.size(), (Float)1);

    // Calculate numer of upstream neurons for each neuron based on synaptic coupling
	int N = offset_in_indices.size() - 1;
	if (N == -1)
	{
		std::cerr << "Connectivity is empty. Returning blank list." << std::endl;
		return {};
	}

	double avg_num_conns = 0;
	for (int i = 1; i < offset_in_indices.size(); ++i)
		avg_num_conns += offset_in_indices[i] - offset_in_indices[i - 1];
	std::cout << "Average number of connections: " << avg_num_conns / (offset_in_indices.size() - 1) << std::endl;

    thrust::host_vector<Float> Tcount_cpu(N, 0);
	for (int ind : indices)
		++Tcount_cpu[ind];
	gpuErrchk(cudaMemcpy(Tcount, Tcount_cpu.data(), N * sizeof(Float), cudaMemcpyHostToDevice));

    std::cout << "Converted connectivity to adjacency list" << std::endl;
    return list;
}

bool UniformRandomNetwork::Generate(int const N)
{
    std::cout << "Generating uniform random coupling..." << std::endl;
    auto& gen{ Utility::UnseededRandomEngine() };
	std::uniform_int_distribution<> int_dist(0, N-1);

	indices.resize(N * Ncouple);
	offset_in_indices.resize(N + 1, 0);
	thrust::host_vector<int> neuron_downstream(Ncouple);
	std::unordered_set<int> set(Ncouple);
	for (int i = 0; i < N; ++i)
	{
        if (i % 1000 == 0)
            std::cout << i / (float)N * 100 << "%\r" << std::flush;
		thrust::fill_n(neuron_downstream.begin(), Ncouple, INT_MAX);
		for (int j = 0; j < Ncouple; ++j)
		{
			// This selects random numbers without making duplicates.
			// We use a set for constant time insert and lookup and clear it each time. 
			int downstream;
			do
			{
				downstream = int_dist(gen);
			} while (set.find(downstream) != set.end());
			neuron_downstream[j] = downstream;
			set.insert(downstream);
		}
		thrust::copy_n(neuron_downstream.begin(), Ncouple, indices.begin() + i * Ncouple);
		offset_in_indices[i] = i * Ncouple;
		set.clear();
	}
    std::cout << "100%\r" << std::endl;
	offset_in_indices[N] = N * Ncouple;
	return true;
}

bool KNNNetwork::Generate(int const N)
{
    std::cout << "Generating KNN coupling..." << std::endl;
    std::vector<int> my_knn(k+1); 
    std::vector<ANNdist> my_knn_dists(k+1);

    indices.resize(N * k);
    offset_in_indices.resize(N + 1, 0);
	for (int n = 0; n < N; ++n)
    {
        if (n % 1000 == 0)
            std::cout << n / (float)N * 100 << "%\r" << std::flush;

		// Note the +1 because we don't want to self-couple this neuron.
		morphology_handle->KNN(n, k+1, my_knn, my_knn_dists);
		std::copy(my_knn.begin() + 1, my_knn.end(), indices.begin() + k * n);
        offset_in_indices[n + 1] = offset_in_indices[n] + k;
    }
    std::cout << "100%\r" << std::endl;
	return true;
}

bool NormalNetwork::Generate(int const N)
{
    std::cout << "Generating connections... " << std::endl;
    std::vector<int> ball(N); 
    std::vector<ANNdist> ball_dists(N);
    auto& gen{ Utility::UnseededRandomEngine() };

    indices.reserve(N * 100);
    offset_in_indices.resize(N + 1, 0);
    ANNdist const variance = stddev * stddev;
	ANNdist const normal_dist_scalar = 1.0 / (stddev * sqrt(2.0 * 3.14159265359));
    std::uniform_real_distribution<ANNdist> unit_dist(0.0, normal_dist_scalar);
    int ball_sz_sum{ 0 };
    for (int from_idx = 0; from_idx < N; ++from_idx)
    {
        if (from_idx % 1000 == 0)
        {
            std::cout << from_idx / (float)N * 100 << "%\r";
            std::cout.flush();
        }	

		int ball_sz{ morphology_handle->WithinBall(from_idx, n_stddevs * stddev, N, ball, ball_dists) };
        ball_sz_sum += ball_sz;

		int n_inds_prev = indices.size();
        for (int i = 1; i < ball_sz; ++i)
        {
            // Calculate normal distribution weight
            ANNdist prob{ normal_dist_scalar * std::exp(-ball_dists[i] / (2 * variance)) };
            ANNdist unit_sample = unit_dist(gen);

			// We have prob chance of this condition being true, so the distribution of points will be normal.
            if (unit_sample < prob)
                indices.push_back(ball[i]);
        }
        offset_in_indices[from_idx + 1] = indices.size();
    }
    std::cout << "100%\r" << std::endl;

    indices.shrink_to_fit(); // Shrink capacity
    std::cout << "Average number of connections: " << indices.size() / (float)N << std::endl;
    std::cout << "Average ball size: " << ball_sz_sum / (float)N << std::endl;
	return true;
}

FileNetwork::FileNetwork(std::string const& fl_name_)
    : fl_name{ "../" + fl_name_ }, format{ fl_name_.substr(fl_name_.find_last_of('.') + 1) }
{
}

bool FileNetwork::Generate(int const N)
{
    std::cout << "Generating coupling from file '" + fl_name + "'..." << std::endl;
    if (format != "h5" && format != "H5")
    {
        std::cerr << "Unsupported format " << format << std::endl;
        return false;
    }
    return GenerateFromHDF(N);
}

bool FileNetwork::GenerateFromHDF(int const N)
{
	using namespace H5;
	int numconns, numpnts;
	std::vector<int> col_idx, row_ptr;
	try
	{
		H5File file(fl_name, H5F_ACC_RDONLY);
		DataSet col_idx_arr = file.openDataSet("/col_idx");
		numconns = (int)(col_idx_arr.getStorageSize() / sizeof(int));
		col_idx.resize(numconns);
		col_idx_arr.read(col_idx.data(), PredType::NATIVE_INT);

		DataSet row_ptr_arr = file.openDataSet("/row_ptr");
		numpnts = (int)(row_ptr_arr.getStorageSize() / sizeof(int));
		row_ptr.resize(numpnts);
		row_ptr_arr.read(row_ptr.data(), PredType::NATIVE_INT);
	}
	catch (...)
	{
		std::cerr << "ERROR : Failed to read in coupling" << std::endl;
		return false;
	}
	numpnts -= 1; // row_ptr has one extra element.
	if (N > numpnts)
		std::cerr << "Not enough points in file " + fl_name  
		           + ". Have N = " + std::to_string(N) + " and " \
		           + std::to_string(numpnts) + " points available." << std::endl;

	int sparsity{ numpnts / N };
	indices.reserve(numconns);
	offset_in_indices.resize(N + 1, 0);
	for (int from_idx = 0; from_idx < N; ++from_idx)
	{
		int const lookup_idx{ from_idx * sparsity };
		offset_in_indices[from_idx + 1] = offset_in_indices[from_idx];
		for (int to_idx = row_ptr[lookup_idx]; to_idx < row_ptr[lookup_idx + 1]; ++to_idx)
			if (col_idx[to_idx] % sparsity == 0)
			{
				indices.push_back(col_idx[to_idx] / sparsity);
				offset_in_indices[from_idx + 1] += 1;
			}
	}
	indices.shrink_to_fit();
}