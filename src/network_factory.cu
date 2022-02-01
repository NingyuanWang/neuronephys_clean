#define _USE_MATH_DEFINES
#include "network_factory.cuh"
#include <unordered_set>

AdjacencyList Network::ToAdjacencyList(Float*& Tcount) const
{
    // Generate coupling
    AdjacencyList list;
    list.indices = indices;
    list.offset_in_indices = offset_in_indices;

    // Calculate numer of upstream neurons for each neuron based on synaptic coupling
    int N = offset_in_indices.size() - 1;
    if (N == -1)
    {
        std::cerr << "Connectivity is empty. Returning blank list." << std::endl;
        return {};
    }

    thrust::host_vector<Float> Tcount_cpu(N, 0);
    for (int ind : indices)
        ++Tcount_cpu[ind];
    cudaMemcpy(Tcount, Tcount_cpu.data(), N * sizeof(Float), cudaMemcpyHostToDevice);

    std::cout << "Converted connectivity to adjacency list" << std::endl;
    return list;
}

ChunkedAdjacencyList Network::ToChunkedAdjacencyList(Float*& Tcount) const
{
    // Generate coupling
    ChunkedAdjacencyList list;
    int N = offset_in_indices.size() - 1;
    if (N == -1)
    {
        std::cerr << "Connectivity is empty. Returning blank list." << std::endl;
        return {};
    }

    // Calculate numer of upstream neurons for each neuron based on synaptic coupling
    thrust::host_vector<Float> Tcount_cpu(N, 0);
    for (int ind : indices)
        ++Tcount_cpu[ind];
    cudaMemcpy(Tcount, Tcount_cpu.data(), N * sizeof(Float), cudaMemcpyHostToDevice);

    // Copy indices into chunks.
    thrust::host_vector<thrust::host_vector<int>>
        indices_chunked(NCHUNKS), offset_in_indices_chunked(NCHUNKS, thrust::host_vector<int>(N+1,0));

    for (int i = 1; i < N + 1; ++i)
    {
        // Initialize i'th element of each offsets for each chunk to (i-1)'th element.
        for (auto& inds_chunk : offset_in_indices_chunked)
            inds_chunk[i] = inds_chunk[i - 1];

        // Copy downstream connections from i'th element to respective chunks.
        for (int j = offset_in_indices[i - 1]; j < offset_in_indices[i]; ++j)
        {
            int const ds_idx = indices[j];
            int const ds_chunk = ds_idx / N_PER_CHUNK;
            indices_chunked[ds_chunk].push_back(ds_idx % N_PER_CHUNK);
            offset_in_indices_chunked[ds_chunk][i] += 1;
        }
    }

    // Copy chunked connections to GPU as int double pointer.
    // First, make nested host_vectors into int* pointer on GPU.
    thrust::host_vector<int*> raw_indices_chunked(NCHUNKS), raw_offset_in_indices_chunked(NCHUNKS);
    for (int i = 0; i < NCHUNKS; ++i)
    {
        int sz = indices_chunked[i].size() * sizeof(int);
        cudaMalloc(&(raw_indices_chunked[i]), sz);
        cudaMemcpy(raw_indices_chunked[i], indices_chunked[i].data(), sz, cudaMemcpyHostToDevice);

        sz = offset_in_indices_chunked[i].size() * sizeof(int);
        cudaMalloc(&(raw_offset_in_indices_chunked[i]), sz);
        cudaMemcpy(raw_offset_in_indices_chunked[i], offset_in_indices_chunked[i].data(), sz, cudaMemcpyHostToDevice);
    }

    // Second, copy outer host_vector to device_vector on GPU.
    list.indices_chunks = raw_indices_chunked;
    list.offset_in_indices_chunks = raw_offset_in_indices_chunked;
    std::cout << "Converted connectivity to chunked adjacency list" << std::endl;
    return list;
}

CuSparseAdjacencyMatrix Network::ToUpstreamSparseMatrix(cusparseHandle_t& handle, AdjacencyList& list) const
{
    // Since CuSparseAdjacencyMatrix is CSC format, and a column of the matrix contains downstream neurons, 
    // we don't need to do any conversion stuff, yay! 
    CuSparseAdjacencyMatrix mat{ handle, list };
    mat.vals.resize(list.indices.size(), 1);
    return mat;
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
    indices.resize(numconns);
    offset_in_indices.resize(numpnts + 1, 0);
    auto it = indices.begin();
    for (int from_idx = 0; from_idx < N; ++from_idx)
    {
        int const lookup_idx{ from_idx * sparsity };
        for (int to_idx = row_ptr[lookup_idx]; to_idx < row_ptr[lookup_idx + 1]; ++to_idx)
            if (col_idx[to_idx] % sparsity == 0)
                *it++ = col_idx[to_idx] / sparsity;
        offset_in_indices[from_idx + 1] = it - indices.begin();
    }
}