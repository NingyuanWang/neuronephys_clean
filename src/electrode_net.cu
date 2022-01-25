#include "electrode_net.cuh"

inline __global__ void electrodes_measure_kernel (
    int const N_regions, 
    int const* regions,
    int const* region_offsets,
    Float* output,
    Float const* V,
    int* firing_per_region,
    int const* is_firing)
{
    for (int tid = blockIdx.x * blockDim.x + threadIdx.x; tid < N_regions; tid += gridDim.x * blockDim.x) 
    {
        // Compute average voltage for region.
        Float V_sum = 0;
        int const* region_start = regions + region_offsets[tid];
        int const* region_end = regions + region_offsets[tid + 1];
        for(int const* it = region_start; it != region_end; ++it)
            V_sum += V[*it];

        output[tid] = V_sum / (region_end - region_start);

        // Compute number of neurons firing in region.
        int n_firing = 0;
        for (int const* it = region_start; it != region_end; ++it)
            n_firing += is_firing[*it];

        firing_per_region[tid] = n_firing;
    }
}

void ElectrodeNet::DoElectrodeMeasurements (Float const* V_ptr, thrust::device_vector<int> const& is_firing)
{
	electrodes_measure_kernel<<<NBLOCKS, NTHREADS>>>(
        N_regions,
        thrust::raw_pointer_cast(d_regions.indices.data()),
        thrust::raw_pointer_cast(d_regions.offset_in_indices.data()),
        thrust::raw_pointer_cast(d_avg_voltages.data()),
        V_ptr,
        thrust::raw_pointer_cast(d_firing_per_region.data()),
        thrust::raw_pointer_cast(is_firing.data())
    );
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaDeviceSynchronize() );

    // Copy averages from GPU to CPU.
    h_avg_voltages = d_avg_voltages;
    h_firing_per_region = d_firing_per_region;
}

Float ElectrodeNet::MeasureAverageEntireNetwork (Float const* V_ptr, int const N)
{
    return thrust::reduce(thrust::device_ptr<Float const>(V_ptr), thrust::device_ptr<Float const>(V_ptr + N)) / N;
}

inline __global__ void variance_kernel (
    int const N,
    int const* region_inds,
    Float const* V,
    double * Vi2_sum, double* Vi_sum)
{
    for (int tid = blockIdx.x * blockDim.x + threadIdx.x; tid < N; tid += gridDim.x * blockDim.x) 
    {
        int const ind = region_inds[tid];
        Vi2_sum[tid] += V[ind] * V[ind];
        Vi_sum[tid] += V[ind];
    }
}

void SynchronyNet::MeasureVariances(Float const* V_ptr)
{
    // Measure V_i(t)^2 and V_i(t) and add to running sum over time.
	variance_kernel<<<NBLOCKS, NTHREADS>>>(
        d_region_inds.size(),
        thrust::raw_pointer_cast(d_region_inds.data()),
        V_ptr,
        thrust::raw_pointer_cast(d_Vi2_sum.data()),
        thrust::raw_pointer_cast(d_Vi_sum.data()));
	gpuErrchk( cudaDeviceSynchronize() );

    // Sum over sub arrays. For how to do this in thrust, see https://stackoverflow.com/questions/10451664/thrust-summing-the-elements-of-an-array-indexed-by-another-array-matlabs-synt. 
    auto V_thrust_ptr = thrust::device_ptr<Float const>(V_ptr);
    Float Vsmpl[3];
    cudaMemcpy(Vsmpl, V_ptr, sizeof(Float) * 3, cudaMemcpyDeviceToHost);
    for (int i = 0; i < sub_sample_counts.size(); ++i) 
    {
        // Welcome to jank land :).
        int const& Nsub = sub_sample_counts[i];
        Float Vsum = thrust::reduce(thrust::make_permutation_iterator(V_thrust_ptr, d_region_inds.begin()),
                                    thrust::make_permutation_iterator(V_thrust_ptr, d_region_inds.begin() + Nsub));
        double Vavg = Vsum / Nsub;
        Vavg_sums[i] += Vavg;
        Vavg2_sums[i] += Vavg * Vavg;
    }
	gpuErrchk( cudaDeviceSynchronize() );
    N_temporal_steps += 1;
}

std::vector<double> SynchronyNet::GetSynchronyMeasures()
{
    std::vector<double> chi_sub(sub_sample_counts.size(), 0);

    // Copy from GPU to CPU and average sums over time.
    thrust::host_vector<double> h_Vi2_avg = d_Vi2_sum;
    thrust::host_vector<double> h_Vi_avg = d_Vi_sum;

    if(N_temporal_steps > 0)
        for (int i = 0; i < h_Vi2_avg.size(); ++i)
        {
            h_Vi2_avg[i] /= N_temporal_steps;
            h_Vi_avg[i] /= N_temporal_steps;
        }

    for (int i = 0; i < sub_sample_counts.size(); ++i)
    {
        int Nsub = sub_sample_counts[i];
        double denom = 0;
        for (int j = 0; j < Nsub; ++j)
            denom += h_Vi2_avg[j] - h_Vi_avg[j] * h_Vi_avg[j]; // = sigma^2_Vj

        // Compute Nsub * sigma^2_Vsub / denom.
        double sigmaV2_sub = (Vavg2_sums[i] - Vavg_sums[i] * Vavg_sums[i] / N_temporal_steps) / N_temporal_steps;
        chi_sub[i] = Nsub * sigmaV2_sub / denom;
    }
    return chi_sub;
}
