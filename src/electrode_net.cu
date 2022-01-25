#include "electrode_net.cuh"

inline __global__ void electrodes_measure_kernel (
    int const N_regions, 
    int const* regions,
    int const* region_offsets,
    Float* output,
    Float const* V)
{
    for (int tid = blockIdx.x * blockDim.x + threadIdx.x; tid < N_regions; tid += gridDim.x * blockDim.x) 
    {
        Float V_sum = 0;

        int const* region_start = regions + region_offsets[tid];
        int const* region_end = regions + region_offsets[tid + 1];
        for(int const* it = region_start; it != region_end; ++it)
            V_sum += V[*it];

        output[tid] = V_sum / (region_end - region_start);
    }
}

void ElectrodeNet::DoElectrodeMeasurements (Float const* V_ptr)
{
	electrodes_measure_kernel<<<NBLOCKS, NTHREADS>>>(
        N_regions,
        thrust::raw_pointer_cast(d_regions.indices.data()),
        thrust::raw_pointer_cast(d_regions.offset_in_indices.data()),
        thrust::raw_pointer_cast(d_avg_voltages.data()),
        V_ptr);
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaDeviceSynchronize() );

    // Copy averages from GPU to CPU.
    h_avg_voltages = d_avg_voltages;
}

inline __global__ void TMP_kernel(Float const* V, int const N, Float* sum)
{
    *sum = 0;
    for(int i = 0; i < N; ++i)
        *sum += V[i]; 
    *sum /= N;
}

Float ElectrodeNet::MeasureAverageEntireNetwork (Float const* V_ptr, int const N)
{
    return thrust::reduce(thrust::device_ptr<Float const>(V_ptr), thrust::device_ptr<Float const>(V_ptr + N)) / N;
}