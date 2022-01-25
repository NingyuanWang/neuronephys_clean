#include "hebbian.cuh"

inline __global__ void set_timers(int const num_firing,
    int const* connectivity,
    int const* connectivity_offsets,
    int const* firing_inds,
    int* timers,
    int window_duration)
{
    for (int tid = blockIdx.x * blockDim.x + threadIdx.x; tid < num_firing; tid += gridDim.x * blockDim.x) {
        int const pre_idx = firing_inds[tid];
        for (int syn_id = connectivity_offsets[pre_idx]; syn_id < connectivity_offsets[pre_idx + 1]; ++syn_id)
            timers[syn_id] = window_duration;
    }
}

inline __global__ void hebbian_rule_kernel(int const Nconns,
    int const* connectivity,
    int const* is_firing,
    int* timers,
    Float* weights,
    Float heb_inc)
{
    for (int syn_id = blockIdx.x * blockDim.x + threadIdx.x; syn_id < Nconns; syn_id += gridDim.x * blockDim.x)
    {
        // If timer is 0, there has been no upstream activity, so Hebbian update does not apply.
        if (timers[syn_id] == 0)
            continue;

        // Downstream index.
        int post_idx = connectivity[syn_id];

        // Decrememnt timers.
        timers[syn_id] -= 1;

        // HEBBIAN RULES :)!!
        // If timer has hit zero, downregulate weight.
        Float alpha = expf(-fabsf(weights[syn_id] - 1));
        if (timers[syn_id] == 0)
        {
            // Downregulate weight.
            weights[syn_id] -= heb_inc * alpha;
        }
        else if (is_firing[post_idx])
        {
            // Upregulate weight and reset timer.
            weights[syn_id] += heb_inc * alpha;
            timers[syn_id] = 0;
        }
    }
}

void Hebbian::UpdateWeights(AdjacencyList& connectivity, thrust::device_vector<int> const& is_firing, thrust::device_vector<int> const& firing_inds)
{
    set_timers << <NBLOCKS, NTHREADS >> > (firing_inds.size(),
        thrust::raw_pointer_cast(connectivity.indices.data()),
        thrust::raw_pointer_cast(connectivity.offset_in_indices.data()),
        thrust::raw_pointer_cast(firing_inds.data()),
        thrust::raw_pointer_cast(timers.data()),
        window_duration);

    hebbian_rule_kernel << <NBLOCKS, NTHREADS >> > (connectivity.indices.size(),
        thrust::raw_pointer_cast(connectivity.indices.data()),
        thrust::raw_pointer_cast(is_firing.data()),
        thrust::raw_pointer_cast(timers.data()),
        thrust::raw_pointer_cast(connectivity.weights.data()),
        heb_inc);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
}