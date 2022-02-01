#include "network_factory.cuh"
#include "graphics.h"
#include "morphology.h"
#include "ephys.cuh"
#include "circadian.h"
#include "paracrine.cuh"
#include "electrode_net.cuh"
#include <glm/gtx/norm.hpp>
#include <glm/gtc/type_ptr.hpp>

#define OUTPUT_ELECTRODES_FILE

// This is simply a struct that contains all the relevant stuff for coupling. 
// It can be called which will perform coupling and store it. 
// Description:
// - list Adjacency list containing connectivity.
// - coupling Coupling class that can be called to do coupling with a variety of connectivity structs.
// - downstream_inputs Input to downstream neurons after coupling.
// - neuron_outputs Output of neurons.
struct CouplingStruct 
{
    ChunkedAdjacencyList& list;
    AtomicCoupling& coupling;
    Float* downstream_inputs;
    Float* neuron_outputs;
    
    // Performs coupling and stores results in downstream_inputs. 
    // Returns number of neurons firing.
    int operator() () 
    {
        return coupling(list, downstream_inputs, neuron_outputs);
    }
};

void set_scn_egaba(Float* const d_egaba, Morphology* morphology, int const N, bool left, bool right)
{
    static const std::vector<double> egaba_key{ -79.4000, -77.7000, -76.0000, -74.3000, -72.7000, -71.0000, -69.3000, -67.6000, -65.9000, -64.3000, -62.6000, -60.9000, -59.2000, -57.5000, -55.9000, -54.2000, -52.5000, -50.8000, -49.1000, -47.5000, -45.8000, -44.1000, -42.4000, -40.7000, -39.1000, -37.4000, -35.7000, -34.0000, -32.3000 };
    static const std::vector<double> cum_dist{ 0.0040, 0.0091, 0.0111, 0.0152, 0.0233, 0.0273, 0.0385, 0.0607, 0.0860, 0.1093, 0.1387, 0.1771, 0.2156, 0.2713, 0.3411, 0.4322, 0.5202, 0.6093, 0.7085, 0.7874, 0.8330, 0.8877, 0.9281, 0.9565, 0.9717, 0.9787, 0.9909, 0.9949, 1.0000 };

    // We split up the distribution into two parts. Those with y less than some y value have inhibitory egaba and those above are selected from the rest of the distribution. 
    // The left lobe of the SCN has less inhibitory neurons, so we set slice to be lower if we are just using the left lobe than the right. 
    static const double left_slice{ -71.0 }, right_slice{ -59.0 }, avg_slice{ (left_slice + right_slice) * 0.5 };
    double slice{ avg_slice };
    if (!right)
        slice = left_slice;
    else if (!left)
        slice = right_slice;
    std::cout << 0 << std::endl;

    int const idx{ (int)(std::lower_bound(egaba_key.begin(), egaba_key.end(), slice) - egaba_key.begin()) };
    int const num_below = cum_dist[idx] * N;
    std::vector<std::pair<int, float>> sorted_by_y_val(N);
    auto const& positions{ morphology->GetPositions() };
    for (int i = 0; i < N; ++i)
        sorted_by_y_val[i] = { i, positions[i].y };
    std::sort(sorted_by_y_val.begin(), sorted_by_y_val.end(), [](std::pair<int, float> lhs, std::pair<int, float> rhs) {return lhs.second < rhs.second; });
    std::cout << 0 << std::endl;

    // First sample inhibitory. These are all neurons with y < y[num_below-1].
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine gen(seed);
    std::uniform_real_distribution<double> dist1(0.0, cum_dist[idx]);
    std::vector<Float> egaba(N);
    for (int i = 0; i < num_below; ++i)
    {
        double const sample{ dist1(gen) };
        int const box{ (int)(std::lower_bound(cum_dist.begin(), cum_dist.end(), sample) - cum_dist.begin()) };
        egaba[sorted_by_y_val[i].first] = egaba_key[box];
    }
    std::cout << 0 << std::endl;

    std::uniform_real_distribution<double> dist2(cum_dist[idx], 1.0);
    for (int i = num_below; i < N; ++i)
    {
        double const sample{ dist2(gen) };
        int const box{ (int)(std::lower_bound(cum_dist.begin(), cum_dist.end(), sample) - cum_dist.begin()) };
        egaba[sorted_by_y_val[i].first] = egaba_key[box];
    }
    std::cout << 0 << std::endl;
    cudaMemcpy(d_egaba, egaba.data(), sizeof(Float) * N, cudaMemcpyHostToDevice);
}

void _RunAndPrintSimulation(int const nincs, int const print_incs, std::ostream& fout,
    int const N, thrust::device_vector<Float*>& sim_vars,  thrust::host_vector<Float*>& sim_vars_cpu,
    Ephys* ephys,
    std::vector<CouplingStruct>& coupling_structs,
    Float* V_ptr,
    float* graphics_voltages_handle = nullptr, int const render_N = -1,
    ElectrodeNet* electrodes = nullptr,
    ParacrineModule* paracrine = nullptr,
    const float circadian_time = 0.0f,
    const Circadian* circadian = nullptr)
{
    std::cout << " HI2" << std::endl;
    // If graphics handle is non-null, allocate CPU mem to copy voltages to.
    uint8_t* frame_data;
    FILE* replay_out;
    bool pause{ false };
    graphics_voltages_handle = nullptr;
    if (graphics_voltages_handle)
    {
        frame_data = (uint8_t*)malloc(3 * sizeof(uint8_t) * Graphics::get_width() * Graphics::get_height());
        replay_out = fopen("REPLAY.raw", "wb");
    }
    float* slice = nullptr;
    if (paracrine) // If using paracrine signalling, display slice.
        slice = paracrine->get_grid().density().slice(Graphics::get_slice_index()).as(f32).host<float>();

    
    for (int i = 0; i < nincs; ++i)
    {

       

        Float sim_time = i * diekman_params::leapfrog_dt;

        if (fmod(sim_time, 1000.0f) < diekman_params::leapfrog_dt) 
        {
            Float sim_time_hr = sim_time / (1000*3600);

            std::vector<float> gk_pair = circadian->get_params_at_time(circadian_time+sim_time_hr);
            std::cout << circadian_time + sim_time_hr << " " << gk_pair[0] << " " << gk_pair[1] << std::endl;
            thrust::fill_n(thrust::device_ptr<Float>(sim_vars_cpu[DIEKMAN_VAR_GKCA]), N, gk_pair[0]);
            thrust::fill_n(thrust::device_ptr<Float>(sim_vars_cpu[DIEKMAN_VAR_GKLEAK]), N, gk_pair[1]);
        }

        if (i % (nincs / 100) == 0)
            std::cout << 100 * (i / (float)nincs) << "%" << std::endl;

        if (i % print_incs == 0)
        {
            electrodes->DoElectrodeMeasurements(V_ptr);
            Float const* avg_V_handle = electrodes->ElectrodeOutputHandle();
#ifdef OUTPUT_ELECTRODES_FILE
            for (int k = 0; k < render_N; ++k)
                fout << avg_V_handle[k] << ','; //which neurons to sample
#endif
            fout << ElectrodeNet::MeasureAverageEntireNetwork(V_ptr, N);
            fout << std::endl;
            // Update graphics if the graphics handle is non-null
            if (graphics_voltages_handle)
            {
                electrodes->DoElectrodeMeasurements(V_ptr);
                Float const* avg_V_handle = electrodes->ElectrodeOutputHandle();
                for (int k = 0; k < render_N; ++k)
                    graphics_voltages_handle[k] = avg_V_handle[k];

                int slice_index;
                slice_index = Graphics::get_slice_index();
                if (paracrine)
                    paracrine->get_grid().density().slice(slice_index % paracrine->z_gridcount).as(f32).host(slice);
                pause = Graphics::render(graphics_voltages_handle, slice, paracrine->x_gridcount, paracrine->y_gridcount, pause);
            }
        }

        if (pause)
        {
            i -= 1;
            continue;
        }

        // Perform coupling. The number of neurons firing is returned by each of these calls. 
        int n_firing;
        for(auto& coup_struct: coupling_structs)
            n_firing = coup_struct();

        if (paracrine)
        {
            if(i% paracrine->timescale == 0)
                paracrine->update_on_own_timescale();
            else 
                paracrine->update_on_fine_timescale(coupling_structs[0].coupling);
        }


        // Update electrophysiology.
        ephys->SimulateEphys(thrust::raw_pointer_cast(sim_vars.data()));
    }

    if (graphics_voltages_handle)
    {
        free(frame_data);
        fclose(replay_out);
    }
}

bool length_predicate(float const* p1, float const* p2)
{
	return true;
}

// N : number of neurons to simulate
// samp_freq : sampling frequency (Hz)
// samp_len : How long you record the activity of the model neuron (sec)
void sanmodel_inhexc_benchmark(int const N = 1, int const samp_freq = 1000, float const samp_len = 10, int render_N = -1)
{
    // Set up graphics and morphology
    FileMorphology morphology("cortex_connectivity_1m_1k_inh.h5");
    morphology.Generate(N);

    float* graphics_voltage_handle;
    Graphics::setup(graphics_voltage_handle, morphology.GetPositionsRaw(), N, render_N);

    // Set up simulation variables. Stored in a struct of arrays.
    thrust::host_vector<Float*> sim_vars_cpu(NUM_AN_IE_VAR);
    thrust::device_vector<Float*> sim_vars;
    std::ofstream fout("OUT_AN.txt");
    {
        thrust::host_vector<Float> cpu_arr(N, 0);

        for (Float*& arr : sim_vars_cpu)
            cudaMalloc(&arr, N * sizeof(Float));

        // Set voltages randomly
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<> voltage_dist(-80, 20);
        for (auto& val : cpu_arr)
            val = voltage_dist(gen);
        cudaMemcpy(sim_vars_cpu[AN_IE_VAR_V], cpu_arr.data(), N * sizeof(Float), cudaMemcpyHostToDevice);

        // Set Ca 
        std::uniform_real_distribution<> Ca_dist(0.1, 0.5);
        for (auto& val : cpu_arr)
            val = Ca_dist(gen);
        cudaMemcpy(sim_vars_cpu[AN_IE_VAR_CA], cpu_arr.data(), N * sizeof(Float), cudaMemcpyHostToDevice);

        // Set potasium gating variable 
        std::uniform_real_distribution<> nK_dist(0.0, 0.5);
        for (auto& val : cpu_arr)
            val = nK_dist(gen);
        cudaMemcpy(sim_vars_cpu[AN_IE_VAR_NK], cpu_arr.data(), N * sizeof(Float), cudaMemcpyHostToDevice);
        thrust::fill_n(cpu_arr.begin(), N, 0.0);
        // AMPA/NMDA gating variable
        cudaMemcpy(sim_vars_cpu[AN_IE_VAR_YE], cpu_arr.data(), N * sizeof(Float), cudaMemcpyHostToDevice);

        // GABA gating variable

        cudaMemcpy(sim_vars_cpu[AN_IE_VAR_YI], cpu_arr.data(), N * sizeof(Float), cudaMemcpyHostToDevice);

        // No neurotransmitter input/output. Connection counts are set to zero for now and edited below. 
        cudaMemcpy(sim_vars_cpu[AN_IE_VAR_EOUTPUT], cpu_arr.data(), N * sizeof(Float), cudaMemcpyHostToDevice);
        cudaMemcpy(sim_vars_cpu[AN_IE_VAR_EIN], cpu_arr.data(), N * sizeof(Float), cudaMemcpyHostToDevice);
        cudaMemcpy(sim_vars_cpu[AN_IE_VAR_ECOUNT], cpu_arr.data(), N * sizeof(Float), cudaMemcpyHostToDevice);
        cudaMemcpy(sim_vars_cpu[AN_IE_VAR_IOUTPUT], cpu_arr.data(), N * sizeof(Float), cudaMemcpyHostToDevice);
        cudaMemcpy(sim_vars_cpu[AN_IE_VAR_IIN], cpu_arr.data(), N * sizeof(Float), cudaMemcpyHostToDevice);
        cudaMemcpy(sim_vars_cpu[AN_IE_VAR_ICOUNT], cpu_arr.data(), N * sizeof(Float), cudaMemcpyHostToDevice);
    }
    // Copy pointers from CPU to GPU 
    sim_vars = sim_vars_cpu;

    // Setup ephys.
    SANEphys ephys(N);

    // Generate coupling 
    AtomicCoupling e_coupling(N), i_coupling(N);
    FileNetwork e_net("cortex_connectivity_1m_1k_exc.h5");
    KNNNetwork  i_net(30, &morphology);
    e_net.Generate(N); i_net.Generate(N);
    ChunkedAdjacencyList e_list{ e_net.ToChunkedAdjacencyList(sim_vars_cpu[AN_IE_VAR_ECOUNT]) };
    ChunkedAdjacencyList i_list{ i_net.ToChunkedAdjacencyList(sim_vars_cpu[AN_IE_VAR_ICOUNT]) };
    std::vector<CouplingStruct> coupling_structs
    {
        {e_list, e_coupling, sim_vars_cpu[AN_IE_VAR_EIN], sim_vars_cpu[AN_IE_VAR_EOUTPUT]},
        {i_list, i_coupling, sim_vars_cpu[AN_IE_VAR_IIN], sim_vars_cpu[AN_IE_VAR_IOUTPUT]},
    };

    // Send the connectivity to graphics.
    Graphics::load_in_connectivity(
        e_net.GetCpuIndices().data(),
        e_net.GetCpuOffsetsIntoIndices().data(),
        N, 1e5,
        length_predicate);

    // Setup electrodes randomly distributed.
    thrust::host_vector<int> h_regions, h_regions_inds;
    std::vector<int> electrode_inds(render_N);
    for(int i = 0; i < render_N; ++i)
        electrode_inds[i] = (N / render_N) * i;
    morphology.GenSphericalRegions(h_regions, h_regions_inds, electrode_inds, 50);
    ElectrodeNet electrode_net(h_regions, h_regions_inds);

    // Note, we multiply by 1000 to convert from sec to ms.
    int const nincs = (1000 * samp_len / an_ie_params::rk4_dt);
    int const print_incs = (int)((1000.0 / samp_freq) / an_ie_params::rk4_dt + 0.5);
    _RunAndPrintSimulation(nincs, print_incs, fout,
        N, sim_vars, sim_vars_cpu,
        &ephys, 
        coupling_structs,
        sim_vars_cpu[AN_IE_VAR_V],
        graphics_voltage_handle,
        render_N,
        &electrode_net);

    for (Float*& arr : sim_vars_cpu)
        cudaFree(arr);
    fout.close();
}

// N : number of neurons to simulate
// samp_freq : sampling frequency (Hz)
// samp_len : How long you record the activity of the model neuron (sec)
void hh_benchmark(int const N = 1, int const samp_freq = 1000, float const samp_len = 10, int render_N = -1)
{
    using namespace hh_params;

    // Set up graphics and morphology
	FileMorphology morphology("cortex_connectivity_1m_1k_inh.h5");
    morphology.Generate(N);
	morphology.Rotate(glm::radians(90.0f), glm::vec3(1, 0, 0));
	morphology.Rotate(glm::radians(-90.0f), glm::vec3(0, 0, 1));

    float* graphics_voltage_handle;
    Graphics::setup(graphics_voltage_handle, morphology.GetPositionsRaw(), N, render_N);

    // Set up simulation variables. Stored in a struct of arrays.
    thrust::host_vector<Float*> sim_vars_cpu(NUM_SIM_PARAMS);
    thrust::device_vector<Float*> sim_vars;
    std::ofstream fout("OUT_HH.txt");
    {
        thrust::host_vector<Float> cpu_arr(N, 0);

        for (Float*& arr : sim_vars_cpu)
            cudaMalloc(&arr, N * sizeof(Float));

        // Set voltages randomly
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<> voltage_dist(-55, 25);
        for (auto& val : cpu_arr)
            val = voltage_dist(gen);
        cudaMemcpy(sim_vars_cpu[SIM_PARAMS_V], cpu_arr.data(), N * sizeof(Float), cudaMemcpyHostToDevice);
        thrust::fill_n(cpu_arr.begin(), N, 0.0);

        // Set gating variables  
        std::normal_distribution<> gating_var_dist(0.0, 0.0);
        cudaMemcpy(sim_vars_cpu[SIM_PARAMS_M], cpu_arr.data(), N * sizeof(Float), cudaMemcpyHostToDevice);

        for (auto& val : cpu_arr)
            val = gating_var_dist(gen);
        cudaMemcpy(sim_vars_cpu[SIM_PARAMS_N], cpu_arr.data(), N * sizeof(Float), cudaMemcpyHostToDevice);

        for (auto& val : cpu_arr)
            val = gating_var_dist(gen);
        cudaMemcpy(sim_vars_cpu[SIM_PARAMS_H], cpu_arr.data(), N * sizeof(Float), cudaMemcpyHostToDevice);

        thrust::fill_n(cpu_arr.begin(), N, 0.0);
        cudaMemcpy(sim_vars_cpu[SIM_PARAMS_LOCAL_Y], cpu_arr.data(), N * sizeof(Float), cudaMemcpyHostToDevice);
        cudaMemcpy(sim_vars_cpu[SIM_PARAMS_IY], cpu_arr.data(), N * sizeof(Float), cudaMemcpyHostToDevice);
        cudaMemcpy(sim_vars_cpu[SIM_PARAMS_EY], cpu_arr.data(), N * sizeof(Float), cudaMemcpyHostToDevice);

        // No neurotransmitter input/output or connections
        cudaMemcpy(sim_vars_cpu[SIM_PARAMS_LOCAL_OUTPUT], cpu_arr.data(), N * sizeof(Float), cudaMemcpyHostToDevice);
        cudaMemcpy(sim_vars_cpu[SIM_PARAMS_LOCAL_IN], cpu_arr.data(), N * sizeof(Float), cudaMemcpyHostToDevice);
        cudaMemcpy(sim_vars_cpu[SIM_PARAMS_LOCAL_COUNT], cpu_arr.data(), N * sizeof(Float), cudaMemcpyHostToDevice);
        cudaMemcpy(sim_vars_cpu[SIM_PARAMS_IOUTPUT], cpu_arr.data(), N * sizeof(Float), cudaMemcpyHostToDevice);
        cudaMemcpy(sim_vars_cpu[SIM_PARAMS_IIN], cpu_arr.data(), N * sizeof(Float), cudaMemcpyHostToDevice);
        cudaMemcpy(sim_vars_cpu[SIM_PARAMS_ICOUNT], cpu_arr.data(), N * sizeof(Float), cudaMemcpyHostToDevice);
        cudaMemcpy(sim_vars_cpu[SIM_PARAMS_EOUTPUT], cpu_arr.data(), N * sizeof(Float), cudaMemcpyHostToDevice);
        cudaMemcpy(sim_vars_cpu[SIM_PARAMS_EIN], cpu_arr.data(), N * sizeof(Float), cudaMemcpyHostToDevice);
        cudaMemcpy(sim_vars_cpu[SIM_PARAMS_ECOUNT], cpu_arr.data(), N * sizeof(Float), cudaMemcpyHostToDevice);

        // Set applied current
        thrust::fill_n(cpu_arr.begin(), N, appcur);
        cudaMemcpy(sim_vars_cpu[SIM_PARAMS_APPCUR], cpu_arr.data(), N * sizeof(Float), cudaMemcpyHostToDevice);
    }
    // Copy pointers from CPU to GPU 
    sim_vars = sim_vars_cpu;

    // Setup Ephys.
    HHEphys ephys(N);

    // Generate coupling
    AtomicCoupling /*loc_coupling(N),*/ e_coupling(N), i_coupling(N);
//    KNNNetwork loc_net(500, &morphology);
//    FileNetwork e_net("cortex_connectivity_1m_1k_exc.h5"), i_net("cortex_connectivity_1m_1k_inh.h5");
    UniformRandomNetwork e_net(800), i_net(200);
    /*loc_net.Generate(N);*/ e_net.Generate(N); i_net.Generate(N);
//    ChunkedAdjacencyList loc_list{ loc_net.ToChunkedAdjacencyList(sim_vars_cpu[SIM_PARAMS_LOCAL_COUNT]) };
    ChunkedAdjacencyList e_list{ e_net.ToChunkedAdjacencyList(sim_vars_cpu[SIM_PARAMS_ECOUNT]) };
    ChunkedAdjacencyList i_list{ i_net.ToChunkedAdjacencyList(sim_vars_cpu[SIM_PARAMS_ICOUNT]) };
    std::vector<CouplingStruct> coupling_structs
    {
//        {loc_list, loc_coupling, sim_vars_cpu[SIM_PARAMS_LOCAL_IN], sim_vars_cpu[SIM_PARAMS_LOCAL_OUTPUT]},
        {e_list, e_coupling, sim_vars_cpu[SIM_PARAMS_EIN], sim_vars_cpu[SIM_PARAMS_EOUTPUT]},
        {i_list, i_coupling, sim_vars_cpu[SIM_PARAMS_IIN], sim_vars_cpu[SIM_PARAMS_IOUTPUT]}
    };

    // Send the connectivity to graphics.
//    Graphics::load_in_connectivity(
//        loc_net.GetCpuIndices().data(),
//        loc_net.GetCpuOffsetsIntoIndices().data(),
//        N, N/10,
//        length_predicate);

    // Setup electrodes based on mouse 2D data from 
    // https://gin.g-node.org/hiobeen/Mouse_hdEEG_ASSR_Hwang_et_al/src/master/montage.csv.
    thrust::host_vector<int> h_regions, h_regions_inds;
    std::vector<int> electrode_inds(render_N);
    for(int i = 0; i < render_N; ++i)
        electrode_inds[i] = (N / render_N) * i;
    morphology.GenSphericalRegions(h_regions, h_regions_inds, electrode_inds, 1);
    ElectrodeNet electrode_net(h_regions, h_regions_inds);

    // Note, we multiply by 1000 to convert from sec to ms.
    int const nincs = (1000 * samp_len / leapfrog_dt);
    int const print_incs = (int)((1000.0 / samp_freq) / leapfrog_dt + 0.5);
    _RunAndPrintSimulation(nincs, print_incs, fout,
        N, sim_vars, sim_vars_cpu,
        &ephys, 
        coupling_structs,
        sim_vars_cpu[SIM_PARAMS_V],
        graphics_voltage_handle,
        render_N,
        &electrode_net);

    for (Float*& arr : sim_vars_cpu)
        cudaFree(arr);
    fout.close();
}

void diekman_benchmark(int const N = 1, int const samp_freq = 1000, float const samp_len = 10, int render_N = -1)
{
    // Set up graphics and morphology
	FileMorphology morphology("scn_atlas.h5");
    morphology.Generate(N);

    float* graphics_voltage_handle;
    Graphics::setup(graphics_voltage_handle, morphology.GetPositionsRaw(), N, render_N);

    // Set up simulation variables. Stored in a struct of arrays.
    thrust::host_vector<Float*> sim_vars_cpu(NUM_DIEKMAN_VAR);
    std::ofstream fout("testcircadian3.txt");
    {
        thrust::host_vector<Float> cpu_arr(N, 0);

        // Set all parameters to zero.
        for (Float*& arr : sim_vars_cpu)
        {
            cudaMalloc(&arr, N * sizeof(Float));
            cudaMemcpy(arr, cpu_arr.data(), N * sizeof(Float), cudaMemcpyHostToDevice);
        }
    }
    // Set EGABA
    set_scn_egaba(sim_vars_cpu[DIEKMAN_VAR_EGABA], &morphology, N, true, true);

    // Copy pointers from CPU to GPU 
    thrust::device_vector<Float*> sim_vars = sim_vars_cpu;

    // Setup Ephys.
    DiekmanEphys ephys(N);

    // Generate coupling
    AtomicCoupling i_coupling(N);
    UniformRandomNetwork i_net(100);
    i_net.Generate(N);
    ChunkedAdjacencyList i_list{ i_net.ToChunkedAdjacencyList(sim_vars_cpu[DIEKMAN_VAR_ICOUNT]) };
    std::vector<CouplingStruct> coupling_structs
    {
        {i_list, i_coupling, sim_vars_cpu[DIEKMAN_VAR_IIN], sim_vars_cpu[DIEKMAN_VAR_IOUTPUT]}
    };

    // Initialize paracrine signalling. 
    Float min_dist = morphology.GetMinimumDistance();
    Float max_dist = morphology.GetMaximumDistance();
    std::cout <<"min/max distances are"<< min_dist<<" "<< max_dist << std::endl;

    Float voxel_size = 10 * min_dist;
    int x_count, y_count, z_count;
    x_count = y_count = z_count = 2 / voxel_size;
    std::cout << "x_count is" << x_count << std::endl;

    thrust::device_vector<Float> d_coords[3]; 
    auto const& positions{ morphology.GetPositions() };

    {
        thrust::host_vector<Float> h_coords[3]; 
        for (int i = 0; i < 3; ++i)
        {
            h_coords[i].resize(N);
            for (int j = 0; j < N; ++j)
                h_coords[i][j] = positions[j][i];
            d_coords[i] = h_coords[i];
        }
    }

    //ParacrineModule paracrine(sim_vars_cpu[DIEKMAN_VAR_PARACRINE], 0, 0, 0, 0, diekman_params::leapfrog_dt, 1, 1, 2, 1, 0.1, {}, {}, {});
    
    ParacrineModule paracrine(sim_vars_cpu[DIEKMAN_VAR_PARACRINE],N,x_count,y_count,z_count,100*diekman_params::leapfrog_dt,100,0.5,voxel_size,2e-5,3e-4,d_coords[0], d_coords[1], d_coords[2]);
    std::cout << paracrine.timescale << std::endl;

    // Send the connectivity to graphics.
//    Graphics::load_in_connectivity(
//        i_net.GetCpuIndices().data(),
//        i_net.GetCpuOffsetsIntoIndices().data(),
//        N, N,
//        length_predicate);

    // Setup electrodes based on mouse 2D data from 
    // https://gin.g-node.org/hiobeen/Mouse_hdEEG_ASSR_Hwang_et_al/src/master/montage.csv.
    thrust::host_vector<int> h_regions, h_regions_inds;
    std::vector<int> electrode_inds(render_N);
    for(int i = 0; i < render_N; ++i)
        electrode_inds[i] = (N / render_N) * i;
    morphology.GenSphericalRegions(h_regions, h_regions_inds, electrode_inds, 1);//last parameter sets the number of nearest neighbors to sample
    ElectrodeNet electrode_net(h_regions, h_regions_inds);
    
    
    // Note, we multiply by 1000 to convert from sec to ms.
    int const nincs = (1000 * samp_len / diekman_params::leapfrog_dt);
    int const print_incs = (int)((1000.0 / samp_freq) / diekman_params::leapfrog_dt + 0.5);

    Float circadian_time = 10.1;
    Circadian circadian("circtable.csv");//initialize circadian class

    _RunAndPrintSimulation(nincs, print_incs, fout,
        N, sim_vars, sim_vars_cpu,
        &ephys, 
        coupling_structs,
        sim_vars_cpu[DIEKMAN_VAR_V],
        graphics_voltage_handle,
        render_N,
        &electrode_net,
        &paracrine,
        circadian_time,
        &circadian);

    for (Float*& arr : sim_vars_cpu)
        cudaFree(arr);
    std::cout << "gaga";
    fout.close();
}

int main()
{
    cusparseCreate(&Csr_matrix::handle);
    int N = 14638;
    int N_electrodes = 1000;
    diekman_benchmark(N, 2500, 100, N_electrodes);
    Graphics::terminate_graphics();
}
