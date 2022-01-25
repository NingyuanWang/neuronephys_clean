#include "mouse_cortex.cuh"
#include "graphics.h"
#include <set>
#include <glm/gtx/norm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <chrono>
#include "regions_tree.h"
#include "hebbian.cuh"

int NumIncsSecond(float dt=HH::leapfrog_dt)
{
    return int(1000 / dt); 
}

void ElectrodesExample(int N)
{
    MouseCortex mouse_cortex(N);

    int KNN_value = 1000;

    // Setup 5 electrodes.
    std::vector<int> electrode_inds = { 0, 1, 2, 3, N - 1 };
    
    // Generate spherical regions at indices.
    thrust::host_vector<int> h_regions, h_regions_inds;
    mouse_cortex.morphology->GenSphericalRegions(h_regions, h_regions_inds, electrode_inds, KNN_value);

    // Construct ElectrodeNet class.
    ElectrodeNet electrode_net(h_regions, h_regions_inds);

    // Get pointer to voltages. 
    Float const* V_ptr = mouse_cortex.ephys->SimVarsCpu()[HH::Params::V];

    // Output file.
    std::ofstream fout("ELECTRODES_OUT.txt");
    for(int i = 0; i < NumIncsSecond() * 3; ++i)
    {
        mouse_cortex.Simulate();

        // Measure average for each individual electrode.
        electrode_net.DoElectrodeMeasurements(V_ptr, mouse_cortex.white_ampa_connectivity->is_firing);

        // Write electrode measurements to file.
        Float const* handle = electrode_net.ElectrodeOutputHandle();
        for (int j = 0; j < electrode_inds.size(); ++j)
            fout << handle[j] << ',';
        fout << std::endl;
    }
    fout.close();
}

void VisualizationTest(int N, int render_N=-1, int electrodeK=10)
{
    if (render_N < 0)
        render_N = N;

    MouseCortex mouse_cortex(N);
    std::vector<int> electrode_inds(render_N);
    int const render_stride = N / render_N;

    // Generate electrodes by sampling render_N neurons uniformly from the N and creating electrodes at every such neuron.
    for (int i = 0; i < render_N; ++i)
        electrode_inds[i] = render_stride * i;
    thrust::host_vector<int> h_regions, h_regions_inds;
    mouse_cortex.morphology->GenSphericalRegions(h_regions, h_regions_inds, electrode_inds, electrodeK);

    ElectrodeNet electrode_net(h_regions, h_regions_inds);
    Float const* avg_V_handle = electrode_net.ElectrodeOutputHandle();
    Float const* V_ptr = mouse_cortex.ephys->SimVarsCpu()[HH::Params::V];

    float* graphics_voltage_handle;
    Graphics::setup(graphics_voltage_handle, mouse_cortex.morphology->GetPositionsRaw(), N, render_N);
    // Hodgkin-Huxley normalized coloring.
    Graphics::set_color_params(15, -60, 0.0);

    // Use super-regions at a specific depth in the regions json tree.
    int visual_region_idx;
    thrust::host_vector<int> h_regions_allen, h_regions_inds_allen, associated_ids;
    try
    {
        RegionsTree tree("CUBIC_regionID_lookup_no_comma.json");
        mouse_cortex.morphology->SpecfiyRegionsDepth(6, tree);
        mouse_cortex.morphology->GenRegionsFromIDs(h_regions_allen, h_regions_inds_allen, associated_ids);

        // Find region corresponding to the visual region in h_regions.
        int ID = tree.GetIdOfRegionByName("VIS");
        std::cout << "Visual region ID: " << ID << std::endl;
        visual_region_idx = std::find(associated_ids.begin(), associated_ids.end(), ID) - associated_ids.begin();

        // Set applied current to zero everywhere but the visual regions.
        thrust::host_vector<Float> appcur(N, 2);
        for (int i = h_regions_inds_allen[visual_region_idx]; i < h_regions_inds_allen[visual_region_idx + 1]; ++i)
            appcur[h_regions_allen[i]] = 7;
        thrust::device_vector<Float> d_appcur(appcur);
        thrust::copy_n(d_appcur.begin(), N, thrust::device_ptr<Float>(mouse_cortex.ephys->SimVarsCpu()[HH::Params::APPCUR]));
    }
    catch (std::runtime_error& e)
    {
        std::cerr << e.what() << std::endl;
        exit(1);
    }

    int switch_incs = NumIncsSecond();
    bool pause = false;
    for(int i = 0; i < NumIncsSecond() * 3; ++i)
    {
        mouse_cortex.Simulate();

        electrode_net.DoElectrodeMeasurements(V_ptr, mouse_cortex.white_ampa_connectivity->is_firing);
        for (int j = 0; j < h_regions_inds.size() - 1; ++j)
            graphics_voltage_handle[j] = avg_V_handle[j];
        Graphics::render(graphics_voltage_handle, nullptr, 0, 0, pause);
    }
}

void RecordNumFiring(int N, std::string fl_suffix="")
{
    std::ofstream fout("N_FIRING_" + fl_suffix + ".txt");
    MouseCortex mouse_cortex(N, "HH", false, false);
    Float const* V_ptr = mouse_cortex.ephys->SimVarsCpu()[HH::Params::V];

    // Transient.
    for(int i = 0; i < NumIncsSecond() * 3; ++i)
        mouse_cortex.Simulate();

    int const n_incs = NumIncsSecond() * 10;
    std::cout << "\r           ";
    for (int i = 0; i < n_incs; ++i)
    {
        if (i % 1000 == 0)
            std::cout << "\r" << i / (float)n_incs * 100 << "%";
        fout << mouse_cortex.Simulate() << '\n';
    }
    std::cout << std::endl;
    fout.close();
}

/// n_incs_hebbian_sample is the number of simulation increments to do before we output hebbian weights to a file.
void HebbianLearningExample(int N, int n_incs_hebbian_sample = 1000)
{
    MouseCortex::use_weights_network = true;

    // Note we are using uniform random weights here and no local connections.
    MouseCortex mouse_cortex(N, "SAN", true, false);
    auto& conns{ mouse_cortex.white_ampa_connectivity->list };
    auto& is_firing{ mouse_cortex.white_ampa_connectivity->is_firing };
    auto& firing_inds{ mouse_cortex.white_ampa_connectivity->is_firing };
    Hebbian heb(300, 0.1, conns.indices.size());
    thrust::host_vector<Float> h_weights = conns.weights;

    // Transient.
    for(int i = 0; i < NumIncsSecond() * 3; ++i)
        mouse_cortex.Simulate();

    int const n_incs = NumIncsSecond() * 10;
    for (int i = 0; i < n_incs; ++i)
    {
        if (i % 1000 == 0)
            std::cout << "\r" << i / (float)n_incs * 100 << "%";

        // Simulation!
        mouse_cortex.Simulate();

        // Perform Hebbian update.
        heb.UpdateWeights(conns, is_firing, firing_inds);

        if (i % n_incs_hebbian_sample == 0)
        {
            // Write hebbian weights to a file.
            std::ofstream fout("HEBBIAN_WEIGHTS_" + std::to_string(i) + ".txt");
            h_weights = conns.weights;
            for (auto& weight : h_weights)
                fout << weight << ',';
            fout.close();
        }
    }
}

// NOTE THAT: this benchmark defaults to using uniform random connections and no gray matter. 
// These can both be switched so that the gray and white matter are realistic. 
// For FIGURE 2.
void SweepBenchmark(std::vector<int> const& choices_of_N, bool uniform_random_white=true, bool use_gray=false)
{
    std::ofstream fout("HH_VARIED_N_OUT.csv");
    int n_incs = NumIncsSecond();
    for (int N : choices_of_N)
    {
        {
            std::cout << "N " << N << std::endl;
            MouseCortex mouse_cortex(N, "HH", uniform_random_white, use_gray);

            // Start of simulation.
            auto start = std::chrono::steady_clock::now();
            for (int i = 0; i < n_incs; ++i)
                mouse_cortex.Simulate();
            auto stop = std::chrono::steady_clock::now();

            // Measure elapsed time and write to row of file in format <N> <TIME>
            float time = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count() / 1000.0f;
            fout << N << ',' << time << std::endl;
        }
    }
    fout.close();
}

// Generate some examples exhibiting waves in the 3D visualization. 
// N is number of neurons.
// render_N is number of neurons to render. If not specified, render_N = N is default.
// ElectrodeK constrols the number of nearest neighbors for electorde measurements.
// For FIGURE 3.
void WavesExamples(int const N, int render_N = -1, int const electrodeK = 1)
{
    if (render_N < 0)
        render_N = N;

    MouseCortex mouse_cortex(N);
    std::vector<int> electrode_inds(render_N);
    int const render_stride = N / render_N;

    // Generate electrodes by sampling render_N neurons uniformly from the N and creating electrodes at every such neuron.
    for (int i = 0; i < render_N; ++i)
        electrode_inds[i] = render_stride * i;
    thrust::host_vector<int> h_regions, h_regions_inds;
    mouse_cortex.morphology->GenSphericalRegions(h_regions, h_regions_inds, electrode_inds, electrodeK);
    ElectrodeNet electrode_net(h_regions, h_regions_inds);
    Float const* avg_V_handle = electrode_net.ElectrodeOutputHandle();
    Float const* V_ptr = mouse_cortex.ephys->SimVarsCpu()[HH::Params::V];

    float* graphics_voltage_handle;
    Graphics::setup(graphics_voltage_handle, mouse_cortex.morphology->GetPositionsRaw(), N, render_N);
    // Hodgkin-Huxley normalized coloring.
    Graphics::set_color_params(15, -60, 0.0);

    // Use super-regions at a specific depth in the regions json tree.
    int visual_region_idx;
    thrust::host_vector<int> h_regions_allen, h_regions_inds_allen, associated_ids;
    try
    {
        RegionsTree tree("CUBIC_regionID_lookup_no_comma.json");
        mouse_cortex.morphology->SpecfiyRegionsDepth(6, tree);
        mouse_cortex.morphology->GenRegionsFromIDs(h_regions_allen, h_regions_inds_allen, associated_ids);

        // Find region corresponding to the visual region in h_regions.
        int ID = tree.GetIdOfRegionByName("AI");
        std::cout << "Visual region ID: " << ID << std::endl;
        visual_region_idx = std::find(associated_ids.begin(), associated_ids.end(), ID) - associated_ids.begin();

        // Set applied current to zero everywhere but the visual regions.
        thrust::host_vector<Float> appcur(N, 2);
        for (int i = h_regions_inds_allen[visual_region_idx]; i < h_regions_inds_allen[visual_region_idx + 1]; ++i)
            appcur[h_regions_allen[i]] = 7;
        thrust::device_vector<Float> d_appcur(appcur);
        thrust::copy_n(d_appcur.begin(), N, thrust::device_ptr<Float>(mouse_cortex.ephys->SimVarsCpu()[HH::Params::APPCUR]));
    }
    catch (std::runtime_error& e)
    {
        std::cerr << e.what() << std::endl;
        exit(1);
    }

    // Simulate and record ERP.
    std::ofstream fout("N_FIRING_APPCUR.csv");
    int switch_incs = NumIncsSecond();
    for (int window = 0; window < 3; ++window)
    {
        int n_incs_transient = NumIncsSecond() * 1;
        int n_incs_record = NumIncsSecond() * 2;
        float stim = window == 0 ? 0 : (window == 1 ? 5 : 10);
        float base = window == 0 ? 0 : 3;
        bool pause = false;
        for (int i = 0; i < n_incs_transient + n_incs_record; ++i)
        {
            thrust::host_vector<Float> appcur(N, base);
            Float appcur_write = base;
            if (i % switch_incs == 0)
                std::cout << "SWITCH : " << (i % (2 * switch_incs)) / switch_incs << ", i = " << i << std::endl;

            if((i < n_incs_transient || i % (2 * switch_incs) < switch_incs) && i % (switch_incs / 500) < 2)
            {
//                for (int j = h_regions_inds_allen[visual_region_idx]; j < h_regions_inds_allen[visual_region_idx + 1]; ++j)
//                    appcur[h_regions_allen[j]] = stim;
                for (int j = 0; j < N; ++j)
                    if (mouse_cortex.morphology->GetPositions()[j].y > 0.0)
                        appcur[j] = stim;
//                thrust::fill_n(appcur.begin(), N, stim);
                appcur_write = stim;
            }
            thrust::device_vector<Float> d_appcur(appcur);
            thrust::copy_n(d_appcur.begin(), N, thrust::device_ptr<Float>(mouse_cortex.ephys->SimVarsCpu()[HH::Params::APPCUR]));

            fout << mouse_cortex.Simulate() << ',' << appcur_write << '\n';

            if (i > n_incs_transient)
            {
                electrode_net.DoElectrodeMeasurements(V_ptr, mouse_cortex.white_ampa_connectivity->is_firing);
                for (int j = 0; j < h_regions_inds.size() - 1; ++j)
                    graphics_voltage_handle[j] = avg_V_handle[j];
                Graphics::render(graphics_voltage_handle, nullptr, 0, 0, pause);
            }
        }
    }
    fout.close();
}

// Simulates ERP for three cases: realistic connectivity, realistic short range and uniform long range, and uniform long range. 
// For FIGURE 4.
void SimulatedERP(int const N=1e6)
{
    MouseCortex::excitatory_conns_file_name = "cortex_connectivity_1m_1k_exc.h5";
    MouseCortex::inhibitory_conns_file_name = "cortex_connectivity_1m_1k_inh.h5";

    // Figure out what the indices of the electrodes should be.
    thrust::host_vector<int> h_regions, h_regions_inds, associated_ids;

    // Regions to sample EEG for. This differs from the above because we want an equal number of neurons per region for EEG.
    thrust::host_vector<int> eeg_sample_regions, eeg_sample_regions_inds; 
    {
        MouseCortex mouse_cortex(N, "HH", false, false);
        // Use super-regions at a specific depth in the regions json tree.
        try
        {
            RegionsTree tree("CUBIC_regionID_lookup_no_comma.json");
            mouse_cortex.morphology->SpecfiyRegionsDepth(6, tree);
            mouse_cortex.morphology->GenRegionsFromIDs(h_regions, h_regions_inds, associated_ids);
            
            // Sample the same number of neurons from each subregion to measure EEG for.
            int N_sample = INT_MAX;
            for (int i = 0; i < h_regions_inds.size() - 1; ++i)
            {
                if(h_regions_inds[i+1] - h_regions_inds[i] < N_sample)
                    N_sample = h_regions_inds[i+1] - h_regions_inds[i];
                if(h_regions_inds[i+1] - h_regions_inds[i] < N_sample)
                    N_sample = h_regions_inds[i+1] - h_regions_inds[i];
            }
            std::cout << "Sampling the first " << N_sample << " neurons from each region" << std::endl;

            eeg_sample_regions.resize((h_regions_inds.size() - 1) * N_sample);
            eeg_sample_regions_inds.resize(h_regions_inds.size(), 0);
            for (int i = 0; i < h_regions_inds.size() - 1; ++i)
            {
                eeg_sample_regions_inds[i + 1] = eeg_sample_regions_inds[i] + N_sample;
                for (int j = 0; j < N_sample; ++j)
                    eeg_sample_regions[eeg_sample_regions_inds[i] + j] = h_regions[h_regions_inds[i] + j];
            }
        }
        catch (std::runtime_error& e)
        {
            std::cerr << e.what() << std::endl;
            exit(1);
        }
    }
    int n_incs_transient = NumIncsSecond() * 3;
    int n_incs_record = NumIncsSecond() * 1;

    ElectrodeNet electrode_net(eeg_sample_regions, eeg_sample_regions_inds);
    Float const* avg_V_handle = electrode_net.ElectrodeOutputHandle();

    // Simulate and record the three cases: gray + white, gray + uniform, uniform only.
    bool use_uniform_white[3]{ false, true, true };
    bool use_gray[3]{ true, true, false };
    std::string fout_names[3]{ "ERP_OUT_GRAY_WHITE.csv", "ERP_OUT_GRAY_UNIFORM.csv", "ERP_OUT_UNIFORM.csv" };
    for(int case_idx = 0; case_idx < 3; ++case_idx)
    {
        std::cout << " USE UNIFORM : " << use_uniform_white[case_idx] << ", USE GRAY : " << use_gray[case_idx] << std::endl;
        MouseCortex mouse_cortex(N, "HH", use_uniform_white[case_idx], use_gray[case_idx]);
        std::ofstream fout(fout_names[case_idx]);
        Float const* V_ptr = mouse_cortex.ephys->SimVarsCpu()[HH::Params::V];

        // Set applied current to 6 everywhere.
        thrust::host_vector<Float> appcur(N, 6);
        thrust::device_vector<Float> d_appcur(appcur);
        thrust::copy_n(d_appcur.begin(), N, thrust::device_ptr<Float>(mouse_cortex.ephys->SimVarsCpu()[HH::Params::APPCUR]));

        // Transient.
        for (int i = 0; i < n_incs_transient; ++i)
            mouse_cortex.Simulate();

        // Simulate and record ERP.
        for (int i = 0; i < n_incs_record; ++i)
        {
            electrode_net.DoElectrodeMeasurements(V_ptr, mouse_cortex.white_ampa_connectivity->is_firing);
            mouse_cortex.Simulate();

            for (int j = 0; j < eeg_sample_regions_inds.size() - 1; ++j)
                fout << avg_V_handle[j] << ',';
            fout << '\n';
        }
        fout.close();
    }
}

// Given connectivity and regions, writes a matrix R to a file 
// where R_{ij} is the number of connections from region i to region j.
void EvaluatePhysicalConnectivity(std::string fout_name, 
    thrust::host_vector<int> const& region_per_neuron, thrust::host_vector<int> const& associated_ids,
    AdjacencyList const& conns)
{
    std::cout << "Writing physical connectivity to file: " << fout_name << "..." << std::endl;
    std::ofstream fout(fout_name);
    int N_regions = associated_ids.size();
    std::vector<std::vector<int>> phys_conns(N_regions, std::vector<int>(N_regions, 0));

    // Copy adjacency list to cpu.
    thrust::host_vector<int> indices = conns.indices;
    thrust::host_vector<int> offset_in_indices = conns.offset_in_indices;

    for (int n = 0; n < region_per_neuron.size(); ++n)
    {
        // Get associated region for this neuron.
        int reg = std::find(associated_ids.begin(), associated_ids.end(), region_per_neuron[n]) - associated_ids.begin();

        // Iterate over downstream neurons and incremement phys_conns respectively for each region.
        for (int d = offset_in_indices[n]; d < offset_in_indices[n + 1]; ++d)
        {
            int downstream = indices[d];
            int downstream_reg = std::find(associated_ids.begin(), associated_ids.end(), region_per_neuron[downstream]) - associated_ids.begin();
            phys_conns[reg][downstream_reg] += 1;
        }
    }
    
    // Write physical connectivity to file.
    for (int i = 0; i < N_regions; ++i)
    {
        for (int j = 0; j < N_regions; ++j)
            fout << phys_conns[i][j] << ',';
        fout << '\n';
    }
    fout.close();
}

// Record fMRI in regions for use in functional connectivity.
// A specified region is stimulated.
// For FIGURE 5-6.
void FunctionalData(int const N=1e6, std::string stim_region="VIS")
{
    MouseCortex::excitatory_conns_file_name = "cortex_connectivity_1m_1k_exc.h5";
    MouseCortex::inhibitory_conns_file_name = "cortex_connectivity_1m_1k_inh.h5";

    // Figure out what the indices of the electrodes should be.
    thrust::host_vector<int> h_regions, h_regions_inds, associated_ids;

    // Regions to sample EEG for. This differs from the above because we want an equal number of neurons per region for EEG.
    thrust::host_vector<int> eeg_sample_regions, eeg_sample_regions_inds; 
    int stim_region_idx = -1;
    std::ofstream region_sizes_fout("REGION_SIZES.csv");
    std::ofstream region_names("REGION_NAMES.txt");
    {
        MouseCortex mouse_cortex(N, "HH", false, false);

        // Use super-regions at a specific depth in the regions json tree.
        try
        {
            RegionsTree tree("CUBIC_regionID_lookup_no_comma.json");
            mouse_cortex.morphology->SpecfiyRegionsDepth(6, tree);
            mouse_cortex.morphology->GenRegionsFromIDs(h_regions, h_regions_inds, associated_ids);

            // Print region names to file.
            for (int id : associated_ids)
                region_names << tree.GetNameOfRegionById(id) << '\n';

            // Find region corresponding to the visual region in h_regions.
            int ID = tree.GetIdOfRegionByName(stim_region);
            std::cout << stim_region << " region ID: " << ID << std::endl;
            stim_region_idx = std::find(associated_ids.begin(), associated_ids.end(), ID) - associated_ids.begin();
            
            // Sample the same number of neurons from each subregion to measure EEG for.
            int N_sample = INT_MAX;
            for (int i = 0; i < h_regions_inds.size() - 1; ++i)
            {
                region_sizes_fout << h_regions_inds[i + 1] - h_regions_inds[i] << '\n';
                if(h_regions_inds[i+1] - h_regions_inds[i] < N_sample)
                    N_sample = h_regions_inds[i+1] - h_regions_inds[i];
                if(h_regions_inds[i+1] - h_regions_inds[i] < N_sample)
                    N_sample = h_regions_inds[i+1] - h_regions_inds[i];
            }
            std::cout << "Sampling the first " << N_sample << " neurons from each region" << std::endl;

            eeg_sample_regions.resize((h_regions_inds.size() - 1) * N_sample);
            eeg_sample_regions_inds.resize(h_regions_inds.size(), 0);
            for (int i = 0; i < h_regions_inds.size() - 1; ++i)
            {
                eeg_sample_regions_inds[i + 1] = eeg_sample_regions_inds[i] + N_sample;
                for (int j = 0; j < N_sample; ++j)
                    eeg_sample_regions[eeg_sample_regions_inds[i] + j] = h_regions[h_regions_inds[i] + j];
            }
            EvaluatePhysicalConnectivity("PHYSICAL_CONNECTIVITY.csv", mouse_cortex.morphology->GetRegions(), associated_ids, mouse_cortex.white_ampa_connectivity->list);
        }
        catch (std::runtime_error& e)
        {
            std::cerr << e.what() << std::endl;
            exit(1);
        }
    }
    region_sizes_fout.close();
    region_names.close();
    int n_incs_transient = NumIncsSecond() * 3;
    int n_incs_record = NumIncsSecond() * 300;

    ElectrodeNet electrode_net(eeg_sample_regions, eeg_sample_regions_inds);
    Float const* avg_V_handle = electrode_net.ElectrodeOutputHandle();
    int const* firing_handle = electrode_net.FiringPerRegionHandle();

    // Run simulation
    MouseCortex mouse_cortex(N, "HH", false, true);
    std::ofstream firing_fout("FMRI_" + stim_region + ".csv");
    Float const* V_ptr = mouse_cortex.ephys->SimVarsCpu()[HH::Params::V];

    // Set applied current to zero everywhere but the stim region.
    thrust::host_vector<Float> appcur(N, 6);
    for (int i = h_regions_inds[stim_region_idx]; i < h_regions_inds[stim_region_idx + 1]; ++i)
        appcur[h_regions[i]] = 10;
    thrust::device_vector<Float> d_appcur(appcur);
    thrust::copy_n(d_appcur.begin(), N, thrust::device_ptr<Float>(mouse_cortex.ephys->SimVarsCpu()[HH::Params::APPCUR]));

    // Transient.
    for (int i = 0; i < n_incs_transient; ++i)
        mouse_cortex.Simulate();

    // Simulate and record ERP.
    for (int i = 0; i < n_incs_record; ++i)
    {
        electrode_net.DoElectrodeMeasurements(V_ptr, mouse_cortex.white_ampa_connectivity->is_firing);
        mouse_cortex.Simulate();

        for (int j = 0; j < eeg_sample_regions_inds.size() - 1; ++j)
            firing_fout << firing_handle[j] << ',';
        firing_fout << '\n';
    }
    firing_fout.close();
}

int main()
{
    FunctionalData(1e6);
//    ElectrodesExample(1e5);
}
