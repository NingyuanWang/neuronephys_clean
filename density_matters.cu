#include "network_factory.cuh"
#include "graphics.h"
#include "morphology.h"
#include <set>
#include "electrode_net.cuh"
#include <glm/gtx/norm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include "ephys.cuh"

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
    AdjacencyList& list;
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

void run(int const N = 1, int const samp_freq = 1000, float const samp_len = 10, int render_N = -1)
{
    using namespace HH;

    // Set up graphics and morphology
	FileMorphology morphology("cortex_connectivity_1m_1k_inh.h5", true);
    morphology.Generate(N);
	morphology.Rotate(glm::radians(90.0f), glm::vec3(1, 0, 0));
	morphology.Rotate(glm::radians(-90.0f), glm::vec3(0, 0, 1));
    morphology.ProjectNeuronsToPlane(1);
	morphology.Rotate(glm::radians(90.0f), glm::vec3(0, 0, 1));

    float* graphics_voltage_handle;
#ifdef OUTPUT_ELECTRODES_FILE
    Graphics::setup(graphics_voltage_handle, (float*)projected_electrode_positions.data(), 36, 36);
    std::ofstream fout("OUT_HH.txt");
#else
    Graphics::setup(graphics_voltage_handle, morphology.GetPositionsRaw(), N, render_N);
#endif
    Graphics::set_color_params(15, -60, 0.0);

    HHEphys ephys(N);
    auto sim_vars_cpu = ephys.SimVarsCpu();

    // Generate coupling
    AtomicCoupling loc_coupling(N), e_coupling(N), i_coupling(N);
    NormalNetwork loc_net(&morphology, 0.03);
    FileNetwork e_net("cortex_connectivity_1m_1k_exc.h5"), i_net("cortex_connectivity_1m_1k_inh.h5");
    loc_net.Generate(N); e_net.Generate(N); i_net.Generate(N);
    AdjacencyList e_list{ e_net.ToAdjacencyList(sim_vars_cpu[Params::ECOUNT]) };
    AdjacencyList loc_list{ loc_net.ToAdjacencyList(sim_vars_cpu[Params::LOCAL_COUNT]) };
    AdjacencyList i_list{ i_net.ToAdjacencyList(sim_vars_cpu[Params::ICOUNT]) };

    std::vector<CouplingStruct> coupling_structs
    {
        {loc_list, loc_coupling, sim_vars_cpu[Params::LOCAL_IN], sim_vars_cpu[Params::LOCAL_OUTPUT]},
        {e_list, e_coupling, sim_vars_cpu[Params::EIN], sim_vars_cpu[Params::EOUTPUT]},
        {i_list, i_coupling, sim_vars_cpu[Params::IIN], sim_vars_cpu[Params::IOUTPUT]}
    };

    thrust::host_vector<int> h_regions, h_regions_inds;
    std::vector<int> electrode_inds(render_N);
    for(int i = 0; i < render_N; ++i)
        electrode_inds[i] = (N / render_N) * i;
#ifdef OUTPUT_ELECTRODES_FILE
//    morphology.GenSphericalRegions(h_regions, h_regions_inds, electrodes_cloud.GetPositions(), 1000);
    morphology.GenSphericalRegions(h_regions, h_regions_inds, electrode_inds, 1000);
 //   morphology.GenRegionsFromIDs(h_regions, h_regions_inds);
//    morphology.GenCustomRegions([](glm::vec3 const& pos) {return int(2 * (pos.x + 1)); }, 4, h_regions, h_regions_inds);
#else
    morphology.GenSphericalRegions(h_regions, h_regions_inds, electrode_inds, 1);
//    morphology.GenRegionsFromIDs(h_regions, h_regions_inds);

    // Send the connectivity to graphics.
    Graphics::load_in_connectivity(
        i_net.GetCpuIndices().data(),
        i_net.GetCpuOffsetsIntoIndices().data(),
        N, 1000);
#endif
    ElectrodeNet electrode_net(h_regions, h_regions_inds);

    // Note, we multiply by 1000 to convert from sec to ms.
    int const nincs = (1000 * samp_len / leapfrog_dt);
    int const print_incs = (int)((1000.0 / samp_freq) / leapfrog_dt + 0.5);

    thrust::host_vector<Float> h_appcur(N);
    bool pause = false;
    for (int i = 0; i < nincs; ++i)
    {
        if (i % (nincs / 100) == 0)
            std::cout << 100 * (i / (float)nincs) << "%" << std::endl;

//        // Periodic input
//        float t = i * ephys->Dt();
//        if (fmod(t, 2000) > 1000 && fmod(t - 1000, 1000.0 / 50) <= 1)
//            thrust::fill_n(thrust::device_ptr<Float>(sim_vars_cpu[Params::APPCUR]), N, hh_params::appcur + 3.0f);
//        else
//            thrust::fill_n(thrust::device_ptr<Float>(sim_vars_cpu[Params::APPCUR]), N, hh_params::appcur);

        if (i % print_incs == 0)
        {
                electrode_net.DoElectrodeMeasurements(sim_vars_cpu[Params::V]);
                Float const* avg_V_handle = electrode_net.ElectrodeOutputHandle();
#ifdef OUTPUT_ELECTRODES_FILE
                for(int k = 0; k < render_N; ++k)
                    fout << avg_V_handle[k] << ',';
                fout << std::endl;
#else
                for (int k = 0; k < render_N; ++k)
                    graphics_voltage_handle[k] = avg_V_handle[k];
                pause = Graphics::render(graphics_voltage_handle, nullptr, 0, 0, pause);
#endif
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

        // Update ODE 
        ephys.SimulateEphys();
    }

    for (Float*& arr : sim_vars_cpu)
        cudaFree(arr);
#ifdef OUTPUT_ELECTRODES_FILE
    fout.close();
#endif
}

int main()
{
#ifdef OUTPUT_ELECTRODES_FILE
    int N_electrodes = 4; // 239;
#else
    int N_electrodes = 1e5;
#endif
    run(1e5, 2500, 12, N_electrodes);
    Graphics::terminate_graphics();
}
