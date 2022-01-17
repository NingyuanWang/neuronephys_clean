#pragma once
#include "defines.cuh"


//! EPhys is an abstract class that handles electrophysiological details for a certain model.
class Ephys
{
protected:
    int const N;
    Float dt;
    thrust::device_vector<curandState_t> random_states;
    thrust::host_vector<Float*> sim_vars_cpu;
    thrust::device_vector<Float*> sim_vars;
public:
    //! Default constructor. This constructor sets up the random_states variable which is a collection of random seeds for curand.
    //! Most Ephys models need random seeds for noise, etc. 
    //! \param [in] N_ Number of neurons.
    //! \param [in] dt_ Timstep for simulation.
    Ephys(int const N_, Float const dt_);

    //! Pure virtual function intended to simulate electrophysiology.
    virtual void SimulateEphys() = 0;

    //! Getter for struct of arrays storing variables. 
    thrust::device_vector<Float*> SimVars() const { return sim_vars; }

    //! Getter for struct of arrays storing variables. CPU version: the pointers to the data (not the data itself) exist on the CPU.
    thrust::host_vector<Float*> SimVarsCpu() const { return sim_vars_cpu; }

    //! Get simulation timestep.
    Float Dt() const { return dt; }

    //! Free all the data allocated dynamically.
    virtual ~Ephys();
};

//! Hodgkin-Huxley model (https://neuronaldynamics.epfl.ch/online/Ch2.S2.html)
class HHEphys : public Ephys
{
public:
    HHEphys(int const N_);
    virtual void SimulateEphys() override;
};
//! HH parameters and constants.
namespace HH
{
    constexpr static const Float rk4_dt = 0.04;                                                          // Timestep used with RK4 numerical method (ms)
    constexpr static const Float leapfrog_dt = 0.4;                                                      // Timestep used with leapfrog numerical method (ms)
    constexpr static const Float C = 1;
    constexpr static const Float gna = 120;  constexpr static const Float ena = 55;                      // Sodium conductance and potential 
    constexpr static const Float gk = 36; constexpr static const Float ek = -72;                         // Potassium conductance and potential
    constexpr static const Float gl = 0.3; constexpr static const Float el = -50;                        // Leak conductance and potential 
    constexpr static const Float Tmax = 1.0;                                                             // Max neurotransmitter output
    constexpr static const Float Vt = -20.0; constexpr static const Float Kp = 3.0;                      // Neurotransmitter potential and scalar 
    constexpr const static Float Ar = 5.0; constexpr const static Float Ad = 0.18;                       // alpha and beta for GABA gating variable
    constexpr static const Float g_ampa_white = 0.0; constexpr static const Float e_ampa = 0.0; // Coupling conductance and potential for AMPA in white matter
    constexpr static const Float g_ampa_grey = 0.1; // Coupling conductance and potential for AMPA in grey matter
    constexpr static const Float g_gaba = 0.0; constexpr static const Float e_gaba = -70.0; // Coupling conductance and potential for GABA
    constexpr static const Float appcur = 0.0;                                                            // Base current to apply when we only apply a constant current. 
    constexpr static const Float white_noise_mean = 0.0f;
    constexpr static const Float white_noise_intensity = 2.0f; // Controls stddev of white noise.
    enum Params
    {
        // Main parameters:
        V=0, M, N, H,
        // Coupling gating variables:
        LOCAL_Y, IY, EY,
        // Local coupling parameters:
        LOCAL_OUTPUT, LOCAL_IN, LOCAL_COUNT,
        // Excitatory coupling parameters:
        EOUTPUT, EIN, ECOUNT,
        // Inhibitory coupling parameters:
        IOUTPUT, IIN, ICOUNT,
        // Applied current:
        APPCUR,
        // Number of parameters:
        PARAM_COUNT 
    };
};

//! SAN Model (https://pubmed.ncbi.nlm.nih.gov/26996081/)
class SANEphys : public Ephys
{
public:
    SANEphys(int const N_);
    virtual void SimulateEphys() override;
};
//! SAN parameters and constants.
namespace SAN
{
    constexpr static const Float rk4_dt = 0.04;
    constexpr static const Float an_Vt = -45; constexpr static const Float an_Kp = 1;
    constexpr static const Float Tmax = 1.0;
    constexpr static const Float C = 1;
    constexpr static const Float A = 0.02;
    constexpr static const Float vL = -60.95;
    constexpr static const Float vNa = 55;
    constexpr static const Float vK = -100;
    constexpr static const Float vCa = 120;
    constexpr static const Float kD = 30;
    constexpr static const Float vAMPA = 0;
    constexpr static const Float vNMDA = 0;
    constexpr static const Float vGABA = -70;
    constexpr static const Float alphaCa = 0.5;
    constexpr static const Float tauAMPA_white = 2;
    constexpr static const Float tauAMPA_grey = 2;
    constexpr static const Float tausNMDA = 100;
    constexpr static const Float tauxNMDA = 2;
    constexpr static const Float tauGABA = 10;
    constexpr static const Float gL = 0.016307;
    constexpr static const Float gNa = 12.2438;
    constexpr static const Float gK = 19.20436;
    constexpr static const Float gNaP = 0.63314;
    constexpr static const Float gCa = 0.1624;
    constexpr static const Float gKCa = 0.7506;
    constexpr static const Float gAMPA_white = 0.0; // Coupling conductance and potential for AMPA in white matter
    constexpr static const Float gAMPA_grey = 0.5; // Coupling conductance and potential for AMPA in grey matter
    constexpr static const Float gNMDA = 0.0; // 0.0434132;
    constexpr static const Float gGABA = 0.5; //TODO: Change to non-averaged value
    constexpr static const Float tauCa = 0.5 * 739.09; // Controls calcium decay rate. Decrease to decrease the length of the resting state between bursts. Around half is close to transition from wake to sleep.
    constexpr static const Float white_noise_mean = 0.1f;
    constexpr static const Float white_noise_intensity = 0.05f; // Controls stddev of white noise.
    enum Params
    {
        // Main parameters:
        V=0, NK, CA,
        // Coupling gating variables:
        LOCAL_Y, IY, EY,
        // Local coupling parameters:
        LOCAL_OUTPUT, LOCAL_IN, LOCAL_COUNT,
        // Excitatory coupling parameters:
        EOUTPUT, EIN, ECOUNT,
        // Inhibitory coupling parameters:
        IOUTPUT, IIN, ICOUNT,
        // Number of parameters:
        PARAM_COUNT 
    };
};