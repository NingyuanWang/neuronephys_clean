#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/sequence.h>
#include <fstream>
#include <stdio.h>
#include <random>
#include <H5Cpp.h>
#include <chrono>
#include <vector>
#include <cusparse.h>
#include <cublas.h>

// we need these includes for CUDA's random number stuff
#include <curand.h>
#include <curand_kernel.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}
#define cusparseErrchk(ans) { cusparseAssert((ans), __FILE__, __LINE__); }
inline void cusparseAssert(cusparseStatus_t code, const char* file, int line, bool abort = true)
{
    if (code != CUSPARSE_STATUS_SUCCESS)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cusparseGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

//! Just some utility functions that can be used throughout the codebase.
namespace Utility
{
    std::default_random_engine& UnseededRandomEngine();
	std::default_random_engine& SeededRandomEngine();
};

#define USE_DOUBLE_PRECISION 0 
#if USE_DOUBLE_PRECISION
typedef double Float;
#else
typedef float Float;
#endif // USE_DOUBLE_PRECISION
//SIM: Hodgkin Huxley
#define SIM_PARAMS_V 0
#define SIM_PARAMS_M 1
#define SIM_PARAMS_N 2
#define SIM_PARAMS_H 3
#define SIM_PARAMS_LOCAL_Y 4
#define SIM_PARAMS_IY 5
#define SIM_PARAMS_EY 6

#define SIM_PARAMS_LOCAL_OUTPUT 7
#define SIM_PARAMS_LOCAL_IN 8//input synaptic neurotransmitter
#define SIM_PARAMS_LOCAL_COUNT 9//count of presynaptic neurons

#define SIM_PARAMS_EOUTPUT 10
#define SIM_PARAMS_EIN 11//input synaptic neurotransmitter
#define SIM_PARAMS_ECOUNT 12//count of presynaptic neurons

#define SIM_PARAMS_IOUTPUT 13
#define SIM_PARAMS_IIN 14//input synaptic neurotransmitter
#define SIM_PARAMS_ICOUNT 15//count of presynaptic neurons

#define SIM_PARAMS_APPCUR 16
#define NUM_SIM_PARAMS 17//Dimension of state space

// TODO : SET NUMBER OF THREADS DYNAMICALLY BASED ON CARD USED
#define NTHREADS 64 
#define NBLOCKS 32
#define NCHUNKS 2
#define N_PER_CHUNK 7319
#define NSAMPLES_PRINT 501

namespace hh_params {
    constexpr static const Float rk4_dt = 0.04;                                                          // Timestep used with RK4 numerical method (ms)
    constexpr static const Float leapfrog_dt = 0.4;                                                      // Timestep used with leapfrog numerical method (ms)
    constexpr static const Float C = 1;
    constexpr static const Float gna = 120;  constexpr static const Float ena = 55;                      // Sodium conductance and potential 
    constexpr static const Float gk = 36; constexpr static const Float ek = -72;                         // Potassium conductance and potential
    constexpr static const Float gl = 0.3; constexpr static const Float el = -50;                        // Leak conductance and potential 
    constexpr static const Float Tmax = 1.0;                                                             // Max neurotransmitter output
    constexpr static const Float Vt = -20.0; constexpr static const Float Kp = 3.0;                      // Neurotransmitter potential and scalar 
    constexpr const static Float Ar = 5.0; constexpr const static Float Ad = 0.18;                       // alpha and beta for GABA gating variable
    constexpr static const Float g_ampa = 0.1; constexpr static const Float e_ampa = 0.0; // Coupling conductance and potential for AMPA
    constexpr static const Float g_gaba = 0.1; constexpr static const Float e_gaba = -70.0; // Coupling conductance and potential for GABA
    constexpr static const Float appcur = 0.0;                                                            // Base current to apply when we only apply a constant current. 
    constexpr static const Float white_noise_mean = 3.0f;
    constexpr static const Float white_noise_intensity = 1.5f; // Controls stddev of white noise.
}


//PARAMETERS FOR AN MODEL with INHIBITORY-EXCITATORY NETWORK
//excitatory: AMPA, NMDA. inhibitory: GABA
#define AN_IE_VAR_V 0
#define AN_IE_VAR_NK 1 
#define AN_IE_VAR_CA 2
#define AN_IE_VAR_YE 3
#define AN_IE_VAR_YI 4
#define AN_IE_VAR_EIN 5 //excitatory input synaptic neurotransmitter
#define AN_IE_VAR_ECOUNT 6//count of presynaptic excitatory neurons
#define AN_IE_VAR_EOUTPUT 7//neurotransmitter output
#define AN_IE_VAR_IIN 8 //inhibitory input synaptic neurotransmitter
#define AN_IE_VAR_ICOUNT 9//count of presynaptic inhibitory neurons
#define AN_IE_VAR_IOUTPUT 10//neurotransmitter output
#define NUM_AN_IE_VAR 11 //Dimension of state space
namespace an_ie_params {
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
    constexpr static const Float tauAMPA = 2;
    constexpr static const Float tausNMDA = 100;
    constexpr static const Float tauxNMDA = 2;
    constexpr static const Float tauGABA = 10;
    constexpr static const Float gL = 0.016307;
    constexpr static const Float gNa = 12.2438;
    constexpr static const Float gK = 19.20436;
    constexpr static const Float gNaP = 0.63314;
    constexpr static const Float gCa = 0.1624;
    constexpr static const Float gKCa = 0.7506;
    constexpr static const Float gAMPA = 0.01;
    constexpr static const Float gNMDA = 0.0; // 0.0434132;
    constexpr static const Float gGABA = 0.5; //TODO: Change to non-averaged value
    constexpr static const Float tauCa = 0.5 * 739.09; // Controls calcium decay rate. Decrease to decrease the length of the resting state between bursts. Around half is close to transition from wake to sleep.
    constexpr static const Float white_noise_mean = 0.1f;
    constexpr static const Float white_noise_intensity = 0.05f; // Controls stddev of white noise.
};

//PARAMETERS FOR DIEKMAN-FORGER MODEL OF SCN NEURONS.
#define DIEKMAN_VAR_V 0 // Voltage
#define DIEKMAN_VAR_M 1 
#define DIEKMAN_VAR_H 2 
#define DIEKMAN_VAR_N 3
#define DIEKMAN_VAR_RL 4
#define DIEKMAN_VAR_RNL 5 
#define DIEKMAN_VAR_FNL 6 
#define DIEKMAN_VAR_S 7 
#define DIEKMAN_VAR_P 8
#define DIEKMAN_VAR_Y 9 
#define DIEKMAN_VAR_CAS 10
#define DIEKMAN_VAR_CAC 11 
#define DIEKMAN_VAR_G_BIOCHEM 12 // Biochemical model input
#define DIEKMAN_VAR_IIN 13
#define DIEKMAN_VAR_ICOUNT 14
#define DIEKMAN_VAR_IOUTPUT 15
#define DIEKMAN_VAR_EGABA 16 // GABA reversal potential
#define DIEKMAN_VAR_PARACRINE 17 // Paracrine signalling input
#define DIEKMAN_VAR_GKCA 18
#define DIEKMAN_VAR_GKLEAK 19
#define NUM_DIEKMAN_VAR 20
namespace diekman_params {
    constexpr static const Float leapfrog_dt = 0.4;// ms Timestep to use.
    constexpr static const Float Cm = 5.7;         // pF/um^2  membrane capacitance per unit area
    constexpr static const Float gna = 229;        // nS/um^2  Na+ conductance per unit area
    constexpr static const Float gnaleak = 0.0576;      // nS/um^2  Na+ leak conductance per unit area
    constexpr static const Float gk = 3;           // nS/um^2  K+ conductance per unit area
    constexpr static const Float gcal = 6;         // nS/um^2  Ca++ conductance per unit area for L-type channel
    constexpr static const Float gcanl = 20;       // nS/um^2  Ca++ conductance per unit area for non L-type channel
    constexpr static const Float Ena = 45;         // mV       equilibrium potential for Na+
    constexpr static const Float Ek = -97;         // mV       equilibrium potential for K+
    constexpr static const Float Eca = 54;         // mV       equilibrium potential for K+
    constexpr static const Float K1 = 3.93 * 1e-5; // mM       parameter for the value of fL
    constexpr static const Float K2 = 6.55 * 1e-4; // mM       parameter for the value of fL
    constexpr static const Float kcas = 1.65e-4;   // mM / fC  calcium current to concentration conversion factor (outer)
    constexpr static const Float kcac = 8.59e-9;   // mM / fC  calcium current to concentration conversion factor (inner)
    constexpr static const Float taucas = 0.1;     // ms       calcium clearance time constant (outer)
    constexpr static const Float taucac = 1.75e3;  // ms       calcium clearance time constant (inner)
    constexpr static const Float bcas = 5.425e-4;  // mM / ms  ????????????? (outer)
    constexpr static const Float bcac = 3.1e-8;    // mM / ms  ????????????? (inner)
    //constexpr static const Float Iapp = 0;         // pA       constant applied current into the cell
    constexpr static const Float ar = 5;           // 1/mM/ms  activation rate of the gaba synapse
    constexpr static const Float ad = 0.18;        // 1/ms     de-activation rate of the gaba synapse
    constexpr static const Float Tmax = 1;         // mM       maximum neurotransmitter in the gaba synapse
    constexpr static const Float Vt = -20;         // mV       neurontransmitter threshold
    constexpr static const Float Kp = 3;           // mV       neurontransmitter activation rate
    constexpr static const Float clk = 2.2;        //          strength of the influence of the molecular clock on the electrical activity
    constexpr static const Float gto = 1.66;        //          gto is the molecular clock variable corresponding to GSK3 activity
    constexpr static const Float g_gaba = 0.01;     // GABA coupling conductance coefficient.
    constexpr static const Float g_paracrine = 0;// Paracrine coupling conductance coefficient.
    //constexpr static const Float e_gaba = 20.0;     //Coupling potential for GABA and paracrine signalling
    constexpr static const Float e_paracrine = 20.0;//Coupling potential for paracrine signalling
    constexpr const static Float Ar = 5.0; constexpr const static Float Ad = 0.18; // alpha and beta for GABA gating variable
    constexpr const static Float appcur = 3.5f;      // Input current
    constexpr static const Float white_noise_mean = 0.1f;
    constexpr static const Float white_noise_intensity = 0.05f; // Controls stddev of white noise.
};
