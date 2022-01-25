#pragma once
#include "defines.cuh"


//! EPhys is an abstract class that handles electrophysiological details for a certain model.
class Ephys
{
protected:
    int const N;
    thrust::device_vector<curandState_t> random_states;
public:
    //! Default constructor. This constructor sets up the random_states variable which is a collection of random seeds for curand.
    //! Most Ephys models need random seeds for noise, etc. 
    //! \param [in] N_ Number of neurons.
    Ephys(int const N_);

    //! Pure virtual function intended to simulate electrophysiology.
	/*!
	  \param [in, out] sim_vars Struct-of-arrays containing relevant simulation variables per neurons (e.g., voltages, gating vars).
	  \param [in] timestep Timestep used for simulation.
	*/
    virtual void SimulateEphys(Float** sim_vars) = 0;
};

//! Hodgkin-Huxley model (https://neuronaldynamics.epfl.ch/online/Ch2.S2.html)
class HHEphys : public Ephys
{
public:
    HHEphys(int const N_) : Ephys(N_) {}
    virtual void SimulateEphys(Float** sim_vars) override;
};

//! SAN Model (https://pubmed.ncbi.nlm.nih.gov/26996081/)
class SANEphys : public Ephys
{
public:
    SANEphys(int const N_) : Ephys(N_) {}
    virtual void SimulateEphys(Float** sim_vars) override;
};

//! Diekman-Forger model (https://journals.sagepub.com/doi/abs/10.1177/0748730409337601)
class DiekmanEphys : public Ephys
{
public:
    DiekmanEphys(int const N_) : Ephys(N_) {}
    virtual void SimulateEphys(Float** sim_vars) override;
};
