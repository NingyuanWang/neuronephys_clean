#pragma once

#include "network_factory.cuh"
#include "morphology.h"
#include "electrode_net.cuh"
#include "ephys.cuh"

//! Mouse cortex model including morphology, connectivity, and underlying neuron model.
class MouseCortex
{
public:
    //! Ncouple value to use for uniform random connectivity. Default is 1000.
    static int uniform_random_Ncouple;
    static std::string excitatory_conns_file_name;
    static std::string inhibitory_conns_file_name;
    static bool use_weights_network;

    //! Pointer to brain morphology.
    std::unique_ptr<FileMorphology> morphology;
    //! Pointer to brain electrophysiology class holding all the data and relevant methods.
    std::unique_ptr<Ephys> ephys;

    std::unique_ptr<AtomicCoupling> white_ampa_connectivity; //! Long range AMPA connections.
    std::unique_ptr<AtomicCoupling> white_gaba_connectivity; //! Long range GABA connections.
    std::unique_ptr<AtomicCoupling> gray_gaba_connectivity; //! Short range GABA connections.

    //! Construct the mouse brain model.
    //! \param[in] N Number of neurons to use for simulations. These will be sampled uniformly from the full population of 13 million neurons.
    //! \param[in] neuron_model One of HH or SAN. Default value is HH.
    //! \param[in] uniform_random_white If this is true, uniform random connections are generated for the white matter. If not, it is read from the realistic connectivity. Default value false.
    //! \param[in] use_gray If this is true, localized gray matter connections are used. If false, no gray matter is used. Default value true.
    MouseCortex(int N, std::string neuron_model = "HH", bool uniform_random_white=false, bool use_gray=true);

    //! Simulate the whole cortex! This consists of first handling synaptic coupling then numerically integrating each individual neuron.
    //! \return The number of neurons firing.
    int Simulate();
};
