#include "coupling.cuh"

class Hebbian
{
private:
    int Nconns;
    thrust::device_vector<int> timers;
public:
    int const window_duration;
    Float const heb_inc;

    //! Construct Hebbian learner.
    //! \param [in] window_duration_ The window size for Hebbian learning rule in dt timestep units.
    //! \param [in] heb_inc_ Hebbian rule increment.
    //! \param [in] Nconns_ Number of synapses in the whole network.
    Hebbian(int window_duration_, Float heb_inc_, int Nconns_)
        : Nconns{ Nconns_ }, window_duration{ window_duration_ }, heb_inc{ heb_inc_ }
    {
        timers = thrust::device_vector<int>(Nconns_, 0);
    }

    //! Do Hebbian learning at a certain instant in time.
    //! \param [in, out] connectivity The synaptic connectivity stored as an adjacency list of downstream connections, along with upstream indices and weights.
    //! \param [in] is_firing Vector where element i is 1/0 if neuron i is/isn't firing.
    //! \param [in] firing_inds Vector of indices of firing neurons.
    void UpdateWeights(AdjacencyList& connectivity, thrust::device_vector<int> const& is_firing, thrust::device_vector<int> const& firing_inds);
};