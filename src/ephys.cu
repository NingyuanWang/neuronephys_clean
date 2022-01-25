#include "ephys.cuh"

Ephys::~Ephys()
{
    for (Float*& dev_ptr : sim_vars_cpu)
        cudaFree(dev_ptr);
}

HHEphys::HHEphys(int const N_)
    : Ephys(N_, HH::leapfrog_dt)
{
    using namespace HH;
    // Set up simulation variables. Stored in a struct of arrays.
    sim_vars_cpu.resize(Params::PARAM_COUNT);
    {
        thrust::host_vector<Float> cpu_arr(N, 0);

        // Zero arrays.
        for (Float*& arr : sim_vars_cpu)
        {
            cudaMalloc(&arr, N * sizeof(Float));
            cudaMemcpy(sim_vars_cpu[Params::V], cpu_arr.data(), N * sizeof(Float), cudaMemcpyHostToDevice);
        }

        // Set voltages randomly
        auto& gen{ Utility::UnseededRandomEngine() };
        std::normal_distribution<> voltage_dist(-55, 25);
        for (auto& val : cpu_arr)
            val = voltage_dist(gen);
        thrust::fill_n(cpu_arr.begin(), N, -55);
        cudaMemcpy(sim_vars_cpu[Params::V], cpu_arr.data(), N * sizeof(Float), cudaMemcpyHostToDevice);
        thrust::fill_n(cpu_arr.begin(), N, 0.0);

        // Set gating variables  
        std::normal_distribution<> gating_var_dist(0.0, 0.0);
        cudaMemcpy(sim_vars_cpu[Params::M], cpu_arr.data(), N * sizeof(Float), cudaMemcpyHostToDevice);

        for (auto& val : cpu_arr)
            val = gating_var_dist(gen);
        cudaMemcpy(sim_vars_cpu[Params::N], cpu_arr.data(), N * sizeof(Float), cudaMemcpyHostToDevice);

        for (auto& val : cpu_arr)
            val = gating_var_dist(gen);
        cudaMemcpy(sim_vars_cpu[Params::H], cpu_arr.data(), N * sizeof(Float), cudaMemcpyHostToDevice);

        thrust::fill_n(cpu_arr.begin(), N, 0.0);
        cudaMemcpy(sim_vars_cpu[Params::LOCAL_Y], cpu_arr.data(), N * sizeof(Float), cudaMemcpyHostToDevice);
        cudaMemcpy(sim_vars_cpu[Params::IY], cpu_arr.data(), N * sizeof(Float), cudaMemcpyHostToDevice);
        cudaMemcpy(sim_vars_cpu[Params::EY], cpu_arr.data(), N * sizeof(Float), cudaMemcpyHostToDevice);

        // No neurotransmitter input/output or connections
        cudaMemcpy(sim_vars_cpu[Params::LOCAL_OUTPUT], cpu_arr.data(), N * sizeof(Float), cudaMemcpyHostToDevice);
        cudaMemcpy(sim_vars_cpu[Params::LOCAL_IN], cpu_arr.data(), N * sizeof(Float), cudaMemcpyHostToDevice);
        cudaMemcpy(sim_vars_cpu[Params::LOCAL_COUNT], cpu_arr.data(), N * sizeof(Float), cudaMemcpyHostToDevice);
        cudaMemcpy(sim_vars_cpu[Params::IOUTPUT], cpu_arr.data(), N * sizeof(Float), cudaMemcpyHostToDevice);
        cudaMemcpy(sim_vars_cpu[Params::IIN], cpu_arr.data(), N * sizeof(Float), cudaMemcpyHostToDevice);
        cudaMemcpy(sim_vars_cpu[Params::ICOUNT], cpu_arr.data(), N * sizeof(Float), cudaMemcpyHostToDevice);
        cudaMemcpy(sim_vars_cpu[Params::EOUTPUT], cpu_arr.data(), N * sizeof(Float), cudaMemcpyHostToDevice);
        cudaMemcpy(sim_vars_cpu[Params::EIN], cpu_arr.data(), N * sizeof(Float), cudaMemcpyHostToDevice);
        cudaMemcpy(sim_vars_cpu[Params::ECOUNT], cpu_arr.data(), N * sizeof(Float), cudaMemcpyHostToDevice);

        // Set applied current
        thrust::fill_n(cpu_arr.begin(), N, appcur);
        cudaMemcpy(sim_vars_cpu[Params::APPCUR], cpu_arr.data(), N * sizeof(Float), cudaMemcpyHostToDevice);
        gpuErrchk(cudaDeviceSynchronize());
    }
    // Copy pointers from CPU to GPU 
    sim_vars = sim_vars_cpu;
}

SANEphys::SANEphys(int const N_)
    : Ephys(N_, SAN::rk4_dt)
{
    // Set up simulation variables. Stored in a struct of arrays.
    using namespace SAN;
    sim_vars_cpu.resize(Params::PARAM_COUNT);
    {
        thrust::host_vector<Float> cpu_arr(N, 0);

        // Zero arrays.
        for (Float*& arr : sim_vars_cpu)
        {
            cudaMalloc(&arr, N * sizeof(Float));
            cudaMemcpy(sim_vars_cpu[Params::V], cpu_arr.data(), N * sizeof(Float), cudaMemcpyHostToDevice);
        }

        // Set voltages randomly
        auto& gen{ Utility::UnseededRandomEngine() };
        std::normal_distribution<> voltage_dist(-80, 20);
        for (auto& val : cpu_arr)
            val = voltage_dist(gen);
        cudaMemcpy(sim_vars_cpu[Params::V], cpu_arr.data(), N * sizeof(Float), cudaMemcpyHostToDevice);

        // Set Ca 
        std::uniform_real_distribution<> Ca_dist(0.1, 0.5);
        for (auto& val : cpu_arr)
            val = Ca_dist(gen);
        cudaMemcpy(sim_vars_cpu[Params::CA], cpu_arr.data(), N * sizeof(Float), cudaMemcpyHostToDevice);

        // Set potasium gating variable 
        std::uniform_real_distribution<> nK_dist(0.0, 0.5);
        for (auto& val : cpu_arr)
            val = nK_dist(gen);
        cudaMemcpy(sim_vars_cpu[Params::NK], cpu_arr.data(), N * sizeof(Float), cudaMemcpyHostToDevice);
        thrust::fill_n(cpu_arr.begin(), N, 0.0);
        // AMPA/NMDA gating variable
        cudaMemcpy(sim_vars_cpu[Params::EY], cpu_arr.data(), N * sizeof(Float), cudaMemcpyHostToDevice);

        // GABA gating variable
        cudaMemcpy(sim_vars_cpu[Params::IY], cpu_arr.data(), N * sizeof(Float), cudaMemcpyHostToDevice);

        // No neurotransmitter input/output. Connection counts are set to zero for now and edited below. 
        cudaMemcpy(sim_vars_cpu[Params::EOUTPUT], cpu_arr.data(), N * sizeof(Float), cudaMemcpyHostToDevice);
        cudaMemcpy(sim_vars_cpu[Params::EIN], cpu_arr.data(), N * sizeof(Float), cudaMemcpyHostToDevice);
        cudaMemcpy(sim_vars_cpu[Params::ECOUNT], cpu_arr.data(), N * sizeof(Float), cudaMemcpyHostToDevice);
        cudaMemcpy(sim_vars_cpu[Params::IOUTPUT], cpu_arr.data(), N * sizeof(Float), cudaMemcpyHostToDevice);
        cudaMemcpy(sim_vars_cpu[Params::IIN], cpu_arr.data(), N * sizeof(Float), cudaMemcpyHostToDevice);
        cudaMemcpy(sim_vars_cpu[Params::ICOUNT], cpu_arr.data(), N * sizeof(Float), cudaMemcpyHostToDevice);
        gpuErrchk(cudaDeviceSynchronize());
    }
    // Copy pointers from CPU to GPU 
    sim_vars = sim_vars_cpu;
}

__global__ void init_random(unsigned int seed, int const N, curandState_t* states) 
{
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < N; idx += gridDim.x * blockDim.x)
    {
        curand_init(seed, idx, 0, &states[idx]);
    }
}

__global__ void get_random_test(curandState_t* states, int const N, int* numbers)
{
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < N; idx += gridDim.x * blockDim.x)
    {
        numbers[idx] = curand(&states[idx]) % 100;
    }
}

Ephys::Ephys(int const N_, Float const dt_) : N{ N_ }, dt{ dt_ }, random_states(N_)
{
    // Setup random seeds. 
    init_random<<<NBLOCKS, NTHREADS>>>(0, N, thrust::raw_pointer_cast(random_states.data()));
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaDeviceSynchronize() );
}

__global__ void Leapfrog_HH(Float** sim_vars, const int N, const Float timestep, curandState_t* random_states)
{
    using namespace HH;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < N; idx += gridDim.x * blockDim.x)
    {
        Float* V_ptr = sim_vars[Params::V] + idx;
        Float* M_ptr = sim_vars[Params::M] + idx;
        Float* N_ptr = sim_vars[Params::N] + idx;
        Float* H_ptr = sim_vars[Params::H] + idx;
        Float* local_Y_ptr = sim_vars[Params::LOCAL_Y] + idx;
        Float* IY_ptr = sim_vars[Params::IY] + idx;
        Float* EY_ptr = sim_vars[Params::EY] + idx;

        const Float local_in = sim_vars[Params::LOCAL_IN][idx];
        const Float local_num_in = sim_vars[Params::LOCAL_COUNT][idx];
        const Float Iin = sim_vars[Params::IIN][idx];
        const Float Inum_in = sim_vars[Params::ICOUNT][idx];
        const Float Ein = sim_vars[Params::EIN][idx];
        const Float Enum_in = sim_vars[Params::ECOUNT][idx];
        const Float appcur = sim_vars[Params::APPCUR][idx];

        const Float V = *V_ptr;
        const Float M = *M_ptr;
        const Float N = *N_ptr;
        const Float H = *H_ptr;
        const Float local_Y = *local_Y_ptr;
        const Float IY = *IY_ptr;
        const Float EY = *EY_ptr;

        const Float Am = (3.5 + 0.1 * V) / (1 - exp(-3.5 - 0.1*V));
        const Float An = (-0.5 - 0.01 * V) / (exp(-5 - 0.1 * V) - 1);
        const Float Ah = 0.07 * exp(-V / 20 - 3);

        const Float Bm = 4 * exp(-(V + 60) / 18);
        const Float Bn = 0.125*exp(-(V + 60) / 80);
        const Float Bh = 1 / (exp(-3 - 0.1*V) + 1);

        const Float new_M = (Am * timestep + (1 - timestep / 2 * (Am + Bm)) * M) / (timestep / 2 * (Am + Bm) + 1);
        const Float new_N = (An * timestep + (1 - timestep / 2 * (An + Bn)) * N) / (timestep / 2 * (An + Bn) + 1);
        const Float new_H = (Ah * timestep + (1 - timestep / 2 * (Ah + Bh)) * H) / (timestep / 2 * (Ah + Bh) + 1);

        const Float local_Tin_avg = local_num_in > 0 ? local_in / local_num_in : 0;
        const Float new_local_Y = (Ar * local_Tin_avg * timestep + (1 - timestep / 2 * (Ar * local_Tin_avg + Ad)) * local_Y) / (timestep / 2 * (Ar * local_Tin_avg + Ad) + 1);

        const Float Iin_avg = Inum_in > 0 ? Iin / Inum_in : 0;
        const Float new_IY = (Ar * Iin_avg * timestep + (1 - timestep / 2 * (Ar * Iin_avg + Ad)) * IY) / (timestep / 2 * (Ar * Iin_avg + Ad) + 1);

        const Float Ein_avg = Enum_in > 0 ? Ein / Enum_in : 0;
        const Float new_EY = (Ar * Ein_avg * timestep + (1 - timestep / 2 * (Ar * Ein_avg + Ad)) * EY) / (timestep / 2 * (Ar * Ein_avg + Ad) + 1);

        const Float G = gna * new_M*new_M*new_M*new_H + gk * (powf(new_N, 4)) + gl + g_ampa_grey * new_local_Y + g_gaba * new_IY + g_ampa_white * new_EY;
        const Float E = gna * new_M*new_M*new_M*new_H * ena + gk * (powf(new_N, 4)) * ek + gl * el + g_ampa_grey * new_local_Y * e_ampa + g_gaba * new_IY * e_gaba + g_ampa_white * new_EY * e_ampa;

        // Add random white noise to V.
        Float white_noise = white_noise_intensity * curand_normal(random_states + idx) + white_noise_mean;

        // Copy back to sim_vars
        *V_ptr = ((white_noise + appcur) * timestep + E * timestep + (1 - timestep / 2 * G)*V) / (1 + timestep / 2 * G);
        *M_ptr = new_M;
        *N_ptr = new_N;
        *H_ptr = new_H;
        *local_Y_ptr = new_local_Y;
        *IY_ptr = new_IY;
        *EY_ptr = new_EY;

        // Calculate amount of neurotransmitter outputted AFTER update
        sim_vars[Params::LOCAL_OUTPUT][idx] = sim_vars[Params::IOUTPUT][idx] = sim_vars[Params::EOUTPUT][idx] = Tmax / (1.0 + exp(-(*V_ptr - Vt) / Kp));
    }
}

void HHEphys::SimulateEphys()
{
    Leapfrog_HH<<<NBLOCKS, NTHREADS>>>(thrust::raw_pointer_cast(sim_vars.data()), N, dt, thrust::raw_pointer_cast(random_states.data()));
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
}

__device__ void dxdt_an_ie_model(const int idx,
    Float const V, Float const nK, Float const Ca, Float const Ye, Float const Yi, Float const Y_loc,
    Float const e_Tinavg, Float const i_Tinavg, Float const loc_Tinavg,
    Float const white_noise,
    Float* dvdt, Float* dnKdt, Float* dCadt, Float* dYedt, Float* dYidt, Float* dY_loc_dt)
{
    using namespace SAN;
    // Calculate intermediate quantities
    Float mCa_inf = 1 / (1 + exp(-(V + 20) / 9));
    Float mKCa_inf = 1 / (1 + powf(kD / Ca, 3.5));
    Float mNaP_inf = 1 / (1 + exp(-(V + 55.7) / 7.7));

    Float alpha_n = V == -34 ? 0.1 : 0.01 * (V + 34) / (1 - exp(-(V + 34) / 10));
    Float beta_n = 0.125 * exp(-(V + 44) / 25);

    // Calculate gating variable derivatives
    *dnKdt = 4 * (alpha_n * (1 - nK) - beta_n * nK);
    *dYedt = 3.48 * e_Tinavg - Ye / tauAMPA_white;
    *dY_loc_dt = 3.48 * loc_Tinavg - Y_loc / tauAMPA_grey;
    *dYidt = i_Tinavg - Yi / tauGABA;

    // Calculate Ca2+ derivative 
    Float ICa = gCa * mCa_inf * mCa_inf * (V - vCa);
    *dCadt = -alphaCa * (10 * A * ICa) - Ca / tauCa;

    // Calculate voltage derivative
    *dvdt = 
        white_noise
        - gK * powf(nK, 4) * (V - vK)
        - ICa
        - gKCa * mKCa_inf * (V - vK)
        - gNaP * mNaP_inf * mNaP_inf * mNaP_inf * (V - vNa)
        - gL * (V - vL)
        - gAMPA_white * Ye * (V - vAMPA)
        - gAMPA_grey * Y_loc * (V - vGABA)
        - gGABA * Yi * (V - vGABA);

    *dvdt /= C;
}

__global__ void RK4_an_ie_model(Float** sim_vars, const int N, const float timestep, curandState_t* random_states)
{
    using namespace SAN;
    Float dvdt1, dnKdt1, dCadt1, dYedt1, dYidt1, dY_loc_dt1;
    Float dvdt2, dnKdt2, dCadt2, dYedt2, dYidt2, dY_loc_dt2;
    Float dvdt3, dnKdt3, dCadt3, dYedt3, dYidt3, dY_loc_dt3;
    Float dvdt4, dnKdt4, dCadt4, dYedt4, dYidt4, dY_loc_dt4;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < N; idx += gridDim.x * blockDim.x)
    {
        Float* V_ptr = sim_vars[Params::V] + idx;
        Float* nK_ptr = sim_vars[Params::NK] + idx;
        Float* Ca_ptr = sim_vars[Params::CA] + idx;
        Float* Ye_ptr = sim_vars[Params::EY] + idx;
        Float* Yi_ptr = sim_vars[Params::IY] + idx;
        Float* Y_loc_ptr = sim_vars[Params::LOCAL_Y] + idx;

        // Calculate average neurotransmitter input
        Float e_Tinavg = sim_vars[Params::ECOUNT][idx];
        e_Tinavg = e_Tinavg == 0 ? 0 : sim_vars[Params::EIN][idx] / e_Tinavg;
        Float loc_Tinavg = sim_vars[Params::LOCAL_COUNT][idx];
        loc_Tinavg = loc_Tinavg == 0 ? 0 : sim_vars[Params::LOCAL_IN][idx] / loc_Tinavg;
        Float i_Tinavg = sim_vars[Params::ICOUNT][idx];
        i_Tinavg = i_Tinavg == 0 ? 0 : sim_vars[Params::IIN][idx] / i_Tinavg;

        Float const V = *V_ptr;
        Float const nK = *nK_ptr;
        Float const Ca = *Ca_ptr;
        Float const Ye = *Ye_ptr;
        Float const Yi = *Yi_ptr;
        Float const Y_loc = *Y_loc_ptr;

        // Add random white noise to V.
        Float white_noise = white_noise_intensity * curand_normal(random_states + idx) + white_noise_mean;

        // k1 = f(X)
        dxdt_an_ie_model(idx, V, nK, Ca, Ye, Yi, Y_loc, e_Tinavg, i_Tinavg, loc_Tinavg, white_noise, &dvdt1, &dnKdt1, &dCadt1, &dYedt1, &dYidt1, &dY_loc_dt1);

        // k2 = f(X + dt/2 * k1)
        dxdt_an_ie_model(idx,
            V + dvdt1 * timestep / 2, nK + dnKdt1 * timestep / 2, Ca + dCadt1 * timestep / 2, Ye + dYedt1 * timestep / 2, Yi + dYidt1 * timestep / 2, Y_loc + dY_loc_dt1 * timestep / 2, white_noise, 
            e_Tinavg, i_Tinavg, loc_Tinavg, &dvdt2, &dnKdt2, &dCadt2, &dYedt2, &dYidt2, &dY_loc_dt2);

        // k3 = f(X + dt/2 * k2)
        dxdt_an_ie_model(idx,
            V + dvdt2 * timestep / 2, nK + dnKdt2 * timestep / 2, Ca + dCadt2 * timestep / 2, Ye + dYedt2 * timestep / 2, Yi + dYidt2 * timestep / 2, Y_loc + dY_loc_dt2 * timestep / 2, white_noise, 
            e_Tinavg, i_Tinavg, loc_Tinavg, &dvdt3, &dnKdt3, &dCadt3, &dYedt3, &dYidt3, &dY_loc_dt3);

        // k4 = f(X + dt * k3)
        dxdt_an_ie_model(idx,
            V + dvdt3 * timestep, nK + dnKdt3 * timestep, Ca + dCadt3 * timestep, Ye + dYedt3 * timestep, Yi + dYidt3 * timestep, Y_loc + dY_loc_dt3 * timestep, white_noise, 
            e_Tinavg, i_Tinavg, loc_Tinavg, &dvdt4, &dnKdt4, &dCadt4, &dYedt4, &dYidt4, &dY_loc_dt4);

        // X(t + dt) = X(t) + dt * (k1 + 2k2 + 2k3 + k4)/6
        *V_ptr = V + timestep * (dvdt1 + dvdt2 * 2 + dvdt3 * 2 + dvdt4) / 6;
        *nK_ptr = nK + timestep * (dnKdt1 + dnKdt2 * 2 + dnKdt3 * 2 + dnKdt4) / 6;
        *Ca_ptr = Ca + timestep * (dCadt1 + dCadt2 * 2 + dCadt3 * 2 + dCadt4) / 6;
        *Ye_ptr = Ye + timestep * (dYedt1 + dYedt2 * 2 + dYedt3 * 2 + dYedt4) / 6;
        *Yi_ptr = Yi + timestep * (dYidt1 + dYidt2 * 2 + dYidt3 * 2 + dYidt4) / 6;
        *Y_loc_ptr = Y_loc + timestep * (dY_loc_dt1 + dY_loc_dt2 * 2 + dY_loc_dt3 * 2 + dY_loc_dt4) / 6;

        // Calculate amount of neurotransmitter outputted AFTER update
        sim_vars[Params::IOUTPUT][idx] = sim_vars[Params::EOUTPUT][idx] = Tmax / (1.0 + exp(-(*V_ptr - an_Vt) / an_Kp));
    }
}

__global__ void Euler_an_ie_model(Float** sim_vars, const int N, const float timestep, curandState_t* random_states)
{
    using namespace SAN;
    Float dvdt, dnKdt, dCadt, dYedt, dYidt, dY_loc_dt;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < N; idx += gridDim.x * blockDim.x)
    {
        Float* V_ptr = sim_vars[Params::V] + idx;
        Float* nK_ptr = sim_vars[Params::NK] + idx;
        Float* Ca_ptr = sim_vars[Params::CA] + idx;
        Float* Ye_ptr = sim_vars[Params::EY] + idx;
        Float* Yi_ptr = sim_vars[Params::IY] + idx;
        Float* Y_loc_ptr = sim_vars[Params::LOCAL_Y] + idx;

        // Calculate average neurotransmitter input
        Float e_Tinavg = sim_vars[Params::ECOUNT][idx];
        e_Tinavg = e_Tinavg == 0 ? 0 : sim_vars[Params::EIN][idx] / e_Tinavg;
        Float loc_Tinavg = sim_vars[Params::LOCAL_COUNT][idx];
        loc_Tinavg = loc_Tinavg == 0 ? 0 : sim_vars[Params::LOCAL_IN][idx] / loc_Tinavg;
        Float i_Tinavg = sim_vars[Params::ICOUNT][idx];
        i_Tinavg = i_Tinavg == 0 ? 0 : sim_vars[Params::IIN][idx] / i_Tinavg;

        Float const V = *V_ptr;
        Float const nK = *nK_ptr;
        Float const Ca = *Ca_ptr;
        Float const Ye = *Ye_ptr;
        Float const Yi = *Yi_ptr;
        Float const Y_loc = *Y_loc_ptr;

        // Add random white noise to V.
        Float white_noise = white_noise_intensity * curand_normal(random_states + idx) + white_noise_mean;

        dxdt_an_ie_model(idx, V, nK, Ca, Ye, Yi, Y_loc, e_Tinavg, i_Tinavg, loc_Tinavg, white_noise, &dvdt, &dnKdt, &dCadt, &dYedt, &dYidt, &dY_loc_dt);
        
        // X(t + dt) = X(t) + dt * dxdt
        *V_ptr = V + timestep * dvdt;
        *nK_ptr = nK + timestep * dnKdt;
        *Ca_ptr = Ca + timestep * dCadt;
        *Ye_ptr = Ye + timestep * dYedt;
        *Yi_ptr = Yi + timestep * dYidt;
        *Y_loc_ptr = Y_loc + timestep * dY_loc_dt;

        // Calculate amount of neurotransmitter outputted AFTER update
        sim_vars[Params::IOUTPUT][idx] = sim_vars[Params::EOUTPUT][idx] = Tmax / (1.0 + exp(-(*V_ptr - an_Vt) / an_Kp));
    }
}

void SANEphys::SimulateEphys()
{
    Euler_an_ie_model<<<NBLOCKS, NTHREADS>>>(thrust::raw_pointer_cast(sim_vars.data()), N, dt, thrust::raw_pointer_cast(random_states.data()));
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
}
