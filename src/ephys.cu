#include "ephys.cuh"
#include "circadian.h"

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

Ephys::Ephys(int const N_) : N{ N_ }, random_states(N_)
{
    // Setup random seeds. 
    init_random<<<NBLOCKS, NTHREADS>>>(0, N, thrust::raw_pointer_cast(random_states.data()));
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaDeviceSynchronize() );
}

__global__ void Leapfrog_HH(Float** sim_vars, const int N, const Float timestep, curandState_t* random_states)
{
    using namespace hh_params;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < N; idx += gridDim.x * blockDim.x)
    {
        Float* V_ptr = sim_vars[SIM_PARAMS_V] + idx;
        Float* M_ptr = sim_vars[SIM_PARAMS_M] + idx;
        Float* N_ptr = sim_vars[SIM_PARAMS_N] + idx;
        Float* H_ptr = sim_vars[SIM_PARAMS_H] + idx;
        Float* local_Y_ptr = sim_vars[SIM_PARAMS_LOCAL_Y] + idx;
        Float* IY_ptr = sim_vars[SIM_PARAMS_IY] + idx;
        Float* EY_ptr = sim_vars[SIM_PARAMS_EY] + idx;

        const Float local_in = sim_vars[SIM_PARAMS_LOCAL_IN][idx];
        const Float local_num_in = sim_vars[SIM_PARAMS_LOCAL_COUNT][idx];
        const Float Iin = sim_vars[SIM_PARAMS_IIN][idx];
        const Float Inum_in = sim_vars[SIM_PARAMS_ICOUNT][idx];
        const Float Ein = sim_vars[SIM_PARAMS_EIN][idx];
        const Float Enum_in = sim_vars[SIM_PARAMS_ECOUNT][idx];
        const Float appcur = sim_vars[SIM_PARAMS_APPCUR][idx];

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

        const Float G = gna * new_M*new_M*new_M*new_H + gk * (powf(new_N, 4)) + gl + g_ampa * new_local_Y + g_gaba * new_IY + g_ampa * new_EY;
        const Float E = gna * new_M*new_M*new_M*new_H * ena + gk * (powf(new_N, 4)) * ek + gl * el + g_ampa * new_local_Y * e_ampa + g_gaba * new_IY * e_gaba + g_ampa * new_EY * e_ampa;

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
        sim_vars[SIM_PARAMS_LOCAL_OUTPUT][idx] = sim_vars[SIM_PARAMS_IOUTPUT][idx] = sim_vars[SIM_PARAMS_EOUTPUT][idx] = Tmax / (1.0 + exp(-(*V_ptr - Vt) / Kp));
    }
}

void HHEphys::SimulateEphys(Float** sim_vars)
{
    Leapfrog_HH<<<NBLOCKS, NTHREADS>>>(sim_vars, N, hh_params::leapfrog_dt, thrust::raw_pointer_cast(random_states.data()));
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
}

__device__ void dxdt_an_ie_model(const int idx,
    Float const V, Float const nK, Float const Ca, Float const Ye, Float const Yi,
    Float const e_Tinavg, Float const i_Tinavg,//e/i_Tinavg: excitatory/inhibitory neurotransmitter input
    Float const white_noise,
    Float* dvdt, Float* dnKdt, Float* dCadt, Float* dYedt, Float* dYidt)
{
    using namespace an_ie_params;
    // Calculate intermediate quantities
    Float mCa_inf = 1 / (1 + exp(-(V + 20) / 9));
    Float mKCa_inf = 1 / (1 + powf(kD / Ca, 3.5));
    Float mNaP_inf = 1 / (1 + exp(-(V + 55.7) / 7.7));

    Float alpha_n = V == -34 ? 0.1 : 0.01 * (V + 34) / (1 - exp(-(V + 34) / 10));
    Float beta_n = 0.125 * exp(-(V + 44) / 25);

    // Calculate gating variable derivatives
    *dnKdt = 4 * (alpha_n * (1 - nK) - beta_n * nK);
    *dYedt = 3.48 * e_Tinavg - Ye / tauAMPA;
    *dYidt = i_Tinavg - Yi / tauGABA;

    // Calculate Ca2+ derivative 
    Float ICa = gCa * mCa_inf * mCa_inf * (V - vCa);
    *dCadt = -alphaCa * (10 * A * ICa) - Ca / tauCa;

    // Calculate voltage derivative
    *dvdt = 
        white_noise
        -gK * powf(nK, 4) * (V - vK)
        - ICa
        - gKCa * mKCa_inf * (V - vK)
        - gNaP * mNaP_inf * mNaP_inf * mNaP_inf * (V - vNa)
        - gL * (V - vL)
        - an_ie_params::gAMPA * Ye * (V - an_ie_params::vAMPA)//TODO: Check value
        - an_ie_params::gGABA * Yi * (V - an_ie_params::vGABA);//TODO: Set right value

    *dvdt /= C;
}

__global__ void RK4_an_ie_model(Float** sim_vars, const int N, const float timestep, curandState_t* random_states)
{
    using namespace an_ie_params;
    Float dvdt1, dnKdt1, dCadt1, dYedt1, dYidt1;
    Float dvdt2, dnKdt2, dCadt2, dYedt2, dYidt2;
    Float dvdt3, dnKdt3, dCadt3, dYedt3, dYidt3;
    Float dvdt4, dnKdt4, dCadt4, dYedt4, dYidt4;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < N; idx += gridDim.x * blockDim.x)
    {
        Float* V_ptr = sim_vars[AN_IE_VAR_V] + idx;
        Float* nK_ptr = sim_vars[AN_IE_VAR_NK] + idx;
        Float* Ca_ptr = sim_vars[AN_IE_VAR_CA] + idx;
        Float* Ye_ptr = sim_vars[AN_IE_VAR_YE] + idx;
        Float* Yi_ptr = sim_vars[AN_IE_VAR_YI] + idx;

        // Calculate average neurotransmitter input
        Float e_Tinavg = sim_vars[AN_IE_VAR_ECOUNT][idx];
        e_Tinavg = e_Tinavg == 0 ? 0 : sim_vars[AN_IE_VAR_EIN][idx] / e_Tinavg;
        Float i_Tinavg = sim_vars[AN_IE_VAR_ICOUNT][idx];
        i_Tinavg = i_Tinavg == 0 ? 0 : sim_vars[AN_IE_VAR_IIN][idx] / i_Tinavg;

        Float const V = *V_ptr;
        Float const nK = *nK_ptr;
        Float const Ca = *Ca_ptr;
        Float const Ye = *Ye_ptr;
        Float const Yi = *Yi_ptr;

        // Add random white noise to V.
        Float white_noise = white_noise_intensity * curand_normal(random_states + idx) + white_noise_mean;

        // k1 = f(X)
        dxdt_an_ie_model(idx, V, nK, Ca, Ye, Yi, e_Tinavg, i_Tinavg, white_noise, &dvdt1, &dnKdt1, &dCadt1, &dYedt1, &dYidt1);

        // k2 = f(X + dt/2 * k1)
        dxdt_an_ie_model(idx,
            V + dvdt1 * timestep / 2, nK + dnKdt1 * timestep / 2, Ca + dCadt1 * timestep / 2, Ye + dYedt1 * timestep / 2, Yi + dYidt1 * timestep / 2, white_noise, 
            e_Tinavg, i_Tinavg, &dvdt2, &dnKdt2, &dCadt2, &dYedt2, &dYidt2);

        // k3 = f(X + dt/2 * k2)
        dxdt_an_ie_model(idx,
            V + dvdt2 * timestep / 2, nK + dnKdt2 * timestep / 2, Ca + dCadt2 * timestep / 2, Ye + dYedt2 * timestep / 2, Yi + dYidt2 * timestep / 2,  white_noise, 
            e_Tinavg, i_Tinavg, &dvdt3, &dnKdt3, &dCadt3, &dYedt3, &dYidt3);

        // k4 = f(X + dt * k3)
        dxdt_an_ie_model(idx,
            V + dvdt3 * timestep, nK + dnKdt3 * timestep, Ca + dCadt3 * timestep, Ye + dYedt3 * timestep, Yi + dYidt3 * timestep / 2, white_noise, 
            e_Tinavg, i_Tinavg, &dvdt4, &dnKdt4, &dCadt4, &dYedt4, &dYidt4);

        // X(t + dt) = X(t) + dt * (k1 + 2k2 + 2k3 + k4)/6
        *V_ptr = V + timestep * (dvdt1 + dvdt2 * 2 + dvdt3 * 2 + dvdt4) / 6;
        *nK_ptr = nK + timestep * (dnKdt1 + dnKdt2 * 2 + dnKdt3 * 2 + dnKdt4) / 6;
        *Ca_ptr = Ca + timestep * (dCadt1 + dCadt2 * 2 + dCadt3 * 2 + dCadt4) / 6;
        *Ye_ptr = Ye + timestep * (dYedt1 + dYedt2 * 2 + dYedt3 * 2 + dYedt4) / 6;
        *Yi_ptr = Yi + timestep * (dYidt1 + dYidt2 * 2 + dYidt3 * 2 + dYidt4) / 6;

        // Calculate amount of neurotransmitter outputted AFTER update
        sim_vars[AN_IE_VAR_IOUTPUT][idx] = sim_vars[AN_IE_VAR_EOUTPUT][idx] = Tmax / (1.0 + exp(-(*V_ptr - an_Vt) / an_Kp));
    }
}

__global__ void Euler_an_ie_model(Float** sim_vars, const int N, const float timestep, curandState_t* random_states)
{
    using namespace an_ie_params;
    Float dvdt, dnKdt, dCadt, dYedt, dYidt;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < N; idx += gridDim.x * blockDim.x)
    {
        Float* V_ptr = sim_vars[AN_IE_VAR_V] + idx;
        Float* nK_ptr = sim_vars[AN_IE_VAR_NK] + idx;
        Float* Ca_ptr = sim_vars[AN_IE_VAR_CA] + idx;
        Float* Ye_ptr = sim_vars[AN_IE_VAR_YE] + idx;
        Float* Yi_ptr = sim_vars[AN_IE_VAR_YI] + idx;

        // Calculate average neurotransmitter input
        Float e_Tinavg = sim_vars[AN_IE_VAR_ECOUNT][idx];
        e_Tinavg = e_Tinavg == 0 ? 0 : sim_vars[AN_IE_VAR_EIN][idx] / e_Tinavg;
        Float i_Tinavg = sim_vars[AN_IE_VAR_ICOUNT][idx];
        i_Tinavg = i_Tinavg == 0 ? 0 : sim_vars[AN_IE_VAR_IIN][idx] / i_Tinavg;

        Float const V = *V_ptr;
        Float const nK = *nK_ptr;
        Float const Ca = *Ca_ptr;
        Float const Ye = *Ye_ptr;
        Float const Yi = *Yi_ptr;

        // Add random white noise to V.
        Float white_noise = white_noise_intensity * curand_normal(random_states + idx) + white_noise_mean;

        dxdt_an_ie_model(idx, V, nK, Ca, Ye, Yi, e_Tinavg, i_Tinavg, white_noise, &dvdt, &dnKdt, &dCadt, &dYedt, &dYidt);
        
        // X(t + dt) = X(t) + dt * dxdt
        *V_ptr = V + timestep * dvdt;
        *nK_ptr = nK + timestep * dnKdt;
        *Ca_ptr = Ca + timestep * dCadt;
        *Ye_ptr = Ye + timestep * dYedt;
        *Yi_ptr = Yi + timestep * dYidt;

        // Calculate amount of neurotransmitter outputted AFTER update
        sim_vars[AN_IE_VAR_IOUTPUT][idx] = sim_vars[AN_IE_VAR_EOUTPUT][idx] = Tmax / (1.0 + exp(-(*V_ptr - an_Vt) / an_Kp));
    }
}

void SANEphys::SimulateEphys(Float** sim_vars)
{
    Euler_an_ie_model<<<NBLOCKS, NTHREADS>>>(sim_vars, N, hh_params::leapfrog_dt, thrust::raw_pointer_cast(random_states.data()));
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
}

__global__ void Leapfrog_diekman(Float** sim_vars, const int N, const float timestep, curandState_t* random_states)
{
    using namespace diekman_params;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < N; idx += gridDim.x * blockDim.x)
    {
        const Float V = sim_vars[DIEKMAN_VAR_V][idx];
        const Float M = sim_vars[DIEKMAN_VAR_M][idx];
        const Float H = sim_vars[DIEKMAN_VAR_H][idx];
        const Float N = sim_vars[DIEKMAN_VAR_N][idx];
        const Float Rl = sim_vars[DIEKMAN_VAR_RL][idx];
        const Float Rnl = sim_vars[DIEKMAN_VAR_RNL][idx];
        const Float Fnl = sim_vars[DIEKMAN_VAR_FNL][idx];
        const Float S = sim_vars[DIEKMAN_VAR_S][idx];
        const Float P = sim_vars[DIEKMAN_VAR_P][idx];
        const Float Y = sim_vars[DIEKMAN_VAR_Y][idx];
        const Float Cas = sim_vars[DIEKMAN_VAR_CAS][idx];
        const Float Cac = sim_vars[DIEKMAN_VAR_CAC][idx];
        const Float G_biochem = sim_vars[DIEKMAN_VAR_G_BIOCHEM][idx];
        const Float Tin = sim_vars[DIEKMAN_VAR_IIN][idx];
        const Float num_in = sim_vars[DIEKMAN_VAR_ICOUNT][idx];
        const Float egaba = sim_vars[DIEKMAN_VAR_EGABA][idx];
        const Float gkca = sim_vars[DIEKMAN_VAR_GKCA][idx];
        const Float gkleak = sim_vars[DIEKMAN_VAR_GKLEAK][idx];
   
        //const Float R = clk * 11.36 * (G_biochem - 0.25); //changed in ~2/2014? = R2
        const Float R = 3.1;
        const Float gnaP = 2.85;

        const Float SS = 21 * (gto - 1.66); 
        
        //const Float gkca = 198.0 / (1.0 + exp(R)) + 2.0;
        //const Float gkleak = 0.2 / (1.0 + exp(R));
        //const Float gnap = 2.13 + 0.14 / (1.0 + exp(-SS)); // if dki
        //const Float gnap = 1.46 + 0.51 / (1.0 + exp(-SS)); // if chir
        //const Float gnap = 1; // if riluzole
        //const Float gnaP = 1.59 + 0.5 / (1.0 + exp(-SS)); // if nothing
        //const Float gnaP = 0;


        const Float M_inf = 1.0 / (1.0 + expf(-(V + 35.2) / 8.1));
        const Float H_inf = 1.0 / (1.0 + expf((V + 62.0) / 2.0));
        const Float N_inf = 1.0 / powf(1.0 + expf((V - 14.0) / (-17.0)), 0.25);
        const Float Rl_inf = 1.0 / (1.0 + expf(-(V + 36.0) / 5.1));
        const Float Rnl_inf = 1.0 / (1.0 + expf(-(V + 21.6) / 6.7));
        const Float Fnl_inf = 1.0 / (1.0 + exp((V + 260.0) / 65.0));
        const Float S_inf = 1e7 * Cas * Cas / (1e7 * Cas * Cas + 5.6);
        const Float P_inf = 1.0 / powf(1.0 + expf(-(V + 25.0) / 7.4), 1.5);

        const Float tau_M = expf(-(V + 286.0) / 160.0);
        const Float tau_H = 0.51 + expf(-(V + 26.6) / 7.1);
        const Float tau_N = expf(-(V - 67.0) / 68.0);
        const Float tau_Rl = 3.1;
        const Float tau_Rnl = 3.1;
       
        const Float tau_Fnl = expf(-(V - 444.0) / 220.0);
        const Float tau_S = 500.0 / (1e7 * Cas * Cas + 5.6);
        const Float tau_P = 100;

        // Update gating variables except S and Y.
        const Float new_M = 2.0 * timestep / (2.0 * tau_M + timestep) * M_inf + (2.0 * tau_M - timestep) / (2.0 * tau_M + timestep) * M;
        const Float new_H = 2.0 * timestep / (2.0 * tau_H + timestep) * H_inf + (2.0 * tau_H - timestep) / (2.0 * tau_H + timestep) * H;
        const Float new_N = 2.0 * timestep / (2.0 * tau_N + timestep) * N_inf + (2.0 * tau_N - timestep) / (2.0 * tau_N + timestep) * N;
        const Float new_Rl = 2.0 * timestep / (2.0 * tau_Rl + timestep) * Rl_inf + (2.0 * tau_Rl - timestep) / (2.0 * tau_Rl + timestep) * Rl;
        const Float new_Rnl = 2.0 * timestep / (2.0 * tau_Rnl + timestep) * Rnl_inf + (2.0 * tau_Rnl - timestep) / (2.0 * tau_Rnl + timestep) * Rnl;
        const Float new_Fnl = 2.0 * timestep / (2.0 * tau_Fnl + timestep) * Fnl_inf + (2.0 * tau_Fnl - timestep) / (2.0 * tau_Fnl + timestep) * Fnl;
        const Float new_P = 2.0 * timestep / (2.0 * tau_P + timestep) * P_inf + (2.0 * tau_P - timestep) / (2.0 * tau_P + timestep) * P;
        // Solve for Cas using quadratic formula then update S gating variable.
        const Float expr = -kcas * (gcal * Rl * K1 / (K2 + Cas) + gcanl * Rnl * Fnl) * (V - Eca) - Cas / taucas + bcas;
        const Float B = (K2 - Cas - timestep / 2 * expr + timestep / 2 * kcas * gcanl * new_Rnl * new_Fnl * (V - Eca) + timestep / 2 / taucas * K2 - bcas * timestep / 2) / (1 + timestep / 2 / taucas);
        const Float C = (-K2 * Cas - K2 * timestep / 2 * expr + timestep / 2 * kcas * gcal * new_Rl * K1 * (V - Eca) + timestep / 2 * kcas * gcanl * new_Rnl * new_Fnl * (V - Eca) * K2 - bcas * timestep / 2 * K2) / (1 + timestep / 2 / taucas);
        const Float new_Cas = (sqrtf(B * B - 4 * C) - B) / 2;

        // Update S.
        const Float S_inf2 = 1e7 * new_Cas * new_Cas / (1e7 * new_Cas * new_Cas + 5.6);
        const Float tau_S2 = 500.0 / (1e7 * new_Cas * new_Cas + 5.6);
        const Float new_S = 1.0 / (1.0 + timestep / (2.0 * tau_S2)) * (S * (1.0 - timestep / (2.0 * tau_S)) + timestep / 2.0 * (S_inf / tau_S + S_inf2 / tau_S2));

        // Update Cac.
        const Float new_Cac = 1.0 / (1.0 + timestep / (2.0 * taucac)) * (Cac * (1.0 - timestep / (2.0 * taucac)) + bcac * timestep - timestep * kcac / 2.0 * (gcal * new_Rl * (K1 / (K2 + Cac)) + gcanl * new_Rl * new_Fnl + gcal * Rl * (K1 / (K2 + Cas)) + gcanl * Rnl * Fnl) * (V - Eca));

        // Update the GABA gating variable.
        const Float Tin_avg = num_in == 0 ? 0.0 : Tin / num_in;
        const Float new_Y = (Ar * Tin_avg * timestep + (1 - timestep / 2 * (Ar * Tin_avg + Ad)) * Y) / (timestep / 2 * (Ar * Tin_avg + Ad) + 1);

        // Update the voltage.
        const Float G = gna * new_M * new_M * new_M * new_H + gk * powf(new_N, 4) + gcal * new_Rl * (K1 / (K2 + new_Cas)) + gcanl * new_Rnl * new_Fnl + gkca * new_S * new_S + gkleak + gnaleak + gnaP*new_P + new_Y;
        const Float E = Ena * (gna * new_M * new_M * new_M * new_H + gnaleak+ gnaP * new_P) + Ek * (gk * powf(new_N, 4) + gkca * new_S * new_S + gkleak) + Eca * (gcal * new_Rl * (K1 / (K2 + new_Cas)) + gcanl * new_Rnl * new_Fnl) + g_gaba * new_Y * egaba;

        sim_vars[DIEKMAN_VAR_V][idx] = (appcur * timestep + E * timestep + (1 - timestep / 2 * G) * V) / (1 + timestep / 2 * G);
        sim_vars[DIEKMAN_VAR_M][idx] = new_M;
        sim_vars[DIEKMAN_VAR_H][idx] = new_H;
        sim_vars[DIEKMAN_VAR_N][idx] = new_N;
        sim_vars[DIEKMAN_VAR_RL][idx] = new_Rl;
        sim_vars[DIEKMAN_VAR_RNL][idx] = new_Rnl;
        sim_vars[DIEKMAN_VAR_FNL][idx] = new_Fnl;
        sim_vars[DIEKMAN_VAR_S][idx] = new_S;
        sim_vars[DIEKMAN_VAR_Y][idx] = new_Y;
        sim_vars[DIEKMAN_VAR_CAS][idx] = new_Cas;
        sim_vars[DIEKMAN_VAR_CAC][idx] = new_Cac;

        // Calculate amount of neurotransmitter outputted AFTER update
        sim_vars[DIEKMAN_VAR_IOUTPUT][idx] = Tmax / (1.0 + exp(-(sim_vars[DIEKMAN_VAR_V][idx] - Vt) / Kp));
    }
}

void DiekmanEphys::SimulateEphys(Float** sim_vars)
{
    Leapfrog_diekman<<<NBLOCKS, NTHREADS>>>(sim_vars, N, hh_params::leapfrog_dt, thrust::raw_pointer_cast(random_states.data()));
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
}
