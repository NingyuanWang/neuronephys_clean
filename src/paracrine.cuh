#pragma once
#include "defines.cuh"
#include <arrayfire.h>
#include <random>
#include <af/cuda.h>
#include "coupling.cuh"

//! TODO: Port this over to CuSparseAdjacencyMatrix!. 
class Csr_matrix {
private:
    int row_count;
    int col_count;
	int nnz;
	thrust::device_vector<int> d_row_entrycount;
	thrust::device_vector<int> d_col_index;
	thrust::device_vector<Float> d_value;
public:
    static cusparseHandle_t handle;
    Csr_matrix() {//Default constructor that does almost nothing.
        row_count = 0;
        col_count = 0;
        nnz = 0;
        d_row_entrycount = thrust::device_vector<int>();
        d_col_index = thrust::device_vector<int>();
        d_value = thrust::device_vector<Float>();
    }
    Csr_matrix(int row_count, int col_count, int nnz) ://Allocates space but nothing else
        row_count(row_count), col_count(col_count), nnz(nnz) {
        d_row_entrycount = thrust::device_vector<int>(row_count+1);
        d_col_index = thrust::device_vector<int>(nnz);
        d_value = thrust::device_vector<Float>(nnz);
    }

    thrust::device_vector<int>::iterator row_entrycount_begin() { return d_row_entrycount.begin(); }
    thrust::device_vector<int>::const_iterator row_entrycount_begin() const { return d_row_entrycount.begin(); }
    thrust::device_vector<int>::iterator row_entrycount_end() { return d_row_entrycount.end(); }
    thrust::device_vector<int>::const_iterator row_entrycount_end() const { return d_row_entrycount.end(); }
    thrust::device_vector<int>::iterator col_index_begin() { return d_col_index.begin(); }
    thrust::device_vector<int>::const_iterator col_index_begin() const { return d_col_index.begin(); }
    thrust::device_vector<int>::iterator col_index_end() { return d_col_index.end(); }
    thrust::device_vector<int>::const_iterator col_index_end() const { return d_col_index.end(); }
    typename thrust::device_vector<Float>::iterator value_begin() { return d_value.begin(); }
    typename thrust::device_vector<Float>::const_iterator value_begin() const { return d_value.begin(); }
    typename thrust::device_vector<Float>::iterator value_end() { return d_value.end(); }
    typename thrust::device_vector<Float>::const_iterator value_end() const { return d_value.end(); }
    int m() const { return row_count; };
    int n() const { return col_count; };
    int entry_count() const { return nnz; };
    void input_from_connection_file(const std::string connFile) {
        // Input file in the format
        // neuronCount nnz
        // rowPtr (length neuronCount + 1)
        // colIdx (length nnz)
        // val (length nnz)
        std::ifstream fin;
        fin.open(connFile.c_str());
        std::string line;
        std::istringstream ls;
        std::getline(fin, line);
        ls.str(line);
        ls >> row_count >> nnz;
        col_count = row_count;
        std::getline(fin, line);
        ls.clear();
        ls.str(line);
        d_row_entrycount = thrust::device_vector<int>(row_count + 1);
        auto h_row_entrycount = thrust::host_vector<int>(row_count + 1);
        for (int i = 0; i < row_count + 1; i++)
            if (!ls.eof())
                ls >> h_row_entrycount[i];
            else
            {
                std::cout << "error reading neuron connection file." << std::endl;
            }
        thrust::copy(h_row_entrycount.begin(), h_row_entrycount.end(), d_row_entrycount.begin());
        h_row_entrycount.clear();
        d_col_index = thrust::device_vector<int>(nnz);
        auto h_col_index = thrust::host_vector<int>(nnz);
        std::getline(fin, line);
        ls.clear();
        ls.str(line);
        for (int i = 0; i < nnz; i++)
            if (!ls.eof())
                ls >> h_col_index[i];
            else
            {
                std::cout << "error reading neuron connection file." << std::endl;
            }
        thrust::copy(h_col_index.begin(), h_col_index.end(), d_col_index.begin());
        h_col_index.clear();
        d_value = thrust::device_vector<Float>(nnz);
        auto h_value = thrust::host_vector<Float>(nnz);
        std::getline(fin, line);
        ls.clear();
        ls.str(line);
        for (int i = 0; i < nnz; i++)
        {
            if (!ls.eof()) {
                ls >> h_value[i];
            }
            else
            {
                std::cout << "error reading neuron connection file." << std::endl;
            }
        }
        thrust::copy(h_value.begin(), h_value.end(), d_value.begin());
        h_value.clear();
    }
    void dense_vector_multiplication(const thrust::device_vector<Float> x, thrust::device_vector<Float>& Ax);//Does not allocate space to Ax.
    void dense_vector_multiplication(const Float* x_ptr, Float* Ax_ptr);//Version for raw data. 
    void sparse_vector_transpose_multiplication(const thrust::device_vector<Float> x, thrust::device_vector<Float>& Ax);//Does not allocate space to A^T x.
    void sparse_vector_transpose_multiplication(const thrust::device_vector<bool> x, thrust::device_vector<Float>& Ax);//Does not allocate space to A^T x. treats as vector of 0s and 1s. 
    Csr_matrix get_transpose() const;//Returns a transpose of the matrix, ALLOCATES MEMORY.
    void get_transpose(Csr_matrix& AT) const;//Returns a transpose of the matrix, does not allocate memory.
    void transpose();//Transpose this matrix.
};

//Finite difference for heat eqn:
//3D version: Note in this version, we consider the grid point extends to the boundary, so that boundary conditions can be imposed later. Whereas currently
// 2D version uses only "interior grid points". 
struct Density_on_grid_3d {
    
    void initialize_data();
    Float spatial_gridsize;//square grid
    int x_count;//width
    int y_count;//height
    int z_count;//depth
    Float diffusion_coefficient;
    Float decay_coefficient;
    Float* raw_data_d_ptr;
    af::array data;
private:
    Float* d_stencil;
    Float* d_A;
    Float* d_b;
    Float* d_r;
    Float* d_p;
    Float* d_Ap;
    Float* d_laplacian_u;
    Float* d_data_next;
    af::array laplacian_u;
    af::array data_next_step;
    af::array A;
    af::array b;
    af::array r;
    af::array p;
    af::array Ap;
    af::array laplace_stencil;
public:
    Density_on_grid_3d(int x_count, int y_count, int z_count, Float spatial_gridsize, Float diffusion_coefficient, Float decay_coefficient) :

        spatial_gridsize(spatial_gridsize),
        x_count(x_count),
        y_count(y_count),
        z_count(z_count),
        diffusion_coefficient(diffusion_coefficient),
        decay_coefficient(decay_coefficient),
        data(),
        raw_data_d_ptr(NULL)
    {
        initialize_data();
    }
    //convolution with discrete stencil laplacian:
    void update_density(const Float timestep, const Float error_bound = 1e-6);
    //Testing functionality:
    void set_delta() {
        data(x_count / 2, y_count / 2, z_count / 2) = 1.0;
        return;
    }
    const af::array& density() const {
        return data;
    }
};

template<typename T>
void print_thrust_vector(const thrust::device_vector<T> thrust_vector, std::string title="") {
    std::cout << title << ": ";
    thrust::copy(thrust_vector.begin(), thrust_vector.end(), std::ostream_iterator<T>(std::cout, " "));
    std::cout << std::endl;
}
//The interpolation and distribution is separated from the neuron d_vars and density grids for a more modular layout.
// (Note: Sure this can be put inside a larger class as a member later on)
//TODO: Eventually 2D and 3D can be inside the same template...
class Scatter_grid_converter_3d
{
    //These matrices are transpose to each other, however since SpMV is faster with csr than csc, both are stored.
    Csr_matrix scatter_to_grid_distribution_matrix;
    Csr_matrix grid_to_scatter_interpolation_matrix;
    thrust::device_vector<Float> Ax;//Distribution found over this time step. 
public:
    const Float spatial_gridsize;
    const int x_count;//width
    const int y_count;//height
    const int z_count;//depth
    const int neuron_count;
    void initialize_distribute_private(const thrust::device_vector<Float> x_coordinate, const thrust::device_vector<Float> y_coordinate, const thrust::device_vector<Float> z_coordinate);//Note: public due to a limit of CUDA, shall not be called outside 
    Scatter_grid_converter_3d(const Density_on_grid_3d& spatial_grid, const int neuron_count, const thrust::device_vector<Float> x_coordinate, const thrust::device_vector<Float> y_coordinate, const thrust::device_vector<Float> z_coordinate) :
        spatial_gridsize(spatial_grid.spatial_gridsize),
        neuron_count(neuron_count),
        x_count(spatial_grid.x_count),
        y_count(spatial_grid.y_count),
        z_count(spatial_grid.z_count)
    {
        Ax = thrust::device_vector<Float>(x_count*y_count*z_count, 0.0);
        initialize_distribute_private(x_coordinate,y_coordinate,z_coordinate);
    }
    void scatter_to_grid(const thrust::device_vector<Float>& scatter, Density_on_grid_3d& grid);
    void interpolate_at_location(thrust::device_vector<Float>& scatter, const Density_on_grid_3d& grid);
};

// Module handling paracrine signalling
class ParacrineModule
{
    Density_on_grid_3d grid;
    Scatter_grid_converter_3d grid_converter;
public:
    int x_gridcount;
    int y_gridcount;
    int z_gridcount;
    int timescale;
private:
    Float* d_g_paracrine; // Location of paracrine conductances used by electrophysiology.
    Float par_dt; 
    Float coup_strength;

    thrust::device_vector<Float> density_at_neurons;
    thrust::device_vector<Float> density_generated_at_neurons;

public:
    ParacrineModule (Float*& d_g_paracrine_, int const neuron_count, int const x_grid_count, int const y_grid_count, int const z_grid_count, Float const par_dt_, int const timescale_, Float const coup_strength_, Float const spatial_gridsize, Float const diffusion_coef, Float const decay_coef,
            thrust::device_vector<Float> const& d_x_coords, thrust::device_vector<Float> const& d_y_coords, thrust::device_vector<Float> const& d_z_coords) 
        : d_g_paracrine{d_g_paracrine_}, par_dt{par_dt_}, coup_strength{coup_strength_}, density_at_neurons(neuron_count, 0.0), 
        x_gridcount(x_grid_count), y_gridcount(y_grid_count), z_gridcount(z_grid_count), timescale{timescale_},density_generated_at_neurons(neuron_count, 0.0),
        grid(x_grid_count, y_grid_count, z_grid_count, spatial_gridsize, diffusion_coef, decay_coef), 
        grid_converter(grid, neuron_count, d_x_coords, d_y_coords, d_z_coords)
    {
        //grid.data.eval();
        grid_converter.interpolate_at_location(density_at_neurons, grid);
        //af::sync();
        std::cout << "Tree.\n";
        gpuErrchk(cudaDeviceSynchronize());
        thrust::copy(density_at_neurons.begin(), density_at_neurons.end(), thrust::device_ptr<Float>(d_g_paracrine));
        std::cout << "Tree2.\n";
    }

    virtual void update_on_fine_timescale(AtomicCoupling const& coupling) 
    {
        // Accumulate density of neurotransmitter on ephys timescale.
        // We add coup_strength to element i of density_generated_at_neurons if neuron i fired.
        thrust::transform_if(density_generated_at_neurons.begin(), density_generated_at_neurons.end(), 
            coupling.GetFiringNeurons().begin(), density_generated_at_neurons.begin(), 
            thrust::placeholders::_1 + coup_strength, thrust::placeholders::_1);
    }

    virtual void update_on_own_timescale()
    {
        grid_converter.scatter_to_grid(density_generated_at_neurons, grid);
        grid.update_density(par_dt);
        grid_converter.interpolate_at_location(density_at_neurons, grid);
        thrust::fill(density_generated_at_neurons.begin(), density_generated_at_neurons.end(), 0.0);
        
        // This copy is only needed since the paracrine signalling takes a device_vector, not a raw pointer. 
        // Therefore, we need to send the device_vector to the paracrine, then copy the results to the pointer.
        thrust::copy(density_at_neurons.begin(), density_at_neurons.end(), d_g_paracrine);
    }
    Density_on_grid_3d const& get_grid () {return grid;}
};
