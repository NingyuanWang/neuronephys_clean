#include "paracrine.cuh"
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/sequence.h>
#include <thrust/functional.h>
#include <thrust/transform.h>
#include <array>

cusparseHandle_t Csr_matrix::handle;

//Helper function
thrust::device_vector<Float> af_array_to_thrust_vector(const af::array& af_array) {
    //refer to http://arrayfire.org/docs/interop_cuda.htm
    Float* data_ptr_on_device = af_array.device<Float>();
    const thrust::device_vector<Float> rval(data_ptr_on_device, data_ptr_on_device + af_array.elements());
    cudaDeviceSynchronize();
    return rval;
}
void Density_on_grid_3d::initialize_data() {
    std::vector<Float> zero_vector(x_count * y_count * z_count, 0.0f);
    gpuErrchk(cudaMalloc(&raw_data_d_ptr, x_count * y_count * z_count * sizeof(Float)));
    gpuErrchk(cudaMemcpy(raw_data_d_ptr, zero_vector.data(), x_count * y_count * z_count * sizeof(Float), cudaMemcpyHostToDevice));
    cudaDeviceSynchronize();
    data = af::array(x_count, y_count, z_count, raw_data_d_ptr, afDevice);
    data.eval();
    af::sync();
    data.device<Float>();
   
    cudaMalloc(&d_stencil, 27 * sizeof(Float));
    cudaMalloc(&d_A, 27 * sizeof(Float));
    cudaMalloc(&d_b, x_count * y_count * z_count * sizeof(Float));
    cudaMalloc(&d_p, x_count * y_count * z_count * sizeof(Float));
    cudaMalloc(&d_Ap, x_count * y_count * z_count * sizeof(Float));
    cudaMalloc(&d_r, x_count * y_count * z_count * sizeof(Float));
    cudaMalloc(&d_laplacian_u, x_count * y_count * z_count * sizeof(Float));
    cudaMalloc(&d_data_next, x_count * y_count * z_count * sizeof(Float));
    gpuErrchk(cudaDeviceSynchronize());
    laplace_stencil = af::array(3, 3, 3, d_stencil, afDevice);
    laplace_stencil.device<Float>();
    const Float laplace_stencil_h[] = { 1,3,1,3,14,3,1,3,1,3,14,3,14,-128,14,3,14,3,1,3,1,3,14,3,1,3,1 };//27 point stencil known as High-Order compact FD Schemes
                                                                  //Ref: https://www.researchgate.net/publication/2591103_High-Order_Compact_Finite_Difference_Schemes_for_Computational_Mechanics/link/00463524456e49822a000000/download page 110(125)
    cudaMemcpy(d_stencil, laplace_stencil_h, 27 * sizeof(Float), cudaMemcpyHostToDevice);
    laplace_stencil = laplace_stencil * (1 / (spatial_gridsize * spatial_gridsize * 30.0));
    laplacian_u = af::array(x_count, y_count, z_count, d_laplacian_u, afDevice);
    data_next_step = af::array(x_count, y_count, z_count, d_data_next, afDevice);
    A = af::array(3, 3, 3, d_A, afDevice);
    b = af::array(x_count, y_count, z_count, d_b, afDevice);
    p = af::array(x_count, y_count, z_count, d_p, afDevice);
    Ap = af::array(x_count, y_count, z_count, d_Ap, afDevice);
    r = af::array(x_count, y_count, z_count, d_r, afDevice);
    laplace_stencil.lock();
    laplacian_u.lock();
    data_next_step.lock();
    A.lock();
    b.lock();
    p.lock();
    Ap.lock();
    r.lock();
    af::sync();
}


void Density_on_grid_3d::update_density(const Float timestep, const Float error_bound) {
    //NOTE: AF_CONV_FREQ is the less optimal method for the size of input. However, since the convolve3 with 
    //AF_CONV_SPATIAL seems broken with the current version of arrayfire, this is the workaround to take now.
    // 1/dt(u^{n+1}-u^n)
    gpuErrchk(cudaDeviceSynchronize());
    af::sync();
    //Initial guess, using explicit method:
    laplacian_u = af::convolve3(data, laplace_stencil, af::convMode::AF_CONV_DEFAULT, af::convDomain::AF_CONV_FREQ);
    af::sync();
    
    data_next_step = (1 - timestep * decay_coefficient) * data + timestep * diffusion_coefficient * laplacian_u;
    //Eqn to solve: 
    //D*: diffusion coeff. k: decay coeff.
    //(1 + \Delta T k / 2) I - \Delta T D*/2 \nabla^27) u^(n+1) = u^n + \Delta T D* \nabla^27 u^n / 2 - \Delta T k u^n / 2
    //Rewrite as Ax = b. 
    //const Float identity_stencil_h[] = { 0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0 };
    //const af::array identity_stencil = af::array(3, 3, 3, identity_stencil_h);
    A = -0.5 * diffusion_coefficient * timestep * laplace_stencil;// +(1 + 0.5 * timestep * decay_coefficient) *identity_stencil;
    A(1, 1, 1) = A(1, 1, 1) + 1 + 0.5 * timestep * decay_coefficient;
#ifdef SOME_SANITY_CHECK_CODE
    af::print("A", A);
    const int test_shape_side_length = 7;
    af::array test_array_2 = af::constant(0.0, test_shape_side_length, test_shape_side_length, test_shape_side_length, f64);
    int entry_index = test_shape_side_length * test_shape_side_length * 4 + test_shape_side_length * 5 + 5;
    test_array_2(entry_index) = 1.0;
    test_array_2.eval();
    af::print("test_array_2", test_array_2);
    af::array concept_check_array = af::convolve3(test_array_2, A, af::convMode::AF_CONV_DEFAULT, af::convDomain::AF_CONV_FREQ);
    concept_check_array.eval();
    af::print("concept_check_array", concept_check_array);

    //af::print("concept check array", concept_check_array);
    std::cout << "Sum of values: expect " << af::sum<Float>(test_array_2) * af::sum<Float>(A) << ", actual " << af::sum<Float>(concept_check_array) << std::endl;
    af::array test_array = af::convolve3(data, A, af::convMode::AF_CONV_DEFAULT, af::convDomain::AF_CONV_FREQ);
    af::print("An entry that should read some value", concept_check_array(entry_index));
    auto data_shape = data.dims();
    std::cout << "Shape of data: [" << data_shape[0] << ", " << data_shape[1] << ", " << data_shape[2] << ", " << data_shape[3] << "]\n";
    data_shape = test_array.dims();
    std::cout << "Shape of test_array: [" << data_shape[0] << ", " << data_shape[1] << ", " << data_shape[2] << ", " << data_shape[3] << "]\n";
    std::cout << "Sum of values: expect " << af::sum<Float>(data) * af::sum<Float>(A) << ", actual " << af::sum<Float>(test_array);
#endif // SOME_SANITY_CHECK_CODE
    
    b = (1 - spatial_gridsize * decay_coefficient / 2) * data + spatial_gridsize * diffusion_coefficient / 2 * laplacian_u;
    const int max_iteration = 20;
    //Solving Ax = b using conjugate gradient descent: 
    //refer to page on wikipedia: https://en.wikipedia.org/wiki/Conjugate_gradient_method
    af::array r(x_count, y_count, z_count, d_r, afDevice);
    r = b - af::convolve3(data_next_step, A, af::convMode::AF_CONV_DEFAULT, af::convDomain::AF_CONV_FREQ);
    af::array p(x_count, y_count, z_count, d_p, afDevice);
    p = r;
    Float r_normsquare_old = pow(af::norm(af::flat(r), af::normType::AF_NORM_VECTOR_2), 2);
    if (r_normsquare_old < error_bound) {
        data.eval();
        return;
    }
    for (int i = 0; i < max_iteration; i++) {
        const af::array Ap = af::convolve3(p, A, af::convMode::AF_CONV_DEFAULT, af::convDomain::AF_CONV_FREQ);//A*p
        const Float pAp = af::sum<Float>(p*Ap);
        const Float alpha = r_normsquare_old / pAp; // (r'*r)/(p'*A*p)
        data_next_step = data_next_step + alpha * p;
        r = r - alpha * Ap;
        Float r_normsquare_new = pow(af::norm(af::flat(r), af::normType::AF_NORM_VECTOR_2), 2);
        if (r_normsquare_new < error_bound)
            break;
        p = r + (r_normsquare_new / r_normsquare_old) * p;
        r_normsquare_old = r_normsquare_new;
    }
    data = data_next_step;
    data.eval();
    af::sync();
    data.lock();
}

void Scatter_grid_converter_3d::initialize_distribute_private(const thrust::device_vector<Float> x_coordinate, const thrust::device_vector<Float> y_coordinate, const thrust::device_vector<Float> z_coordinate) {
    std::array<thrust::device_vector<int>, 8> neuron_to_gridpoint_index;
    std::array<thrust::device_vector<Float>, 8> neuron_to_gridpoint_weight;
    //x direction: 
    thrust::device_vector<int> m_vect(this->neuron_count);
    const Float x_sidelength = (x_count - 1)  * spatial_gridsize;
    const int x_count_copy = x_count;
    auto find_m = [x_sidelength, x_count_copy] __host__ __device__(const Float x) {
        const Float x_rescaled = x / x_sidelength + Float(0.5);
        const int m = ((int)((x_count_copy - 1) * x_rescaled) + 0.5);//Note difference by 1 expected
        return m;
    };
    thrust::transform(x_coordinate.begin(), x_coordinate.end(), m_vect.begin(), find_m);
    //y direction: 
    thrust::device_vector<int> n_vect(this->neuron_count);
    const Float y_sidelength = (y_count - 1)  * spatial_gridsize;
    const int y_count_copy = y_count;
    auto find_n = [y_sidelength, y_count_copy] __host__ __device__(const Float y) {
        const Float y_rescaled = y / y_sidelength + Float(0.5);
        const int n = ((int)((y_count_copy - 1) * y_rescaled) + 0.5);//Note difference by 1 expected
        return n;
    };
    thrust::transform(y_coordinate.begin(), y_coordinate.end(), n_vect.begin(), find_n);
    //z direction: 
    thrust::device_vector<int> p_vect(this->neuron_count);
    const Float z_sidelength = (z_count - 1)  * spatial_gridsize;
    const int z_count_copy = z_count;
    auto find_p = [z_sidelength, z_count_copy] __host__ __device__(const Float z) {
        const Float z_rescaled = z / z_sidelength + Float(0.5);
        const int p = ((int)((z_count_copy - 1) * z_rescaled) + 0.5);//Note difference by 1 expected
        return p;
    };
    thrust::transform(z_coordinate.begin(), z_coordinate.end(), p_vect.begin(), find_p);
    //Set linear index number: 
    auto mnp_tuple_begin = thrust::make_zip_iterator(thrust::make_tuple(m_vect.begin(), n_vect.begin(), p_vect.begin()));
    auto mnp_tuple_end = thrust::make_zip_iterator(thrust::make_tuple(m_vect.end(), n_vect.end(),p_vect.end()));
    auto set_linear_index = [x_count_copy,y_count_copy,z_count_copy] __host__ __device__(const thrust::tuple<int, int, int> mnp_tuple) {
        //Can work without zip. However for generation to 3d case, this form is used. 
        const int m = thrust::get<0>(mnp_tuple);
        const int n = thrust::get<1>(mnp_tuple);
        const int p = thrust::get<2>(mnp_tuple);
        return m + n * x_count_copy + p * x_count_copy * y_count_copy;
    };
    //Note no boundary condition (of index) is required to consider, since obviously the neurons need to be inside the domain.
    neuron_to_gridpoint_index[0] = thrust::device_vector<int>(this->neuron_count);
    thrust::transform(mnp_tuple_begin, mnp_tuple_end, neuron_to_gridpoint_index[0].begin(), set_linear_index);
    //Note the following 7 entries may not need to be stored as they can be inferred from the first one easily. 
    //However since the memory usage is modest, and this function runs only once, the convenience of us later
    //is more important. 
    for (int i = 1; i < 8; i++) {
        neuron_to_gridpoint_index[i] = thrust::device_vector<int>(this->neuron_count);
    }
    
    thrust::transform(neuron_to_gridpoint_index[0].begin(), neuron_to_gridpoint_index[0].end(), neuron_to_gridpoint_index[1].begin(), thrust::placeholders::_1 + 1);
    thrust::transform(neuron_to_gridpoint_index[0].begin(), neuron_to_gridpoint_index[0].end(), neuron_to_gridpoint_index[2].begin(), thrust::placeholders::_1 + x_count_copy);
    thrust::transform(neuron_to_gridpoint_index[0].begin(), neuron_to_gridpoint_index[0].end(), neuron_to_gridpoint_index[3].begin(), thrust::placeholders::_1 + (x_count_copy + 1));
    thrust::transform(neuron_to_gridpoint_index[0].begin(), neuron_to_gridpoint_index[0].end(), neuron_to_gridpoint_index[4].begin(), thrust::placeholders::_1 + (x_count_copy * y_count_copy));
    thrust::transform(neuron_to_gridpoint_index[0].begin(), neuron_to_gridpoint_index[0].end(), neuron_to_gridpoint_index[5].begin(), thrust::placeholders::_1 + (x_count_copy * y_count_copy + 1));
    thrust::transform(neuron_to_gridpoint_index[0].begin(), neuron_to_gridpoint_index[0].end(), neuron_to_gridpoint_index[6].begin(), thrust::placeholders::_1 + (x_count_copy * y_count_copy + x_count_copy));
    thrust::transform(neuron_to_gridpoint_index[0].begin(), neuron_to_gridpoint_index[0].end(), neuron_to_gridpoint_index[7].begin(), thrust::placeholders::_1 + (x_count_copy * y_count_copy + x_count_copy + 1));
    //Done with finding indices. Now, find weights. Using bicubic interpolation (key's kernel)
    //KeyKernel: 1.0 -               2.5 * input ^ 2 + 1.5 * input ^ 3; (input < 1)
    //           2.0 - 4.0 * input + 2.5 * input ^ 2 - 0.5 * input ^ 3; (input < 2)
    //x: 
    thrust::device_vector<Float> x_frac_vect(this->neuron_count);
    auto find_x_frac = [x_count_copy, x_sidelength] __host__ __device__(const Float x, const int m) {
        const Float x_rescaled = x / x_sidelength + Float(0.5);
        const Float rval = (x_count_copy - 1) * x_rescaled - m;
        return rval;
    };
    thrust::transform(x_coordinate.begin(), x_coordinate.end(), m_vect.begin(), x_frac_vect.begin(), find_x_frac);
    //y:
    thrust::device_vector<Float> y_frac_vect(this->neuron_count);
    auto find_y_frac = [y_count_copy, y_sidelength] __host__ __device__(const Float y, const int n) {
        const Float y_rescaled = y / y_sidelength + Float(0.5);
        const Float rval = (y_count_copy - 1) * y_rescaled - n;
        return rval;
    };
    thrust::transform(y_coordinate.begin(), y_coordinate.end(), n_vect.begin(), y_frac_vect.begin(), find_y_frac);
    //z:
    thrust::device_vector<Float> z_frac_vect(this->neuron_count);
    auto find_z_frac = [z_count_copy, z_sidelength] __host__ __device__(const Float z, const int p) {
        const Float z_rescaled = z / z_sidelength + Float(0.5);
        const Float rval = (z_count_copy - 1) * z_rescaled - p;
        return rval;
    };
    thrust::transform(z_coordinate.begin(), z_coordinate.end(), p_vect.begin(), z_frac_vect.begin(), find_z_frac);
    auto bilinear_kernel = [] __host__ __device__(const Float input) {
        return Float(1.0 - input);
    };
    auto bilinear_kernel_1s = [] __host__ __device__(const Float input_1s) {//1s means 1 substract
        const Float input = 1 - input_1s;
        return Float(1.0 - input);
    };
    thrust::device_vector<Float> bilinear_kernel_xfrac(this->neuron_count);
    thrust::transform(x_frac_vect.begin(), x_frac_vect.end(), bilinear_kernel_xfrac.begin(), bilinear_kernel);
    thrust::device_vector<Float> bilinear_kernel_yfrac(this->neuron_count);
    thrust::transform(y_frac_vect.begin(), y_frac_vect.end(), bilinear_kernel_yfrac.begin(), bilinear_kernel);
    thrust::device_vector<Float> bilinear_kernel_zfrac(this->neuron_count);
    thrust::transform(z_frac_vect.begin(), z_frac_vect.end(), bilinear_kernel_zfrac.begin(), bilinear_kernel);
    thrust::device_vector<Float> bilinear_kernel_1s_xfrac(this->neuron_count);
    thrust::transform(x_frac_vect.begin(), x_frac_vect.end(), bilinear_kernel_1s_xfrac.begin(), bilinear_kernel_1s);
    thrust::device_vector<Float> bilinear_kernel_1s_yfrac(this->neuron_count);
    thrust::transform(y_frac_vect.begin(), y_frac_vect.end(), bilinear_kernel_1s_yfrac.begin(), bilinear_kernel_1s);
    thrust::device_vector<Float> bilinear_kernel_1s_zfrac(this->neuron_count);
    thrust::transform(z_frac_vect.begin(), z_frac_vect.end(), bilinear_kernel_1s_zfrac.begin(), bilinear_kernel_1s);
    //A 2 step process that can use binary thrust::transform: 
    for (int i = 0; i < 8; i++) {
        neuron_to_gridpoint_weight[i] = thrust::device_vector<Float>(this->neuron_count);
    }
    //Step 1: populate with x y values.
    thrust::transform(bilinear_kernel_xfrac.begin(), bilinear_kernel_xfrac.end(), bilinear_kernel_yfrac.begin(), neuron_to_gridpoint_weight[0].begin(), thrust::multiplies<Float>());
    thrust::transform(bilinear_kernel_1s_xfrac.begin(), bilinear_kernel_1s_xfrac.end(), bilinear_kernel_yfrac.begin(), neuron_to_gridpoint_weight[1].begin(), thrust::multiplies<Float>());
    thrust::transform(bilinear_kernel_xfrac.begin(), bilinear_kernel_xfrac.end(), bilinear_kernel_1s_yfrac.begin(), neuron_to_gridpoint_weight[2].begin(), thrust::multiplies<Float>());
    thrust::transform(bilinear_kernel_1s_xfrac.begin(), bilinear_kernel_1s_xfrac.end(), bilinear_kernel_1s_yfrac.begin(), neuron_to_gridpoint_weight[3].begin(), thrust::multiplies<Float>());
    //Step 2: Add z information.
    for (int i = 0; i < 4; i++) {
        thrust::transform(neuron_to_gridpoint_weight[i].begin(), neuron_to_gridpoint_weight[i].end(), bilinear_kernel_1s_zfrac.begin(), neuron_to_gridpoint_weight[i + 4].begin(), thrust::multiplies<Float>());
    }
    for (int i = 0; i < 4; i++) {
        thrust::transform(neuron_to_gridpoint_weight[i].begin(), neuron_to_gridpoint_weight[i].end(), bilinear_kernel_zfrac.begin(), neuron_to_gridpoint_weight[i].begin(), thrust::multiplies<Float>());
    }
    //Convert to standard csr matrix:
    grid_to_scatter_interpolation_matrix = Csr_matrix(this->neuron_count, x_count*y_count*z_count, 8 * this->neuron_count);
    //row_entrycount: an increasing sequence 0,8, ..., 8 * this->neuron_count
    thrust::sequence(grid_to_scatter_interpolation_matrix.row_entrycount_begin(), grid_to_scatter_interpolation_matrix.row_entrycount_end(), 0, 8);
    //col_index and value: from a = [a1,a2,a3] b = [b1,b2,b3], c = [c1,c2,c3], ..., h to [a1,b1, ..., h1,a2,b2,c2,d2, ...]
    //IMPORTANT: csr is sorted, need to check ai<bi<ci<di<... < hi
    thrust::device_vector<int> index_vector(this->neuron_count);
    thrust::sequence(index_vector.begin(), index_vector.end(), 0, 8);
    for (int i = 0; i < 8; i++) {
        thrust::copy(neuron_to_gridpoint_index[i].begin(), neuron_to_gridpoint_index[i].end(), thrust::make_permutation_iterator(grid_to_scatter_interpolation_matrix.col_index_begin(), index_vector.begin()));
        thrust::copy(neuron_to_gridpoint_weight[i].begin(), neuron_to_gridpoint_weight[i].end(), thrust::make_permutation_iterator(grid_to_scatter_interpolation_matrix.value_begin(), index_vector.begin()));
        thrust::transform(index_vector.begin(), index_vector.end(), index_vector.begin(), thrust::placeholders::_1 + 1);
    }
    std::cout << "Enter transpose:\n";
    scatter_to_grid_distribution_matrix = grid_to_scatter_interpolation_matrix.get_transpose();
    return;
}

void Scatter_grid_converter_3d::scatter_to_grid(const thrust::device_vector<Float>& scatter, Density_on_grid_3d& grid) {
    scatter_to_grid_distribution_matrix.dense_vector_multiplication(scatter, Ax);
    //Use when a fast SpMSpV is available: grid_to_scatter_interpolation_matrix.sparse_vector_transpose_multiplication(scatter, Ax); 
    //Add newly generated density to grid (of af::array): 
    grid.data.eval();
    af::sync();
    thrust::device_ptr<Float> density_data_ptr = thrust::device_pointer_cast(grid.raw_data_d_ptr);
    thrust::transform(Ax.begin(), Ax.end(), density_data_ptr, density_data_ptr, thrust::plus<Float>());
    cudaDeviceSynchronize();
    //grid.data.unlock();
    return;
}
void Scatter_grid_converter_3d::interpolate_at_location(thrust::device_vector<Float>& scatter, const Density_on_grid_3d& grid) {
    grid.data.eval();
    af::sync();
    const Float* data_ptr_on_device = grid.raw_data_d_ptr;
    Float* scatter_ptr = thrust::raw_pointer_cast(scatter.data());
    grid_to_scatter_interpolation_matrix.dense_vector_multiplication(data_ptr_on_device, scatter_ptr);
    gpuErrchk(cudaDeviceSynchronize());
    //grid.data.unlock();
}

void Csr_matrix::dense_vector_multiplication(const thrust::device_vector<Float> x, thrust::device_vector<Float>& Ax)
{
    dense_vector_multiplication(thrust::raw_pointer_cast(x.data()), thrust::raw_pointer_cast(Ax.data()));
}

void Csr_matrix::dense_vector_multiplication(const Float* x_ptr, Float* Ax_ptr) {
    const Float alpha = 1.0;
    const Float beta = 0.0;
    int* row_entrycount_ptr = thrust::raw_pointer_cast(d_row_entrycount.data());
    int* col_index_ptr = thrust::raw_pointer_cast(d_col_index.data());
    Float* value_ptr = thrust::raw_pointer_cast(d_value.data());

    auto type = (sizeof(Float) < sizeof(double)) ? CUDA_R_32F : CUDA_R_64F;

    cusparseSpMatDescr_t mat_desc;
    cusparseErrchk(cusparseCreateCsr(&mat_desc, row_count, col_count, nnz,
        (void*)thrust::raw_pointer_cast(d_row_entrycount.data()),
        (void*)thrust::raw_pointer_cast(d_col_index.data()),
        (void*)thrust::raw_pointer_cast(d_value.data()),
        CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO,
        type
    ));
    gpuErrchk(cudaDeviceSynchronize());
    cusparseDnVecDescr_t x_vec, Ax_vec;
    cusparseErrchk(cusparseCreateDnVec(&x_vec, col_count, (void*)x_ptr, CUDA_R_32F));
    cusparseErrchk(cusparseCreateDnVec(&Ax_vec, row_count, (void*)Ax_ptr, CUDA_R_32F));
    gpuErrchk(cudaDeviceSynchronize());
    size_t* buffer_size = new size_t;
    Float* d_mv_buffer;
    cusparseErrchk(cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, mat_desc, x_vec, &beta, Ax_vec, type, CUSPARSE_CSRMV_ALG1, buffer_size));
    gpuErrchk(cudaDeviceSynchronize());
    cudaMalloc(&d_mv_buffer, *buffer_size);
    //std::wcout << "BUF SIZE " << *buffer_size << std::endl;
    delete buffer_size;
    cusparseErrchk(cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, mat_desc, x_vec, &beta, Ax_vec, type, CUSPARSE_CSRMV_ALG1, d_mv_buffer));
    gpuErrchk(cudaDeviceSynchronize());
    cudaFree(d_mv_buffer);
}

void Csr_matrix::sparse_vector_transpose_multiplication(const thrust::device_vector<Float> x, thrust::device_vector<Float>& Ax) {
    thrust::device_vector<int> x_nonzero_index(row_count);
    thrust::device_vector<int> index_vector(row_count);
    thrust::sequence(index_vector.begin(), index_vector.end());
    //thrust::counting_iterator<int> index_vector(row_count);
    
    auto is_nonzero = [] __host__ __device__(const Float input) { return input!=0.0; };
    auto x_nonzero_index_end = thrust::copy_if(index_vector.begin(), index_vector.end(), x.begin(), x_nonzero_index.begin(), is_nonzero);
    const int nonzero_count = x_nonzero_index_end - x_nonzero_index.begin();
    thrust::fill(Ax.begin(), Ax.end(), 0.0);
    if (nonzero_count == 0) {
        return;
    }
    //A primitive method: iterate through columns:
    for (auto itr = x_nonzero_index.begin(); itr < x_nonzero_index_end; itr++) {
        const Float alpha = 1.0;

        auto add_alpha = [alpha] __host__ __device__(const Float input, const Float multiplier) { return input + alpha * multiplier; };
        thrust::transform(thrust::make_permutation_iterator(Ax.begin(), d_col_index.begin() + d_row_entrycount[*itr]),
            thrust::make_permutation_iterator(Ax.begin(), d_col_index.begin() + d_row_entrycount[(*itr) + 1]),
            d_value.begin() + d_row_entrycount[*itr],
            thrust::make_permutation_iterator(Ax.begin(), d_col_index.begin() + d_row_entrycount[*itr]),
            add_alpha);
    }
    return;
}

void Csr_matrix::sparse_vector_transpose_multiplication(const thrust::device_vector<bool> x, thrust::device_vector<Float>& Ax) {
    thrust::device_vector<int> x_nonzero_index(row_count);
    thrust::device_vector<int> index_vector(row_count);
    thrust::sequence(index_vector.begin(), index_vector.end());
    //thrust::counting_iterator<int> index_vector(row_count);

    auto is_nonzero = [] __host__ __device__(const bool input) { return input; };
    auto x_nonzero_index_end = thrust::copy_if(index_vector.begin(), index_vector.end(), x.begin(), x_nonzero_index.begin(), is_nonzero);
    const int nonzero_count = x_nonzero_index_end - x_nonzero_index.begin();
    thrust::fill(Ax.begin(), Ax.end(), 0.0);
    if (nonzero_count == 0) {
        return;
    }
    //A primitive method: iterate through columns:
    for (auto itr = x_nonzero_index.begin(); itr < x_nonzero_index_end; itr++) {
        const Float alpha = 1.0;
        
        auto add_alpha = [alpha] __host__ __device__(const Float input, const Float multiplier) { return input + alpha * multiplier; };
        thrust::transform(thrust::make_permutation_iterator(Ax.begin(), d_col_index.begin() + d_row_entrycount[*itr]),
            thrust::make_permutation_iterator(Ax.begin(), d_col_index.begin() + d_row_entrycount[(*itr) + 1]),
            d_value.begin() + d_row_entrycount[*itr],
            thrust::make_permutation_iterator(Ax.begin(), d_col_index.begin() + d_row_entrycount[*itr]),
            add_alpha);
    }
    return;
}

void Csr_matrix::transpose() {
    Csr_matrix AT = get_transpose();
    row_count = AT.row_count;
    col_count = AT.col_count;
    d_col_index = AT.d_col_index;
    d_row_entrycount = AT.d_row_entrycount;
    d_value = AT.d_value;
    return;
}

void Csr_matrix::get_transpose(Csr_matrix& AT) const {
    cudaDataType cuda_data_type;
    if (sizeof(Float) == sizeof(double)) {
        cuda_data_type = CUDA_R_64F;
    }
    else if (sizeof(Float) == sizeof(float)) {
        cuda_data_type = CUDA_R_32F;
    }
    const int* i_row_entrycount_ptr = thrust::raw_pointer_cast(d_row_entrycount.data());
    const int* i_col_index_ptr = thrust::raw_pointer_cast(d_col_index.data());
    const void* i_value_ptr = thrust::raw_pointer_cast(d_value.data());
    int* o_row_entrycount_ptr = thrust::raw_pointer_cast(AT.d_row_entrycount.data());
    int* o_col_index_ptr = thrust::raw_pointer_cast(AT.d_col_index.data());
    void* o_value_ptr = thrust::raw_pointer_cast(AT.d_value.data());
    size_t* buffer_size = new size_t;
    //First: find buffer required.
    cusparseCsr2cscEx2_bufferSize(handle, row_count, col_count, nnz, i_value_ptr, i_row_entrycount_ptr, i_col_index_ptr,
        o_value_ptr, o_row_entrycount_ptr, o_col_index_ptr, cuda_data_type, CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO, CUSPARSE_CSR2CSC_ALG2, buffer_size);
    //Then: allocate buffer
    void* buffer_ptr;
    cudaGetErrorString(cudaMalloc(&buffer_ptr, sizeof(double) * *buffer_size));
    std::cout << "Required buffer size: " << *buffer_size << std::endl;
    delete buffer_size;
    //Perform transpose:
    cusparseCsr2cscEx2(handle, row_count, col_count, nnz, i_value_ptr, i_row_entrycount_ptr, i_col_index_ptr,
        o_value_ptr, o_row_entrycount_ptr, o_col_index_ptr, cuda_data_type, CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO, CUSPARSE_CSR2CSC_ALG2, buffer_ptr);
    //Finally, free buffer:
    cudaFree(buffer_ptr);
    std::cout << "transpose done.\n";
    return;
}
Csr_matrix Csr_matrix::get_transpose() const {
    Csr_matrix rval;
    rval.row_count = col_count;
    rval.col_count = row_count;
    rval.nnz = nnz;
    rval.d_row_entrycount = thrust::device_vector<int>(rval.row_count + 1);
    rval.d_col_index = thrust::device_vector<int>(rval.nnz);
    rval.d_value = thrust::device_vector<Float>(rval.nnz);
    get_transpose(rval);
    return rval;
}
