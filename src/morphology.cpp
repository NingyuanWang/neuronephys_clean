#define _USE_MATH_DEFINES
#include "morphology.h"
#include <set>
#include <glm/ext.hpp>
#include <glm/gtx/norm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <unordered_map>
using namespace H5;

void Morphology::Union(Morphology* rhs)
{
    std::cout << "Computing union of two morphologies..." << std::endl;
	size_t cur_size{ positions.size() };
	positions.resize(cur_size + rhs->positions.size());
	std::copy(rhs->positions.begin(), rhs->positions.end(), positions.begin() + cur_size);
}

void Morphology::Translate(glm::vec3 const& translate)
{
    for (glm::vec3& pt : positions)
        pt += translate;
}

void Morphology::Scale(glm::vec3 const& scale)
{
    for (glm::vec3& pt : positions)
        pt *= scale;
}

void Morphology::Rotate(float const angle, glm::vec3 const& axis)
{
    auto rot_mat = glm::rotate(glm::mat4(1.0f), angle, axis);
    for (glm::vec3& pt : positions)
        pt = glm::vec3(rot_mat * glm::vec4(pt, 1.0f));
}

void Morphology::KNN(int const n, int const k, std::vector<int>& knn, std::vector<ANNdist>& knn_dists)
{
    KNN(glm::value_ptr(positions[n]), k, knn, knn_dists);
}

void Morphology::KNN(ANNpoint pos , int const k, std::vector<int>& knn, std::vector<ANNdist>& knn_dists)
{
    if (!kd_tree)
        SetupANN();
	kd_tree->annkSearch(pos, k, knn.data(), knn_dists.data());
}

int Morphology::WithinBall(int const n, Float dist_not_sqr, int max_n_conns, std::vector<int>& ball, std::vector<ANNdist>& ball_dists)
{
    return WithinBall(glm::value_ptr(positions[n]), dist_not_sqr, max_n_conns, ball, ball_dists);
}

int Morphology::WithinBall(ANNpoint pos, Float dist_not_sqr, int max_n_conns, std::vector<int>& ball, std::vector<ANNdist>& ball_dists)
{
    if (!kd_tree)
        SetupANN();
    return kd_tree->annkFRSearch(pos, dist_not_sqr * dist_not_sqr, max_n_conns, ball.data(), ball_dists.data());
}

void Morphology::GenSphericalRegions(
    thrust::host_vector<int>& h_regions, thrust::host_vector<int>& h_region_inds, 
    std::vector<int> const& electrode_inds, int const k)
{
    if (!kd_tree)
        SetupANN();

    std::cout << "Generating spherical regions..." << std::endl;
    std::vector<int> knn(k+1); 
    std::vector<ANNdist> knn_dists(k+1);

	int const N_regions = electrode_inds.size();
    h_regions.resize(N_regions * k);
    h_region_inds.resize(N_regions + 1, 0);
    for (int n = 0; n < N_regions; ++n)
    {
		// Initialize query point. 
		int const idx = electrode_inds[n];
	
        KNN(idx, k, knn, knn_dists);
		std::copy(knn.begin(), knn.end(), h_regions.begin() + k * n);
        h_region_inds[n + 1] = h_region_inds[n] + k;
    }
}

// TODO: reduce code duplication.
void Morphology::GenSphericalRegions(
    thrust::host_vector<int>& h_regions, thrust::host_vector<int>& h_region_inds, 
    std::vector<glm::vec3> const& electrode_positions, int const k)
{
    if (!kd_tree)
        SetupANN();

    std::cout << "Generating spherical regions..." << std::endl;
    std::vector<int> knn(k+1); 
    std::vector<ANNdist> knn_dists(k+1);

	int const N_regions = electrode_positions.size();
    h_regions.resize(N_regions * k);
    h_region_inds.resize(N_regions + 1, 0);
    for (int n = 0; n < N_regions; ++n)
    {
		// Initialize query point. 
		glm::vec3 pos = electrode_positions[n];
	
        KNN(glm::value_ptr(pos), k, knn, knn_dists);
		std::copy(knn.begin(), knn.end(), h_regions.begin() + k * n);
        h_region_inds[n + 1] = h_region_inds[n] + k;
    }
}

void Morphology::GetBoundaryRegion(
    thrust::host_vector<int>& h_regions, thrust::host_vector<int>& h_region_inds) const
{
    for (int i = 0; i < positions.size(); ++i)
        if (IsBoundaryPoint(positions[i]))
            h_regions.push_back(i);
    h_region_inds.resize(2);
    h_region_inds[0] = 0;
    h_region_inds[1] = (int)h_regions.size();
    std::cout << "Boundary region number of points: " << h_regions.size() << std::endl;
}


void Morphology::SetupANN ()
{
    std::cout << "Setting up ANN..." << std::endl;
    int const N = positions.size();
    ANNpointArray p_vec{ annAllocPts(N, 3) };
    for (int i = 0; i < N; ++i)
        p_vec[i] = glm::value_ptr(positions[i]);
	kd_tree = std::make_unique<ANNkd_tree>(p_vec, N, 3);
}

std::vector<bool> Morphology::ToBinaryVoxels(glm::uvec3 const& dims) const
{
    std::vector<bool> grid(dims[0] * dims[1] * dims[2], false);
    auto dims_float = glm::vec3(dims) - glm::vec3(1);
    for (auto const& pt : positions)
    {
        glm::uvec3 vidx = dims_float * (pt + glm::vec3(1)) / glm::vec3(2);
        grid[vidx.x + dims[0] * (vidx.y + dims[1] * vidx.z)] = true;
    }
    return grid;
}

void Morphology::ProjectNeuronsToPlane(int dim)
{
    for (auto& pos : positions)
        pos[dim] = 0;
}

float Morphology::GetMinimumDistance()
{
    if (!kd_tree)
        SetupANN();

    std::vector<int> knn(2);
    std::vector<ANNdist> knn_dists(2);
    float min_dist2 = FLT_MAX;
    for (int n = 0; n < positions.size(); ++n)
    {
        // Get two nearest neighbors. First nearest neighor is always just the neuron itself.
        // knn_dists[1] stores squared distance to nearest neuron.
        KNN(n, 2, knn, knn_dists);
        if(knn_dists[1] < min_dist2)
            min_dist2 = knn_dists[1];
    }
    return sqrt(min_dist2);
}

bool HollowSphere::Generate(int const N)
{
    std::cout << "Generating hollow sphere morphology..." << std::endl;
    auto& gen{ Utility::UnseededRandomEngine() };
    std::uniform_real_distribution<float> dist(0, 2 * M_PI);
    positions.resize(N);
    for (int i = 0; i < N; ++i)
    {
        float const theta{ dist(gen) }, phi{ dist(gen) };
        positions[i] = { cos(theta)*sin(phi), sin(theta)*sin(phi), cos(phi) };
    }
    return true;
}

bool UnitSquare::Generate(int const N)
{
    std::cout << "Generating unit square morphology..." << std::endl;
    auto& gen{ Utility::UnseededRandomEngine() };
    std::uniform_real_distribution<float> dist(-1, 1);
    positions.resize(N);
    for (int i = 0; i < N; ++i)
    {
        float const x{ dist(gen) }, y{ dist(gen) };
        positions[i] = { x, y, 0 };
    }
    return true;
}

bool UnitCube::Generate(int const N)
{
    std::cout << "Generating unit cube morphology..." << std::endl;
    auto& gen{ Utility::UnseededRandomEngine() };
    std::uniform_real_distribution<float> dist(-1, 1);
    positions.resize(N);
    for (int i = 0; i < N; ++i)
    {
        float const x{ dist(gen) }, y{ dist(gen) }, z{ dist(gen) };
        positions[i] = { x, y, z };
    }
    return true;
}

bool ImplicitSolid::Generate(int const N)
{
    // Generate points uniformly in unit cube and only keep those for which F(p) <= 0.
    std::cout << "Generating implicit solid morphology..." << std::endl;
    auto& gen{ Utility::UnseededRandomEngine() };
    std::uniform_real_distribution<float> dist(-1, 1);
    positions.resize(N);
    int pos_size = 0;
    while (pos_size < N)
    {
        float const x{ dist(gen) }, y{ dist(gen) }, z{ dist(gen) };
        glm::vec3 p{ x, y, z };
        if (F(p) <= 0)
        {
            positions[pos_size] = p;
            pos_size += 1;
        }
    }
    return true;
}

bool ImplicitSolid::IsBoundaryPoint(glm::vec3 const& p) const
{
    static constexpr Float tol = 0.01f;
    return fabs(F(p)) <= tol;
}

Float SolidSphere::F(glm::vec3 const& p) const
{
    return glm::length2(p) - 1;
}

Float SimpleWrinkledSolidSphere::F(glm::vec3 const& p) const
{
    // Convert cartesian to spherical coordinates.
    Float r = glm::length(p);
    Float theta = atan2(p.y, p.x);
    Float phi = atan2(sqrt(p.x * p.x + p.y * p.y), p.z);
    return r - cos(phi) * (0.15 * cos(20 * theta) + 0.85);
}

FileMorphology::FileMorphology(std::string const& fl_name_, bool read_regions_)
    : fl_name{ "../" + fl_name_ }, format{ fl_name_.substr(fl_name_.find_last_of('.') + 1) }, read_regions{ read_regions_ }
{
}

void FileMorphology::SpecfiyRegionsDepth(int const depth_in_tree, RegionsTree const& regions_tree)
{
    std::cout << "Updating regions depth to be " << depth_in_tree << std::endl;

    // Replace regions with parent regions at depth specified in constructor.
    std::unordered_map<int, int> node_to_parent;
    std::set<int> new_unique_regions_set; 
    try
    {
        for (int id : unique_regions)
        {
            int parent_at_depth = regions_tree.GetParentRegionAtDepth(id, depth_in_tree);
            node_to_parent[id] = parent_at_depth;
            new_unique_regions_set.insert(parent_at_depth);
        }
    }
    catch (std::runtime_error& e)
    {
        std::cerr << e.what() << std::endl;
        exit(1);
    }
    // Update regions using map.
    for (int& id : regions)
        id = node_to_parent[id];

    // Update unique_regions with new unique regions at specified depth.
    unique_regions = std::vector<int>(new_unique_regions_set.begin(), new_unique_regions_set.end());
    std::cout << "Done updating region depth. Number of unique regions at depth: " << unique_regions.size() << std::endl;
}

bool FileMorphology::Generate(int const N)
{
    std::cout << "Generating morphology from file '" + fl_name + "'..." << std::endl;
    if (format != "h5" && format != "H5")
    {
        std::cerr << "Unsupported format " << format << std::endl;
        return false;
    }
    return GenerateFromHDF(N);
}

bool FileMorphology::GenerateFromHDF(int const N)
{
    // First read in three individual axes, then put into positions,
    // finally scale and transform into range [-1, 1].
    std::vector<float> axes[4];
    const char* coord_ids[4]{ "/x", "/y", "/z", "/region_id" };
    int numpnts;
    int n_dims = read_regions ? 4 : 3;
    try
    {
        H5File file(fl_name, H5F_ACC_RDONLY);
        for (int i = 0; i < n_dims; ++i)
        {
            DataSet axis = file.openDataSet(coord_ids[i]);
            numpnts = (int)(axis.getStorageSize() / sizeof(float));
            axes[i].resize(numpnts);
            axis.read(axes[i].data(), PredType::NATIVE_FLOAT);
        }
    }
    catch (...)
    {
        return false;
    }

    // Calculate bounding box.
    glm::vec3 wmin, wmax;
    for (int i = 0; i < 3; ++i)
    {
        wmin[i] = *std::min_element(axes[i].begin(), axes[i].end());
        wmax[i] = *std::max_element(axes[i].begin(), axes[i].end());
    }

    // Check if we have enough points.
    if (N > numpnts)
    {
        std::cerr << "Not enough points in file " << fl_name 
                  << ". Have N = " << N << " and " << numpnts 
                  << " points available." << std::endl;
        return false;
    }

    // Sample points uniformly so that we have exactly N, not numpnts. 
    int stride{ numpnts / N };
    positions.resize(N);
    for (int i = 0; i < N; ++i)
    {
        int const j{ stride * i };
        positions[i] = { axes[0][j], axes[1][j], axes[2][j] };
    }

    if (read_regions)
    {
        regions.resize(N);
        for (int i = 0; i < N; ++i)
        {
            int const j{ stride * i };
            regions[i] = axes[n_dims - 1][j];
        }
        std::set<int> unique_regions_set(regions.begin(), regions.end());
        unique_regions = std::vector<int>(unique_regions_set.begin(), unique_regions_set.end());
    }

    // Translate so that min is zero.
    Translate(-wmin);

    // Scale so that everything is in the range [0, 2].
    Scale(glm::vec3(2.0f) / (wmax - wmin));

    // Finally, translate so that min is -1 and everything is in range [-1, 1].
    Translate(glm::vec3(-1));

	return true;
}

void FileMorphology::GenRegionsFromIDs(thrust::host_vector<int>& h_regions, thrust::host_vector<int>& h_region_inds, thrust::host_vector<int>& associated_ids)
{
    if (!kd_tree)
        SetupANN();

    std::cout << "Generating regions from IDs..." << std::endl;
    int const N = positions.size();
	int const N_regions = unique_regions.size();
    h_regions.resize(N);
    h_region_inds.resize(N_regions + 1, 0);
    auto it = h_regions.begin();
    std::vector<int> sizes;
    for (int n = 0; n < N_regions; ++n)
    {
        int region_id = unique_regions[n];
        for (int j = 0; j < N; ++j)
        {
            if (regions[j] == region_id)
                *it++ = j;
        }

        h_region_inds[n + 1] = (it - h_regions.begin());
        sizes.push_back(h_region_inds[n + 1] - h_region_inds[n]);
    }
    associated_ids = unique_regions;
}

MouseBrain2D::MouseBrain2D(std::string const& bmp_scan_)
    : bmp_scan{ "../" + bmp_scan_ }
{}

unsigned char* ReadBMP(std::string const& filename, int& width, int& height)
{
    int i;
    FILE* f = fopen(filename.c_str(), "rb");
    unsigned char info[24];

    // read the 24 bytes
    fread(info, sizeof(unsigned char), 24, f);

    unsigned int start_idx = *(std::uint32_t*)&info[10];

    // extract image height and width from header
    width = *(std::uint16_t*)&info[18];
    height = *(std::uint16_t*)&info[22];

    // skip to start of data
    fseek(f, start_idx, SEEK_SET);

    // allocate 4 bytes per pixel
    int size = 4 * width * height;
    unsigned char* data = new unsigned char[size];

    // read the rest of the data at once
    fread(data, sizeof(unsigned char), size, f);
    fclose(f);
    return data;
}

bool MouseBrain2D::Generate(int const N)
{
    std::cout << "Generating 2D Mouse Brain Morphology..." << std::endl;
    int width, height;
    unsigned char* grid = ReadBMP(bmp_scan, width, height);
    float aspect{ width / (float)height };

    // Generate points uniformly in square [-aspect, aspect] X [-1, 1]
    // and only keep those for which BMP value is black.
    auto& gen{ Utility::UnseededRandomEngine() };
    std::uniform_real_distribution<float> x_dist(-aspect, aspect), y_dist(-1, 1);
    positions.resize(N);
    int pos_size = 0;
    while (pos_size < N)
    {
        float const x{ x_dist(gen) }, y{ y_dist(gen) };
        glm::vec3 p{ x, y, 0.0f };
        int const i = width * (x + aspect) / (2 * aspect), 
                  j = height * (y + 1) / 2;

        // Check if pixel is black. Index of 4 byte pixel data is 4 * (j * width + i.
        // First byte is alpha, so skip to second byte (hence the +1).
        if (grid[4 * (j * width + i) + 1] == 0)
        {
            positions[pos_size] = p;
            pos_size += 1;
        }
    }
    return true;
}
