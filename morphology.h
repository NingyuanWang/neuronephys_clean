// morphology.h : Handles generation and storage of neuron positions. 
#pragma once
#include "defines.cuh"
#include <ANN/ANN.h>
#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>
#include "regions_tree.h"

//! Morphology is an abstract class that supports morphology generation 
//! and operations such as union, nearest-neighbor, etc.
class Morphology
{
protected:
	std::unique_ptr<ANNkd_tree> kd_tree; //! Tree used by ANN for spatial search.
	std::vector<glm::vec3> positions; //! Positions in GLM format.

	//! Creates kd_tree using positions.
	void SetupANN();

	//! Virtual function to check if point p is on the boundary. By default returns false.
	virtual bool IsBoundaryPoint(glm::vec3 const& p) const { return false; }

public:
	Morphology() : kd_tree{ nullptr } {}

	//! Generates a morphology. Pure virtual function.
	/*!
	  \param [in] N Number of points.
      \return If the function is successful for not.
	*/
    virtual bool Generate (int const N) = 0;

	//! Computes the union of this morphology plus another morphology and stores it in updated positions.
	/*!
	  \param [in] rhs The morphology to union with.
	*/
    void Union (Morphology* rhs); 

	//! Translates the positions and overrites the old positions.
	/*!
	  \param [in] translation Translation vector.
	*/
    void Translate(glm::vec3 const& transform);

	//! Scale the positions and overrites the old positions.
	/*!
	  \param [in] scale Scale vector.
	*/
    void Scale(glm::vec3 const& scale);

	//! Rotates the positions and overrites the old positions.
	/*!
	  \param [in] angle Angle of rotation.
	  \param [in] axis Axis of rotation.
	*/
	void Rotate(float const angle, glm::vec3 const& axis);

	//! Finds the k nearest neighbros to neuron n.
	/*!
	  \param [in] n Index of neuron.
	  \param [in] k Number of nearest neighbors to find, including n.
	  \param [out] knn Indices of nearest neighbors.
	  \param [out] knn_dists Distances to each respective neighbor.
	*/
	void KNN(int const n, int const k, std::vector<int>& knn, std::vector<ANNdist>& knn_dists);

	//! Same as other KNN, but position, not neuron index, is sent.
	void KNN(ANNpoint pos, int const k, std::vector<int>& knn, std::vector<ANNdist>& knn_dists);

	//! Finds the neurons within a ball of neuron n.
	int WithinBall(int const n, Float dist_not_sqr, int max_n_conns, std::vector<int>& ball, std::vector<ANNdist>& ball_dists);

	//! Same as other WithinBall, but position, not neuron index, is sent.
	int WithinBall(ANNpoint pos, Float dist_not_sqr, int max_n_conns, std::vector<int>& ball, std::vector<ANNdist>& ball_dists);

	//! Generate a list of KNN regions at specific neurons.
	/*!
		\param [out] h_regions Column indices for region adjacency list. 
		\param [out] h_region_inds Row offsets for region adjacency list.
		\param [in] electrode_inds Indices of neurons at which to place electrodes.
		\param [in] k Number of nearest neighbors for regions.
	*/	
	void GenSphericalRegions(thrust::host_vector<int>& h_regions, thrust::host_vector<int>& h_region_inds, 
		                     std::vector<int> const& electrode_inds, int const k);

	//! Same as other GenSphericalRegions but positions, not neuron indices, are passed.
	void GenSphericalRegions(thrust::host_vector<int>& h_regions, thrust::host_vector<int>& h_region_inds, 
		                     std::vector<glm::vec3> const& electrode_positions, int const k);

	//! Function for getting the boundary points of a morphology using on IsBoundaryPoint function.
	/*!
		\param [out] h_regions Column indices for region adjacency list. 
		\param [out] h_region_inds Row offsets for region adjacency list.
	*/	
	void GetBoundaryRegion(thrust::host_vector<int>& h_regions, thrust::host_vector<int>& h_region_inds) const;

	//! Creates a vector of binary voxels that are true if they contain a point and false otherwise.
	//! To access (i,j,k) voxel, index into vector is i+dims[0]*(j+dims[1]*k).
	/*!
		\param [in] dims Dimensions of voxel grid in X, Y, Z, respectively.
	*/	
	virtual std::vector<bool> ToBinaryVoxels(glm::uvec3 const& dims) const;

	//! Find the minimum distance between any two adjacent points.
	float GetMinimumDistance();
	
	//! Project neurons to a plane in some dimension. 
	//! \param [in] dim Dimension to project. Defaults to 0 (X dimension).
	void ProjectNeuronsToPlane(int dim = 0);

	//! Function for generating custom regions based on a functor defined by the user operating on individual points.
	/*!
		\param [in] selector Functor defined by user that operates on individual points and assigns them to a region index. If region index is -1, they are not put in any region.
		\param [in] N_regions Number of regions defined by user.
	*/	
	template<typename RegionSelector>
	void GenCustomRegions(RegionSelector& selector, int N_regions, thrust::host_vector<int>& h_regions, thrust::host_vector<int>& h_region_inds) const
    {
        std::vector<std::vector<int>> regions_raw(N_regions);
        for (int n = 0; n < positions.size(); ++n)
        {
            auto const& pos{ positions[n] };
            int idx = selector(pos);
            if (idx >= 0)
                regions_raw[idx].push_back(n);
        }

        // Transfer info from regions_raw to h_regions and h_Regions_inds.
        h_region_inds.resize(N_regions + 1);
        for (int i = 0; i < N_regions; ++i)
        {
            h_region_inds[i + 1] = h_region_inds[i] + regions_raw.size();
            int cur_sz = h_regions.size();
            h_regions.resize(cur_sz + regions_raw.size());
            for (int j = 0; j < regions_raw.size(); ++j)
                h_regions[cur_sz + j] = regions_raw[i][j];
        }
    }

	//! Getter for positions. 
	std::vector<glm::vec3> const& GetPositions() { return positions; }
	float* GetPositionsRaw() { return (float*)positions.data(); }

	virtual ~Morphology() { annClose(); }
};

//! Hollow Sphere shell. 2D Manifold.
class HollowSphere : public Morphology
{
public:
    virtual bool Generate(int const N) override;
};

//! 2D Square [-1,1] X [-1,1]. 2D Manifold.
class UnitSquare : public Morphology
{
public:
    virtual bool Generate(int const N) override;
};

//! 3D Square [-1,1] X [-1,1] X [-1,1]. 
class UnitCube : public Morphology
{
public:
	virtual bool Generate(int const N) override;
};

//! Volume given by the solution of the equations F(x,y,z) <= 0.
//! NOTE THAT WE ASSUME THE SOLID IS WITHIN THE UNIT CUBE!
class ImplicitSolid : public Morphology
{
protected:
	//! Defines the inside, outside and boundary by F(x,y,z) < 0, > 0 and = 0, respectively.
	virtual Float F(glm::vec3 const& p) const = 0;
	virtual bool IsBoundaryPoint(glm::vec3 const& p) const override;
public:
	virtual bool Generate(int const N) override;
};

//! Solid unit sphere implemented as an implicit solid.
class SolidSphere : public ImplicitSolid
{
protected:
	//! Returns L2 norm squared minus 1.
	virtual Float F(glm::vec3 const& p) const override;
};

//! Sphere with "wrinkles" that are polar oscillations in the radius.
class SimpleWrinkledSolidSphere : public ImplicitSolid
{
protected:
	//! Returns L2 norm squared minus 1 plus some variation for the "wrinkles.
	virtual Float F(glm::vec3 const& p) const override;
};

//! Morphology read in from a file with format inferred in constructor.
class FileMorphology : public Morphology
{
protected:	
	std::string fl_name;	
	std::string format;
	std::vector<int> unique_regions;
	std::vector<int> regions;
	bool read_regions;
public:
	/*!
	  \param [in] fl_name_ Name of file to read from.
	  \param [in] read_regions_ Defaults to false. If true, read in region ids per neuron.
	*/
	FileMorphology(std::string const& fl_name_, bool read_regions_=false);
	bool GenerateFromHDF(int const N);
	virtual bool Generate(int const N) override;

	//! Replace all regions with parent region at specified depth in Allen atlas region tree. 
	/*!
	  \param [in] depth_in_tree This specifies the depth of the subregion in the Allen atlas regions which are stored as tree. By default, leaf regions are used.
	  \param [in] regions_tree RegionsTree class which represents the heirarchy of regions for the Allen Atlas regions. See: RegionsTree.
	*/
	void SpecfiyRegionsDepth(int const depth_in_tree, RegionsTree const& regions_tree);

	//! Generate a list of regions defined by the region IDs in regions variable, read from file. 
	/*!
		\param [out] h_regions Column indices for region adjacency list. 
		\param [out] h_region_inds Row offsets for region adjacency list.
		\param [out] associated_ids Vector of IDs associated with each of the regions. 
	*/	
	void GenRegionsFromIDs(thrust::host_vector<int>& h_regions, thrust::host_vector<int>& h_region_inds, thrust::host_vector<int>& associated_ids);

	std::vector<int> const& GetRegions() const { return regions; }
};

//! Point cloud provided by user.
class UserPointCloud : public Morphology
{
public:
	UserPointCloud(std::vector<glm::vec3> const& positions_)
		: Morphology()
	{
		positions = positions_;
	}
	virtual bool Generate(int const N) override { return false; } // Should not be called.
};

//! 2D Mouse brain scan generated from a BMP binary image.
class MouseBrain2D : public Morphology
{
protected:	
	std::string bmp_scan;	

public:
	/*!
	  \param [in] bmp_scan_ Name of bmp file to read from.
	*/
	MouseBrain2D(std::string const& bmp_scan_);
	virtual bool Generate(int const N) override;
};
