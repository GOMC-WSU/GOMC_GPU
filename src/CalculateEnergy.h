
#ifndef CALCULATEENERGY_H
#define CALCULATEENERGY_H

#include "../lib/BasicTypes.h"
#include "EnergyTypes.h"
#include "TransformMatrix.h"
#include <cuda_runtime.h> 
#include <cuda.h>
#include <vector>


#define CELL_LIST




class StaticVals;
class System;
class Forcefield;
class Molecules;
class MoleculeLookup;
class MoleculeKind;
class Coordinates;
class COM;
class XYZArray;
class BoxDimensions;

namespace cbmc { class TrialMol; }


static const int MAXTHREADSPERBLOCK = 128; 
static const int MAXTBLOCKS = 65535; 



class CalculateEnergy 
{
public:
	CalculateEnergy(const StaticVals& stat, const System& sys);

	void Init();

	//! Calculates total energy/virial of all boxes in the system
	SystemPotential SystemTotal() ;

	//! Calculates intermolecule energy of all boxes in the system
	//! @param coords Particle coordinates to evaluate for
	//! @param boxAxes Box Dimenions to evaluate in
	//! @return System potential assuming no molecule changes
	SystemPotential SystemInter(SystemPotential potential,
		const XYZArray& coords, 
		XYZArray const& com,
		const BoxDimensions& boxAxes) const;

	/////////////////////////////////////////////////////////////////////
	//TODO_BEGIN: wrap this in GEMC_NPT ensemble tags
	/////////////////////////////////////////////////////////////////////
	//! Calculates total energy/virial of a single box in the system
	SystemPotential BoxNonbonded(SystemPotential potential,
		const uint box,
		const XYZArray& coords, 
		XYZArray const& com,
		const BoxDimensions& boxAxes) const;
	/////////////////////////////////////////////////////////////////////
	//TODO_END: wrap this in GEMC_NPT ensemble tags
	/////////////////////////////////////////////////////////////////////

	//! Calculates intermolecule energy of all boxes in the system
	//! @param coords Particle coordinates to evaluate for
	//! @param boxAxes Box Dimenions to evaluate in
	//! @return System potential assuming no molecule changes
	SystemPotential SystemNonbonded(SystemPotential potential,
		const XYZArray& coords, 
		XYZArray const& com,
		const BoxDimensions& boxAxes) const;


	//! Calculates intermolecular energy of a molecule were it at molCoords
	//! @param molCoords Molecule coordinates
	//! @param molIndex Index of molecule.
	//! @param box Index of box molecule is in. 
	//! @return
	Intermolecular MoleculeInter(const XYZArray& molCoords,
		uint molIndex, uint box,
		XYZ const*const newCOM = NULL) const;

	//checks the intermolecule energy, for debugging purposes
	double CheckMoleculeInter(uint molIndex, uint box) const;


	//! Calculates Nonbonded intra energy for candidate positions
	//! @param trialMol Partially built trial molecule.
	//! @param partIndex Index of particle within the molecule
	//! @param trialPos Contains exactly n potential particle positions
	//! @param energy Return array, must be pre-allocated to size n
	//! @param box Index of box molecule is in
	void ParticleNonbonded(double* energy,
		const cbmc::TrialMol& trialMol,
		XYZArray const& trialPos,
		const uint partIndex,
		const uint box,
		const uint trials) const;

	void ParticleNonbonded_1_4(double* energy,
		cbmc::TrialMol const& trialMol,
		XYZArray const& trialPos,
		const uint partIndex,
		const uint box,
		const uint trials) const;

	void MolNonbond_1_4(double & energy,
		MoleculeKind const& molKind,
		const uint molIndex,
		const uint box) const;

	//! Calculates Nonbonded intra energy for single particle
	//! @return Energy of all 1-4 pairs particle is in
	//! @param trialMol Partially built trial molecule.
	//! @param partIndex Index of particle within the molecule
	//! @param trialPos Position of candidate particle
	//! @param box Index of box molecule is in
	/* double ParticleNonbonded(const cbmc::TrialMol& trialMol, uint partIndex,
	XYZ& trialPos, uint box) const;*/

	/*
	void ParticleInterCache(double* en, XYZ * virCache, const uint partIndex,
	const uint molIndex, XYZArray const& trialPos,
	const uint box) const;
	*/

	//! Calculates Nonbonded intra energy for candidate positions
	//! @param partIndex Index of the particle within the molecule
	//! @param trialPos Array of candidate positions
	//! @param energy Output Array, at least the size of trialpos
	//! @param molIndex Index of molecule
	//! @param box Index of box molecule is in
	void ParticleInter(const uint partIndex, const XYZArray& trialPos,
		double* energy, const uint molIndex, const uint box) const;



	Intermolecular MoleculeInter(const uint molIndex, const uint box) const;

	double EvalCachedVir(XYZ const*const virCache, XYZ const& newCOM,
		const uint molIndex, const uint box) const;

	double MoleculeVirial(const uint molIndex, const uint box) const;


	//! Calculates the change in the TC from adding numChange atoms of a kind
	//! @param box Index of box under consideration
	//! @param kind Kind of particle being added or removed
	//! @param add If removed: false (sign=-1); if added: true (sign=+1)
	Intermolecular MoleculeTailChange(const uint box,
		const uint kind,
		const bool add) const;

	//Calculates intramolecular energy of a full molecule
	void MoleculeIntra(double& bondEn,
		double& nonBondEn,
		const uint molIndex,
		const uint box) const;
	double * PairEnergy;
	uint * atomsMoleculeNo; 
	// GPU data


	uint AtomCount[BOX_TOTAL];

	uint MolCount [BOX_TOTAL];
	// GPU micro cell list

	SystemPotential SystemInterGPU_CellList();
	double EdgeAdjust[BOX_TOTAL*3]; 
	int CellsPerDim[BOX_TOTAL*3]; 
	int CellDim[BOX_TOTAL*3];
	uint * atomCountrs;
	uint * atomCells;
	dim3 BlockSize;
	uint TotalCellsPerBox[BOX_TOTAL];


	// conv cell list

	// conv cell list data and methods
	int TotalNumberOfCells[BOX_TOTAL] ;
	int AdjacencyCellList_size[BOX_TOTAL];
	int NumberOfCells[BOX_TOTAL];
	double CellSize[BOX_TOTAL]; 
	SystemPotential CalculateEnergyCellList();
	SystemPotential CalculateNewEnergyCellList(BoxDimensions &newDim,SystemPotential curpot, int step);
	
	double * dev_EnergyContribCELL_LIST;
	double * dev_VirialContribCELL_LIST;


	double * dev_partEnergy; 
	double * trialPosX,* trialPosY,* trialPosZ; 

	int MaxTrialNumber;

	double * FinalEnergyNVirial;

	int *dev_AdjacencyCellList0;

	uint *dev_CountAtomsInCell0;

	int *AtomsInCells0;

#if ENSEMBLE == GEMC
	uint *dev_CountAtomsInCell1;

	int *AtomsInCells1;
	int *dev_AdjacencyCellList1;
#endif

#ifdef MIE_INT_ONLY
	uint* Gpu_partn;
#else
	double *Gpu_partn;
#endif


	// to be used at mol transfer 
	double* tmpx,*tmpy,*tmpz;
	double *tmpCOMx, *tmpCOMy, *tmpCOMz;
	uint *atmsPerMol;
	uint * CPU_atomKinds;
	uint * tmpMolStart;
	uint * CPU_atomsMoleculeNo;

	double * cordsx;
	double * cordsy;
	double * cordsz;

	double * Gpu_sigmaSq, * Gpu_epsilon_cn, * Gpu_epsilon_cn_6, * Gpu_nOver6, 
		* Gpu_enCorrection, * Gpu_virCorrection;

	double * Gpu_x, *Gpu_y, *Gpu_z;

	double * dev_EnergyContrib, * dev_VirialContrib;

	double * tempCoordsX;
	double * tempCoordsY;
	double * tempCoordsZ;

	uint* Gpu_start;
	uint* Gpu_kIndex;

	uint* Gpu_countByKind;


	double* Gpu_pairEnCorrections;
	double* Gpu_pairVirCorrections;


	double *Gpu_COMX;
	double *Gpu_COMY;
	double *Gpu_COMZ;



	SystemPotential *Gpu_Potential;


	uint * Gpu_atomKinds; 


	uint * NoOfAtomsPerMol;


	bool *Gpu_result;

	cudaStream_t stream0,stream1;

	std::vector<int> particleKind;
	std::vector<int> particleMol;


	double *newCOMX, *newCOMY, *newCOMZ;
	double *newX,*newY,*newZ;

	SystemPotential SystemInterGPU() ;

	SystemPotential NewSystemInterGPU(uint step, BoxDimensions &newDim,uint src,uint dist); 
	SystemPotential NewSystemInterGPUOneBox(BoxDimensions &newDim, uint bPick); 

	void GetParticleEnergyGPU(uint box, double * en,XYZArray positions, int numAtoms, int mOff, int CurrentPos, int MolKind, int nLJTrials);
	void GetParticleEnergyGPU_CellList(uint box, double * en, XYZArray positions, int numAtoms, int mOff, int CurrentPos, int MolKind, int nLJTrials);

	void  GetParticleEnergy(uint box, double * en, XYZArray positions, int numAtoms, int mOff, int CurrentPos, int MolKind, int nLJTrials);// general function

	const Forcefield& forcefield;
	const Molecules& mols;
	const Coordinates& currentCoords;
	const MoleculeLookup& molLookup;
	const BoxDimensions& currentAxes;
	const COM& currentCOM;

private:
	//Calculates intramolecular energy for all molecules in the system
	void SystemIntra(SystemPotential& pot, const Coordinates& coords, 
		const BoxDimensions& boxDims) const; 

	//Calculates full TC for current system
	void FullTailCorrection(SystemPotential& pot, 
		const BoxDimensions& boxAxes) const;



	//Calculates bond vectors of a full molecule, stores them in vecs
	void BondVectors(XYZArray & vecs,
		MoleculeKind const& molKind,
		const uint molIndex,
		const uint box) const;

	//Calculates bond stretch intramolecular energy of a full molecule
	void MolBond(double & energy,
		MoleculeKind const& molKind,
		XYZArray const& vecs,
		const uint box) const;

	//Calculates angular bend intramolecular energy of a full molecule
	void MolAngle(double & energy,
		MoleculeKind const& molKind,
		XYZArray const& vecs,
		const uint box) const;

	//Calculates dihedral torsion intramolecular energy of a full molecule
	void MolDihedral(double & energy,
		MoleculeKind const& molKind,
		XYZArray const& vecs,
		const uint box) const;


	//Calculates Nonbonded intramolecule energy of a full molecule
	void MolNonbond(double & energy,
		MoleculeKind const& molKind,
		const uint molIndex,
		const uint box) const;

	bool SameMolecule(const uint p1, const uint p2) const
	{ return (particleMol[p1] == particleMol[p2]); }
};

// GPU headers
__global__ void Gpu_CalculateSystemInter( uint step,
	uint * NoOfAtomsPerMol,
	uint *AtomKinds,
	uint *molKindIndex,
	double * sigmaSq,
	double * epsilon_cn,
	double * nOver6,
	double * epsilon_cn_6,
	double *x,
	double *y,
	double *z,
	double * Gpu_COMX,
	double * Gpu_COMY,
	double * Gpu_COMZ,
	uint * Gpu_start,
	double xAxis,
	double yAxis,
	double zAxis,
	double xHalfAxis,
	double yHalfAxis,
	double zHalfAxis,
	uint boxOffset,
	uint MoleculeCount,
	uint HalfMoleculeCount,
	uint FFParticleKindCount,
	double rCut,
	uint isEvenMolCount,
	double rCutSq,
	double dev_EnergyContrib[],
	double dev_VirialContrib[],
	uint limit,
	uint MolOffset,

#ifdef MIE_INT_ONLY
	const uint * n,
#else
	const double * n
#endif

	);

__global__ void Gpu_CalculateParticleInter(int trial,
	uint *molKindIndex,
	double * sigmaSq,
	double * epsilon_cn,
	double * nOver6,
	double * epsilon_cn_6,
	double *Gpu_x,
	double *Gpu_y,
	double *Gpu_z,
	double nX,
	double nY,
	double nZ,
	uint * Gpu_atomKinds,
	int len, // mol length
	uint MolId, // mol ID of the particle we are testing now
	uint AtomToProcess,
	uint * Gpu_start,
	double xAxis,
	double yAxis,
	double zAxis,
	double xHalfAxis,
	double yHalfAxis,
	double zHalfAxis,
	uint boxOffset,
	uint AtomCount, // atom count in the current box
	uint FFParticleKindCount,
	double rCut,
	uint molKInd,// mol kind of the tested atom
	double rCutSq,
	double dev_EnergyContrib[],

#ifdef MIE_INT_ONLY
	const uint * n,
#else
	const double * n
#endif
	);



__global__ void TryTransformGpu(uint * NoOfAtomsPerMol, uint *AtomKinds,  SystemPotential * Gpu_Potential ,  double * Gpu_x, double * Gpu_y, double * Gpu_z,
	double * Gpu_COMX, double * Gpu_COMY, double * Gpu_COMZ,

	XYZ shift, double xAxis, double yAxis, double zAxis,
	uint *molKindIndex,
	double * sigmaSq,
	double * epsilon_cn,
	double * nOver6,
	double * epsilon_cn_6,
	double beta,
	double AcceptRand,
	uint * Gpu_start,
	int len,
	double xHalfAxis,
	double yHalfAxis,
	double zHalfAxis,
	uint boxOffset,
	uint MoleculeCount,
	uint mIndex,// mol index with offset
	uint FFParticleKindCount,
	double rCut,
	uint molKInd,
	double rCutSq,
	double dev_EnergyContrib[],
	double dev_VirialContrib[],
	uint boxIndex,
	bool * Gpu_result,
#ifdef MIE_INT_ONLY
	const uint * n,
#else
	const double * n
#endif

	);
__device__ bool InRcutGpuSigned(
	double &distSq,
	double *x,
	double * y,
	double *z,
	double xAxis,
	double yAxis,
	double zAxis,
	double xHalfAxis,
	double yHalfAxis,
	double zHalfAxis,
	const uint i,
	const uint j,
	double rCut,
	double rCutSq, XYZ & dist
	) ;


__device__ bool InRcutGpuSigned(
	double &distSq,
	double *x,
	double *y,
	double *z,
	const double xi,
	const double yi,
	const double zi,
	double xAxis,
	double yAxis,
	double zAxis,
	double xHalfAxis,
	double yHalfAxis,
	double zHalfAxis,
	const uint i,
	const uint j,
	double rCut,
	double rCutSq, XYZ & dist
	);



__device__ void CalcAddGpu(
	double& en,
	double& vir,
	const double distSq,
	const uint kind1,
	const uint kind2,
	uint count,
	double * sigmaSq,
	double * epsilon_cn,
	double * nOver6,
	double * epsilon_cn_6,
#ifdef MIE_INT_ONLY
	const uint * n,
#else
	const double * n
#endif

	) ;


__device__ void CalcAddGpu(
	double& en,
	double& vir,
	const double distSq,
	const uint kind1,
	const uint kind2,
	const uint count,
	const double * sigmaSq,
	const double * epsilon_cn,
	const double * nOver6,
	const double * epsilon_cn_6,
#ifdef MIE_INT_ONLY
	const uint * n,
#else
	const double * n
#endif

	) ;

__device__ double MinImageSignedGpu(double raw,double ax, double halfAx) ;


__global__ void TryRotateGpu( uint * NoOfAtomsPerMol, uint *AtomKinds, SystemPotential * Gpu_Potential , TransformMatrix  matrix,   double * Gpu_x, double * Gpu_y, double * Gpu_z,
	double * Gpu_COMX, double * Gpu_COMY, double * Gpu_COMZ,

	double xAxis, double yAxis, double zAxis,
	uint *molKindIndex,
	double * sigmaSq,
	double * epsilon_cn,
	double * nOver6,
	double * epsilon_cn_6,
	double beta,
	double AcceptRand,
	uint * Gpu_start,
	int len,
	double xHalfAxis,
	double yHalfAxis,
	double zHalfAxis,
	uint boxOffset,

	uint MoleculeCount,
	uint mIndex,// mol index with offset
	uint FFParticleKindCount,
	double rCut,
	uint molKInd,
	double rCutSq,

	double dev_EnergyContrib[],
	double dev_VirialContrib[],
	uint boxIndex,
	bool * Gpu_result,
#ifdef MIE_INT_ONLY
	const uint * n,
#else
	const double * n
#endif



	);


__global__ void ScaleMolecules(uint * noOfAtomsPerMol ,
	uint *molKindIndex, double * Gpu_x, double * Gpu_y, double * Gpu_z,
	double * Gpu_COMX, double * Gpu_COMY, double * Gpu_COMZ,
	double * Gpu_newx, double * Gpu_newy, double * Gpu_newz,
	double * Gpu_newCOMX, double * Gpu_newCOMY, double * Gpu_newCOMZ,
	double scale, int MolCount,
	double newxAxis,
	double newyAxis,
	double newzAxis,
	double xAxis,
	double yAxis,
	double zAxis,
	double xHalfAxis,
	double yHalfAxis,
	double zHalfAxis,
	uint boxOffset,
	uint * Gpu_start
	);

// Function to run the Transform move using celllist
__global__ void TryTransformGpuCellList(
	double *tempCoordsX,
	double *tempCoordsY,
	double *tempCoordsZ,
	uint * NoOfAtomsPerMol,
	uint *AtomKinds,
	SystemPotential * Gpu_Potential ,
	double * Gpu_x,
	double * Gpu_y,
	double * Gpu_z,
	double * Gpu_COMX,
	double * Gpu_COMY,
	double * Gpu_COMZ,
	XYZ shift,
	double xAxis,
	double yAxis,
	double zAxis,
	double EdgeXAdjust,
	double EdgeYAdjust,
	double EdgeZAdjust,
	int CellsXDim,
	int CellsYDim,
	int CellsZDim,
	int CellsPerXDimension,
	int CellsPerYDimension,
	int CellsPerZDimension,
	uint cellOffset,
	uint cellrangeOffset, 
	uint NumberofCells,// number of cells in the search box (cellsXDim * cellsYDim* cellsZDim )
	uint*  atomCountrs,
	uint*  atomCells,
	uint NumberOfCellsInBox, 
	uint * atomsMoleculeNo,
	uint *molKindIndex,
	double * sigmaSq,
	double * epsilon_cn,
	double * nOver6,
	double * epsilon_cn_6,
	double beta,
	double AcceptRand,
	uint * Gpu_start,
	int len,
	double xHalfAxis,
	double yHalfAxis,
	double zHalfAxis,
	uint boxOffset,
	uint MoleculeCount,
	uint mIndex,// mol index with offset
	uint FFParticleKindCount,
	double rCut,
	uint molKInd,
	double rCutSq,
	double dev_EnergyContrib[],
	double dev_VirialContrib[],
	uint boxIndex,
	bool * Gpu_result,
#ifdef MIE_INT_ONLY
	const uint * n,
#else
	const double * n
#endif
	);



// Function to run the Rotate move using celllist
__global__ void TryRotateGpuCellList(
	double *tempCoordsX,
	double *tempCoordsY,
	double *tempCoordsZ,
	uint * NoOfAtomsPerMol,
	uint *AtomKinds,
	SystemPotential * Gpu_Potential ,
	double * Gpu_x,
	double * Gpu_y,
	double * Gpu_z,
	double * Gpu_COMX,
	double * Gpu_COMY,
	double * Gpu_COMZ,
	TransformMatrix  matrix,
	double xAxis,
	double yAxis,
	double zAxis,
	double EdgeXAdjust,
	double EdgeYAdjust,
	double EdgeZAdjust,
	int CellsXDim,
	int CellsYDim,
	int CellsZDim,
	int CellsPerXDimension,
	int CellsPerYDimension,
	int CellsPerZDimension,
	uint cellOffset,
	uint cellrangeOffset, 
	uint NumberofCells,// number of cells in the search box (cellsXDim * cellsYDim* cellsZDim )
	uint*  atomCountrs,
	uint*  atomCells,
	uint NumberOfCellsInBox, 
	uint * atomsMoleculeNo,
	uint *molKindIndex,
	double * sigmaSq,
	double * epsilon_cn,
	double * nOver6,
	double * epsilon_cn_6,
	double beta,
	double AcceptRand,
	uint * Gpu_start,
	int len,
	double xHalfAxis,
	double yHalfAxis,
	double zHalfAxis,
	uint boxOffset,
	uint MoleculeCount,
	uint mIndex,// mol index with offset
	uint FFParticleKindCount,
	double rCut,
	uint molKInd,
	double rCutSq,
	double dev_EnergyContrib[],
	double dev_VirialContrib[],
	uint boxIndex,
	bool * Gpu_result,
#ifdef MIE_INT_ONLY
	const uint * n,
#else
	const double * n
#endif
	);


__global__ void CalculateParticleInter_CellList(

	double *n_x,
	double *n_y,
	double *n_z,
	double* x,
	double* y,
	double* z,
	double xAxis,
	double yAxis,
	double zAxis,
	double xHalfAxis,
	double yHalfAxis,
	double zHalfAxis,
	double EdgeXAdjust,
	double EdgeYAdjust,
	double EdgeZAdjust,
	int CellsXDim,
	int CellsYDim,
	int CellsZDim,
	int CellsPerXDimension,
	int CellsPerYDimension,
	int CellsPerZDimension,
	double rCut,
	double rCutSq,
	uint boxOffset,
	uint cellOffset,
	uint cellrangeOffset, 
	uint NumberofCells,// number of cells in the search box (cellsXDim * cellsYDim* cellsZDim )
	const uint*  atomCountrs,
	const uint*  atomCells,
	uint * Gpu_start,
	uint NumberOfCellsInBox, 
	uint * atomsMoleculeNo,
	uint *AtomKinds,
	double * sigmaSq,
	double * epsilon_cn,
	double * nOver6,
	double * epsilon_cn_6,
	uint FFParticleKindCount,
	double dev_EnergyContrib[],
	int MolToProcess,
	int atomToProcess,
#ifdef MIE_INT_ONLY
	const uint * n,
#else
	const double * n
#endif

	);
#endif /*ENERGY_H*/

