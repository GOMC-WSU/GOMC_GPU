/*******************************************************************************
GPU OPTIMIZED MONTE CARLO (GOMC) BETA 0.97 (GPU version)
Copyright (C) 2015  GOMC Group

A copy of the GNU General Public License can be found in the COPYRIGHT.txt
along with this program, also can be found at <http://www.gnu.org/licenses/>.
********************************************************************************/

#ifndef CALCULATEENERGY_H
#define CALCULATEENERGY_H

#include "../lib/BasicTypes.h"
#include "EnergyTypes.h"
#include "TransformMatrix.h"
#include <cuda_runtime.h> 
#include <cuda.h>
#include <vector>
//
//    CalculateEnergy.h
//    Energy Calculation functions for Monte Carlo simulation
//    Calculates using const references to a particular Simulation's members
//    Brock Jackman Sep. 2013
//    
//    Updated to use radial-based intermolecular pressure
//    Jason Mick    Feb. 2014
//

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
      void CalculateEnergy::ParticleNonbonded(double* energy,
                                        const cbmc::TrialMol& trialMol,
                                        XYZArray const& trialPos,
                                        const uint partIndex,
                                        const uint box,
                                        const uint trials) const;

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
      Intermolecular CalculateEnergy::MoleculeTailChange(const uint box,
                                                   const uint kind,
                                                   const bool add) const;

      //Calculates intramolecular energy of a full molecule
      void CalculateEnergy::MoleculeIntra(double& bondEn,
                                    double& nonBondEn,
                                    const uint molIndex,
                                    const uint box) const;


	    // GPU data


	  uint AtomCount[BOX_TOTAL];

	  uint MolCount [BOX_TOTAL];


   // forcefield data
   // particles 
 //  double* Gpu_mass;

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




   double * Gpu_sigmaSq, * Gpu_epsilon_cn, * Gpu_epsilon_cn_6, * Gpu_nOver6, 
	   * Gpu_enCorrection, * Gpu_virCorrection;
  
   double * Gpu_x, *Gpu_y, *Gpu_z;

   
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

	SystemPotential NewSystemInterGPU(BoxDimensions &newDim,uint src,uint dist); 
	SystemPotential NewSystemInterGPUOneBox(BoxDimensions &newDim, uint bPick); 

	void GetParticleEnergyGPU(uint box, double * en,XYZArray positions, int numAtoms, int mOff, int CurrentPos, int MolKind);


	const Forcefield& forcefield;
	const Molecules& mols;
	const Coordinates& currentCoords;
	const MoleculeLookup& molLookup;
	const BoxDimensions& currentAxes;
	const COM& currentCOM;

	//Internal storage for virial
	/*double * molInter;
	double oldInter;*/

   private:
      //Calculates intramolecular energy for all molecules in the system
      void SystemIntra(SystemPotential& pot, const Coordinates& coords, 
            const BoxDimensions& boxDims) const; 

      //Calculates full TC for current system
      void FullTailCorrection(SystemPotential& pot, 
            const BoxDimensions& boxAxes) const;



      //Calculates bond vectors of a full molecule, stores them in vecs
      void CalculateEnergy::BondVectors(XYZArray & vecs,
                                  MoleculeKind const& molKind,
                                  const uint molIndex,
                                  const uint box) const;

      //Calculates bond stretch intramolecular energy of a full molecule
       void CalculateEnergy::MolBond(double & energy,
                              MoleculeKind const& molKind,
                              XYZArray const& vecs,
                              const uint box) const;

      //Calculates angular bend intramolecular energy of a full molecule
      void CalculateEnergy::MolAngle(double & energy,
                               MoleculeKind const& molKind,
                               XYZArray const& vecs,
                               const uint box) const;

      //Calculates dihedral torsion intramolecular energy of a full molecule
      void CalculateEnergy::MolDihedral(double & energy,
                                  MoleculeKind const& molKind,
                                  XYZArray const& vecs,
                                  const uint box) const;


      //Calculates Nonbonded intramolecule energy of a full molecule
      void CalculateEnergy::MolNonbond(double & energy,
                                 MoleculeKind const& molKind,
                                 const uint molIndex,
                                 const uint box) const;

     bool SameMolecule(const uint p1, const uint p2) const
      { return (particleMol[p1] == particleMol[p2]); }
};

// GPU headers
__global__ void Gpu_CalculateSystemInter(
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
	//double dev_VirialContrib[],
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



#endif /*ENERGY_H*/


