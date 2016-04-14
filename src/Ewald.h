#ifndef EWALD_H
#define EWALD_H

#include "../lib/BasicTypes.h"
#include "EnergyTypes.h"
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>

//
//    Ewald.h
//    Energy Calculation functions for Ewald summation method
//    Calculating self, correction and reciprocate part of ewald    
//
//    Developed by Y. Li and modified by Mohammad S. Barhaghi
// 
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
class CalculateEnergy;

namespace cbmc { class TrialMol; }
namespace config_setup{ class SystemVals; }

static const int MAXTHREADSPERBLOCK = 128;
static const int MAXTBLOCKS = 65535;

class Ewald 
{
  //friend class CalculateEnergy;
   public:
    Ewald(StaticVals const& stat, System & sys);
    
    ~Ewald();
   void freeGpuData();

   void Init();

   void InitEwald();
   void InitGPU();

   //return size of image with defined Kmax value
   //uint GetImageSize();

   //initiliazie term used for ewald calculation
   void RecipInit(uint box, BoxDimensions const& boxAxes);
   
   //calculate self term for a box
   double BoxSelf(BoxDimensions const& boxAxes, uint box) const;

   //calculate reciprocate term for a box
   double BoxReciprocal(int box, XYZArray const& molCoords);

   //calculate correction term for a molecule
   double MolCorrection(uint molIndex, BoxDimensions const& boxAxes,
			uint box)const;

   //calculate reciprocate term for displacement and rotation move
   double MolReciprocal(XYZArray const& molCoords, const uint molIndex,
			const uint box, XYZ const*const newCOM = NULL) ;	

   double MolReciprocal(const uint pStart, const uint pLen, const uint m,
		   const uint b, const XYZ shift);

   //calculate self term for CBMC algorithm
   void SwapSelf(double *self, uint molIndex, uint partIndex, int box, 
		 uint trials) const;
   
   //calculate correction term for linear molecule CBMC algorithm
   void SwapCorrection(double* energy, const cbmc::TrialMol& trialMol, 
		       XYZArray const& trialPos, const uint partIndex, 
		       const uint box, const uint trials) const; 

   //calculate correction term for branched molecule CBMC algorithm
   void SwapCorrection(double* energy, const cbmc::TrialMol& trialMol,
		       XYZArray *trialPos, const int pickedAtom, 
		       uint *partIndexArray, const uint box, const uint trials,
		       const uint PrevIndex, bool Prev) const;  

   //calculate correction term for old configuration
   double CorrectionOldMol(const cbmc::TrialMol& oldMol, const double distSq,
			   const uint i, const uint j) const;

   //calculate reciprocate term in destination box for swap move
   double SwapDestRecip(cbmc::TrialMol newMol, const uint box);	

   //calculate reciprocate term in source box for swap move
   double SwapSourceRecip(cbmc::TrialMol oldMol, const uint box);

   //back up reciprocate values
   void BackUpRecip( uint box)
   {  
     for (uint i = 0; i < imageSize[box]; i++)
      {
	sumRnew[box][i] = sumRref[box][i];
	sumInew[box][i] = sumIref[box][i];
      }
   }

   //update reciprocate values
   void UpdateRecip(uint box)
   {
     /*
     for (uint i = 0; i < imageSize[box]; i++)
      {
	 sumRref[box][i] = sumRnew[box][i];
	 sumIref[box][i] = sumInew[box][i];
      }
     */
     double *tempR, *tempI;
     tempR = sumRref[box];
     tempI = sumIref[box];
     sumRref[box] = sumRnew[box];
     sumIref[box] = sumInew[box];
     sumRnew[box] = tempR;
     sumInew[box] = tempI;
   }

   void UpdateGPU(const uint pStart, const uint pLen, const uint molIndex, const uint box);
   void RestoreGPU(uint box);

   //GPU global arrays, variables, and streams
   uint molPerBox[BOX_TOTAL], atomPerBox[BOX_TOTAL];
   double *gpu_xCoords, *gpu_yCoords, *gpu_zCoords,
   	   *gpu_xTempCoords, *gpu_yTempCoords, *gpu_zTempCoords;
   uint *molStartAtom, *atomPerMol, *gpu_molStartAtom, *gpu_atomPerMol;
   double *gpu_atomCharge;
   double *gpu_recipEnergy;
   double *gpu_imageReal, *gpu_imageImaginary, *gpu_imageRealRef, *gpu_imageImaginaryRef;
   double *h_kx, *h_ky, *h_kz, *gpu_kx, *gpu_ky, *gpu_kz, *h_prefact, *gpu_prefact;
   cudaStream_t streams[BOX_TOTAL];
   double blockRecipEnergy[2];
   double *gpu_newMolPos, *gpu_oldMolPos;

   private: 
   
   
   const Forcefield& forcefield;
   const Molecules& mols;
   const Coordinates& currentCoords;
   const MoleculeLookup& molLookup;
   const BoxDimensions& currentAxes;
   const COM& currentCOM;

   bool electrostatic, ewald;
   int maxMolLen;
   //int kmax;
   double alpha; 
   double recip_rcut, recip_rcut_Sq;
   int *imageSize, imageTotal;
  
   double **sumRnew; //cosine serries
   double **sumInew; //sine serries
   double **sumRref;
   double **sumIref; 
   double **kx;
   double **ky; 
   double **kz;
   double **prefact;
  

   std::vector<int> particleKind;
   std::vector<int> particleMol;
   std::vector<double> particleCharge;
   
};

//GPU function declaration
__global__ void BoxReciprocalGPU(
		const uint atomPerBox, const uint atomOffset,
		double *gpu_xCoords, double *gpu_yCoords,
		double *gpu_zCoords, double *gpu_atomCharge,
		double *gpu_imageReal, double *gpu_imageImaginary,
		double *gpu_kx, double *gpu_ky, double *gpu_kz,
		double *gpu_prefact, const uint box, const uint imageOffset,
		double *gpu_blockRecipEnergy, const uint imageSize);

__global__ void MolReciprocalGPU(
		const uint pStart, const uint pLen, const uint molIndex,
		const XYZ shift, double *gpu_xCoords, double *gpu_yCoords,
		double *gpu_zCoords, double *gpu_atomCharge, double *gpu_imageReal,
		double *gpu_imageImaginary, double *gpu_imageRealRef,
		double *gpu_imageImaginaryRef, double *gpu_kx, double *gpu_ky,
		double *gpu_kz, double *gpu_newMolPos, double *gpu_prefact,
		const uint box, const uint imageOffset, double* gpu_blockRecipNew,
		const uint imageSize, const double xAxis, const double yAxis,
		const double zAxis, const int maxMolLen);

__global__ void AcceptUpdateGPU(
		const uint pStart, const uint pLen, const uint molIndex,
		double *gpu_xCoords, double *gpu_yCoords,
		double *gpu_zCoords, double *gpu_newMolPos,
		const int maxMolLen, const int box);


#endif /*EWALD_H*/
