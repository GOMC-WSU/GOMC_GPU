#include "Ewald.h"
//#include "CalculateEnergy.h"
//#include "EnergyTypes.h"            //Energy structs
#include "EnsemblePreprocessor.h"   //Flags
//#include "../lib/BasicTypes.h"             //uint
#include "System.h"                 //For init
#include "StaticVals.h"             //For init
#include "Forcefield.h"             //
#include "MoleculeLookup.h"
#include "MoleculeKind.h"
#include "Coordinates.h"
#include "BoxDimensions.h"
#include "cbmc/TrialMol.h"
#include "../lib/GeomLib.h"
#include "../lib/NumLib.h"
#include <cassert>

//
//    Ewald.h
//    Energy Calculation functions for Ewald summation method
//    Calculating self, correction and reciprocate part of ewald    
//
//    Developed by Yuanzhe. Li and modified by Mohammad S. Barhaghi
// 
//	  Ewald.cu
//    GPU implementation of Ewald.cpp & Ewald.h
//
//	  Developed by Yuanzhe. Li
//

//#define debug

using namespace geom;
__device__ unsigned int BlocksDone = 0;

Ewald::Ewald(StaticVals const& stat, System & sys) :
   forcefield(stat.forcefield), mols(stat.mol), currentCoords(sys.coordinates),
   currentCOM(sys.com), ewald(false), maxMolLen(0),
#ifdef VARIABLE_PARTICLE_NUMBER
   molLookup(sys.molLookup),
#else
   molLookup(stat.molLookup),
#endif
#ifdef VARIABLE_VOLUME
   currentAxes(sys.boxDimensions)
#else
   currentAxes(stat.boxDimensions)
#endif
{ }



Ewald::~Ewald()
{
   for (uint b = 0; b < BOX_TOTAL; b++)
   {
      if (kx[b] != NULL)
      {
    	  delete[] kx[b];
    	  delete[] ky[b];
    	  delete[] kz[b];
    	  delete[] prefact[b];
    	  delete[] sumRnew[b];
    	  delete[] sumInew[b];
    	  delete[] sumRref[b];
    	  delete[] sumIref[b];
      }
   }
   if (kx != NULL)
   {
      delete[] kx;
      delete[] ky;
      delete[] kz;
      delete[] prefact;
      delete[] sumRnew;
      delete[] sumInew;
      delete[] sumRref;
      delete[] sumIref;
   }
   freeGpuData();
}

void Ewald::Init()
{
   for(uint m = 0; m < mols.count; ++m)
   {
      const MoleculeKind& molKind = mols.GetKind(m);
      if (maxMolLen < molKind.NumAtoms())
    	  maxMolLen = molKind.NumAtoms();
      for(uint a = 0; a < molKind.NumAtoms(); ++a)
      {
         particleKind.push_back(molKind.AtomKind(a));
         particleMol.push_back(m);
         particleCharge.push_back(molKind.AtomCharge(a));
      }
   }
   electrostatic = forcefield.electrostatic;
   ewald = forcefield.ewald; 

   if (ewald)
   {
     alpha = forcefield.alpha;
     recip_rcut = forcefield.recip_rcut;
     recip_rcut_Sq = recip_rcut * recip_rcut;
     InitEwald();
     RecipInit(0, currentAxes);
     RecipInit(1, currentAxes);
     InitGPU();
   }
}


void Ewald::InitEwald()
{   
   if (ewald)
   {
     //get size of image using defined Kmax
     //const uint imageTotal = GetImageSize();
     const uint imageTotal = 100000;
     //imageSize = imageTotal;   
     //allocate memory
     //allocate memory
     imageSize = new int[BOX_TOTAL];
     sumRnew = new double*[BOX_TOTAL];
     sumInew = new double*[BOX_TOTAL];
     sumRref = new double*[BOX_TOTAL];
     sumIref = new double*[BOX_TOTAL];
     kx = new double*[BOX_TOTAL];
     ky = new double*[BOX_TOTAL];
     kz = new double*[BOX_TOTAL];
     prefact = new double*[BOX_TOTAL];

     for (uint b = 0; b < BOX_TOTAL; b++)
     {
        kx[b] = new double[imageTotal];
        ky[b] = new double[imageTotal];
		kz[b] = new double[imageTotal];
		prefact[b] = new double[imageTotal];
		sumRnew[b] = new double[imageTotal];
		sumInew[b] = new double[imageTotal];
		sumRref[b] = new double[imageTotal];
		sumIref[b] = new double[imageTotal];
     }

   }       
}

void Ewald::InitGPU()
{
	if (ewald)
	{
		//allocate host variables
		atomPerMol = new uint[mols.count];
		molStartAtom = new uint[mols.count];
		imageTotal = imageSize[0] + imageSize[1];
		h_prefact = new double[imageTotal];
		h_kx = new double[imageTotal];
		h_ky = new double[imageTotal];
		h_kz = new double[imageTotal];
		std::cout << "host variables initialized." << std::endl;
		//check the correctness of cudaMalloc
		cudaError_t error;
		error = cudaMalloc((void**)&gpu_atomCharge, sizeof(double) * particleCharge.size());
		if (error != cudaSuccess)
		{
			std::cout << "cudaMalloc gpu_atomCharge returned error " <<
					cudaGetErrorString(error) << "code: " << error << std::endl;
			exit(EXIT_FAILURE);
		}

		error = cudaMalloc((void**)&gpu_xCoords, sizeof(double) * currentCoords.Count());
		error = cudaMalloc((void**)&gpu_yCoords, sizeof(double) * currentCoords.Count());
		error = cudaMalloc((void**)&gpu_zCoords, sizeof(double) * currentCoords.Count());
		error = cudaMalloc((void**)&gpu_xCoordsNew, sizeof(double) * currentCoords.Count());
		error = cudaMalloc((void**)&gpu_yCoordsNew, sizeof(double) * currentCoords.Count());
		error = cudaMalloc((void**)&gpu_zCoordsNew, sizeof(double) * currentCoords.Count());

		error = cudaMalloc((void**)&gpu_molStartAtom, sizeof(int) * mols.count);
		if (error != cudaSuccess)
		{
			std::cout << "cudaMalloc gpu_molStartAtom returned error " <<
					cudaGetErrorString(error) << "code: " << error << std::endl;
			exit(EXIT_FAILURE);
		}

		error = cudaMalloc((void**)&gpu_atomPerMol, sizeof(uint) * mols.count);
		if (error != cudaSuccess)
		{
			std::cout << "cudaMalloc gpu_atomPerMol returned error " <<
					cudaGetErrorString(error) << "code: " << error << std::endl;
			exit(EXIT_FAILURE);
		}

		error = cudaMalloc((void**)&gpu_recipEnergy, sizeof(double) * BOX_TOTAL);
		if (error != cudaSuccess)
		{
			std::cout << "cudaMalloc gpu_recipEnergy returned error " <<
					cudaGetErrorString(error) << "code: " << error << std::endl;
			exit(EXIT_FAILURE);
		}

		error = cudaMalloc((void**)&gpu_prefact, sizeof(double) * imageTotal);
		error = cudaMalloc((void**)&gpu_kx, sizeof(double) * imageTotal);
		error = cudaMalloc((void**)&gpu_ky, sizeof(double) * imageTotal);
		error = cudaMalloc((void**)&gpu_kz, sizeof(double) * imageTotal);
		error = cudaMalloc((void**)&gpu_prefactNew, sizeof(double) * imageTotal);
		error = cudaMalloc((void**)&gpu_kxNew, sizeof(double) * imageTotal);
		error = cudaMalloc((void**)&gpu_kyNew, sizeof(double) * imageTotal);
		error = cudaMalloc((void**)&gpu_kzNew, sizeof(double) * imageTotal);

		error = cudaMalloc((void**)&gpu_imageReal, sizeof(double) * imageTotal);
		if (error != cudaSuccess)
		{
			std::cout << "cudaMalloc gpu_imageReal returned error " <<
					cudaGetErrorString(error) << "code: " << error << std::endl;
			exit(EXIT_FAILURE);
		}

		error = cudaMalloc((void**)&gpu_imageImaginary, sizeof(double) * imageTotal);
		if (error != cudaSuccess)
		{
			std::cout << "cudaMalloc gpu_imageImaginary returned error " <<
					cudaGetErrorString(error) << "code: " << error << std::endl;
			exit(EXIT_FAILURE);
		}

		error = cudaMalloc((void**)&gpu_imageRealRef, sizeof(double) * imageTotal);
		if (error != cudaSuccess)
		{
			std::cout << "cudaMalloc gpu_imageRealRef returned error " <<
					cudaGetErrorString(error) << "code: " << error << std::endl;
			exit(EXIT_FAILURE);
		}

		error = cudaMalloc((void**)&gpu_imageImaginaryRef, sizeof(double) * imageTotal);
		if (error != cudaSuccess)
		{
			std::cout << "cudaMalloc gpu_imageImaginaryRef returned error " <<
					cudaGetErrorString(error) << "code: " << error << std::endl;
			exit(EXIT_FAILURE);
		}

		//maxMolLen is the max mol length in system, 3 is for x, y, z coordinates.
		error = cudaMalloc((void**)&gpu_newMolPos, sizeof(double) * maxMolLen * 3 * BOX_TOTAL);
		if (error != cudaSuccess)
		{
			std::cout << "cudaMalloc gpu_newMolPos returned error " <<
					cudaGetErrorString(error) << "code: " << error << std::endl;
			exit(EXIT_FAILURE);
		}

		error = cudaMalloc((void**)&gpu_oldMolPos, sizeof(double) * maxMolLen * 3);
		if (error != cudaSuccess)
		{
			std::cout << "cudaMalloc gpu_oldMolPos returned error " <<
					cudaGetErrorString(error) << "code: " << error << std::endl;
			exit(EXIT_FAILURE);
		}

		//each box has its own gpu arrays for threads reduction and value return.
		if(imageSize[0] <= MAXTHREADSPERBLOCK)
		{
		   recipBlockSize[0] = imageSize[0];
		}
		else
		{
		   recipBlockSize[0] = MAXTHREADSPERBLOCK;
		}
		if(imageSize[1] <= MAXTHREADSPERBLOCK)
		{
		   recipBlockSize[1] = imageSize[1];
		}
		else
		{
		   recipBlockSize[1] = MAXTHREADSPERBLOCK;
		}
		recipGridSize[0] = (imageSize[0] + MAXTHREADSPERBLOCK - 1) / MAXTHREADSPERBLOCK ;
		recipGridSize[1] = (imageSize[1] + MAXTHREADSPERBLOCK - 1) / MAXTHREADSPERBLOCK ;
		cudaMalloc((void**)&gpu_blockRecipNew, sizeof(double) * (recipGridSize[0] + recipGridSize[1]));
		cudaMalloc((void**)&gpu_rotateMatrix, sizeof(double) * 9);

		std::cout << "cudaMalloc is done." << std::endl;
		//load to GPU
		molLookup.TotalAtomsMols(atomPerBox, mols.kinds);
		if (BOX_TOTAL==1)
		{
			molPerBox[0] = mols.count;
			for (int i = 0; i < imageSize[0]; i++)
			{
				h_prefact[i] = prefact[0][i];
				h_kx[i] = kx[0][i];
				h_ky[i] = ky[0][i];
				h_kz[i] = kz[0][i];
			}
		}
		else
		{
			for (int i = 0; i < BOX_TOTAL; i++)
			{
				molPerBox[i] = molLookup.NumInBox(i);
				for (int j = 0; j < imageSize[i]; j++)
				{
					int index = j + i * imageSize[0];
					h_prefact[index] = prefact[i][j];
					h_kx[index] = kx[i][j];
					h_ky[index] = ky[i][j];
					h_kz[index] = kz[i][j];
				}
			}
		}

		for (int i = 0; i < mols.count; i++)
		{
			atomPerMol[i] = mols.kinds[mols.kIndex[i]].NumAtoms();
		}

		error = cudaMemcpy(gpu_atomCharge, particleCharge.data(), sizeof(double) * particleCharge.size(), cudaMemcpyHostToDevice);
		if (error != cudaSuccess)
		{
			std::cout << "cudaMemcpy gpu_atomCharge returned error " <<
					cudaGetErrorString(error) << "code: " << error << std::endl;
			exit(EXIT_FAILURE);
		}

		error = cudaMemcpy(gpu_xCoords, currentCoords.x, sizeof(double) * currentCoords.Count(), cudaMemcpyHostToDevice);
		if (error != cudaSuccess)
		{
			std::cout << "cudaMemcpy gpu_xCoords returned error " <<
					cudaGetErrorString(error) << "code: " << error << std::endl;
			exit(EXIT_FAILURE);
		}

		error = cudaMemcpy(gpu_yCoords, currentCoords.y, sizeof(double) * currentCoords.Count(), cudaMemcpyHostToDevice);
		if (error != cudaSuccess)
		{
			std::cout << "cudaMemcpy gpu_yCoords returned error " <<
					cudaGetErrorString(error) << "code: " << error << std::endl;
			exit(EXIT_FAILURE);
		}

		error = cudaMemcpy(gpu_zCoords, currentCoords.z, sizeof(double) * currentCoords.Count(), cudaMemcpyHostToDevice);
		if (error != cudaSuccess)
		{
			std::cout << "cudaMemcpy gpu_zCoords returned error " <<
					cudaGetErrorString(error) << "code: " << error << std::endl;
			exit(EXIT_FAILURE);
		}

		error = cudaMemcpy(gpu_atomPerMol, atomPerMol, sizeof(uint) * mols.count, cudaMemcpyHostToDevice);
		if (error != cudaSuccess)
		{
			std::cout << "cudaMemcpy gpu_atomPerMol returned error " <<
					cudaGetErrorString(error) << "code: " << error << std::endl;
			exit(EXIT_FAILURE);
		}

		error = cudaMemcpy(gpu_molStartAtom, mols.start, sizeof(uint) * mols.count, cudaMemcpyHostToDevice);
		if (error != cudaSuccess)
		{
			std::cout << "cudaMemcpy gpu_molStartAtom returned error " <<
					cudaGetErrorString(error) << "code: " << error << std::endl;
			exit(EXIT_FAILURE);
		}

		error = cudaMemcpy(gpu_kx, h_kx, sizeof(double) * imageTotal, cudaMemcpyHostToDevice);
		if (error != cudaSuccess)
		{
			std::cout << "cudaMemcpy gpu_kx returned error " <<
					cudaGetErrorString(error) << "code: " << error << std::endl;
			exit(EXIT_FAILURE);
		}

		error = cudaMemcpy(gpu_ky, h_ky, sizeof(double) * imageTotal, cudaMemcpyHostToDevice);
		if (error != cudaSuccess)
		{
			std::cout << "cudaMemcpy gpu_ky returned error " <<
					cudaGetErrorString(error) << "code: " << error << std::endl;
			exit(EXIT_FAILURE);
		}

		error = cudaMemcpy(gpu_kz, h_kz, sizeof(double) * imageTotal, cudaMemcpyHostToDevice);
		if (error != cudaSuccess)
		{
			std::cout << "cudaMemcpy gpu_kz returned error " <<
					cudaGetErrorString(error) << "code: " << error << std::endl;
			exit(EXIT_FAILURE);
		}

		error = cudaMemcpy(gpu_prefact, h_prefact, sizeof(double) * imageTotal, cudaMemcpyHostToDevice);
		if (error != cudaSuccess)
		{
			std::cout << "cudaMemcpy gpu_prefact returned error " <<
					cudaGetErrorString(error) << "code: " << error << std::endl;
			exit(EXIT_FAILURE);
		}

		std::cout << "memcpy is done." << std::endl;
		//cuda stream initialize
		for (int i = 0; i < BOX_TOTAL; i++)
		{
			error = cudaStreamCreate(&streams[i]);
			if (error != cudaSuccess)
			{
				std::cout << "cudaStream " << i << " returned error " <<
						cudaGetErrorString(error) << "code: " << error << std::endl;
				exit(EXIT_FAILURE);
			}

		}
		std::cout << "cuda stream is done." << std::endl;
	}
}

void Ewald::freeGpuData()
{
	delete[] molStartAtom;
	delete[] atomPerMol;
	delete[] h_prefact;
	delete[] h_kx;
	delete[] h_ky;
	delete[] h_kz;
	cudaFree(gpu_atomCharge);
	cudaFree(gpu_xCoords);
	cudaFree(gpu_yCoords);
	cudaFree(gpu_zCoords);
	cudaFree(gpu_xTempCoords);
	cudaFree(gpu_yTempCoords);
	cudaFree(gpu_zTempCoords);
	cudaFree(gpu_molStartAtom);
	cudaFree(atomPerMol);
	cudaFree(gpu_prefact);
	cudaFree(gpu_recipEnergy);
	cudaFree(gpu_kx);
	cudaFree(gpu_ky);
	cudaFree(gpu_kz);
	cudaFree(gpu_imageReal);
	cudaFree(gpu_imageImaginary);
	cudaFree(gpu_newMolPos);
	cudaFree(gpu_oldMolPos);
	cudaFree(gpu_blockRecipNew);
	cudaFree(gpu_rotateMatrix);
	for (int i = 0; i < BOX_TOTAL; i++)
		cudaStreamDestroy(streams[i]);
}

void Ewald::RecipInit(uint box, BoxDimensions const& boxAxes)
{
   uint counter = 0;
   double ksqr;
   double alpsqr4 = 1.0 / (4.0 * alpha * alpha);
   double constValue = 2 * M_PI / boxAxes.axis.BoxSize(box);
   double vol = boxAxes.volume[box] / (4 * M_PI);
//   int kmax = int(recip_rcut * boxAxes.axis.BoxSize(box) / (2 * M_PI)) + 1;
   int kmax = 5;
   for (int x = 0; x <= kmax; x++)
   {
      int nky_max = sqrt(pow(kmax, 2) - pow(x, 2));
      int nky_min = -nky_max;
      if (x == 0.0)
      { 
	 nky_min = 0;
      }
      for (int y = nky_min; y <= nky_max; y++)
      {
	 int nkz_max = sqrt(pow(kmax, 2) - pow(x, 2) - pow(y, 2));
	 int nkz_min = -nkz_max;
	 if (x == 0.0 && y == 0.0)
     {
	    nkz_min = 1;
	 }
	 for (int z = nkz_min; z <= nkz_max; z++)
         {
	   ksqr = pow((constValue * x), 2) + pow((constValue * y), 2) +
	     pow ((constValue * z), 2);
	    
	    if (ksqr < recip_rcut_Sq)
	    {
	       kx[box][counter] = constValue * x;
	       ky[box][counter] = constValue * y;
	       kz[box][counter] = constValue * z;
	       prefact[box][counter] = num::qqFact * exp(-ksqr * alpsqr4)/
		 (ksqr * vol);
	       counter++;
	    }
	 }
      }
   }
   imageSize[box] = counter;
}

//calculate reciprocate term for a box
double Ewald::BoxReciprocal(int box, XYZArray const& molCoords, bool volume)
{
   //This is the global array used to store the addition of reciprocal real and imaginary
   //of each block
   int atomOffset, imageOffset, gridOffset;

   if (box == 0)
   {
	   atomOffset = 0;
	   imageOffset = 0;
	   gridOffset = 0;
   }
   else if (box == 1)
   {
	   atomOffset = atomPerBox[0];
	   imageOffset = imageSize[0];
	   gridOffset = recipGridSize[0];
   }

   if (box < BOXES_WITH_U_NB)
   {
	   if (!volume)
	   {
		  BoxReciprocalGPU<<<recipGridSize[box], recipBlockSize[box], 0, streams[box]>>>(
				  atomPerBox[box], atomOffset, gpu_xCoords, gpu_yCoords,
				  gpu_zCoords, gpu_atomCharge, gpu_imageReal,
				  gpu_imageImaginary, gpu_kx, gpu_ky, gpu_kz,
				  gpu_prefact, box, imageOffset, gridOffset,
				  gpu_blockRecipNew, imageSize[box]);
	   }
	   else
	   {
		   BoxReciprocalGPU<<<recipGridSize[box], recipBlockSize[box], 0, streams[box]>>>(
		   				  atomPerBox[box], atomOffset, gpu_xCoordsNew, gpu_yCoordsNew,
		   				  gpu_zCoordsNew, gpu_atomCharge, gpu_imageRealRef,
		   				  gpu_imageImaginaryRef, gpu_kxNew, gpu_kyNew, gpu_kzNew,
		   				  gpu_prefactNew, box, imageOffset, gridOffset,
		   				  gpu_blockRecipNew, imageSize[box]);
	   }
      cudaMemcpyAsync(&blockRecipEnergy[box], &gpu_blockRecipNew[gridOffset], sizeof(double), cudaMemcpyDeviceToHost, streams[box]);
      cudaStreamSynchronize(streams[box]);
   }

   return blockRecipEnergy[box];
}


//calculate reciprocate term for displacement and rotation move
double Ewald::MolReciprocal(XYZArray const& molCoords,
			    const uint molIndex,
			    const uint box,
			    XYZ const*const newCOM)
{
   double energyRecipNew = 0.0; 
   double energyRecipOld = 0.0;
   
   if (box < BOXES_WITH_U_NB)
   {
      MoleculeKind const& thisKind = mols.GetKind(molIndex);
      uint length = thisKind.NumAtoms();
      uint startAtom = mols.MolStart(molIndex);
      
      for (uint i = 0; i < imageSize[box]; i++)
      { 
		 double sumRealNew = 0.0;
		 double sumImaginaryNew = 0.0;
		 double sumRealOld = 0.0;
		 double sumImaginaryOld = 0.0;
		 double dotProductNew = 0.0;
		 double dotProductOld = 0.0;

		 for (uint p = 0; p < length; ++p)
		 {
			uint atom = startAtom + p;
			dotProductNew = currentAxes.DotProduct(p, kx[box][i], ky[box][i],
							kz[box][i], molCoords, box);
			dotProductOld = currentAxes.DotProduct(atom, kx[box][i], ky[box][i],
							kz[box][i], currentCoords, box);

			sumRealNew += (thisKind.AtomCharge(p) * cos(dotProductNew));
			sumImaginaryNew += (thisKind.AtomCharge(p) * sin(dotProductNew));

			sumRealOld += (thisKind.AtomCharge(p) * cos(dotProductOld));
			sumImaginaryOld += (thisKind.AtomCharge(p) * sin(dotProductOld));
		 }

		 sumRnew[box][i] = sumRref[box][i] - sumRealOld + sumRealNew;
		 sumInew[box][i] = sumIref[box][i] - sumImaginaryOld + sumImaginaryNew;

		 energyRecipNew += (sumRnew[box][i] * sumRnew[box][i] + sumInew[box][i]
					* sumInew[box][i]) * prefact[box][i];

		 energyRecipOld += (sumRref[box][i] * sumRref[box][i] + sumIref[box][i]
					* sumIref[box][i]) * prefact[box][i];
	 
      }
      
   }

   return energyRecipNew - energyRecipOld; 
}

double Ewald::MolReciprocal(const uint pStart, const uint pLen, const uint molIndex,
		   const uint box, const XYZ shiftOrCenter, const int moveType, RotationMatrix matrix)
{
	double rotateMatrix[9];
	int imageOffset, gridOffset;

	if (box == 0)
	{
	   imageOffset = 0;
	   gridOffset = 0;
	}
	else if (box == 1)
	{
	   imageOffset = imageSize[0];
	   gridOffset = recipGridSize[0];
	}

	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			rotateMatrix[i * 3 + j] = matrix.GetMatrix(i, j);
		}
	}
	cudaMemcpy(gpu_rotateMatrix, rotateMatrix, sizeof(double) * 9, cudaMemcpyHostToDevice);

	if (box < BOXES_WITH_U_NB)
	{
		if (moveType == 0)
			MolReciprocalDisplaceGPU<<<recipGridSize[box], recipBlockSize[box], maxMolLen * 3 * sizeof(double), streams[box]>>>(
					pStart, pLen, molIndex, shiftOrCenter,
					gpu_xCoords, gpu_yCoords, gpu_zCoords, gpu_atomCharge,
					gpu_imageReal, gpu_imageImaginary, gpu_imageRealRef,
					gpu_imageImaginaryRef, gpu_kx, gpu_ky, gpu_kz,
					gpu_newMolPos, gpu_prefact, box, imageOffset, gridOffset,
					gpu_blockRecipNew, imageSize[box], currentAxes.axis.x[box],
					currentAxes.axis.y[box], currentAxes.axis.z[box], maxMolLen);
		else if (moveType == 1)
			MolReciprocalRotateGPU<<<recipGridSize[box], recipBlockSize[box], maxMolLen * 3 * sizeof(double), streams[box]>>>(
					pStart, pLen, molIndex, shiftOrCenter,
					gpu_xCoords, gpu_yCoords, gpu_zCoords, gpu_atomCharge,
					gpu_imageReal, gpu_imageImaginary, gpu_imageRealRef,
					gpu_imageImaginaryRef, gpu_kx, gpu_ky, gpu_kz,
					gpu_newMolPos, gpu_prefact, box, imageOffset, gridOffset,
					gpu_blockRecipNew, imageSize[box], currentAxes.axis.x[box],
					currentAxes.axis.y[box], currentAxes.axis.z[box], maxMolLen, gpu_rotateMatrix);

		cudaMemcpy(&blockRecipEnergy[box], &gpu_blockRecipNew[gridOffset], sizeof(double), cudaMemcpyDeviceToHost);
		cudaStreamSynchronize(streams[box]);
	}

	return blockRecipEnergy[box];
}

//calculate self term for a box
double Ewald::BoxSelf(BoxDimensions const& boxAxes, uint box) const
{
   if (box >= BOXES_WITH_U_NB || !ewald)
     return 0.0;

   double self = 0.0;
   for (uint i = 0; i < mols.kindsCount; i++)
   {
     MoleculeKind const& thisKind = mols.kinds[i];
     uint length = thisKind.NumAtoms();
     double molSelfEnergy = 0.0;
     for (uint j = 0; j < length; j++)
     {
       molSelfEnergy += (thisKind.AtomCharge(j) * thisKind.AtomCharge(j));
     }
     self += (molSelfEnergy * molLookup.NumKindInBox(i, box));
   }
   
   self = -1.0 * self * alpha * num::qqFact / sqrt(M_PI);

   return self;
}


//calculate correction term for a molecule
double Ewald::MolCorrection(uint molIndex, BoxDimensions const& boxAxes,
			    uint box) const
{
   if (box >= BOXES_WITH_U_NB || !ewald)
     return 0.0;

   double dist, distSq;
   double correction = 0.0;
   XYZ virComponents; 
   
   MoleculeKind& thisKind = mols.kinds[mols.kIndex[molIndex]];
   for (uint i = 0; i < thisKind.NumAtoms(); i++)
   {
      for (uint j = i + 1; j < thisKind.NumAtoms(); j++)
      {
	 currentAxes.InRcut(distSq, virComponents, currentCoords,
			    mols.start[molIndex] + i,
			    mols.start[molIndex] + j, box);
	 dist = sqrt(distSq);
	 correction += (thisKind.AtomCharge(i) * thisKind.AtomCharge(j) *
			erf(alpha * dist) / dist);
      }
   }

   return correction;
}

//calculate reciprocate term in destination box for swap move
double Ewald::SwapDestRecip(cbmc::TrialMol newMol, const uint box) 
{
   double energyRecipNew = 0.0;  
   
   if (box < BOXES_WITH_U_NB || !ewald)
   {
      MoleculeKind const& thisKind = newMol.GetKind();
      XYZArray molCoords = newMol.GetCoords();
      for (uint i = 0; i < imageSize[box]; i++)
      {
		 double sumRealNew = 0.0;
		 double sumImaginaryNew = 0.0;
		 double dotProductNew = 0.0;
		 uint length = thisKind.NumAtoms();

		 for (uint p = 0; p < length; ++p)
		 {
			dotProductNew = currentAxes.DotProduct(p, kx[box][i], ky[box][i],
							kz[box][i], molCoords, box);

			sumRealNew += (thisKind.AtomCharge(p) * cos(dotProductNew));
			sumImaginaryNew += (thisKind.AtomCharge(p) * sin(dotProductNew));

		 }

		 sumRnew[box][i] = sumRref[box][i] + sumRealNew;
		 sumInew[box][i] = sumIref[box][i] + sumImaginaryNew;

		 energyRecipNew += (sumRnew[box][i] * sumRnew[box][i] + sumInew[box][i]
					* sumInew[box][i]) * prefact[box][i];
      }
   }

   return energyRecipNew;
}


//calculate reciprocate term in source box for swap move
double Ewald::SwapSourceRecip(cbmc::TrialMol oldMol, const uint box) 
{
   double energyRecipNew = 0.0;  
   
   if (box < BOXES_WITH_U_NB || !ewald)
   {
      MoleculeKind const& thisKind = oldMol.GetKind();
      XYZArray molCoords = oldMol.GetCoords();
      for (uint i = 0; i < imageSize[box]; i++)
      {
		 double sumRealNew = 0.0;
		 double sumImaginaryNew = 0.0;
		 double dotProductNew = 0.0;
		 uint length = thisKind.NumAtoms();

		 for (uint p = 0; p < length; ++p)
		 {
			dotProductNew = currentAxes.DotProduct(p, kx[box][i], ky[box][i],
							kz[box][i], molCoords, box);

			sumRealNew += (thisKind.AtomCharge(p) * cos(dotProductNew));
			sumImaginaryNew += (thisKind.AtomCharge(p) * sin(dotProductNew));

		 }

		 sumRnew[box][i] = sumRref[box][i] - sumRealNew;
		 sumInew[box][i] = sumIref[box][i] - sumImaginaryNew;

		 energyRecipNew += (sumRnew[box][i] * sumRnew[box][i] + sumInew[box][i]
					* sumInew[box][i]) * prefact[box][i];
      }
   }

   return energyRecipNew;
}


//calculate self term for CBMC algorithm
void Ewald::SwapSelf(double *self, uint molIndex, uint partIndex, int box, 
	      uint trials) const
{
   if (box >= BOXES_WITH_U_NB || !ewald)
     return;

   MoleculeKind const& thisKind = mols.GetKind(molIndex);

   for (uint t = 0; t < trials; t++)
   {
     self[t] -= (thisKind.AtomCharge(partIndex) *
		 thisKind.AtomCharge(partIndex) * alpha *
		 num::qqFact / sqrt(M_PI)); 
   }

}

//calculate correction term for linear molecule CBMC algorithm
void Ewald::SwapCorrection(double* energy, const cbmc::TrialMol& trialMol, 
		    XYZArray const& trialPos, const uint partIndex, 
		    const uint box, const uint trials) const
{
   if (box >= BOXES_WITH_U_NB || !ewald)
     return;

   double dist;
   const MoleculeKind& thisKind = trialMol.GetKind();

   //loop over all partners of the trial particle
   const uint* partner = thisKind.sortedEwaldNB.Begin(partIndex);
   const uint* end = thisKind.sortedEwaldNB.End(partIndex);
   while (partner != end)
   {
      if (trialMol.AtomExists(*partner))
      {
	 for (uint t = 0; t < trials; ++t)
	 {
	   double distSq;
	   if (currentAxes.InRcut(distSq, trialPos, t, trialMol.GetCoords(),
				  *partner, box))
	   {
	     dist = sqrt(distSq);
	     energy[t] -= (thisKind.AtomCharge(*partner) *
			   thisKind.AtomCharge(partIndex) * erf(alpha * dist) *
			   num::qqFact / dist);
	   }
	 }
      }
      ++partner;
   }
}


//calculate correction term for branched molecule CBMC algorithm
void Ewald::SwapCorrection(double* energy, const cbmc::TrialMol& trialMol,
		    XYZArray *trialPos, const int pickedAtom, 
		    uint *partIndexArray, const uint box, const uint trials,
		    const uint prevIndex, bool prev) const
{
   if (box >= BOXES_WITH_U_NB || !ewald)
     return;

   double dist, distSq;
   const MoleculeKind& thisKind = trialMol.GetKind();
   uint pickedAtomIndex = partIndexArray[pickedAtom];

   if(prev)
      pickedAtomIndex = prevIndex;
	  
   for (int t = 0; t < trials; t++)
   {
      //loop through all previous new atoms generated simultanously,
      //and calculate the pair interactions between the picked atom and
      //the prev atoms.
      for (int newAtom = 0; newAtom < pickedAtom; newAtom++)
      {
	 distSq = 0;
	 if (currentAxes.InRcut(distSq, trialPos[newAtom], t,
				trialPos[pickedAtom], t, box))
	 {
	   dist = sqrt(distSq);
	   energy[t] -= (thisKind.AtomCharge(pickedAtomIndex) *
			 thisKind.AtomCharge(partIndexArray[newAtom]) *
			 erf(alpha * dist) * num::qqFact / dist);
	 }
      }

      //loop through the array of new molecule's atoms, and calculate the pair
      //interactions between the picked atom and the atoms have been created 
      //previously and added
      for (int count = 0; count < thisKind.NumAtoms(); count++)
      {
	 if (trialMol.AtomExists(count))
	 {
	    distSq = 0;
	    if (currentAxes.InRcut(distSq, trialMol.GetCoords(), count,
				   trialPos[pickedAtom], t, box))
	    {
	       dist = sqrt(distSq);
	       energy[t] -= (thisKind.AtomCharge(pickedAtomIndex) * 
			     thisKind.AtomCharge(count) *
			     erf(alpha * dist) * num::qqFact / dist);
	    }
	 }
      }
   }
}

double Ewald::CorrectionOldMol(const cbmc::TrialMol& oldMol,
			       const double distSq, const uint i,
			       const uint j) const
{
  if (oldMol.GetBox() >= BOXES_WITH_U_NB || !ewald)
     return 0.0;

   const MoleculeKind& thisKind = oldMol.GetKind();
   double dist = sqrt(distSq);
   return (-1 * thisKind.AtomCharge(i) * thisKind.AtomCharge(j) *
	   erf(alpha * dist) * num::qqFact / dist);
}


//GPU functions
void Ewald::UpdateGPU(const uint pStart, const uint pLen, const uint molIndex, const uint box)
{
	int offset = 0;
	if (box == 1)
		offset = imageSize[0];

	AcceptUpdateGPU<<<1, 1, 0, streams[box]>>>(
			pStart, pLen, molIndex, gpu_xCoords, gpu_yCoords,
			gpu_zCoords, gpu_newMolPos, maxMolLen, box);

	cudaMemcpyAsync(&gpu_imageReal[offset], &gpu_imageRealRef[offset], sizeof(double) * imageSize[box],
			cudaMemcpyDeviceToDevice, streams[box]);
	cudaMemcpyAsync(&gpu_imageImaginary[offset], &gpu_imageImaginaryRef[offset], sizeof(double) * imageSize[box],
			cudaMemcpyDeviceToDevice, streams[box]);
	cudaStreamSynchronize(streams[box]);
	//update imageReal & imageImaginary
//	double *tempReal, *tempImaginary;
//	tempReal = gpu_imageReal;
//	tempImaginary = gpu_imageImaginary;
//	gpu_imageReal = gpu_imageRealRef;
//	gpu_imageImaginary = gpu_imageImaginaryRef;
//	gpu_imageRealRef = tempReal;
//	gpu_imageImaginaryRef = tempImaginary;
}

void Ewald::RestoreGPU(uint box)
{
	int offset = 0;
	if (box == 1)
		offset = imageSize[0];
		cudaMemcpyAsync(&gpu_imageRealRef[offset], &gpu_imageReal[offset], sizeof(double) * imageSize[box],
				cudaMemcpyDeviceToDevice, streams[box]);
		cudaMemcpyAsync(&gpu_imageImaginaryRef[offset], &gpu_imageImaginary[offset], sizeof(double) * imageSize[box],
				cudaMemcpyDeviceToDevice, streams[box]);
		cudaStreamSynchronize(streams[box]);
}

void Ewald::SetupGPUKernel()
{
	//each box has its own gpu arrays for threads reduction and value return.
	if(imageSize[0] <= MAXTHREADSPERBLOCK)
	{
	   recipBlockSize[0] = imageSize[0];
	}
	else
	{
	   recipBlockSize[0] = MAXTHREADSPERBLOCK;
	}
	if(imageSize[1] <= MAXTHREADSPERBLOCK)
	{
	   recipBlockSize[1] = imageSize[1];
	}
	else
	{
	   recipBlockSize[1] = MAXTHREADSPERBLOCK;
	}
	int imageTotal = imageSize[0] + imageSize[1];
	recipGridSize[0] = (imageSize[0] + MAXTHREADSPERBLOCK - 1) / MAXTHREADSPERBLOCK ;
	recipGridSize[1] = (imageSize[1] + MAXTHREADSPERBLOCK - 1) / MAXTHREADSPERBLOCK ;
}

void Ewald::AcceptVolumeMove()
{
	//check if imageSize is changed or not. If not changed, don't do cudaFree & cudaMalloc
	/*
	cudaMemcpy(gpu_imageReal, gpu_imageRealRef, sizeof(double) * (imageSize[0] + imageSize[1]), cudaMemcpyDeviceToDevice);
	cudaMemcpy(gpu_imageImaginary, gpu_imageImaginaryRef, sizeof(double) * (imageSize[0] + imageSize[1]), cudaMemcpyDeviceToDevice);
	cudaMemcpy(gpu_prefact, gpu_prefactNew, sizeof(double) * (imageSize[0] + imageSize[1]), cudaMemcpyDeviceToDevice);
	cudaMemcpy(gpu_kx, gpu_kxNew, sizeof(double) * (imageSize[0] + imageSize[1]), cudaMemcpyDeviceToDevice);
	cudaMemcpy(gpu_ky, gpu_kyNew, sizeof(double) * (imageSize[0] + imageSize[1]), cudaMemcpyDeviceToDevice);
	cudaMemcpy(gpu_kz, gpu_kzNew, sizeof(double) * (imageSize[0] + imageSize[1]), cudaMemcpyDeviceToDevice);
	cudaMemcpy(gpu_xCoords, gpu_xCoordsNew, sizeof(double) * (imageSize[0] + imageSize[1]), cudaMemcpyDeviceToDevice);
	cudaMemcpy(gpu_yCoords, gpu_yCoordsNew, sizeof(double) * (imageSize[0] + imageSize[1]), cudaMemcpyDeviceToDevice);
	cudaMemcpy(gpu_zCoords, gpu_zCoordsNew, sizeof(double) * (imageSize[0] + imageSize[1]), cudaMemcpyDeviceToDevice);
	*/
	double *tempA, *tempB, *tempC;
	tempA = gpu_imageReal;
	tempB = gpu_imageImaginary;
	gpu_imageReal = gpu_imageRealRef;
	gpu_imageImaginary = gpu_imageImaginaryRef;
	gpu_imageRealRef = tempA;
	gpu_imageImaginaryRef = tempB;

	tempA = gpu_prefact;
	gpu_prefact = gpu_prefactNew;
	gpu_prefactNew = tempA;

	tempA = gpu_kx;
	tempB = gpu_ky;
	tempC = gpu_kz;
	gpu_kx = gpu_kxNew;
	gpu_ky = gpu_kyNew;
	gpu_kz = gpu_kzNew;
	gpu_kxNew = tempA;
	gpu_kyNew = tempB;
	gpu_kzNew = tempC;

	tempA = gpu_xCoords;
	tempB = gpu_yCoords;
	tempC = gpu_zCoords;
	gpu_xCoords = gpu_xCoordsNew;
	gpu_yCoords = gpu_yCoordsNew;
	gpu_zCoords = gpu_zCoordsNew;
	gpu_xCoordsNew = tempA;
	gpu_yCoordsNew = tempB;
	gpu_zCoordsNew = tempC;
}

void Ewald::RejectVolumeMove()
{
	//if imageSize is not changed, do not need to run this function.
	cudaFree(gpu_blockRecipNew);
	SetupGPUKernel();
	cudaMalloc((void**)&gpu_blockRecipNew, sizeof(double) * (recipGridSize[0] + recipGridSize[1]));
}

void Ewald::PrepareVolumeMove(XYZArray const &coords)
{
	//check if imageSize is changed or not. If not change, no necessary to free and re-malloc
	cudaFree(gpu_imageRealRef);
	cudaFree(gpu_imageImaginaryRef);
	cudaFree(gpu_blockRecipNew);
	cudaFree(gpu_prefactNew);
	cudaFree(gpu_kxNew);
	cudaFree(gpu_kyNew);
	cudaFree(gpu_kzNew);
	cudaMalloc((void**)&gpu_prefactNew, sizeof(double) * imageTotal);
	cudaMalloc((void**)&gpu_kxNew, sizeof(double) * imageTotal);
	cudaMalloc((void**)&gpu_kyNew, sizeof(double) * imageTotal);
	cudaMalloc((void**)&gpu_kzNew, sizeof(double) * imageTotal);
	cudaMalloc((void**)&gpu_imageRealRef, sizeof(double) * imageTotal);
	cudaMalloc((void**)&gpu_imageImaginaryRef, sizeof(double) * imageTotal);
	SetupGPUKernel();
	cudaMalloc((void**)&gpu_blockRecipNew, sizeof(double) * (recipGridSize[0] + recipGridSize[1]));

	if (BOX_TOTAL==1)
	{
		molPerBox[0] = mols.count;
		for (int i = 0; i < imageSize[0]; i++)
		{
			h_prefact[i] = prefact[0][i];
			h_kx[i] = kx[0][i];
			h_ky[i] = ky[0][i];
			h_kz[i] = kz[0][i];
		}
	}
	else
	{
		int index;
		for (int i = 0; i < BOX_TOTAL; i++)
		{
			molPerBox[i] = molLookup.NumInBox(i);
			for (int j = 0; j < imageSize[i]; j++)
			{
				index = j + i * imageSize[0];
				h_prefact[index] = prefact[i][j];
				h_kx[index] = kx[i][j];
				h_ky[index] = ky[i][j];
				h_kz[index] = kz[i][j];
			}
		}
	}

	cudaMemcpy(gpu_kxNew, h_kx, sizeof(double) * imageTotal, cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_kyNew, h_ky, sizeof(double) * imageTotal, cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_kzNew, h_kz, sizeof(double) * imageTotal, cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_prefactNew, h_prefact, sizeof(double) * imageTotal, cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_xCoordsNew, coords.x, sizeof(double) * coords.Count(), cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_yCoordsNew, coords.y, sizeof(double) * coords.Count(), cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_zCoordsNew, coords.z, sizeof(double) * coords.Count(), cudaMemcpyHostToDevice);
}

__device__ double DotProductGPU(
		double kx, double ky, double kz,
		double *xCoords, double *yCoords,
		double *zCoords, uint i)
{
	return(xCoords[i] * kx + yCoords[i] * ky + zCoords[i] * kz);
}


__device__ double WrapPBC(double& v, const double ax) {
	// Inspired by :
	// http://graphics.stanford.edu/~seander/bithacks.html#ConditionalNegate
	//
#ifdef NO_BRANCHING_WRAP
	// Inspired by :
	// http://graphics.stanford.edu/~seander/bithacks.html#ConditionalNegate
	//
	// Advantages:
	// No branching
	//
	// Disadvantages:
	// Sometimes a couple of extra ops or a couple of extra compares.
	if (
			bool negate = (v > ax);
			double vNeg = v + (ax ^ -negate) + negate;
			return (fabs(v - halfAx) > halfAx) ? v : vNeg;
#else

	//Note: testing shows that it's most efficient to negate if true.
	//Source:
	// http://jacksondunstan.com/articles/2052
	if (v > ax) //if +, wrap out to low end
		v -= ax;
	else if (v < 0) //if -, wrap to high end
		v += ax;
	return v;
#endif
}

__device__ double UnwrapPBC(double& v, const double ref, const double ax,
		const double halfAx) {
	//If absolute value of X dist btwn pt and ref is > 0.5 * box_axis
	//If ref > 0.5 * box_axis, add box_axis to pt (to wrap out + side)
	//If ref < 0.5 * box_axis, subtract box_axis (to wrap out - side)
	// uses bit hack to avoid branching for conditional
#ifdef NO_BRANCHING_UNWRAP
//	bool negate = ( ref > halfAx );
//	double vDiff = v + (ax ^ -negate) + negate;
//	return (fabs(ref - v) > halfAx ) ? v : vDiff;
#else
	if (fabs(ref - v) > halfAx) {
		//Note: testing shows that it's most efficient to negate if true.
		//Source:
		// http://jacksondunstan.com/articles/2052
		if (ref < halfAx) {
			v -= ax;
		} else {
			v += ax;
		}
	}

	return v;
#endif
}

__device__ void MatrixApply(double& xpos, double& ypos, double& zpos, double *matrix)
		{
	double x, y, z;
	x = matrix[0] * xpos + matrix[1] * ypos + matrix[2] * zpos;
	y = matrix[3] * xpos + matrix[4] * ypos + matrix[5] * zpos;
	z = matrix[6] * xpos + matrix[7] * ypos + matrix[8] * zpos;

	xpos = x;
	ypos = y;
	zpos = z;
}

__device__ void sinecosine(float ref, double &sine, double &cosine)
{
	sine = double(__sinf(ref));
	cosine = double(__cosf(ref));
}

__global__ void BoxReciprocalGPU(
		const uint atomPerBox, const uint atomOffset,
		double *gpu_xCoords, double *gpu_yCoords,
		double *gpu_zCoords, double *gpu_atomCharge,
		double *gpu_imageReal, double *gpu_imageImaginary,
		double *gpu_kx, double *gpu_ky, double *gpu_kz, double *gpu_prefact,
		const uint box, const uint imageOffset, const uint gridOffset,
		double *gpu_blockRecipEnergy, const uint imageSize)
{
	__shared__ double imageSum[MAXTHREADSPERBLOCK];
	__shared__ bool	lastBlock;

	int threadId = blockDim.x * blockIdx.x + threadIdx.x;
	int imageIndex = threadId + imageOffset;

	double kx = 0.0;
	double ky = 0.0;
	double kz = 0.0;
	double sumReal = 0.0;
	double sumImaginary = 0.0;
	double dotProduct = 0.0;
	double sine, cosine;
	imageSum[threadIdx.x] = 0.0;

	if (threadId < imageSize)
	{
		kx = gpu_kx[imageIndex];
		ky = gpu_ky[imageIndex];
		kz = gpu_kz[imageIndex];
		//if use loop unroll here, addition of sumReal & sumImaginary should be using reduction
		for (int i = atomOffset; i < atomOffset + atomPerBox; i++)
		{
			//dotProduct = DotProductGPU(kx, ky, kz, gpu_xCoords, gpu_yCoords, gpu_zCoords, i);
			dotProduct = DotProductGPU(kx, ky, kz, gpu_xCoords, gpu_yCoords, gpu_zCoords, i);
#ifdef debug
			if (i == 30)
				printf("image: %d, atom: %d, dotProduct: %lf, atomCharge: %lf, kx: %lf, ky: %lf, kz: %lf, x: %lf, y: %lf, z: %lf\n",
						imageIndex, i, dotProduct, gpu_atomCharge[i], kx, ky, kz, gpu_xCoords[i], gpu_yCoords[i], gpu_zCoords[i]);
#endif
			sinecosine(dotProduct, sine, cosine);
			sumReal += (gpu_atomCharge[i] * cosine);
			sumImaginary += (gpu_atomCharge[i] * sine);

		}
		gpu_imageReal[imageIndex] = sumReal;
		gpu_imageImaginary[imageIndex] = sumImaginary;
		imageSum[threadIdx.x] = (sumReal * sumReal + sumImaginary * sumImaginary)  * gpu_prefact[imageIndex];
#ifdef debug
		if (threadId == 0)
			printf("box: %d, image: %d, cacheReal: %lf, cacheImaginary: %lf, sum: %lf\n",
					box, threadId, sumReal, sumImaginary, imageSum[threadIdx.x]);
#endif
	}
	//after the calculation of each thread, implement reduction on gpu_imageReal
	//and gpu_imageImaginary within the block, and then another reduction between blocks
	__syncthreads();
	// add data
	int offset = 1 << (int) __log2f((float) blockDim.x);

	if (blockDim.x < MAXTHREADSPERBLOCK) {
		if ((threadIdx.x + offset) < imageSize % MAXTHREADSPERBLOCK) {
			imageSum[threadIdx.x] += imageSum[threadIdx.x + offset];
		}

		__syncthreads();
	}

	for (int i = offset >> 1; i > 32; i >>= 1) {
		if (threadIdx.x < i) {
			imageSum[threadIdx.x] += imageSum[threadIdx.x + i];
		}           //end if

		__syncthreads();
	}           //end for

	if (threadIdx.x < 32) {
		offset = min(offset, 64);

		switch (offset) {
		case 64:
			imageSum[threadIdx.x] += imageSum[threadIdx.x + 32];
			__threadfence_block();

		case 32:
			imageSum[threadIdx.x] += imageSum[threadIdx.x + 16];
			__threadfence_block();

		case 16:
			imageSum[threadIdx.x] += imageSum[threadIdx.x + 8];
			__threadfence_block();

		case 8:
			imageSum[threadIdx.x] += imageSum[threadIdx.x + 4];
			__threadfence_block();

		case 4:
			imageSum[threadIdx.x] += imageSum[threadIdx.x + 2];
			__threadfence_block();

		case 2:
			imageSum[threadIdx.x] += imageSum[threadIdx.x + 1];
		}
	}

	if (threadIdx.x == 0) {
		gpu_blockRecipEnergy[gridOffset + blockIdx.x] = imageSum[0];
		__threadfence();
		lastBlock = (atomicInc(&BlocksDone, gridDim.x) == gridDim.x - 1);
	}       //end if

	__syncthreads();

	if (lastBlock) {
		//If this thread corresponds to a valid block
		if (threadIdx.x < gridDim.x) {
			imageSum[threadIdx.x] = gpu_blockRecipEnergy[gridOffset + threadIdx.x];
		}

		for (int i = blockDim.x; threadIdx.x + i < gridDim.x; i += blockDim.x) {
			if (threadIdx.x + i < gridDim.x) {
				imageSum[threadIdx.x] += gpu_blockRecipEnergy[gridOffset + threadIdx.x + i];
			}       //end if
		}       //end for

		__syncthreads();
		offset = 1 << (int) __log2f((float) min(blockDim.x, gridDim.x));

		for (int i = offset; i > 0; i >>= 1) {
			if (threadIdx.x < i && threadIdx.x + i < min(blockDim.x, gridDim.x)) {
					imageSum[threadIdx.x] += imageSum[threadIdx.x + i];
			}       //end if

			__syncthreads();
		}       //end for

		if (threadIdx.x == 0) {
			BlocksDone = 0;
			gpu_blockRecipEnergy[gridOffset] = imageSum[0];
#ifdef debug
			printf("box: %d, TotalRecipEnergy: %lf\n", box, gpu_blockRecipEnergy[gridOffset]);
#endif
		}
	}
}

__global__ void MolReciprocalDisplaceGPU(
		const uint pStart, const uint pLen, const uint molIndex,
		const XYZ shift, double *gpu_xCoords, double *gpu_yCoords,
		double *gpu_zCoords, double *gpu_atomCharge, double *gpu_imageReal,
		double *gpu_imageImaginary, double *gpu_imageRealRef,
		double *gpu_imageImaginaryRef, double *gpu_kx, double *gpu_ky,
		double *gpu_kz, double *gpu_newMolPos, double *gpu_prefact,
		const uint box, const uint imageOffset, const uint gridOffset,
		double* gpu_blockRecipNew,
		const uint imageSize, const double xAxis, const double yAxis,
		const double zAxis, const int maxMolLen)
{
	__shared__ double imageSumNew[MAXTHREADSPERBLOCK];
	__shared__ bool	lastBlock;
	extern __shared__ double Array[];

	int threadId = blockDim.x * blockIdx.x + threadIdx.x;
	int imageIndex = threadId + imageOffset;

	double kx = 0.0;
	double ky = 0.0;
	double kz = 0.0;
	double sineOld, cosineOld, sineNew, cosineNew;
	imageSumNew[threadIdx.x] = 0.0;
	double sumRealNew = 0.0;
	double sumImaginaryNew = 0.0;
	double sumRealOld = 0.0;
	double sumImaginaryOld = 0.0;
	double dotProductNew = 0.0;
	double dotProductOld = 0.0;
	double prefact = 0.0;

	if (threadIdx.x == 0)
	{
		for (uint p = 0; p < pLen; ++p)
		{
			Array[p] = gpu_xCoords[pStart + p] + shift.x;
			Array[p + pLen] = gpu_yCoords[pStart + p] + shift.y;
			Array[p + 2*pLen] = gpu_zCoords[pStart + p] + shift.z;
			WrapPBC(Array[p], xAxis);
			WrapPBC(Array[pLen + p], yAxis);
			WrapPBC(Array[2*pLen + p], zAxis);
			if (threadId == 0)
			{
				int offset = maxMolLen * 3 * box;		//3 is for coordinates x, y, z
				gpu_newMolPos[p + offset] = Array[p];
				gpu_newMolPos[p + pLen + offset] = Array[p + pLen];
				gpu_newMolPos[p + 2 * pLen + offset] = Array[p + 2 * pLen];
			}
		}
		__threadfence();
	}
	__syncthreads();

	if (threadId < imageSize)
	{
		prefact = gpu_prefact[imageIndex];
		kx = gpu_kx[imageIndex];
		ky = gpu_ky[imageIndex];
		kz = gpu_kz[imageIndex];
		//if use loop unroll here, addition of sumReal & sumImaginary should be using reduction
		for (uint p = 0; p < pLen; ++p)
		{
			int atomIndex = p + pStart;
			dotProductOld = DotProductGPU(kx, ky, kz, gpu_xCoords, gpu_yCoords, gpu_zCoords, atomIndex);
			dotProductNew = DotProductGPU(kx, ky, kz, &Array[0], &Array[pLen], &Array[2*pLen], p);

			sinecosine(dotProductOld, sineOld, cosineOld);
			sinecosine(dotProductNew, sineNew, cosineNew);
			sumRealNew += (gpu_atomCharge[atomIndex] * cosineNew);
			sumImaginaryNew += (gpu_atomCharge[atomIndex] * sineNew);

			sumRealOld += (gpu_atomCharge[atomIndex] * cosineOld);
			sumImaginaryOld += (gpu_atomCharge[atomIndex] * sineOld);
#ifdef debug
			if (imageIndex == 1 + imageOffset)
			{
				printf("kx: %lf, ky: %lf, kz: %lf, dotProductNew: %lf, dotProductOld: %lf\n", kx, ky, kz, dotProductNew, dotProductOld);
				printf("GPU box: %d, mol: %d, atom: %d, x: %lf, y: %lf, z: %lf, newX: %lf, newY: %lf, newZ: %lf\n",
						box, molIndex, p, gpu_xCoords[atomIndex], gpu_yCoords[atomIndex], gpu_zCoords[atomIndex],
						Array[p], Array[p + pLen], Array[p + 2*pLen]);
			}
#endif
		}
		double sumRnew = gpu_imageRealRef[imageIndex] = gpu_imageReal[imageIndex] - sumRealOld + sumRealNew;
		double sumInew = gpu_imageImaginaryRef[imageIndex] = gpu_imageImaginary[imageIndex] - sumImaginaryOld + sumImaginaryNew;

		imageSumNew[threadIdx.x] += (sumRnew * sumRnew + sumInew * sumInew) * prefact;
#ifdef debug
		if (threadId == 0)
			printf("box: %d, image: %d, sumRnew: %lf, sumInew: %lf, sumRold: %lf, sumIold: %lf, cacheRold: %lf, cacheIold: %lf, "
					"newReal: %lf, newImaginary: %lf, prefact: %lf\n", box, threadId, sumRealNew, sumImaginaryNew, sumRealOld,
					sumImaginaryOld, gpu_imageReal[imageIndex], gpu_imageImaginary[imageIndex], sumRnew, sumInew, prefact);
#endif
	}

	//after the calculation of each thread, implement reduction on gpu_imageReal
	//and gpu_imageImaginary within the block, and then another reduction between blocks
	__syncthreads();
	// add data
	int offset = 1 << (int) __log2f((float) blockDim.x);

	if (blockDim.x < MAXTHREADSPERBLOCK) {
		if ((threadIdx.x + offset) < imageSize % MAXTHREADSPERBLOCK) {
			imageSumNew[threadIdx.x] += imageSumNew[threadIdx.x + offset];
		}

		__syncthreads();
	}

	for (int i = offset >> 1; i > 32; i >>= 1) {
		if (threadIdx.x < i) {
			imageSumNew[threadIdx.x] += imageSumNew[threadIdx.x + i];
		}           //end if

		__syncthreads();
	}           //end for

	if (threadIdx.x < 32) {
		offset = min(offset, 64);

		switch (offset) {
		case 64:
			imageSumNew[threadIdx.x] += imageSumNew[threadIdx.x + 32];
			__threadfence_block();

		case 32:
			imageSumNew[threadIdx.x] += imageSumNew[threadIdx.x + 16];
			__threadfence_block();

		case 16:
			imageSumNew[threadIdx.x] += imageSumNew[threadIdx.x + 8];
			__threadfence_block();

		case 8:
			imageSumNew[threadIdx.x] += imageSumNew[threadIdx.x + 4];
			__threadfence_block();

		case 4:
			imageSumNew[threadIdx.x] += imageSumNew[threadIdx.x + 2];
			__threadfence_block();

		case 2:
			imageSumNew[threadIdx.x] += imageSumNew[threadIdx.x + 1];
		}
	}

	if (threadIdx.x == 0) {
		gpu_blockRecipNew[gridOffset + blockIdx.x] = imageSumNew[0];
		__threadfence();
		lastBlock = (atomicInc(&BlocksDone, gridDim.x) == gridDim.x - 1);
	}       //end if

	__syncthreads();

	if (lastBlock) {
		//If this thread corresponds to a valid block
		if (threadIdx.x < gridDim.x) {
			imageSumNew[threadIdx.x] = gpu_blockRecipNew[gridOffset + threadIdx.x];
		}

		for (int i = blockDim.x; threadIdx.x + i < gridDim.x; i += blockDim.x) {
			if (threadIdx.x + i < gridDim.x) {
				imageSumNew[threadIdx.x] += gpu_blockRecipNew[ gridOffset + threadIdx.x + i];
			}       //end if
		}       //end for

		__syncthreads();
		offset = 1 << (int) __log2f((float) min(blockDim.x, gridDim.x));

		for (int i = offset; i > 0; i >>= 1) {
			if (threadIdx.x < i && threadIdx.x + i < min(blockDim.x, gridDim.x)) {
					imageSumNew[threadIdx.x] += imageSumNew[threadIdx.x + i];
			}       //end if

			__syncthreads();
		}       //end for

		if (threadIdx.x == 0) {
			BlocksDone = 0;
			gpu_blockRecipNew[gridOffset] = imageSumNew[0];
#ifdef debug
			printf("box: %d, RecipEnergyNew: %lf\n", box, gpu_blockRecipNew[gridOffset]);
#endif
		}
	}
}

__global__ void MolReciprocalRotateGPU(
		const uint pStart, const uint pLen, const uint molIndex,
		const XYZ center, double *gpu_xCoords, double *gpu_yCoords,
		double *gpu_zCoords, double *gpu_atomCharge, double *gpu_imageReal,
		double *gpu_imageImaginary, double *gpu_imageRealRef,
		double *gpu_imageImaginaryRef, double *gpu_kx, double *gpu_ky,
		double *gpu_kz, double *gpu_newMolPos, double *gpu_prefact,
		const uint box, const uint imageOffset, const uint gridOffset,
		double* gpu_blockRecipNew,
		const uint imageSize, const double xAxis, const double yAxis,
		const double zAxis, const int maxMolLen, double *matrix)
{
	__shared__ double imageSumNew[MAXTHREADSPERBLOCK];
	__shared__ bool	lastBlock;
	extern __shared__ double Array[];

	int threadId = blockDim.x * blockIdx.x + threadIdx.x;
	int imageIndex = threadId + imageOffset;

	double kx = 0.0;
	double ky = 0.0;
	double kz = 0.0;
	double sineOld, cosineOld, sineNew, cosineNew;
	imageSumNew[threadIdx.x] = 0.0;
	double sumRealNew = 0.0;
	double sumImaginaryNew = 0.0;
	double sumRealOld = 0.0;
	double sumImaginaryOld = 0.0;
	double dotProductNew = 0.0;
	double dotProductOld = 0.0;
	double xHalfAxis = xAxis/2;
	double yHalfAxis = yAxis/2;
	double zHalfAxis = zAxis/2;
	double prefact;

	if (threadIdx.x == 0)
	{
		for (uint p = 0; p < pLen; ++p)
		{
			Array[p] = gpu_xCoords[pStart + p];
			Array[p + pLen] = gpu_yCoords[pStart + p];
			Array[p + 2*pLen] = gpu_zCoords[pStart + p];
			UnwrapPBC(Array[p], center.x, xAxis, xHalfAxis);
			UnwrapPBC(Array[p + pLen], center.y, yAxis, yHalfAxis);
			UnwrapPBC(Array[p + 2*pLen], center.z, zAxis, zHalfAxis);
			Array[p] += (-1.0 * center.x);
			Array[p + pLen] += (-1.0 * center.y);
			Array[p + 2*pLen] += (-1.0 * center.z);
			MatrixApply(Array[p], Array[p + pLen], Array[p + 2*pLen], matrix);
			Array[p] += (center.x);
			Array[p + pLen] += (center.y);
			Array[p + 2*pLen] += (center.z);
			WrapPBC(Array[p], xAxis);
			WrapPBC(Array[p + pLen], yAxis);
			WrapPBC(Array[p + 2*pLen], zAxis);

			int offset = maxMolLen * 3 * box;		//3 is for coordinates x, y, z
			gpu_newMolPos[p + offset] = Array[p];
			gpu_newMolPos[p + pLen + offset] = Array[p + pLen];
			gpu_newMolPos[p + 2 * pLen + offset] = Array[p + 2 * pLen];
		}
		__threadfence();
	}
	__syncthreads();

	if (threadId < imageSize)
	{
		prefact = gpu_prefact[imageIndex];
		kx = gpu_kx[imageIndex];
		ky = gpu_ky[imageIndex];
		kz = gpu_kz[imageIndex];
		//if use loop unroll here, addition of sumReal & sumImaginary should be using reduction
		for (uint p = 0; p < pLen; ++p)
		{
			int atomIndex = p + pStart;
			dotProductOld = DotProductGPU(kx, ky, kz, gpu_xCoords, gpu_yCoords, gpu_zCoords, atomIndex);
			dotProductNew = DotProductGPU(kx, ky, kz, &Array[0], &Array[pLen], &Array[2*pLen], p);

			sinecosine(dotProductOld, sineOld, cosineOld);
			sinecosine(dotProductNew, sineNew, cosineNew);
			sumRealNew += (gpu_atomCharge[atomIndex] * cosineNew);
			sumImaginaryNew += (gpu_atomCharge[atomIndex] * sineNew);

			sumRealOld += (gpu_atomCharge[atomIndex] * cosineOld);
			sumImaginaryOld += (gpu_atomCharge[atomIndex] * sineOld);
#ifdef debug
			if (imageIndex == 1 + imageOffset)
			{
				printf("kx: %lf, ky: %lf, kz: %lf, dotProductNew: %lf, dotProductOld: %lf\n", kx, ky, kz, dotProductNew, dotProductOld);
				printf("GPU box: %d, mol: %d, atom: %d, x: %lf, y: %lf, z: %lf, newX: %lf, newY: %lf, newZ: %lf\n",
						box, molIndex, p, gpu_xCoords[atomIndex], gpu_yCoords[atomIndex], gpu_zCoords[atomIndex],
						Array[p], Array[p + pLen], Array[p + 2*pLen]);
			}
#endif
		}
		double sumRold = gpu_imageReal[imageIndex];
		double sumIold = gpu_imageImaginary[imageIndex];
		double sumRnew = gpu_imageRealRef[imageIndex] = sumRold - sumRealOld + sumRealNew;
		double sumInew = gpu_imageImaginaryRef[imageIndex] = sumIold - sumImaginaryOld + sumImaginaryNew;

		imageSumNew[threadIdx.x] += (sumRnew * sumRnew + sumInew * sumInew) * prefact;
#ifdef debug
		if (threadId == 0)
			printf("box: %d, image: %d, sumRnew: %lf, sumInew: %lf, sumRold: %lf, sumIold: %lf, cacheRold: %lf, cacheIold: %lf, cacheRnew: %lf, cacheInew: %lf\n",
				box, threadId, sumRealNew, sumImaginaryNew, sumRealOld, sumImaginaryOld, sumRold, sumIold, sumRnew, sumInew);
#endif
	}

	//after the calculation of each thread, implement reduction on gpu_imageReal
	//and gpu_imageImaginary within the block, and then another reduction between blocks
	__syncthreads();
	// add data
	int offset = 1 << (int) __log2f((float) blockDim.x);

	if (blockDim.x < MAXTHREADSPERBLOCK) {
		if ((threadIdx.x + offset) < imageSize % MAXTHREADSPERBLOCK) {
			imageSumNew[threadIdx.x] += imageSumNew[threadIdx.x + offset];
//			imageSumOld[threadIdx.x] += imageSumOld[threadIdx.x + offset];
		}

		__syncthreads();
	}

	for (int i = offset >> 1; i > 32; i >>= 1) {
		if (threadIdx.x < i) {
			imageSumNew[threadIdx.x] += imageSumNew[threadIdx.x + i];
//			imageSumOld[threadIdx.x] += imageSumOld[threadIdx.x + i];
		}           //end if

		__syncthreads();
	}           //end for

	if (threadIdx.x < 32) {
		offset = min(offset, 64);

		switch (offset) {
		case 64:
			imageSumNew[threadIdx.x] += imageSumNew[threadIdx.x + 32];
//			imageSumOld[threadIdx.x] += imageSumOld[threadIdx.x + 32];
			__threadfence_block();

		case 32:
			imageSumNew[threadIdx.x] += imageSumNew[threadIdx.x + 16];
//			imageSumOld[threadIdx.x] += imageSumOld[threadIdx.x + 16];
			__threadfence_block();

		case 16:
			imageSumNew[threadIdx.x] += imageSumNew[threadIdx.x + 8];
//			imageSumOld[threadIdx.x] += imageSumOld[threadIdx.x + 8];
			__threadfence_block();

		case 8:
			imageSumNew[threadIdx.x] += imageSumNew[threadIdx.x + 4];
//			imageSumOld[threadIdx.x] += imageSumOld[threadIdx.x + 4];
			__threadfence_block();

		case 4:
			imageSumNew[threadIdx.x] += imageSumNew[threadIdx.x + 2];
//			imageSumOld[threadIdx.x] += imageSumOld[threadIdx.x + 2];
			__threadfence_block();

		case 2:
			imageSumNew[threadIdx.x] += imageSumNew[threadIdx.x + 1];
//			imageSumOld[threadIdx.x] += imageSumOld[threadIdx.x + 1];
		}
	}

	if (threadIdx.x == 0) {
		gpu_blockRecipNew[gridOffset + blockIdx.x] = imageSumNew[0];
//		gpu_blockRecipOld[blockIdx.x] = imageSumOld[0];
		__threadfence();
		lastBlock = (atomicInc(&BlocksDone, gridDim.x) == gridDim.x - 1);
	}       //end if

	__syncthreads();

	if (lastBlock) {
		//If this thread corresponds to a valid block
		if (threadIdx.x < gridDim.x) {
			imageSumNew[threadIdx.x] = gpu_blockRecipNew[gridOffset + threadIdx.x];
//			imageSumOld[threadIdx.x] = gpu_blockRecipOld[threadIdx.x];
		}

		for (int i = blockDim.x; threadIdx.x + i < gridDim.x; i += blockDim.x) {
			if (threadIdx.x + i < gridDim.x) {
				imageSumNew[threadIdx.x] += gpu_blockRecipNew[gridOffset + threadIdx.x + i];
//				imageSumOld[threadIdx.x] += gpu_blockRecipOld[threadIdx.x + i];
			}       //end if
		}       //end for

		__syncthreads();
		offset = 1 << (int) __log2f((float) min(blockDim.x, gridDim.x));

		for (int i = offset; i > 0; i >>= 1) {
			if (threadIdx.x < i && threadIdx.x + i < min(blockDim.x, gridDim.x)) {
					imageSumNew[threadIdx.x] += imageSumNew[threadIdx.x + i];
//					imageSumOld[threadIdx.x] += imageSumOld[threadIdx.x + i];
			}       //end if

			__syncthreads();
		}       //end for

		if (threadIdx.x == 0) {
			BlocksDone = 0;
			gpu_blockRecipNew[gridOffset] = imageSumNew[0];
#ifdef debug
			printf("box: %d, RecipEnergyNew: %lf\n", box, gpu_blockRecipNew[gridOffset]);
#endif
		}
	}
}

__global__ void AcceptUpdateGPU(
		const uint pStart, const uint pLen, const uint molIndex,
		double *gpu_xCoords, double *gpu_yCoords,
		double *gpu_zCoords, double *gpu_newMolPos,
		const int maxMolLen, const int box)
{
	int offset = maxMolLen * 3 * box;	//3 is for coordinates x, y, z
	int index;
	//update coordinates
	for (int p = pStart; p < pStart + pLen; p++)
	{
		index = p - pStart + offset;
		gpu_xCoords[p] = gpu_newMolPos[index];
		gpu_yCoords[p] = gpu_newMolPos[index + pLen];
		gpu_zCoords[p] = gpu_newMolPos[index + 2 * pLen];
	}
}
