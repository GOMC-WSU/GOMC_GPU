/*******************************************************************************
GPU OPTIMIZED MONTE CARLO (GOMC) BETA 0.97 (GPU version)
Copyright (C) 2015  GOMC Group

A copy of the GNU General Public License can be found in the COPYRIGHT.txt
along with this program, also can be found at <http://www.gnu.org/licenses/>.
********************************************************************************/

#include "CalculateEnergy.h"        //header for this
#include "EnergyTypes.h"            //Energy structs
#include "EnsemblePreprocessor.h"   //Flags
#include "../lib/BasicTypes.h"             //uint
#include "System.h"                 //For init
#include "StaticVals.h"             //For init
#include "Forcefield.h"             //
#include "MoleculeLookup.h"
#include "MoleculeKind.h"
#include "Coordinates.h"
#include "BoxDimensions.h"
#include "cbmc/TrialMol.h"
#include "../lib/GeomLib.h"
#include <cassert>
#include <cuda_runtime.h> 
#include <cuda.h>
//
//    CalculateEnergy.cpp
//    Energy Calculation functions for Monte Carlo simulation
//    Calculates using const references to a particular Simulation's members
//    Brock Jackman Sep. 2013
//
//    Updated to use radial-based intermolecular pressure
//    Jason Mick    Feb. 2014
//

//  
// USEFUL TO DEBUG GPU CODE
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


using namespace geom;

__device__ int BlockNum = -1;
__device__ unsigned int BlocksDone = 0;

CalculateEnergy::CalculateEnergy(const StaticVals& stat, const System& sys) :
forcefield(stat.forcefield), mols(stat.mol), currentCoords(sys.coordinates),
currentCOM(sys.com),
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
{}


void CalculateEnergy::Init()
{
   for(uint m = 0; m < mols.count; ++m)
   {
      const MoleculeKind& molKind = mols.GetKind(m);
      for(uint a = 0; a < molKind.NumAtoms(); ++a)
      {
         particleKind.push_back(molKind.AtomKind(a));
         particleMol.push_back(m);
      }
   }
}



SystemPotential CalculateEnergy::SystemTotal() 
{
  

   SystemPotential pot = SystemInterGPU();// gpu call 

  if (forcefield.useLRC)
      FullTailCorrection(pot, currentAxes);


   //system intra
   for (uint box = 0; box < BOX_TOTAL; ++box)
   {
      double bondEn = 0.0;
      double nonbondEn = 0.0;
      MoleculeLookup::box_iterator thisMol = molLookup.BoxBegin(box);
      MoleculeLookup::box_iterator end = molLookup.BoxEnd(box);
      while (thisMol != end)
      {
         MoleculeIntra(bondEn, nonbondEn, *thisMol, box);
         ++thisMol;
      }
      pot.boxEnergy[box].intraBond = bondEn;
      pot.boxEnergy[box].intraNonbond = nonbondEn;
   }
  
   pot.Total();
   return pot;
}

SystemPotential CalculateEnergy::SystemInter
(SystemPotential potential,
 const XYZArray& coords,
 const XYZArray& com,
 const BoxDimensions& boxAxes) const
{
   for (uint box = 0; box < BOX_TOTAL; ++box)
   {
      double energy, virial;
      energy = virial = 0.0;
      MoleculeLookup::box_iterator thisMol = molLookup.BoxBegin(box);
      MoleculeLookup::box_iterator end = molLookup.BoxEnd(box);
      while (thisMol != end)
      {
         uint m1 = *thisMol;
         MoleculeKind const& thisKind = mols.GetKind(m1);
         //evaluate interaction with all molecules after this
         MoleculeLookup::box_iterator otherMol = thisMol;
         ++otherMol;
         while (otherMol != end)
         {
            uint m2 = *otherMol;
            XYZ fMol;
            MoleculeKind const& otherKind = mols.GetKind(m2);
            for (uint i = 0; i < thisKind.NumAtoms(); ++i)
            {
               for (uint j = 0; j < otherKind.NumAtoms(); ++j)
               {
                  XYZ virComponents;
                  double distSq;
                  if (boxAxes.InRcut(distSq, virComponents,
                     coords, mols.start[m1] + i,
                     mols.start[m2] + j, box))
                  {

                     double partVirial = 0.0;

                     forcefield.particles.CalcAdd
                        (energy, partVirial, distSq,
                        thisKind.AtomKind(i), otherKind.AtomKind(j));

                     //Add to our pressure components.
                     fMol += (virComponents * partVirial);
                  }
               }
            }

            //Pressure is wrt center of masses of two molecules.
            virial -= geom::Dot(fMol,
               currentAxes.MinImage
               (com.Difference(m1, m2), box));

            ++otherMol;
         }
         ++thisMol;
      }
      //Correct TC for current density
      double densityRatio = currentAxes.volume[box] * boxAxes.volInv[box];
      potential.boxEnergy[box].inter = energy;
      potential.boxEnergy[box].tc *= densityRatio;
      //Account for that virial is 
      potential.boxVirial[box].inter = virial;
      potential.boxVirial[box].tc *= densityRatio;
   }
   potential.Total();
   return potential;
}

double CalculateEnergy::CheckMoleculeInter(uint molIndex, uint box) const
{
   double result = 0.0;
   MoleculeLookup::box_iterator molInBox = molLookup.BoxBegin(box);
   MoleculeLookup::box_iterator end = molLookup.BoxEnd(box);
   MoleculeKind const& thisKind = mols.GetKind(molIndex);

   uint thisStart = mols.MolStart(molIndex);
   uint thisEnd = thisStart + thisKind.NumAtoms();
   //looping over all molecules in box
   while (molInBox != end)
   {
      uint otherMol = *molInBox;
      //except itself
      if (otherMol != molIndex)
      {
         MoleculeKind const& otherKind = mols.GetKind(otherMol);
         uint otherStart = mols.MolStart(otherMol);
         uint otherEnd = otherStart + otherKind.NumAtoms();
         //compare all particle pairs
         for (uint j = otherStart; j < otherEnd; ++j)
         {
            uint kindJ = otherKind.AtomKind(j - otherStart);
            for (uint i = thisStart; i < thisEnd; ++i)
            {
               double distSq;
               int kindI = thisKind.AtomKind(i - thisStart);
               if (currentAxes.InRcut(distSq, currentCoords, i, j, box))
               {
                  result += forcefield.particles.CalcEn(distSq, kindI, kindJ);
               }
            }
         }
      }
      ++molInBox;
   }
   return result;
}

SystemPotential CalculateEnergy::BoxNonbonded
(SystemPotential potential, 
 const uint box,
 const XYZArray& coords, 
 XYZArray const& com,
 const BoxDimensions& boxAxes) const
{
   Intermolecular inter;
   MoleculeLookup::box_iterator thisMol = molLookup.BoxBegin(box);
   MoleculeLookup::box_iterator end = molLookup.BoxEnd(box);
   while (thisMol != end)
   {
      uint m1 = *thisMol;
      MoleculeKind const& thisKind = mols.GetKind(m1);
      //evaluate interaction with all molecules after this
      MoleculeLookup::box_iterator otherMol = thisMol;
      ++otherMol;
      while (otherMol != end)
      {
	 uint m2 = *otherMol;
	 XYZ fMol;
	 MoleculeKind const& otherKind = mols.GetKind(m2);
	 for (uint i = 0; i < thisKind.NumAtoms(); ++i)
         {
	    for (uint j = 0; j < otherKind.NumAtoms(); ++j)
	    {
	       XYZ virComponents;
	       double distSq;
	       if (boxAxes.InRcut(distSq, virComponents,
				  coords, mols.start[m1] + i,
				  mols.start[m2] + j, box))
               {

		  double partVirial = 0.0;
		  
		  forcefield.particles.CalcAdd
		     (inter.energy, partVirial, distSq,
		      thisKind.AtomKind(i), otherKind.AtomKind(j));

		  //Add to our pressure components.
		  fMol += (virComponents * partVirial);
	       }
	    }
	 }
	 
	 //Pressure is wrt center of masses of two molecules.
	 inter.virial -= geom::Dot(fMol,
			     boxAxes.MinImage
			     (com.Difference(m1, m2), box));
	 
	 ++otherMol;
      }
      ++thisMol;
   }
   //Correct TC for current density
   double densityRatio = currentAxes.volume[box] * boxAxes.volInv[box];
   //printf("%lf\n",inter.energy);
   potential.boxEnergy[box].inter = inter.energy;
   potential.boxEnergy[box].tc *= densityRatio;
   //Account for that virial is 
   potential.boxVirial[box].inter = inter.virial;
   potential.boxVirial[box].tc *= densityRatio;

   potential.Total();

   return potential;
}

SystemPotential CalculateEnergy::SystemNonbonded
(SystemPotential potential,
 const XYZArray& coords,
 const XYZArray& com,
 const BoxDimensions& boxAxes) const
{
   for (uint box = 0; box < BOX_TOTAL; ++box)
   {
      potential = BoxNonbonded(potential, box, coords, com, boxAxes);
   }
   potential.Total();
   return potential;
}


Intermolecular CalculateEnergy::MoleculeInter(const XYZArray& molCoords,
   const uint molIndex,
   const uint box,
   XYZ const*const newCOM) const
{
   bool hasNewCOM = !(newCOM == NULL);
   Intermolecular result;
   MoleculeLookup::box_iterator otherMol = molLookup.BoxBegin(box);
   MoleculeLookup::box_iterator end = molLookup.BoxEnd(box);
   uint partStartMolI, partLenMolI, partStartMolJ, partLenMolJ, pkI, pkJ,
      pOldI, pNewI, pJ;
   partStartMolI = partLenMolI = partStartMolJ = partLenMolJ = pkI = pkJ =
      pOldI = pNewI = pJ = 0;
   mols.GetRangeStartLength(partStartMolI, partLenMolI, molIndex);
   const MoleculeKind& thisKind = mols.GetKind(molIndex);
   //looping over all molecules in box
   while (otherMol != end)
   {
      uint m2 = *otherMol;
      XYZ fMolO, fMolN;
      //except itself
      if (m2 != molIndex)
      {
         const MoleculeKind& otherKind = mols.GetKind(m2);
         //compare all particle pairs
         for (uint i = 0; i < partLenMolI; ++i)
         {
            pOldI = i + partStartMolI;
            pkI = thisKind.AtomKind(i);
            mols.GetRangeStartLength(partStartMolJ, partLenMolJ, m2);
            for (uint j = 0; j < partLenMolJ; ++j)
            {
               XYZ virComponents;
               double distSq;
               pJ = j + partStartMolJ;
               pkJ = otherKind.AtomKind(j);
               //Subtract old energy
               if (currentAxes.InRcut(distSq, virComponents,
                  currentCoords, pOldI, pJ, box))
               {
                  double partVirial = 0.0;

                  forcefield.particles.CalcSub
                     (result.energy, partVirial, distSq, pkI, pkJ);

                  fMolO += virComponents * partVirial;
               }
               //Add new energy
               if (currentAxes.InRcut(distSq, virComponents, molCoords, i,
                  currentCoords, pJ, box))
               {
                  double partVirial = 0.0;

                  forcefield.particles.CalcAdd(result.energy, partVirial,
                     distSq, pkI, pkJ);

                  //Add to our pressure components.
                  fMolN += (virComponents * partVirial);
               }
            }
         }
         //Pressure is wrt center of masses of two molecules.
         result.virial -= geom::Dot(fMolO,
            currentAxes.MinImage
            (currentCOM.Difference(molIndex, m2), box));
         if (hasNewCOM)
         {
            result.virial -= geom::Dot(fMolN,
               currentAxes.MinImage
               (*newCOM - currentCOM.Get(m2), box));
         }
         else
         {
            result.virial -= geom::Dot(fMolN,
               currentAxes.MinImage
               (currentCOM.Difference(molIndex, m2), box));
         }
      }

      ++otherMol;
   }
   return result;
}

void CalculateEnergy::ParticleNonbonded(double* energy,
                                        const cbmc::TrialMol& trialMol,
                                        XYZArray const& trialPos,
                                        const uint partIndex,
                                        const uint box,
                                        const uint trials) const
{
   if (box >= BOXES_WITH_U_B)
      return;
   
   const MoleculeKind& kind = trialMol.GetKind();
   //loop over all partners of the trial particle
   const uint* partner = kind.sortedNB.Begin(partIndex);
   const uint* end = kind.sortedNB.End(partIndex);
   while (partner != end)
   {
      if (trialMol.AtomExists(*partner))
      {
         for (uint i = 0; i < trialPos.Count(); ++i)
         {
            double distSq;
            if (currentAxes.InRcut(distSq, trialPos, i, trialMol.GetCoords(),
               *partner, box))
            {
               energy[i] += forcefield.particles.CalcEn(distSq,
                  kind.AtomKind(partIndex), kind.AtomKind(*partner));
            }
         }
      }
      ++partner;
   }
}




void CalculateEnergy::GetParticleEnergyGPU(uint box, double * en, XYZArray positions, int numAtoms, int mOff, int CurrentPos, int MolKind)
{
	
	if (box >= BOXES_WITH_U_NB)
      return;


	int ThreadsPerBlock1=0;
	int BlocksPerGrid1=0;

	/*if (p->calc.AtomCount[box] < MAXTHREADSPERBLOCK)
	ThreadsPerBlock1 = p->calc.AtomCount[box];
	else*/
	ThreadsPerBlock1 = MAXTHREADSPERBLOCK;
	if(ThreadsPerBlock1 == 0)
		ThreadsPerBlock1 = 1;
	BlocksPerGrid1 = ((AtomCount[box])+ ThreadsPerBlock1 - 1) / ThreadsPerBlock1;

	if (BlocksPerGrid1==0) BlocksPerGrid1=1;

	double * dev_EnergyContrib;
	double FinalEnergyNVirial[2];

	cudaMalloc((void**) &dev_EnergyContrib, 4* BlocksPerGrid1 * sizeof(double));
	
	for (uint i = 0; i < positions.count; ++i)
	{
		
		Gpu_CalculateParticleInter <<<BlocksPerGrid1,ThreadsPerBlock1>>>(
			i,
			Gpu_kIndex,
			Gpu_sigmaSq, 
			Gpu_epsilon_cn, 
			Gpu_nOver6, 
			Gpu_epsilon_cn_6,
			Gpu_x,
			Gpu_y,
			Gpu_z,
			positions.x[i],
			positions.y[i],
			positions.z[i], 
			Gpu_atomKinds,
			numAtoms,mOff,
			CurrentPos,Gpu_start,
			currentAxes.axis.x[box],
			currentAxes.axis.y[box],
			currentAxes.axis.z[box],
			currentAxes.halfAx.x[box],
			currentAxes.halfAx.y[box],
			currentAxes.halfAx.z[box],
			(box==0)?0:AtomCount[0],
			AtomCount[box],
			forcefield.particles.NumKinds(),
			currentAxes.rCut,MolKind,
			currentAxes.rCutSq,
			dev_EnergyContrib,
			
			Gpu_partn
			);

		cudaMemcpy(&FinalEnergyNVirial, dev_EnergyContrib, 2*sizeof(double), cudaMemcpyDeviceToHost);

		en[i] += FinalEnergyNVirial[0];
	
		
		cudaDeviceSynchronize();
		cudaError_t  code = cudaGetLastError();
		if (code != cudaSuccess) 
		{ printf ("Cuda error end of transfer energy calc -- %s\n", cudaGetErrorString(code)); 
		}	

	}
	cudaFree(dev_EnergyContrib);
	
	

}




//! Calculates Nonbonded intra energy for candidate positions in trialPos
void CalculateEnergy::ParticleInter
(uint partIndex, const XYZArray& trialPos, double* en, uint molIndex, uint box) const
{
   double distSq;
   MoleculeLookup::box_iterator molInBox = molLookup.BoxBegin(box);
   MoleculeLookup::box_iterator end = molLookup.BoxEnd(box);
   MoleculeKind const& thisKind = mols.GetKind(molIndex);
   uint kindI = thisKind.AtomKind(partIndex);

   //looping over all molecules in box
   while (molInBox != end)
   {
      uint otherMol = *molInBox;
      //except itself
      if (otherMol != molIndex)
      {
         MoleculeKind const& otherKind = mols.GetKind(otherMol);
         uint otherStart = mols.MolStart(otherMol);
         uint otherLength = otherKind.NumAtoms();
         //compare all particle pairs
         for (uint j = 0; j < otherLength; ++j)
         {
            uint kindJ = otherKind.AtomKind(j);
            for (uint i = 0; i < trialPos.Count(); ++i)
            {
               if (currentAxes.InRcut(distSq, trialPos, i, currentCoords,
                  otherStart + j, box))
               {
				   double enr= forcefield.particles.CalcEn(distSq, kindI, kindJ);
                  en[i] += enr;

				
               }
            }
         }
      }
      ++molInBox;
   }
   return;
}



Intermolecular CalculateEnergy::MoleculeInter(const uint molIndex,
   const uint box) const
{
   Intermolecular result;

   MoleculeKind const& thisKind = mols.GetKind(molIndex);
   MoleculeLookup::box_iterator otherMol = molLookup.BoxBegin(box);
   MoleculeLookup::box_iterator end = molLookup.BoxEnd(box);

   uint partStartMolI, partLenMolI, partStartMolJ, partLenMolJ, pkI, pkJ,
      pI, pJ;
   partStartMolI = partLenMolI = partStartMolJ = partLenMolJ = pkI = pkJ =
      pI = pJ = 0;

   mols.GetRangeStartLength(partStartMolI, partLenMolI, molIndex);

   //looping over all molecules in box
   while (otherMol != end)
   {
      uint m2 = *otherMol;
      XYZ fMol;
      //except itself
      if (m2 != molIndex)
      {
         const MoleculeKind& otherKind = mols.GetKind(m2);
         //compare all particle pairs
         for (uint i = 0; i < partLenMolI; ++i)
         {
            pI = i + partStartMolI;
            pkI = thisKind.AtomKind(i);
            mols.GetRangeStartLength(partStartMolJ, partLenMolJ, m2);
            for (uint j = 0; j < partLenMolJ; ++j)
            {
               XYZ virComponents;
               double distSq;
               pJ = j + partStartMolJ;
               pkJ = otherKind.AtomKind(j);
               //Subtract old energy
               if (currentAxes.InRcut(distSq, virComponents,
                  currentCoords, pI, pJ, box))
               {
                  double partVirial = 0.0;

                  forcefield.particles.CalcAdd
                     (result.energy, partVirial, distSq, pkI, pkJ);

                  fMol += virComponents * partVirial;
               }
            }
         }
         //Pressure is wrt center of masses of two molecules.
         result.virial -= geom::Dot(fMol,
            currentAxes.MinImage
            (currentCOM.Difference(molIndex, m2), box));
      }
      ++otherMol;
   }
   return result;
}





double CalculateEnergy::MoleculeVirial(const uint molIndex,
                                       const uint box) const
{
   double virial = 0;
   if (box < BOXES_WITH_U_NB)
   {
      MoleculeLookup::box_iterator molInBox = molLookup.BoxBegin(box);
      MoleculeLookup::box_iterator end = molLookup.BoxEnd(box);
      
      const MoleculeKind& thisKind = mols.GetKind(molIndex);
      uint thisStart = mols.MolStart(molIndex);
      uint thisLength = thisKind.NumAtoms();
      //looping over all molecules in box
      while (molInBox != end)
      {
         uint otherMol = *molInBox;
         //except itself
         if(otherMol == molIndex)
         {
            ++molInBox;
            continue;
         }
         const MoleculeKind& otherKind = mols.GetKind(otherMol);
         XYZ forceOnMol;
         for (uint i = 0; i < thisLength; ++i)
         {
            uint kindI = thisKind.AtomKind(i);
            uint otherStart = mols.MolStart(otherMol);
            uint otherLength = otherKind.NumAtoms();
            for (uint j = 0; j < otherLength; ++j)
            {
               XYZ forceVector;
               double distSq;
               uint kindJ = otherKind.AtomKind(j);
               if (currentAxes.InRcut(distSq, forceVector, 
                                      currentCoords, thisStart + i,
                                      otherStart + j, box))
                  {
                     //sum forces between all particles in molecule pair
                     double mag = forcefield.particles.CalcVir(distSq,
                                                               kindI, kindJ);
                     forceOnMol += forceVector * mag;
                  }
            }
         }
         //correct for center of mass
         virial -= geom::Dot(forceOnMol, 
                             currentAxes.MinImage(currentCOM.Get(molIndex) - 
                                                  currentCOM.Get(otherMol),
                                                  box));
         ++molInBox;
      }
      double what = virial;
   }
   return virial;
}



//Calculates the change in the TC from adding numChange atoms of a kind
Intermolecular CalculateEnergy::MoleculeTailChange(const uint box,
                                                   const uint kind,
                                                   const bool add) const
{
   Intermolecular delta;
   
   if (box < BOXES_WITH_U_NB)
   {
   
      double sign = (add ? 1.0 : -1.0);
      uint mkIdxII = kind * mols.kindsCount + kind;
      for (uint j = 0; j < mols.kindsCount; ++j)
      {
         uint mkIdxIJ = j * mols.kindsCount + kind;
         double rhoDeltaIJ_2 = sign * 2.0 * 
            (double)(molLookup.NumKindInBox(j, box)) * currentAxes.volInv[box];
         delta.energy += mols.pairEnCorrections[mkIdxIJ] * rhoDeltaIJ_2;
         delta.virial -= mols.pairVirCorrections[mkIdxIJ] * rhoDeltaIJ_2;
      }
      //We already calculated part of the change for this type in the loop
      delta.energy += mols.pairEnCorrections[mkIdxII] * 
	 currentAxes.volInv[box];
      delta.virial -= mols.pairVirCorrections[mkIdxII] *
         currentAxes.volInv[box];
   }
   return delta;
}

//Calculates intramolecular energy of a full molecule
void CalculateEnergy::MoleculeIntra(double& bondEn,
                                    double& nonBondEn,
                                    const uint molIndex,
                                    const uint box) const
{
   MoleculeKind& molKind = mols.kinds[mols.kIndex[molIndex]];
   // *2 because we'll be storing inverse bond vectors
   XYZArray bondVec(molKind.bondList.count * 2);
   BondVectors(bondVec, molKind, molIndex, box);
   MolBond(bondEn, molKind, bondVec, box);
   MolAngle(bondEn, molKind, bondVec, box);
   MolDihedral(bondEn, molKind, bondVec, box);
   MolNonbond(nonBondEn, molKind, molIndex, box);
}

void CalculateEnergy::BondVectors(XYZArray & vecs,
                                  MoleculeKind const& molKind,
                                  const uint molIndex,
                                  const uint box) const
{
   for (uint i = 0; i < molKind.bondList.count; ++i)
   {
      uint p1 = mols.start[molIndex] + molKind.bondList.part1[i];
      uint p2 = mols.start[molIndex] + molKind.bondList.part2[i];
      XYZ dist = currentCoords.Difference(p2, p1);
      dist = currentAxes.MinImage(dist, box);

      //store inverse vectors at i+count
      vecs.Set(i, dist);
      vecs.Set(i + molKind.bondList.count, -dist.x, -dist.y, -dist.z);
   }
}


void CalculateEnergy::MolBond(double & energy,
                              MoleculeKind const& molKind,
                              XYZArray const& vecs,
                              const uint box) const
{
   if (box >= BOXES_WITH_U_B)
      return;
   for (uint i = 0; i < molKind.bondList.count; ++i)
   {
      energy += forcefield.bonds.Calc(molKind.bondList.kinds[i],
				      vecs.Get(i).Length());
   }
}

void CalculateEnergy::MolAngle(double & energy,
                               MoleculeKind const& molKind,
                               XYZArray const& vecs,
                               const uint box) const
{
   if (box >= BOXES_WITH_U_B)
      return;
   for (uint i = 0; i < molKind.angles.Count(); ++i)
   {
      double theta = Theta(vecs.Get(molKind.angles.GetBond(i, 0)),
         -vecs.Get(molKind.angles.GetBond(i, 1)));
      energy += forcefield.angles.Calc(molKind.angles.GetKind(i), theta);
   }
}

void CalculateEnergy::MolDihedral(double & energy,
                                  MoleculeKind const& molKind,
                                  XYZArray const& vecs,
                                  const uint box) const
{
   if (box >= BOXES_WITH_U_B)
      return;
   for (uint i = 0; i < molKind.dihedrals.Count(); ++i)
   {
      double phi = Phi(vecs.Get(molKind.dihedrals.GetBond(i, 0)),
         vecs.Get(molKind.dihedrals.GetBond(i, 1)),
         vecs.Get(molKind.dihedrals.GetBond(i, 2)));
      energy += forcefield.dihedrals.Calc(molKind.dihedrals.GetKind(i), phi);
   }
}

void CalculateEnergy::MolNonbond(double & energy,
                                 MoleculeKind const& molKind,
                                 const uint molIndex,
                                 const uint box) const
{
   if (box >= BOXES_WITH_U_B)
      return;
   
   double distSq;
   double virial; //we will throw this away
   for (uint i = 0; i < molKind.nonBonded.count; ++i)
   {
      uint p1 = mols.start[molIndex] + molKind.nonBonded.part1[i];
      uint p2 = mols.start[molIndex] + molKind.nonBonded.part2[i];
      if (currentAxes.InRcut(distSq, currentCoords, p1, p2, box))
      {
         forcefield.particles.CalcAdd(energy, virial, distSq,
                                      molKind.AtomKind
                                      (molKind.nonBonded.part1[i]),
                                      molKind.AtomKind
                                      (molKind.nonBonded.part2[i]));
      }
   }
}

//!Calculates energy and virial tail corrections for the box
void CalculateEnergy::FullTailCorrection(SystemPotential& pot,
					 const BoxDimensions& boxAxes) const
{   //if (box < BOXES_WITH_U_NB)
 //  for (uint box = 0; box < BOX_TOTAL; ++box)
	    for (uint box = 0; box < BOXES_WITH_U_NB; ++box)
   {
      double en = 0.0;
      double vir = 0.0;

      for (uint i = 0; i < mols.kindsCount; ++i)
      {
         uint numI = molLookup.NumKindInBox(i, box);
         for (uint j = 0; j < mols.kindsCount; ++j)
         {
            uint numJ = molLookup.NumKindInBox(j, box);
            en += mols.pairEnCorrections[i * mols.kindsCount + j] * numI * numJ
               * boxAxes.volInv[box];
            vir -= mols.pairVirCorrections[i * mols.kindsCount + j] *
               numI * numJ * boxAxes.volInv[box];
         }
      }
      pot.boxEnergy[box].tc = en;
      pot.boxVirial[box].tc = vir;
   }
}


// GPU code 
SystemPotential CalculateEnergy::SystemInterGPU() {
	int offset = 0;
	int ThreadsPerBlock1 = 0;
	int BlocksPerGrid1 = 0;
	ThreadsPerBlock1 = MAXTHREADSPERBLOCK;

	if(ThreadsPerBlock1 == 0)
	{ ThreadsPerBlock1 = 1; }

	BlocksPerGrid1 = ((MolCount[0] * (MolCount[0] - 1) / 2) + ThreadsPerBlock1 - 1) / ThreadsPerBlock1;

	if (BlocksPerGrid1 == 0)
	{ BlocksPerGrid1 = 1; }

	int XBlocks, YBlocks;

	if (BlocksPerGrid1 < MAXTBLOCKS) {
		XBlocks = 1;
		YBlocks = BlocksPerGrid1;
	} else {
		YBlocks = MAXTBLOCKS;
		XBlocks = (int)ceil((double)BlocksPerGrid1 / (double)MAXTBLOCKS);
	}

	dim3 blcks(XBlocks, YBlocks, 1);
	double * dev_EnergyContrib, * dev_VirialContrib;
	cudaMalloc((void**) &dev_EnergyContrib, 4 * XBlocks * YBlocks * sizeof(double));
	cudaMalloc((void**) &dev_VirialContrib,  4 * XBlocks * YBlocks * sizeof(double));
	double FinalEnergyNVirial[2];
	FinalEnergyNVirial[0] = 0.0;
	FinalEnergyNVirial[1] = 0.0;
	double NewEnergy1 = 0.0;
	double NewVirial1 = 0.0;
	//Box 0
	dim3 thrds (ThreadsPerBlock1, 1, 1);

	
	if ( MolCount[0] != 0) {
		Gpu_CalculateSystemInter <<< blcks, thrds, 0, stream0>>>(NoOfAtomsPerMol, Gpu_atomKinds,
			Gpu_kIndex,
			Gpu_sigmaSq,
			Gpu_epsilon_cn,
			Gpu_nOver6,
			Gpu_epsilon_cn_6,
			Gpu_x,
			Gpu_y,
			Gpu_z,
			Gpu_COMX,
			Gpu_COMY,
			Gpu_COMZ,
			Gpu_start,
			currentAxes.axis.x[0],
			currentAxes.axis.y[0],
			currentAxes.axis.z[0],
			currentAxes.halfAx.x[0],
			currentAxes.halfAx.y[0],
			currentAxes.halfAx.z[0],
			offset,
			MolCount[0],
			MolCount[0] / 2,
			forcefield.particles.NumKinds(),
			currentAxes.rCut,
			(MolCount[0] % 2 == 0) ? 1 : 0,
			currentAxes.rCutSq,
			dev_EnergyContrib,
			dev_VirialContrib,
			(MolCount[0]* (MolCount[0] - 1) / 2),
			0,
			Gpu_partn
			);
		cudaMemcpyAsync(&FinalEnergyNVirial, dev_EnergyContrib, 2 * sizeof(double), cudaMemcpyDeviceToHost, stream0);
		cudaStreamSynchronize(stream0);
		NewEnergy1 = FinalEnergyNVirial[0];
		NewVirial1 = FinalEnergyNVirial[1];
	}

	#if ENSEMBLE == GEMC

	ThreadsPerBlock1 = MAXTHREADSPERBLOCK;

	if(ThreadsPerBlock1 == 0)
	{ ThreadsPerBlock1 = 1; }

	BlocksPerGrid1 = ((MolCount[1] * (MolCount[1] - 1) / 2) + ThreadsPerBlock1 - 1) / ThreadsPerBlock1;

	if (BlocksPerGrid1 == 0)
	{ BlocksPerGrid1 = 1; }

	if (BlocksPerGrid1 < MAXTBLOCKS) {
		XBlocks = 1;
		YBlocks = BlocksPerGrid1;
	} else {
		YBlocks = MAXTBLOCKS;
		XBlocks = (int)ceil((double)BlocksPerGrid1 / (double)MAXTBLOCKS);
	}

	dim3 blcks1(XBlocks, YBlocks, 1);
	double * dev_EnergyContrib1, * dev_VirialContrib1;
	cudaMalloc((void**) &dev_EnergyContrib1, 4 * XBlocks * YBlocks * sizeof(double));
	cudaMalloc((void**) &dev_VirialContrib1,  4 * XBlocks * YBlocks * sizeof(double));
	dim3 thrds1 (ThreadsPerBlock1, 1, 1);
	FinalEnergyNVirial[0] = 0.0;
	FinalEnergyNVirial[1] = 0.0;
	offset = AtomCount[0];
	double NewEnergy2 = 0.0;
	double NewVirial2 = 0.0;

	if (MolCount[1] != 0 ) {
		Gpu_CalculateSystemInter <<< blcks1, thrds1, 0, stream1>>>(NoOfAtomsPerMol, Gpu_atomKinds,
			Gpu_kIndex,
			Gpu_sigmaSq,
			Gpu_epsilon_cn,
			Gpu_nOver6,
			Gpu_epsilon_cn_6,
			Gpu_x,
			Gpu_y,
			Gpu_z,
			Gpu_COMX,
			Gpu_COMY,
			Gpu_COMZ,
			Gpu_start,
			currentAxes.axis.x[1],
			currentAxes.axis.y[1],
			currentAxes.axis.z[1],
			currentAxes.halfAx.x[1],
			currentAxes.halfAx.y[1],
			currentAxes.halfAx.z[1],
			offset,
			MolCount[1],
			MolCount[1] / 2,
			forcefield.particles.NumKinds(),
			currentAxes.rCut,
			(MolCount[1] % 2 == 0) ? 1 : 0,
			currentAxes.rCutSq,
			dev_EnergyContrib1,
			dev_VirialContrib1,
			(MolCount[1]* (MolCount[1] - 1) / 2),
			MolCount[0],
			Gpu_partn
			);
		cudaMemcpyAsync(&FinalEnergyNVirial, dev_EnergyContrib1, 2 * sizeof(double), cudaMemcpyDeviceToHost, stream1);
		cudaStreamSynchronize(stream1);
		NewEnergy2 = FinalEnergyNVirial[0];
		NewVirial2 = FinalEnergyNVirial[1];
	}

   #endif


	SystemPotential currentpot;
	//double densityRatio = currentAxes.volume[0] * currentAxes.volInv[0];
	currentpot.boxEnergy[0].inter = NewEnergy1;
	//currentpot.boxEnergy[0].tc *= densityRatio;
	currentpot.boxVirial[0].inter = NewVirial1;
	//currentpot.boxVirial[0].tc *= densityRatio;

	

	cudaFree( dev_EnergyContrib);
	cudaFree(dev_VirialContrib);

 #if ENSEMBLE == GEMC
	//densityRatio = currentAxes.volume[1] * currentAxes.volInv[1];
	currentpot.boxEnergy[1].inter = NewEnergy2;
	//currentpot.boxEnergy[1].tc *= densityRatio;
	currentpot.boxVirial[1].inter = NewVirial2;
	//currentpot.boxVirial[1].tc *= densityRatio;


	

    cudaFree( dev_EnergyContrib1);
	cudaFree(dev_VirialContrib1);
#endif

	currentpot.Total();

	cudaDeviceSynchronize();
	cudaError_t  code = cudaGetLastError();

	if (code != cudaSuccess) {
		printf ("Cuda error at end of Calculate Total Energy -- %s\n", cudaGetErrorString(code));
		exit(2);
	}
	
	return currentpot;
}



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

	) {
		__shared__
			double cacheEnergy[MAXTHREADSPERBLOCK];
		__shared__
			double cacheVirial[MAXTHREADSPERBLOCK];
		__shared__ bool
			LastBlock;
		int blockId = blockIdx.y * gridDim.x + blockIdx.x;
		int MolId = blockId * blockDim.x + threadIdx.x;
		cacheEnergy[threadIdx.x] = 0.0;
		cacheVirial[threadIdx.x] = 0.0;
		
		if (MolId < limit) {
			double energy = 0.0, virial = 0.0;
			double distSq = 0.0;
			double partVirial = 0.0;	
			XYZ virComponents;
			XYZ fMol;
			XYZ temp;
			int i = MolId / (HalfMoleculeCount) + (isEvenMolCount);
			int j = MolId % (HalfMoleculeCount);
			//int k,m;

			if (j >= i) { // re-map
				i = MoleculeCount - i - 1 + (isEvenMolCount);
				j = MoleculeCount - j - 2 + (isEvenMolCount);
			}

			int iStart = Gpu_start[i + MolOffset];
			int jStart = Gpu_start[j + MolOffset];

			for (int k = 0; k < NoOfAtomsPerMol[i + MolOffset]; k++) {
				for (int m = 0; m < NoOfAtomsPerMol[j + MolOffset]; m++) {

					if (InRcutGpuSigned( distSq, x, y, z, xAxis, yAxis, zAxis, xHalfAxis, yHalfAxis, zHalfAxis, k + iStart, m + jStart, rCut ,  rCutSq, virComponents))
					{
					 partVirial = 0.0;	
					 CalcAddGpu(energy, partVirial, distSq, AtomKinds[iStart + k], AtomKinds[jStart + m], FFParticleKindCount, sigmaSq,  epsilon_cn, nOver6,  epsilon_cn_6, n); 
					 fMol += (virComponents * partVirial);
					
					}
				}
			}


			temp.x=MinImageSignedGpu(Gpu_COMX[i+ MolOffset] - Gpu_COMX[j+ MolOffset],xAxis, xHalfAxis) ;
			temp.y=MinImageSignedGpu(Gpu_COMY[i+ MolOffset] - Gpu_COMY[j+ MolOffset],yAxis, yHalfAxis) ;
			temp.z=MinImageSignedGpu(Gpu_COMZ[i+ MolOffset] - Gpu_COMZ[j+ MolOffset],zAxis, zHalfAxis) ;
			virial -= geom::Dot(fMol,temp);
			cacheEnergy[threadIdx.x] += energy;
			cacheVirial[threadIdx.x] += virial;
			
		}

		__syncthreads();
		// add data
		int offset = 1 << (int) __log2f((float) blockDim.x);

		if (blockDim.x < MAXTHREADSPERBLOCK) {
			if ((threadIdx.x + offset) < 4 * limit) {
				cacheEnergy[threadIdx.x] += cacheEnergy[threadIdx.x + offset];
				cacheVirial[threadIdx.x] += cacheVirial[threadIdx.x + offset];
			}

			__syncthreads();
		}

		for (int i = offset >> 1; i > 32; i >>= 1) {
			if (threadIdx.x < i) {
				cacheEnergy[threadIdx.x] += cacheEnergy[threadIdx.x + i];
				cacheVirial[threadIdx.x] += cacheVirial[threadIdx.x + i];
			}           //end if

			__syncthreads();
		}           //end for

		if (threadIdx.x < 32) {
			offset = min(offset, 64);

			switch (offset) {
			case 64:
				cacheEnergy[threadIdx.x] += cacheEnergy[threadIdx.x + 32];
				cacheVirial[threadIdx.x] += cacheVirial[threadIdx.x + 32];
				__threadfence_block();

			case 32:
				cacheEnergy[threadIdx.x] += cacheEnergy[threadIdx.x + 16];
				cacheVirial[threadIdx.x] += cacheVirial[threadIdx.x + 16];
				__threadfence_block();

			case 16:
				cacheEnergy[threadIdx.x] += cacheEnergy[threadIdx.x + 8];
				cacheVirial[threadIdx.x] += cacheVirial[threadIdx.x + 8];
				__threadfence_block();

			case 8:
				cacheEnergy[threadIdx.x] += cacheEnergy[threadIdx.x + 4];
				cacheVirial[threadIdx.x] += cacheVirial[threadIdx.x + 4];
				__threadfence_block();

			case 4:
				cacheEnergy[threadIdx.x] += cacheEnergy[threadIdx.x + 2];
				cacheVirial[threadIdx.x] += cacheVirial[threadIdx.x + 2];
				__threadfence_block();

			case 2:
				cacheEnergy[threadIdx.x] += cacheEnergy[threadIdx.x + 1];
				cacheVirial[threadIdx.x] += cacheVirial[threadIdx.x + 1];
			}
		}           //end if

		if (threadIdx.x == 0) {
			dev_EnergyContrib[blockId] = cacheEnergy[0];
			dev_VirialContrib[blockId] = cacheVirial[0];
			//Compress these two lines to one to save one variable declaration
			__threadfence();
			LastBlock = (atomicInc(&BlocksDone, gridDim.x * gridDim.y) ==  (gridDim.x * gridDim.y) - 1);
		}       //end if

		__syncthreads();

		if (LastBlock) {
			//If this thread corresponds to a valid block
			if (threadIdx.x <  gridDim.x * gridDim.y) {
				cacheEnergy[threadIdx.x] = dev_EnergyContrib[threadIdx.x];
				cacheVirial[threadIdx.x] = dev_VirialContrib[threadIdx.x];
			}

			for (int i = blockDim.x; threadIdx.x + i < gridDim.x * gridDim.y; i += blockDim.x) {
				if (threadIdx.x + i <  gridDim.x * gridDim.y) {
					cacheEnergy[threadIdx.x] += dev_EnergyContrib[threadIdx.x + i];
					cacheVirial[threadIdx.x] += dev_VirialContrib[threadIdx.x + i];
				}       //end if
			}       //end for

			__syncthreads();
			offset = 1 << (int) __log2f((float) min(blockDim.x,  gridDim.x * gridDim.y));

			for (int i = offset; i > 0; i >>= 1) {
				if (threadIdx.x < i && threadIdx.x + i < min(blockDim.x, gridDim.x * gridDim.y)) {
					cacheEnergy[threadIdx.x] += cacheEnergy[threadIdx.x + i];
					cacheVirial[threadIdx.x] += cacheVirial[threadIdx.x + i];
				}       //end if

				__syncthreads();
			}       //end for

			if (threadIdx.x == 0) {
				BlocksDone = 0;
				dev_EnergyContrib[0] = cacheEnergy[0];
				dev_EnergyContrib[1] = cacheVirial[0];
				dev_VirialContrib[0] = cacheVirial[0];
			
			}
		}
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
	if ( v > ax ) //if +, wrap out to low end
		v -= ax;
	else
		if ( v < 0 ) //if -, wrap to high end
			v += ax;
	return v;
#endif
}
__device__ double Gpu_BoltzW(const double a, const double x)
{ return exp(-1.0 * a * x); }


__device__ void CalcSubGpu(
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

	) {
		uint index = kind1 + kind2 * count;
		double rNeg2 = 1.0/distSq;
		double rRat2 = sigmaSq[index] * rNeg2 ;
		double rRat4 = rRat2 * rRat2;
		double attract = rRat4 * rRat2;
#ifdef MIE_INT_ONLY
		double repulse = pow(rRat2, rRat4, attract, n);
#else
		double repulse = pow(__dsqrt_rn(rRat2), n[index]);
#endif
		en -= epsilon_cn[index] * (repulse - attract);
		vir = -1.0 *epsilon_cn_6[index] * (nOver6[index] * repulse - attract)*rNeg2;
		
}

__device__ bool InRcutGpuSigned(
	double &distSq,
	double *Array,
	int len,
	double *x2,
	double *y2,
	double *z2,
	double xAxis,
	double yAxis,
	double zAxis,
	double xHalfAxis,
	double yHalfAxis,
	double zHalfAxis,
	const uint i,
	const uint j,
	double rCut,
	double rCutSq,
	XYZ &dist
	) {
		distSq = 0;
		dist.x =   Array[i]-x2[j] ;
		dist.x = MinImageSignedGpu(dist.x, xAxis, xHalfAxis);
		dist.y =   Array[len + i]-y2[j];
		dist.y = MinImageSignedGpu(dist.y , yAxis, yHalfAxis);
		dist.z =  Array[2 * len + i]-z2[j] ;
		dist.z = MinImageSignedGpu(dist.z, zAxis, zHalfAxis);
		distSq = dist.x * dist.x + dist.y  * dist.y + dist.z * dist.z;
		return (rCutSq > distSq);
}

__device__ void Gpu_CalculateMoleculeInter(
	uint * NoOfAtomsPerMol, uint *AtomKinds,
	uint *molKindIndex,
	double * sigmaSq,
	double * epsilon_cn,
	double * nOver6,
	double * epsilon_cn_6,
	double *oldx,
	double *oldy,
	double *oldz,
	double * Gpu_COMX,
	double * Gpu_COMY,
	double * Gpu_COMZ,
	double  selectedCOMX,
	double  selectedCOMY,
	double  selectedCOMZ,
	double *newArray,
	int len,
	uint MolId,
	uint * Gpu_start,
	double xAxis,
	double yAxis,
	double zAxis,
	double xHalfAxis,
	double yHalfAxis,
	double zHalfAxis,
	uint boxOffset,
	uint MoleculeCount,
	uint mIndex,
	uint FFParticleKindCount,
	double rCut,
	uint molKInd,
	double rCutSq,
	double dev_EnergyContrib[],
	double dev_VirialContrib[],
#ifdef MIE_INT_ONLY
	const uint * n,
#else
	const double * n
#endif

	) {
		__shared__
			double cacheEnergy[MAXTHREADSPERBLOCK];
		__shared__
			double cacheVirial[MAXTHREADSPERBLOCK];
		__shared__ bool LastBlock;
		cacheEnergy[threadIdx.x] = 0.0;
		cacheVirial[threadIdx.x] = 0.0;

		if (MolId < MoleculeCount + boxOffset && mIndex != MolId ) {
			double energy = 0.0, virial = 0.0;
			double distSq = 0.0;
			XYZ fMolO, fMolN;
			double partVirial = 0.0;
			XYZ virComponents;
			XYZ temp;
			uint MolStart = Gpu_start [MolId];
			uint SelectedMolStart = Gpu_start [mIndex];// put in shared memory to be used always

			for (int i = 0; i < len; i++) { //original mol that we are rotating
				for (int j = 0; j < NoOfAtomsPerMol[MolId]; j++) { // the mol we compare with
					if (InRcutGpuSigned(distSq, oldx, oldy, oldz, xAxis, yAxis, zAxis, xHalfAxis, yHalfAxis, zHalfAxis, i + SelectedMolStart, j + MolStart, rCut ,  rCutSq,virComponents ))
					{   partVirial = 0.0;
						CalcSubGpu(energy, partVirial, distSq, AtomKinds[i + SelectedMolStart ], AtomKinds[j + MolStart], FFParticleKindCount, sigmaSq,  epsilon_cn, nOver6,  epsilon_cn_6, n);
						fMolO += virComponents * partVirial;
					}

					if (InRcutGpuSigned( distSq, newArray, len, oldx, oldy, oldz, xAxis, yAxis, zAxis, xHalfAxis, yHalfAxis, zHalfAxis, i, j + MolStart, rCut ,  rCutSq,virComponents ))
					{   partVirial = 0.0;
						CalcAddGpu(energy, partVirial, distSq, AtomKinds[i + SelectedMolStart ], AtomKinds[j + MolStart], FFParticleKindCount, sigmaSq,  epsilon_cn, nOver6,  epsilon_cn_6, n);
						fMolN += virComponents * partVirial;
					}
				}
			}

			temp.x= MinImageSignedGpu(Gpu_COMX[mIndex] - Gpu_COMX[MolId],xAxis, xHalfAxis) ;
			temp.y= MinImageSignedGpu(Gpu_COMY[mIndex] - Gpu_COMY[MolId],yAxis, yHalfAxis) ;
			temp.z= MinImageSignedGpu(Gpu_COMZ[mIndex] - Gpu_COMZ[MolId],zAxis, zHalfAxis) ;
			virial -= geom::Dot(fMolO,temp);

			temp.x= MinImageSignedGpu(selectedCOMX - Gpu_COMX[MolId],xAxis, xHalfAxis) ;
			temp.y= MinImageSignedGpu(selectedCOMY - Gpu_COMY[MolId],yAxis, yHalfAxis) ;
			temp.z= MinImageSignedGpu(selectedCOMZ - Gpu_COMZ[MolId],zAxis, zHalfAxis) ;
			virial -= geom::Dot(fMolN,temp);


			cacheEnergy[threadIdx.x] += energy;
			cacheVirial[threadIdx.x] += virial;
		}

		__syncthreads();
		// add data
		int offset = 1 << (int) __log2f((float) blockDim.x);

		if (blockDim.x < MAXTHREADSPERBLOCK) {
			if ((threadIdx.x + offset) <  MoleculeCount) {
				cacheEnergy[threadIdx.x] += cacheEnergy[threadIdx.x + offset];
				cacheVirial[threadIdx.x] += cacheVirial[threadIdx.x + offset];
			}

			__syncthreads();
		}

		for (int i = offset >> 1; i > 32; i >>= 1) {
			if (threadIdx.x < i) {
				cacheEnergy[threadIdx.x] += cacheEnergy[threadIdx.x + i];
				cacheVirial[threadIdx.x] += cacheVirial[threadIdx.x + i];
			}           //end if

			__syncthreads();
		}           //end for

		if (threadIdx.x < 32) {
			offset = min(offset, 64);

			switch (offset) {
			case 64:
				cacheEnergy[threadIdx.x] += cacheEnergy[threadIdx.x + 32];
				cacheVirial[threadIdx.x] += cacheVirial[threadIdx.x + 32];
				__threadfence_block();

			case 32:
				cacheEnergy[threadIdx.x] += cacheEnergy[threadIdx.x + 16];
				cacheVirial[threadIdx.x] += cacheVirial[threadIdx.x + 16];
				__threadfence_block();

			case 16:
				cacheEnergy[threadIdx.x] += cacheEnergy[threadIdx.x + 8];
				cacheVirial[threadIdx.x] += cacheVirial[threadIdx.x + 8];
				__threadfence_block();

			case 8:
				cacheEnergy[threadIdx.x] += cacheEnergy[threadIdx.x + 4];
				cacheVirial[threadIdx.x] += cacheVirial[threadIdx.x + 4];
				__threadfence_block();

			case 4:
				cacheEnergy[threadIdx.x] += cacheEnergy[threadIdx.x + 2];
				cacheVirial[threadIdx.x] += cacheVirial[threadIdx.x + 2];
				__threadfence_block();

			case 2:
				cacheEnergy[threadIdx.x] += cacheEnergy[threadIdx.x + 1];
				cacheVirial[threadIdx.x] += cacheVirial[threadIdx.x + 1];
			}
		}

		if (threadIdx.x == 0) {
			dev_EnergyContrib[blockIdx.x] = cacheEnergy[0];
			dev_VirialContrib[blockIdx.x] = cacheVirial[0];
			__threadfence();
			LastBlock = (atomicInc(&BlocksDone, gridDim.x) == gridDim.x - 1);

			if (LastBlock)
			{ BlockNum = blockIdx.x; }
		}       //end if

		__syncthreads();

		if (LastBlock) {
			//If this thread corresponds to a valid block
			if (threadIdx.x < gridDim.x) {
				cacheEnergy[threadIdx.x] = dev_EnergyContrib[threadIdx.x];
				cacheVirial[threadIdx.x] = dev_VirialContrib[threadIdx.x];
			}

			for (int i = blockDim.x; threadIdx.x + i < gridDim.x; i += blockDim.x) {
				if (threadIdx.x + i < gridDim.x) {
					cacheEnergy[threadIdx.x] += dev_EnergyContrib[threadIdx.x + i];
					cacheVirial[threadIdx.x] += dev_VirialContrib[threadIdx.x + i];
				}       //end if
			}       //end for

			__syncthreads();
			offset = 1 << (int) __log2f((float) min(blockDim.x, gridDim.x));

			for (int i = offset; i > 0; i >>= 1) {
				if (threadIdx.x < i
					&& threadIdx.x + i < min(blockDim.x, gridDim.x)) {
						cacheEnergy[threadIdx.x] += cacheEnergy[threadIdx.x + i];
						cacheVirial[threadIdx.x] += cacheVirial[threadIdx.x + i];
				}       //end if

				__syncthreads();
			}       //end for

			if (threadIdx.x == 0) {
				dev_EnergyContrib[0] = cacheEnergy[0];
				dev_EnergyContrib[1] = cacheVirial[0];
				dev_VirialContrib[0] = cacheVirial[0];
			}
		}
}



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



	)


{
	extern __shared__ double Array[];
	__shared__ XYZ newCOM;
	int threadId = blockIdx.x * blockDim.x + threadIdx.x ;
	int MolId = threadId + boxOffset ;

	
	if (threadIdx.x < len ) {
		uint start;
		start = Gpu_start[mIndex];
		newCOM.x = Gpu_COMX[mIndex];
		newCOM.y = Gpu_COMY[mIndex];
		newCOM.z = Gpu_COMZ[mIndex];
		newCOM += shift;
		WrapPBC(newCOM.x , xAxis);
		WrapPBC(newCOM.y , yAxis);
		WrapPBC(newCOM.z , zAxis);
		Array[threadIdx.x] = Gpu_x[start + threadIdx.x] + shift.x;
		Array[len + threadIdx.x] = Gpu_y[start + threadIdx.x] + shift.y;
		Array[2 * len + threadIdx.x] = Gpu_z[start + threadIdx.x] + shift.z;
		WrapPBC(Array[threadIdx.x] , xAxis);
		WrapPBC(Array[len + threadIdx.x] , yAxis);
		WrapPBC(Array[2 * len + threadIdx.x] , zAxis);
	}

	__syncthreads();

	
	 if ( boxIndex< BOXES_WITH_U_NB)
  

	Gpu_CalculateMoleculeInter( NoOfAtomsPerMol, AtomKinds,
		molKindIndex,
		sigmaSq,
		epsilon_cn,
		nOver6,
		epsilon_cn_6,
		Gpu_x,
		Gpu_y,
		Gpu_z,
		Gpu_COMX,
		Gpu_COMY,
		Gpu_COMZ,
		newCOM.x,
		newCOM.y,
		newCOM.z,
		Array,
		len,
		MolId,
		Gpu_start,
		xAxis,
		yAxis,
		zAxis,
		xHalfAxis,
		yHalfAxis,
		zHalfAxis,
		boxOffset,
		MoleculeCount,
		mIndex,// mol index with offset
		FFParticleKindCount,
		rCut,
		molKInd,
		rCutSq,
		dev_EnergyContrib,
		dev_VirialContrib,
		n
		);

	 
	 else
	  {
	   dev_EnergyContrib[0]=0.0;
	   dev_EnergyContrib[1]=0.0;

   }
	 



	if (blockIdx.x == BlockNum) {
		if (threadIdx.x == 0) {
			Gpu_result[0] = false;
			
  		
			if (AcceptRand < Gpu_BoltzW(beta, dev_EnergyContrib[0] )) {
				Gpu_result[0] = true;

				for (int i = 0; i < len; i++) {
					Gpu_x[Gpu_start[mIndex] + i] = Array[i];
					Gpu_y[Gpu_start[mIndex] + i] = Array[len + i];
					Gpu_z[Gpu_start[mIndex] + i] = Array[2 * len + i];
				}

				Gpu_COMX[mIndex] =   newCOM.x;
				Gpu_COMY[mIndex] =   newCOM.y;
				Gpu_COMZ[mIndex] =   newCOM.z;
				Gpu_Potential->Add(boxIndex, dev_EnergyContrib[0], dev_EnergyContrib[1] );
			
				Gpu_Potential->Total();
			}

			BlocksDone = 0;
			BlockNum = -1;
		}
	}
}


	__device__  double UnwrapPBC
	(double& v, const double ref, const double ax, const double halfAx) {
		//If absolute value of X dist btwn pt and ref is > 0.5 * box_axis
		//If ref > 0.5 * box_axis, add box_axis to pt (to wrap out + side)
		//If ref < 0.5 * box_axis, subtract box_axis (to wrap out - side)
		// uses bit hack to avoid branching for conditional
#ifdef NO_BRANCHING_UNWRAP
		bool negate = ( ref > halfAx );
		double vDiff = v + (ax ^ -negate) + negate;
		return (fabs(ref - v) > halfAx ) ? v : vDiff;
#else

		if (fabs(ref - v) > halfAx ) {
			//Note: testing shows that it's most efficient to negate if true.
			//Source:
			// http://jacksondunstan.com/articles/2052
			if ( ref < halfAx )
			{ v -= ax; }
			else
			{ v += ax; }
		}

		return v;
#endif
}


__global__ void TryRotateGpu( uint * NoOfAtomsPerMol, uint *AtomKinds, SystemPotential * Gpu_Potential , TransformMatrix  matrix, double * Gpu_x, double * Gpu_y, double * Gpu_z,
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



	)


{
	extern __shared__ double Array[];
	int threadId = blockIdx.x * blockDim.x + threadIdx.x ;
	int MolId = threadId + boxOffset ;

	if (threadIdx.x < len) {
		uint start;
		start = Gpu_start[mIndex];
		//int i=0;
		XYZ center;
	
		center.x = Gpu_COMX[mIndex];
		center.y = Gpu_COMY[mIndex];
		center.z = Gpu_COMZ[mIndex];
	
		Array[threadIdx.x] = Gpu_x[start + threadIdx.x];
		Array[len + threadIdx.x] = Gpu_y[start + threadIdx.x];
		Array[2 * len + threadIdx.x] = Gpu_z[start + threadIdx.x];

		UnwrapPBC(Array[threadIdx.x] , center.x, xAxis, xHalfAxis);
		UnwrapPBC(Array[len + threadIdx.x] , center.y, yAxis, yHalfAxis);
		UnwrapPBC(Array[2 * len + threadIdx.x] , center.z, zAxis, zHalfAxis);

	
		Array[threadIdx.x] += (-1.0 * center.x);
		Array[len + threadIdx.x] += (-1.0 * center.y);
		Array[2 * len + threadIdx.x] += (-1.0 * center.z);


	    matrix.Apply(Array[threadIdx.x], Array[len + threadIdx.x], Array[2 * len + threadIdx.x]);

		Array[threadIdx.x] += ( center.x);
		Array[len + threadIdx.x] += ( center.y);
		Array[2 * len + threadIdx.x] += ( center.z);



		WrapPBC(Array[threadIdx.x], xAxis);
		WrapPBC(Array[len + threadIdx.x], yAxis);
		WrapPBC(Array[2 * len + threadIdx.x], zAxis);
	}

	__syncthreads();
	Gpu_CalculateMoleculeInter( NoOfAtomsPerMol, AtomKinds,
		molKindIndex,
		sigmaSq,
		epsilon_cn,
		nOver6,
		epsilon_cn_6,
		Gpu_x,
		Gpu_y,
		Gpu_z,
		Gpu_COMX,
		Gpu_COMY,
		Gpu_COMZ,
		Gpu_COMX[mIndex],
		Gpu_COMY[mIndex],
		Gpu_COMZ[mIndex],
		Array,
		len,
		MolId,
		Gpu_start,
		xAxis,
		yAxis,
		zAxis,
		xHalfAxis,
		yHalfAxis,
		zHalfAxis,
		boxOffset,
		MoleculeCount,
		mIndex,// mol index with offset
		FFParticleKindCount,
		rCut,
		molKInd,
		rCutSq,
		dev_EnergyContrib,
		dev_VirialContrib,
		n
		);

	if (blockIdx.x == BlockNum) {
		if (threadIdx.x == 0) {
			Gpu_result[0] = false;

			if (AcceptRand < Gpu_BoltzW(beta, dev_EnergyContrib[0] )) {
				Gpu_result[0] = true;

				for (int i = 0; i < len; i++) {
					Gpu_x[Gpu_start[mIndex] + i] = Array[i];
					Gpu_y[Gpu_start[mIndex] + i] = Array[len + i];
					Gpu_z[Gpu_start[mIndex] + i] = Array[2 * len + i];
				}

				Gpu_Potential->Add(boxIndex, dev_EnergyContrib[0], dev_EnergyContrib[1] );
				Gpu_Potential->Total();
			}

			BlocksDone = 0;
			BlockNum = -1;
		}
	}
}


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
	) {
		distSq = 0;
		dist.x = x[i] - x[j];
		dist.x =MinImageSignedGpu(dist.x, xAxis, xHalfAxis);
		dist.y = y[i] - y[j];
		dist.y = MinImageSignedGpu(dist.y, yAxis, yHalfAxis);
		dist.z = z[i] - z[j];
		dist.z = MinImageSignedGpu(dist.z , zAxis, zHalfAxis);
		distSq = dist.x * dist.x + dist.y * dist.y+ dist.z  * dist.z ;
		return (rCutSq > distSq);
}


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

	) {
		uint index = kind1 + kind2 * count;
		double rNeg2 = 1.0/distSq;
		double rRat2 = rNeg2 * sigmaSq[index];
		double rRat4 = rRat2 * rRat2;
		double attract = rRat4 * rRat2;
#ifdef MIE_INT_ONLY
		double repulse = pow(rRat2, rRat4, attract, n);
#else
		double repulse = pow(__dsqrt_rn(rRat2), n[index]);
#endif

		en += epsilon_cn[index] * (repulse-attract);
		//Virial is the derivative of the pressure... mu
		vir = epsilon_cn_6[index] * (nOver6[index]*repulse-attract)*rNeg2;

}

__device__ double MinImageSignedGpu(double raw,double ax, double halfAx) 
{
   if (raw > halfAx)
      raw -= ax;
   else if (raw < -halfAx)
      raw += ax;
   return raw;
}


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
	)

{
	int threadId = blockIdx.x * blockDim.x + threadIdx.x ;
	int MolId = threadId + boxOffset ; // ID of molecule to do calculations with for this thread

	if (threadId < MolCount) {
		double shiftX, shiftY, shiftZ;
		uint start;
		int len;
		len = noOfAtomsPerMol[MolId];
		start = Gpu_start[MolId];
		Gpu_newCOMX[MolId] = Gpu_COMX [MolId] * scale;
		Gpu_newCOMY[MolId] = Gpu_COMY [MolId] * scale;
		Gpu_newCOMZ[MolId] = Gpu_COMZ [MolId] * scale;
		shiftX = Gpu_newCOMX[MolId];
		shiftY = Gpu_newCOMY[MolId];
		shiftZ = Gpu_newCOMZ[MolId];
		shiftX -= Gpu_COMX[MolId];
		shiftY -= Gpu_COMY[MolId];
		shiftZ -= Gpu_COMZ[MolId];

		double tmp = 0.0;

		for (int i = 0; i < len; i++) {
			Gpu_newx[start + i] = Gpu_x[start + i];
			Gpu_newy[start + i] = Gpu_y[start + i];
			Gpu_newz[start + i] = Gpu_z[start + i];
			UnwrapPBC(Gpu_newx[start + i], Gpu_COMX[MolId], xAxis, xHalfAxis );
			UnwrapPBC(Gpu_newy[start + i], Gpu_COMY[MolId], yAxis, yHalfAxis );
			UnwrapPBC(Gpu_newz[start + i], Gpu_COMZ[MolId], zAxis, zHalfAxis );
			Gpu_newx[start + i] += shiftX;
			Gpu_newy[start + i] += shiftY;
			Gpu_newz[start + i] += shiftZ;
			WrapPBC(Gpu_newx[start + i], newxAxis);
			WrapPBC(Gpu_newy[start + i], newyAxis);
			WrapPBC(Gpu_newz[start + i], newzAxis);
		}
	}
}

SystemPotential CalculateEnergy::NewSystemInterGPU(
	BoxDimensions &newDim,
	uint src,
	uint dist
	) {
		SystemPotential pot;
		double densityRatio;
		cudaMemcpy(&pot, Gpu_Potential, sizeof(SystemPotential)  , cudaMemcpyDeviceToHost);
		// do src box
		uint offset;
		offset = (src == 0 ) ? 0 : AtomCount[0]   ;
		uint MolOffset;
		MolOffset = (src == 0 ) ? 0 : MolCount[0]   ;
		int ThreadsPerBlock1 = 0;
		int BlocksPerGrid1 = 0;
		ThreadsPerBlock1 = MAXTHREADSPERBLOCK;

		if(ThreadsPerBlock1 == 0)
		{ ThreadsPerBlock1 = 1; }

		BlocksPerGrid1 = ((MolCount[src] * (MolCount[src] - 1) / 2) + ThreadsPerBlock1 - 1) / ThreadsPerBlock1;

		if (BlocksPerGrid1 == 0)
		{ BlocksPerGrid1 = 1; }

		int XBlocks, YBlocks;

		if (BlocksPerGrid1 < MAXTBLOCKS) {
			XBlocks = 1;
			YBlocks = BlocksPerGrid1;
		} else {
			YBlocks = MAXTBLOCKS;
			XBlocks = (int)ceil((double)BlocksPerGrid1 / (double)MAXTBLOCKS);
		}

		double * dev_EnergyContrib, * dev_VirialContrib;
		cudaMalloc((void**) &dev_EnergyContrib, 4 * XBlocks * YBlocks * sizeof(double));
		cudaMalloc((void**) &dev_VirialContrib, 4 * XBlocks * YBlocks * sizeof(double));
		double FinalEnergyNVirial[2];
		FinalEnergyNVirial[0] = 0.0;
		FinalEnergyNVirial[1] = 0.0;
		double NewEnergy1 = 0.0;
		double NewVirial1 = 0.0;
		dim3 blcks(XBlocks, YBlocks, 1);
		dim3 thrds (ThreadsPerBlock1, 1, 1);

		if ( MolCount[src] != 0) {
			Gpu_CalculateSystemInter <<< blcks, thrds, 0, stream0>>>(NoOfAtomsPerMol, Gpu_atomKinds,
				Gpu_kIndex,
				Gpu_sigmaSq,
				Gpu_epsilon_cn,
				Gpu_nOver6,
				Gpu_epsilon_cn_6,
				newX,
				newY,
				newZ,
				newCOMX,
				newCOMY,
				newCOMZ,
				Gpu_start,
				newDim.axis.x[src],
				newDim.axis.y[src],
				newDim.axis.z[src],
				newDim.halfAx.x[src],
				newDim.halfAx.y[src],
				newDim.halfAx.z[src],
				offset,
				MolCount[src],
				MolCount[src] / 2,
				forcefield.particles.NumKinds(),
				newDim.rCut,
				(MolCount[src] % 2 == 0) ? 1 : 0,
				newDim.rCutSq,
				dev_EnergyContrib,
				dev_VirialContrib,
				(MolCount[src]* (MolCount[src] - 1) / 2),
				MolOffset,
				Gpu_partn
				);
			cudaMemcpyAsync(&FinalEnergyNVirial, dev_EnergyContrib, 2 * sizeof(double), cudaMemcpyDeviceToHost, stream0);
			cudaStreamSynchronize(stream0);
			NewEnergy1 = FinalEnergyNVirial[0];
			NewVirial1 = FinalEnergyNVirial[1];
		}

		densityRatio = currentAxes.volume[src] * newDim.volInv[src];
		pot.boxEnergy[src].inter = NewEnergy1;
		pot.boxEnergy[src].tc *= densityRatio;
		pot.boxVirial[src].inter = NewVirial1;
		pot.boxVirial[src].tc *= densityRatio;
		offset = (dist  == 0 ) ? 0 : AtomCount[0]  ;
		MolOffset = (dist == 0 ) ? 0 : MolCount[0]   ;
		/*if (MolCount[dist]* (MolCount[dist]-1)/2 < MAXTHREADSPERBLOCK)
		ThreadsPerBlock1 = MolCount[dist]* (MolCount[dist]-1)/2;
		else*/
		ThreadsPerBlock1 = MAXTHREADSPERBLOCK;

		if(ThreadsPerBlock1 == 0)
		{ ThreadsPerBlock1 = 1; }

		BlocksPerGrid1 = ((MolCount[dist] * (MolCount[dist] - 1) / 2) + ThreadsPerBlock1 - 1) / ThreadsPerBlock1;

		if (BlocksPerGrid1 == 0)
		{ BlocksPerGrid1 = 1; }

		if (BlocksPerGrid1 < MAXTBLOCKS) {
			XBlocks = BlocksPerGrid1;
			YBlocks = 1;
		} else {
			YBlocks = MAXTBLOCKS;
			XBlocks = (int)ceil((double)BlocksPerGrid1 / (double)MAXTBLOCKS);
		}

		double * dev_EnergyContrib1, * dev_VirialContrib1;
		cudaMalloc((void**) &dev_EnergyContrib1, 4 * XBlocks * YBlocks * sizeof(double));
		cudaMalloc((void**) &dev_VirialContrib1, 4 * XBlocks * YBlocks * sizeof(double));
		FinalEnergyNVirial[0] = 0.0;
		FinalEnergyNVirial[1] = 0.0;
		double NewEnergy2 = 0.0;
		double NewVirial2 = 0.0;
		dim3 blcks1(XBlocks, YBlocks, 1);
		dim3 thrds1 (ThreadsPerBlock1, 1, 1);

		if ( MolCount[dist] != 0) {
			Gpu_CalculateSystemInter <<< blcks1, thrds1, 0, stream1>>>(NoOfAtomsPerMol, Gpu_atomKinds,
				Gpu_kIndex,
				Gpu_sigmaSq,
				Gpu_epsilon_cn,
				Gpu_nOver6,
				Gpu_epsilon_cn_6,
				newX,
				newY,
				newZ,
				newCOMX,
				newCOMY,
				newCOMZ,
				Gpu_start,
				newDim.axis.x[dist],
				newDim.axis.y[dist],
				newDim.axis.z[dist],
				newDim.halfAx.x[dist],
				newDim.halfAx.y[dist],
				newDim.halfAx.z[dist],
				offset,
				MolCount[dist],
				MolCount[dist] / 2,
				forcefield.particles.NumKinds(),
				newDim.rCut,
				(MolCount[dist] % 2 == 0) ? 1 : 0,
				newDim.rCutSq,
				dev_EnergyContrib1,
				dev_VirialContrib1,
				(MolCount[dist]* (MolCount[dist] - 1) / 2),
				MolOffset,
				Gpu_partn
				);
			cudaMemcpyAsync(&FinalEnergyNVirial, dev_EnergyContrib1, 2 * sizeof(double), cudaMemcpyDeviceToHost, stream1);
			cudaStreamSynchronize(stream1);
			NewEnergy2 = FinalEnergyNVirial[0];
			NewVirial2 = FinalEnergyNVirial[1];
		}

		densityRatio = currentAxes.volume[dist] * newDim.volInv[dist];
		pot.boxEnergy[dist].inter = NewEnergy2;
		pot.boxEnergy[dist].tc *= densityRatio;
		pot.boxVirial[dist].inter = NewVirial2;
		pot.boxVirial[dist].tc *= densityRatio;
		pot.Total();
		cudaFree( dev_EnergyContrib);
		cudaFree(dev_VirialContrib);
		cudaFree( dev_EnergyContrib1);
		cudaFree(dev_VirialContrib1);
		return pot;
}


SystemPotential CalculateEnergy::NewSystemInterGPUOneBox(
	BoxDimensions &newDim,
	uint bPick
	) {
		SystemPotential pot;
		double densityRatio;
		cudaMemcpy(&pot, Gpu_Potential, sizeof(SystemPotential)  , cudaMemcpyDeviceToHost);
		// do src box
		uint offset;
		offset = (bPick == 0 ) ? 0 : AtomCount[0]   ;
		uint MolOffset;
		MolOffset = (bPick == 0 ) ? 0 : MolCount[0]   ;
		int ThreadsPerBlock1 = 0;
		int BlocksPerGrid1 = 0;
		ThreadsPerBlock1 = MAXTHREADSPERBLOCK;

		if(ThreadsPerBlock1 == 0)
		{ ThreadsPerBlock1 = 1; }

		BlocksPerGrid1 = ((MolCount[bPick] * (MolCount[bPick] - 1) / 2) + ThreadsPerBlock1 - 1) / ThreadsPerBlock1;

		if (BlocksPerGrid1 == 0)
		{ BlocksPerGrid1 = 1; }

		int XBlocks, YBlocks;

		if (BlocksPerGrid1 < MAXTBLOCKS) {
			XBlocks = 1;
			YBlocks = BlocksPerGrid1;
		} else {
			YBlocks = MAXTBLOCKS;
			XBlocks = (int)ceil((double)BlocksPerGrid1 / (double)MAXTBLOCKS);
		}

		double * dev_EnergyContrib, * dev_VirialContrib;
		cudaMalloc((void**) &dev_EnergyContrib, 4 * XBlocks * YBlocks * sizeof(double));
		cudaMalloc((void**) &dev_VirialContrib, 4 * XBlocks * YBlocks * sizeof(double));
		double FinalEnergyNVirial[2];
		FinalEnergyNVirial[0] = 0.0;
		FinalEnergyNVirial[1] = 0.0;
		double NewEnergy1 = 0.0;
		double NewVirial1 = 0.0;
		dim3 blcks(XBlocks, YBlocks, 1);
		dim3 thrds (ThreadsPerBlock1, 1, 1);

		if ( MolCount[bPick] != 0) {
			Gpu_CalculateSystemInter <<< blcks, thrds, 0, stream0>>>(NoOfAtomsPerMol, Gpu_atomKinds,
				Gpu_kIndex,
				Gpu_sigmaSq,
				Gpu_epsilon_cn,
				Gpu_nOver6,
				Gpu_epsilon_cn_6,
				newX,
				newY,
				newZ,
				newCOMX,
				newCOMY,
				newCOMZ,
				Gpu_start,
				newDim.axis.x[bPick],
				newDim.axis.y[bPick],
				newDim.axis.z[bPick],
				newDim.halfAx.x[bPick],
				newDim.halfAx.y[bPick],
				newDim.halfAx.z[bPick],
				offset,
				MolCount[bPick],
				MolCount[bPick] / 2,
				forcefield.particles.NumKinds(),
				newDim.rCut,
				(MolCount[bPick] % 2 == 0) ? 1 : 0,
				newDim.rCutSq,
				dev_EnergyContrib,
				dev_VirialContrib,
				(MolCount[bPick]* (MolCount[bPick] - 1) / 2),
				MolOffset,
				Gpu_partn
				);
			cudaMemcpyAsync(&FinalEnergyNVirial, dev_EnergyContrib, 2 * sizeof(double), cudaMemcpyDeviceToHost, stream0);
			cudaStreamSynchronize(stream0);
			NewEnergy1 = FinalEnergyNVirial[0];
			NewVirial1 = FinalEnergyNVirial[1];
		}

		densityRatio = currentAxes.volume[bPick] * newDim.volInv[bPick];
		pot.boxEnergy[bPick].inter = NewEnergy1;
		pot.boxEnergy[bPick].tc *= densityRatio;
		pot.boxVirial[bPick].inter = NewVirial1;
		pot.boxVirial[bPick].tc *= densityRatio;
		pot.Total();
		cudaFree( dev_EnergyContrib);
		cudaFree(dev_VirialContrib);
		return pot;
}


__device__ bool InRcutGpu(
	double &distSq,
	double *x,
	double * y,
	double *z,
	double nX,
	double nY,
	double nZ,
	double xAxis,
	double yAxis,
	double zAxis,
	double xHalfAxis,
	double yHalfAxis,
	double zHalfAxis,
	const uint i,
	double rCut,
	double rCutSq
	) {
		distSq = 0;
		double dX = nX - x[i];
		dX= MinImageSignedGpu(dX, xAxis, xHalfAxis);
		double dY = nY - y[i];
		dY =MinImageSignedGpu(dY, yAxis, yHalfAxis);
		double dZ = nZ - z[i];
		dZ = MinImageSignedGpu(dZ, zAxis, zHalfAxis);
		distSq = dX * dX + dY * dY+ dZ * dZ;
		return (rCutSq > distSq);
}


__device__ void CalcAddGpuNoVirial(
	double& en,
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

	) {
		uint index = kind1 + kind2 * count;
		double rRat2 = sigmaSq[index] / distSq;
		double rRat4 = rRat2 * rRat2;
		double attract = rRat4 * rRat2;
#ifdef MIE_INT_ONLY
		double repulse = pow(rRat2, rRat4, attract, n[index]);
#else
		double repulse = pow(__dsqrt_rn(rRat2), n[index]);
#endif
		en += epsilon_cn[index] * (repulse - attract);
	

	
}


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


	) {
		__shared__ double cacheEnergy[MAXTHREADSPERBLOCK];
		
		__shared__ bool LastBlock;
		int threadID = blockIdx.x * blockDim.x + threadIdx.x;
		int AtomID = threadID + boxOffset;
		cacheEnergy[threadIdx.x] = 0.0;
	
		int start = Gpu_start[MolId];

		if (threadID < AtomCount && (AtomID < start || AtomID >= start + len )) { // skip atoms inside the same molecule
			double energy = 0.0;
			double distSq = 0.0;

			if (InRcutGpu( distSq, Gpu_x, Gpu_y, Gpu_z, nX, nY, nZ, xAxis, yAxis, zAxis, xHalfAxis, yHalfAxis, zHalfAxis, AtomID, rCut ,  rCutSq )) {
			
				CalcAddGpuNoVirial(
					energy,
					distSq,
					Gpu_atomKinds[start + AtomToProcess],
					Gpu_atomKinds[AtomID],
					FFParticleKindCount,
					sigmaSq,
					epsilon_cn,
					nOver6,
					epsilon_cn_6,
					n
					);
				
			}

			cacheEnergy[threadIdx.x] += energy;
			
		}

		__syncthreads();
		// add data
		int offset = 1 << (int) __log2f((float) blockDim.x);

		if (blockDim.x < MAXTHREADSPERBLOCK) {
			if ((threadIdx.x + offset) <  AtomCount) {
				cacheEnergy[threadIdx.x] += cacheEnergy[threadIdx.x + offset];
			
			}

			__syncthreads();
		}

		for (int i = offset >> 1; i > 32; i >>= 1) {
			if (threadIdx.x < i) {
				cacheEnergy[threadIdx.x] += cacheEnergy[threadIdx.x + i];
				
			}           //end if

			__syncthreads();
		}           //end for

		if (threadIdx.x < 32) {
			offset = min(offset, 64);

			switch (offset) {
			case 64:
				cacheEnergy[threadIdx.x] += cacheEnergy[threadIdx.x + 32];
				
				__threadfence_block();

			case 32:
				cacheEnergy[threadIdx.x] += cacheEnergy[threadIdx.x + 16];
				
				__threadfence_block();

			case 16:
				cacheEnergy[threadIdx.x] += cacheEnergy[threadIdx.x + 8];
				
				__threadfence_block();

			case 8:
				cacheEnergy[threadIdx.x] += cacheEnergy[threadIdx.x + 4];
			
				__threadfence_block();

			case 4:
				cacheEnergy[threadIdx.x] += cacheEnergy[threadIdx.x + 2];
			
				__threadfence_block();

			case 2:
				cacheEnergy[threadIdx.x] += cacheEnergy[threadIdx.x + 1];
			
			}
		}           //end if

		if (threadIdx.x == 0) {
			dev_EnergyContrib[blockIdx.x] = cacheEnergy[0];
			
			__threadfence();
			LastBlock = (atomicInc(&BlocksDone, gridDim.x) == gridDim.x - 1);

			if (LastBlock)
			{ BlockNum = blockIdx.x; }
		}       //end if

		__syncthreads();

		if (LastBlock) {
			//If this thread corresponds to a valid block
			if (threadIdx.x < gridDim.x) {
				cacheEnergy[threadIdx.x] = dev_EnergyContrib[threadIdx.x];
				
			}

			for (int i = blockDim.x; threadIdx.x + i < gridDim.x; i += blockDim.x) {
				if (threadIdx.x + i < gridDim.x) {
					cacheEnergy[threadIdx.x] += dev_EnergyContrib[threadIdx.x + i];
					
				}       //end if
			}       //end for

			__syncthreads();
			offset = 1 << (int) __log2f((float) min(blockDim.x, gridDim.x));

			for (int i = offset; i > 0; i >>= 1) {
				if (threadIdx.x < i
					&& threadIdx.x + i < min(blockDim.x, gridDim.x)) {
						cacheEnergy[threadIdx.x] += cacheEnergy[threadIdx.x + i];
						
				}       //end if

				__syncthreads();
			}       //end for

			if (threadIdx.x == 0) {
				dev_EnergyContrib[0] = cacheEnergy[0];
			
				BlockNum = -1;
				BlocksDone = 0;
			}
		}
}


