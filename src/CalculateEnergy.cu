#include "CalculateEnergy.h"        //header for this#include "EnergyTypes.h"            //Energy structs#include "EnsemblePreprocessor.h"   //Flags#include "../lib/BasicTypes.h"             //uint#include "System.h"                 //For init#include "StaticVals.h"             //For init#include "Forcefield.h"             //#include "MoleculeLookup.h"
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
#define REAL
#define RECIP

using namespace geom;

__device__ int BlockNum = -1;
__device__ unsigned int BlocksDone = 0;

CalculateEnergy::CalculateEnergy(const StaticVals& stat, const System& sys) :
		forcefield(stat.forcefield), mols(stat.mol), currentCoords(
				sys.coordinates), currentCOM(sys.com),
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
{
}

void CalculateEnergy::Init(config_setup::SystemVals const& val) {
	DoEwald = (bool*)malloc(sizeof(bool));	DoEwald[0] = val.ewald.enable;
	alpha = (double*)malloc(sizeof(double));	alpha[0] = val.ewald.alpha;
	kmax1 = (double*)malloc(sizeof(double));	kmax1[0] = val.ewald.KMax;	qqfact = (double*)malloc(sizeof(double));	qqfact[0] = 167000.00;	RecipSize = (int*)malloc(sizeof(int) * 2);

	for (uint m = 0; m < mols.count; ++m) {
		const MoleculeKind& molKind = mols.GetKind(m);
		for (uint a = 0; a < molKind.NumAtoms(); ++a) {
			particleKind.push_back(molKind.AtomKind(a));
			particleMol.push_back(m);
			particleCharge.push_back(molKind.atomCharge[a]);
		}
	}
	calp = (double*)malloc(BOX_TOTAL);
	std::fill_n(calp, BOX_TOTAL, 0.0);	RecipSize[0] = RecipSize[1] = 0;

	for(int box = 0; box < BOX_TOTAL; box++){
		Calp(box, currentAxes.axis.BoxSize(box));	//calculate the alpha over box size
#ifdef RECIP
		SetupRecip(box);
#endif
	}
#ifdef RECIP
	RecipSinSum = (double*)malloc(RecipSize[0] + RecipSize[1]);
	RecipCosSum = (double*)malloc(RecipSize[0] + RecipSize[1]);
	SinSumNew = (double*)malloc(RecipSize[0] + RecipSize[1]);
	CosSumNew = (double*)malloc(RecipSize[0] + RecipSize[1]);//	cudaMalloc((void**) &dev_prefact, sizeof(double) * prefact.size());//	cudaMalloc((void**) &dev_kxyz, sizeof(int) * kxyz.size());//	cudaMemcpy(dev_prefact, &prefact[0], sizeof(double) * prefact.size(), cudaMemcpyHostToDevice);//	cudaMemcpy(dev_kxyz, &kxyz[0], sizeof(double) * kxyz.size(), cudaMemcpyHostToDevice);
#endif
	MolSelfEnergy = (double*)malloc(molLookup.GetNumKind());
	std::fill_n(MolSelfEnergy, molLookup.GetNumKind(), 0.0);
}

SystemPotential CalculateEnergy::SystemTotal() {

#ifdef CELL_LIST
	SystemPotential pot = CalculateEnergyCellList();
#else

	SystemPotential pot = SystemInterGPU();
#endif

	if (forcefield.useLRC)
		FullTailCorrection(pot, currentAxes);

	//system intra
	for (uint box = 0; box < BOX_TOTAL; ++box) {
		double bondEn = 0.0;
		double nonbondEn = 0.0;		double correction = 0.0, self = 0.0;
		MoleculeLookup::box_iterator thisMol = molLookup.BoxBegin(box);
		MoleculeLookup::box_iterator end = molLookup.BoxEnd(box);
		while (thisMol != end) {
			MoleculeIntra(bondEn, nonbondEn, *thisMol, box);
			if(DoEwald[0]){
				MolCorrection(correction, *thisMol, box);
			}
			++thisMol;
		}

		pot.boxEnergy[box].intraBond = bondEn;
		pot.boxEnergy[box].intraNonbond = nonbondEn;

		if(DoEwald[0]){
			BoxSelf(self, box);
			self = -1.0 * self * calp[box] * qqfact[0] / sqrt(M_PI);
			pot.boxEnergy[box].self = self;
			pot.boxEnergy[box].correction = -1.0 * correction * qqfact[0];
#ifdef RECIP
			SetupRecip(box);
			pot.boxEnergy[box].recip = SystemRecipGPU().boxEnergy[box].recip;
#endif
			pot.boxEnergy[box].elect += (pot.boxEnergy[box].self + pot.boxEnergy[box].correction
					+ pot.boxEnergy[box].recip);
		}
	}

	pot.Total();
	return pot;
}

SystemPotential CalculateEnergy::SystemInter(SystemPotential potential,
		const XYZArray& coords, const XYZArray& com,
		const BoxDimensions& boxAxes) const {
	for (uint box = 0; box < BOX_TOTAL; ++box) {
		double energy, virial;
		energy = virial = 0.0;
		MoleculeLookup::box_iterator thisMol = molLookup.BoxBegin(box);
		MoleculeLookup::box_iterator end = molLookup.BoxEnd(box);
		while (thisMol != end) {
			uint m1 = *thisMol;
			MoleculeKind const& thisKind = mols.GetKind(m1);
			//evaluate interaction with all molecules after this
			MoleculeLookup::box_iterator otherMol = thisMol;
			++otherMol;
			while (otherMol != end) {
				uint m2 = *otherMol;
				XYZ fMol;
				MoleculeKind const& otherKind = mols.GetKind(m2);
				for (uint i = 0; i < thisKind.NumAtoms(); ++i) {
					for (uint j = 0; j < otherKind.NumAtoms(); ++j) {
						XYZ virComponents;
						double distSq;
						if (boxAxes.InRcut(distSq, virComponents, coords,
								mols.start[m1] + i, mols.start[m2] + j, box)) {

							double partVirial = 0.0;

							forcefield.particles->CalcAdd(energy, partVirial,
									distSq, thisKind.AtomKind(i),
									otherKind.AtomKind(j));

							//Add to our pressure components.
							fMol += (virComponents * partVirial);
						}
					}
				}

				//Pressure is wrt center of masses of two molecules.
				virial -= geom::Dot(fMol,
						currentAxes.MinImage(com.Difference(m1, m2), box));

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

double CalculateEnergy::CheckMoleculeInter(uint molIndex, uint box) const {
	double result = 0.0;
	MoleculeLookup::box_iterator molInBox = molLookup.BoxBegin(box);
	MoleculeLookup::box_iterator end = molLookup.BoxEnd(box);
	MoleculeKind const& thisKind = mols.GetKind(molIndex);

	uint thisStart = mols.MolStart(molIndex);
	uint thisEnd = thisStart + thisKind.NumAtoms();
	//looping over all molecules in box
	while (molInBox != end) {
		uint otherMol = *molInBox;
		//except itself
		if (otherMol != molIndex) {
			MoleculeKind const& otherKind = mols.GetKind(otherMol);
			uint otherStart = mols.MolStart(otherMol);
			uint otherEnd = otherStart + otherKind.NumAtoms();
			//compare all particle pairs
			for (uint j = otherStart; j < otherEnd; ++j) {
				uint kindJ = otherKind.AtomKind(j - otherStart);
				for (uint i = thisStart; i < thisEnd; ++i) {
					double distSq;
					int kindI = thisKind.AtomKind(i - thisStart);
					if (currentAxes.InRcut(distSq, currentCoords, i, j, box)) {
						result += forcefield.particles->CalcEn(distSq, kindI,
								kindJ);
					}
				}
			}
		}
		++molInBox;
	}
	return result;
}

SystemPotential CalculateEnergy::BoxNonbonded(SystemPotential potential,
		const uint box, const XYZArray& coords, XYZArray const& com,
		const BoxDimensions& boxAxes) const {
	Intermolecular inter;
	MoleculeLookup::box_iterator thisMol = molLookup.BoxBegin(box);
	MoleculeLookup::box_iterator end = molLookup.BoxEnd(box);
	while (thisMol != end) {
		uint m1 = *thisMol;
		MoleculeKind const& thisKind = mols.GetKind(m1);
		//evaluate interaction with all molecules after this
		MoleculeLookup::box_iterator otherMol = thisMol;
		++otherMol;
		while (otherMol != end) {
			uint m2 = *otherMol;
			XYZ fMol;
			MoleculeKind const& otherKind = mols.GetKind(m2);
			for (uint i = 0; i < thisKind.NumAtoms(); ++i) {
				for (uint j = 0; j < otherKind.NumAtoms(); ++j) {
					XYZ virComponents;
					double distSq;
					if (boxAxes.InRcut(distSq, virComponents, coords,
							mols.start[m1] + i, mols.start[m2] + j, box)) {

						double partVirial = 0.0;

						forcefield.particles->CalcAdd(inter.energy, partVirial,
								distSq, thisKind.AtomKind(i),
								otherKind.AtomKind(j));

						//Add to our pressure components.
						fMol += (virComponents * partVirial);
					}
				}
			}

			//Pressure is wrt center of masses of two molecules.
			inter.virial -= geom::Dot(fMol,
					boxAxes.MinImage(com.Difference(m1, m2), box));

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

SystemPotential CalculateEnergy::SystemNonbonded(SystemPotential potential,
		const XYZArray& coords, const XYZArray& com,
		const BoxDimensions& boxAxes) const {
	for (uint box = 0; box < BOX_TOTAL; ++box) {
		potential = BoxNonbonded(potential, box, coords, com, boxAxes);
	}
	potential.Total();
	return potential;
}

Intermolecular CalculateEnergy::MoleculeInter(const XYZArray& molCoords,
		const uint molIndex, const uint box, XYZ const* const newCOM) const {
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
	while (otherMol != end) {
		uint m2 = *otherMol;
		XYZ fMolO, fMolN;
		//except itself
		if (m2 != molIndex) {
			const MoleculeKind& otherKind = mols.GetKind(m2);
			//compare all particle pairs
			for (uint i = 0; i < partLenMolI; ++i) {
				pOldI = i + partStartMolI;
				pkI = thisKind.AtomKind(i);
				mols.GetRangeStartLength(partStartMolJ, partLenMolJ, m2);
				for (uint j = 0; j < partLenMolJ; ++j) {
					XYZ virComponents;
					double distSq;
					pJ = j + partStartMolJ;
					pkJ = otherKind.AtomKind(j);
					//Subtract old energy
					if (currentAxes.InRcut(distSq, virComponents, currentCoords,
							pOldI, pJ, box)) {
						double partVirial = 0.0;

						forcefield.particles->CalcSub(result.energy, partVirial,
								distSq, pkI, pkJ);

						fMolO += virComponents * partVirial;
					}
					//Add new energy
					if (currentAxes.InRcut(distSq, virComponents, molCoords, i,
							currentCoords, pJ, box)) {
						double partVirial = 0.0;

						forcefield.particles->CalcAdd(result.energy, partVirial,
								distSq, pkI, pkJ);

						//Add to our pressure components.
						fMolN += (virComponents * partVirial);
					}
				}
			}
			//Pressure is wrt center of masses of two molecules.
			result.virial -= geom::Dot(fMolO,
					currentAxes.MinImage(currentCOM.Difference(molIndex, m2),
							box));
			if (hasNewCOM) {
				result.virial -= geom::Dot(fMolN,
						currentAxes.MinImage(*newCOM - currentCOM.Get(m2),
								box));
			} else {
				result.virial -= geom::Dot(fMolN,
						currentAxes.MinImage(
								currentCOM.Difference(molIndex, m2), box));
			}
		}

		++otherMol;
	}
	return result;
}

void CalculateEnergy::ParticleNonbonded(double* energy,
		const cbmc::TrialMol& trialMol, XYZArray const& trialPos,
		const uint partIndex, const uint box, const uint trials) const {
	if (box >= BOXES_WITH_U_B)
		return;

	const MoleculeKind& kind = trialMol.GetKind();
	//loop over all partners of the trial particle
	const uint* partner = kind.sortedNB.Begin(partIndex);
	const uint* end = kind.sortedNB.End(partIndex);
	while (partner != end) {
		if (trialMol.AtomExists(*partner)) {
			for (uint i = 0; i < trials; ++i) //v1
					{
				double distSq;
				if (currentAxes.InRcut(distSq, trialPos, i,
						trialMol.GetCoords(), *partner, box)) {
					energy[i] += forcefield.particles->CalcEn(distSq,
							kind.AtomKind(partIndex), kind.AtomKind(*partner));
				}
			}
		}
		++partner;
	}
}

// Calculate 1-4 nonbonded intra energy
// Calculate 1-3 nonbonded intra energy for Martini force field
void CalculateEnergy::ParticleNonbonded_1_4(double* energy,
		cbmc::TrialMol const& trialMol, XYZArray const& trialPos,
		const uint partIndex, const uint box, const uint trials) const {
	if (box >= BOXES_WITH_U_B)
		return;

	const MoleculeKind& kind = trialMol.GetKind();

	if (kind.nonBonded_1_4 == NULL)
		return;

	//loop over all partners of the trial particle
	const uint* partner = kind.sortedNB_1_4.Begin(partIndex);
	const uint* end = kind.sortedNB_1_4.End(partIndex);
	while (partner != end) {
		if (trialMol.AtomExists(*partner)) {
			for (uint t = 0; t < trials; ++t) {
				double distSq;
				currentAxes.GetDistSq(distSq, trialPos, t, trialMol.GetCoords(),
						*partner, box);
				forcefield.particles->CalcAdd_1_4(energy[t], distSq,
						kind.AtomKind(partIndex), kind.AtomKind(*partner));
			}
		}
		++partner;
	}
}

void CalculateEnergy::GetParticleEnergyGPU(uint box, double * en,
		XYZArray positions, int numAtoms, int mOff, int CurrentPos, int MolKind,
		int nLJTrials) {

	if (box >= BOXES_WITH_U_NB)
		return;

	int ThreadsPerBlock1 = 0;
	int BlocksPerGrid1 = 0;

	ThreadsPerBlock1 = MAXTHREADSPERBLOCK;
	if (ThreadsPerBlock1 == 0)
		ThreadsPerBlock1 = 1;
	BlocksPerGrid1 = ((AtomCount[box]) + ThreadsPerBlock1 - 1)
			/ ThreadsPerBlock1;

	if (BlocksPerGrid1 == 0)
		BlocksPerGrid1 = 1;

	double * dev_EnergyContrib;
	double FinalEnergyNVirial[2];

	cudaMalloc((void**) &dev_EnergyContrib,
			4 * BlocksPerGrid1 * sizeof(double));

	for (uint i = 0; i < nLJTrials; ++i) {

		Gpu_CalculateParticleInter<<<BlocksPerGrid1, ThreadsPerBlock1>>>(i,
				Gpu_kIndex, Gpu_sigmaSq, Gpu_epsilon_cn, Gpu_nOver6,
				Gpu_epsilon_cn_6, Gpu_x, Gpu_y, Gpu_z, positions.x[i],
				positions.y[i], positions.z[i], Gpu_atomKinds, numAtoms, mOff,
				CurrentPos, Gpu_start, currentAxes.axis.x[box],
				currentAxes.axis.y[box], currentAxes.axis.z[box],
				currentAxes.halfAx.x[box], currentAxes.halfAx.y[box],
				currentAxes.halfAx.z[box], (box == 0) ? 0 : AtomCount[0],
				AtomCount[box], forcefield.particles->NumKinds(),
				currentAxes.rCut, MolKind, currentAxes.rCutSq,
				dev_EnergyContrib,

				Gpu_partn);

		cudaMemcpy(&FinalEnergyNVirial, dev_EnergyContrib, 2 * sizeof(double),
				cudaMemcpyDeviceToHost);

		en[i] += FinalEnergyNVirial[0];

		cudaDeviceSynchronize();
		cudaError_t code = cudaGetLastError();
		if (code != cudaSuccess) {
			printf("Cuda error end of transfer energy calc -- %s\n",
					cudaGetErrorString(code));
			exit(2);
		}

	}
	cudaFree(dev_EnergyContrib);

}

//! Calculates Nonbonded intra energy for candidate positions in trialPos
void CalculateEnergy::ParticleInter(uint partIndex, const XYZArray& trialPos,
		double* en, uint molIndex, uint box) const {
	double distSq;
	MoleculeLookup::box_iterator molInBox = molLookup.BoxBegin(box);
	MoleculeLookup::box_iterator end = molLookup.BoxEnd(box);
	MoleculeKind const& thisKind = mols.GetKind(molIndex);
	uint kindI = thisKind.AtomKind(partIndex);

	//looping over all molecules in box
	while (molInBox != end) {
		uint otherMol = *molInBox;
		//except itself
		if (otherMol != molIndex) {
			MoleculeKind const& otherKind = mols.GetKind(otherMol);
			uint otherStart = mols.MolStart(otherMol);
			uint otherLength = otherKind.NumAtoms();
			//compare all particle pairs
			for (uint j = 0; j < otherLength; ++j) {
				uint kindJ = otherKind.AtomKind(j);
				for (uint i = 0; i < trialPos.Count(); ++i) {
					if (currentAxes.InRcut(distSq, trialPos, i, currentCoords,
							otherStart + j, box)) {
						double enr = forcefield.particles->CalcEn(distSq, kindI,
								kindJ);
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
		const uint box) const {
	Intermolecular result;

	MoleculeKind const& thisKind = mols.GetKind(molIndex);
	MoleculeLookup::box_iterator otherMol = molLookup.BoxBegin(box);
	MoleculeLookup::box_iterator end = molLookup.BoxEnd(box);

	uint partStartMolI, partLenMolI, partStartMolJ, partLenMolJ, pkI, pkJ, pI,
			pJ;
	partStartMolI = partLenMolI = partStartMolJ = partLenMolJ = pkI = pkJ = pI =
			pJ = 0;

	mols.GetRangeStartLength(partStartMolI, partLenMolI, molIndex);

	//looping over all molecules in box
	while (otherMol != end) {
		uint m2 = *otherMol;
		XYZ fMol;
		//except itself
		if (m2 != molIndex) {
			const MoleculeKind& otherKind = mols.GetKind(m2);
			//compare all particle pairs
			for (uint i = 0; i < partLenMolI; ++i) {
				pI = i + partStartMolI;
				pkI = thisKind.AtomKind(i);
				mols.GetRangeStartLength(partStartMolJ, partLenMolJ, m2);
				for (uint j = 0; j < partLenMolJ; ++j) {
					XYZ virComponents;
					double distSq;
					pJ = j + partStartMolJ;
					pkJ = otherKind.AtomKind(j);
					//Subtract old energy
					if (currentAxes.InRcut(distSq, virComponents, currentCoords,
							pI, pJ, box)) {
						double partVirial = 0.0;

						forcefield.particles->CalcAdd(result.energy, partVirial,
								distSq, pkI, pkJ);

						fMol += virComponents * partVirial;
					}
				}
			}
			//Pressure is wrt center of masses of two molecules.
			result.virial -= geom::Dot(fMol,
					currentAxes.MinImage(currentCOM.Difference(molIndex, m2),
							box));
		}
		++otherMol;
	}
	return result;
}

double CalculateEnergy::MoleculeVirial(const uint molIndex,
		const uint box) const {
	double virial = 0;
	if (box < BOXES_WITH_U_NB) {
		MoleculeLookup::box_iterator molInBox = molLookup.BoxBegin(box);
		MoleculeLookup::box_iterator end = molLookup.BoxEnd(box);

		const MoleculeKind& thisKind = mols.GetKind(molIndex);
		uint thisStart = mols.MolStart(molIndex);
		uint thisLength = thisKind.NumAtoms();
		//looping over all molecules in box
		while (molInBox != end) {
			uint otherMol = *molInBox;
			//except itself
			if (otherMol == molIndex) {
				++molInBox;
				continue;
			}
			const MoleculeKind& otherKind = mols.GetKind(otherMol);
			XYZ forceOnMol;
			for (uint i = 0; i < thisLength; ++i) {
				uint kindI = thisKind.AtomKind(i);
				uint otherStart = mols.MolStart(otherMol);
				uint otherLength = otherKind.NumAtoms();
				for (uint j = 0; j < otherLength; ++j) {
					XYZ forceVector;
					double distSq = 0.0;
					uint kindJ = otherKind.AtomKind(j);
					if (currentAxes.InRcut(distSq, forceVector, currentCoords,
							thisStart + i, otherStart + j, box)) {
						//sum forces between all particles in molecule pair
						double mag = forcefield.particles->CalcVir(distSq,
								kindI, kindJ);
						forceOnMol += forceVector * mag;
					}
				}
			}
			//correct for center of mass
			virial -= geom::Dot(forceOnMol,
					currentAxes.MinImage(
							currentCOM.Get(molIndex) - currentCOM.Get(otherMol),
							box));
			++molInBox;
		}
		double what = virial;
	}
	return virial;
}

//Calculates the change in the TC from adding numChange atoms of a kind
Intermolecular CalculateEnergy::MoleculeTailChange(const uint box,
		const uint kind, const bool add) const {
	Intermolecular delta;

	if (box < BOXES_WITH_U_NB) {

		double sign = (add ? 1.0 : -1.0);
		uint mkIdxII = kind * mols.kindsCount + kind;
		for (uint j = 0; j < mols.kindsCount; ++j) {
			uint mkIdxIJ = j * mols.kindsCount + kind;
			double rhoDeltaIJ_2 = sign * 2.0
					* (double) (molLookup.NumKindInBox(j, box))
					* currentAxes.volInv[box];
			delta.energy += mols.pairEnCorrections[mkIdxIJ] * rhoDeltaIJ_2;
			delta.virial -= mols.pairVirCorrections[mkIdxIJ] * rhoDeltaIJ_2;
		}
		//We already calculated part of the change for this type in the loop
		delta.energy += mols.pairEnCorrections[mkIdxII]
				* currentAxes.volInv[box];
		delta.virial -= mols.pairVirCorrections[mkIdxII]
				* currentAxes.volInv[box];
	}
	return delta;
}

//Calculates intramolecular energy of a full molecule
void CalculateEnergy::MoleculeIntra(double& bondEn, double& nonBondEn,
		const uint molIndex, const uint box) const {
	MoleculeKind& molKind = mols.kinds[mols.kIndex[molIndex]];
	// *2 because we'll be storing inverse bond vectors
	XYZArray bondVec(molKind.bondList.count * 2);
	BondVectors(bondVec, molKind, molIndex, box);
	MolBond(bondEn, molKind, bondVec, box);
	MolAngle(bondEn, molKind, bondVec, box);
	MolDihedral(bondEn, molKind, bondVec, box);
	MolNonbond(nonBondEn, molKind, molIndex, box);
	MolNonbond_1_4(nonBondEn, molKind, molIndex, box);
}

void CalculateEnergy::BondVectors(XYZArray & vecs, MoleculeKind const& molKind,
		const uint molIndex, const uint box) const {
	for (uint i = 0; i < molKind.bondList.count; ++i) {
		uint p1 = mols.start[molIndex] + molKind.bondList.part1[i];
		uint p2 = mols.start[molIndex] + molKind.bondList.part2[i];
		XYZ dist = currentCoords.Difference(p2, p1);
		dist = currentAxes.MinImage(dist, box);

		//store inverse vectors at i+count
		vecs.Set(i, dist);
		vecs.Set(i + molKind.bondList.count, -dist.x, -dist.y, -dist.z);
	}
}

void CalculateEnergy::MolBond(double & energy, MoleculeKind const& molKind,
		XYZArray const& vecs, const uint box) const {
	if (box >= BOXES_WITH_U_B)
		return;
	for (uint i = 0; i < molKind.bondList.count; ++i) {
		energy += forcefield.bonds.Calc(molKind.bondList.kinds[i],
				vecs.Get(i).Length());
	}
}

void CalculateEnergy::MolAngle(double & energy, MoleculeKind const& molKind,
		XYZArray const& vecs, const uint box) const {
	if (box >= BOXES_WITH_U_B)
		return;
	for (uint i = 0; i < molKind.angles.Count(); ++i) {
		double theta = Theta(vecs.Get(molKind.angles.GetBond(i, 0)),
				-vecs.Get(molKind.angles.GetBond(i, 1)));
		energy += forcefield.angles->Calc(molKind.angles.GetKind(i), theta);
	}
}

void CalculateEnergy::MolDihedral(double & energy, MoleculeKind const& molKind,
		XYZArray const& vecs, const uint box) const {
	if (box >= BOXES_WITH_U_B)
		return;
	for (uint i = 0; i < molKind.dihedrals.Count(); ++i) {
		double phi = Phi(vecs.Get(molKind.dihedrals.GetBond(i, 0)),
				vecs.Get(molKind.dihedrals.GetBond(i, 1)),
				vecs.Get(molKind.dihedrals.GetBond(i, 2)));
		energy += forcefield.dihedrals.Calc(molKind.dihedrals.GetKind(i), phi);
	}
}

void CalculateEnergy::MolNonbond(double & energy, MoleculeKind const& molKind,
		const uint molIndex, const uint box) const {
	if (box >= BOXES_WITH_U_B)
		return;

	double distSq;
	double virial; //we will throw this away
	for (uint i = 0; i < molKind.nonBonded.count; ++i) {
		uint p1 = mols.start[molIndex] + molKind.nonBonded.part1[i];
		uint p2 = mols.start[molIndex] + molKind.nonBonded.part2[i];
		currentAxes.GetDistSq(distSq, currentCoords, p1, p2, box); // v1
		if (currentAxes.InRcut(distSq, currentCoords, p1, p2, box)) {
			forcefield.particles->CalcAdd(energy, virial, distSq,
					molKind.AtomKind(molKind.nonBonded.part1[i]),
					molKind.AtomKind(molKind.nonBonded.part2[i]));
		}
	}
}

void CalculateEnergy::MolNonbond_1_4(double & energy,
		MoleculeKind const& molKind, const uint molIndex,
		const uint box) const {
	if (box >= BOXES_WITH_U_B || molKind.nonBonded_1_4 == NULL)
		return;

	double distSq;
	for (uint i = 0; i < molKind.nonBonded_1_4->count; ++i) {
		uint p1 = mols.start[molIndex] + molKind.nonBonded_1_4->part1[i];
		uint p2 = mols.start[molIndex] + molKind.nonBonded_1_4->part2[i];
		currentAxes.GetDistSq(distSq, currentCoords, p1, p2, box);
		forcefield.particles->CalcAdd_1_4(energy, distSq,
				molKind.AtomKind(molKind.nonBonded_1_4->part1[i]),
				molKind.AtomKind(molKind.nonBonded_1_4->part2[i]));
	}
}

//!Calculates energy and virial tail corrections for the box
void CalculateEnergy::FullTailCorrection(SystemPotential& pot,
		const BoxDimensions& boxAxes) const {
	for (uint box = 0; box < BOXES_WITH_U_NB; ++box) {
		double en = 0.0;
		double vir = 0.0;

		for (uint i = 0; i < mols.kindsCount; ++i) {
			uint numI = molLookup.NumKindInBox(i, box);
			for (uint j = 0; j < mols.kindsCount; ++j) {
				uint numJ = molLookup.NumKindInBox(j, box);
				en += mols.pairEnCorrections[i * mols.kindsCount + j] * numI
						* numJ * boxAxes.volInv[box];
				vir -= mols.pairVirCorrections[i * mols.kindsCount + j] * numI
						* numJ * boxAxes.volInv[box];
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

	if (ThreadsPerBlock1 == 0) {
		ThreadsPerBlock1 = 1;
	}

	BlocksPerGrid1 = ((MolCount[0] * (MolCount[0] - 1) / 2) + ThreadsPerBlock1
			- 1) / ThreadsPerBlock1;

	if (BlocksPerGrid1 == 0) {
		BlocksPerGrid1 = 1;
	}

	int XBlocks, YBlocks;

	if (BlocksPerGrid1 < MAXTBLOCKS) {
		XBlocks = 1;
		YBlocks = BlocksPerGrid1;
	} else {
		YBlocks = MAXTBLOCKS;
		XBlocks = (int) ceil((double) BlocksPerGrid1 / (double) MAXTBLOCKS);
	}

	dim3 blcks(XBlocks, YBlocks, 1);
	double * dev_EnergyContrib, *dev_VirialContrib, *dev_RealEnergyContrib;
	cudaMalloc((void**) &dev_EnergyContrib,
			4 * XBlocks * YBlocks * sizeof(double));
	cudaMalloc((void**) &dev_VirialContrib,
			4 * XBlocks * YBlocks * sizeof(double));	cudaMalloc((void**) &dev_RealEnergyContrib, 4 * XBlocks * YBlocks * sizeof(double));
	double FinalEnergyNVirial[2];	double RealEnergy[2];
	FinalEnergyNVirial[0] = 0.0;
	FinalEnergyNVirial[1] = 0.0;	RealEnergy[0] = RealEnergy[1] = 0.0;
	double NewEnergy1 = 0.0;
	double NewVirial1 = 0.0;	double RealEnergy1 = 0.0;
	//Box 0
	dim3 thrds(ThreadsPerBlock1, 1, 1);

	if (MolCount[0] != 0) {

		Gpu_CalculateSystemInter<<<blcks, thrds, 0, stream0>>>(0,
				NoOfAtomsPerMol, Gpu_atomKinds, Gpu_kIndex, Gpu_sigmaSq,
				Gpu_epsilon_cn, Gpu_nOver6, Gpu_epsilon_cn_6, Gpu_x, Gpu_y,
				Gpu_z, Gpu_COMX, Gpu_COMY, Gpu_COMZ, Gpu_start,
				currentAxes.axis.x[0], currentAxes.axis.y[0],
				currentAxes.axis.z[0], currentAxes.halfAx.x[0],
				currentAxes.halfAx.y[0], currentAxes.halfAx.z[0], offset,
				MolCount[0], MolCount[0] / 2, forcefield.particles->NumKinds(),
				currentAxes.rCut, (MolCount[0] % 2 == 0) ? 1 : 0,
				currentAxes.rCutSq, dev_EnergyContrib, dev_VirialContrib,
				(MolCount[0] * (MolCount[0] - 1) / 2), 0, dev_particleCharge, dev_calp,				dev_DoEwald, dev_alpha, dev_Kmax1, dev_RealEnergyContrib,
				Gpu_partn);
		cudaMemcpyAsync(&FinalEnergyNVirial, dev_EnergyContrib,
				2 * sizeof(double), cudaMemcpyDeviceToHost, stream0);		cudaMemcpyAsync(&RealEnergy, dev_RealEnergyContrib, 2*sizeof(double), cudaMemcpyDeviceToHost, stream0);
		cudaStreamSynchronize(stream0);
		NewEnergy1 = FinalEnergyNVirial[0];
		NewVirial1 = FinalEnergyNVirial[1];
		RealEnergy1 = RealEnergy[0];
	}

#if ENSEMBLE == GEMC

	ThreadsPerBlock1 = MAXTHREADSPERBLOCK;

	if (ThreadsPerBlock1 == 0) {
		ThreadsPerBlock1 = 1;
	}

	BlocksPerGrid1 = ((MolCount[1] * (MolCount[1] - 1) / 2) + ThreadsPerBlock1
			- 1) / ThreadsPerBlock1;

	if (BlocksPerGrid1 == 0) {
		BlocksPerGrid1 = 1;
	}

	if (BlocksPerGrid1 < MAXTBLOCKS) {
		XBlocks = 1;
		YBlocks = BlocksPerGrid1;
	} else {
		YBlocks = MAXTBLOCKS;
		XBlocks = (int) ceil((double) BlocksPerGrid1 / (double) MAXTBLOCKS);
	}

	dim3 blcks1(XBlocks, YBlocks, 1);
	double * dev_EnergyContrib1, *dev_VirialContrib1, *dev_RealEnergyContrib1;
	cudaMalloc((void**) &dev_EnergyContrib1,
			4 * XBlocks * YBlocks * sizeof(double));
	cudaMalloc((void**) &dev_VirialContrib1,
			4 * XBlocks * YBlocks * sizeof(double));	cudaMalloc((void**) &dev_RealEnergyContrib1, 4 * XBlocks * YBlocks * sizeof(double));
	dim3 thrds1(ThreadsPerBlock1, 1, 1);
	FinalEnergyNVirial[0] = 0.0;
	FinalEnergyNVirial[1] = 0.0;	RealEnergy[0] = RealEnergy[1] = 0.0;
	offset = AtomCount[0];
	double NewEnergy2 = 0.0;
	double NewVirial2 = 0.0;	double RealEnergy2 = 0.0;

	if (MolCount[1] != 0) {

		Gpu_CalculateSystemInter<<<blcks1, thrds1, 0, stream1>>>(0,
				NoOfAtomsPerMol, Gpu_atomKinds, Gpu_kIndex, Gpu_sigmaSq,
				Gpu_epsilon_cn, Gpu_nOver6, Gpu_epsilon_cn_6, Gpu_x, Gpu_y,
				Gpu_z, Gpu_COMX, Gpu_COMY, Gpu_COMZ, Gpu_start,
				currentAxes.axis.x[1], currentAxes.axis.y[1],
				currentAxes.axis.z[1], currentAxes.halfAx.x[1],
				currentAxes.halfAx.y[1], currentAxes.halfAx.z[1], offset,
				MolCount[1], MolCount[1] / 2, forcefield.particles->NumKinds(),
				currentAxes.rCut, (MolCount[1] % 2 == 0) ? 1 : 0,
				currentAxes.rCutSq, dev_EnergyContrib1, dev_VirialContrib1,
				(MolCount[1] * (MolCount[1] - 1) / 2), MolCount[0], dev_particleCharge,				dev_calp, dev_DoEwald, dev_alpha, dev_Kmax1, dev_RealEnergyContrib1,				Gpu_partn);
		cudaMemcpyAsync(&FinalEnergyNVirial, dev_EnergyContrib1,
				2 * sizeof(double), cudaMemcpyDeviceToHost, stream1);		cudaMemcpyAsync(&RealEnergy, dev_RealEnergyContrib1, 2*sizeof(double), cudaMemcpyDeviceToHost, stream0);
		cudaStreamSynchronize(stream1);
		NewEnergy2 = FinalEnergyNVirial[0];
		NewVirial2 = FinalEnergyNVirial[1];
		RealEnergy2 = RealEnergy[0];
		printf("inter energy box 1=%f\n", FinalEnergyNVirial[0]);		printf("real energy box 1 =%f\n", RealEnergy2);
	}

#endif

	SystemPotential currentpot;

	currentpot.boxEnergy[0].inter = NewEnergy1;
	currentpot.boxVirial[0].inter = NewVirial1;	currentpot.boxEnergy[0].real = RealEnergy1 * qqfact[0];		//ewald	currentpot.boxEnergy[0].elect += currentpot.boxEnergy[0].real;	printf("inter energy box 0=%.8lf\n", currentpot.boxEnergy[0].inter);	printf("real energy box 0 =%.8lf\n", currentpot.boxEnergy[0].real);

	cudaFree(dev_EnergyContrib);
	cudaFree(dev_VirialContrib);	cudaFree(dev_RealEnergyContrib);

#if ENSEMBLE == GEMC

	currentpot.boxEnergy[1].inter = NewEnergy2;

	currentpot.boxVirial[1].inter = NewVirial2;	currentpot.boxEnergy[1].real = RealEnergy2 * qqfact[0];		//ewald	currentpot.boxEnergy[1].elect += currentpot.boxEnergy[0].real;

	cudaFree(dev_EnergyContrib1);
	cudaFree(dev_VirialContrib1);	cudaFree(dev_RealEnergyContrib);
#endif

	currentpot.Total();

	cudaDeviceSynchronize();
	cudaError_t code = cudaGetLastError();

	if (code != cudaSuccess) {
		printf("Cuda error at end of Calculate Total Energy -- %s\n",
				cudaGetErrorString(code));
		exit(2);
	}

	return currentpot;
}
__device__ void CalcRealGpu(double distSq, double* dev_calp, double& realEnergy,		double* particleCharge, int tarParticle, int ParticleNumber, int box);
__global__ void Gpu_CalculateSystemInter(uint step, uint * NoOfAtomsPerMol,
		uint *AtomKinds, uint *molKindIndex, double * sigmaSq,
		double * epsilon_cn, double * nOver6, double * epsilon_cn_6, double *x,
		double *y, double *z, double * Gpu_COMX, double * Gpu_COMY,
		double * Gpu_COMZ, uint * Gpu_start, double xAxis, double yAxis,
		double zAxis, double xHalfAxis, double yHalfAxis, double zHalfAxis,
		uint boxOffset, uint MoleculeCount, uint HalfMoleculeCount,
		uint FFParticleKindCount, double rCut, uint isEvenMolCount,
		double rCutSq, double dev_EnergyContrib[], double dev_VirialContrib[],
		uint limit, uint MolOffset, double *dev_particleCharge, double *dev_calp,		bool *dev_DoEwald, double *dev_alpha, double *dev_Kmax1, double *dev_RealEnergyContrib,

#ifdef MIE_INT_ONLY
		const uint * n,
#else
		const double * n
#endif

		) {
	__shared__
	double cacheEnergy[MAXTHREADSPERBLOCK];
	__shared__
	double cacheVirial[MAXTHREADSPERBLOCK];	__shared__ double realEnergy[MAXTHREADSPERBLOCK];
	__shared__ bool LastBlock;
	int blockId = blockIdx.y * gridDim.x + blockIdx.x;
	int MolId = blockId * blockDim.x + threadIdx.x;
	cacheEnergy[threadIdx.x] = 0.0;
	cacheVirial[threadIdx.x] = 0.0;	realEnergy[threadIdx.x] = 0.0;

	if (MolId < limit) {
		double energy = 0.0, virial = 0.0, real = 0.0;
		double distSq = 0.0;
		double partVirial = 0.0;
		XYZ virComponents;
		XYZ fMol;
		XYZ temp;
		int i = MolId / (HalfMoleculeCount) + (isEvenMolCount);
		int j = MolId % (HalfMoleculeCount);

		if (j >= i) { // re-map
			i = MoleculeCount - i - 1 + (isEvenMolCount);
			j = MoleculeCount - j - 2 + (isEvenMolCount);
		}

		int iStart = Gpu_start[i + MolOffset];
		int jStart = Gpu_start[j + MolOffset];

		for (int k = 0; k < NoOfAtomsPerMol[i + MolOffset]; k++) {
			for (int m = 0; m < NoOfAtomsPerMol[j + MolOffset]; m++) {

				if (InRcutGpuSigned(distSq, x, y, z, xAxis, yAxis, zAxis,
						xHalfAxis, yHalfAxis, zHalfAxis, k + iStart, m + jStart,
						rCut, rCutSq, virComponents)) {
					partVirial = 0.0;

					CalcAddGpu(energy, partVirial, distSq,
							AtomKinds[iStart + k], AtomKinds[jStart + m],
							FFParticleKindCount, sigmaSq, epsilon_cn, nOver6,
							epsilon_cn_6, n);

					fMol += (virComponents * partVirial);#ifdef REAL					if(dev_DoEwald[0]){						CalcRealGpu(distSq, dev_calp, real, dev_particleCharge,								k+iStart, m+jStart, (boxOffset == 0) ? 0:1);					}#endif

				}
			}

		}

		temp.x = MinImageSignedGpu(
				Gpu_COMX[i + MolOffset] - Gpu_COMX[j + MolOffset], xAxis,
				xHalfAxis);
		temp.y = MinImageSignedGpu(
				Gpu_COMY[i + MolOffset] - Gpu_COMY[j + MolOffset], yAxis,
				yHalfAxis);
		temp.z = MinImageSignedGpu(
				Gpu_COMZ[i + MolOffset] - Gpu_COMZ[j + MolOffset], zAxis,
				zHalfAxis);
		virial -= geom::Dot(fMol, temp);
		cacheEnergy[threadIdx.x] += energy;
		cacheVirial[threadIdx.x] += virial;		realEnergy[threadIdx.x] += real;
	}

	__syncthreads();
	// add data
	int offset = 1 << (int) __log2f((float) blockDim.x);

	if (blockDim.x < MAXTHREADSPERBLOCK) {
		if ((threadIdx.x + offset) < 4 * limit) {
			cacheEnergy[threadIdx.x] += cacheEnergy[threadIdx.x + offset];
			cacheVirial[threadIdx.x] += cacheVirial[threadIdx.x + offset];			realEnergy[threadIdx.x] += realEnergy[threadIdx.x + offset];
		}

		__syncthreads();
	}
	for (int i = offset >> 1; i > 32; i >>= 1) {
		if (threadIdx.x < i) {
			cacheEnergy[threadIdx.x] += cacheEnergy[threadIdx.x + i];
			cacheVirial[threadIdx.x] += cacheVirial[threadIdx.x + i];			realEnergy[threadIdx.x] += realEnergy[threadIdx.x + i];
		}           //end if

		__syncthreads();
	}           //end for

	if (threadIdx.x < 32) {
		offset = min(offset, 64);

		switch (offset) {
		case 64:
			cacheEnergy[threadIdx.x] += cacheEnergy[threadIdx.x + 32];
			cacheVirial[threadIdx.x] += cacheVirial[threadIdx.x + 32];			realEnergy[threadIdx.x] += realEnergy[threadIdx.x + 32];
			__threadfence_block();

		case 32:
			cacheEnergy[threadIdx.x] += cacheEnergy[threadIdx.x + 16];
			cacheVirial[threadIdx.x] += cacheVirial[threadIdx.x + 16];			realEnergy[threadIdx.x] += realEnergy[threadIdx.x + 16];
			__threadfence_block();

		case 16:
			cacheEnergy[threadIdx.x] += cacheEnergy[threadIdx.x + 8];
			cacheVirial[threadIdx.x] += cacheVirial[threadIdx.x + 8];			realEnergy[threadIdx.x] += realEnergy[threadIdx.x + 8];
			__threadfence_block();

		case 8:
			cacheEnergy[threadIdx.x] += cacheEnergy[threadIdx.x + 4];
			cacheVirial[threadIdx.x] += cacheVirial[threadIdx.x + 4];			realEnergy[threadIdx.x] += realEnergy[threadIdx.x + 4];
			__threadfence_block();

		case 4:
			cacheEnergy[threadIdx.x] += cacheEnergy[threadIdx.x + 2];
			cacheVirial[threadIdx.x] += cacheVirial[threadIdx.x + 2];			realEnergy[threadIdx.x] += realEnergy[threadIdx.x + 2];
			__threadfence_block();

		case 2:
			cacheEnergy[threadIdx.x] += cacheEnergy[threadIdx.x + 1];
			cacheVirial[threadIdx.x] += cacheVirial[threadIdx.x + 1];			realEnergy[threadIdx.x] += realEnergy[threadIdx.x + 1];
		}
	}           //end if

	if (threadIdx.x == 0) {
		dev_EnergyContrib[blockId] = cacheEnergy[0];
		dev_VirialContrib[blockId] = cacheVirial[0];		dev_RealEnergyContrib[blockId] = realEnergy[0];
		//Compress these two lines to one to save one variable declaration
		__threadfence();
		LastBlock = (atomicInc(&BlocksDone, gridDim.x * gridDim.y)
				== (gridDim.x * gridDim.y) - 1);
	}       //end if

	__syncthreads();

	if (LastBlock) {
		//If this thread corresponds to a valid block
		if (threadIdx.x < gridDim.x * gridDim.y) {
			cacheEnergy[threadIdx.x] = dev_EnergyContrib[threadIdx.x];
			cacheVirial[threadIdx.x] = dev_VirialContrib[threadIdx.x];			realEnergy[threadIdx.x] = dev_RealEnergyContrib[threadIdx.x];
		}

		for (int i = blockDim.x; threadIdx.x + i < gridDim.x * gridDim.y; i +=
				blockDim.x) {
			if (threadIdx.x + i < gridDim.x * gridDim.y) {
				cacheEnergy[threadIdx.x] += dev_EnergyContrib[threadIdx.x + i];
				cacheVirial[threadIdx.x] += dev_VirialContrib[threadIdx.x + i];				realEnergy[threadIdx.x] += dev_RealEnergyContrib[threadIdx.x + i];
			}       //end if
		}       //end for

		__syncthreads();
		offset = 1
				<< (int) __log2f(
						(float) min(blockDim.x, gridDim.x * gridDim.y));

		for (int i = offset; i > 0; i >>= 1) {
			if (threadIdx.x < i
					&& threadIdx.x + i
							< min(blockDim.x, gridDim.x * gridDim.y)) {
				cacheEnergy[threadIdx.x] += cacheEnergy[threadIdx.x + i];
				cacheVirial[threadIdx.x] += cacheVirial[threadIdx.x + i];				realEnergy[threadIdx.x] += realEnergy[threadIdx.x + i];
			}       //end if

			__syncthreads();
		}       //end for

		if (threadIdx.x == 0) {
			BlocksDone = 0;
			dev_EnergyContrib[0] = cacheEnergy[0];
			dev_EnergyContrib[1] = cacheVirial[0];
			dev_VirialContrib[0] = cacheVirial[0];
			dev_RealEnergyContrib[0] = realEnergy[0];			printf("real: %lf\n", dev_RealEnergyContrib[0]);
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
	if (v > ax) //if +, wrap out to low end
		v -= ax;
	else if (v < 0) //if -, wrap to high end
		v += ax;
	return v;
#endif
}
__device__ double Gpu_BoltzW(const double a, const double x) {
	return exp(-1.0 * a * x);
}

__device__ void CalcSubGpu(double& en, double& vir, const double distSq,
		const uint kind1, const uint kind2, uint count, double * sigmaSq,
		double * epsilon_cn, double * nOver6, double * epsilon_cn_6,
#ifdef MIE_INT_ONLY
		const uint * n,
#else
		const double * n
#endif

		) {
	uint index = kind1 + kind2 * count;
	double rNeg2 = 1.0 / distSq;
	double rRat2 = sigmaSq[index] * rNeg2;
	double rRat4 = rRat2 * rRat2;
	double attract = rRat4 * rRat2;
#ifdef MIE_INT_ONLY
	double repulse = pow(rRat2, rRat4, attract, n);
#else
	double repulse = pow(__dsqrt_rn(rRat2), n[index]);
#endif
	en -= epsilon_cn[index] * (repulse - attract);
	vir = -1.0 * epsilon_cn_6[index] * (nOver6[index] * repulse - attract)
			* rNeg2;

}

__device__ bool InRcutGpuSigned(double &distSq, double *Array, int len,
		double *x2, double *y2, double *z2, double xAxis, double yAxis,
		double zAxis, double xHalfAxis, double yHalfAxis, double zHalfAxis,
		const uint i, const uint j, double rCut, double rCutSq, XYZ &dist) {
	distSq = 0;
	dist.x = Array[i] - x2[j];
	dist.x = MinImageSignedGpu(dist.x, xAxis, xHalfAxis);
	dist.y = Array[len + i] - y2[j];
	dist.y = MinImageSignedGpu(dist.y, yAxis, yHalfAxis);
	dist.z = Array[2 * len + i] - z2[j];
	dist.z = MinImageSignedGpu(dist.z, zAxis, zHalfAxis);
	distSq = dist.x * dist.x + dist.y * dist.y + dist.z * dist.z;
	return (rCutSq > distSq);
}

__device__ bool InRcutGpuSigned(double &distSq, double *x, double *y, double *z,
		const double xi, const double yi, const double zi, double xAxis,
		double yAxis, double zAxis, double xHalfAxis, double yHalfAxis,
		double zHalfAxis, const uint i, const uint j, double rCut,
		double rCutSq, XYZ & dist) {
	distSq = 0;
	dist.x = xi - x[j];
	dist.x = MinImageSignedGpu(dist.x, xAxis, xHalfAxis);
	dist.y = yi - y[j];
	dist.y = MinImageSignedGpu(dist.y, yAxis, yHalfAxis);
	dist.z = zi - z[j];
	dist.z = MinImageSignedGpu(dist.z, zAxis, zHalfAxis);
	distSq = dist.x * dist.x + dist.y * dist.y + dist.z * dist.z;
	return (rCutSq > distSq);
}

__device__ bool InRcutGpuSignedNoVir(double &distSq, double *x, double *y,
		double *z, const double xi, const double yi, const double zi,
		double xAxis, double yAxis, double zAxis, double xHalfAxis,
		double yHalfAxis, double zHalfAxis, const uint i, const uint j,
		double rCut, double rCutSq) {
	double dx, dy, dz;
	distSq = 0;
	dx = xi - x[j];
	dx = MinImageSignedGpu(dx, xAxis, xHalfAxis);
	dy = yi - y[j];
	dy = MinImageSignedGpu(dy, yAxis, yHalfAxis);
	dz = zi - z[j];
	dz = MinImageSignedGpu(dz, zAxis, zHalfAxis);
	distSq = dx * dx + dy * dy + dz * dz;
	return (rCutSq > distSq);
}

__device__ void Gpu_CalculateMoleculeInter(uint * NoOfAtomsPerMol,
		uint *AtomKinds, uint *molKindIndex, double * sigmaSq,
		double * epsilon_cn, double * nOver6, double * epsilon_cn_6,
		double *oldx, double *oldy, double *oldz, double * Gpu_COMX,
		double * Gpu_COMY, double * Gpu_COMZ, double selectedCOMX,
		double selectedCOMY, double selectedCOMZ, double *newArray, int len,
		uint MolId, uint * Gpu_start, double xAxis, double yAxis, double zAxis,
		double xHalfAxis, double yHalfAxis, double zHalfAxis, uint boxOffset,
		uint MoleculeCount, uint mIndex, uint FFParticleKindCount, double rCut,
		uint molKInd, double rCutSq, double dev_EnergyContrib[],
		double dev_VirialContrib[], double *dev_calp, double *particleCharge,		double *dev_RealEnergyContrib, int box,
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
	__shared__ bool LastBlock;	__shared__ double cacheRealEnergy[MAXTHREADSPERBLOCK];
	cacheEnergy[threadIdx.x] = 0.0;
	cacheVirial[threadIdx.x] = 0.0;	cacheRealEnergy[threadIdx.x] = 0.0;

	if (MolId < MoleculeCount + boxOffset && mIndex != MolId) {
		double energy = 0.0, virial = 0.0, realSub = 0.0, realAdd = 0.0;
		double distSq = 0.0;
		XYZ fMolO, fMolN;
		double partVirial = 0.0;
		XYZ virComponents;
		XYZ temp;
		uint MolStart = Gpu_start[MolId];
		uint SelectedMolStart = Gpu_start[mIndex]; // put in shared memory to be used always

		for (int i = 0; i < len; i++) { //original mol that we are moving
			for (int j = 0; j < NoOfAtomsPerMol[MolId]; j++) { // the mol we compare with
				if (InRcutGpuSigned(distSq, oldx, oldy, oldz, xAxis, yAxis,
						zAxis, xHalfAxis, yHalfAxis, zHalfAxis,
						i + SelectedMolStart, j + MolStart, rCut, rCutSq,
						virComponents)) {
					partVirial = 0.0;
					CalcSubGpu(energy, partVirial, distSq,
							AtomKinds[i + SelectedMolStart],
							AtomKinds[j + MolStart], FFParticleKindCount,
							sigmaSq, epsilon_cn, nOver6, epsilon_cn_6, n);					CalcRealGpu(distSq, dev_calp, realSub, particleCharge,							i + SelectedMolStart, j + MolStart, box);
					fMolO = virComponents * partVirial;

					temp.x = MinImageSignedGpu(
							Gpu_COMX[mIndex] - Gpu_COMX[MolId], xAxis,
							xHalfAxis);
					temp.y = MinImageSignedGpu(
							Gpu_COMY[mIndex] - Gpu_COMY[MolId], yAxis,
							yHalfAxis);
					temp.z = MinImageSignedGpu(
							Gpu_COMZ[mIndex] - Gpu_COMZ[MolId], zAxis,
							zHalfAxis);
					virial -= geom::Dot(fMolO, temp);

				}

				if (InRcutGpuSigned(distSq, newArray, len, oldx, oldy, oldz,
						xAxis, yAxis, zAxis, xHalfAxis, yHalfAxis, zHalfAxis, i,
						j + MolStart, rCut, rCutSq, virComponents)) {
					partVirial = 0.0;

					CalcAddGpu(energy, partVirial, distSq,
							AtomKinds[i + SelectedMolStart],
							AtomKinds[j + MolStart], FFParticleKindCount,
							sigmaSq, epsilon_cn, nOver6, epsilon_cn_6, n);					CalcRealGpu(distSq, dev_calp, realAdd, particleCharge,							i + SelectedMolStart, j + MolStart, box);
					fMolN = virComponents * partVirial;

					temp.x = MinImageSignedGpu(selectedCOMX - Gpu_COMX[MolId],
							xAxis, xHalfAxis);
					temp.y = MinImageSignedGpu(selectedCOMY - Gpu_COMY[MolId],
							yAxis, yHalfAxis);
					temp.z = MinImageSignedGpu(selectedCOMZ - Gpu_COMZ[MolId],
							zAxis, zHalfAxis);
					virial -= geom::Dot(fMolN, temp);

				}
			}
		}

		cacheEnergy[threadIdx.x] += energy;
		cacheVirial[threadIdx.x] += virial;
		cacheRealEnergy[threadIdx.x] += (realAdd - realSub);
	}

	__syncthreads();
	// add data
	int offset = 1 << (int) __log2f((float) blockDim.x);

	if (blockDim.x < MAXTHREADSPERBLOCK) {
		if ((threadIdx.x + offset) < MoleculeCount) {
			cacheEnergy[threadIdx.x] += cacheEnergy[threadIdx.x + offset];
			cacheVirial[threadIdx.x] += cacheVirial[threadIdx.x + offset];			cacheRealEnergy[threadIdx.x] += cacheRealEnergy[threadIdx.x + offset];
		}
		__syncthreads();
	}

	for (int i = offset >> 1; i > 32; i >>= 1) {
		if (threadIdx.x < i) {
			cacheEnergy[threadIdx.x] += cacheEnergy[threadIdx.x + i];
			cacheVirial[threadIdx.x] += cacheVirial[threadIdx.x + i];			cacheRealEnergy[threadIdx.x] += cacheRealEnergy[threadIdx.x + i];
		}           //end if

		__syncthreads();
	}           //end for

	if (threadIdx.x < 32) {
		offset = min(offset, 64);

		switch (offset) {
		case 64:
			cacheEnergy[threadIdx.x] += cacheEnergy[threadIdx.x + 32];
			cacheVirial[threadIdx.x] += cacheVirial[threadIdx.x + 32];			cacheRealEnergy[threadIdx.x] += cacheRealEnergy[threadIdx.x + 32];
			__threadfence_block();

		case 32:
			cacheEnergy[threadIdx.x] += cacheEnergy[threadIdx.x + 16];
			cacheVirial[threadIdx.x] += cacheVirial[threadIdx.x + 16];			cacheRealEnergy[threadIdx.x] += cacheRealEnergy[threadIdx.x + 16];
			__threadfence_block();

		case 16:
			cacheEnergy[threadIdx.x] += cacheEnergy[threadIdx.x + 8];
			cacheVirial[threadIdx.x] += cacheVirial[threadIdx.x + 8];			cacheRealEnergy[threadIdx.x] += cacheRealEnergy[threadIdx.x + 8];
			__threadfence_block();

		case 8:
			cacheEnergy[threadIdx.x] += cacheEnergy[threadIdx.x + 4];
			cacheVirial[threadIdx.x] += cacheVirial[threadIdx.x + 4];			cacheRealEnergy[threadIdx.x] += cacheRealEnergy[threadIdx.x + 4];
			__threadfence_block();

		case 4:
			cacheEnergy[threadIdx.x] += cacheEnergy[threadIdx.x + 2];
			cacheVirial[threadIdx.x] += cacheVirial[threadIdx.x + 2];			cacheRealEnergy[threadIdx.x] += cacheRealEnergy[threadIdx.x + 2];
			__threadfence_block();

		case 2:
			cacheEnergy[threadIdx.x] += cacheEnergy[threadIdx.x + 1];
			cacheVirial[threadIdx.x] += cacheVirial[threadIdx.x + 1];			cacheRealEnergy[threadIdx.x] += cacheRealEnergy[threadIdx.x + 1];
		}
	}

	if (threadIdx.x == 0) {
		dev_EnergyContrib[blockIdx.x] = cacheEnergy[0];
		dev_VirialContrib[blockIdx.x] = cacheVirial[0];		dev_RealEnergyContrib[blockIdx.x] = cacheRealEnergy[0];
		__threadfence();
		LastBlock = (atomicInc(&BlocksDone, gridDim.x) == gridDim.x - 1);

		if (LastBlock) {
			BlockNum = blockIdx.x;
		}
	}       //end if

	__syncthreads();

	if (LastBlock) {
		//If this thread corresponds to a valid block
		if (threadIdx.x < gridDim.x) {
			cacheEnergy[threadIdx.x] = dev_EnergyContrib[threadIdx.x];
			cacheVirial[threadIdx.x] = dev_VirialContrib[threadIdx.x];			cacheRealEnergy[threadIdx.x] = dev_RealEnergyContrib[threadIdx.x];
		}

		for (int i = blockDim.x; threadIdx.x + i < gridDim.x; i += blockDim.x) {
			if (threadIdx.x + i < gridDim.x) {
				cacheEnergy[threadIdx.x] += dev_EnergyContrib[threadIdx.x + i];
				cacheVirial[threadIdx.x] += dev_VirialContrib[threadIdx.x + i];				cacheRealEnergy[threadIdx.x] = dev_RealEnergyContrib[threadIdx.x + i];
			}       //end if
		}       //end for

		__syncthreads();
		offset = 1 << (int) __log2f((float) min(blockDim.x, gridDim.x));

		for (int i = offset; i > 0; i >>= 1) {
			if (threadIdx.x < i
					&& threadIdx.x + i < min(blockDim.x, gridDim.x)) {
				cacheEnergy[threadIdx.x] += cacheEnergy[threadIdx.x + i];
				cacheVirial[threadIdx.x] += cacheVirial[threadIdx.x + i];				cacheRealEnergy[threadIdx.x] += cacheRealEnergy[threadIdx.x + i];
			}       //end if

			__syncthreads();
		}       //end for

		if (threadIdx.x == 0) {			BlocksDone = 0;
			dev_EnergyContrib[0] = cacheEnergy[0];
			dev_EnergyContrib[1] = cacheVirial[0];
			dev_VirialContrib[0] = cacheVirial[0];			dev_RealEnergyContrib[0] = cacheRealEnergy[0];//			printf("inter change: %lf, real change; %lf\n", dev_EnergyContrib[0], dev_RealEnergyContrib[0]);
		}
	}
}

__global__ void TryTransformGpu(uint * NoOfAtomsPerMol, uint *AtomKinds,
		SystemPotential * Gpu_Potential, double * Gpu_x, double * Gpu_y,
		double * Gpu_z, double * Gpu_COMX, double * Gpu_COMY, double * Gpu_COMZ,

		XYZ shift, double xAxis, double yAxis, double zAxis, uint *molKindIndex,
		double * sigmaSq, double * epsilon_cn, double * nOver6,
		double * epsilon_cn_6, double beta, double AcceptRand, uint * Gpu_start,
		int len, double xHalfAxis, double yHalfAxis, double zHalfAxis,
		uint boxOffset,

		uint MoleculeCount,
		uint mIndex,       // mol index with offset
		uint FFParticleKindCount, double rCut, uint molKInd, double rCutSq,

		double dev_EnergyContrib[], double dev_VirialContrib[], uint boxIndex,
		bool * Gpu_result, double *dev_calp, double *dev_particleCharge,		double *dev_RealEnergyContrib, XYZ *dev_newCOM, double *dev_Array,
#ifdef MIE_INT_ONLY
		const uint * n,
#else
		const double * n
#endif

		)

		{
	extern __shared__ double Array[];
	__shared__ XYZ newCOM;
	int threadId = blockIdx.x * blockDim.x + threadIdx.x;
	int MolId = threadId + boxOffset;

	if (threadIdx.x < len) {
		uint start;
		start = Gpu_start[mIndex];
		newCOM.x = Gpu_COMX[mIndex];
		newCOM.y = Gpu_COMY[mIndex];
		newCOM.z = Gpu_COMZ[mIndex];
		newCOM += shift;
		WrapPBC(newCOM.x, xAxis);
		WrapPBC(newCOM.y, yAxis);
		WrapPBC(newCOM.z, zAxis);		dev_newCOM[0].x = newCOM.x;		dev_newCOM[0].y = newCOM.y;		dev_newCOM[0].z = newCOM.z;
		Array[threadIdx.x] = Gpu_x[start + threadIdx.x] + shift.x;
		Array[len + threadIdx.x] = Gpu_y[start + threadIdx.x] + shift.y;
		Array[2 * len + threadIdx.x] = Gpu_z[start + threadIdx.x] + shift.z;
		WrapPBC(Array[threadIdx.x], xAxis);
		WrapPBC(Array[len + threadIdx.x], yAxis);
		WrapPBC(Array[2 * len + threadIdx.x], zAxis);		dev_Array[threadIdx.x] = Array[threadIdx.x];		dev_Array[len + threadIdx.x] = Array[len + threadIdx.x];		dev_Array[2 * len + threadIdx.x] = Array[2 * len + threadIdx.x];
	}

	__syncthreads();

	if (boxIndex < BOXES_WITH_U_NB)

		Gpu_CalculateMoleculeInter(NoOfAtomsPerMol, AtomKinds, molKindIndex,
				sigmaSq, epsilon_cn, nOver6, epsilon_cn_6, Gpu_x, Gpu_y, Gpu_z,
				Gpu_COMX, Gpu_COMY, Gpu_COMZ, newCOM.x, newCOM.y, newCOM.z,
				Array, len, MolId, Gpu_start, xAxis, yAxis, zAxis, xHalfAxis,
				yHalfAxis, zHalfAxis, boxOffset, MoleculeCount,
				mIndex,       // mol index with offset
				FFParticleKindCount, rCut, molKInd, rCutSq, dev_EnergyContrib,
				dev_VirialContrib, dev_calp, dev_particleCharge, dev_RealEnergyContrib,				boxIndex, n);

	else {
		dev_EnergyContrib[0] = 0.0;
		dev_EnergyContrib[1] = 0.0;

	}
}

__device__ double UnwrapPBC(double& v, const double ref, const double ax,
		const double halfAx) {
	//If absolute value of X dist btwn pt and ref is > 0.5 * box_axis
	//If ref > 0.5 * box_axis, add box_axis to pt (to wrap out + side)
	//If ref < 0.5 * box_axis, subtract box_axis (to wrap out - side)
	// uses bit hack to avoid branching for conditional
#ifdef NO_BRANCHING_UNWRAP
	bool negate = ( ref > halfAx );
	double vDiff = v + (ax ^ -negate) + negate;
	return (fabs(ref - v) > halfAx ) ? v : vDiff;
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

__global__ void TryRotateGpu(uint * NoOfAtomsPerMol, uint *AtomKinds,
		SystemPotential * Gpu_Potential, TransformMatrix matrix, double * Gpu_x,
		double * Gpu_y, double * Gpu_z, double * Gpu_COMX, double * Gpu_COMY,
		double * Gpu_COMZ,

		double xAxis, double yAxis, double zAxis, uint *molKindIndex,
		double * sigmaSq, double * epsilon_cn, double * nOver6,
		double * epsilon_cn_6, double beta, double AcceptRand, uint * Gpu_start,
		int len, double xHalfAxis, double yHalfAxis, double zHalfAxis,
		uint boxOffset, uint MoleculeCount,
		uint mIndex,			// mol index with offset
		uint FFParticleKindCount, double rCut, uint molKInd, double rCutSq,

		double dev_EnergyContrib[], double dev_VirialContrib[], uint boxIndex,
		bool * Gpu_result, double *dev_calp, double *dev_particleCharge,		double *dev_RealEnergyContrib, XYZ *dev_newCOM, double *dev_Array,
#ifdef MIE_INT_ONLY
		const uint * n,
#else
		const double * n
#endif

		)

		{
	extern __shared__ double Array[];
	int threadId = blockIdx.x * blockDim.x + threadIdx.x;
	int MolId = threadId + boxOffset;

	if (threadIdx.x < len) {
		uint start;
		start = Gpu_start[mIndex];
		//int i=0;
		XYZ center;

		dev_newCOM[0].x = center.x = Gpu_COMX[mIndex];
		dev_newCOM[0].y = center.y = Gpu_COMY[mIndex];
		dev_newCOM[0].z = center.z = Gpu_COMZ[mIndex];

		Array[threadIdx.x] = Gpu_x[start + threadIdx.x];
		Array[len + threadIdx.x] = Gpu_y[start + threadIdx.x];
		Array[2 * len + threadIdx.x] = Gpu_z[start + threadIdx.x];

		UnwrapPBC(Array[threadIdx.x], center.x, xAxis, xHalfAxis);
		UnwrapPBC(Array[len + threadIdx.x], center.y, yAxis, yHalfAxis);
		UnwrapPBC(Array[2 * len + threadIdx.x], center.z, zAxis, zHalfAxis);

		Array[threadIdx.x] += (-1.0 * center.x);
		Array[len + threadIdx.x] += (-1.0 * center.y);
		Array[2 * len + threadIdx.x] += (-1.0 * center.z);

		matrix.Apply(Array[threadIdx.x], Array[len + threadIdx.x],
				Array[2 * len + threadIdx.x]);

		Array[threadIdx.x] += (center.x);
		Array[len + threadIdx.x] += (center.y);
		Array[2 * len + threadIdx.x] += (center.z);

		WrapPBC(Array[threadIdx.x], xAxis);
		WrapPBC(Array[len + threadIdx.x], yAxis);
		WrapPBC(Array[2 * len + threadIdx.x], zAxis);		dev_Array[threadIdx.x] = Array[threadIdx.x];		dev_Array[len + threadIdx.x] = Array[len + threadIdx.x];		dev_Array[2 * len + threadIdx.x] = Array[2 * len + threadIdx.x];

	}

	__syncthreads();
	Gpu_CalculateMoleculeInter(NoOfAtomsPerMol, AtomKinds, molKindIndex,
			sigmaSq, epsilon_cn, nOver6, epsilon_cn_6, Gpu_x, Gpu_y, Gpu_z,
			Gpu_COMX, Gpu_COMY, Gpu_COMZ, Gpu_COMX[mIndex], Gpu_COMY[mIndex],
			Gpu_COMZ[mIndex], Array, len, MolId, Gpu_start, xAxis, yAxis, zAxis,
			xHalfAxis, yHalfAxis, zHalfAxis, boxOffset, MoleculeCount,
			mIndex,			// mol index with offset
			FFParticleKindCount, rCut, molKInd, rCutSq, dev_EnergyContrib,
			dev_VirialContrib, dev_calp, dev_particleCharge, dev_RealEnergyContrib,			boxIndex, n);

}

__device__ bool InRcutGpuSigned(double &distSq, double *x, double * y,
		double *z, double xAxis, double yAxis, double zAxis, double xHalfAxis,
		double yHalfAxis, double zHalfAxis, const uint i, const uint j,
		double rCut, double rCutSq, XYZ & dist) {
	distSq = 0;
	dist.x = x[i] - x[j];
	dist.x = MinImageSignedGpu(dist.x, xAxis, xHalfAxis);
	dist.y = y[i] - y[j];
	dist.y = MinImageSignedGpu(dist.y, yAxis, yHalfAxis);
	dist.z = z[i] - z[j];
	dist.z = MinImageSignedGpu(dist.z, zAxis, zHalfAxis);
	distSq = dist.x * dist.x + dist.y * dist.y + dist.z * dist.z;
	return (rCutSq > distSq);
}

__device__ void CalcAddGpu(double& en, double& vir, const double distSq,
		const uint kind1, const uint kind2, uint count, double * sigmaSq,
		double * epsilon_cn, double * nOver6, double * epsilon_cn_6,
#ifdef MIE_INT_ONLY
		const uint * n,
#else
		const double * n
#endif

		) {
	uint index = kind1 + kind2 * count;
	double rNeg2 = 1.0 / distSq;
	double rRat2 = rNeg2 * sigmaSq[index];
	double rRat4 = rRat2 * rRat2;
	double attract = rRat4 * rRat2;
#ifdef MIE_INT_ONLY
	double repulse = pow(rRat2, rRat4, attract, n);
#else
	double repulse = pow(__dsqrt_rn(rRat2), n[index]);
#endif

	en += epsilon_cn[index] * (repulse - attract);
	//Virial is the derivative of the pressure... mu
	vir = epsilon_cn_6[index] * (nOver6[index] * repulse - attract) * rNeg2;

}

__device__ double MinImageSignedGpu(double raw, double ax, double halfAx) {
	if (raw > halfAx)
		raw -= ax;
	else if (raw < -halfAx)
		raw += ax;
	return raw;
}

__global__ void ScaleMolecules(uint * noOfAtomsPerMol, uint *molKindIndex,
		double * Gpu_x, double * Gpu_y, double * Gpu_z, double * Gpu_COMX,
		double * Gpu_COMY, double * Gpu_COMZ, double * Gpu_newx,
		double * Gpu_newy, double * Gpu_newz, double * Gpu_newCOMX,
		double * Gpu_newCOMY, double * Gpu_newCOMZ, double scale, int MolCount,
		double newxAxis, double newyAxis, double newzAxis, double xAxis,
		double yAxis, double zAxis, double xHalfAxis, double yHalfAxis,
		double zHalfAxis, uint boxOffset, uint * Gpu_start)

		{
	int threadId = blockIdx.x * blockDim.x + threadIdx.x;
	int MolId = threadId + boxOffset; // ID of molecule to do calculations with for this thread

	if (threadId < MolCount) {
		double shiftX, shiftY, shiftZ;
		uint start;
		int len;
		len = noOfAtomsPerMol[MolId];
		start = Gpu_start[MolId];
		Gpu_newCOMX[MolId] = Gpu_COMX[MolId] * scale;
		Gpu_newCOMY[MolId] = Gpu_COMY[MolId] * scale;
		Gpu_newCOMZ[MolId] = Gpu_COMZ[MolId] * scale;

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

			UnwrapPBC(Gpu_newx[start + i], Gpu_COMX[MolId], xAxis, xHalfAxis);
			UnwrapPBC(Gpu_newy[start + i], Gpu_COMY[MolId], yAxis, yHalfAxis);
			UnwrapPBC(Gpu_newz[start + i], Gpu_COMZ[MolId], zAxis, zHalfAxis);
			Gpu_newx[start + i] += shiftX;
			Gpu_newy[start + i] += shiftY;
			Gpu_newz[start + i] += shiftZ;
			WrapPBC(Gpu_newx[start + i], newxAxis);
			WrapPBC(Gpu_newy[start + i], newyAxis);
			WrapPBC(Gpu_newz[start + i], newzAxis);

		}
	}
}

SystemPotential CalculateEnergy::NewSystemInterGPU(uint step,
		BoxDimensions &newDim, uint src, uint dist) {

	SystemPotential pot;
	double densityRatio;
	cudaMemcpy(&pot, Gpu_Potential, sizeof(SystemPotential),
			cudaMemcpyDeviceToHost);
	// do src box
	uint offset;
	offset = (src == 0) ? 0 : AtomCount[0];
	uint MolOffset;
	MolOffset = (src == 0) ? 0 : MolCount[0];
	int ThreadsPerBlock1 = 0;
	int BlocksPerGrid1 = 0;
	ThreadsPerBlock1 = MAXTHREADSPERBLOCK;

	if (ThreadsPerBlock1 == 0) {
		ThreadsPerBlock1 = 1;
	}

	BlocksPerGrid1 = ((MolCount[src] * (MolCount[src] - 1) / 2)
			+ ThreadsPerBlock1 - 1) / ThreadsPerBlock1;

	if (BlocksPerGrid1 == 0) {
		BlocksPerGrid1 = 1;
	}

	int XBlocks, YBlocks;

	if (BlocksPerGrid1 < MAXTBLOCKS) {
		XBlocks = 1;
		YBlocks = BlocksPerGrid1;
	} else {
		YBlocks = MAXTBLOCKS;
		XBlocks = (int) ceil((double) BlocksPerGrid1 / (double) MAXTBLOCKS);
	}

	double * dev_EnergyContrib, *dev_VirialContrib, *dev_RealEnergyContrib;
	cudaMalloc((void**) &dev_EnergyContrib,
			4 * XBlocks * YBlocks * sizeof(double));
	cudaMalloc((void**) &dev_VirialContrib,
			4 * XBlocks * YBlocks * sizeof(double));	cudaMalloc((void**) &dev_RealEnergyContrib,			4 * XBlocks * YBlocks * sizeof(double));
	double FinalEnergyNVirial[2], RealEnergy[2];
	FinalEnergyNVirial[0] = 0.0;
	FinalEnergyNVirial[1] = 0.0;	RealEnergy[0] = RealEnergy[1] = 0.0;
	double NewEnergy1 = 0.0;
	double NewVirial1 = 0.0;	double RealEnergy1 = 0.0;
	dim3 blcks(XBlocks, YBlocks, 1);
	dim3 thrds(ThreadsPerBlock1, 1, 1);

	if (MolCount[src] != 0) {

		Gpu_CalculateSystemInter<<<blcks, thrds, 0, stream0>>>(step,
				NoOfAtomsPerMol, Gpu_atomKinds, Gpu_kIndex, Gpu_sigmaSq,
				Gpu_epsilon_cn, Gpu_nOver6, Gpu_epsilon_cn_6, newX, newY, newZ,
				newCOMX, newCOMY, newCOMZ, Gpu_start, newDim.axis.x[src],
				newDim.axis.y[src], newDim.axis.z[src], newDim.halfAx.x[src],
				newDim.halfAx.y[src], newDim.halfAx.z[src], offset,
				MolCount[src], MolCount[src] / 2,
				forcefield.particles->NumKinds(), newDim.rCut,
				(MolCount[src] % 2 == 0) ? 1 : 0, newDim.rCutSq,
				dev_EnergyContrib, dev_VirialContrib,
				(MolCount[src] * (MolCount[src] - 1) / 2), MolOffset,				dev_particleCharge, dev_calp, dev_DoEwald, dev_alpha,				dev_Kmax1, dev_RealEnergyContrib, Gpu_partn);

		cudaError_t code = cudaGetLastError();

		if (code != cudaSuccess) {
			printf(
					"Cuda error at volume move energy for box 0-- %s, LINE: %d\n",
					cudaGetErrorString(code), __LINE__);
			exit(2);
		}

		cudaMemcpyAsync(&FinalEnergyNVirial, dev_EnergyContrib,
				2 * sizeof(double), cudaMemcpyDeviceToHost, stream0);		cudaMemcpyAsync(&RealEnergy, dev_RealEnergyContrib,						2 * sizeof(double), cudaMemcpyDeviceToHost, stream0);
		cudaStreamSynchronize(stream0);
		NewEnergy1 = FinalEnergyNVirial[0];

		NewVirial1 = FinalEnergyNVirial[1];
		RealEnergy1 = RealEnergy[0];
	}

	densityRatio = currentAxes.volume[src] * newDim.volInv[src];

	pot.boxEnergy[src].inter = NewEnergy1;
	pot.boxEnergy[src].tc *= densityRatio;
	pot.boxVirial[src].inter = NewVirial1;
	pot.boxVirial[src].tc *= densityRatio;	pot.boxEnergy[src].real = RealEnergy1;
	offset = (dist == 0) ? 0 : AtomCount[0];
	MolOffset = (dist == 0) ? 0 : MolCount[0];

	ThreadsPerBlock1 = MAXTHREADSPERBLOCK;

	if (ThreadsPerBlock1 == 0) {
		ThreadsPerBlock1 = 1;
	}

	BlocksPerGrid1 = ((MolCount[dist] * (MolCount[dist] - 1) / 2)
			+ ThreadsPerBlock1 - 1) / ThreadsPerBlock1;

	if (BlocksPerGrid1 == 0) {
		BlocksPerGrid1 = 1;
	}

	if (BlocksPerGrid1 < MAXTBLOCKS) {
		XBlocks = BlocksPerGrid1;
		YBlocks = 1;
	} else {
		YBlocks = MAXTBLOCKS;
		XBlocks = (int) ceil((double) BlocksPerGrid1 / (double) MAXTBLOCKS);
	}

	double * dev_EnergyContrib1, *dev_VirialContrib1, *dev_RealEnergyContrib1;
	cudaMalloc((void**) &dev_EnergyContrib1,
			4 * XBlocks * YBlocks * sizeof(double));
	cudaMalloc((void**) &dev_VirialContrib1,
			4 * XBlocks * YBlocks * sizeof(double));	cudaMalloc((void**) &dev_RealEnergyContrib1,			4 * XBlocks * YBlocks * sizeof(double));
	FinalEnergyNVirial[0] = 0.0;
	FinalEnergyNVirial[1] = 0.0;	RealEnergy[0] = RealEnergy[1] = 0.0;
	double NewEnergy2 = 0.0;
	double NewVirial2 = 0.0;	double RealEnergy2 = 0.0;
	dim3 blcks1(XBlocks, YBlocks, 1);
	dim3 thrds1(ThreadsPerBlock1, 1, 1);

	if (MolCount[dist] != 0) {

		Gpu_CalculateSystemInter<<<blcks1, thrds1, 0, stream1>>>(step,
				NoOfAtomsPerMol, Gpu_atomKinds, Gpu_kIndex, Gpu_sigmaSq,
				Gpu_epsilon_cn, Gpu_nOver6, Gpu_epsilon_cn_6, newX, newY, newZ,
				newCOMX, newCOMY, newCOMZ, Gpu_start, newDim.axis.x[dist],
				newDim.axis.y[dist], newDim.axis.z[dist], newDim.halfAx.x[dist],
				newDim.halfAx.y[dist], newDim.halfAx.z[dist], offset,
				MolCount[dist], MolCount[dist] / 2,
				forcefield.particles->NumKinds(), newDim.rCut,
				(MolCount[dist] % 2 == 0) ? 1 : 0, newDim.rCutSq,
				dev_EnergyContrib1, dev_VirialContrib1,
				(MolCount[dist] * (MolCount[dist] - 1) / 2), MolOffset,
				dev_particleCharge, dev_calp, dev_DoEwald, dev_alpha,				dev_Kmax1, dev_RealEnergyContrib1, Gpu_partn);
		cudaMemcpyAsync(&FinalEnergyNVirial, dev_EnergyContrib1,
				2 * sizeof(double), cudaMemcpyDeviceToHost, stream1);		cudaMemcpyAsync(&RealEnergy, dev_RealEnergyContrib1,				2 * sizeof(double), cudaMemcpyDeviceToHost, stream1);
		cudaStreamSynchronize(stream1);
		NewEnergy2 = FinalEnergyNVirial[0];

		NewVirial2 = FinalEnergyNVirial[1];
		RealEnergy2 = RealEnergy[0];
	}

	densityRatio = currentAxes.volume[dist] * newDim.volInv[dist];

	pot.boxEnergy[dist].inter = NewEnergy2;
	pot.boxEnergy[dist].tc *= densityRatio;
	pot.boxVirial[dist].inter = NewVirial2;
	pot.boxVirial[dist].tc *= densityRatio;	pot.boxEnergy[dist].real = RealEnergy2;
	pot.Total();
	cudaFree(dev_EnergyContrib);
	cudaFree(dev_VirialContrib);	cudaFree(dev_RealEnergyContrib);
	cudaFree(dev_EnergyContrib1);
	cudaFree(dev_VirialContrib1);	cudaFree(dev_RealEnergyContrib1);
	return pot;
}

SystemPotential CalculateEnergy::NewSystemInterGPUOneBox(BoxDimensions &newDim,
		uint bPick) {
	SystemPotential pot;
	double densityRatio;
	cudaMemcpy(&pot, Gpu_Potential, sizeof(SystemPotential),
			cudaMemcpyDeviceToHost);
	// do src box
	uint offset;
	offset = (bPick == 0) ? 0 : AtomCount[0];
	uint MolOffset;
	MolOffset = (bPick == 0) ? 0 : MolCount[0];
	int ThreadsPerBlock1 = 0;
	int BlocksPerGrid1 = 0;
	ThreadsPerBlock1 = MAXTHREADSPERBLOCK;

	if (ThreadsPerBlock1 == 0) {
		ThreadsPerBlock1 = 1;
	}

	BlocksPerGrid1 = ((MolCount[bPick] * (MolCount[bPick] - 1) / 2)
			+ ThreadsPerBlock1 - 1) / ThreadsPerBlock1;

	if (BlocksPerGrid1 == 0) {
		BlocksPerGrid1 = 1;
	}

	int XBlocks, YBlocks;

	if (BlocksPerGrid1 < MAXTBLOCKS) {
		XBlocks = 1;
		YBlocks = BlocksPerGrid1;
	} else {
		YBlocks = MAXTBLOCKS;
		XBlocks = (int) ceil((double) BlocksPerGrid1 / (double) MAXTBLOCKS);
	}

	double * dev_EnergyContrib, *dev_VirialContrib, *dev_RealEnergyContrib;
	cudaMalloc((void**) &dev_EnergyContrib,
			4 * XBlocks * YBlocks * sizeof(double));
	cudaMalloc((void**) &dev_VirialContrib,
			4 * XBlocks * YBlocks * sizeof(double));	cudaMalloc((void**) &dev_RealEnergyContrib,			4 * XBlocks * YBlocks * sizeof(double));
	double FinalEnergyNVirial[2], RealEnergy[2];
	FinalEnergyNVirial[0] = 0.0;
	FinalEnergyNVirial[1] = 0.0;	RealEnergy[0] = RealEnergy[1] = 0.0;
	double NewEnergy1 = 0.0;
	double NewVirial1 = 0.0;	double RealEnergy1 = 0.0;
	dim3 blcks(XBlocks, YBlocks, 1);
	dim3 thrds(ThreadsPerBlock1, 1, 1);

	if (MolCount[bPick] != 0) {

		Gpu_CalculateSystemInter<<<blcks, thrds, 0, stream0>>>(0,
				NoOfAtomsPerMol, Gpu_atomKinds, Gpu_kIndex, Gpu_sigmaSq,
				Gpu_epsilon_cn, Gpu_nOver6, Gpu_epsilon_cn_6, newX, newY, newZ,
				newCOMX, newCOMY, newCOMZ, Gpu_start, newDim.axis.x[bPick],
				newDim.axis.y[bPick], newDim.axis.z[bPick],
				newDim.halfAx.x[bPick], newDim.halfAx.y[bPick],
				newDim.halfAx.z[bPick], offset, MolCount[bPick],
				MolCount[bPick] / 2, forcefield.particles->NumKinds(),
				newDim.rCut, (MolCount[bPick] % 2 == 0) ? 1 : 0, newDim.rCutSq,
				dev_EnergyContrib, dev_VirialContrib,
				(MolCount[bPick] * (MolCount[bPick] - 1) / 2), MolOffset,				dev_particleCharge, dev_calp, dev_DoEwald, dev_alpha, dev_Kmax1,				dev_RealEnergyContrib, Gpu_partn);
		cudaMemcpyAsync(&FinalEnergyNVirial, dev_EnergyContrib,
				2 * sizeof(double), cudaMemcpyDeviceToHost, stream0);		cudaMemcpyAsync(&RealEnergy, dev_RealEnergyContrib,				2 * sizeof(double), cudaMemcpyDeviceToHost, stream0);
		cudaStreamSynchronize(stream0);
		NewEnergy1 = FinalEnergyNVirial[0];
		NewVirial1 = FinalEnergyNVirial[1];		RealEnergy1 = RealEnergy[0];
	}

	densityRatio = currentAxes.volume[bPick] * newDim.volInv[bPick];
	pot.boxEnergy[bPick].inter = NewEnergy1;
	pot.boxEnergy[bPick].tc *= densityRatio;
	pot.boxVirial[bPick].inter = NewVirial1;
	pot.boxVirial[bPick].tc *= densityRatio;	pot.boxEnergy[bPick].real = RealEnergy1;
	pot.Total();
	cudaFree(dev_EnergyContrib);
	cudaFree(dev_VirialContrib);	cudaFree(dev_RealEnergyContrib);
	return pot;
}

__device__ bool InRcutGpu(double &distSq, double *x, double * y, double *z,
		double nX, double nY, double nZ, double xAxis, double yAxis,
		double zAxis, double xHalfAxis, double yHalfAxis, double zHalfAxis,
		const uint i, double rCut, double rCutSq) {
	distSq = 0;
	double dX = nX - x[i];
	dX = MinImageSignedGpu(dX, xAxis, xHalfAxis);
	double dY = nY - y[i];
	dY = MinImageSignedGpu(dY, yAxis, yHalfAxis);
	double dZ = nZ - z[i];
	dZ = MinImageSignedGpu(dZ, zAxis, zHalfAxis);
	distSq = dX * dX + dY * dY + dZ * dZ;
	return (rCutSq > distSq);
}

__device__ void CalcAddGpuNoVirial(double& en, const double distSq,
		const uint kind1, const uint kind2, uint count, double * sigmaSq,
		double * epsilon_cn, double * nOver6, double * epsilon_cn_6,
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

__global__ void Gpu_CalculateParticleInter(int trial, uint *molKindIndex,
		double * sigmaSq, double * epsilon_cn, double * nOver6,
		double * epsilon_cn_6, double *Gpu_x, double *Gpu_y, double *Gpu_z,
		double nX, double nY, double nZ, uint * Gpu_atomKinds,
		int len, // mol length
		uint MolId, // mol ID of the particle we are testing now
		uint AtomToProcess, uint * Gpu_start, double xAxis, double yAxis,
		double zAxis, double xHalfAxis, double yHalfAxis, double zHalfAxis,
		uint boxOffset, uint AtomCount, // atom count in the current box
		uint FFParticleKindCount, double rCut, uint molKInd, // mol kind of the tested atom
		double rCutSq, double dev_EnergyContrib[],

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

	if (threadID < AtomCount && (AtomID < start || AtomID >= start + len)) { // skip atoms inside the same molecule
		double energy = 0.0;
		double distSq = 0.0;

		if (InRcutGpu(distSq, Gpu_x, Gpu_y, Gpu_z, nX, nY, nZ, xAxis, yAxis,
				zAxis, xHalfAxis, yHalfAxis, zHalfAxis, AtomID, rCut, rCutSq)) {

			CalcAddGpuNoVirial(energy, distSq,
					Gpu_atomKinds[start + AtomToProcess], Gpu_atomKinds[AtomID],
					FFParticleKindCount, sigmaSq, epsilon_cn, nOver6,
					epsilon_cn_6, n);

		}

		cacheEnergy[threadIdx.x] += energy;

	}

	__syncthreads();
	// add data
	int offset = 1 << (int) __log2f((float) blockDim.x);

	if (blockDim.x < MAXTHREADSPERBLOCK) {
		if ((threadIdx.x + offset) < AtomCount) {
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

		if (LastBlock) {
			BlockNum = blockIdx.x;
		}
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

__device__ int GetMinimumCell(int CellCoord, const int CellsPerDimension) {
	if (CellCoord >= CellsPerDimension)
		CellCoord -= CellsPerDimension;
	else if (CellCoord < 0)
		CellCoord += CellsPerDimension;
	return CellCoord;
}

__device__ void CalcAddGpu(double& en, double& vir, const double distSq,
		const uint kind1, const uint kind2, const uint count,
		const double * sigmaSq, const double * epsilon_cn,
		const double * nOver6, const double * epsilon_cn_6,
#ifdef MIE_INT_ONLY
		const uint * n,
#else
		const double * n
#endif

		) {
	uint index = kind1 + kind2 * count;
	double rNeg2 = 1.0 / distSq;
	double rRat2 = rNeg2 * sigmaSq[index];
	double rRat4 = rRat2 * rRat2;
	double attract = rRat4 * rRat2;
#ifdef MIE_INT_ONLY
	double repulse = pow(rRat2, rRat4, attract, n);
#else
	double repulse = pow(__dsqrt_rn(rRat2), n[index]);
#endif

	en += epsilon_cn[index] * (repulse - attract);
	//Virial is the derivative of the pressure... mu
	vir = epsilon_cn_6[index] * (nOver6[index] * repulse - attract) * rNeg2;

}

__device__ void CalcAddGpuNoVirCell(double& en,

const double distSq, const uint kind1, const uint kind2, const uint count,
		const double * sigmaSq, const double * epsilon_cn,
		const double * nOver6, const double * epsilon_cn_6,
#ifdef MIE_INT_ONLY
		const uint * n,
#else
		const double * n
#endif

		) {
	uint index = kind1 + kind2 * count;
	double rNeg2 = 1.0 / distSq;
	double rRat2 = rNeg2 * sigmaSq[index];
	double rRat4 = rRat2 * rRat2;
	double attract = rRat4 * rRat2;
#ifdef MIE_INT_ONLY
	double repulse = pow(rRat2, rRat4, attract, n);
#else
	double repulse = pow(__dsqrt_rn(rRat2), n[index]);
#endif

	en += epsilon_cn[index] * (repulse - attract);

}

__device__ void CalculateParticleValuesFast(const int atomToProcess,
		const double ParticleXPosition, const double ParticleYPosition,
		const double ParticleZPosition, double* __restrict__ x,
		double* __restrict__ y, double* __restrict__ z, const double xAxis,
		const double yAxis, const double zAxis, const double xHalfAxis,
		const double yHalfAxis, const double zHalfAxis,
		const double EdgeXAdjust, const double EdgeYAdjust,
		const double EdgeZAdjust, const int CellsXDim, const int CellsYDim,
		const int CellsZDim, const int CellsPerXDimension,
		const int CellsPerYDimension, const int CellsPerZDimension,
		const double rCut, const double rCutSq, const uint cellOffset,
		uint cellrangeOffset,
		const uint NumberofCells, // number of cells in the search box (cellsXDim * cellsYDim* cellsZDim )
		const unsigned int* __restrict__ dev_ParticleCntrs,
		const unsigned int* __restrict__ dev_ParticleNums,
		const uint NumberOfCellsInBox, const uint * atomsMoleculeNo,
		const uint *AtomKinds, const double * sigmaSq,
		const double * epsilon_cn, const double * nOver6,
		const double * epsilon_cn_6, const uint FFParticleKindCount,
		const uint atomToProcessKind, double &deltaEnergy,

#ifdef MIE_INT_ONLY
		const uint * n,
#else
		const double * n
#endif
		) {

	//Local variables

	int CellNumber, ParticleIndex, ArrayLocation, xcoord, ycoord, zcoord;

	//If there aren't many cells, then we just skip processing of threads that aren't in a valid cell.
	if (threadIdx.x < CellsXDim && threadIdx.y < CellsYDim
			&& threadIdx.z < CellsZDim) {

		xcoord = threadIdx.x;
		ycoord = threadIdx.y;
		zcoord = threadIdx.z;

		double energy = 0.0, virial = 0.0;
		double distSq = 0.0;
		double partVirial = 0.0;
		XYZ virComponents;

		//Figure out which cell this thread is accessing.
		//Need to use the floor function so that, for instance, -2.3 becomes -3.0, not -2.0.
		if (ParticleXPosition < rCut)
			xcoord = GetMinimumCell(
					((int) floor((ParticleXPosition - rCut - EdgeXAdjust))
							>> HALF_MICROCELL_DIM) + xcoord,
					CellsPerXDimension);
		else
			xcoord = GetMinimumCell(
					(((int) (ParticleXPosition - rCut) >> HALF_MICROCELL_DIM))
							+ xcoord, CellsPerXDimension);

		if (ParticleYPosition < rCut)
			ycoord = GetMinimumCell(
					((int) floor((ParticleYPosition - rCut - EdgeYAdjust))
							>> HALF_MICROCELL_DIM) + ycoord,
					CellsPerYDimension);
		else
			ycoord = GetMinimumCell(
					(((int) (ParticleYPosition - rCut) >> HALF_MICROCELL_DIM))
							+ ycoord, CellsPerYDimension);

		if (ParticleZPosition < rCut)
			zcoord = GetMinimumCell(
					((int) floor((ParticleZPosition - rCut - EdgeZAdjust))
							>> HALF_MICROCELL_DIM) + zcoord,
					CellsPerZDimension);
		else
			zcoord = GetMinimumCell(
					(((int) (ParticleZPosition - rCut) >> HALF_MICROCELL_DIM))
							+ zcoord, CellsPerZDimension);

		CellNumber = (zcoord * CellsPerZDimension + ycoord) * CellsPerYDimension
				+ xcoord;

		ArrayLocation = 0;
		for (int j = 0; j < dev_ParticleCntrs[CellNumber + cellOffset]; j++) {
			ParticleIndex = dev_ParticleNums[ArrayLocation + CellNumber
					+ (cellrangeOffset)];

			if (atomsMoleculeNo[atomToProcess] != atomsMoleculeNo[ParticleIndex]
					&& atomToProcess > ParticleIndex) {

				if (InRcutGpuSigned(distSq, x, y, z, ParticleXPosition,
						ParticleYPosition, ParticleZPosition, xAxis, yAxis,
						zAxis, xHalfAxis, yHalfAxis, zHalfAxis, atomToProcess,
						ParticleIndex, rCut, rCutSq, virComponents)) {

					CalcAddGpu(energy, partVirial, distSq, atomToProcessKind,
							AtomKinds[ParticleIndex], FFParticleKindCount,
							sigmaSq, epsilon_cn, nOver6, epsilon_cn_6, n);

				}

			}		//End check if this is the selected particle (if statement).
			ArrayLocation += NumberOfCellsInBox;
		}		//End for particles in cell

		__syncthreads();

		deltaEnergy = energy;

	}		//End for if (a valid cell location)
}

__device__ void CalculateParticleValues(
		//int centerCell, 
		//int *dev_AdjacencyCellList,
		const int atomToProcess, const int ParticleNumber,
		const double ParticleXPosition, const double ParticleYPosition,
		const double ParticleZPosition, double* __restrict__ x,
		double* __restrict__ y, double* __restrict__ z, const double xAxis,
		const double yAxis, const double zAxis, const double xHalfAxis,
		const double yHalfAxis, const double zHalfAxis,
		const double EdgeXAdjust, const double EdgeYAdjust,
		const double EdgeZAdjust, const int CellsXDim, const int CellsYDim,
		const int CellsZDim, const int CellsPerXDimension,
		const int CellsPerYDimension, const int CellsPerZDimension,
		const double rCut, const double rCutSq, const uint cellOffset,
		const uint cellrangeOffset, // cellOffset*MAX_ATOMS_PER_CELL
		const uint NumberofCells, // number of cells in the search box (cellsXDim * cellsYDim* cellsZDim )
		const unsigned int* dev_ParticleCntrs,
		const unsigned int* dev_ParticleNums, const uint NumberOfCellsInBox,
		const uint * atomsMoleculeNo, const uint *AtomKinds,
		const double * sigmaSq, const double * epsilon_cn,
		const double * nOver6, const double * epsilon_cn_6,
		const uint FFParticleKindCount, const uint atomToProcessKind,
		const uint atomToProcessMoleculeNo, double &deltaEnergy,
#ifdef MIE_INT_ONLY
		const uint * n,
#else
		const double * n
#endif
		) {

	//Local variables
	double RadialDistanceSquared, AttractiveTerm, RepulsiveTerm;
	double SeparationOnXAxis, SeparationOnYAxis, SeparationOnZAxis;
	int CellNumber, ParticleIndex, ArrayLocation, xcoord, ycoord, zcoord;

	double energy = 0.0, virial = 0.0;
	double distSq = 0.0;
	double partVirial = 0.0;
	XYZ virComponents;

	for (int i = ParticleNumber; i < NumberofCells; i += BLOCK_SIZE) {

		xcoord = i % CellsXDim;
		ycoord = (i / CellsYDim) % CellsYDim; // check 
		zcoord = i / (CellsYDim * CellsZDim); // check 

		if (ParticleXPosition < rCut)
			xcoord = GetMinimumCell(
					((int) floor((ParticleXPosition - rCut - EdgeXAdjust))
							>> HALF_MICROCELL_DIM) + xcoord,
					CellsPerXDimension);
		else
			xcoord = GetMinimumCell(
					(((int) (ParticleXPosition - rCut) >> HALF_MICROCELL_DIM))
							+ xcoord, CellsPerXDimension);

		if (ParticleYPosition < rCut)
			ycoord = GetMinimumCell(
					((int) floor((ParticleYPosition - rCut - EdgeYAdjust))
							>> HALF_MICROCELL_DIM) + ycoord,
					CellsPerYDimension);
		else
			ycoord = GetMinimumCell(
					(((int) (ParticleYPosition - rCut) >> HALF_MICROCELL_DIM))
							+ ycoord, CellsPerYDimension);

		if (ParticleZPosition < rCut)
			zcoord = GetMinimumCell(
					((int) floor((ParticleZPosition - rCut - EdgeZAdjust))
							>> HALF_MICROCELL_DIM) + zcoord,
					CellsPerZDimension);
		else
			zcoord = GetMinimumCell(
					(((int) (ParticleZPosition - rCut) >> HALF_MICROCELL_DIM))
							+ zcoord, CellsPerZDimension);

		CellNumber = (zcoord * CellsPerZDimension + ycoord) * CellsPerYDimension
				+ xcoord;

		ArrayLocation = 0;
		for (int j = 0; j < dev_ParticleCntrs[CellNumber + cellOffset]; j++) {
			ParticleIndex = dev_ParticleNums[ArrayLocation + CellNumber
					+ (cellrangeOffset)];
			if (atomsMoleculeNo[atomToProcess] != atomsMoleculeNo[ParticleIndex]
					&& atomToProcess > ParticleIndex) {
				if (InRcutGpuSigned(distSq, x, y, z, ParticleXPosition,
						ParticleYPosition, ParticleZPosition, xAxis, yAxis,
						zAxis, xHalfAxis, yHalfAxis, zHalfAxis, atomToProcess,
						ParticleIndex, rCut, rCutSq, virComponents)) {

					CalcAddGpu(energy, partVirial, distSq, atomToProcessKind,
							AtomKinds[ParticleIndex], FFParticleKindCount,
							sigmaSq, epsilon_cn, nOver6, epsilon_cn_6, n);

				}

			} //End check if this is the selected particle (if statement).
			ArrayLocation += NumberOfCellsInBox;
		} //End for particles in cell

		__syncthreads();

		deltaEnergy = energy;
	} //End for i
}

__global__ void CalculateSystemInter_CellList(
		//	int *dev_AdjacencyCellList,
		double* __restrict__ x, double* __restrict__ y, double* __restrict__ z,
		const double xAxis, const double yAxis, const double zAxis,
		const double xHalfAxis, const double yHalfAxis, const double zHalfAxis,
		const double EdgeXAdjust, const double EdgeYAdjust,
		const double EdgeZAdjust, const int CellsXDim, const int CellsYDim,
		const int CellsZDim, const int CellsPerXDimension,
		const int CellsPerYDimension, const int CellsPerZDimension,
		const double rCut, const double rCutSq, const uint boxOffset,
		const uint cellOffset, const uint cellrangeOffset,
		const uint NumberofCells, // number of cells in the search box (cellsXDim * cellsYDim* cellsZDim )
		const uint* atomCountrs, const uint* atomCells,
		const uint NumberOfCellsInBox, const uint * atomsMoleculeNo,
		const uint *AtomKinds, const double * sigmaSq,
		const double * epsilon_cn, const double * nOver6,
		const double * epsilon_cn_6, const uint FFParticleKindCount,
		double dev_EnergyContrib[], const int NoOfAtomsToProcess,
		double * EnergyPair, // remove later 
#ifdef MIE_INT_ONLY
		const uint * n,
#else
		const double * n
#endif

		) {

	__shared__ double cacheEnergy[BLOCK_SIZE]; // should be max number of cells this block will process ()

	__shared__ bool LastBlock;
	int atomToProcess;
	double deltaEnergy = 0.0, deltaVirial = 0.0;
	int flatThreadid;
	flatThreadid = (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x
			+ threadIdx.x;
	cacheEnergy[flatThreadid] = 0.0;
	for (int i = 0; i < NoOfAtomsToProcess; i++) {

		atomToProcess = i * gridDim.x + boxOffset + blockIdx.x;

		deltaEnergy = 0.0, deltaVirial = 0.0;

		//Calcualte the energy and virial for the current particle location.
		if (CellsXDim <= BLOCK_DIM) // need to check 

		{

			CalculateParticleValuesFast(atomToProcess, x[atomToProcess],
					y[atomToProcess], z[atomToProcess], x, y, z, xAxis, yAxis,
					zAxis, xHalfAxis, yHalfAxis, zHalfAxis, EdgeXAdjust,
					EdgeYAdjust, EdgeZAdjust, CellsXDim, CellsYDim, CellsZDim,
					CellsPerXDimension, CellsPerYDimension, CellsPerZDimension,
					rCut, rCutSq, cellOffset, cellrangeOffset, NumberofCells,
					atomCountrs, atomCells, NumberOfCellsInBox, atomsMoleculeNo,
					AtomKinds, sigmaSq, epsilon_cn, nOver6, epsilon_cn_6,
					FFParticleKindCount, AtomKinds[atomToProcess], deltaEnergy,

					n);
		}

		else {
			CalculateParticleValues(atomToProcess, flatThreadid,
					x[atomToProcess], y[atomToProcess], z[atomToProcess], x, y,
					z, xAxis, yAxis, zAxis, xHalfAxis, yHalfAxis, zHalfAxis,
					EdgeXAdjust, EdgeYAdjust, EdgeZAdjust, CellsXDim, CellsYDim,
					CellsZDim, CellsPerXDimension, CellsPerYDimension,
					CellsPerZDimension, rCut, rCutSq, cellOffset,
					cellrangeOffset, NumberofCells, atomCountrs, atomCells,
					NumberOfCellsInBox, atomsMoleculeNo, AtomKinds, sigmaSq,
					epsilon_cn, nOver6, epsilon_cn_6, FFParticleKindCount,
					AtomKinds[atomToProcess], atomsMoleculeNo[atomToProcess],
					deltaEnergy, n);
		}

		cacheEnergy[flatThreadid] += deltaEnergy;

	}

	__syncthreads();

	// add data
	//This code assumes we always have an 8x8x8 = 512 thread block.
	int offset = 0;
	if (flatThreadid < 256) {
		cacheEnergy[flatThreadid] += cacheEnergy[flatThreadid + 256];

	}
	__syncthreads();

	if (flatThreadid < 128) {
		cacheEnergy[flatThreadid] += cacheEnergy[flatThreadid + 128];

	}
	__syncthreads();

	if (flatThreadid < 64) {
		cacheEnergy[flatThreadid] += cacheEnergy[flatThreadid + 64];

	}
	__syncthreads();

	if (flatThreadid < 32) {
		cacheEnergy[flatThreadid] += cacheEnergy[flatThreadid + 32];

		__threadfence_block();
		cacheEnergy[flatThreadid] += cacheEnergy[flatThreadid + 16];

		__threadfence_block();
		cacheEnergy[flatThreadid] += cacheEnergy[flatThreadid + 8];

		__threadfence_block();
		cacheEnergy[flatThreadid] += cacheEnergy[flatThreadid + 4];

		__threadfence_block();
		cacheEnergy[flatThreadid] += cacheEnergy[flatThreadid + 2];

		__threadfence_block();
		cacheEnergy[flatThreadid] += cacheEnergy[flatThreadid + 1];

	}	//end if

	__syncthreads();

	if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
		dev_EnergyContrib[blockIdx.x] = cacheEnergy[0];
		if (cacheEnergy[0] != 0)

			__threadfence();
		LastBlock = (atomicInc(&BlocksDone, gridDim.x) == (gridDim.x) - 1);
	}       //end if

	__syncthreads();

	if (LastBlock) {
		//If this thread corresponds to a valid block
		if (flatThreadid < gridDim.x) {
			cacheEnergy[flatThreadid] = dev_EnergyContrib[flatThreadid];

		}

		for (int i = BLOCK_DIM; flatThreadid + i < gridDim.x; i += BLOCK_DIM) {
			if (flatThreadid + i < gridDim.x) {
				cacheEnergy[flatThreadid] +=
						dev_EnergyContrib[flatThreadid + i];

			}       //end if
		}       //end for

		__syncthreads();
		offset = 1 << (int) __log2f((float) min(BLOCK_DIM, gridDim.x));

		for (int i = offset; i > 0; i >>= 1) {
			if (flatThreadid < i
					&& flatThreadid + i < min(BLOCK_DIM, gridDim.x)) {
				cacheEnergy[flatThreadid] += cacheEnergy[flatThreadid + i];

			}       //end if

			__syncthreads();
		}       //end for

		if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
			BlocksDone = 0;
			dev_EnergyContrib[0] = cacheEnergy[0];

		}
	}
}

SystemPotential CalculateEnergy::SystemInterGPU_CellList() {

	/*double * dev_EnergyContrib, * dev_VirialContrib;
	 cudaMalloc((void**) &dev_EnergyContrib,  BLOCK_SIZE * sizeof(double));
	 cudaMalloc((void**) &dev_VirialContrib,  BLOCK_SIZE * sizeof(double));*/

	CalculateSystemInter_CellList<<<AtomCount[0] / NumberofOps, BlockSize>>>(

	Gpu_x, Gpu_y, Gpu_z, currentAxes.axis.x[0], currentAxes.axis.y[0],
			currentAxes.axis.z[0], currentAxes.halfAx.x[0],
			currentAxes.halfAx.y[0], currentAxes.halfAx.z[0], EdgeAdjust[0],
			EdgeAdjust[1], EdgeAdjust[2], CellDim[0], CellDim[1], CellDim[2],
			CellsPerDim[0], CellsPerDim[1], CellsPerDim[2], currentAxes.rCut,
			currentAxes.rCutSq, 0, 0, 0, CellDim[0] * CellDim[1] * CellDim[2],
			atomCountrs, atomCells, TotalCellsPerBox[0], atomsMoleculeNo,
			Gpu_atomKinds, Gpu_sigmaSq, Gpu_epsilon_cn, Gpu_nOver6,
			Gpu_epsilon_cn_6, forcefield.particles->NumKinds(),
			dev_EnergyContrib,

			NumberofOps, PairEnergy, Gpu_partn);

	double FinalEnergyNVirial[2];
	cudaMemcpy(&FinalEnergyNVirial, dev_EnergyContrib, 2 * sizeof(double),
			cudaMemcpyDeviceToHost);

	double NewEnergy1 = FinalEnergyNVirial[0];

#if ENSEMBLE == GEMC

	/*double * dev_EnergyContrib1, * dev_VirialContrib1;
	 cudaMalloc((void**) &dev_EnergyContrib1,  BLOCK_SIZE * sizeof(double));
	 cudaMalloc((void**) &dev_VirialContrib1,  BLOCK_SIZE * sizeof(double));*/

	CalculateSystemInter_CellList<<<AtomCount[1] / NumberofOps, BlockSize>>>(

	Gpu_x, Gpu_y, Gpu_z, currentAxes.axis.x[1], currentAxes.axis.y[1],
			currentAxes.axis.z[1], currentAxes.halfAx.x[1],
			currentAxes.halfAx.y[1], currentAxes.halfAx.z[1], EdgeAdjust[3],
			EdgeAdjust[4], EdgeAdjust[5], CellDim[3], CellDim[4], CellDim[5],
			CellsPerDim[3], CellsPerDim[4], CellsPerDim[5], currentAxes.rCut,
			currentAxes.rCutSq, AtomCount[0], TotalCellsPerBox[0],
			TotalCellsPerBox[0] * MAX_ATOMS_PER_CELL,
			CellDim[3] * CellDim[4] * CellDim[5], atomCountrs, atomCells,
			TotalCellsPerBox[1], atomsMoleculeNo, Gpu_atomKinds, Gpu_sigmaSq,
			Gpu_epsilon_cn, Gpu_nOver6, Gpu_epsilon_cn_6,
			forcefield.particles->NumKinds(), dev_EnergyContrib,

			NumberofOps, PairEnergy, Gpu_partn);

	cudaDeviceSynchronize();

	cudaMemcpy(&FinalEnergyNVirial, dev_EnergyContrib, 2 * sizeof(double),
			cudaMemcpyDeviceToHost);

	double NewEnergy2 = FinalEnergyNVirial[0];

#endif

	SystemPotential currentpot;
	double densityRatio = currentAxes.volume[0] * currentAxes.volInv[0];
	currentpot.boxEnergy[0].inter = NewEnergy1;
	currentpot.boxEnergy[0].tc *= densityRatio;

	currentpot.boxVirial[0].tc *= densityRatio;
	/*cudaFree( dev_EnergyContrib);
	 cudaFree(dev_VirialContrib);*/

#if ENSEMBLE == GEMC
	densityRatio = currentAxes.volume[1] * currentAxes.volInv[1];
	currentpot.boxEnergy[1].inter = NewEnergy2;
	currentpot.boxEnergy[1].tc *= densityRatio;

	currentpot.boxVirial[1].tc *= densityRatio;
	/*cudaFree( dev_EnergyContrib1);
	 cudaFree(dev_VirialContrib1);*/
#endif

	currentpot.Total();

	cudaDeviceSynchronize();
	cudaError_t code = cudaGetLastError();

	if (code != cudaSuccess) {
		printf("Cuda error at end of Calculate Total Energy -- %s\n",
				cudaGetErrorString(code));
		exit(2);
	}

	return currentpot;

}

// conv cell list
__device__ bool InRcutGpuSignedNoVir(double &distSq, double *x, double *y,
		double *z, const double xi, const double yi, const double zi,
		double xAxis, double yAxis, double zAxis, double xHalfAxis,
		double yHalfAxis, double zHalfAxis,

		const uint j, double rCut, double rCutSq) {
	XYZ dist;
	distSq = 0;
	dist.x = xi - x[j];
	dist.x = MinImageSignedGpu(dist.x, xAxis, xHalfAxis);
	dist.y = yi - y[j];
	dist.y = MinImageSignedGpu(dist.y, yAxis, yHalfAxis);
	dist.z = zi - z[j];
	dist.z = MinImageSignedGpu(dist.z, zAxis, zHalfAxis);
	distSq = dist.x * dist.x + dist.y * dist.y + dist.z * dist.z;
	return (rCutSq > distSq);
}

__global__ void CalculateTotalCellEnergy

(double* __restrict__ x, double* __restrict__ y, double* __restrict__ z,
		double * Gpu_COMX, double * Gpu_COMY, double * Gpu_COMZ,
		const double xAxis, const double yAxis, const double zAxis,
		const double xHalfAxis, const double yHalfAxis, const double zHalfAxis,
		const uint * atomsMoleculeNo, const uint *AtomKinds,
		const double * sigmaSq, const double * epsilon_cn,
		const double * nOver6, const double * epsilon_cn_6, const double rCut,
		const double rCutSq, const uint boxOffset,
		const uint FFParticleKindCount, double dev_EnergyContribCELL_LIST[],
		double dev_VirialContribCELL_LIST[], double dev_particleCharge[], double dev_calp[],		bool *dev_DoEwald, double *dev_alpha, double *dev_Kmax1, double dev_RealEnergyContribCELL_LIST[],		const int * dev_AdjacencyCellList, const uint * dev_CountAtomsInCell, const int * AtomsInCells,
		//double * ENR, 
		uint step,
#ifdef MIE_INT_ONLY
		const uint * n,
#else
		const double * n
#endif

		) {

	//Shared variables
	__shared__
	double cacheEnergy[MaxParticleInCell];
	__shared__
	double cacheVirial[MaxParticleInCell];
	__shared__ double cacheRealEnergy[MaxParticleInCell];

	__shared__ double centerX[MaxParticleInCell];
	__shared__ double centerY[MaxParticleInCell];
	__shared__ double centerZ[MaxParticleInCell];
	__shared__ int cachedAtomKinds[MaxParticleInCell];

	__shared__ uint cachedCenterAtoms[MaxParticleInCell];
	__shared__ bool LastBlock;

	//Local variables

	uint ParticleNumber, CurrentCell;

	double energy = 0.0, virial = 0.0, realEnergy = 0.0;
	double distSq = 0.0;
	double partVirial = 0.0;
	double erfc_variable = 0.0;
	XYZ virComponents;
	XYZ temp;
	XYZ fMolO;

	uint threadId = blockIdx.x * blockDim.x + threadIdx.x;

	uint CenterCell = blockIdx.x / 27;
	uint CellIndex = blockIdx.x % 27;

	CurrentCell = dev_AdjacencyCellList[CellIndex + CenterCell * 27];

	cacheEnergy[threadIdx.x] = 0.0;
	cacheVirial[threadIdx.x] = 0.0;

	if (threadIdx.x < dev_CountAtomsInCell[CenterCell]) {

		cachedCenterAtoms[threadIdx.x] = AtomsInCells[threadIdx.x
				+ CenterCell * MaxParticleInCell] + boxOffset;

		centerX[threadIdx.x] = x[cachedCenterAtoms[threadIdx.x]];
		centerY[threadIdx.x] = y[cachedCenterAtoms[threadIdx.x]];
		centerZ[threadIdx.x] = z[cachedCenterAtoms[threadIdx.x]];
		cachedAtomKinds[threadIdx.x] =
				AtomKinds[cachedCenterAtoms[threadIdx.x]];
	}

	__syncthreads();

	if (dev_CountAtomsInCell[CurrentCell] > 0
			&& threadIdx.x < dev_CountAtomsInCell[CurrentCell]) {

		ParticleNumber = AtomsInCells[threadIdx.x
				+ CurrentCell * MaxParticleInCell] + boxOffset;

		for (int i = 0; i < dev_CountAtomsInCell[CenterCell]; i++) {

			if (cachedCenterAtoms[i] > ParticleNumber
					&& atomsMoleculeNo[cachedCenterAtoms[i]]
							!= atomsMoleculeNo[ParticleNumber]) {

				if (InRcutGpuSigned(distSq, centerX, centerY, centerZ,
						x[ParticleNumber], y[ParticleNumber], z[ParticleNumber],
						xAxis, yAxis, zAxis, xHalfAxis, yHalfAxis, zHalfAxis,
						ParticleNumber, i, rCut, rCutSq, virComponents)) {

					CalcAddGpu(energy, partVirial, distSq,
							AtomKinds[ParticleNumber], cachedAtomKinds[i],
							FFParticleKindCount, sigmaSq, epsilon_cn, nOver6,
							epsilon_cn_6, n);

					fMolO = virComponents * partVirial;
					temp.x = MinImageSignedGpu(
							Gpu_COMX[atomsMoleculeNo[cachedCenterAtoms[i]]]
									- Gpu_COMX[atomsMoleculeNo[ParticleNumber]],
							xAxis, xHalfAxis);
					temp.y = MinImageSignedGpu(
							Gpu_COMY[atomsMoleculeNo[cachedCenterAtoms[i]]]
									- Gpu_COMY[atomsMoleculeNo[ParticleNumber]],
							yAxis, yHalfAxis);
					temp.z = MinImageSignedGpu(
							Gpu_COMZ[atomsMoleculeNo[cachedCenterAtoms[i]]]
									- Gpu_COMZ[atomsMoleculeNo[ParticleNumber]],
							zAxis, zHalfAxis);
					virial += geom::Dot(fMolO, temp);

#ifdef REAL					if(dev_DoEwald[0])
						CalcRealGpu(distSq, dev_calp, realEnergy, dev_particleCharge,
							cachedCenterAtoms[i], ParticleNumber, (boxOffset > 0) ? 0:1);
#endif

				}
			}
		}

	}

	cacheEnergy[threadIdx.x] = energy;
	cacheVirial[threadIdx.x] = virial;
	cacheRealEnergy[threadIdx.x] = realEnergy;
	__syncthreads();

	//First reduce to make sure that we have an exact power of two threads left, then use a loop.
	//We only need to do this if there is only one block. We assume the MAXTHREADSPERBLOCK is an
	//exact power of two.

	int offset = 1 << (int) __log2f((float) blockDim.x);

	//if (blockDim.x < MAXTHREADSPERBLOCK)
	{
		if ((threadIdx.x + offset) < blockDim.x) {
			cacheEnergy[threadIdx.x] += cacheEnergy[threadIdx.x + offset];
			cacheVirial[threadIdx.x] += cacheVirial[threadIdx.x + offset];
			cacheRealEnergy[threadIdx.x] += cacheRealEnergy[threadIdx.x + offset];
		}
		__syncthreads();
	}

	for (int i = offset >> 1; i > 32; i >>= 1) {
		if (threadIdx.x < i) {

			cacheEnergy[threadIdx.x] += cacheEnergy[threadIdx.x + i];
			cacheVirial[threadIdx.x] += cacheVirial[threadIdx.x + i];
			cacheRealEnergy[threadIdx.x] += cacheRealEnergy[threadIdx.x + i];
		}		//end if
		__syncthreads();

	}		//end for

	if (threadIdx.x < offset / 2 && threadIdx.x < 32) {
		__shared__ double *cEnergy, *cVirial, *cRealEnergy;
		cEnergy = cacheEnergy;
		cVirial = cacheVirial;
		cRealEnergy = cacheRealEnergy;
		offset = min(offset, 64);
		switch (offset) {

		case 64:
			cEnergy[threadIdx.x] += cEnergy[threadIdx.x + 32];
			cVirial[threadIdx.x] += cVirial[threadIdx.x + 32];
			cRealEnergy[threadIdx.x] += cRealEnergy[threadIdx.x + 32];
		case 32:
			cEnergy[threadIdx.x] += cEnergy[threadIdx.x + 16];
			cVirial[threadIdx.x] += cVirial[threadIdx.x + 16];
			cRealEnergy[threadIdx.x] += cRealEnergy[threadIdx.x + 16];
		case 16:
			cEnergy[threadIdx.x] += cEnergy[threadIdx.x + 8];
			cVirial[threadIdx.x] += cVirial[threadIdx.x + 8];
			cRealEnergy[threadIdx.x] += cRealEnergy[threadIdx.x + 8];
		case 8:
			cEnergy[threadIdx.x] += cEnergy[threadIdx.x + 4];
			cVirial[threadIdx.x] += cVirial[threadIdx.x + 4];
			cRealEnergy[threadIdx.x] += cRealEnergy[threadIdx.x + 5];
		case 4:

			cEnergy[threadIdx.x] += cEnergy[threadIdx.x + 2];
			cVirial[threadIdx.x] += cVirial[threadIdx.x + 2];
			cRealEnergy[threadIdx.x] += cRealEnergy[threadIdx.x + 2];
		case 2:
			cEnergy[threadIdx.x] += cEnergy[threadIdx.x + 1];
			cVirial[threadIdx.x] += cVirial[threadIdx.x + 1];
			cRealEnergy[threadIdx.x] += cRealEnergy[threadIdx.x + 1];
		}		//End Switch
	}		//end if

	if (threadIdx.x == 0) {

		dev_EnergyContribCELL_LIST[blockIdx.x] = cacheEnergy[0];
		dev_VirialContribCELL_LIST[blockIdx.x] = cacheVirial[0];
		dev_RealEnergyContribCELL_LIST[blockIdx.x] = cacheRealEnergy[0];
		__threadfence();
		LastBlock = (atomicInc(&BlocksDone, gridDim.x) == gridDim.x - 1);
		if (LastBlock)
			BlockNum = blockIdx.x;
	}		//end if
	__syncthreads();

	if (LastBlock) {
		//If this thread corresponds to a valid block
		if (threadIdx.x < gridDim.x) {
			cacheEnergy[threadIdx.x] = dev_EnergyContribCELL_LIST[threadIdx.x];
			cacheVirial[threadIdx.x] = dev_VirialContribCELL_LIST[threadIdx.x];
			cacheEnergy[threadIdx.x] = dev_RealEnergyContribCELL_LIST[threadIdx.x];
		}
		for (int i = blockDim.x; threadIdx.x + i < gridDim.x; i += blockDim.x) {
			if (threadIdx.x + i < gridDim.x) {
				cacheEnergy[threadIdx.x] +=
						dev_EnergyContribCELL_LIST[threadIdx.x + i];
				cacheVirial[threadIdx.x] +=
						dev_VirialContribCELL_LIST[threadIdx.x + i];
				cacheRealEnergy[threadIdx.x] += dev_RealEnergyContribCELL_LIST[threadIdx.x+i];
			}		//end if
		}		//end for
		__syncthreads();

		offset = 1 << (int) __log2f((float) min(blockDim.x, gridDim.x));

		for (int i = offset; i > 0; i >>= 1) {
			if (threadIdx.x < i
					&& threadIdx.x + i < min(blockDim.x, gridDim.x)) {
				cacheEnergy[threadIdx.x] += cacheEnergy[threadIdx.x + i];
				cacheVirial[threadIdx.x] += cacheVirial[threadIdx.x + i];
				cacheRealEnergy[threadIdx.x] += cacheRealEnergy[threadIdx.x + i];
			}		//end if
			__syncthreads();
		}		//end for

		if (threadIdx.x == 0) {
			dev_EnergyContribCELL_LIST[0] = cacheEnergy[0];
			dev_EnergyContribCELL_LIST[1] = cacheVirial[0];
			dev_VirialContribCELL_LIST[0] = cacheVirial[0];
			dev_RealEnergyContribCELL_LIST[0] = cacheRealEnergy[0];
			BlocksDone = 0;

		}
	}		//end if LastBlock

}
/*
SystemPotential CalculateEnergy::CalculateNewEnergyCellListOneBox(
		BoxDimensions &newDim, int step, int bPick) {

	SystemPotential currentpot;
	cudaMemcpy(&currentpot, Gpu_Potential, sizeof(SystemPotential),
			cudaMemcpyDeviceToHost);
	double FinalEnergy[2];

	FinalEnergy[0] = FinalEnergy[1] = 0.0;

	int NofCellsCube = NumberOfCells[bPick] * NumberOfCells[bPick]
			* NumberOfCells[bPick];
	cudaMalloc((void**) &dev_EnergyContribCELL_LIST,
			27 * NofCellsCube * sizeof(double));
	cudaMalloc((void**) &dev_VirialContribCELL_LIST,
			27 * NofCellsCube * sizeof(double));

	CalculateTotalCellEnergy<<<27 * NofCellsCube, MaxParticleInCell>>>

	(newX, newY, newZ, newCOMX, newCOMY, newCOMZ, newDim.axis.x[bPick],
			newDim.axis.y[bPick], newDim.axis.z[bPick], newDim.halfAx.x[bPick],
			newDim.halfAx.y[bPick], newDim.halfAx.z[bPick], atomsMoleculeNo,
			Gpu_atomKinds, Gpu_sigmaSq, Gpu_epsilon_cn, Gpu_nOver6,
			Gpu_epsilon_cn_6, currentAxes.rCut, currentAxes.rCutSq,
			(bPick == 0) ? 0 : AtomCount[0], forcefield.particles->NumKinds(),
			dev_EnergyContribCELL_LIST, dev_VirialContribCELL_LIST, dev_particleCharge,			dev_calp, dev_DoEwald, dev_alpha, dev_Kmax1, dev_RealEnergyContribCELL_LIST,
			(bPick == 0) ? dev_AdjacencyCellList0 : dev_AdjacencyCellList1,
			(bPick == 0) ? dev_CountAtomsInCell0 : dev_CountAtomsInCell1,
			(bPick == 0) ? AtomsInCells0 : AtomsInCells1,
			//ENR,
			step, Gpu_partn

			);
	cudaMemcpy(&FinalEnergy, dev_EnergyContribCELL_LIST, 2 * sizeof(double),
			cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

	double NewEnergy1 = FinalEnergy[0];
	double NewVirial1 = FinalEnergy[1];

	//SystemPotential currentpot
	//double densityRatio = currentAxes.volume[0] * currentAxes.volInv[0];
	double densityRatio = currentAxes.volume[bPick] * newDim.volInv[bPick];
	//printf("densityRatio=%f, tc =%f\n", densityRatio, currentpot.boxEnergy[0].tc);
	currentpot.boxEnergy[bPick].inter = NewEnergy1;
	currentpot.boxEnergy[bPick].tc *= densityRatio;
	currentpot.boxVirial[bPick].inter = NewVirial1;
	currentpot.boxVirial[bPick].tc *= densityRatio;

	if (bPick == 0) {
		cudaFree(dev_AdjacencyCellList0);
		cudaFree(dev_CountAtomsInCell0);
		cudaFree(AtomsInCells0);
	} else {
		cudaFree(dev_AdjacencyCellList1);
		cudaFree(dev_CountAtomsInCell1);
		cudaFree(AtomsInCells1);

	}

	currentpot.Total();

	cudaFree(dev_EnergyContribCELL_LIST);
	cudaFree(dev_VirialContribCELL_LIST);

	cudaDeviceSynchronize();
	cudaError_t code = cudaGetLastError();

	if (code != cudaSuccess) {
		printf("Cuda error at end of Calculate Total Energy at step %d -- %s\n",
				step, cudaGetErrorString(code));
		exit(2);
	}

	return currentpot;

}

SystemPotential CalculateEnergy::CalculateNewEnergyCellList(
		BoxDimensions &newDim, SystemPotential currentpot, int step) {

	double FinalEnergy[2];

	FinalEnergy[0] = FinalEnergy[1] = 0.0;

	int NofCellsCube = NumberOfCells[0] * NumberOfCells[0] * NumberOfCells[0];
	cudaMalloc((void**) &dev_EnergyContribCELL_LIST,
			27 * NofCellsCube * sizeof(double));
	cudaMalloc((void**) &dev_VirialContribCELL_LIST,
			27 * NofCellsCube * sizeof(double));

	CalculateTotalCellEnergy<<<27 * NofCellsCube, MaxParticleInCell>>>

	(newX, newY, newZ, newCOMX, newCOMY, newCOMZ, newDim.axis.x[0],
			newDim.axis.y[0], newDim.axis.z[0], newDim.halfAx.x[0],
			newDim.halfAx.y[0], newDim.halfAx.z[0], atomsMoleculeNo,
			Gpu_atomKinds, Gpu_sigmaSq, Gpu_epsilon_cn, Gpu_nOver6,
			Gpu_epsilon_cn_6, currentAxes.rCut, currentAxes.rCutSq, 0,
			forcefield.particles->NumKinds(), dev_EnergyContribCELL_LIST,			dev_VirialContribCELL_LIST, dev_particleCharge, dev_calp, dev_DoEwald, dev_alpha,			dev_Kmax1, dev_RealEnergyContribCELL_LIST, dev_AdjacencyCellList0,
			dev_CountAtomsInCell0, AtomsInCells0, step, Gpu_partn

			);
	cudaMemcpy(&FinalEnergy, dev_EnergyContribCELL_LIST, 2 * sizeof(double),
			cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

	double NewEnergy1 = FinalEnergy[0];
	double NewVirial1 = FinalEnergy[1];

	double NewEnergy2 = 0.0;
	double NewVirial2 = 0.0;

#if ENSEMBLE == GEMC
	cudaFree(dev_EnergyContribCELL_LIST);
	cudaFree(dev_VirialContribCELL_LIST);

	FinalEnergy[0] = FinalEnergy[1] = 0.0;
	NofCellsCube = NumberOfCells[1] * NumberOfCells[1] * NumberOfCells[1];
	cudaMalloc((void**) &dev_EnergyContribCELL_LIST,
			27 * NofCellsCube * sizeof(double));
	cudaMalloc((void**) &dev_VirialContribCELL_LIST,
			27 * NofCellsCube * sizeof(double));

	CalculateTotalCellEnergy<<<27 * NofCellsCube, MaxParticleInCell>>>

	(newX, newY, newZ, newCOMX, newCOMY, newCOMZ, newDim.axis.x[1],
			newDim.axis.y[1], newDim.axis.z[1], newDim.halfAx.x[1],
			newDim.halfAx.y[1], newDim.halfAx.z[1], atomsMoleculeNo,
			Gpu_atomKinds, Gpu_sigmaSq, Gpu_epsilon_cn, Gpu_nOver6,
			Gpu_epsilon_cn_6, currentAxes.rCut, currentAxes.rCutSq,
			AtomCount[0], forcefield.particles->NumKinds(),
			dev_EnergyContribCELL_LIST, dev_VirialContribCELL_LIST, dev_particleCharge,			dev_calp, dev_DoEwald, dev_alpha, dev_Kmax1, dev_RealEnergyContribCELL_LIST,
			dev_AdjacencyCellList1, dev_CountAtomsInCell1, AtomsInCells1, step,
			Gpu_partn

			);
	cudaDeviceSynchronize();
	cudaMemcpy(&FinalEnergy, dev_EnergyContribCELL_LIST, 2 * sizeof(double),
			cudaMemcpyDeviceToHost);

	NewEnergy2 = FinalEnergy[0];
	NewVirial2 = FinalEnergy[1];

#endif

	//SystemPotential currentpot
	//double densityRatio = currentAxes.volume[0] * currentAxes.volInv[0];
	double densityRatio = currentAxes.volume[0] * newDim.volInv[0];

	currentpot.boxEnergy[0].inter = NewEnergy1;
	currentpot.boxEnergy[0].tc *= densityRatio;
	currentpot.boxVirial[0].inter = NewVirial1;
	currentpot.boxVirial[0].tc *= densityRatio;
	cudaFree(dev_AdjacencyCellList0);
	cudaFree(dev_CountAtomsInCell0);
	cudaFree(AtomsInCells0);

#if ENSEMBLE == GEMC

	densityRatio = currentAxes.volume[1] * newDim.volInv[1];

	currentpot.boxEnergy[1].inter = NewEnergy2;
	currentpot.boxEnergy[1].tc *= densityRatio;
	currentpot.boxVirial[1].inter = NewVirial2;
	currentpot.boxVirial[1].tc *= densityRatio;
	cudaFree(dev_AdjacencyCellList1);
	cudaFree(dev_CountAtomsInCell1);
	cudaFree(AtomsInCells1);

#endif

	currentpot.Total();

	cudaFree(dev_EnergyContribCELL_LIST);
	cudaFree(dev_VirialContribCELL_LIST);

	cudaDeviceSynchronize();
	cudaError_t code = cudaGetLastError();

	if (code != cudaSuccess) {
		printf("Cuda error at end of Calculate Total Energy at step %d -- %s\n",
				step, cudaGetErrorString(code));
		exit(2);
	}

	return currentpot;

}
*/
// conv cell list total energy 
SystemPotential CalculateEnergy::CalculateEnergyCellList() {

	int NofCellsCube = NumberOfCells[0] * NumberOfCells[0] * NumberOfCells[0];
	double FinalEnergy[2];	double RealEnergy[2];	//ewald

	FinalEnergy[0] = FinalEnergy[1] = 0.0;	RealEnergy[0] = RealEnergy[1] = 0.0;	//ewald

	NofCellsCube = NumberOfCells[0] * NumberOfCells[0] * NumberOfCells[0];
	cudaMalloc((void**) &dev_EnergyContribCELL_LIST,
			27 * NofCellsCube * sizeof(double));
	cudaMalloc((void**) &dev_VirialContribCELL_LIST,
			27 * NofCellsCube * sizeof(double));	//ewald	cudaMalloc((void**) &dev_RealEnergyContribCELL_LIST, 27 * NofCellsCube * sizeof(double));

	CalculateTotalCellEnergy<<<27 * NofCellsCube, MaxParticleInCell>>>

	(Gpu_x, Gpu_y, Gpu_z, Gpu_COMX, Gpu_COMY, Gpu_COMZ, currentAxes.axis.x[0],
			currentAxes.axis.y[0], currentAxes.axis.z[0],
			currentAxes.halfAx.x[0], currentAxes.halfAx.y[0],
			currentAxes.halfAx.z[0], atomsMoleculeNo, Gpu_atomKinds,
			Gpu_sigmaSq, Gpu_epsilon_cn, Gpu_nOver6, Gpu_epsilon_cn_6,
			currentAxes.rCut, currentAxes.rCutSq, 0,
			forcefield.particles->NumKinds(), dev_EnergyContribCELL_LIST,
			dev_VirialContribCELL_LIST, dev_particleCharge, dev_calp, dev_DoEwald, dev_alpha,			dev_Kmax1, dev_RealEnergyContribCELL_LIST/*ewald*/,			dev_AdjacencyCellList0, dev_CountAtomsInCell0, AtomsInCells0, 0, Gpu_partn

			);
	cudaMemcpy(&FinalEnergy, dev_EnergyContribCELL_LIST, 2 * sizeof(double), cudaMemcpyDeviceToHost);	cudaMemcpy(RealEnergy, dev_RealEnergyContribCELL_LIST, 2 * sizeof(double), cudaMemcpyDeviceToHost);	//ewald
	cudaDeviceSynchronize();
	cudaError_t code1 = cudaGetLastError();

	if (code1 != cudaSuccess) {
		printf("Cuda error at end of Calculate Total Energyyyyyyyyy -- %s\n",
				cudaGetErrorString(code1));
		exit(2);
	}

	double NewEnergy1 = FinalEnergy[0];
	double NewVirial1 = FinalEnergy[1];
	double RealEnergy1 = RealEnergy[0];	//ewald
	double NewEnergy2 = 0.0;
	double NewVirial2 = 0.0;
	double RealEnergy2 = 0.0;	//ewald
#if ENSEMBLE == GEMC
	cudaFree(dev_EnergyContribCELL_LIST);
	cudaFree(dev_VirialContribCELL_LIST);	cudaFree(dev_RealEnergyContribCELL_LIST);	//ewald

	FinalEnergy[0] = FinalEnergy[1] = RealEnergy[0] = RealEnergy[1] = 0.0;	//ewald
	NofCellsCube = NumberOfCells[1] * NumberOfCells[1] * NumberOfCells[1];
	cudaMalloc((void**) &dev_EnergyContribCELL_LIST,
			27 * NofCellsCube * sizeof(double));
	cudaMalloc((void**) &dev_VirialContribCELL_LIST,
			27 * NofCellsCube * sizeof(double));
	cudaMalloc((void**) &dev_RealEnergyContribCELL_LIST,				27 * NofCellsCube * sizeof(double));		//ewald
	CalculateTotalCellEnergy<<<27 * NofCellsCube, MaxParticleInCell>>>
	(Gpu_x, Gpu_y, Gpu_z, Gpu_COMX, Gpu_COMY, Gpu_COMZ, currentAxes.axis.x[1],
			currentAxes.axis.y[1], currentAxes.axis.z[1],
			currentAxes.halfAx.x[1], currentAxes.halfAx.y[1],
			currentAxes.halfAx.z[1], atomsMoleculeNo, Gpu_atomKinds,
			Gpu_sigmaSq, Gpu_epsilon_cn, Gpu_nOver6, Gpu_epsilon_cn_6,
			currentAxes.rCut, currentAxes.rCutSq, AtomCount[0],
			forcefield.particles->NumKinds(), dev_EnergyContribCELL_LIST,			dev_VirialContribCELL_LIST, dev_particleCharge, dev_calp, dev_DoEwald, dev_alpha,			dev_Kmax1, dev_RealEnergyContribCELL_LIST, dev_AdjacencyCellList1,
			dev_CountAtomsInCell1, AtomsInCells1, 0, Gpu_partn

			);
	cudaDeviceSynchronize();
	cudaMemcpy(&FinalEnergy, dev_EnergyContribCELL_LIST, 2 * sizeof(double),
			cudaMemcpyDeviceToHost);
	cudaMemcpy(RealEnergy, dev_RealEnergyContribCELL_LIST, 2 * sizeof(double), cudaMemcpyDeviceToHost);	//ewald
	NewEnergy2 = FinalEnergy[0];
	NewVirial2 = FinalEnergy[1];
	RealEnergy2 = RealEnergy[0];	//ewald
#endif

	SystemPotential currentpot;
	double densityRatio = currentAxes.volume[0] * currentAxes.volInv[0];
	currentpot.boxEnergy[0].inter = NewEnergy1;
	currentpot.boxEnergy[0].tc *= densityRatio;
	currentpot.boxVirial[0].inter = NewVirial1;
	currentpot.boxVirial[0].tc *= densityRatio;	currentpot.boxEnergy[0].real = RealEnergy1 * qqfact[0];		//ewald	currentpot.boxEnergy[0].elect += currentpot.boxEnergy[0].real;

#if ENSEMBLE == GEMC
	densityRatio = currentAxes.volume[1] * currentAxes.volInv[1];
	currentpot.boxEnergy[1].inter = NewEnergy2;
	currentpot.boxEnergy[1].tc *= densityRatio;
	currentpot.boxVirial[1].inter = NewVirial2;
	currentpot.boxVirial[1].tc *= densityRatio;	currentpot.boxEnergy[1].real = RealEnergy2 * qqfact[0];		//ewald	currentpot.boxEnergy[1].elect += currentpot.boxEnergy[1].real;

#endif

	currentpot.Total();

	cudaFree(dev_EnergyContribCELL_LIST);
	cudaFree(dev_VirialContribCELL_LIST);	cudaFree(dev_RealEnergyContribCELL_LIST);

	cudaDeviceSynchronize();
	cudaError_t code = cudaGetLastError();

	if (code != cudaSuccess) {
		printf("Cuda error at end of Calculate Total Energy -- %s\n",
				cudaGetErrorString(code));
		exit(2);
	}

	return currentpot;

}

__device__ void CalculateParticleValuesFastMolInter(int start, int MolToProcess,
		const int atomToProcess, const double ParticleXPosition,
		const double ParticleYPosition, const double ParticleZPosition,
		double* __restrict__ x, double* __restrict__ y, double* __restrict__ z,
		double * Gpu_COMX, double * Gpu_COMY, double * Gpu_COMZ, double COMX,
		double COMY, double COMZ, const double xAxis, const double yAxis,
		const double zAxis, const double xHalfAxis, const double yHalfAxis,
		const double zHalfAxis, const double EdgeXAdjust,
		const double EdgeYAdjust, const double EdgeZAdjust, const int CellsXDim,
		const int CellsYDim, const int CellsZDim, const int CellsPerXDimension,
		const int CellsPerYDimension, const int CellsPerZDimension,
		const double rCut, const double rCutSq, const uint cellOffset,
		uint cellrangeOffset,
		const uint NumberofCells,// number of cells in the search box (cellsXDim * cellsYDim* cellsZDim )
		const unsigned int* __restrict__ dev_ParticleCntrs,
		const unsigned int* __restrict__ dev_ParticleNums,
		const uint NumberOfCellsInBox, const uint * atomsMoleculeNo,
		const uint *AtomKinds, const double * sigmaSq,
		const double * epsilon_cn, const double * nOver6,
		const double * epsilon_cn_6, const uint FFParticleKindCount,
		const uint atomToProcessKind, double &deltaEnergy, double &deltaVirial,
		double& deltaReal, int len, uint boxOffset, double dev_calp[],		double dev_particleCharge[], bool *dev_DoEwald,
#ifdef MIE_INT_ONLY
		const uint * n,
#else
		const double * n
#endif
		) {

	//Local variables
	//double RadialDistanceSquared, AttractiveTerm, RepulsiveTerm;
	//double SeparationOnXAxis, SeparationOnYAxis, SeparationOnZAxis;

	//If there aren't many cells, then we just skip processing of threads that aren't in a valid cell.
	if (threadIdx.x < CellsXDim && threadIdx.y < CellsYDim
			&& threadIdx.z < CellsZDim) {
		int CellNumber, ParticleIndex, ArrayLocation, xcoord, ycoord, zcoord;

		xcoord = threadIdx.x;
		ycoord = threadIdx.y;
		zcoord = threadIdx.z;

		double energy = 0.0, virial = 0.0, realEnergy = 0.0;
		double distSq = 0.0;
		double partVirial = 0.0;
		XYZ virComponents;
		XYZ fMolO, temp;

		int PartIndexMolNo;

		//Figure out which cell this thread is accessing.
		//Need to use the floor function so that, for instance, -2.3 becomes -3.0, not -2.0.
		if (ParticleXPosition < rCut)
			xcoord = GetMinimumCell(
					((int) floor((ParticleXPosition - rCut - EdgeXAdjust))
							>> HALF_MICROCELL_DIM) + xcoord,
					CellsPerXDimension);
		else
			xcoord = GetMinimumCell(
					(((int) (ParticleXPosition - rCut) >> HALF_MICROCELL_DIM))
							+ xcoord, CellsPerXDimension);

		if (ParticleYPosition < rCut)
			ycoord = GetMinimumCell(
					((int) floor((ParticleYPosition - rCut - EdgeYAdjust))
							>> HALF_MICROCELL_DIM) + ycoord,
					CellsPerYDimension);
		else
			ycoord = GetMinimumCell(
					(((int) (ParticleYPosition - rCut) >> HALF_MICROCELL_DIM))
							+ ycoord, CellsPerYDimension);

		if (ParticleZPosition < rCut)
			zcoord = GetMinimumCell(
					((int) floor((ParticleZPosition - rCut - EdgeZAdjust))
							>> HALF_MICROCELL_DIM) + zcoord,
					CellsPerZDimension);
		else
			zcoord = GetMinimumCell(
					(((int) (ParticleZPosition - rCut) >> HALF_MICROCELL_DIM))
							+ zcoord, CellsPerZDimension);

		CellNumber = (zcoord * CellsPerZDimension + ycoord) * CellsPerYDimension
				+ xcoord;

		ArrayLocation = 0;
		for (int j = 0; j < dev_ParticleCntrs[CellNumber + cellOffset]; j++) {
			ParticleIndex = dev_ParticleNums[ArrayLocation + CellNumber
					+ (cellrangeOffset)];

			PartIndexMolNo = atomsMoleculeNo[ParticleIndex];

			if (ParticleIndex < start || ParticleIndex >= start + len) {

				if (InRcutGpuSigned(distSq, x, y, z, ParticleXPosition,
						ParticleYPosition, ParticleZPosition, xAxis, yAxis,
						zAxis, xHalfAxis, yHalfAxis, zHalfAxis, atomToProcess,
						ParticleIndex, rCut, rCutSq, virComponents)) {

					CalcAddGpu(energy, partVirial, distSq, atomToProcessKind,
							AtomKinds[ParticleIndex], FFParticleKindCount,
							sigmaSq, epsilon_cn, nOver6, epsilon_cn_6, n);

					fMolO = virComponents * partVirial;
					temp.x = MinImageSignedGpu(COMX - Gpu_COMX[PartIndexMolNo],
							xAxis, xHalfAxis);
					temp.y = MinImageSignedGpu(COMY - Gpu_COMY[PartIndexMolNo],
							yAxis, yHalfAxis);
					temp.z = MinImageSignedGpu(COMZ - Gpu_COMZ[PartIndexMolNo],
							zAxis, zHalfAxis);
					virial -= geom::Dot(fMolO, temp);#ifdef REAL					if(dev_DoEwald[0]){					CalcRealGpu(distSq, dev_calp, realEnergy, dev_particleCharge,							atomToProcess, ParticleIndex, (boxOffset > 0) ? 0:1);					}#endif
				}

			}		//End check if this is the selected particle (if statement).
			ArrayLocation += NumberOfCellsInBox;
		}		//End for particles in cell

		__syncthreads();

		deltaEnergy = energy;
		deltaVirial = virial;		deltaReal = realEnergy;

	}		//End for if (a valid cell location)
}

__device__ void CalculateParticleValuesFastPartInter(int len, int MolToProcess,
		int MolStart, const int atomToProcess, const double ParticleXPosition,
		const double ParticleYPosition, const double ParticleZPosition,
		double* __restrict__ x, double* __restrict__ y, double* __restrict__ z,
		const double xAxis, const double yAxis, const double zAxis,
		const double xHalfAxis, const double yHalfAxis, const double zHalfAxis,
		const double EdgeXAdjust, const double EdgeYAdjust,
		const double EdgeZAdjust, const int CellsXDim, const int CellsYDim,
		const int CellsZDim, const int CellsPerXDimension,
		const int CellsPerYDimension, const int CellsPerZDimension,
		const double rCut, const double rCutSq, uint boxOffset,
		const uint cellOffset, uint cellrangeOffset,
		const uint NumberofCells,// number of cells in the search box (cellsXDim * cellsYDim* cellsZDim )
		const unsigned int* __restrict__ dev_ParticleCntrs,
		const unsigned int* __restrict__ dev_ParticleNums,
		const uint NumberOfCellsInBox, const uint * atomsMoleculeNo,
		const uint *AtomKinds, const double * sigmaSq,
		const double * epsilon_cn, const double * nOver6,
		const double * epsilon_cn_6, const uint FFParticleKindCount,
		const uint atomToProcessKind, double &deltaEnergy,
#ifdef MIE_INT_ONLY
		const uint * n,
#else
		const double * n
#endif
		) {

	//Local variables
	//double RadialDistanceSquared, AttractiveTerm, RepulsiveTerm;
	//double SeparationOnXAxis, SeparationOnYAxis, SeparationOnZAxis;
	int CellNumber, ParticleIndex, ArrayLocation, xcoord, ycoord, zcoord;

	//If there aren't many cells, then we just skip processing of threads that aren't in a valid cell.
	if (threadIdx.x < CellsXDim && threadIdx.y < CellsYDim
			&& threadIdx.z < CellsZDim) {

		xcoord = threadIdx.x;
		ycoord = threadIdx.y;
		zcoord = threadIdx.z;

		double energy = 0.0;
		double distSq = 0.0;

		//Figure out which cell this thread is accessing.
		//Need to use the floor function so that, for instance, -2.3 becomes -3.0, not -2.0.
		if (ParticleXPosition < rCut)
			xcoord = GetMinimumCell(
					((int) floor((ParticleXPosition - rCut - EdgeXAdjust))
							>> HALF_MICROCELL_DIM) + xcoord,
					CellsPerXDimension);
		else
			xcoord = GetMinimumCell(
					(((int) (ParticleXPosition - rCut) >> HALF_MICROCELL_DIM))
							+ xcoord, CellsPerXDimension);

		if (ParticleYPosition < rCut)
			ycoord = GetMinimumCell(
					((int) floor((ParticleYPosition - rCut - EdgeYAdjust))
							>> HALF_MICROCELL_DIM) + ycoord,
					CellsPerYDimension);
		else
			ycoord = GetMinimumCell(
					(((int) (ParticleYPosition - rCut) >> HALF_MICROCELL_DIM))
							+ ycoord, CellsPerYDimension);

		if (ParticleZPosition < rCut)
			zcoord = GetMinimumCell(
					((int) floor((ParticleZPosition - rCut - EdgeZAdjust))
							>> HALF_MICROCELL_DIM) + zcoord,
					CellsPerZDimension);
		else
			zcoord = GetMinimumCell(
					(((int) (ParticleZPosition - rCut) >> HALF_MICROCELL_DIM))
							+ zcoord, CellsPerZDimension);

		CellNumber = (zcoord * CellsPerZDimension + ycoord) * CellsPerYDimension
				+ xcoord;

		ArrayLocation = 0;
		for (int j = 0; j < dev_ParticleCntrs[CellNumber + cellOffset]; j++) {
			ParticleIndex = dev_ParticleNums[ArrayLocation + CellNumber
					+ (cellrangeOffset)];

			if (ParticleIndex < MolStart || ParticleIndex >= MolStart + len) {

				if (InRcutGpuSignedNoVir(distSq, x, y, z, ParticleXPosition,
						ParticleYPosition, ParticleZPosition, xAxis, yAxis,
						zAxis, xHalfAxis, yHalfAxis, zHalfAxis, atomToProcess,
						ParticleIndex, rCut, rCutSq)) {

					CalcAddGpuNoVirCell(energy, distSq, atomToProcessKind,
							AtomKinds[ParticleIndex], FFParticleKindCount,
							sigmaSq, epsilon_cn, nOver6, epsilon_cn_6, n);

				}

			}		//End check if this is the selected particle (if statement).
			ArrayLocation += NumberOfCellsInBox;
		}		//End for particles in cell

		__syncthreads();

		deltaEnergy = energy;

	}		//End for if (a valid cell location)
}

__device__ void CalculateParticleValuesPartInter(

int len, int MolStart, int MolToProcess, const int atomToProcess,
		const int ParticleNumber, const double ParticleXPosition,
		const double ParticleYPosition, const double ParticleZPosition,
		double* __restrict__ x, double* __restrict__ y, double* __restrict__ z,
		const double xAxis, const double yAxis, const double zAxis,
		const double xHalfAxis, const double yHalfAxis, const double zHalfAxis,
		const double EdgeXAdjust, const double EdgeYAdjust,
		const double EdgeZAdjust, const int CellsXDim, const int CellsYDim,
		const int CellsZDim, const int CellsPerXDimension,
		const int CellsPerYDimension, const int CellsPerZDimension,
		const double rCut, const double rCutSq, const uint cellOffset,
		const uint cellrangeOffset, // cellOffset*MAX_ATOMS_PER_CELL
		const uint NumberofCells, // number of cells in the search box (cellsXDim * cellsYDim* cellsZDim )
		const unsigned int* dev_ParticleCntrs,
		const unsigned int* dev_ParticleNums, const uint NumberOfCellsInBox,
		const uint * atomsMoleculeNo, const uint *AtomKinds,
		const double * sigmaSq, const double * epsilon_cn,
		const double * nOver6, const double * epsilon_cn_6,
		const uint FFParticleKindCount, const uint atomToProcessKind,
		const uint atomToProcessMoleculeNo, double &deltaEnergy,
#ifdef MIE_INT_ONLY
		const uint * n,
#else
		const double * n
#endif
		) {

	//Local variables
	double RadialDistanceSquared, AttractiveTerm, RepulsiveTerm;
	double SeparationOnXAxis, SeparationOnYAxis, SeparationOnZAxis;
	int CellNumber, ParticleIndex, ArrayLocation, xcoord, ycoord, zcoord;

	double energy = 0.0;
	double distSq = 0.0;

	for (int i = ParticleNumber; i < NumberofCells; i += BLOCK_SIZE) {
		//Calculate the x, y, and z coordindates of the cell for this block.
		xcoord = i % CellsXDim;
		ycoord = (i / CellsYDim) % CellsYDim;
		zcoord = i / (CellsYDim * CellsZDim);

		if (ParticleXPosition < rCut)
			xcoord = GetMinimumCell(
					((int) floor((ParticleXPosition - rCut - EdgeXAdjust))
							>> HALF_MICROCELL_DIM) + xcoord,
					CellsPerXDimension);
		else
			xcoord = GetMinimumCell(
					(((int) (ParticleXPosition - rCut) >> HALF_MICROCELL_DIM))
							+ xcoord, CellsPerXDimension);

		if (ParticleYPosition < rCut)
			ycoord = GetMinimumCell(
					((int) floor((ParticleYPosition - rCut - EdgeYAdjust))
							>> HALF_MICROCELL_DIM) + ycoord,
					CellsPerYDimension);
		else
			ycoord = GetMinimumCell(
					(((int) (ParticleYPosition - rCut) >> HALF_MICROCELL_DIM))
							+ ycoord, CellsPerYDimension);

		if (ParticleZPosition < rCut)
			zcoord = GetMinimumCell(
					((int) floor((ParticleZPosition - rCut - EdgeZAdjust))
							>> HALF_MICROCELL_DIM) + zcoord,
					CellsPerZDimension);
		else
			zcoord = GetMinimumCell(
					(((int) (ParticleZPosition - rCut) >> HALF_MICROCELL_DIM))
							+ zcoord, CellsPerZDimension);

		CellNumber = (zcoord * CellsPerZDimension + ycoord) * CellsPerYDimension
				+ xcoord;

		ArrayLocation = 0;
		for (int j = 0; j < dev_ParticleCntrs[CellNumber + cellOffset]; j++) {
			ParticleIndex = dev_ParticleNums[ArrayLocation + CellNumber
					+ (cellrangeOffset)];
			if (ParticleIndex < MolStart || ParticleIndex >= MolStart + len)

			{

				if (InRcutGpuSignedNoVir(distSq, x, y, z, ParticleXPosition,
						ParticleYPosition, ParticleZPosition, xAxis, yAxis,
						zAxis, xHalfAxis, yHalfAxis, zHalfAxis, atomToProcess,
						ParticleIndex, rCut, rCutSq)) {
					CalcAddGpuNoVirCell(energy, distSq, atomToProcessKind,
							AtomKinds[ParticleIndex], FFParticleKindCount,
							sigmaSq, epsilon_cn, nOver6, epsilon_cn_6, n);
				}

			} //End check if this is the selected particle (if statement).
			ArrayLocation += NumberOfCellsInBox;
		} //End for particles in cell

		__syncthreads();

	} //End for i
	deltaEnergy = energy;
}
__device__ void CalculateParticleValuesMolInter(

int MolStart, int len, const int MolToProcess, const int atomToProcess,
		const int ParticleNumber, const double ParticleXPosition,
		const double ParticleYPosition, const double ParticleZPosition,
		double* __restrict__ x, double* __restrict__ y, double* __restrict__ z,
		double * Gpu_COMX, double * Gpu_COMY, double * Gpu_COMZ, double COMX,
		double COMY, double COMZ, const double xAxis, const double yAxis,
		const double zAxis, const double xHalfAxis, const double yHalfAxis,
		const double zHalfAxis, const double EdgeXAdjust,
		const double EdgeYAdjust, const double EdgeZAdjust, const int CellsXDim,
		const int CellsYDim, const int CellsZDim, const int CellsPerXDimension,
		const int CellsPerYDimension, const int CellsPerZDimension,
		const double rCut, const double rCutSq, const uint cellOffset,
		const uint cellrangeOffset, // cellOffset*MAX_ATOMS_PER_CELL
		const uint NumberofCells, // number of cells in the search box (cellsXDim * cellsYDim* cellsZDim )
		const unsigned int* dev_ParticleCntrs,
		const unsigned int* dev_ParticleNums, const uint NumberOfCellsInBox,
		const uint * atomsMoleculeNo, const uint *AtomKinds,
		const double * sigmaSq, const double * epsilon_cn,
		const double * nOver6, const double * epsilon_cn_6,
		const uint FFParticleKindCount, const uint atomToProcessKind,
		const uint atomToProcessMoleculeNo, double &deltaEnergy,
		double &deltaVirial, uint boxOffset, double dev_calp[],		double dev_particleCharge[], bool *dev_DoEwald,
#ifdef MIE_INT_ONLY
		const uint * n,
#else
		const double * n
#endif
		) {

	//Local variables
	//double RadialDistanceSquared, AttractiveTerm, RepulsiveTerm;
	//double SeparationOnXAxis, SeparationOnYAxis, SeparationOnZAxis;
	int CellNumber, ParticleIndex, ArrayLocation, xcoord, ycoord, zcoord;

	double energy = 0.0, virial = 0.0, realEnergy = 0.0;
	double distSq = 0.0;
	double partVirial = 0.0;
	XYZ virComponents;
	XYZ temp, fMolO;
	int ParticleMolNo;

	for (int i = ParticleNumber; i < NumberofCells; i += BLOCK_SIZE) {
		//Calculate the x, y, and z coordindates of the cell for this block.
		xcoord = i % CellsXDim;
		ycoord = (i / CellsYDim) % CellsYDim;
		zcoord = i / (CellsYDim * CellsZDim);

		if (ParticleXPosition < rCut)
			xcoord = GetMinimumCell(
					((int) floor((ParticleXPosition - rCut - EdgeXAdjust))
							>> HALF_MICROCELL_DIM) + xcoord,
					CellsPerXDimension);
		else
			xcoord = GetMinimumCell(
					(((int) (ParticleXPosition - rCut) >> HALF_MICROCELL_DIM))
							+ xcoord, CellsPerXDimension);

		if (ParticleYPosition < rCut)
			ycoord = GetMinimumCell(
					((int) floor((ParticleYPosition - rCut - EdgeYAdjust))
							>> HALF_MICROCELL_DIM) + ycoord,
					CellsPerYDimension);
		else
			ycoord = GetMinimumCell(
					(((int) (ParticleYPosition - rCut) >> HALF_MICROCELL_DIM))
							+ ycoord, CellsPerYDimension);

		if (ParticleZPosition < rCut)
			zcoord = GetMinimumCell(
					((int) floor((ParticleZPosition - rCut - EdgeZAdjust))
							>> HALF_MICROCELL_DIM) + zcoord,
					CellsPerZDimension);
		else
			zcoord = GetMinimumCell(
					(((int) (ParticleZPosition - rCut) >> HALF_MICROCELL_DIM))
							+ zcoord, CellsPerZDimension);

		CellNumber = (zcoord * CellsPerZDimension + ycoord) * CellsPerYDimension
				+ xcoord;

		ArrayLocation = 0;
		for (int j = 0; j < dev_ParticleCntrs[CellNumber + cellOffset]; j++) {
			ParticleIndex = dev_ParticleNums[ArrayLocation + CellNumber
					+ (cellrangeOffset)];
			ParticleMolNo = atomsMoleculeNo[ParticleIndex];

			if (ParticleIndex < MolStart || ParticleIndex >= MolStart + len) {

				if (InRcutGpuSigned(distSq, x, y, z, ParticleXPosition,
						ParticleYPosition, ParticleZPosition, xAxis, yAxis,
						zAxis, xHalfAxis, yHalfAxis, zHalfAxis, atomToProcess,
						ParticleIndex, rCut, rCutSq, virComponents)) {

					CalcAddGpu(energy, partVirial, distSq, atomToProcessKind,
							AtomKinds[ParticleIndex], FFParticleKindCount,
							sigmaSq, epsilon_cn, nOver6, epsilon_cn_6, n);
					fMolO = virComponents * partVirial;
					temp.x = MinImageSignedGpu(COMX - Gpu_COMX[ParticleMolNo],
							xAxis, xHalfAxis);
					temp.y = MinImageSignedGpu(COMY - Gpu_COMY[ParticleMolNo],
							yAxis, yHalfAxis);
					temp.z = MinImageSignedGpu(COMZ - Gpu_COMZ[ParticleMolNo],
							zAxis, zHalfAxis);
					virial -= geom::Dot(fMolO, temp);
#ifdef REAL					if(dev_DoEwald[0]){						CalcRealGpu(distSq, dev_calp, realEnergy, dev_particleCharge,										atomToProcess, ParticleIndex, (boxOffset > 0) ? 0:1);					}#endif
				}

			}	//End check if this is the selected particle (if statement).
			ArrayLocation += NumberOfCellsInBox;
		}	//End for particles in cell

		__syncthreads();

	}	//End for i

	deltaEnergy = energy;
	deltaVirial = virial;
}

__device__ void CalculateMolInter_CellList(int start, int flatThreadid,
		double n_x, double n_y, double n_z, double* x, double* y, double* z,
		double * Gpu_COMX, double * Gpu_COMY, double * Gpu_COMZ, XYZ newCOM,
		XYZ oldCOM, double xAxis, double yAxis, double zAxis, double xHalfAxis,
		double yHalfAxis, double zHalfAxis, double EdgeXAdjust,
		double EdgeYAdjust, double EdgeZAdjust, int CellsXDim, int CellsYDim,
		int CellsZDim, int CellsPerXDimension, int CellsPerYDimension,
		int CellsPerZDimension, double rCut, double rCutSq, uint boxOffset,
		uint cellOffset, uint cellrangeOffset,
		uint NumberofCells,	// number of cells in the search box (cellsXDim * cellsYDim* cellsZDim )
		const uint* atomCountrs, const uint* atomCells, uint * Gpu_start,
		uint NumberOfCellsInBox, uint * atomsMoleculeNo, uint *AtomKinds,
		double * sigmaSq, double * epsilon_cn, double * nOver6,
		double * epsilon_cn_6, uint FFParticleKindCount,
		double dev_EnergyContrib[], double dev_VirialContrib[], double dev_RealEnergyContrib[],
		double dev_calp[], double dev_particleCharge[], bool *dev_DoEwald, double *dev_qqfact,		int MolToProcess, int atomToProcess, int len,
#ifdef MIE_INT_ONLY
		const uint * n,
#else
		const double * n
#endif

		) {

	__shared__ double cacheEnergy[BLOCK_SIZE];// should be max number of cells this block will process ()
	__shared__ double cacheVirial[BLOCK_SIZE];
	__shared__ double cacheRealEnergy[BLOCK_SIZE];//ewald
	__shared__ bool LastBlock;

	double deltaEnergy = 0.0, deltaVirial = 0.0, deltaReal = 0.0;

	cacheEnergy[flatThreadid] = 0.0;
	cacheVirial[flatThreadid] = 0.0;	cacheRealEnergy[flatThreadid] = 0.0;

	//calculate old first 
	if (CellsXDim <= BLOCK_DIM)	// need to check 

	{

		if (blockIdx.x % 2 == 0) {
			CalculateParticleValuesFastMolInter(Gpu_start[MolToProcess],
					MolToProcess, atomToProcess, x[atomToProcess],
					y[atomToProcess], z[atomToProcess], x, y, z, Gpu_COMX,
					Gpu_COMY, Gpu_COMZ, oldCOM.x, oldCOM.y, oldCOM.z, xAxis,
					yAxis, zAxis, xHalfAxis, yHalfAxis, zHalfAxis, EdgeXAdjust,
					EdgeYAdjust, EdgeZAdjust, CellsXDim, CellsYDim, CellsZDim,
					CellsPerXDimension, CellsPerYDimension, CellsPerZDimension,
					rCut, rCutSq, cellOffset, cellrangeOffset, NumberofCells,
					atomCountrs, atomCells, NumberOfCellsInBox, atomsMoleculeNo,
					AtomKinds, sigmaSq, epsilon_cn, nOver6, epsilon_cn_6,
					FFParticleKindCount, AtomKinds[atomToProcess], deltaEnergy,
					deltaVirial, deltaReal, len , boxOffset,					dev_calp, dev_particleCharge, dev_DoEwald, n);

			cacheEnergy[flatThreadid] -= deltaEnergy;
			cacheVirial[flatThreadid] -= deltaVirial;			cacheRealEnergy[flatThreadid] -= deltaReal;
		} else

		{

			CalculateParticleValuesFastMolInter(Gpu_start[MolToProcess],

			MolToProcess, atomToProcess, n_x, n_y, n_z, x, y, z, Gpu_COMX,
					Gpu_COMY, Gpu_COMZ, newCOM.x, newCOM.y, newCOM.z, xAxis,
					yAxis, zAxis, xHalfAxis, yHalfAxis, zHalfAxis, EdgeXAdjust,
					EdgeYAdjust, EdgeZAdjust, CellsXDim, CellsYDim, CellsZDim,
					CellsPerXDimension, CellsPerYDimension, CellsPerZDimension,
					rCut, rCutSq, cellOffset, cellrangeOffset, NumberofCells,
					atomCountrs, atomCells, NumberOfCellsInBox, atomsMoleculeNo,
					AtomKinds, sigmaSq, epsilon_cn, nOver6, epsilon_cn_6,
					FFParticleKindCount, AtomKinds[atomToProcess], deltaEnergy,
					deltaVirial, deltaReal, len, boxOffset,					dev_calp, dev_particleCharge, dev_DoEwald, n);

			cacheEnergy[flatThreadid] += deltaEnergy;
			cacheVirial[flatThreadid] += deltaVirial;			cacheRealEnergy[flatThreadid] += deltaReal;
		}

	} else {
		CalculateParticleValuesMolInter(Gpu_start[MolToProcess], len,

		MolToProcess, atomToProcess, flatThreadid, x[atomToProcess],
				y[atomToProcess], z[atomToProcess], x, y, z, Gpu_COMX, Gpu_COMY,
				Gpu_COMZ, oldCOM.x, oldCOM.y, oldCOM.z, xAxis, yAxis, zAxis,
				xHalfAxis, yHalfAxis, zHalfAxis, EdgeXAdjust, EdgeYAdjust,
				EdgeZAdjust, CellsXDim, CellsYDim, CellsZDim,
				CellsPerXDimension, CellsPerYDimension, CellsPerZDimension,
				rCut, rCutSq, cellOffset, cellrangeOffset, NumberofCells,
				atomCountrs, atomCells, NumberOfCellsInBox, atomsMoleculeNo,
				AtomKinds, sigmaSq, epsilon_cn, nOver6, epsilon_cn_6,
				FFParticleKindCount,

				AtomKinds[atomToProcess], atomsMoleculeNo[atomToProcess],
				deltaEnergy, deltaVirial, boxOffset,				dev_calp, dev_particleCharge, dev_DoEwald, n);

		cacheEnergy[flatThreadid] -= deltaEnergy;
		cacheVirial[flatThreadid] -= deltaVirial;		cacheRealEnergy[flatThreadid] -= deltaReal;

		deltaEnergy = 0;
		deltaVirial = 0;
		CalculateParticleValuesMolInter(Gpu_start[MolToProcess], len,
				MolToProcess, atomToProcess, flatThreadid, n_x, n_y, n_z, x, y,
				z, Gpu_COMX, Gpu_COMY, Gpu_COMZ, newCOM.x, newCOM.y, newCOM.z,
				xAxis, yAxis, zAxis, xHalfAxis, yHalfAxis, zHalfAxis,
				EdgeXAdjust, EdgeYAdjust, EdgeZAdjust, CellsXDim, CellsYDim,
				CellsZDim, CellsPerXDimension, CellsPerYDimension,
				CellsPerZDimension, rCut, rCutSq, cellOffset, cellrangeOffset,
				NumberofCells, atomCountrs, atomCells, NumberOfCellsInBox,
				atomsMoleculeNo, AtomKinds, sigmaSq, epsilon_cn, nOver6,
				epsilon_cn_6, FFParticleKindCount,

				AtomKinds[atomToProcess], atomsMoleculeNo[atomToProcess],
				deltaEnergy, deltaVirial, boxOffset,				dev_calp, dev_particleCharge, dev_DoEwald, n);
		cacheEnergy[flatThreadid] += deltaEnergy;
		cacheVirial[flatThreadid] += deltaVirial;		cacheRealEnergy[flatThreadid] += deltaReal;

	}

	__syncthreads();

	// add data
	//This code assumes we always have an 8x8x8 = 512 thread block.
	int offset = 0;
	if (flatThreadid < 256) {
		cacheEnergy[flatThreadid] += cacheEnergy[flatThreadid + 256];
		cacheVirial[flatThreadid] += cacheVirial[flatThreadid + 256];		cacheRealEnergy[flatThreadid] += cacheRealEnergy[flatThreadid + 256];
	}
	__syncthreads();

	if (flatThreadid < 128) {
		cacheEnergy[flatThreadid] += cacheEnergy[flatThreadid + 128];
		cacheVirial[flatThreadid] += cacheVirial[flatThreadid + 128];		cacheRealEnergy[flatThreadid] += cacheRealEnergy[flatThreadid + 128];
	}
	__syncthreads();

	if (flatThreadid < 64) {
		cacheEnergy[flatThreadid] += cacheEnergy[flatThreadid + 64];
		cacheVirial[flatThreadid] += cacheVirial[flatThreadid + 64];		cacheRealEnergy[flatThreadid] += cacheRealEnergy[flatThreadid + 64];
	}
	__syncthreads();

	if (flatThreadid < 32) {
		cacheEnergy[flatThreadid] += cacheEnergy[flatThreadid + 32];
		cacheVirial[flatThreadid] += cacheVirial[flatThreadid + 32];		cacheRealEnergy[flatThreadid] += cacheRealEnergy[flatThreadid + 32];
		__threadfence_block();
		cacheEnergy[flatThreadid] += cacheEnergy[flatThreadid + 16];
		cacheVirial[flatThreadid] += cacheVirial[flatThreadid + 16];		cacheRealEnergy[flatThreadid] += cacheRealEnergy[flatThreadid + 16];
		__threadfence_block();
		cacheEnergy[flatThreadid] += cacheEnergy[flatThreadid + 8];
		cacheVirial[flatThreadid] += cacheVirial[flatThreadid + 8];		cacheRealEnergy[flatThreadid] += cacheRealEnergy[flatThreadid + 8];
		__threadfence_block();
		cacheEnergy[flatThreadid] += cacheEnergy[flatThreadid + 4];
		cacheVirial[flatThreadid] += cacheVirial[flatThreadid + 4];		cacheRealEnergy[flatThreadid] += cacheRealEnergy[flatThreadid + 4];
		__threadfence_block();
		cacheEnergy[flatThreadid] += cacheEnergy[flatThreadid + 2];
		cacheVirial[flatThreadid] += cacheVirial[flatThreadid + 2];		cacheRealEnergy[flatThreadid] += cacheRealEnergy[flatThreadid + 2];
		__threadfence_block();
		cacheEnergy[flatThreadid] += cacheEnergy[flatThreadid + 1];
		cacheVirial[flatThreadid] += cacheVirial[flatThreadid + 1];		cacheRealEnergy[flatThreadid] += cacheRealEnergy[flatThreadid + 1];
	}	//end if

	__syncthreads();

	if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {

		dev_EnergyContrib[blockIdx.x] = cacheEnergy[0];
		dev_VirialContrib[blockIdx.x] = cacheVirial[0];		dev_RealEnergyContrib[blockIdx.x] = cacheRealEnergy[0];

		//Compress these two lines to one to save one variable declaration
		__threadfence();
		int res = (atomicInc(&BlocksDone, gridDim.x));

		if (res == (gridDim.x - 1)) {
			BlockNum = blockIdx.x;

			LastBlock = 1;
		} else

			LastBlock = 0;

	}       //end if

	__syncthreads();

	if (LastBlock) {
		//If this thread corresponds to a valid block
		if (flatThreadid < gridDim.x) {
			cacheEnergy[flatThreadid] = dev_EnergyContrib[flatThreadid];
			cacheVirial[flatThreadid] = dev_VirialContrib[flatThreadid];			cacheRealEnergy[flatThreadid] = dev_RealEnergyContrib[flatThreadid];

		}

		for (int i = BLOCK_DIM; flatThreadid + i < gridDim.x; i += BLOCK_DIM) {
			if (flatThreadid + i < gridDim.x) {
				cacheEnergy[flatThreadid] +=
						dev_EnergyContrib[flatThreadid + i];
				cacheVirial[flatThreadid] +=
						dev_VirialContrib[flatThreadid + i];				cacheRealEnergy[flatThreadid] += dev_RealEnergyContrib[flatThreadid + i];
			}       //end if
		}       //end for

		__syncthreads();
		offset = 1 << (int) __log2f((float) min(BLOCK_DIM, gridDim.x));

		for (int i = offset; i > 0; i >>= 1) {
			if (flatThreadid < i
					&& flatThreadid + i < min(BLOCK_DIM, gridDim.x)) {
				cacheEnergy[flatThreadid] += cacheEnergy[flatThreadid + i];
				cacheVirial[flatThreadid] += cacheVirial[flatThreadid + i];				cacheRealEnergy[flatThreadid] += cacheRealEnergy[flatThreadid +i];
			}       //end if

			__syncthreads();
		}       //end for

		if (flatThreadid == 0) {

			dev_EnergyContrib[0] = cacheEnergy[0];
			dev_EnergyContrib[1] = cacheVirial[0];
			dev_RealEnergyContrib[0] = cacheRealEnergy[0] * dev_qqfact[0];
		}
	}
}

// Function to run the Transform move using celllist
__global__ void TryTransformGpuCellList(		double *tempCoordsX, double *tempCoordsY, double *tempCoordsZ,
		uint *AtomKinds, double * Gpu_x, double * Gpu_y, double * Gpu_z,		double * Gpu_COMX, double * Gpu_COMY, double * Gpu_COMZ,		XYZ shift, double xAxis, double yAxis, double zAxis,
		double EdgeXAdjust, double EdgeYAdjust, double EdgeZAdjust,
		int CellsXDim, int CellsYDim, int CellsZDim,		int CellsPerXDimension, int CellsPerYDimension, int CellsPerZDimension,		uint cellOffset, uint cellrangeOffset, uint NumberofCells, // number of cells in the search box (cellsXDim * cellsYDim* cellsZDim )
		uint* atomCountrs, uint* atomCells, uint NumberOfCellsInBox, uint * atomsMoleculeNo,		double * sigmaSq, double * epsilon_cn, double * nOver6, double * epsilon_cn_6,
		uint * Gpu_start, int len,
		double xHalfAxis, double yHalfAxis, double zHalfAxis,		uint boxOffset, uint mIndex,       // mol index with offset
		uint FFParticleKindCount, double rCut, double rCutSq,
		double dev_EnergyContrib[], double dev_VirialContrib[], double dev_RealEnergyContrib[],		double dev_calp[], double dev_particleCharge[], bool *dev_DoEwald,		double *dev_qqfact, XYZ *dev_newCOM,
#ifdef MIE_INT_ONLY
		const uint * n,
#else
		const double * n
#endif
		) {

	__shared__ double n_x, n_y, n_z;
	__shared__ XYZ newCOM;
	__shared__ XYZ oldCOM;
	__shared__ uint start;
	__shared__ uint atomToProcess;

	int flatThreadid = (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x
			+ threadIdx.x;

	if (flatThreadid == 0) {
		uint blockOn2 = blockIdx.x >> 1;

		start = Gpu_start[mIndex];
		n_x = Gpu_x[start + blockOn2] + shift.x;
		n_y = Gpu_y[start + blockOn2] + shift.y;
		n_z = Gpu_z[start + blockOn2] + shift.z;

		WrapPBC(n_x, xAxis);
		WrapPBC(n_y, yAxis);
		WrapPBC(n_z, zAxis);

		tempCoordsX[blockIdx.x] = n_x;
		tempCoordsY[blockIdx.x] = n_y;
		tempCoordsZ[blockIdx.x] = n_z;
		oldCOM.x = newCOM.x = Gpu_COMX[mIndex];
		oldCOM.y = newCOM.y = Gpu_COMY[mIndex];
		oldCOM.z = newCOM.z = Gpu_COMZ[mIndex];
		newCOM += shift;

		WrapPBC(newCOM.x, xAxis);
		WrapPBC(newCOM.y, yAxis);
		WrapPBC(newCOM.z, zAxis);
		atomToProcess = Gpu_start[mIndex] + (blockOn2);		dev_newCOM[0] = newCOM;
	}

	__syncthreads();

	CalculateMolInter_CellList(start, flatThreadid, n_x, n_y, n_z, Gpu_x, Gpu_y,
			Gpu_z, Gpu_COMX, Gpu_COMY, Gpu_COMZ, newCOM, oldCOM, xAxis, yAxis,
			zAxis, xHalfAxis, yHalfAxis, zHalfAxis, EdgeXAdjust, EdgeYAdjust,
			EdgeZAdjust, CellsXDim, CellsYDim, CellsZDim, CellsPerXDimension,
			CellsPerYDimension, CellsPerZDimension, rCut, rCutSq, boxOffset,
			cellOffset, cellrangeOffset,
			NumberofCells, // number of cells in the search box (cellsXDim * cellsYDim* cellsZDim )
			atomCountrs, atomCells, Gpu_start, NumberOfCellsInBox,
			atomsMoleculeNo, AtomKinds, sigmaSq, epsilon_cn, nOver6,
			epsilon_cn_6, FFParticleKindCount, dev_EnergyContrib,
			dev_VirialContrib, dev_RealEnergyContrib, dev_calp, dev_particleCharge, dev_DoEwald,			dev_qqfact, mIndex, atomToProcess, len, n);
}

// Function to run the Transform move using celllist
__global__ void MoleculeVirialCellList(double *tempCoordsX, double *tempCoordsY,
		double *tempCoordsZ, uint * NoOfAtomsPerMol, uint *AtomKinds,
		SystemPotential * Gpu_Potential, double * Gpu_x, double * Gpu_y,
		double * Gpu_z, double * Gpu_COMX, double * Gpu_COMY, double * Gpu_COMZ,
		XYZ shift, double xAxis, double yAxis, double zAxis, double EdgeXAdjust,
		double EdgeYAdjust, double EdgeZAdjust, int CellsXDim, int CellsYDim,
		int CellsZDim, int CellsPerXDimension, int CellsPerYDimension,
		int CellsPerZDimension, uint cellOffset, uint cellrangeOffset,
		uint NumberofCells,	// number of cells in the search box (cellsXDim * cellsYDim* cellsZDim )
		uint* atomCountrs, uint* atomCells, uint NumberOfCellsInBox,
		uint * atomsMoleculeNo, uint *molKindIndex, double * sigmaSq,
		double * epsilon_cn, double * nOver6, double * epsilon_cn_6,
		double beta, double AcceptRand, uint * Gpu_start, int len,
		double xHalfAxis, double yHalfAxis, double zHalfAxis, uint boxOffset,
		uint MoleculeCount,
		uint mIndex,					// mol index with offset
		uint FFParticleKindCount, double rCut, uint molKInd, double rCutSq,
		double dev_EnergyContrib[], double dev_VirialContrib[], uint boxIndex,
		bool * Gpu_result, double dev_RealEnergyContrib[], double dev_calp[],		double dev_particleCharge[], bool *dev_DoEwald, double *dev_qqfact,
#ifdef MIE_INT_ONLY
		const uint * n,
#else
		const double * n
#endif
		) {

	__shared__ double n_x, n_y, n_z;
	__shared__ XYZ newCOM;
	__shared__ XYZ oldCOM;
	__shared__ uint start;
	__shared__ uint atomToProcess;

	int flatThreadid = (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x
			+ threadIdx.x;

	if (flatThreadid == 0) {
		uint blockOn2 = blockIdx.x >> 1;

		start = Gpu_start[mIndex];
		n_x = Gpu_x[start + blockOn2];
		n_y = Gpu_y[start + blockOn2];
		n_z = Gpu_z[start + blockOn2];

		oldCOM.x = Gpu_COMX[mIndex];
		oldCOM.y = Gpu_COMY[mIndex];
		oldCOM.z = Gpu_COMZ[mIndex];

		atomToProcess = Gpu_start[mIndex] + (blockOn2);
	}

	__syncthreads();

	CalculateMolInter_CellList(start, flatThreadid, n_x, n_y, n_z, Gpu_x, Gpu_y,
			Gpu_z, Gpu_COMX, Gpu_COMY, Gpu_COMZ, newCOM, oldCOM, xAxis, yAxis,
			zAxis, xHalfAxis, yHalfAxis, zHalfAxis, EdgeXAdjust, EdgeYAdjust,
			EdgeZAdjust, CellsXDim, CellsYDim, CellsZDim, CellsPerXDimension,
			CellsPerYDimension, CellsPerZDimension, rCut, rCutSq, boxOffset,
			cellOffset, cellrangeOffset,
			NumberofCells,// number of cells in the search box (cellsXDim * cellsYDim* cellsZDim )
			atomCountrs, atomCells, Gpu_start, NumberOfCellsInBox,
			atomsMoleculeNo, AtomKinds, sigmaSq, epsilon_cn, nOver6,
			epsilon_cn_6, FFParticleKindCount, dev_EnergyContrib,
			dev_VirialContrib, dev_RealEnergyContrib, dev_calp,			dev_particleCharge, dev_DoEwald, dev_qqfact, mIndex, atomToProcess, len, n);

	__syncthreads();

	if (blockIdx.x == BlockNum) {

		if (flatThreadid == 0) {
			start = Gpu_start[mIndex];
			Gpu_result[0] = false;

			if (AcceptRand < Gpu_BoltzW(beta, dev_EnergyContrib[0])) {
				Gpu_result[0] = true;
				int xCellPos;
				int yCellPos;
				int zCellPos;
				int position;
				int index = -1;
				for (int i = 0; i < len; i++) {

					/////////////////////////////////////////////////////////
					// change the cells (take out the old and put the new)
					xCellPos = ((int) Gpu_x[start + i] >> HALF_MICROCELL_DIM);
					yCellPos = ((int) Gpu_y[start + i] >> HALF_MICROCELL_DIM);
					zCellPos = ((int) Gpu_z[start + i] >> HALF_MICROCELL_DIM);

					position = (zCellPos * CellsPerZDimension + yCellPos)
							* CellsPerYDimension + xCellPos;// flat 3d to 1d 

					index = -1;
					for (int j = 0; j < MAX_ATOMS_PER_CELL; j++) {
						if (atomCells[(j * NumberOfCellsInBox + position)
								+ cellOffset * MAX_ATOMS_PER_CELL]
								== (start + i)) {

							index = j;
							break;

						}

					}

					for (int j = index;
							j < atomCountrs[position + cellOffset] - 1; j++) {

						atomCells[(j * NumberOfCellsInBox + position)
								+ cellOffset * MAX_ATOMS_PER_CELL] =
								atomCells[((j + 1) * NumberOfCellsInBox
										+ position)
										+ cellOffset * MAX_ATOMS_PER_CELL];

					}

					atomCountrs[position + cellOffset] -= 1;

					// add new coords to cells
					Gpu_x[start + i] = tempCoordsX[i * 2];
					Gpu_y[start + i] = tempCoordsY[i * 2];
					Gpu_z[start + i] = tempCoordsZ[i * 2];

					xCellPos = ((int) Gpu_x[start + i] >> HALF_MICROCELL_DIM);
					yCellPos = ((int) Gpu_y[start + i] >> HALF_MICROCELL_DIM);
					zCellPos = ((int) Gpu_z[start + i] >> HALF_MICROCELL_DIM);

					position = (zCellPos * CellsPerZDimension + yCellPos)
							* CellsPerYDimension + xCellPos;// flat 3d to 1d 

					atomCells[(atomCountrs[position + cellOffset]
							* NumberOfCellsInBox + position)
							+ cellOffset * MAX_ATOMS_PER_CELL] = start + i;

					atomCountrs[position + cellOffset] += 1;

				}

				Gpu_COMX[mIndex] = newCOM.x;
				Gpu_COMY[mIndex] = newCOM.y;
				Gpu_COMZ[mIndex] = newCOM.z;

				Gpu_Potential->Add(boxIndex, dev_EnergyContrib[0],
						dev_EnergyContrib[1]);

				Gpu_Potential->Total();

			}
			BlocksDone = 0;
			BlockNum = -1;

		}
	}

}

// Function to run the Rotate move using celllist
__global__ void TryRotateGpuCellList(double *tempCoordsX,		double *tempCoordsY, double *tempCoordsZ, uint * NoOfAtomsPerMol,		uint *AtomKinds, SystemPotential * Gpu_Potential, double * Gpu_x,		double * Gpu_y, double * Gpu_z, double * Gpu_COMX, double * Gpu_COMY,		double * Gpu_COMZ, TransformMatrix matrix, double xAxis, double yAxis, double zAxis,		double EdgeXAdjust, double EdgeYAdjust, double EdgeZAdjust,		int CellsXDim, int CellsYDim, int CellsZDim, int CellsPerXDimension,		int CellsPerYDimension, int CellsPerZDimension, int cellOffset,		int cellrangeOffset,		uint NumberofCells, // number of cells in the search box (cellsXDim * cellsYDim* cellsZDim )		uint* atomCountrs, uint* atomCells, uint NumberOfCellsInBox,		uint * atomsMoleculeNo, uint *molKindIndex, double * sigmaSq,		double * epsilon_cn, double * nOver6, double * epsilon_cn_6,		double beta, double AcceptRand, uint * Gpu_start, int len,		double xHalfAxis, double yHalfAxis, double zHalfAxis, uint boxOffset,		uint MoleculeCount,		uint mIndex, // mol index with offset		uint FFParticleKindCount, double rCut, uint molKInd, double rCutSq,		double dev_EnergyContrib[], double dev_VirialContrib[],		double dev_RealEnergyContrib[],		double dev_calp[], double dev_particleCharge[], bool *dev_DoEwald,		double *dev_qqfact, uint boxIndex, bool * Gpu_result, XYZ *dev_newCOM,
#ifdef MIE_INT_ONLY
		const uint * n,
#else
		const double * n
#endif
		) {

	__shared__ double n_x, n_y, n_z;

	__shared__ XYZ center;
	__shared__ int atomToProcess;

	int flatThreadid = (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x
			+ threadIdx.x;

	uint start;

	if (flatThreadid == 0) {
		uint blockOn2 = blockIdx.x >> 1;
		start = Gpu_start[mIndex];

		center.x = Gpu_COMX[mIndex];
		center.y = Gpu_COMY[mIndex];
		center.z = Gpu_COMZ[mIndex];

		n_x = Gpu_x[start + blockOn2];
		n_y = Gpu_y[start + blockOn2];
		n_z = Gpu_z[start + blockOn2];

		UnwrapPBC(n_x, center.x, xAxis, xHalfAxis);
		UnwrapPBC(n_y, center.y, yAxis, yHalfAxis);
		UnwrapPBC(n_z, center.z, zAxis, zHalfAxis);

		n_x += (-1.0 * center.x);
		n_y += (-1.0 * center.y);
		n_z += (-1.0 * center.z);

		matrix.Apply(n_x, n_y, n_z);

		n_x += (center.x);
		n_y += (center.y);
		n_z += (center.z);

		WrapPBC(n_x, xAxis);
		WrapPBC(n_y, yAxis);
		WrapPBC(n_z, zAxis);

		tempCoordsX[blockIdx.x] = n_x;
		tempCoordsY[blockIdx.x] = n_y;
		tempCoordsZ[blockIdx.x] = n_z;

		atomToProcess = Gpu_start[mIndex] + (blockOn2);		dev_newCOM[0].x = center.x;		dev_newCOM[0].y = center.y;		dev_newCOM[0].z = center.z;
	}

	__syncthreads();

	CalculateMolInter_CellList(start, flatThreadid, n_x, n_y, n_z, Gpu_x, Gpu_y,
			Gpu_z, Gpu_COMX, Gpu_COMY, Gpu_COMZ, center, center, xAxis, yAxis,
			zAxis, xHalfAxis, yHalfAxis, zHalfAxis, EdgeXAdjust, EdgeYAdjust,
			EdgeZAdjust, CellsXDim, CellsYDim, CellsZDim, CellsPerXDimension,
			CellsPerYDimension, CellsPerZDimension, rCut, rCutSq, boxOffset,
			cellOffset, cellrangeOffset,
			NumberofCells,// number of cells in the search box (cellsXDim * cellsYDim* cellsZDim )
			atomCountrs, atomCells, Gpu_start, NumberOfCellsInBox,
			atomsMoleculeNo, AtomKinds, sigmaSq, epsilon_cn, nOver6,
			epsilon_cn_6, FFParticleKindCount, dev_EnergyContrib,
			dev_VirialContrib, dev_RealEnergyContrib, dev_calp, dev_particleCharge, dev_DoEwald,			dev_qqfact, mIndex, atomToProcess, len, n);

}

__global__ void CalculateParticleInter_CellList(

int len, double *n_x, double *n_y, double *n_z, double* x, double* y, double* z,
		double xAxis, double yAxis, double zAxis, double xHalfAxis,
		double yHalfAxis, double zHalfAxis, double EdgeXAdjust,
		double EdgeYAdjust, double EdgeZAdjust, int CellsXDim, int CellsYDim,
		int CellsZDim, int CellsPerXDimension, int CellsPerYDimension,
		int CellsPerZDimension, double rCut, double rCutSq, uint boxOffset,
		uint cellOffset, uint cellrangeOffset,
		uint NumberofCells,	// number of cells in the search box (cellsXDim * cellsYDim* cellsZDim )
		const uint* atomCountrs, const uint* atomCells, uint * Gpu_start,
		uint NumberOfCellsInBox, uint * atomsMoleculeNo, uint *AtomKinds,
		double * sigmaSq, double * epsilon_cn, double * nOver6,
		double * epsilon_cn_6, uint FFParticleKindCount,
		double dev_EnergyContrib[], int MolToProcess, int atomToProcess,
#ifdef MIE_INT_ONLY
		const uint * n,
#else
		const double * n
#endif

		) {

	__shared__ double cacheEnergy[BLOCK_SIZE];// should be max number of cells this block will process ()

	int flatThreadid = (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x
			+ threadIdx.x;

	double deltaEnergy = 0.0;

	cacheEnergy[flatThreadid] = 0.0;

	//calculate old first 
	if (CellsXDim <= BLOCK_DIM)					// need to check 

	{

		CalculateParticleValuesFastPartInter(

		len, MolToProcess, Gpu_start[MolToProcess],
				atomToProcess + Gpu_start[MolToProcess], n_x[blockIdx.x],
				n_y[blockIdx.x], n_z[blockIdx.x], x, y, z, xAxis, yAxis, zAxis,
				xHalfAxis, yHalfAxis, zHalfAxis, EdgeXAdjust, EdgeYAdjust,
				EdgeZAdjust, CellsXDim, CellsYDim, CellsZDim,
				CellsPerXDimension, CellsPerYDimension, CellsPerZDimension,
				rCut, rCutSq, boxOffset, cellOffset, cellrangeOffset,
				NumberofCells, atomCountrs, atomCells, NumberOfCellsInBox,
				atomsMoleculeNo, AtomKinds, sigmaSq, epsilon_cn, nOver6,
				epsilon_cn_6, FFParticleKindCount,
				AtomKinds[atomToProcess + Gpu_start[MolToProcess]], deltaEnergy,

				n);

	} else {
		CalculateParticleValuesPartInter(len, Gpu_start[MolToProcess],
				MolToProcess, atomToProcess + Gpu_start[MolToProcess],
				flatThreadid, n_x[blockIdx.x], n_y[blockIdx.x], n_z[blockIdx.x],
				x, y, z, xAxis, yAxis, zAxis, xHalfAxis, yHalfAxis, zHalfAxis,
				EdgeXAdjust, EdgeYAdjust, EdgeZAdjust, CellsXDim, CellsYDim,
				CellsZDim, CellsPerXDimension, CellsPerYDimension,
				CellsPerZDimension, rCut, rCutSq, cellOffset, cellrangeOffset,
				NumberofCells, atomCountrs, atomCells, NumberOfCellsInBox,
				atomsMoleculeNo, AtomKinds, sigmaSq, epsilon_cn, nOver6,
				epsilon_cn_6, FFParticleKindCount,
				AtomKinds[atomToProcess + Gpu_start[MolToProcess]],
				atomsMoleculeNo[atomToProcess], deltaEnergy, n);
		__syncthreads();

	}

	cacheEnergy[flatThreadid] += deltaEnergy;

	__syncthreads();

	// add data
	//This code assumes we always have an 8x8x8 = 512 thread block.
	int offset = 0;
	if (flatThreadid < 256) {
		cacheEnergy[flatThreadid] += cacheEnergy[flatThreadid + 256];

	}
	__syncthreads();

	if (flatThreadid < 128) {
		cacheEnergy[flatThreadid] += cacheEnergy[flatThreadid + 128];

	}
	__syncthreads();

	if (flatThreadid < 64) {
		cacheEnergy[flatThreadid] += cacheEnergy[flatThreadid + 64];

	}
	__syncthreads();

	if (flatThreadid < 32) {
		cacheEnergy[flatThreadid] += cacheEnergy[flatThreadid + 32];

		__threadfence_block();
		cacheEnergy[flatThreadid] += cacheEnergy[flatThreadid + 16];

		__threadfence_block();
		cacheEnergy[flatThreadid] += cacheEnergy[flatThreadid + 8];

		__threadfence_block();
		cacheEnergy[flatThreadid] += cacheEnergy[flatThreadid + 4];

		__threadfence_block();
		cacheEnergy[flatThreadid] += cacheEnergy[flatThreadid + 2];

		__threadfence_block();
		cacheEnergy[flatThreadid] += cacheEnergy[flatThreadid + 1];

	}	//end if

	__syncthreads();

	if (flatThreadid == 0) {
		dev_EnergyContrib[blockIdx.x] = cacheEnergy[0];

	}       //end if

}

void CalculateEnergy::GetParticleEnergyGPU_CellList(uint box, double * en,
		XYZArray positions, int numAtoms, int mOff, int CurrentPos, int MolKind,
		int nLJTrials) {

	if (box >= BOXES_WITH_U_NB)
		return;

	int cellOffset;
	int cellrangeOffset;

	if (box == 0) {

		cellOffset = 0;
		cellrangeOffset = 0;

	} else {

		cellOffset = TotalCellsPerBox[0];
		cellrangeOffset = TotalCellsPerBox[0] * MAX_ATOMS_PER_CELL;
	}

	// copy positions to GPU array

	cudaMemcpy(trialPosX, positions.x, sizeof(double) * MaxTrialNumber,
			cudaMemcpyHostToDevice);
	cudaMemcpy(trialPosY, positions.y, sizeof(double) * MaxTrialNumber,
			cudaMemcpyHostToDevice);
	cudaMemcpy(trialPosZ, positions.z, sizeof(double) * MaxTrialNumber,
			cudaMemcpyHostToDevice);

	CalculateParticleInter_CellList<<<nLJTrials, BlockSize>>>(numAtoms,
			trialPosX, trialPosY, trialPosZ, Gpu_x, Gpu_y, Gpu_z,
			currentAxes.axis.x[box], currentAxes.axis.y[box],
			currentAxes.axis.z[box], currentAxes.halfAx.x[box],
			currentAxes.halfAx.y[box], currentAxes.halfAx.z[box],
			EdgeAdjust[box * 3], EdgeAdjust[box * 3 + 1],
			EdgeAdjust[box * 3 + 2], CellDim[box * 3], CellDim[box * 3 + 1],
			CellDim[box * 3 + 2], CellsPerDim[box * 3],
			CellsPerDim[box * 3 + 1], CellsPerDim[box * 3 + 2],
			currentAxes.rCut, currentAxes.rCutSq, (box == 0) ? 0 : AtomCount[0],
			cellOffset, cellrangeOffset,
			CellDim[box * 3] * CellDim[box * 3 + 1] * CellDim[box * 3 + 2],
			atomCountrs, atomCells, Gpu_start, TotalCellsPerBox[box],
			atomsMoleculeNo, Gpu_atomKinds, Gpu_sigmaSq, Gpu_epsilon_cn,
			Gpu_nOver6, Gpu_epsilon_cn_6, forcefield.particles->NumKinds(),
			dev_partEnergy, mOff, CurrentPos, Gpu_partn);

	cudaMemcpy(FinalEnergyNVirial, dev_partEnergy, nLJTrials * sizeof(double),
			cudaMemcpyDeviceToHost);

	for (int i = 0; i < nLJTrials; ++i)

		en[i] += FinalEnergyNVirial[i];

	cudaError_t code = cudaGetLastError();
	if (code != cudaSuccess) {
		printf("Cuda error end of transferrrrr energy calc -- %s\n",
				cudaGetErrorString(code));
		exit(2);

	}

}

void CalculateEnergy::GetParticleEnergy(uint box, double * en,
		XYZArray positions, int numAtoms, int mOff, int CurrentPos, int MolKind,
		int nLJTrials) {
#ifdef CELL_LIST
	GetParticleEnergyGPU_CellList(box, en, positions, numAtoms, mOff,
			CurrentPos, MolKind, nLJTrials);
#else
	GetParticleEnergyGPU(box, en,positions,numAtoms, mOff, CurrentPos, MolKind,nLJTrials); // GPU call
#endif
}//******************************** ewald functions *******************************************************void CalculateEnergy::SetupRecip(int box){	int count = 0;	double alpsqr4 = 4.0 * calp[box] * calp[box];	double cst = 2 * M_PI / currentAxes.axis.BoxSize(box);	double ksqr;	for(double x = 0.0; x <= kmax1[0]; x++){		int nky_max = sqrt(pow(kmax1[0],2) - pow(x,2));		int nky_min = -nky_max;		if(x == 0.0)nky_min = 0.0;		for(double y = nky_min; y <= nky_max; y++){			int nkz_max = sqrt(pow(kmax1[0],2) - pow(x,2) - pow(y,2));			int nkz_min = -nkz_max;			if(x == 0.0 && y == 0.0) nkz_min = 1;			for(double z = nkz_min; z <= nkz_max; z++){				kxyz.push_back(cst * x);				kxyz.push_back(cst * y);				kxyz.push_back(cst * z);				ksqr = cst * x * cst * x + cst * y * cst * y + cst * z * cst * z;				prefact.push_back(qqfact[0] * exp(-ksqr/alpsqr4)/(ksqr*(currentAxes.volume[box]/4/M_PI)));				count++;			}		}	}	RecipSize[box] = count;}void CalculateEnergy::MolCorrection(double& correction, uint molIndex, int box) const{	if(box >= BOXES_WITH_U_NB) return;	double dist, distSq;	XYZ virComponents;	MoleculeKind& thisKind = mols.kinds[mols.kIndex[molIndex]];	for(uint i = 0; i < thisKind.NumAtoms()-1; i++){		for(uint j = i+1; j < thisKind.NumAtoms(); j++){			currentAxes.InRcut(distSq, virComponents, currentCoords, mols.start[molIndex] + i,					mols.start[molIndex] + j, box);			dist = sqrt(distSq);			correction = correction + (thisKind.atomCharge[i] * thisKind.atomCharge[j] * erf(calp[box]*dist)/dist);		}	}}void CalculateEnergy::BoxSelf(double& self, int box){	if(box >= BOXES_WITH_U_NB) return;	for(int i=0; i<mols.kindsCount; i++){		MoleculeKind const& thisKind = mols.kinds[i];		int length = thisKind.NumAtoms();		MolSelfEnergy[i] = 0.0;		for(int j=0; j<length; j++){			MolSelfEnergy[i] += (thisKind.atomCharge[j] * thisKind.atomCharge[j]);		}		self += (MolSelfEnergy[i] * molLookup.NumKindInBox(i, box));	}}__device__ double DotProduct(int atom, double kx, double ky, double kz, double *x, double *y, double *z){	double X = x[atom], Y = y[atom], Z = z[atom];	return (X * kx + Y * ky + Z * kz);}__device__ void CalcRealGpu(double distSq, double* dev_calp, double& realEnergy,		double* particleCharge, int tarParticle, int ParticleNumber, int box){	double dist = sqrt(distSq);	double erfc_variable = dev_calp[box] * dist;	realEnergy += particleCharge[tarParticle] * particleCharge[ParticleNumber] * erfc(erfc_variable) / dist;}void CalculateEnergy::SwitchRecipVector(){	double *tS, *tC;	tS = dev_RecipSinSum;	dev_RecipSinSum = dev_SinSumNew;	dev_SinSumNew = tS;	tC = dev_RecipCosSum;	dev_RecipCosSum = dev_CosSumNew;	dev_CosSumNew = tC;}__global__ void Gpu_CalculateSystemRecip(uint * NoOfAtomsPerMol,		double *x, double *y, double *z, uint *Gpu_start, uint boxOffset, uint MolCount,		int imageOffset, double dev_prefact[], double dev_kxyz[],		double dev_RecipSinSum[], double dev_RecipCosSum[], double dev_RecipEnergyContrib[],		double *dev_particleCharge, uint MolOffset, int box,#ifdef MIE_INT_ONLY		const uint * n,#else		const double * n#endif		) {	__shared__ double cacheSinSum[MAXTHREADSPERBLOCK];	__shared__ double cacheCosSum[MAXTHREADSPERBLOCK];	__shared__ bool LastBlock;	double kx, ky, kz;	int imageId = blockIdx.x;	double DotProductVal;	int MolPerThread = (MolCount + blockDim.x - 1) / blockDim.x;	int MolId, MolStart;	cacheSinSum[threadIdx.x] = 0.0;	cacheCosSum[threadIdx.x] = 0.0;	kx = dev_kxyz[imageId * 3 + imageOffset];		//			kx = kxyz[box][imageId][0];	ky = dev_kxyz[imageId * 3 + 1 + imageOffset];	//			ky = kxyz[box][imageId][1];	kz = dev_kxyz[imageId * 3 + 2 + imageOffset];	//			kz = kxyz[box][imageId][2];	for(int i = 0; i < MolPerThread; i++) {		MolId = threadIdx.x + i * blockDim.x;		if(MolId < MolCount){			MolStart = Gpu_start[MolId + MolOffset];			for (int k = 0; k < NoOfAtomsPerMol[MolId + MolOffset]; k++) {				int atom = k + MolStart;				DotProductVal = DotProduct(atom, kx, ky, kz, x, y, z);				cacheSinSum[threadIdx.x] += dev_particleCharge[atom] * sin(DotProductVal);				cacheCosSum[threadIdx.x] += dev_particleCharge[atom] * cos(DotProductVal);			}		}	}	__syncthreads();	// add data	int offset = 1 << (int) __log2f((float) blockDim.x);	if (blockDim.x < MAXTHREADSPERBLOCK) {		if ((threadIdx.x + offset) < blockDim.x) {			cacheSinSum[threadIdx.x] += cacheSinSum[threadIdx.x + offset];			cacheCosSum[threadIdx.x] += cacheCosSum[threadIdx.x + offset];		}		__syncthreads();	}	for (int i = offset >> 1; i > 32; i >>= 1) {		if (threadIdx.x < i) {			cacheSinSum[threadIdx.x] += cacheSinSum[threadIdx.x + i];			cacheCosSum[threadIdx.x] += cacheCosSum[threadIdx.x + i];		}           //end if		__syncthreads();	}           //end for	if (threadIdx.x < 32) {		offset = min(offset, 64);		switch (offset) {		case 64:			cacheSinSum[threadIdx.x] += cacheSinSum[threadIdx.x + 32];			cacheCosSum[threadIdx.x] += cacheCosSum[threadIdx.x + 32];			__threadfence_block();		case 32:			cacheSinSum[threadIdx.x] += cacheSinSum[threadIdx.x + 16];			cacheCosSum[threadIdx.x] += cacheCosSum[threadIdx.x + 16];			__threadfence_block();		case 16:			cacheSinSum[threadIdx.x] += cacheSinSum[threadIdx.x + 8];			cacheCosSum[threadIdx.x] += cacheCosSum[threadIdx.x + 8];			__threadfence_block();		case 8:			cacheSinSum[threadIdx.x] += cacheSinSum[threadIdx.x + 4];			cacheCosSum[threadIdx.x] += cacheCosSum[threadIdx.x + 4];			__threadfence_block();		case 4:			cacheSinSum[threadIdx.x] += cacheSinSum[threadIdx.x + 2];			cacheCosSum[threadIdx.x] += cacheCosSum[threadIdx.x + 2];			__threadfence_block();		case 2:			cacheSinSum[threadIdx.x] += cacheSinSum[threadIdx.x + 1];			cacheCosSum[threadIdx.x] += cacheCosSum[threadIdx.x + 1];		}	}           //end if	if (threadIdx.x == 0) {		dev_RecipSinSum[imageId + imageOffset] = cacheSinSum[0];		dev_RecipCosSum[imageId + imageOffset] = cacheCosSum[0];		//Compress these two lines to one to save one variable declaration		dev_RecipEnergyContrib[imageId] = ((cacheSinSum[0] * cacheSinSum[0]) + (cacheCosSum[0] * cacheCosSum[0]))							* dev_prefact[imageOffset + imageId];		__threadfence();		LastBlock = (atomicInc(&BlocksDone, gridDim.x)				== gridDim.x - 1);	}       //end if	__syncthreads();	if (LastBlock) {		//If this thread corresponds to a valid block		cacheSinSum[threadIdx.x] = cacheCosSum[threadIdx.x] = 0;		if (threadIdx.x < gridDim.x) {			cacheSinSum[threadIdx.x] = dev_RecipEnergyContrib[threadIdx.x];		}		for (int i = blockDim.x; threadIdx.x + i < gridDim.x; i += blockDim.x) {			cacheSinSum[threadIdx.x] += dev_RecipEnergyContrib[threadIdx.x + i];		}       //end for		__syncthreads();		offset = 1 << (int) __log2f((float) min(blockDim.x, gridDim.x));		for (int i = offset; i > 0; i >>= 1) {			if (threadIdx.x < i && threadIdx.x + i < min(blockDim.x, gridDim.x)) {				cacheSinSum[threadIdx.x] += cacheSinSum[threadIdx.x + i];			}       //end if			__syncthreads();		}       //end for		if (threadIdx.x == 0) {			BlocksDone = 0;			dev_RecipEnergyContrib[0] = cacheSinSum[0];		}	}}
SystemPotential CalculateEnergy::SystemRecipGPU() {	int offset = 0;	int ThreadsPerBlock = 0;	int BlocksPerGrid = 0;	ThreadsPerBlock = MAXTHREADSPERBLOCK;	if (ThreadsPerBlock == 0) {		ThreadsPerBlock = 1;	}	BlocksPerGrid = RecipSize[0];	if (BlocksPerGrid == 0) {		BlocksPerGrid = 1;	}	double FinalEnergyNVirial[2], *dev_RecipEnergyContrib;	FinalEnergyNVirial[0] = 0.0;	FinalEnergyNVirial[1] = 0.0;	double NewEnergy1 = 0.0;	double NewVirial1 = 0.0;	cudaMalloc((void**) &dev_RecipEnergyContrib, sizeof(double) * BlocksPerGrid);	//Box 0	if (MolCount[0] != 0) {		Gpu_CalculateSystemRecip<<<BlocksPerGrid, ThreadsPerBlock, 0, stream0>>>(				NoOfAtomsPerMol, Gpu_x, Gpu_y, Gpu_z,				Gpu_start, offset, MolCount[0],				0, dev_prefact, dev_kxyz,				dev_RecipSinSum, dev_RecipCosSum, dev_RecipEnergyContrib,				dev_particleCharge, 0, 0, Gpu_partn);		cudaMemcpyAsync(&FinalEnergyNVirial, dev_RecipEnergyContrib,				2 * sizeof(double), cudaMemcpyDeviceToHost, stream0);		cudaStreamSynchronize(stream0);		NewEnergy1 = FinalEnergyNVirial[0];		NewVirial1 = FinalEnergyNVirial[1];		printf("Reciprocal energy box 0=%f\n", FinalEnergyNVirial[0]);	}#if ENSEMBLE == GEMC	ThreadsPerBlock1 = MAXTHREADSPERBLOCK;	if (ThreadsPerBlock1 == 0) {		ThreadsPerBlock1 = 1;	}	BlocksPerGrid1 = RecipSize[1];	if (BlocksPerGrid1 == 0) {		BlocksPerGrid1 = 1;	}	double *dev_RecipEnergyContrib1;	cudaMalloc((void**) dev_RecipEnergyContrib1, BlocksPerGrid1 * sizeof(double));	FinalEnergyNVirial[0] = 0.0;	FinalEnergyNVirial[1] = 0.0;	offset = AtomCount[0];	double NewEnergy2 = 0.0;	double NewVirial2 = 0.0;	if (MolCount[1] != 0) {		Gpu_CalculateSystemRecip<<<BlocksPerGrid1, ThreadsPerBlock1, 0, stream1>>>(						NoOfAtomsPerMol, Gpu_x, Gpu_y, Gpu_z,						Gpu_start, offset, MolCount[1],						RecipSize[0], dev_prefact, dev_kxyz,						dev_RecipSinSum, dev_RecipCosSum, dev_RecipEnergyContrib1,						dev_particleCharge, MolCount[0], 1, Gpu_partn);		cudaMemcpyAsync(&FinalEnergyNVirial, dev_RecipEnergyContrib1,				2 * sizeof(double), cudaMemcpyDeviceToHost, stream1);		cudaStreamSynchronize(stream1);		NewEnergy2 = FinalEnergyNVirial[0];		NewVirial2 = FinalEnergyNVirial[1];		printf("Reciprocal energy box 1=%f\n", FinalEnergyNVirial[0]);	}#endif	SystemPotential currentpot;	currentpot.boxEnergy[0].recip = NewEnergy1;//	currentpot.boxVirial[0].recip = NewVirial1;	cudaFree(dev_RecipEnergyContrib);#if ENSEMBLE == GEMC	currentpot.boxEnergy[1].recip = NewEnergy2;//	currentpot.boxVirial[1].recip = NewVirial2;	cudaFree(dev_RecipEnergyContrib1);#endif	currentpot.Total();	cudaDeviceSynchronize();	cudaError_t code = cudaGetLastError();	if (code != cudaSuccess) {		printf("Cuda error at end of Calculate Total Energy -- %s\n",				cudaGetErrorString(code));		exit(2);	}	return currentpot;}__global__ void Gpu_CalculateMolRecip(double *tempCoordsX,		double *tempCoordsY, double *tempCoordsZ, double *dev_particleCharge,		int box, double *dev_calp, double *dev_prefact, double *dev_kxyz,		double *dev_RecipSinSum, double *dev_RecipCosSum, double *dev_SinSumNew,		double *dev_CosSumNew, double *x, double *y, double *z, uint *Gpu_start,		int len, int mIndex, int *RecipSize, double *dev_RecipEnergyContrib,		double *dev_Array){	int imageId = blockIdx.x * blockDim.x + threadIdx.x;	__shared__ double RecipNew[MAXTHREADSPERBLOCK];	__shared__ double RecipOld[MAXTHREADSPERBLOCK];	__shared__ double RecipDif[MAXTHREADSPERBLOCK];	__shared__ bool LastBlock;	if(imageId < RecipSize[box]){		RecipNew[threadIdx.x] = 0.0;		RecipOld[threadIdx.x] = 0.0;		RecipDif[threadIdx.x] = 0.0;		double start = Gpu_start[mIndex];		double kx, ky, kz;		double OldSin = 0.0, OldCos = 0.0, NewSin = 0.0, NewCos = 0.0, SinSum = 0.0, CosSum = 0.0				, RecipSin = 0.0, RecipCos = 0.0;		double DotProductOld, DotProductNew;		int RecipOffset = (box == 0)? 0:RecipSize[0];		int atomId, RecipId = (RecipOffset + imageId), kId = RecipId * 3;		//each thread loop through all atoms of the selected molecule and calculate the new reciprocal energy.		for(int atomOffset = 0; atomOffset < len; atomOffset++){			atomId = start + atomOffset;			kx = dev_kxyz[kId];			ky = dev_kxyz[kId + 1];			kz = dev_kxyz[kId + 2];			DotProductOld = DotProduct(atomId, kx, ky, kz, x, y, z);			if(dev_Array == NULL)				DotProductNew = DotProduct(atomOffset, kx, ky, kz, tempCoordsX, tempCoordsY, tempCoordsZ);			else				DotProductNew = kx * dev_Array[atomOffset] + ky * dev_Array[atomOffset + len] +						kz * dev_Array[atomOffset + len * 2];			OldSin += dev_particleCharge[atomId] * sin(DotProductOld);			OldCos += dev_particleCharge[atomId] * cos(DotProductOld);			NewSin += dev_particleCharge[atomId] * sin(DotProductNew);			NewCos += dev_particleCharge[atomId] * cos(DotProductNew);		}		RecipSin = dev_RecipSinSum[RecipId];		RecipCos = dev_RecipCosSum[RecipId];		SinSum = dev_SinSumNew[RecipId] = RecipSin - OldSin + NewSin;		CosSum = dev_CosSumNew[RecipId] = RecipCos - OldCos + NewCos;		RecipNew[threadIdx.x] = (SinSum * SinSum + CosSum * CosSum) * dev_prefact[RecipId];		RecipOld[threadIdx.x] = (RecipSin * RecipSin + RecipCos * RecipCos) * dev_prefact[RecipId];	}	//end if imageId < RecipSize//	if(blockIdx.x == 0 && threadIdx.x == 0)//		printf("image: %d, recip dif: %lf\n", imageId, RecipNew[threadIdx.x] - RecipOld[threadIdx.x]);	__syncthreads();	//loop unroll, parallel reduction, log function enables the case that NO of molecules < MAXTHREADSPERBLOCK	int workSize = min(blockDim.x, RecipSize[box] - blockDim.x * blockIdx.x);	int offset = 1 << (int) __log2f((float) workSize);	if (blockDim.x < MAXTHREADSPERBLOCK) {		if ((threadIdx.x + offset) < workSize) {			RecipNew[threadIdx.x] += RecipNew[threadIdx.x + offset];			RecipOld[threadIdx.x] += RecipOld[threadIdx.x + offset];		}		__syncthreads();	}	for (int i = offset >> 1; i > 32; i >>= 1) {		if (threadIdx.x < i) {			RecipNew[threadIdx.x] += RecipNew[threadIdx.x + i];			RecipOld[threadIdx.x] += RecipOld[threadIdx.x + i];		}           //end if		__syncthreads();	}           //end for	if (threadIdx.x < 32) {		offset = min(offset, 64);		switch (offset) {		case 64:			RecipNew[threadIdx.x] += RecipNew[threadIdx.x + 32];			RecipOld[threadIdx.x] += RecipOld[threadIdx.x + 32];			__threadfence_block();		case 32:			RecipNew[threadIdx.x] += RecipNew[threadIdx.x + 16];			RecipOld[threadIdx.x] += RecipOld[threadIdx.x + 16];			__threadfence_block();		case 16:			RecipNew[threadIdx.x] += RecipNew[threadIdx.x + 8];			RecipOld[threadIdx.x] += RecipOld[threadIdx.x + 8];			__threadfence_block();		case 8:			RecipNew[threadIdx.x] += RecipNew[threadIdx.x + 4];			RecipOld[threadIdx.x] += RecipOld[threadIdx.x + 4];			__threadfence_block();		case 4:			RecipNew[threadIdx.x] += RecipNew[threadIdx.x + 2];			RecipOld[threadIdx.x] += RecipOld[threadIdx.x + 2];			__threadfence_block();		case 2:			RecipNew[threadIdx.x] += RecipNew[threadIdx.x + 1];			RecipOld[threadIdx.x] += RecipOld[threadIdx.x + 1];		}	//end switch	}	//end if	if (threadIdx.x == 0) {		dev_RecipEnergyContrib[blockIdx.x] = RecipNew[0] - RecipOld[0];//		printf("Recip New: %lf, Old: %lf, Dif: %lf\n", RecipNew[0], RecipOld[0], dev_RecipEnergyContrib[blockIdx.x]);		//Compress these two lines to one to save one variable declaration		__threadfence();		LastBlock = (atomicInc(&BlocksDone, gridDim.x)				== gridDim.x - 1);	}       //end if	__syncthreads();	if (LastBlock) {		//If this thread corresponds to a valid block		if (threadIdx.x < gridDim.x) {			RecipDif[threadIdx.x] = dev_RecipEnergyContrib[threadIdx.x];		}		for (int i = blockDim.x; threadIdx.x + i < gridDim.x; i += blockDim.x) {			RecipDif[threadIdx.x] += dev_RecipEnergyContrib[threadIdx.x + i];		}       //end for		__syncthreads();		offset = 1 << (int) __log2f((float) min(blockDim.x, gridDim.x));		for (int i = offset; i > 0; i >>= 1) {			if (threadIdx.x < i && threadIdx.x + i < min(blockDim.x, gridDim.x)) {				RecipDif[threadIdx.x] += RecipDif[threadIdx.x + i];			}       //end if			__syncthreads();		}       //end for		if (threadIdx.x == 0) {			BlocksDone = 0;			dev_RecipEnergyContrib[0] = RecipDif[0];//			printf("Recip dif: %lf\n", dev_RecipEnergyContrib[0]);		}	}}__global__ void Gpu_AcceptOrReject(bool *Gpu_result, double *dev_EnergyContrib,		double *dev_RealEnergyContrib, double *dev_RecipEnergyContrib, uint *Gpu_start,		double beta, double *Gpu_x, double *Gpu_y, double *Gpu_z, double CellsPerXDimension,		double CellsPerYDimension, double CellsPerZDimension, int NumberOfCellsInBox,		int cellOffset, uint *atomCountrs, uint *atomCells, double *tempCoordsX,		double *tempCoordsY, double *tempCoordsZ, double *Gpu_COMX, double *Gpu_COMY,		double *Gpu_COMZ, SystemPotential *Gpu_Potential, double AcceptRand, int len,		double *dev_RecipSinSum, double *dev_RecipCosSum, double *dev_SinSumNew,		double *dev_CosSumNew, uint mIndex, XYZ *dev_newCOM, int box, double *dev_qqfact){		int start = Gpu_start[mIndex];		Gpu_result[0] = false;		dev_RealEnergyContrib[0] *= dev_qqfact[0];		if (AcceptRand < Gpu_BoltzW(beta, dev_EnergyContrib[0] + dev_RealEnergyContrib[0]									   + dev_RecipEnergyContrib[0])) {			Gpu_result[0] = true;			int xCellPos;			int yCellPos;			int zCellPos;			int position;			int index = -1;			for (int i = 0; i < len; i++) {				/////////////////////////////////////////////////////////				// change the cells (take out the old and put the new)				xCellPos = ((int) Gpu_x[start + i] >> HALF_MICROCELL_DIM);				yCellPos = ((int) Gpu_y[start + i] >> HALF_MICROCELL_DIM);				zCellPos = ((int) Gpu_z[start + i] >> HALF_MICROCELL_DIM);				position = (zCellPos * CellsPerZDimension + yCellPos)						* CellsPerYDimension + xCellPos;// flat 3d to 1d				index = -1;				for (int j = 0; j < MAX_ATOMS_PER_CELL; j++) {					if (atomCells[(j * NumberOfCellsInBox + position)							+ cellOffset * MAX_ATOMS_PER_CELL]							== (start + i)) {						index = j;						break;					}				}				for (int j = index;						j < atomCountrs[position + cellOffset] - 1; j++) {					atomCells[(j * NumberOfCellsInBox + position)							+ cellOffset * MAX_ATOMS_PER_CELL] =							atomCells[((j + 1) * NumberOfCellsInBox									+ position)									+ cellOffset * MAX_ATOMS_PER_CELL];				}				atomCountrs[position + cellOffset] -= 1;				// add new coords to cells				Gpu_x[start + i] = tempCoordsX[i * 2];				Gpu_y[start + i] = tempCoordsY[i * 2];				Gpu_z[start + i] = tempCoordsZ[i * 2];				xCellPos = ((int) Gpu_x[start + i] >> HALF_MICROCELL_DIM);				yCellPos = ((int) Gpu_y[start + i] >> HALF_MICROCELL_DIM);				zCellPos = ((int) Gpu_z[start + i] >> HALF_MICROCELL_DIM);				position = (zCellPos * CellsPerZDimension + yCellPos)						* CellsPerYDimension + xCellPos;// flat 3d to 1d				atomCells[(atomCountrs[position + cellOffset]						* NumberOfCellsInBox + position)						+ cellOffset * MAX_ATOMS_PER_CELL] = start + i;				atomCountrs[position + cellOffset] += 1;			}			Gpu_COMX[mIndex] = dev_newCOM[0].x;			Gpu_COMY[mIndex] = dev_newCOM[0].y;			Gpu_COMZ[mIndex] = dev_newCOM[0].z;			Gpu_Potential->Add(box, dev_EnergyContrib[0],					dev_EnergyContrib[1]);			Gpu_Potential->AddReal(box, dev_RealEnergyContrib);			Gpu_Potential->AddRecip(box, dev_RecipEnergyContrib);			Gpu_Potential->Total();		}}__global__ void Gpu_AcceptOrReject(bool *Gpu_result, double *dev_EnergyContrib,		double *dev_RealEnergyContrib, double *dev_RecipEnergyContrib,		uint *Gpu_start, double beta, double *Gpu_x, double *Gpu_y,		double *Gpu_z, double *Gpu_COMX, double *Gpu_COMY,		double *Gpu_COMZ, SystemPotential *Gpu_Potential, double AcceptRand, int len,		double *dev_RecipSinSum, double *dev_RecipCosSum, double *dev_SinSumNew,		double *dev_CosSumNew, int mIndex, XYZ *dev_newCOM, int box, double *dev_Array,		double *dev_qqfact, uint step){		int start = Gpu_start[mIndex];		Gpu_result[0] = false;		dev_RealEnergyContrib[0] *= dev_qqfact[0];		double blotzw = Gpu_BoltzW(beta, dev_EnergyContrib[0] + dev_RealEnergyContrib[0]		                     + dev_RecipEnergyContrib[0]);		if(step == 306529||step == 306530||step == 306531)			printf("rand: %lf, blotzw: %lf, beta: %lf, inter: %lf, real: %lf, recip: %lf\n", AcceptRand, blotzw, beta,					dev_EnergyContrib[0], dev_RealEnergyContrib[0], dev_RecipEnergyContrib[0]);		if (AcceptRand < blotzw) {			Gpu_result[0] = true;			if(step == 306529||step == 306530||step == 306531){				if(Gpu_result[0])					printf("");			}			for (int i = 0; i < len; i++) {				Gpu_x[start + i] = dev_Array[i];				Gpu_y[start + i] = dev_Array[len + i];				Gpu_z[start + i] = dev_Array[2 * len + i];			}			Gpu_COMX[mIndex] = dev_newCOM[0].x;			Gpu_COMY[mIndex] = dev_newCOM[0].y;			Gpu_COMZ[mIndex] = dev_newCOM[0].z;			Gpu_Potential->Add(box, dev_EnergyContrib[0],					dev_EnergyContrib[1]);			Gpu_Potential->AddReal(box, dev_RealEnergyContrib);			Gpu_Potential->AddRecip(box, dev_RecipEnergyContrib);			Gpu_Potential->Total();		}}