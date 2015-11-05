
#ifndef TRANSFORMABLE_BASE_H
#define TRANSFORMABLE_BASE_H

#include "../lib/BasicTypes.h" //For uint.
#include "Molecules.h" //For start
#include "BoxDimensions.h" //For pbc wrapping
#include "XYZArray.h" //Parent class
#include "MoveSettings.h"
#include "Coordinates.h"
#include "EnergyTypes.h"
#include "COM.h"
#include "MoveConst.h"
#include "System.h"
#include "StaticVals.h"
#include "CalculateEnergy.h"
#include "MolPick.h"
#include "Forcefield.h"

#define NDEBUG_MOVES




class MoveBase
{
public:

	MoveBase(System & sys, StaticVals const& statV) :
	  boxDimRef(sys.boxDimRef), moveSetRef(sys.moveSettings), 
		  sysPotRef(sys.potential),
		  calcEnRef(sys.calcEnergy), comCurrRef(sys.com), 
		  coordCurrRef(sys.coordinates), prng(sys.prng), molRef(statV.mol), 
		  beta(statV.forcefield.beta)
	  {}

	  //Based on the random draw, determine the move kind, box, and 
	  //(if necessary) molecule kind.
	  virtual uint Prep(const double subDraw, const double movPerc) = 0;

	  //Note, in general this function is responsible for generating the new
	  //configuration to test.
	  virtual uint Transform() = 0;

	  //In general, this function is responsible for calculating the
	  //energy of the system for the new trial configuration.
	  virtual void CalcEn() = 0;

	  //This function carries out actions based on the internal acceptance state.
	  virtual void Accept(const uint rejectState, uint step) = 0;



	  // protected: //  
	  uint subPick;
	  //If a single molecule move, this is set by the target.
	  MoveSettings & moveSetRef;
	  SystemPotential & sysPotRef;
	  Coordinates & coordCurrRef;
	  COM & comCurrRef;
	  CalculateEnergy & calcEnRef;
	  PRNG & prng;
	  BoxDimensions & boxDimRef;
	  Molecules const& molRef;
	  double beta;
};

//Data needed for transforming a molecule's position via inter or intrabox 
//moves.
class MolTransformBase

{ 

public : 
	uint Getm()

	{
		return m;
	}

	uint Getmk()

	{
		return mk;
	}

	uint Getb()

	{
		return b;
	}

	uint GetmOff()

	{
		return mOff;
	}


protected:
	uint GetBoxAndMol(PRNG & prng, Molecules const& molRef,
		const double subDraw, const double movPerc);
	void ReplaceWith(MolTransformBase const& other);

	//Box, molecule, and molecule kind
	uint b, m, mk;
	uint pStart, pLen;

	uint mOff;  

	//Position
	XYZArray newMolPos; 
};

inline uint MolTransformBase::GetBoxAndMol
	(PRNG & prng, Molecules const& molRef,
	const double subDraw, const double movPerc)
{ 
#if ENSEMBLE == GCMC
	b = mv::BOX0;
	uint state = prng.PickMol(m, mk, b, subDraw, movPerc,mOff);// add mOff later
	//uint state =prng.PickMolAndBox(m, mk, b, mOff, subDraw, movPerc);
#else
	uint state =prng.PickMolAndBox(m, mk, b, mOff, subDraw, movPerc);
#endif
	pStart = pLen = 0;
	molRef.GetRangeStartLength(pStart, pLen, m);
	newMolPos.Uninit();
	newMolPos.Init(pLen);
	return state;
}

inline void MolTransformBase::ReplaceWith(MolTransformBase const& other)
{
	m = other.m;
	mk = other.mk;
	b = other.b;
	mOff= other.mOff; 
	pStart = other.pStart;
	pLen = other.pLen;
	newMolPos = other.newMolPos;
}



class Rotate;

class Translate : public MoveBase, public MolTransformBase
{
public:

	Translate(System &sys, StaticVals const& statV) : MoveBase(sys, statV) {}

	virtual uint Prep(const double subDraw, const double movPerc);
	uint ReplaceRot(Rotate const& other);
	virtual uint Transform();
	virtual void CalcEn();
	virtual void Accept(const uint rejectState, uint step);
	inline void AcceptGPU(const uint rejectState, bool Gpuresult, uint step);
private:
	Intermolecular inter;
	XYZ newCOM;
};
inline void Translate::AcceptGPU(const uint rejectState, bool Gpuresult, uint step)

{

	bool result = (rejectState == mv::fail_state::NO_FAIL) &&
		Gpuresult;

	subPick = mv::GetMoveSubIndex(mv::DISPLACE, b);
	moveSetRef.Update(result, subPick,step);
}
inline uint Translate::Prep(const double subDraw, const double movPerc) 
{ return GetBoxAndMol(prng, molRef, subDraw, movPerc); }

inline uint Translate::Transform()
{
	subPick = mv::GetMoveSubIndex(mv::DISPLACE, b);

	return mv::fail_state::NO_FAIL;
}

inline void Translate::CalcEn()
{ inter = calcEnRef.MoleculeInter(newMolPos, m, b, &newCOM); }

inline void Translate::Accept(const uint rejectState,uint step)
{
	bool res =false;
	if (rejectState == mv::fail_state::NO_FAIL)
	{
		double pr = prng();
		res = pr < exp(-beta * inter.energy);


	}
	bool result = (rejectState == mv::fail_state::NO_FAIL) && res;


	if (result)
	{
		//Set new energy.
		sysPotRef.Add(b, inter);
		sysPotRef.Total();
		//Copy coords
		newMolPos.CopyRange(coordCurrRef, 0, pStart, pLen);	       
		comCurrRef.Set(m, newCOM);
	}
	subPick = mv::GetMoveSubIndex(mv::DISPLACE, b);
	moveSetRef.Update(result, subPick,step); 
}

class Rotate : public MoveBase, public MolTransformBase
{
public:
	Rotate(System &sys, StaticVals const& statV) : MoveBase(sys, statV) {}

	virtual uint Prep(const double subDraw, const double movPerc);
	virtual uint Transform();
	virtual void CalcEn();
	virtual void Accept(const uint earlyReject, uint step);
	void AcceptGPU(const uint rejectState, bool Gpuresult, uint step );// 
private:
	Intermolecular inter;
};

inline uint Rotate::Prep(const double subDraw, const double movPerc) 
{ 
	uint state = GetBoxAndMol(prng, molRef, subDraw, movPerc); 
	if (state == mv::fail_state::NO_FAIL && molRef.NumAtoms(mk)  <= 1)
		state = mv::fail_state::ROTATE_ON_SINGLE_ATOM;
	return state;
}

inline uint Translate::ReplaceRot(Rotate const& other)
{
	ReplaceWith(other);
	return mv::fail_state::NO_FAIL;
}

inline uint Rotate::Transform()
{
	subPick = mv::GetMoveSubIndex(mv::ROTATE, b);

	return mv::fail_state::NO_FAIL;
}

inline void Rotate::CalcEn()
{ inter = calcEnRef.MoleculeInter(newMolPos, m, b); }

inline void Rotate::Accept(const uint rejectState, uint step )
{

	bool res =false;
	if (rejectState == mv::fail_state::NO_FAIL)
	{
		double pr = prng();
		res = pr < exp(-beta * inter.energy);


	}
	bool result = (rejectState == mv::fail_state::NO_FAIL) && res;

	if (result)
	{
		//Set new energy.
		sysPotRef.Add(b, inter);
		sysPotRef.Total();

		//Copy coords
		newMolPos.CopyRange(coordCurrRef, 0, pStart, pLen);
	}
	subPick = mv::GetMoveSubIndex(mv::ROTATE, b);
	moveSetRef.Update(result, subPick,step);
}

inline void Rotate::AcceptGPU(const uint rejectState, bool Gpuresult, uint step)

{
	bool result = (rejectState == mv::fail_state::NO_FAIL) &&
		Gpuresult;


	subPick = mv::GetMoveSubIndex(mv::ROTATE, b);
	moveSetRef.Update(result, subPick,step);
}




#if ENSEMBLE == GEMC

class VolumeTransfer : public MoveBase
{
public:
	VolumeTransfer(System &sys, StaticVals const& statV);

	virtual uint Prep(const double subDraw, const double movPerc);
	virtual void CalcEn();
	virtual uint Transform();
	double GetCoeff() const;
	virtual void Accept(const uint rejectState ,uint step );
	inline void AcceptGPU(const uint rejectState, SystemPotential newPot , SystemPotential curPot, uint bPick, uint step,System * sys);//  

	double scaleO, scaleN;
	double scaleP;

	double randN;


	// private: // 
	uint bPick; //note: This is only used for GEMC-NPT
	uint b_i, b_ii;
	SystemPotential sysPotNew;
	BoxDimensions newDim;
	Coordinates newMolsPos;
	COM newCOMs;
	MoleculeLookup & molLookRef;
	const uint GEMC_KIND;
	const double PRESSURE;
};

inline VolumeTransfer::VolumeTransfer(System &sys, StaticVals const& statV)  : 
MoveBase(sys, statV), molLookRef(sys.molLookupRef),
	newDim(sys.boxDimRef), newMolsPos(boxDimRef, newCOMs, sys.molLookupRef,
	sys.prng, statV.mol),
	newCOMs(sys.boxDimRef, newMolsPos, sys.molLookupRef, statV.mol),
	GEMC_KIND(statV.kindOfGEMC), PRESSURE(statV.pressure)
{
	newMolsPos.Init(sys.coordinates.Count());
	newCOMs.Init(statV.mol.count);
}

inline uint VolumeTransfer::Prep(const double subDraw, const double movPerc) 
{ 
	uint state = mv::fail_state::NO_FAIL;

	if(GEMC_KIND == mv::GEMC_NVT)
	{
		subPick = mv::GetMoveSubIndex(mv::VOL_TRANSFER);

	}
	if(GEMC_KIND == mv::GEMC_NPT)
	{
		prng.PickBox(bPick, subDraw, movPerc);
		subPick = mv::GetMoveSubIndex(mv::VOL_TRANSFER, bPick);
	}

	subPick = movPerc;
	b_i = mv::BOX0;
	b_ii = mv::BOX1;

	newDim = boxDimRef;
	coordCurrRef.CopyRange(newMolsPos,0,0,coordCurrRef.Count());
	comCurrRef.CopyRange(newCOMs, 0, 0, comCurrRef.Count());
	return state;
}

inline uint VolumeTransfer::Transform()
{
	uint state = mv::fail_state::NO_FAIL;
	if(GEMC_KIND == mv::GEMC_NVT)
	{
		subPick = mv::GetMoveSubIndex(mv::VOL_TRANSFER);
	}
	else
	{
		subPick = mv::GetMoveSubIndex(mv::VOL_TRANSFER, bPick);
	}
	double max = moveSetRef.Scale(subPick);

	if(GEMC_KIND == mv::GEMC_NVT)
	{
		coordCurrRef.VolumeTransferTranslate(state, newMolsPos, newCOMs, newDim,
			comCurrRef, max, scaleO, scaleN, randN); 
	}
	else
	{
		double scale = 0.0, delta = prng.Sym(max);
		state = boxDimRef.ShiftVolume(newDim, scale, bPick, delta);

		scaleP = scale;

	}


	return state;
}

inline void VolumeTransfer::CalcEn()
{ 


	if(GEMC_KIND == mv::GEMC_NVT)

		sysPotNew = calcEnRef.SystemNonbonded(sysPotRef, newMolsPos, newCOMs, newDim);
	else
		sysPotNew = calcEnRef.BoxNonbonded(sysPotRef, bPick, newMolsPos, newCOMs, newDim);
}

inline double VolumeTransfer::GetCoeff() const
{

	double coeff = 1.0;
	if (GEMC_KIND == mv::GEMC_NVT)
	{
		for (uint b = 0; b < BOX_TOTAL; ++b)
		{
			coeff *= pow(newDim.volume[b]/boxDimRef.volume[b],
				(double)molLookRef.NumInBox(b));
		}
	}
	else
	{
		coeff *= pow(newDim.volume[bPick]/boxDimRef.volume[bPick],
			(double)molLookRef.NumInBox(bPick))*
			exp(-beta * PRESSURE * (newDim.volume[bPick]-boxDimRef.volume[bPick]));
	}
	return coeff;
}

inline void VolumeTransfer::Accept(const uint rejectState, uint step)
{
	double volTransCoeff = GetCoeff();
	bool result = (rejectState == mv::fail_state::NO_FAIL) &&
		prng() < volTransCoeff * exp(-beta * (sysPotNew.Total() - sysPotRef.Total()));
	if (result)
	{
		//Set new energy.
		sysPotRef = sysPotNew;
		//Swap... next time we'll use the current members.
		swap(coordCurrRef, newMolsPos);
		swap(comCurrRef, newCOMs);
		boxDimRef = newDim;
	}

	if (GEMC_KIND == mv::GEMC_NVT)
	{
		subPick = mv::GetMoveSubIndex(mv::VOL_TRANSFER);
	}
	if (GEMC_KIND == mv::GEMC_NPT)
	{
		subPick = mv::GetMoveSubIndex(mv::VOL_TRANSFER, bPick);
	}


	moveSetRef.Update(result, subPick,step);
}



inline void VolumeTransfer::AcceptGPU(const uint rejectState, SystemPotential newPot , SystemPotential curPot, uint bPick,uint step, System * sys )

{


	double volTransCoeff = GetCoeff();

	double uBoltz = exp(-beta * (newPot.Total() - sysPotRef.Total()));
	double accept = volTransCoeff * uBoltz;

	bool result = (rejectState == mv::fail_state::NO_FAIL) && prng() < accept;


	if (result) {
		sysPotRef = newPot;



		if (GEMC_KIND == mv::GEMC_NVT)
		{
			// copy new coords to the GPU
			cudaMemcpy(calcEnRef.Gpu_x, calcEnRef.newX, sizeof(double) *calcEnRef.currentCoords.Count() ,    cudaMemcpyDeviceToDevice);
			cudaMemcpy(calcEnRef.Gpu_y, calcEnRef.newY, sizeof(double) *calcEnRef.currentCoords.Count() ,    cudaMemcpyDeviceToDevice);
			cudaMemcpy(calcEnRef.Gpu_z, calcEnRef.newZ, sizeof(double) *calcEnRef.currentCoords.Count() ,    cudaMemcpyDeviceToDevice);
			// copy new COM to GPU
			cudaMemcpy(calcEnRef.Gpu_COMX, calcEnRef.newCOMX, sizeof(double) *  comCurrRef.Count() , cudaMemcpyDeviceToDevice);
			cudaMemcpy(calcEnRef.Gpu_COMY, calcEnRef.newCOMY, sizeof(double) *  comCurrRef.Count() , cudaMemcpyDeviceToDevice);
			cudaMemcpy(calcEnRef.Gpu_COMZ, calcEnRef.newCOMZ, sizeof(double) *  comCurrRef.Count() , cudaMemcpyDeviceToDevice);

		}
		else
		{
			if(bPick == 0)
			{
				cudaMemcpy(calcEnRef.Gpu_x, calcEnRef.newX, sizeof(double) *calcEnRef.AtomCount[0] ,    cudaMemcpyDeviceToDevice);
				cudaMemcpy(calcEnRef.Gpu_y, calcEnRef.newY, sizeof(double) *calcEnRef.AtomCount[0] ,    cudaMemcpyDeviceToDevice);
				cudaMemcpy(calcEnRef.Gpu_z, calcEnRef.newZ, sizeof(double) *calcEnRef.AtomCount[0] ,    cudaMemcpyDeviceToDevice);
				// copy new COM to GPU
				cudaMemcpy(calcEnRef.Gpu_COMX, calcEnRef.newCOMX, sizeof(double) *  calcEnRef.MolCount[0] , cudaMemcpyDeviceToDevice);
				cudaMemcpy(calcEnRef.Gpu_COMY, calcEnRef.newCOMY, sizeof(double) *  calcEnRef.MolCount[0] , cudaMemcpyDeviceToDevice);
				cudaMemcpy(calcEnRef.Gpu_COMZ, calcEnRef.newCOMZ, sizeof(double) *  calcEnRef.MolCount[0] , cudaMemcpyDeviceToDevice);
			}
			else
			{
				cudaMemcpy(calcEnRef.Gpu_x + calcEnRef.AtomCount[0],calcEnRef.newX + calcEnRef.AtomCount[0],sizeof(double) *calcEnRef.AtomCount[1] ,    cudaMemcpyDeviceToDevice);
				cudaMemcpy(calcEnRef.Gpu_y + calcEnRef.AtomCount[0],calcEnRef.newY + calcEnRef.AtomCount[0],sizeof(double) *calcEnRef.AtomCount[1] ,    cudaMemcpyDeviceToDevice);
				cudaMemcpy(calcEnRef.Gpu_z + calcEnRef.AtomCount[0],calcEnRef.newZ + calcEnRef.AtomCount[0],sizeof(double) *calcEnRef.AtomCount[1] ,    cudaMemcpyDeviceToDevice);
				// copy new COM to GPU
				cudaMemcpy(calcEnRef.Gpu_COMX + calcEnRef.MolCount[0], calcEnRef.newCOMX + calcEnRef.MolCount[0], sizeof(double) *  calcEnRef.MolCount[1] , cudaMemcpyDeviceToDevice);
				cudaMemcpy(calcEnRef.Gpu_COMY + calcEnRef.MolCount[0], calcEnRef.newCOMY + calcEnRef.MolCount[0], sizeof(double) *  calcEnRef.MolCount[1] , cudaMemcpyDeviceToDevice);
				cudaMemcpy(calcEnRef.Gpu_COMZ + calcEnRef.MolCount[0], calcEnRef.newCOMZ + calcEnRef.MolCount[0], sizeof(double) *  calcEnRef.MolCount[1] , cudaMemcpyDeviceToDevice);
			}
		}
		cudaMemcpy(calcEnRef.Gpu_Potential, &newPot, sizeof(SystemPotential)  , cudaMemcpyHostToDevice);
		boxDimRef = newDim;

		// micro cell list re-init

		cudaFree(calcEnRef.atomCountrs);
		cudaFree(calcEnRef.atomCells);

		sys->LoadMolsToCells();



	}
	if (GEMC_KIND == mv::GEMC_NVT)
	{
		subPick = mv::GetMoveSubIndex(mv::VOL_TRANSFER);
	}
	if (GEMC_KIND == mv::GEMC_NPT)
	{
		subPick = mv::GetMoveSubIndex(mv::VOL_TRANSFER, bPick);
	}
	moveSetRef.Update(result, subPick,step);
}



#endif

#endif /*TRANSFORMABLE_BASE_H*/

