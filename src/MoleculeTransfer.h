/*******************************************************************************
GPU OPTIMIZED MONTE CARLO (GOMC) BETA 0.97 (GPU version)
Copyright (C) 2015  GOMC Group

A copy of the GNU General Public License can be found in the COPYRIGHT.txt
along with this program, also can be found at <http://www.gnu.org/licenses/>.
********************************************************************************/

#ifndef MOLCULETRANSFER_H
#define MOLCULETRANSFER_H

#if ENSEMBLE==GCMC || ENSEMBLE==GEMC

#include "MoveBase.h"
#include "cbmc/TrialMol.h"

//#define DEBUG_MOVES

class MoleculeTransfer : public MoveBase
{
 public:

   MoleculeTransfer(System &sys, StaticVals const& statV) : 
      ffRef(statV.forcefield), molLookRef(sys.molLookupRef), 
      MoveBase(sys, statV) {}

   virtual uint Prep(const double subDraw, const double movPerc);
   virtual uint Transform();
   virtual void CalcEn();
   virtual void Accept(const uint earlyReject, uint step);

   inline void AcceptGPU(const uint rejectState, uint step); //  
 private:
   double GetCoeff() const;
   uint GetBoxPairAndMol(const double subDraw, const double movPerc);
   MolPick molPick;
   uint sourceBox, destBox;
   uint pStart, pLen;
   uint molIndex, kindIndex;
   uint mOff;   
   double W_tc, oldVirial;
   cbmc::TrialMol oldMol, newMol;
   Intermolecular tcLose, tcGain;
   MoleculeLookup & molLookRef;
   Forcefield const& ffRef;
};

inline uint MoleculeTransfer::GetBoxPairAndMol
(const double subDraw, const double movPerc)
{
   uint state = prng.PickMolAndBoxPair(molIndex, kindIndex,mOff, sourceBox, destBox,
					  subDraw, movPerc);// 
 
   if ( state != mv::fail_state::NO_MOL_OF_KIND_IN_BOX)
   {
      pStart = pLen = 0;
      molRef.GetRangeStartLength(pStart, pLen, molIndex);
   }
   return state;
}

inline uint MoleculeTransfer::Prep(const double subDraw, const double movPerc)
{
   uint state = GetBoxPairAndMol(subDraw, movPerc);
   newMol = cbmc::TrialMol(molRef.kinds[kindIndex], boxDimRef, destBox);
   oldMol = cbmc::TrialMol(molRef.kinds[kindIndex], boxDimRef, sourceBox);
  



   int kindStart = molLookRef.boxAndKindStart[sourceBox * molLookRef.numKinds + kindIndex];
	mOff = mOff + kindStart;
	//mOff = mIndexOff;

	/*double * selectedX, * selectedY, * selectedZ;
	selectedX = (double *)malloc(sizeof (double) * pLen);
	selectedY = (double *)malloc(sizeof (double) * pLen);
	selectedZ = (double *)malloc(sizeof (double) * pLen);*/


	uint gpuPstart[1];
	cudaMemcpy(gpuPstart, &calcEnRef.Gpu_start[mOff], sizeof(uint) , cudaMemcpyDeviceToHost);
	cudaMemcpy(&coordCurrRef.x[pStart], &calcEnRef.Gpu_x[gpuPstart[0]], sizeof(double) * pLen, cudaMemcpyDeviceToHost);
	cudaMemcpy(&coordCurrRef.y[pStart], &calcEnRef.Gpu_y[gpuPstart[0]], sizeof(double) * pLen, cudaMemcpyDeviceToHost);
	cudaMemcpy(&coordCurrRef.z[pStart], &calcEnRef.Gpu_z[gpuPstart[0]], sizeof(double) * pLen, cudaMemcpyDeviceToHost);
    
	
	oldMol.SetCoords(coordCurrRef, pStart);

	oldMol.mOff = mOff;

	newMol.mOff= mOff;


	oldMol.molKindIndex= kindIndex;
	newMol.molKindIndex= kindIndex;


	oldMol.molLength = pLen;

	newMol.molLength = pLen;


	

	//for (int i = 0; i < pLen; i++) {
		/*coordCurrRef.x[pStart + i] = selectedX[i];
		coordCurrRef.y[pStart + i] = selectedY[i];
		coordCurrRef.z[pStart + i] = selectedZ[i];*/

		//printf("x=%f,y=%f,z=%f\n",coordCurrRef.x[pStart + i],coordCurrRef.y[pStart + i],coordCurrRef.z[pStart + i] );

	//	printf("x=%f,y=%f,z=%f\n",oldMol.tCoords.x[i],oldMol.tCoords.y[i],oldMol.tCoords.z[i] );

	//}




	//free(selectedX);
	//free(selectedY);
	//free(selectedZ);







   W_tc = 1.0;
   return state;
}


inline uint MoleculeTransfer::Transform()
{   oldVirial = calcEnRef.MoleculeVirial(molIndex, sourceBox);
   subPick = mv::GetMoveSubIndex(mv::MOL_TRANSFER, sourceBox);
   molRef.kinds[kindIndex].Build(oldMol, newMol, molIndex);

  /* for (int i=0;i< newMol.molLength;i++)
   {
   
   printf("x=%f,y=%f,z=%f\n", oldMol.tCoords.x[i],oldMol.tCoords.y[i],oldMol.tCoords.z[i] );
   }*/
	   
  // printf("new energy=%f\n",newMol.en.inter);
   


	 
   return mv::fail_state::NO_FAIL;
}

inline void MoleculeTransfer::CalcEn()
{
   if (ffRef.useLRC)
   {
      tcLose = calcEnRef.MoleculeTailChange(sourceBox, kindIndex, false);

	  //printf("energy lost=%f\n",tcLose.energy);
      tcGain = calcEnRef.MoleculeTailChange(destBox, kindIndex, true);
	  //printf("energy gain=%f\n",tcGain.energy);
      W_tc = exp(-1.0*ffRef.beta*(tcGain.energy + tcLose.energy));
   }
}

inline double MoleculeTransfer::GetCoeff() const
{
#if ENSEMBLE == GEMC
   return (double)(molLookRef.NumKindInBox(kindIndex, sourceBox)) /
      (double)(molLookRef.NumKindInBox(kindIndex, destBox) + 1) *
      boxDimRef.volume[destBox] * boxDimRef.volInv[sourceBox];
#elif ENSEMBLE == GCMC
   if (sourceBox == mv::BOX0) //Delete case
   {
      return (double)(molLookRef.NumKindInBox(kindIndex, sourceBox)) *
         boxDimRef.volInv[sourceBox] *
         exp(-beta * molRef.kinds[kindIndex].chemPot);
   }
   else //Insertion case
   {
      return boxDimRef.volume[destBox]/
         (double)(molLookRef.NumKindInBox(kindIndex, destBox)+1) *
         exp(beta * molRef.kinds[kindIndex].chemPot);
   }
#endif
}


inline void MoleculeTransfer::AcceptGPU(const uint rejectState, uint step)

{
	try {
		bool result;

		//If we didn't skip the move calculation

		if(rejectState == mv::fail_state::NO_FAIL) {
			double molTransCoeff = GetCoeff();
			double Wo = oldMol.GetWeight();
			double Wn = newMol.GetWeight();
			double Wrat = Wn / Wo * W_tc;
#ifndef NDEBUG_MOVES
			SanityCheck(pStart, pLen, bSrc);
#endif



			/* for (int i=0; i < pLen;i++)
	  {
	  printf("x=%f,y=%f,z=%f\n", newMol->GetCoords().x[i], newMol->GetCoords().y[i], newMol->GetCoords().z[i] );
	  
	  
	  }*/



			result = prng() < molTransCoeff * Wrat;

			if (result) {

				// printf("Accepted, mols in box 0=%d, mols in box1=%d, source=%d, dist=%d\n", molLookRef.NumInBox(0),molLookRef.NumInBox(1) ,bSrc,bDest);

				cudaMemcpy(calcEnRef.tmpx, calcEnRef.Gpu_x, sizeof(double) *calcEnRef.currentCoords.Count() , cudaMemcpyDeviceToHost);
				cudaMemcpy(calcEnRef.tmpy, calcEnRef.Gpu_y, sizeof(double) *calcEnRef.currentCoords.Count() , cudaMemcpyDeviceToHost);
				cudaMemcpy(calcEnRef.tmpz, calcEnRef.Gpu_z, sizeof(double) *calcEnRef.currentCoords.Count() , cudaMemcpyDeviceToHost);
				cudaMemcpy( calcEnRef.tmpCOMx, calcEnRef.Gpu_COMX, sizeof(double) *  comCurrRef.Count() , cudaMemcpyDeviceToHost);
				cudaMemcpy( calcEnRef.tmpCOMy, calcEnRef.Gpu_COMY, sizeof(double) *  comCurrRef.Count() , cudaMemcpyDeviceToHost);
				cudaMemcpy( calcEnRef.tmpCOMz, calcEnRef.Gpu_COMZ, sizeof(double) *  comCurrRef.Count() , cudaMemcpyDeviceToHost);

				cudaDeviceSynchronize();
				cudaError_t  code = cudaGetLastError();

				if (code != cudaSuccess) {
					printf ("Cuda error end of coords and com -- %s\n", cudaGetErrorString(code));
				}

				int ctr = 0;
				int ctr1 = 0;
				MoleculeLookup::box_iterator tm = molLookRef.BoxBegin(0);
				MoleculeLookup::box_iterator en = molLookRef.BoxEnd(1);

				while(tm != en) {
					//  printf("x=%f,y=%f,z=%f\n", comCurrRef.x[*tm], comCurrRef.y[*tm],comCurrRef.z[*tm]);
					comCurrRef.x[*tm] = calcEnRef.tmpCOMx[ctr1];
					comCurrRef.y[*tm] = calcEnRef.tmpCOMy[ctr1];
					comCurrRef.z[*tm] = calcEnRef.tmpCOMz[ctr1];

					for(uint i = 0; i <  calcEnRef.mols.GetKind(*tm).NumAtoms(); ++i) {
						coordCurrRef.x[ calcEnRef.mols.start[*tm] + i ] = calcEnRef.tmpx[ctr ];
						coordCurrRef.y[ calcEnRef.mols.start[*tm] + i ] = calcEnRef.tmpy[ctr ];
						coordCurrRef.z[ calcEnRef.mols.start[*tm] + i ] = calcEnRef.tmpz[ctr ];
						ctr++;
					}

					ctr1++;
					++tm;
				}

				//cudaMemcpy(& sysPotRef, calcEnRef.Gpu_Potential, sizeof(SystemPotential)  , cudaMemcpyDeviceToHost);

				//memcpy(& sysPotRef, calcEnRef.Gpu_Potential, sizeof(SystemPotential) );
				cudaMemcpy(& sysPotRef, calcEnRef.Gpu_Potential, sizeof(SystemPotential)  , cudaMemcpyDeviceToHost);
				//Add tail corrections
				sysPotRef.boxEnergy[sourceBox].tc += tcLose.energy;
				sysPotRef.boxEnergy[destBox].tc += tcGain.energy;
				sysPotRef.boxVirial[sourceBox].tc += tcLose.virial;
				sysPotRef.boxVirial[destBox].tc += tcGain.virial;
				//Add rest of energy.
				sysPotRef.boxEnergy[sourceBox] -= oldMol.GetEnergy();
				sysPotRef.boxEnergy[destBox] += newMol.GetEnergy();
				//sysPotRef.boxVirial[bSrc].inter -= oldMol->GetVirial();
				//sysPotRef.boxVirial[bDest].inter += newMol->GetVirial();

				sysPotRef.boxVirial[sourceBox].inter -= calcEnRef.MoleculeVirial(molIndex, sourceBox);// replace with a GPU function 

				///////////////////////////////////////////////
		//		int offset;

		//		if (bSrc == 0)
		//		{ offset = 0; }
		//		else
		//		{ offset = calcEnRef.MolCount[0]; }


		//		int ThreadsPerBlock1 = 0;
		//		int BlocksPerGrid1 = 0;
		//		/*if (calcEnergy.MolCount[selectedBox] < MAXTHREADSPERBLOCK)

		//		ThreadsPerBlock1 = calcEnergy.MolCount[selectedBox];

		//		else*/
		//		ThreadsPerBlock1 = MAXTHREADSPERBLOCK;

		//		if(ThreadsPerBlock1 == 0)
		//		{ ThreadsPerBlock1 = 1; }
		//		
		//		BlocksPerGrid1 = ((calcEnRef.MolCount[bSrc]) + ThreadsPerBlock1 - 1) / ThreadsPerBlock1;

		//		if (BlocksPerGrid1 == 0)
		//		{ BlocksPerGrid1 = 1; }

		//		double * dev_EnergyContrib, * dev_VirialContrib;
		//		cudaMalloc((void**) &dev_EnergyContrib, 4 * BlocksPerGrid1 * sizeof(double));
		//		cudaMalloc((void**) &dev_VirialContrib, 4 * BlocksPerGrid1 * sizeof(double));

		//		
		//		Gpu_CalculateMoleculeVirial( calcEnRef.NoOfAtomsPerMol, calcEnRef.Gpu_atomKinds,
		//calcEnRef.Gpu_kIndex,
		//calcEnRef.Gpu_sigmaSq,
		//calcEnRef.Gpu_epsilon_cn,
		//calcEnRef.Gpu_nOver6,
		//calcEnRef.Gpu_epsilon_cn_6,
		//calcEnRef.Gpu_x,
		//calcEnRef.Gpu_y,
		//calcEnRef.Gpu_z,
		//calcEnRef.Gpu_COMX,
		//calcEnRef.Gpu_COMY,
		//calcEnRef.Gpu_COMZ,
		//Array,
		//len,
		//calcEnRef.Gpu_start,
		//boxDimRef.axis.x[bSrc],
		//boxDimRef.axis.y[bSrc],
		//boxDimRef.axis.z[bSrc],
		//boxDimRef.halfAx.x[bSrc],
		//boxDimRef.halfAx.y[bSrc],
		//boxDimRef.halfAx.z[bSrc],
		//offset,
		//calcEnRef.MolCount[bSrc],
		//mIndex,// mol index with offset
		//calcEnRef.forcefield.particles.NumKinds(),
		//boxDimRef.rCut,
		//molKInd,
		//boxDimRef.rCutSq,
		//dev_EnergyContrib,
		//dev_VirialContrib,
		//calcEnRef.Gpu_partn
		//);



				////////////////////////////////////////////////////////////////////////////


				

#ifndef NDEBUG_MOVES
				double oldBond = 0.0;
				double oldNonbond = 0.0;
				double newBond = 0.0;
				double newNonbond = 0.0;
				calcEnRef.MoleculeIntra(oldBond, oldNonbond, m, bSrc);
#endif
				newMol.GetCoords().CopyRange(coordCurrRef, 0, pStart, pLen);
				comCurrRef.SetNew(molIndex, destBox);
				molLookRef.ShiftMolBox(molIndex, sourceBox, destBox, kindIndex);
				ctr = 0;
				ctr1 = 0;
				uint sum = 0;
				tm = molLookRef.BoxBegin(0);
				en = molLookRef.BoxEnd(1);

				while(tm != en) {
					calcEnRef.tmpCOMx[ctr1] = comCurrRef.x[*tm];
					calcEnRef.tmpCOMy[ctr1] = comCurrRef.y[*tm];
					calcEnRef.tmpCOMz[ctr1] = comCurrRef.z[*tm];
					calcEnRef.tmpMolStart[ctr1] = sum;
					sum += calcEnRef.mols.GetKind(*tm).NumAtoms();
					calcEnRef.atmsPerMol[ctr1] = calcEnRef.mols.GetKind(*tm).NumAtoms();

					for(uint i = 0; i <  calcEnRef.mols.GetKind(*tm).NumAtoms(); ++i) {
						
						calcEnRef.tmpx[ctr ] = coordCurrRef.x[ calcEnRef.mols.start[*tm] + i ]  ;
						calcEnRef.tmpy[ctr ] = coordCurrRef.y[ calcEnRef.mols.start[*tm] + i ]  ;
						calcEnRef.tmpz[ctr ] = coordCurrRef.z[ calcEnRef.mols.start[*tm] + i ]  ;

						calcEnRef.CPU_atomKinds[ctr] =  calcEnRef.mols.GetKind(*tm).atomKind[i];
						ctr++;
					}

					ctr1++;
					++tm;
				}

				cudaMemcpy(calcEnRef.Gpu_start, calcEnRef.tmpMolStart,  sizeof(uint) * (calcEnRef.mols.count + 1) ,    cudaMemcpyHostToDevice);
				cudaMemcpy(calcEnRef.Gpu_COMX, calcEnRef.tmpCOMx, sizeof(double) *  comCurrRef.Count() , cudaMemcpyHostToDevice);
				cudaMemcpy(calcEnRef.Gpu_COMY, calcEnRef.tmpCOMy, sizeof(double) *  comCurrRef.Count() , cudaMemcpyHostToDevice);
				cudaMemcpy(calcEnRef.Gpu_COMZ, calcEnRef.tmpCOMz, sizeof(double) *  comCurrRef.Count() , cudaMemcpyHostToDevice);
				cudaMemcpy(calcEnRef.Gpu_x, calcEnRef.tmpx, sizeof(double) *calcEnRef.currentCoords.Count() , cudaMemcpyHostToDevice);
				cudaMemcpy(calcEnRef.Gpu_y, calcEnRef.tmpy, sizeof(double) *calcEnRef.currentCoords.Count() , cudaMemcpyHostToDevice);
				cudaMemcpy(calcEnRef.Gpu_z, calcEnRef.tmpz, sizeof(double) *calcEnRef.currentCoords.Count() , cudaMemcpyHostToDevice);
				cudaDeviceSynchronize();
				code = cudaGetLastError();

				if (code != cudaSuccess) {
					printf ("Cuda error end of coords and com to gpu  -- %s\n", cudaGetErrorString(code));
				}

				cudaMemcpy(calcEnRef.Gpu_atomKinds, calcEnRef.CPU_atomKinds, sizeof(uint) * (calcEnRef.currentCoords.Count()), cudaMemcpyHostToDevice );
				cudaMemcpy(calcEnRef.NoOfAtomsPerMol, calcEnRef.atmsPerMol, sizeof(uint) * (calcEnRef.mols.count), cudaMemcpyHostToDevice );
				cudaDeviceSynchronize();
				code = cudaGetLastError();

				if (code != cudaSuccess) {
					printf ("Cuda error end of atom kinds to gpu -- %s\n", cudaGetErrorString(code));
				}

				if (sourceBox == 0 ) {
					calcEnRef.AtomCount[0] -= pLen;
					calcEnRef.AtomCount[1] += pLen;
					calcEnRef.MolCount[0] -= 1;
					calcEnRef.MolCount[1] += 1;
				} else {
					calcEnRef.AtomCount[1] -= pLen;
					calcEnRef.AtomCount[0] += pLen;
					calcEnRef.MolCount[1] -= 1;
					calcEnRef.MolCount[0] += 1;
				}

				 sysPotRef.boxVirial[destBox].inter += calcEnRef.MoleculeVirial(molIndex, destBox);// replace with a GPU function 

				 sysPotRef.Total();

				//memcpy(calcEnRef.Gpu_Potential, &sysPotRef, sizeof(SystemPotential));
				cudaMemcpy(calcEnRef.Gpu_Potential, &sysPotRef, sizeof(SystemPotential)  , cudaMemcpyHostToDevice);
				cudaDeviceSynchronize();
				code = cudaGetLastError();

				if (code != cudaSuccess) {
					printf ("Cuda error end of energy move-- %s\n", cudaGetErrorString(code));
				}


#ifndef NDEBUG_MOVES
				calcEnRef.MoleculeIntra(newBond, newNonbond, m, bDest);
				std::cout << "OldMol " << oldMol->GetEnergy()
					<< "NewMol " << newMol->GetEnergy();
				std::cout << "OldIntra: B: " << oldBond << "\tNB: " << oldNonbond
					<< '\n';
				std::cout << "NewIntra: B: " << newBond << "\tNB: " << newNonbond
					<< '\n';
				SanityCheck(pStart, pLen, bDest);
#endif
			}

			////Clean up.
			//delete oldMol;
			//oldMol = NULL;
			//delete newMol;
			//newMol = NULL;
		} else //else we didn't even try because we knew it would fail
		{ result = false; }
		 subPick = mv::GetMoveSubIndex(mv::MOL_TRANSFER, sourceBox);
		moveSetRef.Update(result, subPick,step);
	} catch
		( const std::exception & ex ) {
			std::cout << ex.what() << std::endl;
			getchar();
	}
}



inline void MoleculeTransfer::Accept(const uint rejectState, uint step )
{
   bool result;
   //If we didn't skip the move calculation
   if(rejectState == mv::fail_state::NO_FAIL)
   {
      double molTransCoeff = GetCoeff();
      double Wo = oldMol.GetWeight();
      double Wn = newMol.GetWeight();
      double Wrat = Wn / Wo * W_tc;

      result = prng() < molTransCoeff * Wrat;
      if (result)
      {
         //std::cout << "ACCEPTED\n";
         //Add tail corrections
         sysPotRef.boxEnergy[sourceBox].tc += tcLose.energy;
         sysPotRef.boxEnergy[destBox].tc += tcGain.energy;
         sysPotRef.boxVirial[sourceBox].tc += tcLose.virial;
         sysPotRef.boxVirial[destBox].tc += tcGain.virial;

         //Add rest of energy.
         sysPotRef.boxEnergy[sourceBox] -= oldMol.GetEnergy();
         sysPotRef.boxEnergy[destBox] += newMol.GetEnergy();
        // double oldVirial = calcEnRef.MoleculeVirial(molIndex, sourceBox);
         sysPotRef.boxVirial[sourceBox].inter -= oldVirial;
        sysPotRef.Total();
#ifdef DEBUG_MOVES
         double oldBond = 0.0;
         double oldNonbond = 0.0;
         double newBond = 0.0;
         double newNonbond = 0.0;
         calcEnRef.MoleculeIntra(oldBond, oldNonbond, molIndex, sourceBox);
         double oldInter = calcEnRef.CheckMoleculeInter(molIndex, sourceBox);
#endif

         newMol.GetCoords().CopyRange(coordCurrRef, 0, pStart, pLen);
         comCurrRef.SetNew(molIndex, destBox);
         molLookRef.ShiftMolBox(molIndex, sourceBox, destBox, kindIndex);

         double newVirial = calcEnRef.MoleculeVirial(molIndex, destBox);
         sysPotRef.boxVirial[destBox].inter += newVirial;
         
#ifdef DEBUG_MOVES
         double newInter = calcEnRef.CheckMoleculeInter(molIndex, destBox);
         calcEnRef.MoleculeIntra(newBond, newNonbond, molIndex, destBox);
         std::cout << "Energy from CBMC:\n";
         std::cout << "OldMol " << oldMol.GetEnergy()
            << "NewMol " << newMol.GetEnergy();
         std::cout << "Energy from Recalculation:\n";
         std::cout << "Old Inter: " << oldInter << "\tIntraB: " << oldBond << "\tNB: " << oldNonbond
            << '\n';
         std::cout << "New Inter: " << newInter << "\tIntraB: " << newBond << "\tNB: " << newNonbond
            << "\n\n";
#endif
      }
   }
   else  //else we didn't even try because we knew it would fail
      result = false;
    subPick = mv::GetMoveSubIndex(mv::MOL_TRANSFER, sourceBox);
   moveSetRef.Update(result, subPick,step);
}

#endif

#endif


