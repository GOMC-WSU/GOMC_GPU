/*******************************************************************************
GPU OPTIMIZED MONTE CARLO (GOMC) BETA 0.97 (GPU version)
Copyright (C) 2015  GOMC Group

A copy of the GNU General Public License can be found in the COPYRIGHT.txt
along with this program, also can be found at <http://www.gnu.org/licenses/>.
********************************************************************************/

#include "EnsemblePreprocessor.h"
#include "System.h"

#include "CalculateEnergy.h"
#include "EnergyTypes.h"
#include "Setup.h"               //For source of setup data.
#include "ConfigSetup.h"         //For types directly read from config. file
#include "StaticVals.h"
#include "Molecules.h"           //For indexing molecules.
#include "MoveConst.h"           //For array of move objects.
#include "MoveBase.h"            //For move bases....
#include "MoleculeTransfer.h"


#include <cuda_runtime.h> 

System::System(StaticVals& statics) : 
   statV(statics),
#ifdef VARIABLE_VOLUME
   boxDimRef(boxDimensions),
#else
   boxDimRef(statics.boxDimensions),
#endif
#ifdef VARIABLE_PARTICLE_NUMBER
   molLookupRef(molLookup),
#else
   molLookupRef(statics.molLookup),
#endif
   prng(molLookupRef),
   coordinates(boxDimRef, com, molLookupRef, prng, statics.mol),
   com(boxDimRef, coordinates, molLookupRef, statics.mol),
   moveSettings(boxDimRef),
   calcEnergy(statics, *this) {}


   void System::LoadDataToGPU()

{
	uint numKinds = molLookupRef.GetNumKind();
	uint * numByBox = new uint [BOX_TOTAL];
	
		
	uint * numByKindBox = new uint [BOX_TOTAL * numKinds];
	uint * numAtomsByBox = new uint[BOX_TOTAL];
	molLookupRef.TotalAtomsMols(numByBox, numByKindBox, numAtomsByBox, calcEnergy.mols.kinds);

for (int i=0;i<BOX_TOTAL;i++)
	{calcEnergy.AtomCount[i] = numAtomsByBox[i];
	
    }

if (BOX_TOTAL==1)
	calcEnergy.MolCount[0] = calcEnergy.mols.count;

else
for (int i=0;i<BOX_TOTAL;i++)
	{
	calcEnergy.MolCount[i] = molLookupRef.NumInBox(i);
	
    }

	



	cudaMalloc((void**)&calcEnergy.Gpu_Potential, sizeof(SystemPotential));

	

	int count2 = calcEnergy.forcefield.particles.NumKinds() * calcEnergy.forcefield.particles.NumKinds();
	cudaMalloc((void**)&calcEnergy.Gpu_sigmaSq , sizeof(double) * count2);
	cudaMalloc((void**)&calcEnergy.Gpu_epsilon_cn , sizeof(double) * count2);
	cudaMalloc((void**)&calcEnergy.Gpu_epsilon_cn_6 , sizeof(double) * count2);
	cudaMalloc((void**)&calcEnergy.Gpu_nOver6 , sizeof(double) * count2);
	cudaMalloc((void**)&calcEnergy.Gpu_enCorrection , sizeof(double) * count2);
	cudaMalloc((void**)&calcEnergy.Gpu_virCorrection , sizeof(double) * count2);
	cudaMemcpy(calcEnergy.Gpu_sigmaSq, calcEnergy.forcefield.particles.sigmaSq, sizeof(double)*count2, cudaMemcpyHostToDevice);
	cudaMemcpy(calcEnergy.Gpu_epsilon_cn, calcEnergy.forcefield.particles.epsilon_cn, sizeof(double)*count2, cudaMemcpyHostToDevice);
	cudaMemcpy(calcEnergy.Gpu_epsilon_cn_6, calcEnergy.forcefield.particles.epsilon_cn_6, sizeof(double)*count2, cudaMemcpyHostToDevice);
	cudaMemcpy(calcEnergy.Gpu_nOver6, calcEnergy.forcefield.particles.nOver6, sizeof(double)*count2, cudaMemcpyHostToDevice);
	cudaMemcpy(calcEnergy.Gpu_enCorrection, calcEnergy.forcefield.particles.enCorrection, sizeof(double)* count2, cudaMemcpyHostToDevice);
	cudaMemcpy(calcEnergy.Gpu_virCorrection, calcEnergy.forcefield.particles.virCorrection, sizeof(double)*count2, cudaMemcpyHostToDevice);



#ifdef MIE_INT_ONLY
	cudaMalloc ( (void**)  &calcEnergy.Gpu_partn , sizeof(uint) * count2);
	cudaMemcpy(calcEnergy.Gpu_partn,    calcEnergy.forcefield.particles.n,  sizeof(uint) * count2  ,    cudaMemcpyHostToDevice);
#else
	cudaMalloc ( (void**)  &calcEnergy.Gpu_partn , sizeof(double) * count2);
	cudaMemcpy(calcEnergy.Gpu_partn,    calcEnergy.forcefield.particles.n,  sizeof(double) * count2  ,  cudaMemcpyHostToDevice);
#endif


	cudaMalloc ( (void**)  & calcEnergy.Gpu_x, sizeof(double) * calcEnergy.currentCoords.Count());
	cudaMalloc ( (void**)  & calcEnergy.Gpu_y, sizeof(double) * calcEnergy.currentCoords.Count());
	cudaMalloc ( (void**)  & calcEnergy.Gpu_z, sizeof(double) * calcEnergy.currentCoords.Count());
	cudaMemcpy(calcEnergy.Gpu_x, calcEnergy.currentCoords.x, sizeof(double) *calcEnergy.currentCoords.Count() ,  cudaMemcpyHostToDevice);
	cudaMemcpy(calcEnergy.Gpu_y, calcEnergy.currentCoords.y, sizeof(double) *calcEnergy.currentCoords.Count() ,  cudaMemcpyHostToDevice);
	cudaMemcpy(calcEnergy.Gpu_z, calcEnergy.currentCoords.z, sizeof(double) *calcEnergy.currentCoords.Count() ,  cudaMemcpyHostToDevice);



	#if ENSEMBLE == GEMC || ENSEMBLE == GCMC
	// for mol transfer
	calcEnergy.tmpx = (double*) malloc (sizeof(double) * calcEnergy.currentCoords.Count());
	calcEnergy.tmpy = (double*) malloc (sizeof(double) * calcEnergy.currentCoords.Count());
	calcEnergy.tmpz = (double*) malloc (sizeof(double) * calcEnergy.currentCoords.Count());


	calcEnergy.tmpCOMx = (double*) malloc (sizeof(double) * com.Count());
	calcEnergy.tmpCOMy = (double*) malloc  (sizeof(double) * com.Count());
	calcEnergy.tmpCOMz = (double*) malloc ( sizeof(double) * com.Count());
	calcEnergy.tmpMolStart = (uint*) malloc ( sizeof(uint) * (calcEnergy.mols.count + 1));
    #endif


	// mols data
	cudaMalloc ( (void**)  &calcEnergy.Gpu_start , sizeof(uint) * (calcEnergy.mols.count + 1));
	cudaMalloc ( (void**)  &calcEnergy.Gpu_kIndex , sizeof(uint) * calcEnergy.mols.resKindsCount);
	cudaMalloc ( (void**)  &calcEnergy.Gpu_countByKind , sizeof(uint) * calcEnergy.mols.kindsCount);
	cudaMalloc ( (void**)  &calcEnergy.Gpu_pairEnCorrections , sizeof(double) * calcEnergy.mols.kindsCount * calcEnergy.mols.kindsCount );
	cudaMalloc ( (void**)  &calcEnergy.Gpu_pairVirCorrections , sizeof(double) * calcEnergy.mols.kindsCount * calcEnergy.mols.kindsCount );
	cudaMemcpy(calcEnergy.Gpu_start,    calcEnergy.mols.start,  sizeof(uint) * (calcEnergy.mols.count + 1) ,   cudaMemcpyHostToDevice);
	cudaMemcpy(calcEnergy.Gpu_kIndex,   calcEnergy.mols.kIndex, sizeof(uint) * (calcEnergy.mols.resKindsCount) , cudaMemcpyHostToDevice);
	cudaMemcpy(calcEnergy.Gpu_countByKind,  calcEnergy.mols.countByKind,    sizeof(uint) * (calcEnergy.mols.kindsCount) ,    cudaMemcpyHostToDevice);
	cudaMemcpy(calcEnergy.Gpu_pairEnCorrections,    calcEnergy.mols.pairEnCorrections,  sizeof(double) * (calcEnergy.mols.kindsCount * calcEnergy.mols.kindsCount) , cudaMemcpyHostToDevice);
	cudaMemcpy(calcEnergy.Gpu_pairVirCorrections,   calcEnergy.mols.pairVirCorrections, sizeof(double) * (calcEnergy.mols.kindsCount * calcEnergy.mols.kindsCount) , cudaMemcpyHostToDevice);
	calcEnergy.CPU_atomKinds = (uint *) malloc (sizeof(uint) * (calcEnergy.currentCoords.Count()) );
	calcEnergy.atmsPerMol = (uint *) malloc (sizeof(uint) * (calcEnergy.mols.count) );
	int ctr = 0;

	for (int i = 0; i < calcEnergy.mols.count; i++) {
		calcEnergy.atmsPerMol[i] = calcEnergy.mols.kinds[calcEnergy.mols.kIndex[i]].numAtoms;

		for (int j = 0; j < calcEnergy.mols.kinds[calcEnergy.mols.kIndex[i]].numAtoms; j++ ) {
			calcEnergy.CPU_atomKinds[ctr] = calcEnergy.mols.kinds[calcEnergy.mols.kIndex[i]].atomKind[j];
			ctr++;
		}
	}

	cudaMalloc ( (void**)  &calcEnergy.Gpu_atomKinds , sizeof(uint) * (calcEnergy.currentCoords.Count()));
	cudaMemcpy(calcEnergy.Gpu_atomKinds, calcEnergy.CPU_atomKinds, sizeof(uint) * (calcEnergy.currentCoords.Count()), cudaMemcpyHostToDevice );
	cudaMalloc ( (void**)  &calcEnergy.NoOfAtomsPerMol , sizeof(uint) * (calcEnergy.mols.count));
	cudaMemcpy(calcEnergy.NoOfAtomsPerMol, calcEnergy.atmsPerMol, sizeof(uint) * (calcEnergy.mols.count), cudaMemcpyHostToDevice );
	
	

	cudaMalloc ( (void**)  &calcEnergy.Gpu_COMX , sizeof(double) * com.Count());
	cudaMalloc ( (void**)  &calcEnergy.Gpu_COMY , sizeof(double) * com.Count());
	cudaMalloc ( (void**)  &calcEnergy.Gpu_COMZ , sizeof(double) * com.Count());
	cudaMemcpy(calcEnergy.Gpu_COMX, com.x, sizeof(double) *  com.Count(), cudaMemcpyHostToDevice );
	cudaMemcpy(calcEnergy.Gpu_COMY, com.y, sizeof(double) *  com.Count() , cudaMemcpyHostToDevice);
	cudaMemcpy(calcEnergy.Gpu_COMZ, com.z, sizeof(double) *  com.Count(), cudaMemcpyHostToDevice );





	cudaMalloc ( (void **) &calcEnergy.Gpu_result, sizeof(bool) );

	#if ENSEMBLE == GEMC
	cudaMalloc ( (void**)  &calcEnergy.newCOMX , sizeof(double) * com.Count());
	cudaMalloc ( (void**)  &calcEnergy.newCOMY , sizeof(double) * com.Count());
	cudaMalloc ( (void**)  &calcEnergy.newCOMZ , sizeof(double) * com.Count());
	cudaMemcpy(calcEnergy.newCOMX, calcEnergy.Gpu_COMX, sizeof(double) *  com.Count() , cudaMemcpyDeviceToDevice);
	cudaMemcpy(calcEnergy.newCOMY, calcEnergy.Gpu_COMY, sizeof(double) *  com.Count() , cudaMemcpyDeviceToDevice);
	cudaMemcpy(calcEnergy.newCOMZ, calcEnergy.Gpu_COMZ, sizeof(double) *  com.Count() , cudaMemcpyDeviceToDevice);
	cudaMalloc ( (void**)  & calcEnergy.newX, sizeof(double) * calcEnergy.currentCoords.Count());
	cudaMalloc ( (void**)  & calcEnergy.newY, sizeof(double) * calcEnergy.currentCoords.Count());
	cudaMalloc ( (void**)  & calcEnergy.newZ, sizeof(double) * calcEnergy.currentCoords.Count());
	#endif


	// streams
	cudaStreamCreate(&calcEnergy.stream0);
	cudaStreamCreate(&calcEnergy.stream1);

	
	printf("Data load to GPU done!\n");
	cudaDeviceSynchronize();
	cudaError_t  code = cudaGetLastError();

	if (code != cudaSuccess) {
		printf ("Cuda error at end of data load to GPU -- %s\n", cudaGetErrorString(code));
		exit(2);
	}


	
}


void System::FreeGPUDATA()
{   // free CUDA variables 
	cudaFree(calcEnergy.Gpu_Potential);
	cudaFree(calcEnergy.Gpu_sigmaSq);
	cudaFree(calcEnergy.Gpu_epsilon_cn);
	cudaFree(calcEnergy.Gpu_epsilon_cn_6);
	cudaFree(calcEnergy.Gpu_nOver6);
	cudaFree(calcEnergy.Gpu_enCorrection);
	cudaFree(calcEnergy.Gpu_virCorrection);
	cudaFree(calcEnergy.Gpu_partn);
	cudaFree(calcEnergy.Gpu_x);
	cudaFree(calcEnergy.Gpu_y);
	cudaFree(calcEnergy.Gpu_z);
	cudaFree(calcEnergy.Gpu_start);
	cudaFree(calcEnergy.Gpu_kIndex);
	cudaFree(calcEnergy.Gpu_countByKind);
	cudaFree(calcEnergy.Gpu_pairEnCorrections);
	cudaFree(calcEnergy.Gpu_pairVirCorrections);
	cudaFree(calcEnergy.Gpu_atomKinds);
	cudaFree(calcEnergy.NoOfAtomsPerMol);
	cudaFree(calcEnergy.Gpu_COMX);
	cudaFree(calcEnergy.Gpu_COMY);
	cudaFree(calcEnergy.Gpu_COMZ);
	cudaFree(calcEnergy.Gpu_result);
	cudaFree(calcEnergy.newX);
	cudaFree(calcEnergy.newY);
	cudaFree(calcEnergy.newZ);
	cudaFree(calcEnergy.newCOMX);
	cudaFree(calcEnergy.newCOMY);
	cudaFree(calcEnergy.newCOMZ);
	// free streams
	cudaStreamDestroy(calcEnergy.stream0);
	cudaStreamDestroy(calcEnergy.stream1);




}

System::~System()
{  FreeGPUDATA();//  
   delete moves[mv::DISPLACE];
   delete moves[mv::ROTATE];
#if ENSEMBLE == GEMC
   delete moves[mv::VOL_TRANSFER];
#endif
#if ENSEMBLE == GEMC || ENSEMBLE == GCMC
   delete moves[mv::MOL_TRANSFER];
#endif
}

// Beginning of GPU Architecture definitions
inline int System::_ConvertSMVer2Cores(int major, int minor)
{
	// Defines for GPU Architecture types (using the SM version to determine the # of cores per SM)
	typedef struct {
		int SM; // 0xMm (hexidecimal notation), M = SM Major version, and m = SM minor version
		int Cores;
	} sSMtoCores;

	sSMtoCores nGpuArchCoresPerSM[] = {
		{ 0x10,  8 }, // Tesla Generation (SM 1.0) G80 class
		{ 0x11,  8 }, // Tesla Generation (SM 1.1) G8x class
		{ 0x12,  8 }, // Tesla Generation (SM 1.2) G9x class
		{ 0x13,  8 }, // Tesla Generation (SM 1.3) GT200 class
		{ 0x20, 32 }, // Fermi Generation (SM 2.0) GF100 class
		{ 0x21, 48 }, // Fermi Generation (SM 2.1) GF10x class
		{ 0x30, 192}, // Kepler Generation (SM 3.0) GK10x class
		{ 0x35, 192}, // Kepler Generation (SM 3.5) GK11x class
		{ 0x50, 128}, // Maxwell Generation (SM 5.0) GM10x class
		{   -1, -1 }
	};

	int index = 0;
	while (nGpuArchCoresPerSM[index].SM != -1) {
		if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor) ) {
			return nGpuArchCoresPerSM[index].Cores;
		}
		index++;
	}
	printf("MapSMtoCores undefined SMversion %d.%d!\n", major, minor);
	return -1;
}



void System::DeviceQuery(){
	printf("================================================================\n\n");
	//Get device info
	cudaDeviceProp prop;
	int count, driverVersion;
	size_t heapsize;

	cudaGetDeviceCount(&count);
	if (count < 1) {
		printf("Error: No GPUs found.\nSimulation will Terminate\n");
		exit(3); 
	} //end if

	printf(" --- GPU System Configuration ---\n");
	cudaDriverGetVersion(&driverVersion);
	printf("CUDA Version: %d.%d\n\n", driverVersion/1000, (driverVersion%100)/10);


	


	bool hasRequiredComputeCapability = false;

	for (int i = 0; i < count; i++) {
		cudaGetDeviceProperties(&prop, i);
		cudaDeviceGetLimit(&heapsize, cudaLimitMallocHeapSize);
		printf(" --- General Information for device %d ---\n", i);
		printf("Name: %s\n", prop.name);
		printf("Compute capability: %d.%d\n", prop.major, prop.minor);
		printf("Clock rate: %.2f GHz\n", prop.clockRate * 1e-6f);
		printf(" --- Memory information for device %d ---\n", i);
		printf("Total global mem: %.2f GBytes\n", prop.totalGlobalMem/(1048576.0f*1024.0f));
		printf("Total constant mem: %.2f KBytes\n", prop.totalConstMem/(1024.0f));
		printf("Total dynamic heap mem: %.2f MBytes\n", heapsize/(1024.0f*1024.0f));
		printf(" --- Information on cores for device %d ---\n", i);
		printf("Streaming Multiprocessor (SM) count: %d\n", prop.multiProcessorCount);
		printf("Number of cores per SM: %d\n", _ConvertSMVer2Cores(prop.major, prop.minor));
		printf("Total number of cores: %d\n", _ConvertSMVer2Cores(prop.major, prop.minor) * prop.multiProcessorCount);
		printf("Shared mem per SM: %.2f KBytes\n", prop.sharedMemPerBlock/1024.0f);
		printf("Registers per SM: %d\n", prop.regsPerBlock);
		printf("Threads in warp: %d\n", prop.warpSize);
		printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
		printf("Max thread dimensions: (%d, %d, %d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
		printf("Max grid dimensions: (%d, %d, %d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
		printf("<<<<<<<=>>>>>>>\n\n");

		if (prop.major>= Min_CC_Major && prop.minor >= Min_CC_Minor)
		{
		hasRequiredComputeCapability=true;
		}
	}

	

	printf("================================================================\n\n");

}
void System::Init(Setup const& set)
{
   prng.Init(set.prng.prngMaker.prng);
#ifdef VARIABLE_VOLUME
   boxDimensions.Init(set.config.in.restart, 
		      set.config.sys.volume, set.pdb.cryst,
		      statV.forcefield.rCut,
		      statV.forcefield.rCutSq);
#endif
#ifdef VARIABLE_PARTICLE_NUMBER
   molLookup.Init(statV.mol, set.pdb.atoms); 
#endif
   moveSettings.Init(statV);
   //Note... the following calls use box iterators, so must come after
   //the molecule lookup initialization, in case we're in a constant 
   //particle/molecule ensemble, e.g. NVT
   coordinates.InitFromPDB(set.pdb.atoms);
   com.CalcCOM();

   DeviceQuery();

   LoadDataToGPU();

   potential = calcEnergy.SystemTotal();
   printf("Total system energy at start =%f\n", potential.totalEnergy.total);
   printf("Total system virial at start =%f\n", potential.totalVirial.total);
  

    #if ENSEMBLE == GEMC
   cudaMemcpy(calcEnergy.newX, calcEnergy.Gpu_x, sizeof(double) * calcEnergy.currentCoords.Count(), cudaMemcpyDeviceToDevice);
   cudaMemcpy(calcEnergy.newY, calcEnergy.Gpu_y, sizeof(double) * calcEnergy.currentCoords.Count(), cudaMemcpyDeviceToDevice);
   cudaMemcpy(calcEnergy.newZ, calcEnergy.Gpu_z, sizeof(double) * calcEnergy.currentCoords.Count(), cudaMemcpyDeviceToDevice);
   cudaMemcpy(calcEnergy.newCOMX, calcEnergy.Gpu_COMX, sizeof(double) * calcEnergy.currentCOM.Count(), cudaMemcpyDeviceToDevice);
   cudaMemcpy(calcEnergy.newCOMY, calcEnergy.Gpu_COMY, sizeof(double) * calcEnergy.currentCOM.Count(), cudaMemcpyDeviceToDevice);
   cudaMemcpy(calcEnergy.newCOMZ, calcEnergy.Gpu_COMZ, sizeof(double) * calcEnergy.currentCOM.Count(), cudaMemcpyDeviceToDevice);
  
   #endif

 cudaMemcpy(calcEnergy.Gpu_Potential, &potential, sizeof(SystemPotential), cudaMemcpyHostToDevice );

   






   InitMoves();
}

void System::InitMoves()
{
   moves[mv::DISPLACE] = new Translate(*this, statV);
   moves[mv::ROTATE] = new Rotate(*this, statV);
#if ENSEMBLE == GEMC
   moves[mv::VOL_TRANSFER] = new VolumeTransfer(*this, statV);
#endif
#if ENSEMBLE == GEMC || ENSEMBLE == GCMC
   moves[mv::MOL_TRANSFER] = new MoleculeTransfer(*this, statV);
#endif
}

void System::ChooseAndRunMove(const uint step)
{
   double draw=0;
   uint majKind=0;
   PickMove(majKind, draw);
    RunMove(majKind, draw, step);
}
void System::PickMove(uint & kind, double & draw)
{ 
   prng.PickArbDist(kind, draw, statV.movePerc, statV.totalPerc, 
		    mv::MOVE_KINDS_TOTAL);
}

void System::RunMove(uint majKind, double draw,const uint step)
{
   double Uo = potential.totalEnergy.total; // why do we need this ? 
   ////return now if move targets molecule and there's none in that box.
   uint rejectState = SetParams(majKind, draw);
    

   //If single atom, redo move as displacement

	if (rejectState == mv::fail_state::ROTATE_ON_SINGLE_ATOM) {
		majKind = mv::DISPLACE;
		Translate * disp = static_cast<Translate *>(moves[mv::DISPLACE]);
		Rotate * rot = static_cast<Rotate *>(moves[mv::ROTATE]);
		rejectState = disp->ReplaceRot(*rot);
		}

	switch (majKind) {
	case 0:
		RunDisplaceMove(rejectState, majKind);
		break;

	case 1:
		RunRotateMove(rejectState, majKind);
		break;
    #if ENSEMBLE == GEMC
	case 2:
		RunVolumeMove(rejectState, majKind);
		break;
#endif
		 #if ENSEMBLE == GEMC || ENSEMBLE == GCMC
	case 3:
		RunMolTransferMove(rejectState, majKind);
		break;
    #endif
	}

   //If large change, recalculate the system energy to compensate for errors.
   /*if (potential.totalEnergy.total-Uo > 1e4)
      potential = calcEnergy.SystemTotal();*/
      
}
uint System::SetParams(const uint kind, const double draw) 
{ return moves[kind]->Prep(draw, statV.movePerc[kind]); }

uint System::Transform(const uint kind) { return moves[kind]->Transform(); }

void System::CalcEn(const uint kind) { moves[kind]->CalcEn(); }

void System::Accept(const uint kind, const uint rejectState, const uint step)
{ moves[kind]->Accept(rejectState,step); }

/////////////////////////////////////////

// GPU Code

// Author: Kamel Rushaidat

////////////////////////////////////////



void System::RunDisplaceMove(uint rejectState, uint majKind)

{
	if (rejectState == mv::fail_state::NO_FAIL)
	{ rejectState = Transform(majKind); }

	bool resultFromMove[1];
	resultFromMove[0] = false;


	
	if (rejectState == mv::fail_state::NO_FAIL ) {
		
		int selectedBox = ((Translate*)moves[majKind])->Getb();
		int offset;

		if (selectedBox == 0)
		{ offset = 0; }
		else
		{ offset = calcEnergy.MolCount[0]; }

		int ThreadsPerBlock1 = 0;
		int BlocksPerGrid1 = 0;
		/*if (calcEnergy.MolCount[selectedBox] < MAXTHREADSPERBLOCK)

		ThreadsPerBlock1 = calcEnergy.MolCount[selectedBox];

		else*/
		ThreadsPerBlock1 = MAXTHREADSPERBLOCK;

		if(ThreadsPerBlock1 == 0)
		{ ThreadsPerBlock1 = 1; }

		BlocksPerGrid1 = ((calcEnergy.MolCount[selectedBox]) + ThreadsPerBlock1 - 1) / ThreadsPerBlock1;

		if (BlocksPerGrid1 == 0)
		{ BlocksPerGrid1 = 1; }

		double * dev_EnergyContrib, * dev_VirialContrib;
		cudaMalloc((void**) &dev_EnergyContrib, 4 * BlocksPerGrid1 * sizeof(double));
		cudaMalloc((void**) &dev_VirialContrib, 4 * BlocksPerGrid1 * sizeof(double));


		double max = moveSettings.Scale(mv::GetMoveSubIndex(mv::DISPLACE, selectedBox));
		 XYZ shift = prng.SymXYZ(max);


		int len;
		len = calcEnergy.mols.kinds[ ((Translate*)moves[majKind])->Getmk()].numAtoms;
		int kindStart = molLookupRef.boxAndKindStart[selectedBox * molLookupRef.numKinds + ((Translate*)moves[majKind])->Getmk()];

		

		MTRand *r = prng.gen;
		double randToSend = (*r)();

	 //Intermolecular result;
   if (selectedBox < BOXES_WITH_U_NB)
   {

		TryTransformGpu <<< BlocksPerGrid1, ThreadsPerBlock1, len*3*sizeof(double)>>>(
			calcEnergy.NoOfAtomsPerMol, calcEnergy.Gpu_atomKinds, calcEnergy.Gpu_Potential,
			calcEnergy.Gpu_x,  calcEnergy.Gpu_y,  calcEnergy.Gpu_z,
			calcEnergy.Gpu_COMX,  calcEnergy.Gpu_COMY, calcEnergy.Gpu_COMZ,
			shift, boxDimRef.axis.x[selectedBox], boxDimRef.axis.y[selectedBox],  boxDimRef.axis.z[selectedBox],
			calcEnergy.Gpu_kIndex,
			calcEnergy.Gpu_sigmaSq,
			calcEnergy.Gpu_epsilon_cn,
			calcEnergy.Gpu_nOver6,
			calcEnergy.Gpu_epsilon_cn_6,
			moves[majKind]->beta,
			randToSend,
			calcEnergy.Gpu_start,
			len,
			boxDimRef.halfAx.x[selectedBox],
			boxDimRef.halfAx.y[selectedBox],
			boxDimRef.halfAx.z[selectedBox],
			offset,
			calcEnergy.MolCount[selectedBox],
			((Translate*)moves[majKind])->GetmOff() + kindStart,
			calcEnergy.forcefield.particles.NumKinds(),
			boxDimRef.rCut,
			((Translate*)moves[majKind])->Getmk(),
			boxDimRef.rCutSq,
			dev_EnergyContrib,
			dev_VirialContrib,
			selectedBox,
			calcEnergy.Gpu_result,
			calcEnergy.Gpu_partn
			);
		cudaMemcpy(resultFromMove, calcEnergy.Gpu_result, sizeof(bool), cudaMemcpyDeviceToHost);
		cudaMemcpy(&potential, calcEnergy.Gpu_Potential, sizeof(SystemPotential), cudaMemcpyDeviceToHost);

   }



		cudaFree (dev_EnergyContrib);
		cudaFree(dev_VirialContrib);
	}

	((Translate*)moves[majKind])->AcceptGPU(rejectState, resultFromMove[0],step );
	cudaDeviceSynchronize();
	cudaError_t  code = cudaGetLastError();

	if (code != cudaSuccess) {
		printf ("Cuda error at Displace run-- %s\n", cudaGetErrorString(code));
		exit(2);
	}

	
}
void System::RunRotateMove(uint rejectState, uint majKind )

{
	if (rejectState == mv::fail_state::NO_FAIL)
	{ rejectState = Transform(majKind); }

	bool resultFromMove[1];
	resultFromMove[0] = false;

	if (rejectState == mv::fail_state::NO_FAIL ) {
		
		int selectedBox = ((Rotate*)moves[majKind])->Getb();

	


		int offset;

		if (selectedBox == 0)
		{ offset = 0; }
		else
		{ offset = calcEnergy.MolCount[0]; }

		int ThreadsPerBlock1 = 0;
		int BlocksPerGrid1 = 0;
		/*if (calcEnergy.MolCount[selectedBox] < MAXTHREADSPERBLOCK)

		ThreadsPerBlock1 = calcEnergy.MolCount[selectedBox];

		else*/
		ThreadsPerBlock1 = MAXTHREADSPERBLOCK;

		if(ThreadsPerBlock1 == 0)
		{ ThreadsPerBlock1 = 1; }

		BlocksPerGrid1 = ((calcEnergy.MolCount[selectedBox]) + ThreadsPerBlock1 - 1) / ThreadsPerBlock1;

		if (BlocksPerGrid1 == 0)
		{ BlocksPerGrid1 = 1; }

		double * dev_EnergyContrib, * dev_VirialContrib;
		cudaMalloc((void**) &dev_EnergyContrib, 4 * BlocksPerGrid1 * sizeof(double));
		cudaMalloc((void**) &dev_VirialContrib, 4 * BlocksPerGrid1 * sizeof(double));
		
		int molLen = calcEnergy.mols.kinds[ ((Rotate*)moves[majKind])->Getmk()].numAtoms;
		
		double max = moveSettings.Scale(mv::GetMoveSubIndex(mv::ROTATE, selectedBox));

		double sym=prng.Sym(max);

		XYZ PickonSphere= prng.PickOnUnitSphere();

		RotationMatrix matrix = RotationMatrix::FromAxisAngle(sym,PickonSphere );

	
		
	
	
		int kindStart = molLookupRef.boxAndKindStart[selectedBox * molLookupRef.numKinds + ((Translate*)moves[majKind])->Getmk()];
		MTRand *r = prng.gen;
		double randToSend = (*r)();
		


		TryRotateGpu <<< BlocksPerGrid1, ThreadsPerBlock1, molLen* sizeof(double)*3>>>(
			calcEnergy.NoOfAtomsPerMol,
			calcEnergy.Gpu_atomKinds,
			calcEnergy.Gpu_Potential,
			matrix,
			calcEnergy.Gpu_x,
			calcEnergy.Gpu_y,
			calcEnergy.Gpu_z,
			calcEnergy.Gpu_COMX,
			calcEnergy.Gpu_COMY,
			calcEnergy.Gpu_COMZ,
			boxDimRef.axis.x[selectedBox],
			boxDimRef.axis.y[selectedBox],
			boxDimRef.axis.z[selectedBox],
			calcEnergy.Gpu_kIndex,
			calcEnergy.Gpu_sigmaSq,
			calcEnergy. Gpu_epsilon_cn,
			calcEnergy.Gpu_nOver6,
			calcEnergy.Gpu_epsilon_cn_6,
			moves[majKind]->beta,
			randToSend ,
			calcEnergy.Gpu_start,
			molLen,
			boxDimRef.halfAx.x[selectedBox],
			boxDimRef.halfAx.x[selectedBox],
			boxDimRef.halfAx.x[selectedBox],
			offset,
			calcEnergy.MolCount[selectedBox],
			((Rotate*)moves[majKind])->GetmOff() + kindStart,
			calcEnergy.forcefield.particles.NumKinds(),
			boxDimRef.rCut,
			((Rotate*)moves[majKind])->Getmk(),
			boxDimRef.rCutSq,
			dev_EnergyContrib,
			dev_VirialContrib,
			selectedBox,
			calcEnergy.Gpu_result,
			calcEnergy.Gpu_partn
			);
		cudaMemcpy(resultFromMove, calcEnergy.Gpu_result, sizeof(bool), cudaMemcpyDeviceToHost);
		cudaMemcpy(&potential, calcEnergy.Gpu_Potential, sizeof(SystemPotential), cudaMemcpyDeviceToHost);

		cudaFree (dev_EnergyContrib);
		cudaFree(dev_VirialContrib);
	}

	((Rotate*)moves[majKind])->AcceptGPU(rejectState, resultFromMove[0],step  );
	cudaDeviceSynchronize();
	cudaError_t  code = cudaGetLastError();

	if (code != cudaSuccess) {
		printf ("Cuda error at Rotate Move-- %s\n", cudaGetErrorString(code));
		exit(2);
	}
}

#if ENSEMBLE == GEMC
void System::RunVolumeMove(uint rejectState, uint majKind)

{
	if(((VolumeTransfer*)moves[majKind])->GEMC_KIND == mv::GEMC_NVT)
	{
		int srcBox, distBox;

		if (rejectState == mv::fail_state::NO_FAIL) {
			rejectState = Transform(majKind);
			int ThreadsPerBlock1 = 0;
			int BlocksPerGrid1 = 0;
			double scaleO, scaleN;
			double randN;
			srcBox = ((VolumeTransfer*)moves[majKind])->b_i;
			distBox = ((VolumeTransfer*)moves[majKind])->b_ii;
			scaleO = ((VolumeTransfer*)moves[majKind])->scaleO;
			scaleN = ((VolumeTransfer*)moves[majKind])->scaleN;
			randN = ((VolumeTransfer*)moves[majKind])->randN;

			if (rejectState == mv::fail_state::NO_FAIL) {
				if (calcEnergy.MolCount[srcBox] < MAXTHREADSPERBLOCK)
				{ ThreadsPerBlock1 = calcEnergy.MolCount[srcBox]; }
				else
				{ ThreadsPerBlock1 = MAXTHREADSPERBLOCK; }

				if(ThreadsPerBlock1 == 0)
				{ ThreadsPerBlock1 = 1; }

				BlocksPerGrid1 = ((calcEnergy.MolCount[srcBox]) + ThreadsPerBlock1 - 1) / ThreadsPerBlock1;

				if (BlocksPerGrid1 == 0)
				{ BlocksPerGrid1 = 1; }

				ScaleMolecules <<< BlocksPerGrid1, ThreadsPerBlock1, 0, calcEnergy.stream0>>>(calcEnergy.NoOfAtomsPerMol,
					calcEnergy.Gpu_kIndex, calcEnergy.Gpu_x, calcEnergy.Gpu_y,  calcEnergy.Gpu_z,
					calcEnergy.Gpu_COMX,  calcEnergy.Gpu_COMY,  calcEnergy.Gpu_COMZ,
					calcEnergy.newX, calcEnergy.newY, calcEnergy.newZ,
					calcEnergy.newCOMX, calcEnergy.newCOMY, calcEnergy.newCOMZ,
					scaleO, calcEnergy.MolCount[srcBox],
					((VolumeTransfer*)moves[majKind])->newDim.axis.x[srcBox],
					((VolumeTransfer*)moves[majKind])->newDim.axis.y[srcBox],
					((VolumeTransfer*)moves[majKind])->newDim.axis.z[srcBox],
					boxDimRef.axis.x[srcBox],
					boxDimRef.axis.y[srcBox],
					boxDimRef.axis.z[srcBox],
					boxDimRef.halfAx.x[srcBox],
					boxDimRef.halfAx.x[srcBox],
					boxDimRef.halfAx.x[srcBox],
					(srcBox == 0) ? 0 : calcEnergy.MolCount[0],
					calcEnergy.Gpu_start
					);
				cudaStreamSynchronize(calcEnergy.stream0);

				if (calcEnergy.MolCount[distBox] < MAXTHREADSPERBLOCK)
					ThreadsPerBlock1 = calcEnergy.MolCount[distBox];
				else
				{ ThreadsPerBlock1 = MAXTHREADSPERBLOCK; }
				if(ThreadsPerBlock1 == 0)
					ThreadsPerBlock1 = 1;
				BlocksPerGrid1 = ((calcEnergy.MolCount[distBox]) + ThreadsPerBlock1 - 1) / ThreadsPerBlock1; 
				if (BlocksPerGrid1 == 0) BlocksPerGrid1 = 1;
				ScaleMolecules <<< BlocksPerGrid1, ThreadsPerBlock1, 0, calcEnergy.stream1>>>(calcEnergy.NoOfAtomsPerMol,
					calcEnergy.Gpu_kIndex, calcEnergy.Gpu_x, calcEnergy.Gpu_y,  calcEnergy.Gpu_z,
					calcEnergy.Gpu_COMX,  calcEnergy.Gpu_COMY,  calcEnergy.Gpu_COMZ,
					calcEnergy.newX, calcEnergy.newY, calcEnergy.newZ,
					calcEnergy.newCOMX, calcEnergy.newCOMY, calcEnergy.newCOMZ,
					scaleN, calcEnergy.MolCount[distBox],
					((VolumeTransfer*)moves[majKind])->newDim.axis.x[distBox],
					((VolumeTransfer*)moves[majKind])->newDim.axis.y[distBox],
					((VolumeTransfer*)moves[majKind])->newDim.axis.z[distBox],
					boxDimRef.axis.x[distBox],
					boxDimRef.axis.y[distBox],
					boxDimRef.axis.z[distBox],
					boxDimRef.halfAx.x[distBox],
					boxDimRef.halfAx.x[distBox],
					boxDimRef.halfAx.x[distBox],
					(distBox == 0) ? 0 : calcEnergy.MolCount[0],
					calcEnergy.Gpu_start
					);
				cudaStreamSynchronize(calcEnergy.stream1);
			}
		}

		SystemPotential curpot ;
		SystemPotential newpot;

		if (rejectState == mv::fail_state::NO_FAIL ) {
			cudaMemcpy(& curpot, calcEnergy.Gpu_Potential, sizeof(SystemPotential)  , cudaMemcpyDeviceToHost);
			newpot =  calcEnergy.NewSystemInterGPU(  ((VolumeTransfer*)moves[majKind])->newDim, srcBox, distBox);
		}

		((VolumeTransfer*)moves[majKind])->AcceptGPU(rejectState, newpot, curpot, 0,step);
		cudaDeviceSynchronize();
		cudaError_t code = cudaGetLastError();

		if (code != cudaSuccess) {
			printf ("Cuda error at volume move-- %s, LINE: %d\n", cudaGetErrorString(code), __LINE__);
			exit(2);
		}
	}
	else
	{
		uint bPick;
		if (rejectState == mv::fail_state::NO_FAIL) {
			rejectState = Transform(majKind);
			int ThreadsPerBlock1 = 0;
			int BlocksPerGrid1 = 0;
			double scaleO, scaleN, scaleP;
			double randN;
			bPick  = ((VolumeTransfer*)moves[majKind])->bPick;
			scaleO = ((VolumeTransfer*)moves[majKind])->scaleO;
			scaleN = ((VolumeTransfer*)moves[majKind])->scaleN;
			scaleP = ((VolumeTransfer*)moves[majKind])->scaleP;
			randN  = ((VolumeTransfer*)moves[majKind])->randN;
			if (rejectState == mv::fail_state::NO_FAIL) {


				if (calcEnergy.MolCount[bPick] < MAXTHREADSPERBLOCK)
				{ ThreadsPerBlock1 = calcEnergy.MolCount[bPick]; }
				else
				{ ThreadsPerBlock1 = MAXTHREADSPERBLOCK; }

				if(ThreadsPerBlock1 == 0)
				{ ThreadsPerBlock1 = 1; }

				BlocksPerGrid1 = ((calcEnergy.MolCount[bPick]) + ThreadsPerBlock1 - 1) / ThreadsPerBlock1;

				if (BlocksPerGrid1 == 0)
				{ BlocksPerGrid1 = 1; }

				ScaleMolecules <<< BlocksPerGrid1, ThreadsPerBlock1, 0, calcEnergy.stream0>>>(calcEnergy.NoOfAtomsPerMol,
					calcEnergy.Gpu_kIndex, calcEnergy.Gpu_x, calcEnergy.Gpu_y,  calcEnergy.Gpu_z,
					calcEnergy.Gpu_COMX,  calcEnergy.Gpu_COMY,  calcEnergy.Gpu_COMZ,
					calcEnergy.newX, calcEnergy.newY, calcEnergy.newZ,
					calcEnergy.newCOMX, calcEnergy.newCOMY, calcEnergy.newCOMZ,
					scaleP, calcEnergy.MolCount[bPick],
					((VolumeTransfer*)moves[majKind])->newDim.axis.x[bPick],
					((VolumeTransfer*)moves[majKind])->newDim.axis.y[bPick],
					((VolumeTransfer*)moves[majKind])->newDim.axis.z[bPick],
					boxDimRef.axis.x[bPick],
					boxDimRef.axis.y[bPick],
					boxDimRef.axis.z[bPick],
					boxDimRef.halfAx.x[bPick],
					boxDimRef.halfAx.x[bPick],
					boxDimRef.halfAx.x[bPick],
					(bPick == 0) ? 0 : calcEnergy.MolCount[0],
					calcEnergy.Gpu_start
					);
				cudaStreamSynchronize(calcEnergy.stream0);
			}
		}

		SystemPotential curpot ;
		SystemPotential newpot;

		if (rejectState == mv::fail_state::NO_FAIL ) {
			cudaMemcpy(& curpot, calcEnergy.Gpu_Potential, sizeof(SystemPotential)  , cudaMemcpyDeviceToHost);
			newpot =  calcEnergy.NewSystemInterGPUOneBox(  ((VolumeTransfer*)moves[majKind])->newDim, bPick);
		}

		((VolumeTransfer*)moves[majKind])->AcceptGPU(rejectState, newpot, curpot, bPick,step);
		cudaDeviceSynchronize();
		cudaError_t code = cudaGetLastError();

		if (code != cudaSuccess) {
			printf ("Cuda error at volume move-- %s, LINE: %d\n", cudaGetErrorString(code), __LINE__);
			exit(2);
		}
	}
}

#endif

#if ENSEMBLE == GEMC || ENSEMBLE == GCMC
void System::RunMolTransferMove(uint rejectState, uint majKind)

{
	if (rejectState == mv::fail_state::NO_FAIL )
	{ rejectState = Transform(majKind); }

	if (rejectState == mv::fail_state::NO_FAIL ) {
		CalcEn(majKind);
	}

	((MoleculeTransfer*)moves[majKind])->AcceptGPU(rejectState, step);
	cudaDeviceSynchronize();
	cudaError_t  code = cudaGetLastError();

	if (code != cudaSuccess) {
		printf ("Cuda error at Molecule transfer Move -- %s\n", cudaGetErrorString(code));
		exit(2);
	}
}

#endif



