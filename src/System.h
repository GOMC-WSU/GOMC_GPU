
#ifndef SYSTEM_H
#define SYSTEM_H

#include "EnsemblePreprocessor.h" 
#include "CalculateEnergy.h" 


//Member variables
#include "EnergyTypes.h"
#include "Coordinates.h"
#include "PRNG.h"
#include "BoxDimensions.h"
#include "MoleculeLookup.h"
#include "MoveSettings.h"


//Initialization variables
class Setup;
class StaticVals;
class MoveBase;

#define Min_CUDA_Major 6 //  
#define Min_CUDA_Minor 0 //  
#define Min_CC_Major 3 // min compute capability major
#define Min_CC_Minor 0 // min compute capability minor



#define BLOCK_DIM 8
#define BLOCK_SIZE BLOCK_DIM * BLOCK_DIM * BLOCK_DIM
#define MICROCELL_DIM 4
#define HALF_MICROCELL_DIM MICROCELL_DIM/2
#define NumberofOps 4

// micro cell list
#define MAX_ATOMS_PER_CELL 18
// conv cell list
#define MaxParticleInCell 84



class System
{
 public:
   explicit System(StaticVals& statics);

   void Init(Setup const& setupData);

   //Runs move, picked at random
   void ChooseAndRunMove(const uint step);

   void LoadDataToGPU();// 
   void FreeGPUDATA();//  
   void LoadMolsToCells();//
   uint step; //  
   inline int _ConvertSMVer2Cores(int major, int minor);// 
   void DeviceQuery(); //  
   void RunDisplaceMove(uint rejectState, uint majKind);//  
   void RunRotateMove(uint rejectState, uint majKind ); //  

   //cell list
   void RunDisplaceMoveUsingCellList(uint rejectState, uint majKind);//
   void RunRotateMoveUsingCellList(uint rejectState, uint majKind);//

   SystemPotential ConvCellListSystemTotalEnergy();
  

   #if ENSEMBLE == GEMC
   void RunVolumeMove(uint rejectState, uint majKind, System * sys);// 
   void RunVolumeMoveCell(uint rejectState, uint majKind, System * sys);//
    SystemPotential NewConvCellListSystemTotalEnergy(uint majKind,SystemPotential curpot);// for volume move 
	SystemPotential NewConvCellListSystemTotalEnergyOneBox(uint majKind, int bPick);// for NPT 
   #endif
   #if ENSEMBLE == GEMC || ENSEMBLE==GCMC
   void RunMolTransferMove(uint rejectState, uint majKind, System * sys);// 
   #endif

   //NOTE:
   //This must also come first... as subsequent values depend on obj.
   //That may be in here, i.e. Box Dimensions
   StaticVals & statV;

   //NOTE:
   //Important! These must come first, as other objects may depend
   //on their val for init!
   //Only include these variables if they vary for this ensemble...
#ifdef VARIABLE_VOLUME
   BoxDimensions boxDimensions;
#endif
//#ifdef  VARIABLE_PARTICLE_NUMBER //  
   MoleculeLookup molLookup;
   //TODO CellGrid grid;
//#endif

   //Use as we don't know where they are...
   BoxDimensions & boxDimRef;
   MoleculeLookup & molLookupRef;

   MoveSettings moveSettings;
   SystemPotential potential;
   Coordinates coordinates;
   COM com;

   int MaxTrialNumber;

   CalculateEnergy calcEnergy;
   PRNG prng;

   //Procedure to run once move is picked... can also be called directly for
   //debugging...
   void RunMove(uint majKind, double draw, const uint step);

   ~System();

   // conv cell  list 

   void LoadAtomsToCells();
   void  CreateAdjCellList();
    #if ENSEMBLE == GEMC
   void  CreateAdjCellListForScaledMols(uint majKind);
   void LoadAtomsToCellsVolumeMove(uint majKind);
   #endif

 private:
   void InitMoves();
   void PickMove(uint & kind, double & draw);
   uint SetParams(const uint kind, const double draw);
   uint Transform(const uint kind);
   void CalcEn(const uint kind);
  void Accept(const uint kind, const uint rejectState, const uint step);

   MoveBase * moves[mv::MOVE_KINDS_TOTAL];
};

#endif /*SYSTEM_H*/

