/*******************************************************************************
GPU OPTIMIZED MONTE CARLO (GOMC) BETA 0.97 (GPU version)
Copyright (C) 2015  GOMC Group

A copy of the GNU General Public License can be found in the COPYRIGHT.txt
along with this program, also can be found at <http://www.gnu.org/licenses/>.
********************************************************************************/

#ifndef SYSTEM_H
#define SYSTEM_H

#include "EnsemblePreprocessor.h" //For VARIABLE_<QUANTITY> conditional defines
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
class System
{
 public:
   explicit System(StaticVals& statics);

   void Init(Setup const& setupData);

   //Runs move, picked at random
   void ChooseAndRunMove(const uint step);

   void LoadDataToGPU();// 
   void FreeGPUDATA();//  

   uint step; //  
   inline int _ConvertSMVer2Cores(int major, int minor);// 
   void DeviceQuery(); //  
   void RunDisplaceMove(uint rejectState, uint majKind);//  
   void RunRotateMove(uint rejectState, uint majKind ); //  
   #if ENSEMBLE == GEMC
   void RunVolumeMove(uint rejectState, uint majKind);// 
   #endif
   #if ENSEMBLE == GEMC || ENSEMBLE==GCMC
   void RunMolTransferMove(uint rejectState, uint majKind);// 
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

   CalculateEnergy calcEnergy;
   PRNG prng;

   //Procedure to run once move is picked... can also be called directly for
   //debugging...
   void RunMove(uint majKind, double draw, const uint step);

   ~System();

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


