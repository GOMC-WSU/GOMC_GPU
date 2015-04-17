/*******************************************************************************
GPU OPTIMIZED MONTE CARLO (GOMC) BETA 0.97 (GPU version)
Copyright (C) 2015  GOMC Group

A copy of the GNU General Public License can be found in the COPYRIGHT.txt
along with this program, also can be found at <http://www.gnu.org/licenses/>.
********************************************************************************/
/*******************************************************************************
GPU OPTIMIZED MONTE CARLO (GOMC) BETA 0.95 (GPU version)
Copyright (C) 2014  GOMC Group

A copy of the GNU General Public License can be found in the COPYRIGHT.txt
along with this program, also can be found at <http://www.gnu.org/licenses/>.
********************************************************************************/
#ifndef DCDIATOMIC_H
#define DCDIATOMIC_H

#include "DCComponent.h"
#include "../../lib/BasicTypes.h"

namespace mol_setup { struct MolKind; }

namespace cbmc {
   class DCData;

   class DCDiatomic : public DCComponent
   {
   public:
      DCDiatomic(DCData* data, const mol_setup::MolKind kind, uint first, uint second);
      void PrepareNew() {};
      void PrepareOld() {};
      void BuildOld(TrialMol& oldMol, uint molIndex);
      void BuildNew(TrialMol& newMol, uint molIndex);
      DCComponent* Clone() { return new DCDiatomic(*this); };

  // private:
      DCData* data;
      uint first, second;
      double bondLength;
   };
}
#endif

