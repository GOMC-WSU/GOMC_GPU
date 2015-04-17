/*******************************************************************************
GPU OPTIMIZED MONTE CARLO (GOMC) BETA 0.97 (GPU version)
Copyright (C) 2015  GOMC Group

A copy of the GNU General Public License can be found in the COPYRIGHT.txt
along with this program, also can be found at <http://www.gnu.org/licenses/>.
********************************************************************************/

#ifndef FF_ANGLES_H
#define FF_ANGLES_H

#include "PRNG.h"
#include "FFSetup.h" //For initialization data
#include "../lib/BasicTypes.h" //For "uint"
#include "../lib/VectorLib.h" //For transfer vect --> array function
#include "../lib/NumLib.h" //For "Sq" function

class FFAngles
{
 public:
   FFAngles(void) : Ktheta(NULL), theta0(NULL) {}
   ~FFAngles(void)
   { 
      delete[] Ktheta;
      delete[] theta0;
   }

   void AngGen(double * ang, double * en, double & weightAng, 
	       PRNG & prng, const uint angKind, const uint numTrials,
	       const uint numPicksPerTrial, const double beta) const;

   double Calc(const uint kind, const double ang) const
   { return (Ktheta[kind] * num::Sq(ang-theta0[kind])); }

   void Init(ff_setup::Angle const& angle)
   {
      count = angle.Ktheta.size();
      Ktheta = vect::transfer(angle.Ktheta);
      theta0 = vect::transfer(angle.theta0);
   }

 private:
   double * Ktheta, * theta0;
   uint count;
};

#endif /*FF_ANGLES_H*/


