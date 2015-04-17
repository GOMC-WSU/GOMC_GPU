/*******************************************************************************
GPU OPTIMIZED MONTE CARLO (GOMC) BETA 0.97 (GPU version)
Copyright (C) 2015  GOMC Group

A copy of the GNU General Public License can be found in the COPYRIGHT.txt
along with this program, also can be found at <http://www.gnu.org/licenses/>.
********************************************************************************/

#include "BoxDimensions.h"
#include "MoveConst.h" //For cutoff-related fail condition

void BoxDimensions::Init(config_setup::RestartSettings const& restart,
			 config_setup::Volume const& confVolume, 
			 pdb_setup::Cryst1 const& cryst,
			 double rc, double rcSq)
{ 
   const double TENTH_ANGSTROM = 0.1;
   rCut = rc;
   rCutSq = rcSq;
   minBoxSize = rc * rcSq * 8 + TENTH_ANGSTROM;
   if (restart.enable && cryst.hasVolume)
      axis = cryst.axis;
   else if (confVolume.hasVolume)
      axis = confVolume.axis;
   else
   {
      fprintf(stderr, 
            "Error: Box Volume(s) not specified in PDB or in.dat files.\n");
      exit(EXIT_FAILURE);
   }
   halfAx.Init(BOX_TOTAL);
   axis.CopyRange(halfAx, 0, 0, BOX_TOTAL);
   halfAx.ScaleRange(0, BOX_TOTAL, 0.5);
   //Init volume/inverse volume.
   for (uint b = 0; b < BOX_TOTAL; b++)
   {
      volume[b] = axis.x[b] * axis.y[b] * axis.z[b]; 
      volInv[b] = 1.0 / volume[b];
   }
}

uint BoxDimensions::ShiftVolume
(BoxDimensions & newDim, double & scale, const uint b, const double delta) const
{
   uint rejectState = mv::fail_state::NO_FAIL;
   double newVolume = volume[b] + delta;
   newDim = *this;

   //If move would shrink any box axis to be less than 2 * rcut, then
   //automatically reject to prevent errors.
   if ( newVolume < minBoxSize )
   {
      std::cout << "WARNING!!! box shrunk below 2*rc! Auto-rejecting!" << std::endl;
      rejectState = mv::fail_state::VOL_TRANS_WOULD_SHRINK_BOX_BELOW_CUTOFF;
   }
   else
   {
      newDim.SetVolume(b, newVolume);
      scale = newDim.axis.Get(b).x / axis.Get(b).x;
   }
   return rejectState;
}

uint BoxDimensions::ExchangeVolume
(BoxDimensions & newDim, double & scaleO, double & scaleN,
 const uint bO, const uint bN, const double transfer) const
{
   //uint rejectState = mv::fail_state::NO_FAIL;
   ////double vRat = volume[bO]*volInv[bN];
   ////double expTr = vRat*exp(transfer);
   //double vTot = volume[bO] + volume[bN];
   //newDim = *this;
   ////newDim.volume[bO] = expTr * vTot / (1 + expTr);
   //newDim.volume[bO] = volume[bO] - transfer;
   //newDim.volInv[bO] = 1.0/newDim.volume[bO];
   //newDim.volume[bN] = vTot - newDim.volume[bO];
   //newDim.volInv[bN] = 1.0/newDim.volume[bN];

   ////If move would shrink any box axis to be less than 2 * rcut, then
   ////automatically reject to prevent errors.
   //if ( newDim.volume[bO] < minBoxSize || newDim.volume[bN] < minBoxSize )
   //   rejectState = mv::fail_state::VOL_TRANS_WOULD_SHRINK_BOX_BELOW_CUTOFF;
   //else
   //{
   //   double newAxX_bO = pow(newDim.volume[bO], (1.0/3.0));
   //   double newAxX_bN = pow(newDim.volume[bN], (1.0/3.0));
   //   scaleO = newAxX_bO / axis.Get(bO).x;
   //   scaleN = newAxX_bN / axis.Get(bN).x;
   //   XYZ newAx_bO(newAxX_bO, newAxX_bO, newAxX_bO);
   //   XYZ newAx_bN(newAxX_bN, newAxX_bN, newAxX_bN);
   //   newDim.axis.Set(bO, newAx_bO);
   //   newDim.axis.Set(bN, newAx_bN);
   //   newAx_bO *= 0.5;
   //   newAx_bN *= 0.5;
   //   newDim.halfAx.Set(bO, newAx_bO);
   //   newDim.halfAx.Set(bN, newAx_bN);
   //}
   //return rejectState;

	uint state = mv::fail_state::NO_FAIL;
   //double vRat = volume[bO]*volInv[bN];
   //double expTr = vRat*exp(transfer);
   double vTot = volume[bO] + volume[bN];
   newDim = *this;
   //newDim.volume[bO] = expTr * vTot / (1 + expTr);

   newDim.SetVolume(bO, volume[bO] + transfer);
   newDim.SetVolume(bN, vTot - newDim.volume[bO]);
   //If move would shrink any box axis to be less than 2 * rcut, then
   //automatically reject to prevent errors.
   
	 scaleO = newDim.axis.Get(bO).x / axis.Get(bO).x;
	 if (newDim.volume[bO] < minBoxSize)
	 {
	    state = mv::fail_state::VOL_TRANS_WOULD_SHRINK_BOX_BELOW_CUTOFF;
	 }
	 scaleN = newDim.axis.Get(bN).x / axis.Get(bN).x;
	 if (newDim.volume[bN] < minBoxSize)
	 {
	    state = mv::fail_state::VOL_TRANS_WOULD_SHRINK_BOX_BELOW_CUTOFF;
	 }
   return state;
}


