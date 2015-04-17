/*******************************************************************************
GPU OPTIMIZED MONTE CARLO (GOMC) BETA 0.97 (GPU version)
Copyright (C) 2015  GOMC Group

A copy of the GNU General Public License can be found in the COPYRIGHT.txt
along with this program, also can be found at <http://www.gnu.org/licenses/>.
********************************************************************************/

#include "Simulation.h"
#include "GOMC_Config.h"    //For version number
#include <iostream>
#include <ctime>

//find and include appropriate files for getHostname
#ifdef _WIN32
#include <Winsock2.h>
#define HOSTNAME
#elif defined(__linux__) || defined(__apple__) || defined(__FreeBSD__)
#include <unistd.h>
#define HOSTNAME
#endif

namespace{
   
    void PrintSimulationHeader();
    void PrintSimulationFooter();
}

int main(void)
{
   const char * nm = "in.dat";
   
   Simulation sim(nm);
   sim.RunSimulation();
 
   return 0;
}






