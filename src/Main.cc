#include "Simulation.h"
#include "GOMC_Config.h"    //For version number#include <iostream>
#include <ctime>

//find and include appropriate files for getHostname
#ifdef _WIN32
#include <Winsock2.h>
#define HOSTNAME
#elif defined(__linux__) || defined(__apple__) || defined(__FreeBSD__)
#include <unistd.h>
#define HOSTNAME
#endif

#define GOMCMajor 1
#define GOMCMinor 0

//namespace {

//void PrintSimulationHeader();
//void PrintSimulationFooter();
//}
void PrintTime(char * sTime) {

	// current date/time based on current system
	time_t now = time(0);

	// convert now to string form
	char* dt = ctime(&now);

	cout << "\n    ============================= GOMC " << GOMCMajor << "."
			<< GOMCMinor << " =============================\n\n";
	cout << "         Simulation " << sTime << " date and time is: " << dt
			<< endl;
	cout
			<< "    ====================================================================\n\n";

}
int main(void) {
	PrintTime("start");
	const char * nm = "in.dat";

	Simulation sim(nm);
	sim.RunSimulation();
	PrintTime("end");
	return 0;
}

