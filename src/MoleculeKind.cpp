#include "MoleculeKind.h"
#include "MolSetup.h"
#include "FFSetup.h"
#include "Forcefield.h"
#include "PDBConst.h" //For resname length.
#include "PRNG.h"
#include "Geometry.h"
#include "Setup.h"
#include "CBMC.h"

#include <vector>
#include <map>
#include <string>
#include <cstdio>
#include <algorithm>
#include <utility>
#include <cstdio>
#include <cstdlib>      //for exit

void MoleculeKind::Init(std::string const& l_name, Setup const& setup,
		Forcefield const& forcefield, System& sys)

		{
	mol_setup::MolMap::const_iterator dataIterator = setup.mol.kindMap.find(
			l_name);
	if (dataIterator == setup.mol.kindMap.end()) {
		std::cerr << "================================================"
				<< std::endl << std::endl << "Error: Molecule " << l_name
				<< " not found in PDB file. Exiting." << std::endl;

		exit(EXIT_FAILURE);
	}
	const mol_setup::MolKind& molData = dataIterator->second;
	name = l_name;

#if ENSEMBLE == GCMC
	std::map<std::string, double>::const_iterator kindCPIt =
	setup.config.sys.chemPot.cp.find(name),
	lastOne = setup.config.sys.chemPot.cp.end();

	//If we don't find a chemical potential for a kind in GCMC mode,
	//then quit.
	if (kindCPIt == lastOne)
	{
		std::cerr << "================================================"
		<< std::endl << std::endl
		<< "Error: chemical potential is missing for "
		<< name <<"." << std::endl << std::endl
		<< "Here are the listed chemical potentials:"
		<< std::endl
		<< "----------------------------------------"
		<< std::endl;

		//Print out whatever chemical potentials were read.
		for (kindCPIt = setup.config.sys.chemPot.cp.begin();
				kindCPIt != lastOne;
				++kindCPIt)
		{
			std::cerr << "Resname: " << kindCPIt->first
			<< "      Value: " << kindCPIt->second
			<< std::endl;
		}
		exit(EXIT_FAILURE);
	}
	else
	{
		chemPot = kindCPIt->second;
	}
#endif

	InitAtoms(molData);

	//Once-through topology objects

	if (forcefield.OneThree) {
		nonBonded_1_4 = new Nonbond_1_3;
		nonBonded_1_4->Init(molData);
	}
	if (forcefield.OneFour) {
		nonBonded_1_4 = new Nonbond_1_4;
		nonBonded_1_4->Init(molData);
	}
	if (forcefield.OneN) {
		nonBonded.Init(molData);
	}

	EwaldNonbonded.Init(molData);
	sortedNB_1_4.Init(*nonBonded_1_4, numAtoms);
	sortedNB.Init(nonBonded, numAtoms);
	sortedEwaldNB.Init(EwaldNonbonded, numAtoms);
	bondList.Init(molData.bonds);
	angles.Init(molData.angles, bondList);
	dihedrals.Init(molData.dihedrals, bondList);

#ifdef VARIABLE_PARTICLE_NUMBER
	builder = cbmc::MakeCBMC(sys, forcefield, *this, setup);
	//builder = new cbmc::LinearVlugt(sys, forcefield, *this, setup);
#endif
}

MoleculeKind::MoleculeKind() :
		angles(3), dihedrals(4), atomMass(NULL), atomCharge(NULL), builder(
				NULL), atomKind(NULL), nonBonded_1_4(NULL) {
}

MoleculeKind::~MoleculeKind() {
	delete[] atomKind;
	delete[] atomMass;
	delete[] atomCharge;
	delete builder;
	delete[] nonBonded_1_4;
}

void MoleculeKind::InitAtoms(mol_setup::MolKind const& molData) {
	numAtoms = molData.atoms.size();
	atomKind = new uint[numAtoms];
	atomMass = new double[numAtoms];
	atomCharge = new double[numAtoms];
	molMass = 0;
	atomNames.clear();

	//convert array of structures to structure of arrays
	for (uint i = 0; i < numAtoms; ++i) {
		const mol_setup::Atom& atom = molData.atoms[i];
		atomNames.push_back(atom.name);
		atomMass[i] = atom.mass;
		molMass += atom.mass;
		atomCharge[i] = atom.charge;
		atomKind[i] = atom.kind;
	}
}
