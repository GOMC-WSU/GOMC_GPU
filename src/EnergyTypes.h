
#ifndef ENERGYTYPES_H
#define ENERGYTYPES_H

/*
 *    EnergyTypes.h
 *    Defines structs containing energy values
 *
 */

#include "EnsemblePreprocessor.h" //For # box const.
#include "../lib/BasicTypes.h" //For uint

#ifndef NDEBUG
#include <iostream>
#endif
#ifndef BOXES_WITH_U_NB
#if ENSEMBLE == GCMC || ENSEMBLE == NVT
#define BOXES_WITH_U_NB 1
#elif ENSEMBLE == GEMC
//case for NVT, GCMC
#define BOXES_WITH_U_NB 2
#endif
#endif

#ifndef BOXES_WITH_U_B
#if ENSEMBLE == NVT
#define BOXES_WITH_U_B 1
#elif ENSEMBLE == GEMC || ENSEMBLE == GCMC
//case for NVT, GCMC
#define BOXES_WITH_U_B 2
#endif
#endif
struct Intermolecular
{ 
   //MEMBERS
   double virial, energy;

   //CONSTRUCTORS
    __host__ __device__ Intermolecular() : virial(0.0), energy(0.0) {}
    __host__ __device__ Intermolecular(const double vir, const double en) : 
      virial(vir), energy(en) {} 



   //VALUE SETTER
   void Zero() { virial = energy = 0.0; }

   //OPERATORS
    __host__ __device__ Intermolecular& operator=(Intermolecular const& rhs) 
   { virial = rhs.virial; energy = rhs.energy; return *this; }
    __host__ __device__ Intermolecular& operator-=(Intermolecular const& rhs) 
   { virial -= rhs.virial; energy -= rhs.energy; return *this; }
    __host__ __device__ Intermolecular& operator+=(Intermolecular const& rhs) 
   { virial += rhs.virial; energy += rhs.energy; return *this; }
    __host__ __device__ Intermolecular operator-(Intermolecular const& rhs) 
   { return Intermolecular(virial - rhs.virial, energy - rhs.energy); }
    __host__ __device__ Intermolecular operator+(Intermolecular const& rhs) 
   { return Intermolecular(virial + rhs.virial, energy + rhs.energy); }
};

struct Energy
{
   //MEMBERS
   double intraBond, intraNonbond, inter, tc, total;

   Energy() : intraBond(0.0), intraNonbond(0.0), inter(0.0), 
      tc(0.0), total(0.0) {}


   Energy(double bond, double nonbond, double inter) :
      intraBond(bond), intraNonbond(nonbond), inter(inter),
	 tc(0.0), total(0.0) {}

   //VALUE SETTERS
    __host__ __device__ double Total() 
   { total = intraBond + intraNonbond + inter + tc; return total; }
    __host__ __device__ void Zero() { 
      intraBond = 0.0;
      intraNonbond = 0.0;
      inter = 0.0;
      tc = 0.0;
      total = 0.0; 
   }

   //OPERATORS
    __host__ __device__ Energy& operator-=(Intermolecular const& rhs)
   { inter -= rhs.energy; return *this; }
    __host__ __device__ Energy& operator+=(Intermolecular const& rhs)
   { inter += rhs.energy; return *this; }
    __host__ __device__ Energy& operator-=(Energy const& rhs);
    __host__ __device__ Energy& operator+=(Energy const& rhs);
};

 __host__ __device__ inline Energy& Energy::operator-=(Energy const& rhs)
{ 
   inter -= rhs.inter;
   intraBond -= rhs.intraBond;
   intraNonbond -= rhs.intraNonbond;
   tc -= rhs.tc;
   total -= rhs.total;
   return *this; 
}

 __host__ __device__ inline Energy& Energy::operator+=(Energy const& rhs)
{ 
   inter += rhs.inter;
   intraBond += rhs.intraBond;
   intraNonbond += rhs.intraNonbond;
   tc += rhs.tc;
   total += rhs.total;
   return *this; 
}

struct Virial
{
   //MEMBERS
   double inter, tc, total;

   Virial() { Zero(); }

   //VALUE SETTERS
    __host__ __device__ double Total() { return total = inter + tc; }
    __host__ __device__ void Zero() { inter = tc = total = 0.0; }

   //OPERATORS
    __host__ __device__ Virial& operator-=(Virial const& rhs)
   { inter -= rhs.inter; tc -= rhs.tc; total -= rhs.total; return *this; }
    __host__ __device__ Virial& operator+=(Virial const& rhs)
   { inter += rhs.inter; tc += rhs.tc; total += rhs.total; return *this; }
    __host__ __device__ Virial& operator-=(Intermolecular const& rhs)
   { inter -= rhs.virial; return *this; }
    __host__ __device__ Virial& operator+=(Intermolecular const& rhs)
   { inter += rhs.virial; return *this; }
   //For accounting for dimensionality
    __host__ __device__ Virial& operator=(Virial const& rhs)
   { inter = rhs.inter; tc = rhs.tc; total = rhs.total; return *this; } 
    __host__ __device__ Virial& operator/=(const double rhs)
   { inter /= rhs; tc /= rhs; total /= rhs; return *this; }

};


struct SystemPotential
{
   void Zero();
    __host__ __device__ double Total();
   void Add(const uint b, Intermolecular const& rhs)
   { boxVirial[b] += rhs; boxEnergy[b] += rhs; } 
   void Add(const uint b, Energy const& en, Virial vir)
   { boxVirial[b] += vir; boxEnergy[b] += en; } 

     __host__ __device__ void Add(const uint b, double  en, double vir) // 
   { boxVirial[b].inter += vir; boxEnergy[b].inter += en; } 


   void Sub(const uint b, Energy const& en, Virial vir)
   { boxVirial[b] -= vir; boxEnergy[b] -= en; } 
   SystemPotential& operator=(SystemPotential const& rhs)
   {
      for (uint b = 0; b < BOX_TOTAL; b++)
      {
	 boxVirial[b] = rhs.boxVirial[b];
	 boxEnergy[b] = rhs.boxEnergy[b]; 
      }
      totalEnergy = rhs.totalEnergy;
      totalVirial = rhs.totalVirial;
      return *this;
   } 
   SystemPotential& operator+=(SystemPotential const& rhs)
   {
      for (uint b = 0; b < BOX_TOTAL; b++)
      {
	 boxVirial[b] += rhs.boxVirial[b];
	 boxEnergy[b] += rhs.boxEnergy[b]; 
      }
      Total();
      return *this;
   } 
   SystemPotential& operator-=(SystemPotential const& rhs)
   {
      for (uint b = 0; b < BOX_TOTAL; b++)
	 Add(b, rhs.boxEnergy[b], rhs.boxVirial[b]);
      Total();      return *this;
   } 

   Virial boxVirial[BOX_TOTAL], totalVirial; 
   Energy boxEnergy[BOX_TOTAL], totalEnergy; 
};

inline void SystemPotential::Zero()
{
   for (uint b = 0; b < BOX_TOTAL; b++)
   {
      boxEnergy[b].Zero();
      boxVirial[b].Zero();
   }
   totalEnergy.Zero();
   totalVirial.Zero();
}

 __host__ __device__ inline double SystemPotential::Total()
{
   totalEnergy.Zero();
   totalVirial.Zero();
   for (uint b = 0; b < BOX_TOTAL; b++)
   {
      boxEnergy[b].Total();
      totalEnergy += boxEnergy[b];
      boxVirial[b].Total();
      totalVirial += boxVirial[b];
   }
   return totalEnergy.total;
}

#ifndef NDEBUG
inline std::ostream& operator << (std::ostream& out, const Energy& en)
{
 out << "Total: " << en.total << "   Inter: " << en.inter
       << "   IntraB: " << en.intraBond << "   IntraNB: "
       << en.intraNonbond << '\n';
   return out;
}
#endif

#endif

