This directory will contain a sample setup based Väisälä (2023) adapted to most
recent version of the Astaroth code. 

Reference: https://doi.org/10.3847/1538-4357/acfb00 

Needed defines: 

hostdefine LDENSITY (1)
hostdefine LHYDRO (1)
hostdefine LMAGNETIC (1)
hostdefine LENTROPY (1)
hostdefine LTEMPERATURE (0)
hostdefine LFORCING (0)
hostdefine LUPWD (1)
hostdefine LBFIELD (1 && LMAGNETIC) // bfield only relevant if magnetic is on


hostdefine LSINK (1)
hostdefine LSHOCK (1)
hostdefine LRESOHMIC (1)

