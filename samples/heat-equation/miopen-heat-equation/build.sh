#!/usr/bin/bash
hipcc main.cu -lMIOpen -I /opt/rocm-5.2.3/include/ -L /opt/rocm-5.2.3/lib/