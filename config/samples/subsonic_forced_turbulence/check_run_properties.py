'''
    Copyright (C) 2014-2023, Johannes Pekkila, Miikka Vaisala.

    This file is part of Astaroth.

    Astaroth is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Astaroth is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Astaroth.  If not, see <http://www.gnu.org/licenses/>.
'''

import numpy as np 
import pylab as plt 
import ma

# ---------------------------------------------------------------------------
# This Python script does a number of check on the simulation data. 
# 
# NOTE: Right now it focused on testing special reduction but can be added to
# feature other checks when we need to. 
# ---------------------------------------------------------------------------

#
# This is a test which tests windowed and other special reduction by comparing
# them to equivalent Python based reductions
#


