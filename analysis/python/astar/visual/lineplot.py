'''
    Copyright (C) 2014-2019, Johannes Pekkilae, Miikka Vaeisalae.

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
import pylab as plt 
import numpy as np 
import matplotlib.gridspec as gridspec
import matplotlib.colors as colors

CM_INFERNO = plt.get_cmap('inferno')

end_rm = -1 #-35#-40

def plot_min_man_rms(ts, xaxis, yaxis1, yaxis2, yaxis3):
    plt.plot(ts.var[xaxis][:end_rm], ts.var[yaxis1][:end_rm], label=yaxis1)
    plt.plot(ts.var[xaxis][:end_rm], ts.var[yaxis2][:end_rm], label=yaxis2)
    plt.plot(ts.var[xaxis][:end_rm], ts.var[yaxis3][:end_rm], label=yaxis3)
    plt.xlabel(xaxis)
    plt.legend()

def plot_ts(ts, show_all=False, lnrho=False, uutot=False, uux=False, uuy=False, uuz=False, ss=False):

    if show_all:
        lnrho=True
        uutot=True
        uux=True
        uuy=True
        uuz=True
        ss=True

    if lnrho:
        plt.figure()
        xaxis  = 't_step'
        yaxis1 = 'lnrho_rms'
        yaxis2 = 'lnrho_min'
        yaxis3 = 'lnrho_max'
        plot_min_man_rms(ts, xaxis, yaxis1, yaxis2, yaxis3)
     
    if uutot:   
        plt.figure()
        xaxis = 't_step'
        yaxis1 = 'uutot_rms'
        yaxis2 = 'uutot_min'
        yaxis3 = 'uutot_max'
        plot_min_man_rms(ts, xaxis, yaxis1, yaxis2, yaxis3)
        
    if uux:   
        plt.figure()
        xaxis = 't_step'
        yaxis1 = 'uux_rms'
        yaxis2 = 'uux_min'
        yaxis3 = 'uux_max'
        plot_min_man_rms(ts, xaxis, yaxis1, yaxis2, yaxis3)
        
    if uuy:   
        plt.figure()
        xaxis = 't_step'
        yaxis1 = 'uuy_rms'
        yaxis2 = 'uuy_min'
        yaxis3 = 'uuy_max'
        plot_min_man_rms(ts, xaxis, yaxis1, yaxis2, yaxis3)
        
    if uuz:   
        plt.figure()
        xaxis = 't_step'
        yaxis1 = 'uuz_rms'
        yaxis2 = 'uuz_min'
        yaxis3 = 'uuz_max'
        plot_min_man_rms(ts, xaxis, yaxis1, yaxis2, yaxis3)
  
    if ss:   
        plt.figure()
        xaxis = 't_step'
        yaxis1 = 'ss_rms'
        yaxis2 = 'ss_min'
        yaxis3 = 'ss_max'
        plot_min_man_rms(ts, xaxis, yaxis1, yaxis2, yaxis3)
  
    plt.show()
