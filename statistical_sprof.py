""" Make statistical analyses of NOAA S-profiler radar

	Raul Valenzuela
	December, 2015
	raul.valenzuela@colorado.edu
"""

import matplotlib.pyplot as plt
import read_sprof 
import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm 

from common import find_index, format_yaxis, format_xaxis, shiftedColorMap
from scipy import fftpack
from scipy.interpolate import interp1d
from datetime import datetime, timedelta
from matplotlib.ticker import FuncFormatter, ScalarFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable


base_directory='/home/rvalenzuela/SPROF/matfiles'
casenum = raw_input('\nIndicate case number (i.e. 1): ')

reqdates={ '1': {'ini':[1998,1,18,16],'end':[1998,1,18,19]},
			'2': {'ini':[1998,1,26,4],'end':[1998,1,26,14]},
			'3': {'ini':[2001,1,23,21],'end':[2001,1,24,1]},
			'6': {'ini':[2001,2,11,3],'end':[2001,2,11,10]},
			'7': {'ini':[2001,2,17,17],'end':[2001,2,17,19]},
			'10': {'ini':[2003,2,16,0],'end':[2003,2,16,7]},
			'11': {'ini':[2004,1,9,19],'end':[2004,1,9,22]},
			'12': {'ini':[2004,2,2,12],'end':[2004,2,2,16]},
			'13': {'ini':[2004,2,17,12],'end':[2004,2,18,0]},
			'14': {'ini':[2004,2,25,0],'end':[2004,2,26,0]}
			}

def main():
	dbz,vvel, ht, dayt = read_sprof.retrieve_arrays(base_directory, casenum)

	idx_st=find_index(dayt,reqdates[casenum]['ini'])
	idx_end=find_index(dayt,reqdates[casenum]['end'])

	dayt=dayt[idx_st:idx_end]
	vvel=vvel[:,idx_st:idx_end]
	dbz=dbz[:,idx_st:idx_end]

	vvel[vvel<=-10.]=np.nan
	vvel[vvel>=4.]=np.nan

	' add total seconds to array'
	ts=[]
	for d in dayt:
		ts.append((d-dayt[0]).total_seconds())

	' create equally spaced time grid '
	day = dayt[0]
	end = dayt[-1]
	step = timedelta(seconds=40)
	ts2 = []
	while day < end:
		result=(day-dayt[0]).total_seconds()
		ts2.append(result)
		day += step

	' setup new figure '	
	fig = plt.figure(figsize=(8,10))

	' images axes'
	gs = gridspec.GridSpec(4,1)
	gs.update(top=0.95, bottom=0.46, hspace=.05)
	ax0 = plt.subplot(gs[0])
	ax1 = plt.subplot(gs[1])

	' vvel axis'
	gs = gridspec.GridSpec(1,1)
	gs.update(top=0.7, bottom=0.48,right=0.88)
	ax2 = plt.subplot(gs[0])

	' power spectrum axis'
	gs = gridspec.GridSpec(1,1)
	gs.update(top=0.4, bottom=0.05,right=0.88)
	ax3 = plt.subplot(gs[0])

	'colormap for vvel'
	orig_cmap = cm.bwr
	shifted_cmap = shiftedColorMap(orig_cmap, midpoint=0.7, name='shifted')

	' add images'
	im0= ax0.imshow(dbz,interpolation='none',cmap='nipy_spectral',vmin=-20,vmax=60,aspect='auto',origin='lower')
	im1= ax1.imshow(vvel,interpolation='none',cmap=shifted_cmap,vmin=-10,vmax=4,aspect='auto',origin='lower')
	
	divider0 = make_axes_locatable(ax0)
	cax0 = divider0.append_axes("right", size="2%", pad=0.05)
	cbar0 = plt.colorbar(im0, cax=cax0)

	divider1 = make_axes_locatable(ax1)
	cax1 = divider1.append_axes("right", size="2%", pad=0.05)
	cbar1 = plt.colorbar(im1, cax=cax1)

	' add levels, Doppler velocity, and power spectrum'
	levels=[10, 20,30,40]
	colors=['y', 'k', 'g', 'r']
	variance_y=[0.7, 0.78, 0.86, 0.94]	
	for n,x in enumerate(levels):
		' add levels '
		ax0.plot([x]*len(vvel[x,:]),color=colors[n])
		ax1.plot([x]*len(vvel[x,:]),color=colors[n])

		' get traces at level x'
		trace_doppler=vvel[x,:]
		trace_dbz=dbz[x,:]	

		' extract hydrometeor vvel from Doppler vvel'
		trace_z = 10.**(trace_dbz/10.)
		vvel_hydrometeor = 0.817*trace_z**0.063 # Atlas et al. 1973
		trace_wind = trace_doppler - (-vvel_hydrometeor)

		' plot Doppler velocity at given levels '
		ax2.plot(trace_doppler,color=colors[n])
		ax2.plot(vvel_hydrometeor,color=colors[n])
		ax2.plot(trace_wind,color=colors[n], linestyle=':',linewidth=1)
		ax2.set_ylim([-10,8])

		' add variance annotation '
		variance=np.nanvar(trace_doppler)
		ax2.text(0.2, variance_y[n], 'Variance: '+'{:3.2f}'.format(variance), 
					color=colors[n], transform=ax2.transAxes)
		

		' create interpolant with regular time grid of 40 seconds'
		asd = check_trace(trace_doppler)
		f=interp1d(ts,asd)
		trace_doppler2= f(ts2)

		' plot power spectrum density'
		if ~np.any(np.isnan(trace_doppler2)):
			print 'level: '+str(x)
			asd = check_trace(trace_wind)
			power_spectrum(ax3, asd,colors[n],linestyle=':',marker=None)
			power_spectrum(ax3,trace_doppler2,colors[n],linestyle='-',marker=None)

	format_yaxis(ax0,ht)
	format_yaxis(ax1,ht)

	format_xaxis(ax0, dayt, freqMinutes=60,labels=False )
	format_xaxis(ax1, dayt, freqMinutes=60,labels=False )
	format_xaxis(ax2, dayt, freqMinutes=60,labels=True )


	ax0.set_ylabel('Hgt MSL [km]')
	ax1.set_ylabel('Hgt MSL [km]')
	ax2.set_ylabel('VVel [ms^-1]')
	ax2.set_xlabel(r'$\Leftarrow$'+' Time [UTC]')

	ax0.set_xlim([0, len(vvel[0,:])])
	ax0.invert_xaxis()
	ax0.set_xticklabels([])
	ax1.set_xlim([0, len(vvel[0,:])])
	# ax1.set_ylim([0, 20])
	ax1.invert_xaxis()
	ax1.set_xticklabels([])	
	ax2.set_xlim([0, len(vvel[0,:])])
	ax2.invert_xaxis()

	plt.suptitle('SPROF observations. Date: '+dayt[0].strftime('%Y-%b'))
	plt.draw()

	plt.show(block=False)


def power_spectrum(ax,array,color,linestyle,marker):


	F1 = fftpack.fft(array)
	cut_half = int(len(array)/2)
	ps = 2*np.abs( F1[:cut_half] )**2
	freq=np.linspace(1,len(ps),len(ps))/len(F1)
	# ax.plot(freq, ps, color=color,linestyle=linestyle,marker=marker)
	# ax.semilogy(freq, ps, color=color,linestyle=linestyle,marker=marker)
	ax.loglog(freq, ps, color=color,linestyle=linestyle,marker=marker)

	' intertial subrange '
	x=np.linspace(0.005, 0.5, 1000)
	inertial = x**(-5/3.)
	# ax.plot(x,inertial)
	# ax.semilogy(freq,inertial)
	ln=ax.loglog(x,inertial,linestyle='--',color='k',linewidth=3,label='-5/3')
	
	ax.set_ylim([1e-2,1e7])
	ax.set_xticks([0.002, 0.004, 0.008, 0.02, 0.04, 0.08, 0.2, 0.4, 0.8])

	xSeconds=[]
	seconds=40
	for x in ax.get_xticks():
		xSeconds.append(int(seconds/x))
	ax.set_xticklabels(xSeconds)

	ax.grid(True)
	ax.set_xlabel('seconds')
	ax.set_ylabel('2|F|^2')
	ax.legend(handles=ln)
	plt.draw

def decimalf(x,pos):
	return '%2.0f' % x

def integerf(x,pos):
	print x
	return '{:d}'.format(int(x))

def decimalx(x,pos):
	# return ('{{:.{:1d}f}}'.format(int(np.maximum(-np.log10(x),0)))).format(x)
	return ('{{:.{:1d}f}}'.format(int(np.ceil(-np.log10(x))))).format(x)

def check_trace(trace):

	' if there are only 2 nans on the trace the trace is ok '
	total_nans = np.sum(np.isnan(trace))
	if total_nans<=2:
		idx = np.where(np.isnan(trace))[0]
		for i in idx:
			' if nans have non-nans neighbors then interpolate'
			x=np.sum(np.isnan(trace[i-1:i+2]))
			if x==1:
				trace[i]=(trace[i-1]+trace[i+1])/2.
		return trace
	else:
		return trace

main()