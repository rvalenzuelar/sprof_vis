""" Plot S-profiler radar

	Raul Valenzuela
	November, 2015
	raul.valenzuela@colorado.edu	
"""

# import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import read_partition
import read_sprof
import sys

from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import MultipleLocator
from scipy.interpolate import interp1d
from datetime import datetime,timedelta
from common import find_index,  format_yaxis, format_xaxis

base_directory='/home/rvalenzuela/SPROF/matfiles'

'''  	Dates for zooming in a given period
	reqdates = [yyyy, m, d, H]'''
reqdates={ '1': {'ini':[1998,1,18,14],'end':[1998,1,19,23]},
			'2': {'ini':[1998,1,26,0],'end':[1998,1,27,4]},
			'3': {'ini':[2001,1,23,20],'end':[2001,1,23,22]},
			'5': {'ini':[2001,2,9,13],'end':[2001,2,9,15]},
			'7': {'ini':[2001,2,17,17],'end':[2001,2,17,19]},
			'8': {'ini':[2003,1,12,0],'end':[2003,1,13,0]},
			'9': {'ini':[2003,1,21,0],'end':[2003,1,23,5]},
			'11': {'ini':[2004,1,9,0],'end':[2004,1,10,0]},
			'13': {'ini':[2004,2,17,12],'end':[2004,2,18,0]},
			'14': {'ini':[2004,2,25,0],'end':[2004,2,26,0]}
			}
extent=[]
dayt=[]

def main():

	global extent
	global dayt

	casenum = raw_input('\nIndicate case number (i.e. 1): ')

	dbz,vvel, ht, dayt = read_sprof.retrieve_arrays(base_directory, casenum)

	extent=[0, len(dayt), 0, len(ht)]

	''' set index for time zooming '''
	try:
		idx_st=find_index(dayt,reqdates[casenum]['ini'])
		idx_end=find_index(dayt,reqdates[casenum]['end'])
	except KeyError:
		idx_st=0
		idx_end=len(dayt)-1


	''' Precipitation partition 
	**************************'''
	partition=read_partition.partition(dayt[0].year)
	bbht=partition.get_bbht()
	rtype=partition.get_rtype()
	begdayt=partition.get_begdayt()
	enddayt=partition.get_enddayt()
	
	beg_aux = np.asarray([datetime(t.year, t.month, t.day, t.hour,0,0) for t in begdayt])
	end_aux = np.asarray([datetime(t.year, t.month, t.day, t.hour,0,0) for t in enddayt])
	dayt_aux=np.asarray([datetime(t.year, t.month, t.day, t.hour,0,0) for t in dayt])

	beg_index = np.where(beg_aux==dayt_aux[0])[0][0]
	end_index = np.where(end_aux==dayt_aux[-1])[0][1]

	bbht = bbht[beg_index+1:end_index] 
	rtype = rtype[beg_index+1:end_index]
	begdayt = begdayt[beg_index:end_index]
	enddayt = enddayt[beg_index:end_index]

	d = [[t.year, t.month, t.day, t.hour,t.minute] for t in begdayt]
	idx_be=np.asarray([find_index(dayt,x) for x in d[1:]])
	d = [[t.year, t.month, t.day, t.hour,t.minute] for t in enddayt]
	idx_en=np.asarray([find_index(dayt,x) for x in d[1:]])
	idx_bbht=(idx_be+idx_en)/2


	''' 	Plots
	**************'''
	fig,ax=plt.subplots(2,1,sharex=True, figsize=(12,8))
	
	plot_reflectivity(ax[0],dbz,ht, cmap='nipy_spectral',vmin=-20,vmax=60)
	plot_velocity(ax[1],vvel,ht, cmap='bwr',vmin=-3,vmax=3)

	f = interp1d(ht, range(len(ht)))
	ax[0].plot(idx_bbht, f(bbht),marker='o',linestyle='--',color='black')
	for n, s in enumerate(rtype):
		if s[0] != 'NaN':
			ax[0].text(idx_bbht[n], -5, s[0][0].upper(), horizontalalignment='center',clip_on=True)
	ax[1].plot(idx_bbht, f(bbht),marker='o',linestyle='--',color='black')

	' it has to be before set_xlim'
	# format_xaxis(ax[1],dayt,2,casenum)
	format_xaxis(ax[1],dayt,labels=True,format='%d\n%H')

	ax[0].set_xlim([idx_st,idx_end])
	ax[0].set_ylim([-10,len(ht)])
	ax[1].set_ylim([-10,len(ht)])


	fig.subplots_adjust(hspace=.07)	
	
	ax[1].invert_xaxis()
	ax[1].set_xlabel(r'$\Leftarrow$'+' Time [UTC]')
	# ax[1].set_xlabel(' Time [UTC]'+r'$\Rightarrow$')

	# ax[0].grid(b=True, which='major', color='b', linestyle='-')
	plt.suptitle('SPROF observations. Date: '+dayt[0].strftime('%Y-%b'))

	plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.1)
	plt.draw()
	# plt.show()
	plt.show(block=False)


def plot_reflectivity(ax,dbz,ht, **kwargs):

	cmap=kwargs['cmap']
	vmin=kwargs['vmin']
	vmax=kwargs['vmax']	
	im0=ax.imshow(dbz,origin='lower',
						interpolation='none',
						aspect='auto',
						cmap=cmap,
						extent=extent,
						vmin=vmin,
						vmax=vmax)
	add_colorbar(im0,ax)
	format_yaxis(ax,ht)
	ax.set_ylabel('Altitude MSL [km]')
	ax.text(0.05, 0.9, 'Reflectivity [dBZ]',
					weight='bold',
					transform = ax.transAxes)

def plot_velocity(ax,vvel,ht, **kwargs):
	
	cmap=kwargs['cmap']
	vmin=kwargs['vmin']
	vmax=kwargs['vmax']
	im1=ax.imshow(vvel,origin='lower',
						interpolation='none',
						extent=extent,
						aspect='auto',
						cmap=cmap,
						vmin=vmin,
						vmax=vmax)
	add_colorbar(im1,ax)
	format_yaxis(ax,ht)
	ax.set_ylabel('Altitude MSL [km]')	
	ax.text(0.05, 0.9, 'Vertical Velocity [ms-1]',
				weight='bold',
				transform = ax.transAxes)


def time_period(start, end, delta):
    curr = start
    while curr < end:
        yield curr
        curr += delta

def add_colorbar(im,ax):

	''' Create divider for existing axes instance '''
	divider = make_axes_locatable(ax)
	''' Append axes to the right of ax3, with x% width of ax '''
	cax = divider.append_axes("right", size="1%", pad=0.05)
	''' Create colorbar in the appended axes
		Tick locations can be set with the kwarg `ticks`
		and the format of the ticklabels with kwarg `format` '''
	plt.colorbar(im, cax=cax)





main()