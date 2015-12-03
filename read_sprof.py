""" Read and plot S-profiler radar

	Raul Valenzuela
	November, 2015
	raul.valenzuela@colorado.edu	
"""

import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as mtick
import os
import read_partition
import sys

from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import MultipleLocator
from scipy.interpolate import interp1d
from datetime import datetime,timedelta


base_directory='/home/rvalenzuela/SPROF/matfiles'

' reqdates = [yyyy, m, d, H]'
reqdates={ '1': {'ini':[1998,1,18,14],'end':[1998,1,19,23]},
			'2': {'ini':[1998,1,26,0],'end':[1998,1,27,4]},
			# '3': {'ini':[2001,1,23,21],'end':[2001,1,24,1]},
			# '7': {'ini':[2001,2,17,17],'end':[2001,2,17,19]},
			'9': {'ini':[2003,1,21,0],'end':[2003,1,23,5]},
			'13': {'ini':[2004,2,17,12],'end':[2004,2,18,0]},
			'14': {'ini':[2004,2,25,0],'end':[2004,2,26,0]}
			}
extent=[]
ht=[]
dayt=[]

def main():

	global extent
	global ht
	global dayt
	global czc_elev

	casenum = raw_input('\nIndicate case number (i.e. 1): ')
	sprof_files =get_filenames(casenum)

	if len(sprof_files)>1:

		for k,s in enumerate(sprof_files):
			mat=sio.loadmat(s)
			ht=mat['czc_sprof_ht'][0] # km MSL
			
			if k==0:
				dyt=[datenum_to_datetime(x) for x in mat['czc_sprof_dayt']]
				dayt=np.asarray(dyt)	
				dbz=mat['czc_sprof_dbz'].T
				vvel=mat['czc_sprof_vvel'].T
			else:
				dyt=np.asarray([datenum_to_datetime(x) for x in mat['czc_sprof_dayt']])
				dayt=np.hstack((dayt, dyt))
				dbz=np.hstack((dbz, mat['czc_sprof_dbz'].T))
				vvel=np.hstack((vvel, mat['czc_sprof_vvel'].T))
				
	else:
		mat=sio.loadmat(sprof_files[0])
		ht=mat['czc_sprof_ht'][0] # km MSL
		dayt=mat['czc_sprof_dayt']
		dayt = np.asarray([datenum_to_datetime(x) for x in dayt])
		dbz=mat['czc_sprof_dbz'].T
		vvel=mat['czc_sprof_vvel'].T


	extent=[0, len(dayt), 0, len(ht)]

	' set index for time zooming'
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
	format_xaxis(ax,dayt,1)
	ax.set_ylabel('Altitude MSL [km]')	
	ax.text(0.05, 0.9, 'Vertical Velocity [ms-1]',
				weight='bold',
				transform = ax.transAxes)

def find_index(datetime_array, t):

	idx=[[]]
	sec=0
	mnt=0
	
	if len(t)==4:
		mnt=0
	elif len(t)==5:		
		mnt=t[4]

	while not idx[0]:
		idx = np.where(datetime_array == datetime(t[0], t[1], t[2], t[3], mnt, sec))
		sec+=1
		if sec==60:
			mnt+=1
			sec=0
		if mnt==60:
			mnt=0
			t[3]+=1

	return idx[0][0]

def format_xaxis(ax,time,freq_hr):

	date_fmt='%d\n%H'
	xtlabel=[]
	new_xticks=[]
	idx_old=0
	n=0
	for idx,t in enumerate(time):
		tstr=t.strftime(date_fmt)
		
		# '1998 data has times where hours start at minute 1 or 2'
		if np.mod(t.hour,freq_hr) == 0 and t.minute in [0,1,2] and tstr not in xtlabel:
			xtlabel.append(t.strftime(date_fmt))
			new_xticks.append(idx)
			n+=1
		# ' some data have jumps in time so half-hour could be minute 30 or 31'
		elif np.mod(t.minute,30) in [0,1] and idx != new_xticks[n-1]+1:
			xtlabel.append('')
			new_xticks.append(idx)
			n+=1

	ax.set_xticks(new_xticks)
	ax.set_xticklabels(xtlabel)

def format_yaxis(ax,hgt):
	
	f = interp1d(hgt,range(len(hgt)))
	
	ys=np.arange(1.,8.)
	new_yticks = f(ys)
	ytlabel = ['{:2.1f}'.format(y) for y in ys]
	ax.set_yticks(new_yticks)
	ax.set_yticklabels(ytlabel)



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

def datenum_to_datetime(datenum):
    """
    Convert Matlab datenum into Python datetime.
    :param datenum: Date in datenum format
    :return:        Datetime object corresponding to datenum.

    source: https://gist.github.com/vicow
    """
    days = datenum % 1
    hours = days % 1 * 24
    minutes = hours % 1 * 60
    seconds = minutes % 1 * 60
    return datetime.fromordinal(int(datenum)) \
           + timedelta(days=int(days)) \
           + timedelta(hours=int(hours)) \
           + timedelta(minutes=int(minutes)) \
           + timedelta(seconds=round(seconds)) \
           - timedelta(days=366)

def get_filenames(usr_case):

	case='case'+usr_case.zfill(2)
	casedir=base_directory+'/'+case
	out=os.listdir(casedir)
	out.sort()
	file_sprof=[]
	for f in out:
		file_sprof.append(casedir+'/'+f)
	return file_sprof

main()