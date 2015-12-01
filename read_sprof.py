""" Read and plot S-profiler radar

	Raul Valenzuela
	November, 2015
"""

import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as mtick
import os

from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import MultipleLocator
from scipy.interpolate import interp1d
from datetime import datetime,timedelta


base_directory='/home/rvalenzuela/SPROF/matfiles'


def main():


	# case='03'



	sprof_files =get_filenames('3')
	mat={}
	dayt={}
	dbz={}
	vvel={}
	if len(sprof_files)>1:

		for k,s in enumerate(sprof_files):
			# # mat1=sio.loadmat('/home/raul/Desktop/SPROF/czc_cal_sprof_01023_120gates.mat')
			# # mat2=sio.loadmat('/home/raul/Desktop/SPROF/czc_cal_sprof_01024_120gates.mat')
			# mat1=sio.loadmat('/home/raul/Desktop/SPROF/czc_cal_sprof_01048_120gates.mat')
			# mat2=sio.loadmat('/home/raul/Desktop/SPROF/czc_cal_sprof_01049_120gates.mat')


			mat[k]=sio.loadmat(s)
			ht=mat[k]['czc_sprof_ht'][0]
			daytin=mat[k]['czc_sprof_dayt']
			dayt[k]=np.asarray([datenum_to_datetime(x) for x in daytin])
			dbz[k]=mat[k]['czc_sprof_dbz']
			vvel[k]=mat[k]['czc_sprof_vvel']

		dayt=np.hstack((dayt[0],dayt[1]))
		dbz=np.hstack((dbz[0].T,dbz[1].T))
		vvel=np.hstack((vvel[0].T,vvel[1].T))

	else:
		mat=sio.loadmat(sprof_files[0])

		ht=mat['czc_sprof_ht'][0]
		dayt=mat['czc_sprof_dayt']
		dayt = np.asarray([datenum_to_datetime(x) for x in dayt])
		dbz=mat['czc_sprof_dbz']
		vvel=mat['czc_sprof_vvel']

	extent=[0, len(dayt), 0, len(ht)]

	for t in dayt:
		print t.strftime('%d %H:%M:%S')

	idx_st=find_index(dayt,[2001,1,23,21])
	idx_end=find_index(dayt,[2001,1,24,1])
	# idx_st=find_index(dayt,[2001,2,17,17])
	# idx_end=find_index(dayt,[2001,2,17,19])

	fig,ax=plt.subplots(2,1,sharex=True)
	im0=ax[0].imshow(dbz,origin='lower',
						interpolation='nearest',
						aspect='auto',
						cmap='nipy_spectral',
						extent=extent,
						vmin=-20,
						vmax=60)
	add_colorbar(im0,ax[0])
	format_yaxis(ax[0],ht)
	ax[0].set_ylabel('Altitude AGL [km]')
	ax[0].text(0.05, 0.9, 'Reflectivity [dBZ]',
					weight='bold',
					transform = ax[0].transAxes)


	im1=ax[1].imshow(vvel,origin='lower',
						interpolation='none',
						extent=extent,
						aspect='auto',
						cmap='bwr',
						vmin=-3,
						vmax=3)
	add_colorbar(im1,ax[1])
	format_yaxis(ax[1],ht)
	format_xaxis(ax[0],dayt,1)
	ax[1].text(0.05, 0.9, 'Vertical Velocity [ms-1]',
				weight='bold',
				transform = ax[1].transAxes)

	ax[0].set_xlim([idx_st,idx_end])

	fig.subplots_adjust(hspace=.07)	
	
	ax[1].invert_xaxis()
	ax[1].set_xlabel(r'$\Leftarrow$'+' Time [UTC]')
	ax[1].set_ylabel('Altitude AGL [km]')
	plt.draw()
	# plt.show()
	plt.show(block=False)

def find_index(datetime_array,t):

	idx=[[]]
	s=0
	m=0
	while not idx[0]:
		idx = np.where(datetime_array == datetime(t[0], t[1], t[2], t[3], m, s))
		s+=1
		if s>=60:
			m+=1	

	return idx[0][0]

def format_xaxis(ax,time,freq_hr):

	date_fmt='%d\n%H'
	xtlabel=[]
	new_xticks=[]
	idx_old=0
	n=0
	for idx,t in enumerate(time):
		tstr=t.strftime(date_fmt)
		if np.mod(t.hour,freq_hr) == 0 and \
			t.minute==0 and tstr not in xtlabel:
			xtlabel.append(t.strftime(date_fmt))
			new_xticks.append(idx)
			n+=1
		elif np.mod(t.minute,10) == 0 and idx != new_xticks[n-1]+1:
			xtlabel.append('')
			new_xticks.append(idx)
			n+=1

	ax.set_xticks(new_xticks)
	ax.set_xticklabels(xtlabel)

def format_yaxis(ax,hgt):

	f = interp1d(hgt,range(len(hgt)))
	
	ys=np.arange(1,8)
	new_yticks = f(ys)
	ytlabel = ['{:2.1f}'.format(y) for y in ys]
	ax.set_yticks(new_yticks)
	ax.set_yticklabels(ytlabel)

	ax.set_ylim([-10,len(hgt)])


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