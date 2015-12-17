'''
	Common functions used in sprof_vis

	Raul Valenzuela
	December, 2015
	raul.valenzuela@colorado.edu
'''


import numpy as np
from datetime import datetime, timedelta
from scipy.interpolate import interp1d
import matplotlib
import matplotlib.pyplot as plt 

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

def format_xaxis(ax,time,**kwargs):

	freqMin=kwargs['minutes_tick']
	labels=kwargs['labels']
	# print time
	date_fmt='%d\n%H'
	xtlabel=[]
	new_xticks=[]

	' identifies beginning of hours '
	d0=time[0].hour
	hrbool=[True]
	for d in time[1:]:
		if d.hour == d0:
			hrbool.append(False)
		else:
			hrbool.append(True)
			d0=d.hour
	idxhr = np.where(hrbool)[0]

	if time[0].year == 1998:
		choose=[0,1,2]
	else:
		choose=[0]
	mntbool=[True]
	for n, d in enumerate(time[1:]):
		mod=np.mod(d.minute,freqMin)
		if mod in choose and mntbool[n]==False:
			mntbool.append(True)
		else:
			mntbool.append(False)
			
	idxmnt = np.where(mntbool)[0]

	if labels:
		for n, i in enumerate(idxmnt):	
				if i in idxhr:
					tstr=time[i].strftime(date_fmt)
					xtlabel.append(tstr)
				else:
					xtlabel.append('')
	else:
		[xtlabel.append('') for i in idxmnt]

	ax.set_xlim([0, len(time)])
	ax.set_xticks(idxmnt)
	ax.set_xticklabels(xtlabel)

def format_yaxis(ax,hgt,**kwargs):
	
	hgt_res = np.unique(np.diff(hgt))[0]
	if 'toplimit' in kwargs:
		toplimit=kwargs['toplimit']
		''' extentd hgt to toplimit km so all 
		time-height sections have a common yaxis'''
		hgt=np.arange(hgt[0],toplimit, hgt_res)
	belowrad_gates = np.arange(hgt[0]-hgt_res, 0, -hgt_res)
	f = interp1d(hgt,range(len(hgt)))
	ys=np.arange(np.ceil(hgt[0]), np.floor(hgt[-1])+1)
	new_yticks = f(ys)
	# ytlabel = ['{:2.1f}'.format(y) for y in ys]
	ytlabel = ['{:g}'.format(y) for y in ys]
	ax.set_ylim([-len(belowrad_gates), len(hgt)])
	ax.set_yticks(new_yticks)
	ax.set_yticklabels(ytlabel)	


def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):
    '''
    Function to offset the "center" of a colormap. Useful for
    data with a negative min and positive max and you want the
    middle of the colormap's dynamic range to be at zero

    Input
    -----
      cmap : The matplotlib colormap to be altered
      start : Offset from lowest point in the colormap's range.
          Defaults to 0.0 (no lower ofset). Should be between
          0.0 and `midpoint`.
      midpoint : The new center of the colormap. Defaults to 
          0.5 (no shift). Should be between 0.0 and 1.0. In
          general, this should be  1 - vmax/(vmax + abs(vmin))
          For example if your data range from -15.0 to +5.0 and
          you want the center of the colormap at 0.0, `midpoint`
          should be set to  1 - 5/(5 + 15)) or 0.75
      stop : Offset from highets point in the colormap's range.
          Defaults to 1.0 (no upper ofset). Should be between
          `midpoint` and 1.0.

       Author: http://stackoverflow.com/users/1552748/paul-h
    '''
    cdict = {
        'red': [],
        'green': [],
        'blue': [],
        'alpha': []
    }

    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)

    # shifted index to match the data
    shift_index = np.hstack([
        np.linspace(0.0, midpoint, 128, endpoint=False), 
        np.linspace(midpoint, 1.0, 129, endpoint=True)
    ])

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))

    newcmap = matplotlib.colors.LinearSegmentedColormap(name, cdict)
    plt.register_cmap(cmap=newcmap)

    return newcmap