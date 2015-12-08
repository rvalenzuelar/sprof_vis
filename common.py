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

	freqMin=kwargs['freqMinutes']
	labels=kwargs['labels']

	date_fmt='%d\n%H'
	xtlabel=[]
	new_xticks=[]

	asd = [np.mod(d.minute, freqMin) for d in time]
	asd=np.asarray(asd)
	
	'1998 data has no constant time step so this fix it'
	if time[0].year == 1998:
		idx = np.where((asd==0) | (asd==1))[0]
	else:
		idx = np.where(asd==0)[0]

	for n, i in enumerate(idx):
		tstr=time[i].strftime(date_fmt)
		if n==0:
			new_xticks.append(i)
			if labels:
				xtlabel.append(tstr)
			else:
				xtlabel.append('')			
		else:
			if i>idx[n-1]+1:
				new_xticks.append(i)
				if labels:
					xtlabel.append(tstr)
				else:
					xtlabel.append('')

	ax.set_xticks(new_xticks)
	ax.set_xticklabels(xtlabel)

def format_yaxis(ax,hgt):
	
	f = interp1d(hgt,range(len(hgt)))
	
	ys=np.arange(1.,8.)
	new_yticks = f(ys)
	ytlabel = ['{:2.1f}'.format(y) for y in ys]
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