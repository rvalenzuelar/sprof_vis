""" Read NOAA S-profiler radar

	Raul Valenzuela
	November, 2015
	raul.valenzuela@colorado.edu	
"""


import scipy.io as sio
import numpy as np
import os

from datetime import datetime,timedelta

def retrieve_arrays(*args):

	base_directory=args[0]
	casenum=args[1]

	sprof_files =get_filenames(base_directory, casenum)

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

	return dbz,vvel, ht, dayt 

def get_filenames(base_directory, casenum):

	case='case'+casenum.zfill(2)
	casedir=base_directory+'/'+case
	out=os.listdir(casedir)
	out.sort()
	file_sprof=[]
	for f in out:
		file_sprof.append(casedir+'/'+f)
	return file_sprof

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