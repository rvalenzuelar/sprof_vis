


import numpy as np
from datetime import datetime, timedelta

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