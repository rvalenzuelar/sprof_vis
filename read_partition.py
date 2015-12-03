""" Read  S-profiler radar precipitation partition

	Raul Valenzuela
	December, 2015
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


base_directory='/home/rvalenzuela/SPROF/partition/'

class partition(object):
	def __init__(self, *args):

		if args[0] == 1998:
			self.pfile = base_directory+'czc_sprof_rtype_98.mat'
		elif args[0] == 2001:
			self.pfile = base_directory+'czc_sprof_rtype_01.mat'
		elif args[0] == 2003:
			self.pfile = base_directory+'czc_sprof_rtype_03.mat'
		elif args[0] == 2004:
			self.pfile  = base_directory+'czc_sprof_rtype_04.mat'

	def get_rtype(self):

		mat=sio.loadmat(self.pfile)
		rtype = mat['czc_sprof_rtype'][0]
		return rtype

	def get_bbht(self):

		mat=sio.loadmat(self.pfile)
		bbht = mat['czc_sprof_rtype_bbht'][0]
		return bbht

	def get_begdayt(self):

		mat=sio.loadmat(self.pfile)
		begdayt = [datenum_to_datetime(x) for x in mat['czc_sprof_rtype_begdayt'][0]]
		return begdayt

	def get_enddayt(self):

		mat=sio.loadmat(self.pfile)
		enddayt = [datenum_to_datetime(x) for x in mat['czc_sprof_rtype_enddayt'][0]]
		return enddayt

	def get_numpcprof30(self):

		mat=sio.loadmat(self.pfile)
		numpcprof30 = mat['czc_sprof_rtype_numpcpprof30'][0]
		return numpcprof30

	def get_numpcprof30bb(self):

		mat=sio.loadmat(self.pfile)
		numpcprof30bb = mat['czc_sprof_rtype_numpcpprof30bb'][0]
		return numpcprof30bb

	def get_numpcprof30nbb(self):

		mat=sio.loadmat(self.pfile)
		numpcprof30nbb = mat['czc_sprof_rtype_numpcpprof30nbb'][0]
		return numpcprof30nbb

	def get_numprof30(self):

		mat=sio.loadmat(self.pfile)
		numprof30 = mat['czc_sprof_rtype_numprof30'][0]
		return numprof30

	def get_precip(self):

		mat=sio.loadmat(self.pfile)
		precip = mat['czc_sprof_rtype_precip'][0]
		return precip

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

