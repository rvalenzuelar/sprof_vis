""" Read NOAA S-profiler radar precipitation partition

	Raul Valenzuela
	December, 2015
	raul.valenzuela@colorado.edu
"""

import scipy.io as sio
import numpy as np

from common import datenum_to_datetime, find_index
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

	def get_bbht(self,**kwargs):

		mat=sio.loadmat(self.pfile)
		bbht = mat['czc_sprof_rtype_bbht'][0]
		if 'time' in kwargs:
			idx, time_idx = self.get_index(kwargs['time'])
			return bbht[idx], time_idx
		else:
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

	def get_index(self,time):

		begdayt=self.get_begdayt()
		enddayt=self.get_enddayt()

		beg_aux = np.asarray([datetime(t.year, t.month, t.day, t.hour,0,0) for t in begdayt])
		end_aux = np.asarray([datetime(t.year, t.month, t.day, t.hour,0,0) for t in enddayt])
		dayt_aux=np.asarray([datetime(t.year, t.month, t.day, t.hour,0,0) for t in time])

		beg_index = np.where(beg_aux==dayt_aux[0])[0][0]
		end_index = np.where(end_aux==dayt_aux[-1])[0][1]

		begdayt = begdayt[beg_index:end_index]
		enddayt = enddayt[beg_index:end_index]

		beg = [[t.year, t.month, t.day, t.hour,t.minute] for t in begdayt]
		idx_be=np.asarray([find_index(time, x) for x in beg])
		end = [[t.year, t.month, t.day, t.hour,t.minute] for t in enddayt]
		idx_en=np.asarray([find_index(time, x) for x in end])
		idx_bbht=(idx_be+idx_en)/2

		return range(beg_index,end_index), idx_bbht
