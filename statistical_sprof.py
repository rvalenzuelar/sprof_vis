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
# import seaborn as sns 
import read_partition
import matplotlib
import plotPrecip as precip

from common import find_index, format_yaxis, format_xaxis, shiftedColorMap, is_numeric
from scipy import fftpack
from scipy.interpolate import interp1d
from datetime import datetime, timedelta
from matplotlib.ticker import FuncFormatter, ScalarFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.backends.backend_pdf import PdfPages

base_directory='/home/rvalenzuela/SPROF/matfiles'

' global variables used in plot_profile_variance'
ti=[] # title list
n=0 # line counter
colors=[] #color list

def main(plot=False):

	if plot:

		# casenum = raw_input('\nIndicate case number (i.e. 1): ')
		# dbz,vvel,ht,ts,ts2,dayt = get_arrays(casenum)
		# plot_turbulence_spectra(dbz,vvel,ts,ts2,ht,dayt)
		
		' Time-height sections '
		'*********************************'
		# thrs=0
		# dbz,vvel,ht,ts,ts2,dayt = get_arrays('3')
		# dbz[dbz<thrs]=np.nan
		# plot_thsections(dbz,vvel,ts,ts2,ht,dayt)
		# dbz,vvel,ht,ts,ts2,dayt = get_arrays('4')
		# dbz[dbz<thrs]=np.nan
		# plot_thsections(dbz,vvel,ts,ts2,ht,dayt)


		' Profile variance '
		'*********************************'
		# # ncases = range(1,15)
		# ncases = range(1,8)
		# # ncases = range(8,15)
		# # ncases = [1,2,3]
		# thrs=0
		# for c in ncases:
		# 	dbz,vvel,ht,_,_,_ = get_arrays(str(c))
		# 	idx = np.where(ht<3.75)[0]
		# 	dbz[dbz<thrs]=np.nan		
		# 	if c == ncases[0]:
		# 		ax=plot_profile_variance(dbz[idx,:],vvel[idx,:],ht[idx],False,str(c),len(ncases))
		# 	else:
		# 		plot_profile_variance(dbz[idx,:],vvel[idx,:],ht[idx],ax,str(c),len(ncases))

		' Echo-top '
		'*********************************'
		# # ncases=range(1,15)
		# ncases=[1]
		# for c in ncases:
		# 	dbz,vvel,ht,ts,ts2,dayt = get_arrays(str(c))
		# 	with np.errstate(invalid='ignore'):
		# 		dbz[dbz<0]=np.nan
		# 	echot = timeserie_echotop(dbz,ht,plot=False,retrieve='km')
		# 	print 'Echotop variance case' + str(c)+': {:2.1f}'.format(np.nanvar(echot))
		# 	echot = timeserie_echotop(dbz,ht,plot=False,retrieve='index')		
		# 	plot_thsections(dbz,vvel,ht,dayt,echotop=echot)
		# plt.show(block=False)

		' Print single panels to pdf'
		'*******************************'
		# ppdbz=PdfPages('sprof_dbz_significant.pdf')
		# ppvvel=PdfPages('sprof_vvel_significant.pdf')
		# ncases=range(1,15)
		# # ncases=[1,2,3]
		# for c in ncases:
		# 	dbz,vvel,ht,ts,ts2,dayt = get_arrays(str(c))
		# 	# with np.errstate(invalid='ignore'):
		# 	# 	dbz[dbz<0]=np.nan		
		# 	if c==3:
		# 		cbar=True
		# 	else:
		# 		cbar=False
		# 	echoti = timeserie_echotop(dbz,ht,plot=False,retrieve='index')		
		# 	echotm = timeserie_echotop(dbz,ht,plot=False,retrieve='km')
		# 	partition=read_partition.partition(dayt[0].year)
		# 	bbht,bbtimeidx=partition.get_bbht(time=dayt)
		# 	plot_thsections_single(ht,dayt,dbz=dbz,colorbar=cbar,case=c,echotop=[echoti,echotm], bband=[bbht, bbtimeidx])
		# 	ppdbz.savefig()
		# 	plot_thsections_single(ht,dayt,vvel=vvel,colorbar=cbar,case=c,echotop=[echoti,echotm], bband=[bbht, bbtimeidx])
		# 	ppvvel.savefig()
		# 	plt.close('all')
		# ppdbz.close()
		# ppvvel.close()

		' Layer variance '
		'**********************'
		# for c in range(1,15):
		# 	dbz,vvel,ht,ts,ts2,dayt = get_arrays(str(c))
		# 	partition=read_partition.partition(dayt[0].year)
		# 	bbht,bbtimeidx=partition.get_bbht(time=dayt)
		# 	center = np.nanmax(bbht)+1.0
		# 	lin_var,log_var=layer_mean(dbz,ht,center=center)
		# 	str1=', center = '+'{:3.1f}'.format(center)
		# 	str2=' lin var = {:6d}, log var = {:5.1f}'.format(int(lin_var/1000.),log_var)
		# 	print 'case '+str(c).zfill(2)+str1+str2


		' Print single panels to pdf'
		'*******************************'
		ppdbz=PdfPages('sprof_dbz_cfac_signifcant.pdf')
		ppvvel=PdfPages('sprof_vvel_cfac_significant.pdf')
		ncases=range(1,15)
		# ncases=[3,7]
		for c in ncases:
			dbz,vvel,ht,ts,ts2,dayt = get_arrays(str(c))
			if c==3:
				cbar=True
			else:
				cbar=False		
			plot_cfac(dbz=dbz, hgt=ht, case=c,colorbar=cbar)
			ppdbz.savefig()
			plot_cfac(vvel=vvel,hgt=ht,case=c,colorbar=cbar)
			ppvvel.savefig()
			plt.close('all')
		ppdbz.close()
		ppvvel.close()

def plot_cfac(**kwargs):

	ht = kwargs['hgt']
	case=kwargs['case']
	if 'dbz' in kwargs:
		data=kwargs['dbz']
		bin_size=5 #[dBZ]
		bins=range(-20,70,bin_size)
		xlim=[-20,60]
		str_data='dBZ'
	elif 'vvel' in kwargs:
		data=kwargs['vvel']
		bin_size=2 #[m/s]
		bins=range(-10,6,bin_size)
		xlim=[-10,4]
		str_data='VVEL'
	cfac=[]
	for h in range(len(ht)):
		hist = np.histogram(data[h,:], bins=bins)[0]
		hist_normalized=hist/float(np.sum(hist))
		cfac.append(hist_normalized)
	cfac=np.asarray(cfac)

	xtlab=[str(x) for x in bins]
	X,Y = np.meshgrid(bins[:-1], ht)
	fig,ax = plt.subplots()
	cb_values=np.linspace(0.1,1,10)
	cf =ax.contourf(X,Y,cfac,cb_values)
	if kwargs['colorbar']:
		divider = make_axes_locatable(ax)
		cax = divider.append_axes("right", size="2%", pad=0.05)
		cbar = plt.colorbar(cf, ticks=cb_values,cax=cax)		 
	ax.set_ylim([0, 11])
	ax.set_xlim(xlim)
	plt.suptitle('Case '+str(case).zfill(2) +' - CFAC '+str_data)
	plt.subplots_adjust(top=0.98, bottom=0.05,left=0.04, right=0.93)
	plt.draw()

def layer_mean(vvel=None,dbz=None, height=None, depth=1.0, 
					center=False,bottom=False,plot=False,variance=False):

	ht=height
	
	if center:
		idx=np.where( (ht>center-depth/2) & (ht<center+depth/2))

	if is_numeric(bottom):
		idx=np.where( ht>bottom)

	if is_numeric(dbz):
		data=dbz
		data_layer=data[idx,:][0]
		data_layer_linear=np.power(10,data_layer/10.)
		linear_mean=np.nanmean(data_layer_linear,axis=0)
		lin_varian=np.nanvar(linear_mean)
		log_mean=np.nanmean(data_layer,axis=0)
		log_varian=np.nanvar(log_mean)		
	
	if is_numeric(vvel):
		data=vvel
		data_layer=data[idx,:][0]
		linear_mean=np.nanmean(data_layer,axis=0)
		lin_varian=np.nanvar(linear_mean)
		log_mean=np.copy(linear_mean)
		log_mean[:]=np.nan
		log_varian=np.nan

	if plot:
		' setup new figure '	
		fig = plt.figure(figsize=(8,8))

		' images axes'
		gs = gridspec.GridSpec(3,1)
		ax0 = plt.subplot(gs[0])
		ax1 = plt.subplot(gs[1])
		gs = gridspec.GridSpec(1,1)
		gs.update(top=0.3,right=0.88)
		ax2 = plt.subplot(gs[0])

		im1=ax0.imshow(data_layer,interpolation='none',cmap='nipy_spectral',vmin=-20,vmax=60,aspect='auto',origin='lower')
		add_colorbar(ax0,im1)
		im2=ax1.imshow(data_layer_linear,interpolation='none',cmap='nipy_spectral',aspect='auto',origin='lower')
		add_colorbar(ax1,im2)
		ax2.plot(linear_mean)	
		ax22=ax2.twinx()
		ax22.plot(log_mean,color='red')
		str_var_linear='Lin Var={:3.2f}'.format(lin_varian)
		str_var_log='\nLog Var={:3.2f}'.format(log_varian)
		ax2.text(0.05,0.85,str_var_linear+str_var_log,transform=ax2.transAxes)
		ax2.set_xlim([0, len(linear_mean)])

		ax0.invert_xaxis()
		ax1.invert_xaxis()
		ax2.invert_xaxis()
		plt.show(block=False)

	if variance :
		return lin_varian,log_varian
	else:
		return linear_mean,log_mean	
	

def plot_thsections_single(ht,dayt,**kwargs):

	matplotlib.rcParams.update({'font.size':20})
	fig,ax = plt.subplots()
	str_etop=''
	if 'dbz' in kwargs:
		im=ax.imshow(kwargs['dbz'],interpolation='none',cmap='nipy_spectral',vmin=-30,vmax=60,aspect='auto',origin='lower')
	elif 'vvel' in kwargs:
		orig_cmap = cm.bwr
		shifted_cmap = shiftedColorMap(orig_cmap, midpoint=0.7, name='shifted')
		im=ax.imshow(kwargs['vvel'],interpolation='none',cmap=shifted_cmap,vmin=-10,vmax=4,aspect='auto',origin='lower')

	if 'echotop' in kwargs:
		ax.plot(kwargs['echotop'][0],color='black')
		units=r'$\mathregular{km^{-2}}$'
		str_etop='etop_var: {:2.1f}'.format(np.nanvar(kwargs['echotop'][1]))
		str_etop+=units

	if 'bband' in kwargs:
		bbht=kwargs['bband'][0]
		idx_bbht=kwargs['bband'][1]
		f = interp1d(ht, range(len(ht)))
		ax.plot(idx_bbht, f(bbht),marker='o',linestyle='--',color='black')

	if kwargs['colorbar']:
		divider = make_axes_locatable(ax)
		cax = divider.append_axes("right", size="2%", pad=0.)
		cbar = plt.colorbar(im, cax=cax)

	format_yaxis(ax,ht,toplimit=11)
	format_xaxis(ax, dayt, labels=True,format='%H')
	ax.invert_xaxis()

	str_date=dayt[0].strftime('%Y-%b')
	str_tbeg=dayt[0].strftime(' %dT%H:00')+'-'
	str_tend=(dayt[-1]).strftime('%dT%H:00')
	c=kwargs['case']
	annot='Case '+str(c)+' '+str_date+str_tbeg+str_tend+' UTC'+' '+str_etop
	ax.text(0.02,0.02,annot,transform=ax.transAxes,fontsize=14)
	plt.subplots_adjust(top=0.98, bottom=0.05,left=0.04, right=0.91)
	plt.draw()

def timeserie_echotop(dbz,ht,plot,retrieve):

	(rows, cols) = dbz.shape
	echotop=[]
	h=ht[::-1]
	for c in range(cols):
		prof = dbz[::-1,c]
		thres_dbz=10.
		idx, val = next(((n,v) for (n,v) in enumerate(prof) if v>=thres_dbz), (False, np.nan))
		if idx:
			if retrieve == 'index':
				echotop.append(len(h)-idx)
			elif retrieve == 'km':
				echotop.append(h[idx])
		else:
			echotop.append(val)

	# ma=np.asarray(echotop)
	ma = moving_average(echotop,5)
	ma= np.around(ma,decimals=2)
	' remove edge values'
	ma[0:2]=np.nan
	ma[-3:-1]=np.nan
	if plot:
		fig,ax=plt.subplots()
		ax.plot(ma,linewidth=3)
		ax.plot(echotop)
		ax.invert_xaxis()
		plt.draw()
		return ma
	else:
		return ma


def plot_profile_variance(dbz,vvel,ht, ax,case,ncases):

	dbz_variance=[]
	vvel_variance=[]
	count_gates=[]
	global ti
	global n
	global colors 

	if n==0:
		# colors=sns.color_palette('hls',ncases)
		colors=sns.color_palette('Paired',ncases)

	for i in range(len(ht)):
		dbz_variance.append(np.nanvar(dbz[i,:]))
		vvel_variance.append(np.nanvar(vvel[i,:]))
		count_gates.append(vvel[i,:].size-np.sum(np.isnan(vvel[i,:])))

	inid=datetime(*(reqdates[case]['ini']+[0,0]))
	endd=datetime(*(reqdates[case]['end']+[0,0]))
	ti.append('\nCase '+case+': '+inid.strftime('%Y-%b %dT%H:%M')+endd.strftime(' - %dT%H:%M UTC'))

	if n<7:
		marker='None'
		# marker='o'
	else:
		marker='o'

	dbzv=[0,180]
	vvelv=[0,6]
	if np.any(ax):
		ax[0].plot(dbz_variance,ht,marker=marker,color=colors[n])
		ax[1].plot(vvel_variance,ht,marker=marker,color=colors[n])
		ax[2].plot(count_gates,ht,marker=marker,color=colors[n],label='case '+case)
		n+=1
	else:
		fig,ax=plt.subplots(1,3,sharey=True,figsize=(12,8))
		ax[0].plot(dbz_variance,ht,color=colors[n])
		ax[1].plot(vvel_variance,ht,color=colors[n])
		ax[2].plot(count_gates,ht,color=colors[n], label='case '+case)
		ax[0].set_ylabel('Height MSL [km]')
		ax[0].set_xlabel('Reflectivity [dBZ^2]')
		ax[1].set_xlabel('Vertical velocity [m2 s^-2]')
		ax[2].set_xlabel('Count good gates')
		ax[0].set_xlim(dbzv)
		ax[1].set_xlim(vvelv)
		n+=1
		return ax

	if n==ncases and ncases==4:
		plt.suptitle('SPROF time variance'+''.join(ti))
		plt.subplots_adjust(top=0.85, left=0.05, right=0.95, wspace=0.05)
		ax[2].legend(loc='lower left')		
	elif n==ncases and ncases>4:
		plt.suptitle('SPROF time variance')
		plt.subplots_adjust(top=0.9, left=0.05, right=0.95, wspace=0.06)
		ax[2].legend()		

	plt.draw()

def plot_thsections(dbz,vvel,ht,dayt,**kwargs):

	fig,ax = plt.subplots(2,1,sharex=True)

	'colormap for vvel'
	orig_cmap = cm.bwr
	shifted_cmap = shiftedColorMap(orig_cmap, midpoint=0.7, name='shifted')

	' add images and colorbar'
	im0=ax[0].imshow(dbz,interpolation='none',cmap='nipy_spectral',vmin=-20,vmax=60,aspect='auto',origin='lower')
	im1=ax[1].imshow(vvel,interpolation='none',cmap=shifted_cmap,vmin=-10,vmax=4,aspect='auto',origin='lower')
	divider0 = make_axes_locatable(ax[0])
	cax0 = divider0.append_axes("right", size="2%", pad=0.05)
	cbar0 = plt.colorbar(im0, cax=cax0)
	divider1 = make_axes_locatable(ax[1])
	cax1 = divider1.append_axes("right", size="2%", pad=0.05)
	cbar1 = plt.colorbar(im1, cax=cax1)

	if 'echotop' in kwargs:
		echot=kwargs['echotop']
		ax[0].plot(echot,color='k')

	format_yaxis(ax[0],ht)
	format_yaxis(ax[1],ht)

	format_xaxis(ax[1], dayt, minutes_tick=30, labels=True)
	ax[1].invert_xaxis()

	ax[0].set_ylabel('Hgt MSL [km]')
	ax[1].set_ylabel('Hgt MSL [km]')
	ax[1].set_xlabel(r'$\Leftarrow$'+' Time [UTC]')

	plt.subplots_adjust(hspace=0.05)
	plt.suptitle('SPROF observations. Date: '+dayt[0].strftime('%Y-%b'))
	plt.draw()

def plot_turbulence_spectra(dbz,vvel,ts,ts2,ht,dayt):

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

	' add images and colorbar'
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

def get_arrays(casenum):

	dbz,vvel, ht, dayt = read_sprof.retrieve_arrays(base_directory, casenum)

	reqdates = precip.request_dates_significant(casenum)
	idx_st=find_index(dayt,reqdates['ini'])
	idx_end=find_index(dayt,reqdates['end'])

	dayt=dayt[idx_st:idx_end+1]
	vvel=vvel[:,idx_st:idx_end+1]
	dbz=dbz[:,idx_st:idx_end+1]

	with np.errstate(invalid='ignore'):
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

	return dbz,vvel,ht,ts,ts2,dayt

def print_resolution(c):
	_,_,ht,_,_,dayt = get_arrays(str(c))

	hgt_diff = np.diff(ht)
	hgt_res=np.mean(hgt_diff)

	dayt_diff = np.diff(dayt)
	dayt_seconds = np.asarray([d.seconds for d in dayt_diff])
	dayt_res = np.mean(dayt_seconds)
	
	pout='Aver. vertical res {:3d} meters ; Aver. time res: {: 5.1f} seconds'
	print pout.format(int(hgt_res*1000), dayt_res)

def decimalf(x,pos):
	return '%2.0f' % x

def integerf(x,pos):
	print x
	return '{:d}'.format(int(x))

def decimalx(x,pos):
	# return ('{{:.{:1d}f}}'.format(int(np.maximum(-np.log10(x),0)))).format(x)
	return ('{{:.{:1d}f}}'.format(int(np.ceil(-np.log10(x))))).format(x)

def moving_average(interval, window_size):
	window = np.ones(int(window_size))/float(window_size)
	return np.convolve(interval, window, 'same')

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

def add_colorbar(ax,im):
	divider = make_axes_locatable(ax)
	cax = divider.append_axes("right", size="2%", pad=0.05)
	cbar = plt.colorbar(im, cax=cax)

main()