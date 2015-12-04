""" Make statistical analyses of NOAA S-profiler radar

	Raul Valenzuela
	December, 2015
	raul.valenzuela@colorado.edu
"""

import matplotlib.pyplot as plt
import read_sprof 
import numpy as np

from common import find_index

base_directory='/home/rvalenzuela/SPROF/matfiles'
casenum = raw_input('\nIndicate case number (i.e. 1): ')

reqdates={ '1': {'ini':[1998,1,18,14],'end':[1998,1,19,23]},
			'2': {'ini':[1998,1,26,0],'end':[1998,1,27,4]},
			'3': {'ini':[2001,1,23,21],'end':[2001,1,24,1]},
			'7': {'ini':[2001,2,17,17],'end':[2001,2,17,19]},
			'9': {'ini':[2003,1,21,0],'end':[2003,1,23,5]},
			'13': {'ini':[2004,2,17,12],'end':[2004,2,18,0]},
			'14': {'ini':[2004,2,25,0],'end':[2004,2,26,0]}
			}

dbz,vvel, ht, dayt = read_sprof.retrieve_arrays(base_directory, casenum)

idx_st=find_index(dayt,reqdates[casenum]['ini'])
idx_end=find_index(dayt,reqdates[casenum]['end'])


''' imshow '''
fig,ax=plt.subplots(2,1,sharex=True)
ax[0].imshow(vvel[:,idx_st:idx_end],interpolation='none',cmap='bwr',vmin=-3,vmax=3,aspect='auto',origin='lower')
for x in [40,50,60]:
	ax[0].plot([x]*len(vvel[x,idx_st:idx_end]))

''' lines '''
for x in [40,50,60]:
	trace=vvel[x,idx_st:idx_end]
	trace[trace<-6.]=np.nan
	ax[1].plot(trace)

ax[1].set_xlim([0, len(vvel[0,idx_st:idx_end])])
ax[1].invert_xaxis()

plt.draw()

plt.show(block=False)


