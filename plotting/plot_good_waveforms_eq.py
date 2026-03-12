import obspy
import pyasdf
import numpy as np
import matplotlib.pyplot as plt
from sys import argv
import matplotlib
import random
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['font.family'] = 'Helvetica'

input_file = argv[1]
output_dir = argv[2]
output_name = argv[3]
outfile = '{}/{}'.format(output_dir,output_name)

preset = 120.1
offset = 120.1
ds = pyasdf.ASDFDataSet(input_file)

#fmin = 10.0
#fmax = 18.0
fmin = 0.1
fmax = 18.0

snr_thresh = 2.0
psr_max = 5.0
sf = 10.0
#alpha = 0.35

dist_max = 250.

ps_all = []
dist_all = []
snr_all = []
e_types = []

p_times = []
s_times = []
dists = []

times = []
signals = []

n_good = 0
plot_bad = False

fig,ax = plt.subplots(figsize=[3,5])

for event in ds.events:

    origin = event.preferred_origin() or event.origins[0]
    event_name = '{}'.format(origin.time)

    print(event.event_type)
    if event.event_type == 'explosion':
        continue

    try:
        distances = ds.auxiliary_data.distances.distances[event_name].parameters
        ps_ratios = ds.auxiliary_data.PS_ratios['{}'.format(event_name)]['f_{:2.2f}_{:2.2f}'.format(fmin,fmax)].parameters
        snr = ds.auxiliary_data.SNR['{}'.format(event_name)]['f_{:2.2f}_{:2.2f}'.format(fmin,fmax)].parameters
        P_times = ds.auxiliary_data.travel_times.P_times[event_name].parameters
        S_times = ds.auxiliary_data.travel_times.S_times[event_name].parameters
    except:
        continue

    print(P_times)

    for station in ds.ifilter(ds.q.event == event):

        seis = station.processed
        seis.filter('bandpass',freqmin=fmin,freqmax=fmax,corners=2,zerophase=True)
        net_code = seis[0].stats.network
        sta_code = seis[0].stats.station

        try:
            ps_here = ps_ratios['{}.{}'.format(net_code,sta_code)]
            dist_here = distances['{}.{}'.format(net_code,sta_code)]
            snr_here = snr['{}.{}'.format(net_code,sta_code)]
            P_time = P_times['{}.{}'.format(net_code,sta_code)]
            S_time = S_times['{}.{}'.format(net_code,sta_code)]

            if dist_here > dist_max:
                continue

            #if snr_here >= snr_thresh and ps_here < psr_max:
            if snr_here >= snr_thresh:

                n_good += 1

                tr = seis.select(channel='*HZ')[0]
                tr.normalize()
                time = tr.times()
                data = (tr.data*sf)+dist_here
                times.append(time-preset)
                signals.append(data)
                #ax.plot(time-preset,data,alpha=alpha,linewidth=0.5,color='k')

                p_times.append(P_time)
                s_times.append(S_time)
                dists.append(dist_here)

            #--------------------------------------------------------
            #DONT USE
            #--------------------------------------------------------
            else:
                if plot_bad:
                    tr = seis.select(channel='*HZ')[0]
                    tr.normalize()
                    time = tr.times()
                    data = (tr.data*sf)+dist_here
                    ax.plot(time,data,alpha=0.5,linewidth=0.5,color='r')

        except:
            #print('cant find ratios for {}.{}'.format(net_code,sta_code))
            continue

nsignals = len(signals)
#alpha = (1./nsignals)*50
alpha=0.1

signals_max = 500
for i,signal in enumerate(signals):
    if i > signals_max:
        continue
    else:
        ax.plot(times[i],signal,alpha=alpha,linewidth=0.25,color='k')
        #ax.plot(times[i],signal,alpha=alpha,linewidth=0.5,color='k',rasterized=True)

dists_p_times = zip(dists,p_times)
dists_s_times = zip(dists,s_times)
sorted_p_times = sorted(dists_p_times,key=lambda x: x[0])
sorted_s_times = sorted(dists_s_times,key=lambda x: x[0])
sorted_dists_p, sorted_p = zip(*sorted_p_times)
sorted_dists_s, sorted_s = zip(*sorted_s_times)

stride=15
ax.plot(sorted_p[::stride],sorted_dists_p[::stride],c='r',linewidth=1,alpha=0.75,linestyle='dashed')
ax.plot(sorted_s[::stride],sorted_dists_s[::stride],c='b',linewidth=1,alpha=0.75,linestyle='dashed')

ax.set_ylim([-5,265])
ax.set_xlim([0,100])
ax.set_title('{} waveforms'.format(n_good))
ax.set_xlabel('time (s)')
ax.set_ylabel('distance (km)')
plt.tight_layout()
#plt.show()
plt.savefig(outfile,transparent=True)
#plt.savefig(outfile,transparent=False,dpi=300)
