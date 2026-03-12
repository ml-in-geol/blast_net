import obspy
import pyasdf
import numpy as np
import matplotlib.pyplot as plt
from sys import argv

input_file = argv[1]
output_file = argv[2]

ds = pyasdf.ASDFDataSet(input_file)
fmin = 10.0
fmax = 18.0
snr_thresh = 2.0
psr_max = 5.0
dist_max = 250.0

ex_ps = []
eq_ps = []

ex_dist = []
eq_dist = []

n_events = len(ds.events)

for i_ev,event in enumerate(ds.events):

    print('working on event {} of {} ({})'.format(i_ev,n_events,event.event_type))

    origin = event.preferred_origin() or event.origins[0]
    event_name = '{}'.format(origin.time)

    ps_all = []
    dist_all = []
    snr_all = []
    e_types = []
    ps_event = []

    try:
        distances = ds.auxiliary_data.distances.distances[event_name].parameters
        ps_ratios = ds.auxiliary_data.PS_ratios['{}'.format(event_name)]['f_{:2.2f}_{:2.2f}'.format(fmin,fmax)].parameters
        snr = ds.auxiliary_data.SNR['{}'.format(event_name)]['f_{:2.2f}_{:2.2f}'.format(fmin,fmax)].parameters

    except:
        print('Either distances, ps_ratios, or SNR doesnt exist')
        continue

    for station in ds.ifilter(ds.q.event == event):

        seis = station.processed
        net_code = seis[0].stats.network
        sta_code = seis[0].stats.station

        #ps_here = ps_ratios['{}.{}'.format(net_code,sta_code)]
        #dist_here = distances['{}.{}'.format(net_code,sta_code)]
        #snr_here = snr['{}.{}'.format(net_code,sta_code)]

        try:
            ps_here = ps_ratios['{}.{}'.format(net_code,sta_code)]
            dist_here = distances['{}.{}'.format(net_code,sta_code)]
            snr_here = snr['{}.{}'.format(net_code,sta_code)]

            if snr_here >= snr_thresh and ps_here < psr_max and dist_here < dist_max:
                print("SHOULD HAVE ADDED")
                print(dist_here)
                if event.event_type == 'explosion':
                    ex_dist.append(dist_here)
                elif event.event_type == 'earthquake':
                    eq_dist.append(dist_here)

                dist_all.append(dist_here)
                ps_all.append(ps_here)
                snr_all.append(snr_here)

                #add psratio to event
                ps_event.append(ps_here)

        except:
            print('cant find ratios for {}.{}'.format(net_code,sta_code))
            continue

    if len(ps_event) > 0: 
        #ps_avg = np.average(ps_event)
        ps_avg = np.median(ps_event)
    else:
        print('NO DATA FOR EVENT')
        continue

    if event.event_type == 'explosion':
        ex_ps.append(ps_avg)
    elif event.event_type == 'earthquake':
        eq_ps.append(ps_avg)

fig = plt.figure(figsize=[5,7])
ax1 = fig.add_axes([0.1,0.6,0.8,0.35])
ax2 = ax1.twinx()
ax3 = fig.add_axes([0.1,0.15,0.8,0.35])
ax4 = ax3.twinx()

bins = np.linspace(0.0,3.0,20);
#ax1.hist(ex_ps,bins=bins,rwidth=0.5,label='explosion',alpha=0.77,color='C0')
#ax2.hist(eq_ps,bins=bins,rwidth=0.5,label='earthquake',alpha=0.77,color='C1')
ax1.hist(ex_ps,bins=bins,rwidth=0.75,alpha=0.77,color='C0',align='mid',label='borehole')
ax2.hist(eq_ps,bins=bins,rwidth=0.75,alpha=0.77,color='C1',align='left',label='earthquake')
#hists = hist1+hist2
#labs = [l.get_label() for l in hists]
#ax1.legend(lns, labs,location='upper right')

ax1.tick_params(axis='y', labelcolor='C0')
ax2.tick_params(axis='y', labelcolor='C1')
ax1.set_xlabel('P/S ratio')

bins = np.linspace(0,360,18)
ax3.hist(ex_dist,bins=bins,rwidth=0.75,alpha=0.77,color='C0',align='mid')
ax4.hist(eq_dist,bins=bins,rwidth=0.75,alpha=0.77,color='C1',align='left')
ax3.tick_params(axis='y', labelcolor='C0')
ax4.tick_params(axis='y', labelcolor='C1')
ax3.set_xlabel('distance (km)')

#plt.legend()
#plt.tight_layout()
#plt.show()
plt.savefig(output_file)
