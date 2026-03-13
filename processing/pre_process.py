import pyasdf
import numpy as np
from sys import argv
from obspy.geodetics import gps2dist_azimuth

input_file = argv[1]
output_file = argv[2]

ds = pyasdf.ASDFDataSet(input_file)
ds_out = pyasdf.ASDFDataSet(output_file)

pre_filt= (0.005,0.01,18.0,20.0)

#req_time = 150.0
req_time = 240.0
sampling_rate = 40.0
#preset = 30.0
preset = 120.

npts = int(req_time*sampling_rate)

for event in ds.events:

    print('working on event {}'.format(event))

    origin = event.preferred_origin() or event.origins[0]
    event_latitude = origin.latitude
    event_longitude = origin.longitude
    starttime = origin.time - preset

    distance_dict = {}
    distances = []

    ds_out.add_quakeml(event)

    for station in ds.ifilter(ds.q.event == event):

        inv = station.StationXML
        if inv is None:
            continue
        
        station_latitude = station.coordinates['latitude']
        station_longitude = station.coordinates['longitude']

        seis = station.raw_recording
        print('seis: ')
        print(seis)

        #check that all traces have the same sampling rate
        for tr in seis:
            if tr.stats.sampling_rate != seis[0].stats.sampling_rate:
                continue

        #remove instrument response
        seis.detrend('linear')
        seis.detrend('demean')
        seis.taper(max_percentage=0.05,type='hann')
        seis.attach_response(inv)

        try:
            seis.remove_response(output="DISP", pre_filt=pre_filt, zero_mean=False, taper=False)
        except:
            print('COULDNT REMOVE RESPONSE...')
            continue

        seis.detrend('linear')
        seis.detrend('demean')
        seis.taper(max_percentage=0.05,type='hann')

        if len(seis) > 3:
            try:
                seis.merge()
                assert(len(seis) == 3)
            except:
                print('----------------------------')
                print('skipping stream, len > 3, cant merge')
                print(seis)
                print('----------------------------')

        print('seis after merge: ')
        print(seis)

        seis.resample(sampling_rate)

        #resample to new sampling rate
        try:
            seis.interpolate(sampling_rate=sampling_rate, starttime=starttime, npts=npts)
        except:
            print('interpolation failed for {}... skipping'.format(seis[0].stats.station))
            print(seis)
            for tr in seis:
                print(tr.stats.npts)
            continue
        #seis.resample(sampling_rate)

        #calculate distance and rotate
        dist_m, baz, az = gps2dist_azimuth(station_latitude,
                                           station_longitude,
                                           event_latitude,
                                           event_longitude)

        components = [tr.stats.channel[-1] for tr in seis]
        if "N" in components and "E" in components:
            seis.rotate(method="NE->RT", back_azimuth=baz)

        elif "1" in components and "2" in components:

            print('rotation from H1/N2 -> NE for {}'.format(seis[0].id))
            try:
                seis.rotate(method="->ZNE",inventory=inv)
            except:
                continue
            print(seis)
            seis.rotate(method="NE->RT",back_azimuth=baz)
            print(seis)
            print('*************************\n')

        print('seis here:')
        print(seis)
        for tr in seis:
            tr.stats.distance = dist_m / 1000.0
            print('distance {}'.format(tr.stats.distance))

        #add processed waveform
        #ds.add_waveforms(seis,event_id=event,tag='processed')
        ds_out.add_waveforms(seis,event_id=event,tag='processed')

        #add station
        ds_out.add_stationxml(inv)

        #build distance dict
        distance_dict['{}.{}'.format(tr.stats.network,tr.stats.station)] = dist_m / 1000.0
        distances.append(dist_m / 1000.)

    #ds.add_auxiliary_data(data = np.array(distances), data_type='distances', path = 'distances/{}'.format(origin.time), parameters = distance_dict)
    ds_out.add_auxiliary_data(data = np.array(distances), data_type='distances', path = 'distances/{}'.format(origin.time), parameters = distance_dict)

#ds.add_auxiliary_data(data = distances, data_type='distances', path='distances')
ds.flush()
ds._close()
ds_out.flush()
ds_out._close()
ds._ASDFDataSet__file = None
ds_out._ASDFDataSet__file = None
