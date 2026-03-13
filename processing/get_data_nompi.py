import os
import sys
import obspy
import pyasdf
import numpy as np

from sys import argv
from obspy import UTCDateTime
from obspy.clients.fdsn import Client
from configobj import ConfigObj
from obspy.geodetics import gps2dist_azimuth

from obspy.core.event.origin import Origin
from obspy.core.event.event import Event
from obspy.core.event.event import EventDescription
from obspy.core.event.source import FocalMechanism
from obspy.core.event.source import MomentTensor
from obspy.core.event.source import Tensor
from obspy.core.event.magnitude import Magnitude
from obspy.core.inventory.network import Network
from obspy.core.inventory.network import Station
from obspy.core.inventory.channel import Channel

#set up iris client
iris = Client('iris')

def read_param_dict(inparam_file):
    params = ConfigObj(inparam_file).dict()
    return params

def main(params,debug=False):

    #open dataset
    #--------------------------------------------------------------------
    ds = pyasdf.ASDFDataSet(params['filename'])

    #read events catalog
    #--------------------------------------------------------------------
    f_cat = open(params['catalog'],'r')
    lines = f_cat.readlines()
    n_events = len(lines)
    event_dict = {}

    for line in lines:
        items = line.strip().split()
        year = int(items[0])
        julday = int(items[1])
        month = int(items[2])
        day = int(items[3])
        hour = int(items[4])
        minute = int(items[5])
        second = int(items[6])
        fsec = int(items[7])
        latitude = float(items[8])
        longitude = float(items[9])
        depth = float(items[10])
        magnitude = float(items[11])
        etype = int(items[12])

        evtid = '{}:{:02d}:{:02d}:{:02d}:{:02d}:{:02d}.{:02d}'.format(year,month,day,hour,minute,second,fsec)
        event_dict[evtid] = {}
        event_dict[evtid]['latitude'] = latitude
        event_dict[evtid]['longitude'] = longitude
        event_dict[evtid]['depth'] = depth
        event_dict[evtid]['magnitude'] = magnitude
        event_dict[evtid]['etype'] = etype

    n_events = len(event_dict)
    event_names = list(event_dict.keys())

    #--------------------------------------------------------------------
    #Loop through events
    #--------------------------------------------------------------------
    for i_event in range(0,n_events):

        print('*******')
        print(i_event)
        print('*******')

        #add metadata to event object
        #----------------------------------------------------------------
        event_name = event_names[i_event]
        e = Event()
        e.resource_id = obspy.core.event.ResourceIdentifier()

        #description
        descr1 = EventDescription(text=event_name,type='earthquake name')
        e.event_descriptions = [descr1]

        #origins
        origin_time = UTCDateTime(event_name)
        evlo = event_dict[event_name]['longitude']
        evla = event_dict[event_name]['latitude']
        evdp = event_dict[event_name]['depth']
        o1 = Origin(time=origin_time,longitude=evlo,latitude=evla,depth=evdp)
        o1.resource_id = obspy.core.event.origin.ResourceIdentifier()
        e.origins = [o1]

        #magnitude
        m1 = Magnitude(mag=event_dict[event_name]['magnitude'],magnitude_type='Ml')
        m1.resource_id = obspy.core.event.magnitude.ResourceIdentifier()
        e.magnitudes = [m1]

        #event type
        etype = event_dict[event_name]['etype']
        if etype == 0:
            e.event_type = 'explosion'
            e.event_type_certainty = 'known'
        elif etype == 1:
            e.event_type = 'earthquake'
            e.event_type_certainty = 'known'
        elif etype == -1:
            # QuakeML/ObsPy enum value for unknown event type.
            e.event_type = 'not reported'
        else:
            e.event_type = 'not reported'
            print('warning: unrecognized etype {} for event {}; using not reported'.format(
                etype, event_name))

        #add quakeml
        ds.add_quakeml(e)

        #-----------------------------------------------------------
        #Get stations
        #-----------------------------------------------------------
        starttime = origin_time - float(params['preset'])
        endtime = origin_time + float(params['offset'])
        station_box = params['station_box']
        max_dist = float(params['max_dist'])
        box_bounds = station_box.split('/')
        minlatitude = float(box_bounds[0])
        maxlatitude = float(box_bounds[1])
        minlongitude = float(box_bounds[2])
        maxlongitude = float(box_bounds[3])

        if type(params['channel'] == list):
            channel = params['channel'][0]
            for chn in params['channel'][1:]:
                channel = channel +',{}'.format(chn)
        else:
            channel = params['channel']

        inv = iris.get_stations(starttime = starttime, endtime = endtime,
                network = params['network'], channel = channel,
                minlatitude=minlatitude,maxlatitude=maxlatitude,
                minlongitude=minlongitude,maxlongitude=maxlongitude,level='response')

        #add station metadata
        ds.add_stationxml(inv)

        for net in inv:
            for sta in net:

                stla = sta.latitude
                stlo = sta.longitude
                dist_m, baz, az = gps2dist_azimuth(stla,stlo,evla,evlo)
                dist_km = dist_m / 1000.

                if dist_km > max_dist:
                    print('skipping {}.{} because distance is {:.1f} km (> {:.1f} km)'.format(
                        net.code, sta.code, dist_km, max_dist))
                    continue

                seis = obspy.Stream()

                #try to get BH? data
                try:
                    seis = iris.get_waveforms(net.code,sta.code,'*',"BH*",starttime,endtime)
                except:
                    try:
                        seis = iris.get_waveforms(net.code,sta.code,'*',"HH*",starttime,endtime)
                    except:
                        try:
                            seis = iris.get_waveforms(net.code,sta.code,'*',"EH*",starttime,endtime)
                        except:
                            print('could not find any data for BH?,HH?, or EH?, for station {}.{}'.format(net.code,sta.code))

                if len(seis) > 1:

                    ds.add_waveforms(seis,tag='raw_recording',event_id=e)

                else:
                    continue

    ds.flush()
    ds._close()
    ds._ASDFDataSet__file = None

#params = read_param_dict('./params.dat')
params = read_param_dict(argv[1])
main(params)
