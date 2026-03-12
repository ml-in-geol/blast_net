import os
import sys
import obspy
import pyasdf
import numpy as np

from sys import argv
from mpi4py import MPI
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

def read_param_dict(inparam_file,debug=True):
    params = ConfigObj(inparam_file).dict()
    if debug:
        print('--------------------------------------------------------')
        print('coming from read_param_dict in get_data.py\n')
        print('inparam_file: {}'.format(inparam_file)) 
        print('params:')
        print(params)
        print('--------------------------------------------------------')
    return params

def main(params,debug=False):
    #--------------------------------------------------------------------
    #Initialize MPI
    #--------------------------------------------------------------------
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:


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
            #evtid = '{}.{:02d}.{:02d}.{:02d}.{:02d}.{:02d}.{:02d}'.format(year,month,day,hour,minute,second,fsec)
            event_dict[evtid] = {}
            event_dict[evtid]['latitude'] = latitude
            event_dict[evtid]['longitude'] = longitude
            event_dict[evtid]['depth'] = depth
            event_dict[evtid]['magnitude'] = magnitude
            event_dict[evtid]['etype'] = etype

        n_events = len(event_dict)
        splits = n_events

    else:
        n_events = None
        splits = None
        params = None
        event_dict = None
        #ds = None

    #--------------------------------------------------------------------
    #Loop through events
    #--------------------------------------------------------------------
    n_events = comm.bcast(n_events,root=0)
    splits = comm.bcast(splits,root=0)
    params = comm.bcast(params,root=0)
    event_dict = comm.bcast(event_dict,root=0)
    event_names = list(event_dict.keys())

    for i_event in range(rank,splits,size):

        ds = pyasdf.ASDFDataSet(params['filename'])

        print('coming from rank... {} (size {})'.format(rank,size))
        print('working on event {} ... ({}/{})'.format(event_names[i_event],i_event+1,n_events))
        iris = Client('iris')

        #open asdf dataset
        #--------------------------------------------------------------------

        #add metadata to event object
        #----------------------------------------------------------------
        event_name = event_names[i_event]
        e = Event()
        #e.resource_id = 'smi:local/{}#eventID'.format(event_name)
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
        if event_dict[event_name]['etype'] == 0:
            e.event_type = 'explosion'
        elif event_dict[event_name]['etype'] == 1:
            e.event_type = 'earthquake'
        e.event_type_certainty = 'known'

        ds.add_quakeml(e)

        #-----------------------------------------------------------
        #Get stations
        #-----------------------------------------------------------
        starttime = origin_time - float(params['preset'])
        endtime = origin_time + float(params['offset'])
        station_box = params['station_box']
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

        ds.add_stationxml(inv)

        for net in inv:
            for sta in net:

                #check if station is farther away than max_gcarc
                stla = sta.latitude
                stlo = sta.longitude
                evlo = longitude
                evla = latitude
                dist_m, baz, az = gps2dist_azimuth(stla,stlo,evla,evlo)
                dist_km = dist_m / 1000.

                #if dist_km  > float(params['max_dist']):
                #    print('skipping because distance is {}'.format(dist_km))
                #    continue

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

                    print(seis)
                    #ds.add_waveforms(seis,tag='raw_recording',event_id=e.resource_id)
                    ds.add_waveforms(seis,tag='raw_recording',event_id=e)

                else:
                    continue

        if i_event%size!=rank: continue

    if rank == 0:
        sys.exit()

#params = read_param_dict('./params.dat')
params = read_param_dict(argv[1])
main(params)
