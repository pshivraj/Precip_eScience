#import the different packages used throughout
import xarray as xr
import numpy as np
import pandas as pd
import glob
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.metrics import pairwise_distances, davies_bouldin_score
from mpl_toolkits.basemap import Basemap
from bayes_opt import BayesianOptimization
import math
import boto3
import os
from os.path import expanduser
import json
import time
import logging
logging.basicConfig(filename='trmm.log', level=logging.INFO)

def save_s3_data(labels,eps,minSamples,Data,Time):
    #package the matrices as a dataset to save as a netcdf
    data_events = xr.Dataset(
        data_vars = {'Data': (('time', 'vector'),Data), 
                     'Labels': (('time'),labels)},
        coords = {'time': Time,
                  'vector': range(len(Data[0,:]))},
        attrs = {'eps': eps,
                'minimumSamples': minSamples})

    #save as a netcdf
    data_events.to_netcdf(path = "SortedData.nc4", compute = True)
    
    home = expanduser("~")

    with open(os.path.join(home,'creds.json')) as creds_file:
        creds_data = json.load(creds_file)

    #Access from S3
    s3 = boto3.resource('s3',aws_access_key_id=creds_data['key_id'],
             aws_secret_access_key=creds_data['key_access'],region_name='us-west-2')
    bucket = s3.Bucket('himatdata')
    home = os.getcwd()
    
    bucket.upload_file('SortedData.nc4','Trmm/EPO/2000_01')

#function that connects to the S3 bucket, downloads the file, reads in the data, and deletes the file
def load_s3_data(SR_minrate):
    #create empty matrices to hold the extracted data
    Lat_Heat = []
    surf_r = []
    LAT = []
    LON = []
    TIME = []
    count = 0
    
    home = expanduser("~")

    with open(os.path.join(home,'creds.json')) as creds_file:
        creds_data = json.load(creds_file)

    #Access from S3
    s3 = boto3.resource('s3',aws_access_key_id=creds_data['key_id'],
             aws_secret_access_key=creds_data['key_access'],region_name='us-west-2')
    bucket = s3.Bucket('himatdata')
    home = os.getcwd()

    for obj in bucket.objects.filter(Delimiter='', Prefix='Trmm/EPO/2000_01'):
        bucket.download_file(obj.key,os.path.join(os.path.join(home,'S3_downloads/',obj.key[17:])))
        logging.info("Downloaded file: %s", obj.key[17:])

    #file = 'oneProfile/TPR7_uw1_00538.19980101.000558_EPO.nc4'
        L, S, A, la, lo, Ti = extract_data(xr.open_dataset(os.path.join(home,'S3_downloads/',obj.key[17:])),SR_minrate)
       #append the new data in the matrices
        if count==0:
            Lat_Heat = L
            LAT = la[:,0]
            LON = lo[:,0]
            TIME = Ti
            count += 1
        else:
            Lat_Heat = np.concatenate((Lat_Heat,L),axis =0)
            LAT = np.concatenate((LAT,la[:,0]),axis =0)
            LON = np.concatenate((LON,lo[:,0]),axis =0)
            TIME = np.concatenate((TIME,Ti),axis =0)
        surf_r = np.append(surf_r,S)
        
        #delete the local file
        os.remove(os.path.join(home,'S3_downloads/',obj.key[17:]))
        

    #Put all the data into one array, where rows are individual observations and the columns are 
    #[Latitude, Longitude, Surface Rain, Latent Heat Profile]
    Data = np.concatenate((LAT.reshape(len(LAT),1),LON.reshape(len(LON),1),surf_r.reshape(len(surf_r),1),Lat_Heat),axis=1)
    Data = np.squeeze(Data)
    
    #Remove repeated values
    uniqueData = np.unique(Data,axis=0)
    
    return uniqueData, TIME, A
    
#Translate the time into delta time since the first datapoint (in hours)
def time_to_deltaTime(Time):
    InitialTime = np.min(Time)
    DeltaTime = []
    DeltaTime[:] = [int(x-InitialTime)/(10**9*60*60) for x in Time] #from nanoseconds to hours
    DeltaTime = np.array(DeltaTime) #convert from list to array
    
    return DeltaTime

#Create array to Cluster the rainfall events, Scale the grid lat/lon so it is weighted 'fairly' compared to time
def data_to_cluster(Data):
    #Extract [Lat, Lon, DeltaTime]
    Xdata = np.vstack((Data[:,1],Data[:,2],Data[:,0]))
    Xdata = Xdata.T
    return Xdata

def cluster_and_label_data(Distance,eps,min_samps):
    model = DBSCAN(eps=eps, min_samples=min_samps,metric='precomputed')
    model.fit(Distance)

    labels = model.labels_
    
    return labels

#Use Bayesian Optimization on the data to get the best parameters for the clustering
def optimal_params(Data):
    Opt = optimize_dbscan(Data,'davies') #it seems like silhouette takes substantially longer?
    min_samps = int(Opt['params']['min_samp'])
    eps = Opt['params']['EPS']

    return eps, min_samps

#function that fits dbscan for given parameters and returns the davies bouldin score evaluation metric 
def dbscan_eval_db(eps,min_samples,data):
    model = DBSCAN(eps=eps, min_samples=min_samples,metric='precomputed')
    model.fit(data)
    labels = model.labels_
    if len(set(labels))<2:
        score = 0
    else:
        score = davies_bouldin_score(data, labels)
        
    return score

#function that fits dbscan for given parameters and returns the silhouette score evaluation metric 
def dbscan_eval_sil(eps,min_samples,data):
    model = DBSCAN(eps=eps, min_samples=min_samples,metric='precomputed')
    model.fit(data)
    labels = model.labels_
    if len(set(labels))<2:
        score = 0
    else:
        score = metrics.silhouette_score(data,labels)
        
    return score

#Applies bayesian optimization to determine DBSCAN parameters that maximize the evaluation metric (specified as input)
def optimize_dbscan(data,metric='silhouette'):
    """Apply Bayesian Optimization to DBSCAN parameters."""
    def dbscan_evaluation_sil(EPS, min_samp):
        """Wrapper of DBSCAN evaluation."""
        min_samp = int(min_samp) #insure that you are using an integer value for the minimum samples parameter
        return dbscan_eval_sil(eps=EPS, min_samples=min_samp, data=data)

    def dbscan_evaluation_db(EPS, min_samp):
        """Wrapper of DBSCAN evaluation."""
        min_samp = int(min_samp) #insure that you are using an integer value for the minimum samples parameter
        return dbscan_eval_db(eps=EPS, min_samples=min_samp, data=data)

    if metric == 'davies':
        optimizer = BayesianOptimization(
            f=dbscan_evaluation_db,
            pbounds={"EPS": (100, 600), "min_samp": (5, 30)}, #bounds on my parameters - these are very rough guesses right now
            random_state=1234,
            verbose=2
        )
        
    else:
        optimizer = BayesianOptimization(
            f=dbscan_evaluation_sil,
            pbounds={"EPS": (100, 600), "min_samp": (5, 30)}, #bounds on my parameters - these are very rough guesses right now
            random_state=1234,
            verbose=2
        )
    
    optimizer.maximize(n_iter=10)

    logging.info("Final Result: %s", optimizer.max)
    return optimizer.max

#This function takes in a file name (downloaded from trmm.atmos.washington.edu) and extracts the variables
#that I care about (latitude, longitude, altitude, surface rain, latent heat). It does the inital data checks and
#throws out profiles with missing information or minimal rainfall. It returns the variables I care about that pass these
#checks

def extract_data(file, SR_min=5):
    #Extract the data you want from file
    altitude_lh = file.altitude_lh.data
    surf_rain = file.surf_rain.data
    latent_heating = file.latent_heating.data

    lat = file.latitude.data
    lon = file.longitude.data
    time = file.time.data
    
    #create grid of altitude, lat, and lon coordinates
    LAT, ALTITUDE, LON = np.meshgrid(lat, altitude_lh, lon)

    #size of lat and lon as variables
    nlat = len(lat)
    nlon = len(lon)
    nalt = len(altitude_lh)

    #reshape as column vector (note the indicing is now column*ncolumns+row)
    surf_rain = np.reshape(surf_rain,[nlat*nlon])
    LH = np.reshape(latent_heating,[nalt,nlat*nlon])
    ALTITUDE = np.reshape (ALTITUDE,[nalt,nlat*nlon])
    LON = np.reshape (LON,[nalt,nlat*nlon])
    LAT = np.reshape (LAT,[nalt,nlat*nlon])

    #Remove values with NaN and rainfall less than cut-off
    surf_R = surf_rain[~np.isnan(surf_rain)]
    surf_r = surf_R[surf_R>=SR_min]

    Lat_Heat = LH[:,~np.isnan(surf_rain)]
    Lat_Heat = Lat_Heat[:,surf_R>=SR_min]
    Lat_Heat = np.squeeze(Lat_Heat)

    ALTITUDE = ALTITUDE[:,~np.isnan(surf_rain)]
    ALTITUDE = ALTITUDE[:,surf_R>=SR_min]
    ALTITUDE = np.squeeze(ALTITUDE)

    LAT = LAT[:,~np.isnan(surf_rain)]
    LAT = LAT[:,surf_R>=SR_min]
    LAT = np.squeeze(LAT)

    LON = LON[:,~np.isnan(surf_rain)]
    LON = LON[:,surf_R>=SR_min]
    LON = np.squeeze(LON)

    #Remove any profiles where there is missing latent heat info
    surf_r = surf_r[~pd.isnull(Lat_Heat).any(axis=0)]
    LAT = LAT[:,~pd.isnull(Lat_Heat).any(axis=0)]
    LON = LON[:,~pd.isnull(Lat_Heat).any(axis=0)]
    ALTITUDE = ALTITUDE[:,~pd.isnull(Lat_Heat).any(axis=0)]
    Lat_Heat = Lat_Heat[:,~pd.isnull(Lat_Heat).any(axis=0)]
    Time = np.repeat(time,len(surf_r))
    
    return Lat_Heat.T, surf_r.T, ALTITUDE.T, LAT.T, LON.T, Time.T

#calcuate the distance (in degrees) between 2 points in lat/long
def lat_long_to_arc(lat1,long1,lat2,long2):
    # Convert latitude and longitude to
    # spherical coordinates in radians.
    degrees_to_radians = math.pi/180.0

    # phi = 90 - latitude
    phi1 = (90.0 - lat1)*degrees_to_radians
    phi2 = (90.0 - lat2)*degrees_to_radians

    # theta = longitude
    theta1 = long1*degrees_to_radians
    theta2 = long2*degrees_to_radians

    # Compute spherical distance from spherical coordinates.

    # For two locations in spherical coordinates
    # (1, theta, phi) and (1, theta', phi')
    # cosine( arc length ) =
    # sin phi sin phi' cos(theta-theta') + cos phi cos phi'
    # distance = rho * arc length

    cos = (math.sin(phi1)*math.sin(phi2)*math.cos(theta1 - theta2) +
    math.cos(phi1)*math.cos(phi2))
    if cos>1: cos=1
    if cos<-1: cos=-1
    arc = math.acos( cos )

    return arc

def create_distance_matrix(Data,FrontSpeed,Rad_Earth):
    Scale_Time_to_Distance = FrontSpeed

    Distance = np.zeros((len(Data),len(Data)))
    for i in range(len(Data)):
        for j in range(i,len(Data)):
            d = Rad_Earth*lat_long_to_arc(Data[i,0],Data[i,1],Data[j,0],Data[j,1])
            D = math.sqrt(d**2+(Scale_Time_to_Distance*(Data[i,2]-Data[j,2]))**2)
            Distance[i,j] = D
            Distance[j,i] = D
    return Distance

if __name__ == '__main__':
	#Define Key Values Here
	start_time = time.time()
    SR_minrate = 5 #only keep data with rainrate greater than this value
    opt_frac = .05 #fraction of data to use when determining the optimal dbscan parameters
    Rad_Earth = 6371 #km earth's radius
    MesoScale = 200 #Mesoscale is up to a few hundred km'
    FrontSpeed = 30 # km/h speed at which a front often moves

    Data, Time, A = load_s3_data(SR_minrate)
    DeltaTime = time_to_deltaTime(Time)
    
    Data = np.concatenate((DeltaTime.reshape(len(DeltaTime),1), Data), axis=1)
    Data = np.squeeze(Data)
    
    DatatoCluster = data_to_cluster(Data)
    
    Distance = create_distance_matrix(DatatoCluster,FrontSpeed,Rad_Earth)
    
    eps, minSamples = optimal_params(Distance[0:int(len(DatatoCluster)*opt_frac),0:int(len(DatatoCluster)*opt_frac)])
    
    labels = cluster_and_label_data(Distance,eps,minSamples)
    
    save_s3_data(labels,eps,minSamples,Data,Time)
    print("Done")
    print("--- %s seconds ---" % (time.time() - start_time))
