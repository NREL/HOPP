'''
Functions for clustering weather data and electricity pricing, and calculations of full-year or cluster-average values
'''

import numpy as np
import pandas as pd
import datetime

from pvlib_cspopt.location import Location      # Use pvlib instead     https://pvlib-python.readthedocs.io/en/stable/installation.html?highlight=installation#standardrelease
from pvlib_cspopt import solarposition


class cluster:
    
    def __init__(self):                     
        self.algorithm = 'affinity-propagation'  # Clustering algorithm     
        self.Ncluster = 40                       # Number of clusters 
        self.Nmaxiter = 200                      # Maximum iterations for clustering algorithm  
        self.sim_hard_partitions = True          # Use hard partitioning for simulation weighting factors?  
        
        self.afp_preference_mult = 1.0           # Multiplier for default preference values (median of input similarities = negative Euclidean distance b/w points i and k) --> Larger multiplier = fewer clusters
        self.afp_Nconverge = 10                  # Number of iterations without change in solution for convergence
        
        self.afp_enforce_Ncluster = False        # Iterate on afp_preference_mult to create the number of clusters specified in Ncluster
        self.afp_enforce_Ncluster_tol = 1        # Tolerance for number of clusters 
        self.afp_enforce_Ncluster_maxiter = 50   # Maximum number of iterations 
        

    def form_clusters(self,data):
        clusters = {}
        Ngroup = data.shape[0]
        
        if Ngroup == 1:
            clusters['Ncluster'] = 1
            clusters['wcss'] = 0.0
            clusters['index'] = np.zeros((Ngroup),int)
            clusters['count'] = np.ones((1),int)
            clusters['means'] = np.ones((1,data.shape[1])) * data
            clusters['partition_matrix'] = np.ones((1,1))
            clusters['exemplars'] = np.zeros((1),int)
            clusters['data_pts'] = np.zeros((1,1),int)
            return clusters
            
        else:

            if self.afp_preference_mult == 1.0:  # Run with default preference
                pref = None
            else:
                distsqr = []
                for g in range(Ngroup):
                    dist = ((data[g,:]-data[g:Ngroup,:])**2).sum(1)
                    distsqr = np.append(distsqr,-dist)
                pref = (np.median(distsqr)) * self.afp_preference_mult

            alg = affinity_propagation(max_iter = self.Nmaxiter,  convergence_iter = self.afp_Nconverge, preference = pref)
            alg.fit_predict(data)
            clusters['index'] = alg.cluster_index
            clusters['Ncluster'] = alg.n_clusters
            clusters['means'] = alg.cluster_means
            clusters['wcss'] = alg.wcss
            clusters['exemplars'] = alg.exemplars
        
            Ncluster = clusters['Ncluster']
            clusters['count'] = np.zeros((Ncluster),int)                # Number of data points nominally assigned to each cluster 
            clusters['partition_matrix'] = np.zeros((Ngroup,Ncluster))  
            
            for k in range(Ncluster):
                clusters['count'][k] = np.sum(clusters['index'] == k)
                
            if self.sim_hard_partitions:
                inds = np.arange(Ngroup)
                clusters['partition_matrix'][inds, clusters['index'][inds]] = 1.0
                    
            else:  # Compute "fuzzy" partition matrix
                distsqr = np.zeros((Ngroup,Ncluster))
                for k in range(Ncluster):
                    distsqr[:,k] = ((data-clusters['means'][k,:])**2).sum(1)  # Squared distance between all data points and cluster mean k
                distsqr[distsqr == 0] = 1.e-10          
                sumval = (distsqr**(-2./(self.mfuzzy-1))).sum(1)    # Sum of dik^(-2/m-1) over all clusters k
                for k in range(Ncluster):
                    clusters['partition_matrix'][:,k] = ( distsqr[:,k]**(2./(self.mfuzzy-1)) * sumval)**-1
                    
            clusters['weights'] = clusters['partition_matrix'].sum(0)/Ngroup     # Sum of wij over all data points (i) / Ngroup

            return clusters    
  

#============================================================================
#============================================================================ 
# Affinity propagation algorithm
class affinity_propagation:
    
    def __init__(self, damping = 0.5, max_iter = 300,  convergence_iter = 10, preference = None):
        self.damping = damping                      # Damping factor for update of responsibility and availability matricies (0.5 - 1)
        self.max_iter = max_iter                    # Maximum number of iterations
        self.convergence_iter = convergence_iter    # Number of iterations without change in clusters or exemplars to define convergence
        self.preference = preference                # Preference for all data points to serve as exemplar.  If None, the preference will be set to the median of the input similarities
        self.random_seed = 123

    
    def compute_wcss(self, data, cluster_index, means):
        nclusters = means.shape[0]
        wcss = 0.0   
        for k in range(nclusters):
            dist = ((data-means[k,:])**2).sum(1)   # Distance to cluster k centroid
            wcss += (dist * (cluster_index == k)).sum() 
        return wcss
    
    def fit_predict(self, data):
        nobs, nfeatures = data.shape  # Number of observations and features
        
        # Compute similarities between data points (negative of Euclidean distance)
        S = np.zeros((nobs,nobs))
        inds = np.arange(nobs)
        for p in range(nobs):
            S[p,:] =  -((data[p,:] - data[:,:])**2).sum(1)  # Negative squared Euclidean distance between pt p and all other points

        if self.preference:   # Preference is specified
            S[inds,inds] = self.preference
        else: 
            pref = np.median(S)
            S[inds,inds] = pref
        
        np.random.seed(self.random_seed)  
        S += 1.e-14*S*(np.random.random_sample((nobs, nobs))-0.5)

        # Initialize availability and responsibility matricies
        A = np.zeros((nobs, nobs))
        R = np.zeros((nobs,nobs))
        posR = np.zeros((nobs,nobs))
        M = np.zeros((nobs,nobs))
        exemplars = np.zeros((nobs),bool)
        exemplars_prev = np.zeros((nobs), bool)

        q = 0
        count = 0
        while (q<self.max_iter and count < self.convergence_iter):
            exemplars_prev = exemplars
            update = np.zeros((nobs,nobs))
            
            # Update responsibilty
            M = A+S
            k = M.argmax(axis = 1)        # Location of maximum value in each row of M
            maxval = M[inds,k]            # Maximum values in each row of M
            update = S - np.reshape(maxval,(nobs,1))  # S - max value in each row
            M[inds,k] = -np.inf         
            k2 = M.argmax(axis = 1)     # Location of second highest value in each row of M
            maxval = M[inds,k2]         # Second highest value in each row of M
            update[inds,k] = S[inds,k] - maxval
            R = self.damping * R + (1.-self.damping)*update

            # Update availability
            posR = R.copy()
            posR[posR<0] = 0.0      # Only positive values of R matrix
            sumR = posR.sum(0)
            vals = sumR - np.diag(posR) 
            update[:] = vals        # Sum positive values of R over all rows (i)
            update -= posR
            update[:,inds] += np.diag(R)
            update[update>0] = 0.0
            update[inds,inds] = vals
            A = self.damping*A + (1.-self.damping)*update
            

            # Identify exemplars
            exemplars = (A+R).argmax(1) == inds     # Exemplar for point i is value of k that maximizes A[i,k]+R[i,k] 
            found_exemplars = sum(exemplars)
            diff = (exemplars != exemplars_prev).sum()
            if diff == 0:
                count +=1
            else:
                count = 0
            q+=1
        
        if count < self.convergence_iter and q>= self.max_iter:
            converged = False
        else:
            converged = True

        exemplars = np.where(exemplars == True)[0]
        found_exemplars = exemplars.shape[0]

        # Modify final set of clusters to ensure that the chosen exemplars minimize wcss
        S[inds,inds] = 0.0                    # Replace diagonal entries in S with 0
        S[:,:] = -S[:,:]                      # Revert back to actual distance
        clusters = S[:,exemplars].argmin(1)   # Assign points to clusters based on distance to the possible exemplars
        for k in range(found_exemplars):      # Loop over clusters
            pts = np.where(clusters == k)[0]  # All points in cluster k
            npts = len(pts)
            distsum = np.zeros((npts))
            for p in range(npts):  
                distsum[p] += (S[pts[p],pts]).sum()   # Calculate total distance between point p and all other points in cluster k
            i = distsum.argmin()   
            exemplars[k] = pts[i]       # Replace exemplar k with point that minimizes wcss   
            
        cluster_index = S[:,exemplars].argmin(1)   # Assign points to clusters based on distance to the possible exemplars
        cluster_means = data[exemplars,:] 
        cluster_index = clusters
        wcss = self.compute_wcss(data, cluster_index, cluster_means)


        self.n_clusters = found_exemplars
        self.cluster_means = cluster_means
        self.cluster_index = cluster_index
        self.wcss = wcss     
        self.exemplars = exemplars
        self.converged = converged
                
        return self   
             


#============================================================================
#============================================================================
def read_weather(weatherfile, calc_clearsky = True):
    
    weather = {'year':[], 'month':[], 'day':[], 'hour':[], 'ghi':[], 'dhi':[], 'dni':[], 'tdry':[], 'wspd':[]}    

    zones = {-7:'MST',-8:'US/Pacific',0:'UTC'}
    
    #Get header info
    header = np.genfromtxt(weatherfile, dtype = str, delimiter = ',', max_rows = 1, skip_header = 1)
    lat = float(header[5])
    lon = float(header[6])
    z = int(header[7])
    alt = float(header[8])
    
    # Read in weather data
    labels = {'year':['Year'], 'month':['Month'], 'day':['Day'], 'hour':['Hour'], 'ghi':['GHI'], 'dhi':['DHI'], 'dni':['DNI'], 'tdry':['Tdry', 'Temperature'], 'wspd':['Wspd', 'Wind Speed']}    
    header = np.genfromtxt(weatherfile, dtype = str, delimiter = ',', max_rows = 1, skip_header = 2)
    data = np.genfromtxt(weatherfile, dtype = float, delimiter =',', skip_header=3)
    for k in labels.keys():
        found = False
        for j in labels[k]:
            if j in header:
                found = True
                c = header.tolist().index(j)
                weather[k] = data[:,c]
        if not found:
           print('Failed to find data for ' + k + ' in weather file') 
    
    if calc_clearsky:
        Npts = len(weather['dni'])     
        steps_per_hour = int(Npts/8760)
        freq = 60/steps_per_hour
        
        # make a location
        loc = Location(lat, lon, altitude=alt, tz=zones[z])
        # create a range of times for the local time zone (accounts for DST)
        dtstart = datetime.datetime(int(weather['year'][0]),int(weather['month'][0]),int(weather['day'][0]))
        times = pd.date_range(
                              start=datetime.datetime(int(weather['year'][0]),int(weather['month'][0]),int(weather['day'][0])), 
                              end=dtstart+datetime.timedelta(minutes=8760*60-freq), 
                              freq='%dMin'%freq, 
                              tz=zones[z]
                             ).shift(freq/2,'Min')         
        
        solpos =  solarposition.get_solarposition(times, lat, lon, altitude = alt)  # Calculate solar position throughout the year
        csky = loc.get_clearsky(times,model = 'ineichen', solar_position = solpos)  # Calculate clear sky radiation 
        
        weather['clearsky'] = []
        for c in csky['dni']:
            weather['clearsky'].append(c)
        weather['clearsky'] = np.array(weather['clearsky'])
    
    return weather

#============================================================================
#============================================================================  
def calc_metrics(weatherfile, Ndays = 2, ppa = None, sfavail = None, user_weights = None, user_divisions = None, normalize = True, stowlimit = None):

    '''
    weatherfile = file name containing weather data
    Ndays = number of simulation days in each group
    ppa = ppa multiplier array with same time step as weather file
    sfavail = solar field availability with same time step as weather file
    user_weights = user-selected metric weights 
    user_divisions = user-specified # of daily time-domain divisions per metric
    normalize = calculate metrics after normalization to the maximum value?
    stowlimit = wind velocity (m/s) for heliostat slow limit.  If specified, DNI in hours with velocity > stow limit will be set to zero before calculating clustering metrics
    '''
    weights = {'avgghi':0.,
                'avgghi_prev':0., 
                'avgghi_next':0., 
                #'clearsky': 0.,
                'avgt':0., 
                #'avg_sfavail': 0.,
                'avgwspd':0.,
                'avgwspd_prev': 0.,
                'avgwspd_next': 0.,
                'avgppa':0.,
                'avgppa_prev':0, 
                'avgppa_next':0}        # Weighting factors
    divisions = {'avgghi': 1,
                'avgghi_prev': 1, 
                'avgghi_next': 1, 
                'avgt': 1, 
                'avgwspd': 1,
                'avgwspd_prev': 1,
                'avgwspd_next': 1,
                'avgppa': 1,
                'avgppa_prev': 1, 
                'avgppa_next': 1}      # Integer # of divisions per day
    
    # Hourly calculation boundaries for classification metrics: 'summer_daylight' = daylight hours at summer solstice
    bounds = {'avgghi': 'summer_daylight',
                'avgghi_prev': 'summer_daylight', 
                'avgghi_next': 'summer_daylight', 
                'avgt': 'fullday', 
                'avgwspd': 'fullday',
                'avgwspd_prev': 'fullday',
                'avgwspd_next': 'fullday',
                'avgppa': 'fullday',
                'avgppa_prev': 'fullday', 
                'avgppa_next': 'fullday'} 

    if user_weights is not None and user_divisions is not None:  # User-specified clustering inputs
        for key in weights.keys():
            weights[key] = user_weights[key]
        for key in divisions.keys():    
            divisions[key] = int(user_divisions[key]) 
            
    else:  # Default case
        ## Weighting factors
        weights = {'avgghi': 1.,
                'avgghi_prev': 1., 
                'avgghi_next': 0., 
                'avgt': 0., 
                'avgwspd': 1.,
                'avgwspd_prev': 1.,
                'avgwspd_next': 0.,
                'avgppa': 1.,
                'avgppa_prev': 1., 
                'avgppa_next': 0.}  
        
        # Integer # of divisions per day       
        divisions = {'avgghi': 4,
                'avgghi_prev': 1, 
                'avgghi_next': 1, 
                'avgt': 1, 
                'avgwspd': 4,
                'avgwspd_prev': 1,
                'avgwspd_next': 1,
                'avgppa': 4,
                'avgppa_prev': 1, 
                'avgppa_next': 1}    
    
    # Read in weather data, ppa multipliers and solar field availability
    hourlydata = read_weather(weatherfile, calc_clearsky = False)#True  
    npts = len(hourlydata['ghi'])     
    nptsday = int(npts/365)
    # This could be used for both PV 1- or 2-axis tracking system and wind turbines (however, will need seperate inputs)           
    if stowlimit: 
        hourlydata['ghi'][hourlydata['wspd']>stowlimit] = 0.0    # Replace ghi at all points with windspeed > stow limit
    
    ## read in PPA price data
    hourlydata['ppa'] = np.ones((npts))
    if ppa is None:
        if weights['avgppa']> 0 or weights['avgppa_prev']> 0 or weights['avgppa_next']> 0:
            print('Warning: PPA price multipliers were not provided. Classification metrics will be calculated with a uniform multiplier')
    else:
        if len(ppa) == npts:
            hourlydata['ppa'] = np.array(ppa)
        else:
            print('Warning: Specified ppa multiplier array and data in weather file have different lengths.  Classification metrics will be calculated with a uniform multiplier')
        
    ## REMOVED FOR NOW
    ## read in solar field availability data (could be adapted for wind farm or pv field availability)
    # hourlydata['sfavail'] = np.ones((npts))   
    # if sfavail is None:
    #     if weights['avg_sfavail']>0:
    #         print('Warning: solar field availability was not provided.  Weighting factor for average solar field availability will be reset to zero')
    #         weights['avg_sfavail'] = 0.0
    # else:
    #     if len(sfavail) == npts:
    #         hourlydata['sfavail'] = np.array(sfavail)
    #     else:
    #         print('Warning: Specified solar field availability array and data in weather file have different lengths.  Weighting factor for average solar field availability will be reset to zero')
    #         weights['avg_sfavail'] = 0.0
                
    # Identify "daylight" hours 
    daylight_pts = np.zeros((365,2),int)   
    for d in range(365):
        nonzero = np.nonzero(hourlydata['ghi'][d*nptsday:(d+1)*nptsday])[0]   # Points in day d with nonzero clear-sky ghi
        daylight_pts[d,0] = nonzero[0]        # First morning point with measureable sunlight
        daylight_pts[d,1] = nonzero[-1] + 1   # First evening point without measurable sunlight

    # Calculate daily values for selected classification metrics
    daily_metrics = {'avgghi': [],
                'avgghi_prev': [], 
                'avgghi_next': [], 
                'avgt': [], 
                'avgwspd': [],
                'avgwspd_prev': [],
                'avgwspd_next': [],
                'avgppa': [],
                'avgppa_prev': [], 
                'avgppa_next': []}  
    
    datakeys = {'avgghi': 'ghi',
                'avgghi_prev': 'ghi', 
                'avgghi_next': 'ghi', 
                'avgt': 'tdry', 
                'avgwspd': 'wspd',
                'avgwspd_prev': 'wspd',
                'avgwspd_next': 'wspd',
                'avgppa': 'ppa',
                'avgppa_prev': 'ppa', 
                'avgppa_next': 'ppa'}  

    Nmetrics = 0
    for key in weights.keys():
        if weights[key] > 0.0:    # Metric weighting factor is non-zero
            Ndiv = divisions[key]  # Number of divisions per day
            daily_metrics[key] = np.zeros((365,Ndiv))
            if '_prev' in key or '_next' in key:
                Nmetrics += Ndiv    # TODO: should this *Nnext or *Nprev depending?? (This assumes 1 day for each)
            else:
                Nmetrics += Ndiv * Ndays
                
            # Determine total number of hours considered in metric calculations
            if bounds[key] == 'fullday':
                Npts = nptsday
                p1 = 0
            elif bounds[key] == 'summer_daylight':
                Npts = daylight_pts[172,1]-daylight_pts[172,0]
                p1 = daylight_pts[172,0]  
            
            # Calculate average value in each division (Averages with non-integer number of time points in a division are computed from weighted averages)    
            n = float(Npts)/Ndiv       # Number of time points per division
            pts = []
            wts = []
            for i in range(Ndiv):
                pstart = i*n            # Start time pt
                pend = (i+1)*n          # End time pt    
                npt = int(pend) - int(pstart) + 1   # Number of discrete hourly points included in the time period average
                pts.append( np.linspace(int(pstart), int(pend), npt, dtype = int) )  # Discrete points which are at least partially included in the time period average
                wts.append( 1./n * np.ones((npt)) )
                wts[i][0] = float(1.0 - (pstart - int(pstart)) ) / n      # Weighting factor for first point
                wts[i][npt-1] = float(pend - int(pend)) / n               # Weighting factor for last point

            # Calculate metrics for each day and each division
            for d in range(365):
                for i in range(Ndiv):
                    for h in range(len(pts[i])):    # Loop over hours which are at least partially contained in division i
                        if pts[i][h] == Npts:       # Hour falls outside of allowed number of hours in the day (allowed as long as weighting factor is 0)
                            if  wts[i][h] > 0.0:
                                print('Error calculating weighted average for key ' + key + ' and division ' + str(i))
                        else:
                            p = d*nptsday + p1 + pts[i][h]  # Point in yearly array
                            daily_metrics[key][d,i] += (hourlydata[datakeys[key]][p] * wts[i][h])
                    
            # Normalize daily metrics
            if normalize:
                maxmetric = daily_metrics[key].max()
                minmetric = daily_metrics[key].min()
                daily_metrics[key] = (daily_metrics[key] - minmetric) / (maxmetric - minmetric)
                    
                    
    # Create arrays of classification data for simulation days
    feature_order = ['avgghi',
                        'avgghi_prev', 
                        'avgghi_next', 
                        'avgt', 
                        'avgwspd',
                        'avgwspd_prev',
                        'avgwspd_next',
                        'avgppa',
                        'avgppa_prev', 
                        'avgppa_next']    # Order in which data features will be created
    Ngroup = int((363./Ndays))    # Number of daily ghi groupings (first and last days of the year are excluded)
    data = np.zeros((Ngroup,int(Nmetrics)))      # Classification data for clustering level j
    for g in range(Ngroup):
        f = 0
        for key in feature_order:
             if weights[key] > 0.0:   # Weighting factor > 0
                 Ndiv = divisions[key]
                 if '_prev' in key:
                     days = [g*Ndays]
                 elif '_next' in key:
                     days = [(g+1)*Ndays+1]
                 else:
                     days = np.arange(g*Ndays+1, (g+1)*Ndays+1)
                     
                 for d in days:
                    data[g,f:f+Ndiv] = daily_metrics[key][d,:] * weights[key]
                    f += Ndiv
                   

    # Evaluate subset of classification metrics for days at beginning and end of the year (not included as "simulated" days)
    data_first = None
    data_last = None
    if Ndays != 2:
        print('Extra classification metrics for first/last days are currently only defined for Ndays = 2')
    else:
        data_firstlast = np.zeros((2,Nmetrics))
        for p in range(2):      # Metrics for first and last days
            d1 = 0
            if p == 1:
                d1 = 363
            f = 0
            for key in feature_order:
                if weights[key] > 0.0:   # Weighting factor > 0
                    Ndiv = divisions[key]
                    days = [d1,d1+1]
                    if key == 'avgghi_prev' or key == 'avgppa_prev':
                        days = [d1-1]
                    elif key == 'avgghi_next' or key == 'avgppa_next':
                        days = [d1+2]
                     
                    for d in days:
                        if d>=0 and d<365:
                            data_firstlast[p,f:f+Ndiv] = daily_metrics[key][d,:] * weights[key]
                        else:
                            data_firstlast[p,f:f+Ndiv] = -1.e8
                        f += Ndiv
        data_first = data_firstlast[0,:]
        data_last = data_firstlast[1,:]
           
    classification_data = {'data':data, 'Nmetrics':Nmetrics, 'firstday':data_first, 'lastday':data_last}
    
    return classification_data
 

#============================================================================
#============================================================================  
# Create clusters from classification data.  Includes iterations of affinity propagation algorithm to create desired number of clusters if specified.
def create_clusters(data, cluster_inputs, verbose = False):        

    if cluster_inputs.algorithm == 'affinity-propagation' and cluster_inputs.afp_enforce_Ncluster:
        maxiter = cluster_inputs.afp_enforce_Ncluster_maxiter
        Ntarget = cluster_inputs.Ncluster
        tol = cluster_inputs.afp_enforce_Ncluster_tol
        urf = 1.0       

        mult = 1.0
        mult_prev = 1.0
        Nc_prev = 0
        i = 0
        finished = False

        while i<maxiter and not finished:
            cluster_inputs.afp_preference_mult = mult
            clusters = cluster_inputs.form_clusters(data)
            converged = True
            if 'converged' in clusters.keys():
                converged = clusters['converged']  # Did affinity propagataion algorithm converge? 

            if verbose:  
                print ('    Formed %d clusters with preference multiplier %f' %(clusters['Ncluster'], mult))
                if not converged: 
                    print('        Affinity propagation algorithm did not converge within the maximum allowable iterations')
                    
            if not converged:  # Don't use this solution to create next guess for preference multiplier -> increase maximum iterations and damping and try again
                cluster_inputs.afp_damping += 0.05
                cluster_inputs.afp_damping = min(cluster_inputs.afp_damping, 0.95)
                if verbose:  
                    print ('        Damping factor increased to %f' % (cluster_inputs.afp_damping))
  
            else:   # Algorithm converged -> use this solution and revert back to original damping and maximum number of iterations for next guess
                Nc = clusters['Ncluster']
                if abs(clusters['Ncluster'] - Ntarget)<=tol:
                    finished = True
                else:
                    
                    if Nc_prev == 0 or Nc == Nc_prev: # First successful iteration, or no change in clusters with change in preference mutliplier
                        mult_new  = mult * float(clusters['Ncluster'])/Ntarget
                    else: 
                        dNcdmult = float(Nc-Nc_prev) / float(mult - mult_prev)
                        mult_new = mult - urf * float(Nc-Ntarget) / dNcdmult
                        
                    if mult_new <=0:
                        mult_new  = mult * float(clusters['Ncluster'])/Ntarget
                    
                    mult_prev = mult
                    Nc_prev = Nc
                    mult = mult_new

            i+=1    
        
        if not finished:
            print ('Maximum number of iterations reached without finding %d clusters. The current number of clusters is %d' %(Ntarget, clusters['Ncluster']))
    
    else:
        clusters = cluster_inputs.form_clusters(data)

    if verbose:
        print ('    Created %d clusters' % (clusters['Ncluster']))

    # Sort clusters in order of lowest to highest exemplar points
    ngroup = data.shape[0]          # Number of data points 
    ncluster = clusters['Ncluster']  
    inds = clusters['exemplars'].argsort()
    clusters_sorted = {}
    for key in clusters.keys():
        if key in ['Ncluster','wcss']:
            clusters_sorted[key] = clusters[key]
        else:
            clusters_sorted[key] = np.empty_like(clusters[key])  
    for i in range(ncluster):
        k = inds[i]
        clusters_sorted['partition_matrix'][:,i] = clusters['partition_matrix'][:,k] 
        for key in ['count', 'weights', 'exemplars']:
            clusters_sorted[key][i] = clusters[key][k]
        for key in ['means']:
            clusters_sorted[key][i,:] = clusters[key][k,:]
    for g in range(ngroup):
        k = clusters['index'][g]
        clusters_sorted['index'][g] = inds.argsort()[k]

    return clusters_sorted



#============================================================================
#============================================================================  
# Adjust cluster weighting to account for first/last days of the year (excluded from original clustering algorithm because these days cannot be used as exemplar points)
def adjust_weighting_firstlast(data, data_first, data_last, clusters, Ndays = 2):
    '''
    data = data for clustering (Npts x Nmetrics)  
    data_first = data for neglected points at the beginning of the year 
    data_last = data for neglected points at the end of the year 
    clusters = clusters formed from original data set
    Ndays = # of consecutive simulation days 
    '''

    if Ndays != 2:
        print( 'Cluster weighting factor adjustment to include first/last days is not currently defined for ' + str(Ndays) + 'consecutive simulation days.  Cluster weights will not include days excluded from original clustering algorithm')
        clusters['weights_adjusted'] = clusters['weights']
        return [clusters, -1,-1]
    else:

        ngroup, nfeatures = data.shape
        nclusters = clusters['Ncluster']
        dist_first = np.zeros((nclusters))
        dist_last = np.zeros((nclusters))
        for k in range(nclusters):
            for f in range(nfeatures):
                if data_first[f] >-1.e7:       # Data feature f is defined for first set 
                    dist_first[k] += (data_first[f] - clusters['means'][k,f])**2
                if data_last[f] >-1.e7:
                    dist_last[k] += (data_last[f] - clusters['means'][k,f])**2
                    
        kfirst = dist_first.argmin()        # Cluster which best represents first days
        klast = dist_last.argmin()          # Cluster which best represents last days

        # Recompute cluster weights
        ngroup_adj = ngroup + 1.5          # Adjusted total number of groups 
        s = clusters['partition_matrix'].sum(0)
        s[kfirst] = s[kfirst]+0.5
        s[klast] = s[klast]+1
        clusters['weights_adjusted'] = s/ngroup_adj    
        
        return [clusters, kfirst, klast]

#============================================================================
#============================================================================  
# Compute cluster-average hourly values from full-year hourly array and partition matrix 
def compute_cluster_avg_from_timeseries(hourly, partition_matrix, Ndays, Nprev = 1, Nnext = 1, adjust_wt = False, k1 = None, k2 = None):      
    '''
    hourly = full annual array of data 
    partition_matrix = partition matrix from clustering (rows = data points, columns = clusters)
    Ndays = number of simulated days (not including previous/next)
    Nprev = number of previous days that will be included in the simulation
    Nnext = number of subsequent days that will be included in the simulation
    adjust_wt = adjust calculations with first/last days allocated to a cluster
    k1 = cluster to which first day belongs 
    k2 = cluster to which last day belongs
    
    ouput = list of cluster-average hourly arrays for the (Nprev+Ndays+Nnext) days simulated within the cluster
    '''
    Ngroup,Ncluster = partition_matrix.shape   
    Ndaystot = Ndays + Nprev + Nnext           # Number of days that will be included in the simulation (including previous / next days)
    Nptshr = int(len(hourly)/8760)             

    avg = np.zeros((Ncluster, Ndaystot*24*Nptshr))
    for g in range(Ngroup):                         
        d = g*Ndays + 1                             # First day to be counted in simulation group g
        d1 = max(0, d - Nprev)                      # First day to be included in simulation group g (Nprev days before day d if possible)
        Nprev_actual = d-d1                         # Actual number of previous days that can be included
        Ndaystot_actual = Ndays + Nprev_actual + Nnext
        h = d1*24*Nptshr                            # First time point included in simulation group g
        if Nprev == Nprev_actual:
            vals = np.array(hourly[h:h+Ndaystot*24*Nptshr])       # Hourly values for only the days included in the simulation for group g
        else:                               # Number of previous days was reduced (only occurs at beginning of the year)
            Nvoid = Nprev-Nprev_actual      # Number of previous days which don't exist in the data file (only occurs when Nprev >1)
            vals = []
            for v in range(Nvoid):   # Days for which data doesn't exist
                vals = np.append(vals,hourly[0:24*Nptshr])   # Use data from first day
            vals = np.append(vals, hourly[h:h+Ndaystot_actual*24*Nptshr])   
                
        for k in range(Ncluster):      
            avg[k,:] += vals*partition_matrix[g,k]         # Sum of hourly array * partition_matrix value for cluster k over all points (g)
    
    for k in range(Ncluster): 
        avg[k,:] = avg[k,:] / partition_matrix.sum(0)[k]   # Divide by sum of partition matrix over all groups to normalize
        
        
    if adjust_wt and Ndays == 2:     # Adjust averages to include first/last days of the year
        avgnew = avg[k1, Nprev*24*Nptshr:(Nprev+1)*24*Nptshr]* partition_matrix.sum(0)[k1]      # Revert back to non-normalized values for first simulation day in which results will be counted 
        avgnew += hourly[0:24*Nptshr]                                                           # Update values to include first day
        avg[k1,Nprev*24*Nptshr:(Nprev+1)*24*Nptshr] = avgnew /(partition_matrix.sum(0)[k1]+1)   # Normalize values for first day and insert back into average array
    
        avgnew = avg[k2,0:(Ndays+Nprev)*24*Nptshr] * partition_matrix.sum(0)[k2]     # Revert back to non-normalized values for the previous day and two simulated days
        avgnew += hourly[(363-Nprev)*24*Nptshr:365*24*Nptshr]                        # Update values to include the last days of the year
        avg[k2,0:(Ndays+Nprev)*24*Nptshr] = avgnew /(partition_matrix.sum(0)[k2]+1)  # Normalize values and insert back into average array

    return avg.tolist()

#============================================================================
#============================================================================  
# Clustering inputs that have no dependence on independent variables
def setup_clusters(weatherfile, ppamult, Nclusters, Ndays = 2, Nprev = 1, Nnext = 1, user_weights = None, user_divisions = None):
    algorithm = 'affinity-propagation'
    hard_partitions = True
    afp_enforce_Ncluster = True
    
    # Calculate classification metrics
    ret =  calc_metrics(weatherfile = weatherfile, Ndays = Ndays, ppa = ppamult, stowlimit = None, user_weights = user_weights, user_divisions = user_divisions)
    data = ret['data']
    data_first = ret['firstday']
    data_last = ret['lastday']
    
    # Create clusters
    cluster_ins = cluster()
    cluster_ins.algorithm = algorithm
    cluster_ins.Ncluster = Nclusters
    cluster_ins.sim_hard_partitions = hard_partitions
    cluster_ins.afp_enforce_Ncluster = afp_enforce_Ncluster
    clusters = create_clusters(data, cluster_ins)     
    sim_start_days = (1 + clusters['exemplars']*Ndays).tolist()

    # Adjust weighting for first and last days
    ret = adjust_weighting_firstlast(data, data_first, data_last, clusters, Ndays)
    clusters = ret[0]
    firstpt_cluster = ret[1]
    lastpt_cluster = ret[2]
    
    # Calculate cluster-average PPA multipliers and solar field adjustment factors
    avg_ppamult = compute_cluster_avg_from_timeseries(ppamult, clusters['partition_matrix'], Ndays = Ndays, Nprev = Nprev, Nnext = Nnext, adjust_wt = True, k1 = firstpt_cluster, k2 = lastpt_cluster)
    
    cluster_inputs = {}
    cluster_inputs['exemplars']= clusters['exemplars']
    cluster_inputs['weights'] = clusters['weights_adjusted']
    cluster_inputs['day_start'] = sim_start_days
    cluster_inputs['partition_matrix'] = clusters['partition_matrix']
    cluster_inputs['first_pt_cluster'] = firstpt_cluster
    cluster_inputs['last_pt_cluster'] = lastpt_cluster
    cluster_inputs['avg_ppamult'] = avg_ppamult 

    return cluster_inputs

#============================================================================
#============================================================================  
# Create full year array of hourly data with sections corresponding to cluster exemplar simulations overwritten with cluster-average values
def create_annual_array_with_cluster_average_values(hourly, cluster_average, start_days, Nsim_days, Nprev = 1, Nnext = 1, overwrite_surrounding_days = False):
                                            
    '''
    hourly = full year of hourly input data
    cluster_average = groups of cluster-average input data
    start_days = list of cluster start days
    Nsim_days = list of number of days simulated within each cluster
    Nprev = number of previous days included in the simulation
    Nnext = number of subsequent days included in teh simulation
    '''
    Ng = len(start_days) 
    output = hourly
    Nptshr = int(len(hourly)/8760)       
    Nptsday = Nptshr*24
    count_days = []   
    for g in range(Ng):
        for d in range(Nsim_days[g]):
            count_days.append(start_days[g] + d)
    
    for g in range(Ng):                        # Number of simulation groupings  
        Nday = Nsim_days[g]                    # Number of days counted in simulation group g
        Nsim = Nsim_days[g] + Nprev + Nnext    # Number of simulated days in group g
        for d in range(Nsim):                  # Days included in simulation for group g
            day_of_year = (start_days[g] - Nprev) + d 
            if d >= Nprev and d<Nprev+Nday:    # Days that will be counted in results
                for h in range(Nptsday):
                    output[day_of_year*Nptsday + h] = cluster_average[g][d*Nptsday + h]
 
            else:   # Days that will not be counted in results 
                if overwrite_surrounding_days:
                    if day_of_year not in count_days and day_of_year>=0 and day_of_year<365:  
                        for h in range(Nptsday):
                            output[day_of_year*Nptsday + h] = cluster_average[g][d*Nptsday + h]
                        
    return output

#============================================================================
#============================================================================  
# Create full year hourly array from hourly array containing only data at exemplar points
def compute_annual_array_from_clusters(exemplardata, clusters, Ndays, adjust_wt = False, k1 = None, k2 = None, dtype = float):
    '''
    exemplardata = full-year hourly array with data exising only at days within exemplar groupings
    clusters = cluster information
    Ndays = number of consecutive simulation days within each group
    adjust_wt = adjust calculations with first/last days allocated to a cluster
    k1 = cluster to which first day belongs 
    k2 = cluster to which last day belongs
    '''
    npts = len(exemplardata)
    fulldata = np.zeros((npts))
    ngroup, ncluster = clusters['partition_matrix'].shape
    nptshr = int(npts / 8760)
    nptsday = nptshr * 24
    
    data = np.zeros((nptsday*Ndays, ncluster))   # Hourly data for each cluster exemplar
    for k in range(ncluster):       
        d = clusters['exemplars'][k]*Ndays+1   # Starting days for each exemplar grouping
        data[:,k] = exemplardata[d*nptsday:(d+Ndays)*nptsday]
        
    for g in range(ngroup):
        d = g*Ndays+1                     # Starting day for data group g
        avg = (clusters['partition_matrix'][g,:] * data).sum(1)      # Sum of partition matrix x exemplar data points for each hour
        fulldata[d*nptsday:(d+Ndays)*nptsday] = avg
    
    # Fill in first/last days 
    if adjust_wt and k1 >= 0 and k2 >= 0 and Ndays == 2:
        d = (clusters['exemplars'][k1])*Ndays+1         # Starting day for group to which day 0 is assigned
        fulldata[0:nptsday] = fulldata[d*nptsday:(d+1)*nptsday]
        d = (clusters['exemplars'][k2])*Ndays+1         # Starting day for group to which days 363 and 364 are assigned
        fulldata[363*nptsday:(363+Ndays)*nptsday] = fulldata[d*nptsday:(d+Ndays)*nptsday]      
    else:
        navg = 5
        if max(fulldata[0:24]) == 0:   # No data for first day of year
            print( 'First day of the year was not assigned to a cluster and will be assigned average generation profile from the next '+ str(navg) + ' days.')
            hourly_avg = np.zeros((nptsday))
            for d in range(1,navg+1):   
                for h in range(24*nptshr):
                    hourly_avg[h] += fulldata[d*nptsday+h] / navg
            fulldata[0:nptsday] = hourly_avg
        
        nexclude = 364 - ngroup*Ndays   # Number of excluded days at the end of the year
        if nexclude>0:
            h1 = 8760*nptshr - nexclude*nptsday   # First excluded hour at the end of the year
            if max(fulldata[h1 : h1+nexclude*nptsday]) == 0:
                print( 'Last ' + str(nexclude) + ' days were not assigned to a cluster and will be assigned average generation profile from prior '+ str(navg) + ' days.')
                hourly_avg = np.zeros((nexclude*nptsday))
                d1 = 365-nexclude-navg    # First day to include in average
                for d in range(d1,d1+navg):   
                    for h in range(nptsday):
                        hourly_avg[h] += fulldata[d*nptsday+h] / navg
                fulldata[h1 : h1+nexclude*nptsday] = hourly_avg

    if dtype is bool:
        fulldata = np.array(fulldata, dtype = bool)

    return fulldata.tolist()




#============================================================================
#============================================================================  
# Combine consecutive exemplars into a single simulation
def combine_consecutive_exemplars(days, weights, avg_ppamult, avg_sfadjust, Ndays = 2, Nprev = 1, Nnext = 1):
    '''
    days = starting days for simulations (not including previous days)
    weights = cluster weights
    avg_ppamult = average hourly ppa multipliers for each cluster (note: arrays include all previous and subsequent days)
    avg_sfadjust = average hourly solar field adjustment factors for each cluster (note: arrays include all previous and subsequent days)
    Ndays = number of consecutive days for which results will be counted
    Nprev = number of previous days which are included before simulation days
    Nnext = number of subsequent days which are included after simulation days
    '''
    
    Ncombine = sum(np.diff(days)==Ndays)    # Number of simulation groupings that can be combined (starting days represent consecutive groups)
    Nsim = len(days) - Ncombine             # Number of simulation grouping after combination
    Nptshr = int(len(avg_ppamult[0]) / ((Ndays + Nprev + Nnext)*24))   # Number of points per hour in input arrays
    group_index = np.zeros((len(days)))
    start_days = np.zeros((Nsim),int)
    sim_days = np.zeros((Nsim),int)
    g = -1
    for i in range(len(days)):
        if i == 0 or days[i]-days[i-1] != Ndays:  # Day i starts new simulation grouping
            g+=1
            start_days[g] = days[i]
        sim_days[g] += Ndays
        group_index[i] = g
    
    group_weight = []
    group_avgppa = []
    group_avgsfadj = []    
    h1 = Nprev*24*Nptshr                    # First hour of "simulation" day in any cluster
    h2 = (Ndays+Nprev)*24*Nptshr            # Last hour of "simulation" days in any cluster
    hend = (Ndays+Nprev+Nnext)*24*Nptshr    # Last hour of "next" day in any cluster  
    for i in range(len(days)):
        g = group_index[i]
        if i == 0 or g != group_index[i-1]:   # Start of new group
            wt = [float(weights[i])]
            avgppa = avg_ppamult[i][0:h2]
            avgsfadj = avg_sfadjust[i][0:h2]
        else:  # Continuation of previous group
            wt.append(weights[i])
            avgppa = np.append(avgppa, avg_ppamult[i][h1:h2])
            avgsfadj = np.append(avgsfadj, avg_sfadjust[i][h1:h2])
        
        if i == len(days)-1 or g != group_index[i+1]:  # End of group
            avgppa =  np.append(avgppa, avg_ppamult[i][h2:hend])
            avgsfadj =  np.append(avgsfadj, avg_sfadjust[i][h2:hend])
            group_weight.append(wt)  
            group_avgppa.append(avgppa.tolist())
            group_avgsfadj.append(avgsfadj.tolist())
        
    combined = {}      
    combined['start_days'] = start_days.tolist()
    combined['Nsim_days'] = sim_days.tolist()
    combined['avg_ppa'] = group_avgppa
    combined['avg_sfadj'] = group_avgsfadj
    combined['weights'] = group_weight
        
    return combined



       