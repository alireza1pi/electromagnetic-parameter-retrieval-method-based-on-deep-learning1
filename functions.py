import pandas as pd
import numpy as np
import math
import statistics
from statistics import *
from sklearn import metrics
#from pandas import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import warnings


from numpy.random import uniform, randn, random, seed
from filterpy.monte_carlo import multinomial_resample
import scipy.stats
seed(7)
 

 
def accuracy(predictions, labels):
    #az tarighe formul neveshte shode ast:
    a=np.mean(np.sqrt(np.sum((predictions - labels)**2, 1)))    
    return a



def create_particles(x_range, y_range, v_mean, v_std, N):
    #N is number of particles
        #particle state be sourate (x pos,y pos, direction of motion, speed of motion) det shode ast
            #direction of motion, speed of motion baraye predict kardan karbord darnd
    particles = np.empty((N, 4))
    particles[:, 0] = uniform(x_range[0], x_range[1], size=N)
    particles[:, 1] = uniform(y_range[0], y_range[1], size=N)
    particles[:, 2] = uniform(0, 2 * np.pi, size=N)
    particles[:, 3] = v_mean + (randn(N) * v_std)
    return particles
 
def predict_particles(particles, std_heading, std_v, x_range, y_range):
        # "" "function predict dar in ja avarde shode ast: jahate particleha bar asase speed v jahat khahad bood, ke be soorat random change khahad shod v dobare tekrar...," ""
    idx = np.array([True] * len(particles))
    particles_last = np.copy(particles)
    for i in range (100): # Try at most 100 times
        if i == 0:
            particles[idx, 2] = particles_last[idx, 2] + (randn(np.sum(idx)) * std_heading)
        else:
            particles [idx, 2] = uniform (0, 2 *np.pi, size = np.sum (idx)) # randomly change direction
        particles[idx, 3] = particles_last[idx, 3] + (randn(np.sum(idx)) * std_v)
        particles[idx, 0] = particles_last[idx, 0] + np.cos(particles[idx, 2] ) * particles[idx, 3]
        particles[idx, 1] = particles_last[idx, 1] + np.sin(particles[idx, 2] ) * particles[idx, 3]
                 # mohasebeye particleha ba tavajoh be boundary
        idx = ((particles[:, 0] < x_range[0])
                | (particles[:, 0] > x_range[1])
                | (particles[:, 1] < y_range[0]) 
                | (particles[:, 1] > y_range[1]))
        if np.sum(idx) == 0:
            break


#update particleha,update vazn ba tavajoh be mogeeyat pdf v ettelaate be dast amade az moshahedeye natayej          
def update_particles(particles, weights, z, d_std):
        #fasele mogeeyate moshahede shode v real ra toziea gussian dar nazar migirim.
    distances = np.linalg.norm(particles[:, 0:2] - z, axis=1)
    weights *= scipy.stats.norm(0, d_std).pdf(distances)
    weights += 1.e-300
    weights /= sum(weights)
    
    # "" "function baraye Estimate Location" ""
def estimate(particles, weights):
    a=np.average(particles, weights=weights, axis=0)
    return a

#function dar nemune bardari pey dar pey estefade mishavad
def neff(weights):
    return 1. / np.sum(np.square(weights))

#in function baraye nemune bardari mojadad motabegh nemune moshakhas shode
def resample_from_index(particles, weights, indexes):
         
    particles[:] = particles[indexes]
    weights[:] = weights[indexes]
    weights /= np.sum(weights)

from numpy.random import random

#########################################################################################



def residual_resample(weights):
    """ Performs the residual resampling algorithm used by particle filters.
    Based on observation that we don't need to use random numbers to select
    most of the weights. Take int(N*w^i) samples of each particle i, and then
    resample any remaining using a standard resampling algorithm [1]
    Parameters
    ----------
    weights : list-like of float
        list of weights as floats
    Returns
    -------
    indexes : ndarray of ints
        array of indexes into the weights defining the resample. i.e. the
        index of the zeroth resample is indexes[0], etc.
    References
    ----------
    .. [1] J. S. Liu and R. Chen. Sequential Monte Carlo methods for dynamic
       systems. Journal of the American Statistical Association,
       93(443):1032–1044, 1998.
    """

    N = len(weights)
    indexes = np.zeros(N, 'i')

    # take int(N*w) copies of each weight, which ensures particles with the
    # same weight are drawn uniformly
    num_copies = (np.floor(N*np.asarray(weights))).astype(int)
    k = 0
    for i in range(N):
        for _ in range(num_copies[i]): # make n copies
            indexes[k] = i
            k += 1

    # use multinormal resample on the residual to fill up the rest. This
    # maximizes the variance of the samples
    residual = weights - num_copies     # get fractional part
    residual /= sum(residual)           # normalize
    cumulative_sum = np.cumsum(residual)
    cumulative_sum[-1] = 1. # avoid round-off errors: ensures sum is exactly one
    indexes[k:N] = np.searchsorted(cumulative_sum, random(N-k))

    return indexes



def stratified_resample(weights):
    """ Performs the stratified resampling algorithm used by particle filters.
    This algorithms aims to make selections relatively uniformly across the
    particles. It divides the cumulative sum of the weights into N equal
    divisions, and then selects one particle randomly from each division. This
    guarantees that each sample is between 0 and 2/N apart.
    Parameters
    ----------
    weights : list-like of float
        list of weights as floats
    Returns
    -------
    indexes : ndarray of ints
        array of indexes into the weights defining the resample. i.e. the
        index of the zeroth resample is indexes[0], etc.
    """

    N = len(weights)
    # make N subdivisions, and chose a random position within each one
    positions = (random(N) + range(N)) / N

    indexes = np.zeros(N, 'i')
    cumulative_sum = np.cumsum(weights)
    i, j = 0, 0
    while i < N:
        if positions[i] < cumulative_sum[j]:
            indexes[i] = j
            i += 1
        else:
            j += 1
    return indexes


def systematic_resample(weights):
    """ Performs the systemic resampling algorithm used by particle filters.
    This algorithm separates the sample space into N divisions. A single random
    offset is used to to choose where to sample from for all divisions. This
    guarantees that every sample is exactly 1/N apart.
    Parameters
    ----------
    weights : list-like of float
        list of weights as floats
    Returns
    -------
    indexes : ndarray of ints
        array of indexes into the weights defining the resample. i.e. the
        index of the zeroth resample is indexes[0], etc.
    """
    N = len(weights)

    # make N subdivisions, and choose positions with a consistent random offset
    positions = (random() + np.arange(N)) / N

    indexes = np.zeros(N, 'i')
    cumulative_sum = np.cumsum(weights)
    i, j = 0, 0
    while i < N:
        if positions[i] < cumulative_sum[j]:
            indexes[i] = j
            i += 1
        else:
            j += 1
    return indexes


def multinomial_resample(weights):
    """ This is the naive form of roulette sampling where we compute the
    cumulative sum of the weights and then use binary search to select the
    resampled point based on a uniformly distributed random number. Run time
    is O(n log n). You do not want to use this algorithm in practice; for some
    reason it is popular in blogs and online courses so I included it for
    reference.
   Parameters
   ----------
    weights : list-like of float
        list of weights as floats
    Returns
    -------
    indexes : ndarray of ints
        array of indexes into the weights defining the resample. i.e. the
        index of the zeroth resample is indexes[0], etc.
    """
    cumulative_sum = np.cumsum(weights)
    cumulative_sum[-1] = 1.  # avoid round-off errors: ensures sum is exactly one
    return np.searchsorted(cumulative_sum, random(len(weights)))




############################################################################################



#in function filter particle ha ra yek bar anjam midahad v natije ra barmigardanad.
def run_pf(particles, weights, z, x_range, y_range):
   
    x_range, y_range = [5, 21], [-12,4]
    predict_particles (particles, 0.6, 0.01, x_range, y_range) # 1. Predict
    update_particles (particles, weights, z, 4) # 2. Update
    if neff (weights) <len (particles) / 2: # 3. Resampling
        indexes = multinomial_resample(weights)
        resample_from_index(particles, weights, indexes)
    
    return estimate (particles, weights) # 4. State estimation
