import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import random
from PIL import Image
import pickle as pkl
import datetime
import scipy.stats as scistat

def getInfo():
	
# getInfo does all the data gathering from the csv, puts it in a dictionary, and saves it. It keeps track of a lot of things I don't end up using.

# i want a dictionary that contains all workers. the value of the worker dictionary is another dictionary that contains the cumulative false positive
# and false negative rate for that particle

	# Read the file with all the info. Turn it into a numpy array of strings.
	filename = 'results-Actual.csv'; # whichever csv we wanna process
	key_dict = pkl.load(open('LakshmiKeydict.pkl', 'r')); # 
	with open(filename, 'r') as target:
		#strmat = np.genfromtxt(target, dtype='str', missing_values='NONE');
		datamat = pd.read_csv(filename) # make pandas dataframe
	npdatamat = np.array(datamat); # make numpy array from df

	# x-y locations of .....
	#xylocs = np.array([]);		

	# Initialize a dictionary to keep track of stats for each worker (person)
	workerdict = {};

	for workerID in datamat['MTurk Worker ID']:
		if workerID in workerdict: continue;
		workerdict[workerID] = {}

	# For datetime, conversion dictionary
	dateconv = {'Jan':1, 'Feb':2, 'Mar':3, 'Apr':4, 'May':5, 'Jun':6, 'Jul':7, 'Aug':8, 'Sep':9, 'Oct':10, 'Nov':11, 'Dec':12}

	# Populate a dictionary with the stats for each particle (good and bad both)
	particledict = {}; # i wont actually end up using this
	particlelist = []
	for i in range(10):
		# File that names all the particles in this montage set
		this_mon = 'Montages/S' + str(i) + '/S' + str(i) + 'key.txt'
		with open(this_mon, 'r') as target:
			for line in target:
				particle = line.strip('\n')
				if len(particle) == 0: continue; # skip empty lines
				if particle in ['Real particle names:','Bad particle names:']: continue 	# Remove all occurances of these false particle names			
				
				# Give this particle and entry
				# Missed is bad particle not clicked, Bcorrect is bad particle clicked, Gmissclasssfied is good particle clicked, appeared is # times seen,
				# idx is index in particlelist (for fast lookup)
				if particle in particledict: 
					continue; # skip if already initialized
				else: 
					particledict[particle] = {'Missed':0, 'BCorrect':0, 'Appeared':0, 'GMisclassified':0, 'idx':len(particlelist)}
					particlelist.append(particle)
				# Give every worker a sub-dictionary for this particle
				for ID in workerdict:
#					workerdict[ID][particle] = {'Appeared':0., 'CumAppeared':[0.], 'ErrorRate':0., 'CumErrorRate':[0.]}
					workerdict[ID][particle] = {'CumAppeared':[0.], 'CumFPRate':[0.], 'CumFNRate':[0.], 'CumErrorRate':[0.]}

	# Main data storgae dictionary. Really only the fields defined below are the ones I end up using
	for ID in workerdict:
		# cumulative # seen, cumulative error, individual montage error, time submitted montage
		# MType: 0 => 10 base bads,  1 => 30 base bads,  2 => 50 base bads
		workerdict[ID]['Montage'] = {'CumMontagesSeen':[], 'CumMontageError':[], 'Error':[], 'SubmitTime':[], 'MType':[], 'MComplexity':[]}


	# Go through the rows of npdatamat
	# Each row contains all info about a time that a worker looked at an image! 
	for rownum in np.arange(npdatamat.shape[0]): # for each row in csv
		workerID = npdatamat[rownum, 2]; # Which worker did this? 
		current_MontageError = 0.; # Error for this montage
		if len(workerdict[workerID]['Montage']['CumMontagesSeen']) == 0:
			Nmon_seen = 0.; # number of montages seen by this worker
			previous_MontageError = 0.;
		else:
			Nmon_seen = workerdict[workerID]['Montage']['CumMontagesSeen'][-1] # number of montages seen by this worker
			previous_MontageError = workerdict[workerID]['Montage']['CumMontageError'][-1]
		workerdict[workerID]['Montage']['CumMontagesSeen'].append( Nmon_seen + 1)
		Nmon_seen += 1; # number of montages seen by this worker

		# Figure out which montage we're looking at
		tmp = npdatamat[rownum, 0].split('S')[1].split('M'); #get filename					 
		path = 'Montages/S' + tmp[0] + '/' + npdatamat[rownum, 0].split('.')[0] + 'key.txt'; # path to file containing the info about montage

		# Get info only for particles in this image 
 		with open(path, 'r') as keyfile:
			key = pd.read_csv(keyfile, sep='\t'); # make pandas dataframe for the csv with info about montage

		key_sublocs   = key['Sublocation'][0:100]; # ?? Why did I do this? I think its just the ints 0:99...
		num_bads	  = key['Sublocation'][100];   # number of bad particles in this image by MY classification
		key_filenames = key['Filename'][0:100];	# list of particle filenames in this image



		# Note the appearance of each particle in appropriate dicts
		for particle in key_filenames:
			# particle info: particle was in an image
			particledict[particle]['Appeared'] += 1.; 
			# worker info: worker saw the particle
			workerdict[workerID][particle]['CumAppeared'].append(workerdict[workerID][particle]['CumAppeared'][-1] + 1.)
#			workerdict[workerID][particle]['Appeared'] += 1.


		# Track locations of bad particles		
		badlookup = {} # dictionary that gives true (is bad) or false (is good) for each image location
		badlocs = [] # image locations of bad particles; eventually we want to reduce this down to bad particles NOT ID'ed
		for i, val in enumerate(key_sublocs):
#			badlookup[val] = ('BAD' in key_filenames[i])
			badlookup[val] = not key_dict[key_filenames[i]] # use lakshmi's classifications
			if badlookup[val]: badlocs.append(val)


		# classify the montage type based on how many bad particles were in there
		# Montage Types have a base of 10, 30, or 50 bad particles. My montage generator allowed for +/- 5 bad particles.
		# Can be more or less than +/- in reality, cause lakshmi and I have different definitions of bad particles (9 out of 500)
		# 20 and 40 were picked cause they're in between 
#		num_bads_lakshmi = np.sum([1 for val in badlookup if badlookup[val]]) # count number of true's
		num_bads_lakshmi = len(badlocs)
		if num_bads_lakshmi < 20:
			workerdict[workerID]['Montage']['MType'].append(0); # type1
		elif num_bads_lakshmi < 40:
			workerdict[workerID]['Montage']['MType'].append(1); # type2
		else:
			workerdict[workerID]['Montage']['MType'].append(2); # type3


		# Go through locations clicked on (i.e. marked as bad) and organize the information about the location
		numobj = npdatamat[rownum, 4]; # number of objects identified (good or bad) by worker
		pairs = np.reshape(npdatamat[rownum, 5:numobj*2+5], (-1, 2)); #pairs of xy points (in csv they are just a 1D array, must reshape)
		picked = []; # subpos of something clicked on, to check if they clicked on a particle more than once
		for i, pair in enumerate(pairs):
			if True in pd.isnull(pair): continue # skip null entries

			its_subpos = get_subpos(pair) # convert xy positions into img sublocs (0-99)
			if its_subpos in picked: continue; # skip things we've seen before, ie they clicked on more than once

			part_file = key_filenames[its_subpos] # store the name of the particle, looked up by its img subloc
#			N_app = workerdict[workerID][part_file]['Appeared'] 
			N_app = workerdict[workerID][part_file]['CumAppeared'][-1] # number of times that this particle has been viewed by this worker

			picked.append(its_subpos); # sublocs we haven't visited before; add to list of things we've now visited. Later, becomes sublocs already processed

			# need to make sure all bad particles are in a workers picked particles
			if its_subpos in badlocs: # TRUE NEGATIVE
				badlocs.pop(badlocs.index(its_subpos)) # correctly ID'ed -> remove from badlocs
				particledict[part_file]['BCorrect'] += 1.; # note particle was correctly ID'ed

				# keep cumulative track of accuracy; running average over all images
				# cumulative track of worker accuracy for this particle
				workerdict[workerID][part_file]['CumFNRate'].append( ((N_app-1)*workerdict[workerID][part_file]['CumFNRate'][-1] + 0.) / N_app )
			else: # FALSE NEGATIVE
				particledict[part_file]['GMisclassified'] += 1.; # note that particle was misclassified

				# cumulative track of worker accuracy for this particle
				workerdict[workerID][part_file]['CumFNRate'].append( ((N_app-1)*workerdict[workerID][part_file]['CumFNRate'][-1] + 1.) / N_app )
				current_MontageError += 1; # made either a false negative of false positive. error on montage level

		# Now see how many the workers missed (bad particles they thought were good)
		for badloc in badlocs: # FALSE POSITIVE
			picked.append(badloc); # no longer picked locations in picked, but now sublocs we have processed
			x, y = get_xypos(badloc); # get x,y position from subloc
			part_file = key_filenames[badloc] # get filename of particle in subloc
			N_app = workerdict[workerID][part_file]['CumAppeared'][-1]  # number of times that this particle has been viewed by this worker

			particledict[part_file]['Missed'] += 1.; # note that particle was missed
			# cumulative track of worker accuracy for this particle
			workerdict[workerID][part_file]['CumFPRate'].append( ((N_app-1.)*(workerdict[workerID][part_file]['CumFPRate'][-1]) + 1.) / N_app )
			current_MontageError += 1; # made either a false negative or false positive. error on montage level

		# Go to every leftover sublocation not picked and mark correct (good particle not clicked, i.e. noted as good)
		for leftoverloc in key_sublocs: # TRUE POSITIVE
			if leftoverloc in picked: continue; # if we have already processed this subloc, don't doublecount
			part_file = key_filenames[leftoverloc] # get filename of particle in subloc
			N_app = workerdict[workerID][part_file]['CumAppeared'][-1] # number of times that this particle has been viewed by this worker

			# cumulative track of worker accuracy. update to reflect TRUE POSITIVE identification of particle
			workerdict[workerID][part_file]['CumFPRate'].append( ((N_app-1.)*(workerdict[workerID][part_file]['CumFPRate'][-1]) + 0.) / N_app )

		# keep track of accuracy on the montage level for plotting later
		workerdict[workerID]['Montage']['CumMontageError'].append( ((Nmon_seen-1.)*previous_MontageError + current_MontageError/100.) / Nmon_seen )
		workerdict[workerID]['Montage']['Error'].append( current_MontageError/100. ) # non-cumulative error
		dateparts = npdatamat[rownum, 3].split(' '); # splits submit time string from csv
		timeparts = dateparts[3].split(':'); # splits time string from the previous string
		workerdict[workerID]['Montage']['SubmitTime'].append( datetime.datetime(int(dateparts[5]), int(dateconv[dateparts[1]]), int(dateparts[2]),
																				int(timeparts[0]), int(timeparts[1]), int(timeparts[2])) );

	pkl.dump( workerdict, open('workerdict-actual.pkl', 'w') )
	return particlelist, particledict, workerdict


#########
# PLOTTING
#########
def make_colors(Ncolors):
	return sns.color_palette("husl",Ncolors)


def main1():
	wdict = pkl.load( open('workerdict-actual.pkl', 'r') )
	colors = make_colors(len(wdict))

	# sliding window averaging
	endM = 100; # look at first endM Montages only. Alex suggested 50
	subplots = 9; # number of subplots on figure. windowsizes will go in subplots
	windowsize_arr = np.linspace(1, endM/2-1, subplots+2) # the +2 means we will skip elements 0 and end. (a windowsize of 1 isnt averaging...)

	xarr_mean = np.zeros( endM ); # for the window-averaged mean_Error
	error_vals = {}; # temporary for storing values
	mean_error = np.zeros(endM); # mean error, no averaging
	std_plus = np.zeros(endM); # std + mean will go in here, for plotting
	std_minus = np.zeros(endM); # std - mean will go in here, for plotting
	workers_index = 0; # for looping
	workers_active = {}; # gives the ID of the workers that have done more than endM montages
	for j in range(endM):
		error_vals[j] = [];
		for i, ID in enumerate(wdict):
			if len(wdict[ID]['Montage']['CumMontagesSeen']) < 1: continue;
			if len(wdict[ID]['Montage']['CumMontagesSeen']) < endM: continue;
			if ID not in workers_active: 
				workers_active[ID] = workers_index;
				workers_index += 1;
			error_vals[j].append( wdict[ID]['Montage']['Error'][j] )

		mean_error[j] = np.mean(error_vals[j])
	Cnorm = mean_error[endM-1]

	xarr = np.zeros((subplots, len(workers_active.keys()), endM)) # for plotting
	Nwork = len(workers_active.keys()) # number of workers that has done at least endM montages
	print(Nwork)

	# time between montages
	times = {};
	times['tot'] = [];
	for i, ID in enumerate(wdict):
		times[ID] = [];
		sorted_t = sorted(wdict[ID]['Montage']['SubmitTime'])
#		print(sorted(zip(wdict[ID]['Montage']['SubmitTime'], wdict[ID]['Montage']['Error'])))
		wdict[ID]['Montage']['Error'] = [y for x, y in sorted(zip(wdict[ID]['Montage']['SubmitTime'], wdict[ID]['Montage']['Error']))]
		wdict[ID]['Montage']['MType'] = [y for x, y in sorted(zip(wdict[ID]['Montage']['SubmitTime'], wdict[ID]['Montage']['MType']))]		
		for j in range(len(sorted_t)-1):
			dt = (sorted_t[j+1] - sorted_t[j]).total_seconds()
			times[ID].append(dt);
			times['tot'].append(dt);
		plt.plot(range(1, len(times[ID])+1), times[ID], color=colors[i], alpha=0.7, lw=0.5)
	plt.title('time to complete per worker')
	plt.xlabel('Montage')
	plt.ylabel('time to complete, s')
	print('mean time to copmlete ', np.mean(times['tot']))
	print('median time to complete ', np.median(times['tot']))
	plt.show()
	
	#worker accuracy sliding window
	for i, ID in enumerate(wdict):
		if len(wdict[ID]['Montage']['CumMontagesSeen']) < endM: continue;
		for nsub in range(1, subplots+1):
			windowsize = int(windowsize_arr[nsub]); # round windowsize to int, set to variable for convenience
			plt.subplot(3, 3, nsub) 
			# want to use the prev values, use cumulative mean until windowsize
			xarr[nsub-1,workers_active[ID]][0:windowsize] = get_runningAvg(wdict[ID]['Montage']['Error'][0:windowsize])

			for xpos in range(windowsize, endM):
				# after you have reached element 'windowsize', use previous elements
				xarr[nsub-1,workers_active[ID], xpos] = np.mean(wdict[ID]['Montage']['Error'][xpos-windowsize:xpos])
			plt.plot(range(endM), xarr[nsub-1,workers_active[ID]]/Cnorm, color=colors[i], alpha=0.7, lw=0.5) # plot the window-averaged error for that worker
	for nsub in range(1, subplots+1):
		windowsize = int(windowsize_arr[nsub]); # round windowsize to int, set to variable for convenience
		for xpos in range(0, windowsize): # don't have a cumulative mean for mean_error, so before windowsize, cumulative...
			xarr_mean[xpos] = np.mean(mean_error[0:xpos]);
		for xpos in range(windowsize, endM): # use windowsize for averaging
			xarr_mean[xpos] = np.mean(mean_error[xpos-windowsize:xpos]) # after you have reached element 'windowsize', use previous elements...
		plt.subplot(3, 3, nsub)
		plt.text(25, 0, 'window=' + str(windowsize)) # write the window size 
		Cnorm = xarr_mean[endM-1]
		plt.plot(range(endM), xarr_mean/Cnorm, color='k')	# plot the mean error
#		plt.plot(range(endM), std_plus/Cnorm, linestyle='--', color='b', lw=0.5) # plot the stddevs
#		plt.plot(range(endM), std_minus/Cnorm, linestyle='--', color='b', lw=0.5)
		for j in range(endM):
			std_plus[j] = xarr_mean[j] + np.std(xarr[nsub-1, :, j])
			std_minus[j] = xarr_mean[j] - np.std(xarr[nsub-1, :, j])

		plt.fill_between(range(endM), std_plus/Cnorm, std_minus/Cnorm, where=std_minus/Cnorm>0, color='blue', alpha=0.1) # fill between the stddevs
	plt.subplot(3,3,2)
	plt.title('Mean Error Rate, Actual Worker Error Rate vs Montages Seen with Various Averaging Windows, N=' + str(Nwork))
	plt.show()
	
	# correlation between errors
	for nsub in range(1, subplots+1):
		plt.subplot(3, 3, nsub)
		for j in range(endM):
			std_plus[j] = np.std(xarr[nsub-1, :, j])
			Cnorm = std_plus[endM-1];
		plt.plot(std_plus/Cnorm)
	plt.subplot(3,3,2)
	plt.title('Mean Std dev Seen with Various Averaging Windows')
	plt.show()
	corrmat = np.zeros((Nwork, Nwork))
	corrmat_sig = np.zeros((Nwork, Nwork))
	for i, ID in enumerate(workers_active):
		for j, ID2 in enumerate(workers_active):
			#finda  better correlation funciton
			corrmat[i, j], corrmat_sig[i,j] = scistat.stats.pearsonr(wdict[ID]['Montage']['Error'][0:endM], wdict[ID2]['Montage']['Error'][0:endM])
	print( 'Correlation overall', (np.sum(corrmat)-Nwork)/(Nwork**2 - Nwork), 'significance', (np.sum(corrmat_sig)-Nwork)/(Nwork**2 - Nwork) )
	sns.heatmap(corrmat)
#	plt.colorbar()
	plt.title('Correlation between errors')
	plt.show()

	#mean, median time ot complete

	#error rate vs # completed
	thresh_N = 10;
	xseen = [];
	yerror = [];
	for i, ID in enumerate(wdict):
		if wdict[ID]['Montage']['CumMontagesSeen'] < thresh_N: continue;
		xseen.append(wdict[ID]['Montage']['CumMontagesSeen'][-1])
		yerror.append(np.mean(wdict[ID]['Montage']['Error']))
		sns.plt.scatter(xseen, yerror)
	plt.title('Error Rate vs Completion')
	plt.ylabel('Total Error Rate')
	plt.xlabel('Montages Completed')
	plt.show()
	
	# error by montage type
	errorsT0 = [];
	errorsT1 = [];
	errorsT2 = [];
	width = 0.35;
	for i, ID in enumerate(wdict):
		for j, val in enumerate(wdict[ID]['Montage']['Error']):
			if wdict[ID]['Montage']['MType'][j] == 0:
				errorsT0.append(val)
			elif wdict[ID]['Montage']['MType'][j] == 1:
				errorsT1.append(val)
			else:
				errorsT2.append(val)
	plt.bar(np.arange(3), [np.mean(errorsT0), np.mean(errorsT1), np.mean(errorsT2)], width, ecolor='k', yerr=[np.std(errorsT0), np.std(errorsT1), np.std(errorsT2)])
	print('means t0, t1, t2,', [np.mean(errorsT0), np.mean(errorsT1), np.mean(errorsT2)])
	print('stds t0 t1 t2', [np.std(errorsT0), np.std(errorsT1), np.std(errorsT2)])
	plt.title('Mean Error and Stddev by Type of Montage')
	plt.ylabel('Mean Error')
	plt.xlabel('Type of Montage: 10, 30, 50')
	plt.xticks(np.arange(3) + width/2., ('Type1', 'Type2', 'Type3'))
	plt.show()


def get_subpos(xy):
	#10x10, so each subimg is 50x50
	#subpos ordering is column major
#	row = xy[0] // 50.
#	col = xy[1] // 50.
	if xy[0] > 499: xy[0] = 499
	if xy[1] > 499: xy[1] = 499
	return (xy[0] // 50.)*10 + (xy[1] // 50.)

def get_xypos(subloc):
	#go from sublocation to xy pos
	return (subloc // 10) * 50, (9 - (subloc % 10))*50.

def get_rand50():
	#add a random number
	return random.randint(1, 49);

def get_runningAvg(arr):
	#return a running average of given array
	return [np.mean(arr[0:i]) for i in range(1, len(arr)+1)]

if __name__ == "__main__":
#	getInfo();
	main1();

