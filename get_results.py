import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import random
from PIL import Image
import pickle as pkl

def getInfo():
#	x, y = get_xypos(0)
#	print(x, y)
#	plt.scatter(x, y)  
#	x, y = get_xypos(9)
#	print(x, y)		   
#	plt.scatter(x, y)  
#	x, y = get_xypos(90)
#	print(x, y)
#	plt.scatter(x, y)   
#
#	x, y = get_xypos(71)
#	print(x, y)		   
#	plt.scatter(x, y)  	
#	plt.show()
#	return;


	# Read the file with all the info. Turn it into a numpy array of strings.
	filename = 'results-Actual.csv';
	with open(filename, 'r') as target:
		#strmat = np.genfromtxt(target, dtype='str', missing_values='NONE');
		datamat = pd.read_csv(filename)
	npdatamat = np.array(datamat);

	# x-y locations of .....
	#xylocs = np.array([]);		

	# Initialize a dictionary to keep track of stats for each worker (person)
	workerdict = {};

	for workerID in datamat['MTurk Worker ID']:
		if workerID in workerdict: continue;
		workerdict[workerID] = {}


# This might be helpful for accuracy and complexity
#	for row in np.arange(npdatamat.shape[0]):											 
#		tmp = npdatamat[row, 0].split('S')[1].split('M');#get filename					 
#		path = 'Montages/S' + tmp[0] + '/' + npdatamat[row, 0].split('.')[0] + 'key.txt';
#		workerID = npdatamat[row, 2];
#
#		with open(path, 'r') as keyfile:
#			key = pd.read_csv(keyfile, sep='\t');
#
#		key_filenames = key['Filename'];
#		num_bads = key['Sublocation'][100];
#		numobj = npdatamat[row, 4];								  
#		pairs = np.reshape(npdatamat[row, 5:numobj*2+5], (-1, 2));
#		correct = np.zeros(numobj)
#		workerdict[workerID]['Actual'] += numbads
#		workerdict[workerID]['Attempted'] += 1;
#		#convert xy positions into sub positions and mark if correct
#		for i, pair in enumerate(pairs):
#			if 'BAD' in key_filenames[get_subpos(int(pair))]: correct[i] = 1;
#				
#		#Organize results
#		if len(correct) == np.sum(correct) and len(correct) == num_bads:
#			workerdict[workerID]['Diff'] += 0
#		else:
#			#identified is sum(crrect)
#			#misidentified is len(correct)-sum(correct)
#			workerdict[workerID]['Diff'] += np.abs(num_bads-np.sum(correct)+(len(correct)-np.sum(correct)))

	# #4 and $8, Which particle is hardest to classify? Sample Montage with clicks on it

	# Populate a dictionary with the stats for each particle (good and bad both)
	particledict = {};
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
					workerdict[ID][particle] = {'Appeared': 0., 'CumAppeared':[0.], 'ErrorRate':0., 'CumErrorRate':[0.]}


	# Go through the rows of npdatamat
	# Each row contains all info about a time that a worker looked at an image! 
	for rownum in np.arange(npdatamat.shape[0]):											 
		# Who did this?
		workerID = npdatamat[rownum, 2];

		# Figure out which montage we're looking at
		tmp = npdatamat[rownum, 0].split('S')[1].split('M'); #get filename					 
		path = 'Montages/S' + tmp[0] + '/' + npdatamat[rownum, 0].split('.')[0] + 'key.txt';

		# Get info only for particles in this image 
 		with open(path, 'r') as keyfile:
			key = pd.read_csv(keyfile, sep='\t');			

		key_sublocs   = key['Sublocation'][0:100]; # ?? 
		num_bads	  = key['Sublocation'][100];   # number of bad particles in this image
		key_filenames = key['Filename'][0:100];	# list of particles in this image

		# Note the appearance of each particle in appropriate dicts
		for particle in key_filenames:
			# particle info: particle was in an image
			particledict[particle]['Appeared'] += 1.; 
			# worker info: worker saw the particle
			workerdict[workerID][particle]['Appeared'] += 1.
			workerdict[workerID][particle]['CumAppeared'].append(workerdict[workerID][particle]['Appeared'] + 1.)


		# Track locations of bad particles		
		badlookup = {} # dictionary that gives true (is bad) or false (is good) for each image location
		badlocs = [] # image locations of bad particles; eventually we want to reduce this down to bad particles NOT ID'ed
		for i, val in enumerate(key_sublocs):
			badlookup[val] = ('BAD' in key_filenames[i])
			if badlookup[val]: badlocs.append(val)


		# Go through locations clicked on (i.e. marked as bad) and organize the information about the location
		numobj = npdatamat[rownum, 4];								 #number of objects identified (good or bad)
		pairs = np.reshape(npdatamat[rownum, 5:numobj*2+5], (-1, 2)); #pairs of xy points
		picked = []; # subpos of something clicked on, to check if they clicked on a particle more than once
		for i, pair in enumerate(pairs):
			if True in pd.isnull(pair): continue # skip null entries

			its_subpos = get_subpos(pair)
			if its_subpos in picked: continue; # skip things we've seen before

			part_file = key_filenames[its_subpos]
			N_app = workerdict[workerID][part_file]['Appeared'] 

			picked.append(its_subpos); # haven't seen before; add to list of things we've now seen

			#need to make sure all bad particles are in a workers picked particles
			if its_subpos in badlocs:
				badlocs.pop(badlocs.index(its_subpos)) # correctly ID'ed -> remove from badlocs
				particledict[part_file]['BCorrect'] += 1.; # note particle was correctly ID'ed
				#keep cumulative track of accuracy
				# running average over all images
				workerdict[workerID][part_file]['ErrorRate'] = ((N_app-1.)*workerdict[workerID][part_file]['ErrorRate'] + 0.) / N_app 
				# add current value to "history" list
				workerdict[workerID][part_file]['CumErrorRate'].append(workerdict[workerID][part_file]['ErrorRate'])
			else: 
				particledict[part_file]['GMisclassified'] += 1.; # note that particle was misclassified
				#cumulative track of worker accuracy for this particle
				workerdict[workerID][part_file]['ErrorRate'] = ((N_app-1.)*workerdict[workerID][part_file]['ErrorRate'] + 1.) / N_app
				workerdict[workerID][part_file]['CumErrorRate'].append(workerdict[workerID][part_file]['ErrorRate'])

		# Now see how many the workers missed (bad particles they thought were good)
		for badloc in badlocs:
			picked.append(badloc); # no longer picked locations in picked, but now sublocs we have processed
			x, y = get_xypos(badloc);
			part_file = key_filenames[badloc]
			N_app = workerdict[workerID][part_file]['Appeared']

			particledict[part_file]['Missed'] += 1.; # note that particle was missed
			# cumulative track of worker accuracy for this particle
			workerdict[workerID][part_file]['ErrorRate'] = ((N_app-1.)*(workerdict[workerID][part_file]['ErrorRate']) + 1.) / N_app				   
			workerdict[workerID][part_file]['CumErrorRate'].append(workerdict[workerID][part_file]['ErrorRate'])
		# Go to every leftover sublocation not picked and mark correct (good particle not clicked, i.e. noted as good)
		for leftoverloc in key_sublocs:
			if leftoverloc in picked: continue; # if we have already processed this subloc, don't doublecount
			part_file = key_filenames[leftoverloc]
			N_app = workerdict[workerID][part_file]['Appeared']
			workerdict[workerID][part_file]['ErrorRate'] = ((N_app-1.)*(workerdict[workerID][part_file]['ErrorRate']) + 0.) / N_app				   
			workerdict[workerID][part_file]['CumErrorRate'].append(workerdict[workerID][part_file]['ErrorRate'])


	return particlelist, particledict, workerdict


#########
# PLOTTING
#########
def make_colors(Ncolors):
	return sns.color_palette("husl",Ncolors)


def main1():
	parts, pdict, wdict = getInfo()
	colors = make_colors(len(wdict))

	allacc = [];
	for i, ID in enumerate(wdict):
		for particle in wdict[ID]:
			if not 'BAD' in particle: continue
#			print(np.arange(wdict[ID][particle]['Appeared']))
#			print(1-np.array(wdict[ID][particle]['CumErrorRate']))
	
			#if wdict[ID][particle]['Appeared'] == 0: continue
			if wdict[ID][particle]['Appeared'] < 2: continue

			#diff = wdict[ID][particle]['CumErrorRate'][1] - wdict[ID][particle]['CumErrorRate'][-1]
			#plt.plot( i, diff, marker='o', ms=3, mec='None', mfc=colors[0] )
	#		plt.plot( 1.0 - np.array(wdict[ID][particle]['CumErrorRate']), color=colors[i], lw=0.5 )
			allacc.append(1.0 - np.array(wdict[ID][particle]['CumErrorRate'])[-1])

	plt.hist(allacc, bins=np.sqrt(len(allacc)))
	plt.xlabel('Accuracy')
	plt.ylabel('N')
	plt.title('Histogram of All Final Worker Accuracies')
	plt.show()
	plt.xlabel('Number of times particle viewed')
	plt.ylabel('Accuracy')
	plt.title('Accuracy vs Times Viewed per \'Expert\'')
	plt.show()


def main2():
	parts, pdict, wdict = getInfo()
	colors = make_colors(len(wdict))

	particleFinaldict = {}
	for i, ID in enumerate(wdict):
		part_finacc = -1*np.ones(len(particlelist))
		for particle in wdict[ID]:
			if not 'BAD' in particle: continue
#			print(np.arange(wdict[ID][particle]['Appeared']))
#			print(1-np.array(wdict[ID][particle]['CumErrorRate']))
			if wdict[ID][particle]==0: print(ID, particle, wdict[ID][particle])
			# for plotting individual error
			if wdict[ID][particle]['Appeared'] == 0: continue
			# for plotting delta in error rate
			#if wdict[ID][particle]['Appeared'] < 2: continue

			err = np.array(wdict[ID][particle]['CumErrorRate'])
			app = np.array(wdict[ID][particle]['CumAppeared'])
			
			# For plotting final accuracy per particle
			part_finacc[ pdict[particle]['idx'] ] = 1 - err[-1]

			if particle in particleFinaldict:
				particleFinaldict[particle]['CumAccuracy'] = err*app
				#particleFinaldict[particle]['CumAppeared'] += 
			else:
				particleFinaldict[particle]['CumAccuracy'] = err*app
				
			diff = wdict[ID][particle]['CumErrorRate'][1] - wdict[ID][particle]['CumErrorRate'][-1]
			plt.plot( i, diff, marker='o', ms=3, mec='None', mfc=colors[0] )
	#		plt.plot( 1.0 - np.array(wdict[ID][particle]['CumErrorRate']), color=colors[i], lw=0.5 )

	plt.show()	
	return;

	



	mostMissed = ['', 0];
	mostMisclassified = ['', 0];
	easiestbad = ['', 0];
	expert = pkl.load(open('expertparticledict.pkl'));
	misclasses = np.zeros(len(pdict));
	easys = np.zeros(len(pdict));
	misses = np.zeros(len(pdict));
	
	for i, particle in enumerate(pdict):
		appeared = pdict[particle]['Appeared']
		appearedE = expert[particle]['Appeared']
		if appearedE:
			misclassE = expert[particle]['GMisclassified'] / appearedE
			missedE = expert[particle]['Missed'] / appearedE
			easyE = expert[particle]['BCorrect'] / appearedE		  
			
			if misclassE: misclass = pdict[particle]['GMisclassified'] / appeared / misclassE
			if missedE: missed = pdict[particle]['Missed'] / appeared / missedE
			if easyE: easy = pdict[particle]['BCorrect'] / appeared / easyE
		else:
			misclass = pdict[particle]['GMisclassified'] / appeared
			missed = pdict[particle]['Missed'] / appeared		  
			easy = pdict[particle]['BCorrect'] / appeared		  

		misclasses[i] = misclass
		easys[i] = easy
		misses[i] = missed

		if misclass > mostMisclassified[1]:
			mostMisclassified = [particle, misclass];
		if missed > mostMissed[1]:			
			mostMissed = [particle, missed];
		if easy > easiestbad[1]:			
			easiestbad = [particle, easy];
			
	print(mostMissed, mostMisclassified, easiestbad)
	plt.hist(misclasses[np.nonzero(misclasses)], bins=np.sqrt(len(misclasses[np.nonzero(misclasses)])))	
	plt.xlabel('errors')								
	plt.ylabel('frequency')								
	plt.title('misclassifications')						
	plt.savefig('histMisClasses.jpeg')					
	plt.clf()										   

	plt.hist(misses[np.nonzero(misses)], bins=np.sqrt(len(misses[np.nonzero(misses)])))	
	plt.xlabel('errors')								
	plt.ylabel('frequency')								
	plt.title('Missed particles')						
	plt.savefig('histMisses.jpeg')					
	plt.clf()										   
	
	plt.hist(easys[np.nonzero(easys)], bins=np.sqrt(len(easys[np.nonzero(easys)])))	
	plt.xlabel('errors')								
	plt.ylabel('frequency')								
	plt.title('Easiest Bad to identify')						
	plt.savefig('histEasys.jpeg')					
	plt.clf()										   


def main():
	parts, pdict, wdict = getInfo()
	colors = make_colors(len(wdict))

	parts_plotted = []
	for p in parts:
		if (pdict[p]['Appeared']>0) and ('BAD' in p): 
			parts_plotted.append(p)

	for i, ID in enumerate(wdict):
		part_finacc = -1*np.ones(len(parts_plotted))
		for particle in parts_plotted:
			if not 'BAD' in particle: continue
			# in case *this worker* didn't see it
			if wdict[ID][particle]['Appeared'] == 0: continue

			err = np.array(wdict[ID][particle]['CumErrorRate'])
			app = np.array(wdict[ID][particle]['CumAppeared'])
			
			# For plotting final accuracy per particle
			#idx is returning numbers that are indices for particlelist, not part_finacc
			part_finacc[ pdict[particle]['idx'] ] = 1 - err[-1] 

		# Plot final accuracy for each particle, for each worker (connect worker points)
		plt.plot( part_finacc, ls='-', marker='o', lw=0.5, ms=2, color=colors[i], mec=colors[i], mfc='None' )

	plt.show()	



def get_subpos(xy):
	#10x10, so each subimg is 50x50
	#data is in col, row format
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
if __name__ == "__main__":
	main();
#	main1();
