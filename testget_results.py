import pickle as pkl
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import random
from PIL import Image

def main():
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

	filename = 'results-test.csv';
	workerdict = {};

	xylocs = np.array([]);        

	with open(filename, 'r') as target:
#		strmat = np.genfromtxt(target, dtype='str', missing_values='NONE');
		datamat = pd.read_csv(filename)

	npdatamat = np.array(datamat);

	for workerID in datamat['MTurk Worker ID']:
		if workerID in workerdict: continue;
		workerdict[workerID] = {'Diff':0, 'Actual':0, 'Attempted':0};


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
	particledict = {};
	for i in range(10):
		with open('Montages/S' + str(i) + '/S' + str(i) + 'key.txt', 'r') as target:
			for line in target:
				particle = line.strip('\n')
				if len(particle) == 0: continue;
				#Missed is bad particle not clicked, Bcorrect is bad particle clicked, Gmissclasssfied is good particle clicked, appeared is # times seen
				if particle in particledict: continue;
				else: particledict[particle] = {'Missed':0, 'BCorrect':0, 'Appeared':0, 'GMisclassified':0}
			

	particledict.pop('Real particle names:')
	particledict.pop('Bad particle names:')

	for rownum in np.arange(npdatamat.shape[0]):											 
		tmp = npdatamat[rownum, 0].split('S')[1].split('M');#get filename					 
		path = 'Montages/S' + tmp[0] + '/' + npdatamat[rownum, 0].split('.')[0] + 'key.txt';

#		samplefile = 'S0M0.gif';
	#	im = Image.open('Montages/S0/'+samplefile)
		#plt.imshow(im)
# 		with open('Montages/S0/S0M0key.txt', 'r') as keyfile:
#			key = pd.read_csv(keyfile, sep='\t');            
 		with open(path, 'r') as keyfile:
			key = pd.read_csv(keyfile, sep='\t');            
	
		key_filenames = key['Filename'][0:100];
		for particle in key_filenames:
			particledict[particle]['Appeared'] += 1.;
		key_sublocs = key['Sublocation'][0:100];
		num_bads = key['Sublocation'][100];
	
		badlookup = {}
		badlocs = []
		for i, val in enumerate(key_sublocs):
			if 'BAD' in key_filenames[i]: 
				badlookup[val] = True;
				badlocs.append(val)
			else: badlookup[val] = False;
	
#		for rownum in range(npdatamat.shape[0]):
#			if npdatamat[rownum, 0] != samplefile: break
		numobj = npdatamat[rownum, 4];								 #number of objects identified  
		pairs = np.reshape(npdatamat[rownum, 5:numobj*2+5], (-1, 2)); #pairs of xy points

		picked = [];#in case they cliked on a particle more than once
		for i, pair in enumerate(pairs):
			if True in pd.isnull(pair): continue
			if get_subpos(pair) in picked: continue;
			picked.append(get_subpos(pair));
			#need to make sure all bad particles are in a workers picked particles
			if get_subpos(pair) in badlocs:
				badlocs.pop(badlocs.index(get_subpos(pair)))
				particledict[key_filenames[get_subpos(pair)]]['BCorrect'] += 1.;
			else:
				particledict[key_filenames[get_subpos(pair)]]['GMisclassified'] += 1.;
		for badloc in badlocs:
			x, y = get_xypos(badloc);
			particledict[key_filenames[badloc]]['Missed'] += 1.;
#				plt.scatter(x+get_rand50(), y+get_rand50(), color='g')
			#if you clicked on it and its bad, that's good
#			c = 'b' if badlookup[get_subpos(pair)] else 'r'
#			plt.scatter(pair[0], pair[1], color=c)

#	plt.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')
#	plt.tick_params(axis='y', which='both', left='off', right='off', labelleft='off')
#	plt.show()

	mostMissed = ['', 0];
	mostMisclassified = ['', 0];
	easiestbad = ['', 0];
	for particle in particledict:
		appeared = particledict[particle]['Appeared']
		if appeared == 0: continue;
		misclass = particledict[particle]['GMisclassified'] / appeared
		missed = particledict[particle]['Missed'] / appeared
		easy = particledict[particle]['BCorrect'] / appeared

		if misclass > mostMisclassified[1]:
			mostMisclassified = [particle, misclass];
		if missed > mostMissed[1]:			
			mostMissed = [particle, missed];
		if easy > easiestbad[1]:			
			easiestbad = [particle, easy];
			
	print(mostMissed, mostMisclassified, easiestbad)
	pkl.dump(particledict, open('expertparticledict.pkl', 'w'))


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
