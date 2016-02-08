import numpy as np
#import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import random
from PIL import Image

def main():

	filename = 'results-Actual.csv';
	workerdict = {};

	xylocs = np.array([]);        

	with open(filename, 'r') as target:
#		strmat = np.genfromtxt(target, dtype='str', missing_values='NONE');
		datamat = pd.read_csv(filename)

	npdatamat = np.array(datamat);

	for workerID in datamat['MTurk Worker ID']:
		if workerID in workerdict: continue;
		workerdict[workerID] = {'Diff':0, 'Actual':0, 'Attempted':0};



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

	samplefile = 'S0M0.gif';							 
	im = Image.open('Montages/S0/'+samplefile)			 
	plt.imshow(im)		

#	x, y = get_xypos(0);
#	print(x, y, 0)			
#	plt.scatter(x, y)   
#
#	x, y = get_xypos(8);
#	print(x, y, 8)			
#	plt.scatter(x, y)   
#
#	x, y = get_xypos(17);
#	print(x, y, 17)			
#	plt.scatter(x, y)   
#	plt.show()
#	return;

	for rownum in np.arange(20):											 
		tmp = npdatamat[rownum, 0].split('S')[1].split('M');#get filename					 
		path = 'Montages/S' + tmp[0] + '/' + npdatamat[rownum, 0].split('.')[0] + 'key.txt';

# 		with open('Montages/S0/S0M0key.txt', 'r') as keyfile:
#			key = pd.read_csv(keyfile, sep='\t');            
 		with open(path, 'r') as keyfile:
			key = pd.read_csv(keyfile, sep='\t');            
	
		key_filenames = key['Filename'][0:100];
		for particle in key_filenames:
			particledict[particle]['Appeared'] += 1;
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
		numobj = npdatamat[rownum, 4];								  
		pairs = np.reshape(npdatamat[rownum, 5:numobj*2+5], (-1, 2));

		for i, pair in enumerate(pairs):
			#need to make sure all bad particles are in a workers picked particles
			if get_subpos(pair) in badlocs:
				badlocs.pop(badlocs.index(get_subpos(pair)))
				particledict[key_filenames[get_subpos(pair)]]['BCorrect'] += 1;
			else:
				particledict[key_filenames[get_subpos(pair)]]['GMisclassified'] += 1;

			#if you clicked on it and its bad, that's good					
#			print(badlookup)
			c = 'b' if badlookup[get_subpos(pair)] else 'r'					
			plt.scatter(pair[0], pair[1], color=c, s=9)                          

		for badloc in badlocs:
			print(badloc)
			x, y = get_xypos(badloc);
			particledict[key_filenames[badloc]]['Missed'] += 1;
			plt.scatter(x+get_rand50(), y+get_rand50(), color='g', s=9)		




	plt.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')
	plt.tick_params(axis='y', which='both', left='off', right='off', labelleft='off')
	plt.show()
	





def get_subpos(xy):
	#10x10, so each subimg is 50x50
	#data is in col, row format
#	row = xy[0] // 50.
#	col = xy[1] // 50.
	return (xy[0] // 50.)*10 + (xy[1] // 50.)

def get_xypos(subloc):
	#go from sublocation to xy pos
	return (subloc // 10) * 50, ((subloc % 10))*50.

def get_rand50():
	#add a random number
	return random.randint(1, 49);

if __name__ == "__main__":
	main();
