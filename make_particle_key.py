import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import random
from PIL import Image
import pickle as pkl


def gen_dict():

	particledict = {};
	filename = 'results-Lakshmi.csv';
	datamat = pd.read_csv(filename);
	npdatamat = np.array(datamat);

	for rownum in range(npdatamat.shape[0]):
		numobj = npdatamat[rownum, 4]; #number of objects identified (good or bad)
		pairs = np.reshape(npdatamat[rownum, 5:numobj*2+5], (-1, 2)); #pairs of xy points
		keyfile = 'LakshmiKey/S0/' + npdatamat[rownum, 0].split('.')[0] + 'key.txt'

		keydf = pd.read_csv(keyfile, sep='\t')
		badlocs = [];
		for pair in pairs:
			subloc = get_subpos(pair);
			particledict[keydf['Filename'][subloc]] = False;
			badlocs.append(subloc);
			
		for i in range(100):
			if i in badlocs: continue;
			particledict[keydf['Filename'][i]] = True;


	pkl.dump(particledict, open('LakshmiKeydict.pkl', 'w'))
	return;
			











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
	gen_dict();
