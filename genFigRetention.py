import cPickle as pkl
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# generate figure a, fraction of workers vs montages attempted
def main():
	wdict = pkl.load(open('workerdict-actual.pkl', 'r'))
	
	Nworkers = float(len(np.unique(wdict.keys())));
	nwork = np.zeros(300);
	for xm in range(1, 301):
		for i, ID in enumerate(wdict):
			if len(wdict[ID]['Montage']['CumMontagesSeen']) >= xm: nwork[xm-1] += 1

		
	
	print(nwork)
	plt.plot( range(300), nwork / Nworkers )
	plt.title('Worker Retention')
	plt.xlabel('Number of Montages Attempted')
	plt.ylabel('Fraction of Workers')
	plt.xticks( np.arange(0, 300, 30) )
#	plt.savefig('Figures/FigA_retention', format='png')
	plt.show()











if __name__ == "__main__":
	main()
