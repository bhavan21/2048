import sys
from math import log
import matplotlib.pyplot as plt

with open(sys.argv[1]) as f:

	high_value = []
	max_score =[]
	timesteps = []

	for line in f:

		x = line.split()
		x= [int(i) for i in x]

		high_value.append(int(log(x[1])/log(2)))
		max_score.append(x[2])
		timesteps.append(x[3])

	percentile_high = [0 for i in range(12)]
	highcount = [0 for i in range(12)]
	for i in high_value:
		highcount[i] += 1
		for j in range(1,i+1):
			percentile_high[j] += 1

	percentile_high = [float(i/len(high_value))*100 for i in percentile_high]
	del percentile_high[0]

	x_high = [pow(2,i) for i in range(1,12)]
	fig = plt.figure()
	plt.plot(percentile_high)
	fig.suptitle(' Performance of agent on tile value observed', fontsize=16)
	plt.xlabel('Tile Value in powers of 2')
	plt.ylabel('percent of runs the tile value is observed')
	# plt.show()
	fig.savefig('percentile.jpg')


	del highcount[0]
	highcount = [float(i/len(high_value))*100 for i in highcount]

	xs = [i for i in range(1,12)]
	fig = plt.figure()

	plt.bar(xs,highcount,1,align='center')
	plt.xticks(xs)
	fig.suptitle(' Percentage of highest tile value observed', fontsize=16)
	plt.xlabel('Tile Value in powers of 2')
	plt.ylabel('percent of runs the highest tile value is observed')
	# plt.show()
	fig.savefig('highcountpercent.jpg')



	fig = plt.figure()
	plt.hist(max_score,density =True,bins = 15)
	fig.suptitle('Histogram for scores observed',fontsize=16)
	plt.xlabel('Scores per game')
	plt.ylabel('Probability for score per episode')
	# plt.show()
	fig.savefig('histogram.jpg')
