if __name__ == '__main__':
	predictions = []
	with open('poss_results.txt', 'rU') as f:
		for line in f:
			sentence = line.rsplit('\t', 1)[1]
			predictions.append(sentence)
	with open('nums.txt', 'w') as g:
		for prediction in predictions:
			g.write(prediction)	
