from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.metrics import classification_report
from sklearn.svm import SVC

def load_file(path):
	author_dict = {}
	paragraphs = []
	boolean = []
	with open(path, 'rU') as f:
		i = 0
		for line in f:
			i+=1
			#if(i == 1500):
			#	break	
			trim_line = line.rsplit('\n', 1)[0]
			para_bool_line = trim_line.split('\t')
			paragraphs.append(para_bool_line[0])
			boolean.append(int(para_bool_line[1]))
	author_dict["paragraphs"] = paragraphs
	author_dict["booleans"] = boolean
	return author_dict
def extract_features(author_dict, test_dict):
	word_vector = TfidfVectorizer( analyzer="word", ngram_range=(3,3),
		max_features = 2000, binary = False )
	char_vector = TfidfVectorizer( analyzer="char", ngram_range=(3,3), 
		max_features = 2000, binary=False, min_df=0 )
	corpus = author_dict["paragraphs"]
	classes = author_dict["booleans"]
	vectorizer = FeatureUnion([ ("chars", char_vector), ("words", word_vector) ])
	#x = vectorizer.fit(corpus, classes).transform(corpus)
	svm = SVC(kernel="linear")
	pipeline = Pipeline([("features", vectorizer), ("svm", svm)])
	pipeline.fit(corpus, classes)	
	print("My total number of keys is " + str(len(author_dict["paragraphs"])))
	y = pipeline.predict(test_dict["paragraphs"])
	print("My total length is " + str(len(y)))
	print(classification_report(y, test_dict["booleans"]))
	return y

def check_accuracy(test_dict, predictions):
	i = 0
	correct = 0
	booleans = test_dict["booleans"]
	for prediction in predictions:
		if prediction == booleans[i]:
			correct += 1			
		i += 1
	print("We checked " + str(i) + " booleans and predictions")
	return float(correct) / i

if __name__ == '__main__':
	path = '../project_articles_train'
	author_dict = load_file(path)
	test_path = '../actual_test/poss_results.txt'	
	test_dict = load_file(test_path)
	predictions = extract_features(author_dict, test_dict)
	accuracy = check_accuracy(test_dict, predictions)
	print("My accuracy is " + str(accuracy))
	with open("test.txt", 'w') as f:
		for entry in predictions:
			f.write(str(entry) + '\n')

