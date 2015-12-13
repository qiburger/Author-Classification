from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.metrics import classification_report
from sklearn.svm import SVC

def load_file(path):
	author_dict = {}
	with open(path, 'rU') as f:
		i = 0
		for line in f:
			i+=1
			if(i == 200):
				break	
			trim_line = line.rsplit('\n', 1)[0]
			para_bool_line = trim_line.split('\t')
			author_dict[para_bool_line[0]] = int(para_bool_line[1])			 
	return author_dict
def extract_features(author_dict, test_dict):
	word_vector = TfidfVectorizer( analyzer="word", ngram_range=(3,3),
		max_features = 2000, binary = False )
	char_vector = TfidfVectorizer( analyzer="char", ngram_range=(3,3), 
		max_features = 2000, binary=False, min_df=0 )
	corpus = author_dict.keys()
	classes = author_dict.values()
	vectorizer = FeatureUnion([ ("chars", char_vector), ("words", word_vector) ])
	#x = vectorizer.fit(corpus, classes).transform(corpus)
	svm = SVC(kernel="linear")
	pipeline = Pipeline([("features", vectorizer), ("svm", svm)])
	pipeline.fit(corpus, classes)
	y = pipeline.predict(test_dict.keys())
	print(y)	
	print(classification_report(y, test_dict.values()))

if __name__ == '__main__':
	path = '../project_articles_train'
	author_dict = load_file(path)
	test_path = '../actual_test/poss_results.txt'	
	test_dict = load_file(path)
	feature_set = extract_features(author_dict, test_dict)
