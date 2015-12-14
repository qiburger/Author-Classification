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
			trim_line = line.rsplit('\n', 1)[0]
			para_bool_line = trim_line.split('\t')
			paragraphs.append(para_bool_line[0])
			boolean.append(int(para_bool_line[1]))
	author_dict["paragraphs"] = paragraphs
	author_dict["booleans"] = boolean
	return author_dict
def extract_features(author_dict, test_list, stop_list):
	word_vector = TfidfVectorizer( analyzer="word", ngram_range=(3,3),
		max_features = None, binary=False, stop_words=stop_list)
	char_vector = TfidfVectorizer( analyzer="char", ngram_range=(4,5), 
		max_features = None, binary=False, min_df=0)
	corpus = author_dict["paragraphs"]
	classes = author_dict["booleans"]
	vectorizer = FeatureUnion([ ("chars", char_vector), ("words", word_vector) ])
	svm = SVC(kernel="linear")
	pipeline = Pipeline([("features", vectorizer), ("svm", svm)])
	pipeline.fit(corpus, classes)	
	y = pipeline.predict(test_list)
	return y
def combine_results():
	qi_path = '../baseline_v2.txt'
	chris_path = 'test.txt'
	qi_data = []
	chris_data = []
	with open(qi_path, 'rU') as f:
		for line in f:
			trim_line = line.rsplit('\n', 1)[0]
			qi_data.append(int(trim_line))
	with open(chris_path, 'rU') as f2:
		for line in f2:
			trim_line = line.rsplit('\n', 1)[0]
			chris_data.append(int(trim_line))
	count = 0
	for i in xrange(0, len(qi_data)):
		if(chris_data[i] == 1 and qi_data[i] == 0):
			count += 1
			chris_data[i] = 0
	with open('combined.txt', 'w') as f3:
		for value in chris_data:
			f3.write(str(value) + '\n')	
if __name__ == '__main__':
	path = '../project_articles_train'
	author_dict = load_file(path)
	stop_list = []
	with open("../stopwords.txt", 'rU') as stop:
		for line in stop:
			trim_line = line.rsplit('\n', 1)[0]
			stop_list.append(trim_line)	
	test_list = []
	with open("../project_articles_test", 'rU') as testfile:
		for line in testfile:
			trim_line = line.rsplit('\n', 1)[0]
			test_list.append(trim_line)	
	predictions = extract_features(author_dict, test_list, stop_list)
	with open("test.txt", 'w') as f:
		for entry in predictions:
			f.write(str(entry) + '\n')
	#combine_results()

