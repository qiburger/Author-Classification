from single_feature import *
import pickle

'''
	Hoping to further remove false negatives, we will examine if
	there are unique token that kolata uses. 

'''

def get_unique_tokens(training_file):
	unique_author_tokens = defaultdict(lambda: 0)
	unique_non_author_tokens = defaultdict(lambda: 0)

	test_paragraphs, kolata_list, non_kolata_list, corpus_distribution, stop_words, kolata_paragraph, non_kolata_paragraph, lemma_bank = load_serialized_lists()

	kolata_vocab = set(kolata_list)
	non_kolata_vocab = set(non_kolata_list)

	for token in kolata_vocab:
		if token not in non_kolata_vocab:
			unique_author_tokens[token] = corpus_distribution[token]

	for token in non_kolata_vocab:
		if token not in kolata_vocab:
			unique_non_author_tokens[token] = corpus_distribution[token]

	print unique_author_tokens
	print unique_non_author_tokens

	pickle.dump(unique_author_tokens, open('unique_author_tokens.p', 'wb'))
	pickle.dump(unique_non_author_tokens, open('unique_non_author_tokens.p', 'wb'))

	return unique_author_tokens, unique_non_author_tokens

if __name__ == "__main__":
	get_unique_tokens("project_articles_train")