import json
import requests
import os
import urllib2
import time 
api_key = "55bf221011d6f639454054ae5044e5fb:1:73719482"
api_key2 = "e04f440978b898f0490bf123bfb5db5a:14:73722280"

def get_all_files(directory):
	dir_list = os.listdir(directory)
	filepaths = []
	for file in dir_list:
		filepaths.append(directory + "/" +file)
	return filepaths

def times_call(sentence, key):
	url = 'http://api.nytimes.com/svc/search/v2/articlesearch.json?fq=body%3A+%28%22'
	url = url + sentence + "%22%29&"
	url = url + "api-key=" + key
	r = requests.get(url)

	#switch between api_key's if one is no good
	if(not r and key == api_key):
		print(url)
		print("Bad request, first time")
		time.sleep(1)
		return times_call(sentence[:100], api_key2)
	elif(not r and key == api_key2):
		print(url)
		print("Bad request, second time")
		time.sleep(1)
		return times_call(sentence[:100], api_key)
	#grab the json object and check if we have returns
	json_obj = r.json()
	hits = json_obj["response"]["meta"]["hits"]
	if(hits == 0):
		print(url)
		print("couldn't find any data on the json pull")
		return False
	if(not json_obj):
		print("Couldn't find json_obj")
		return False
	different_results = json_obj["response"]["docs"]
	with open('poss_results.txt', 'w') as g:
		for result in different_results:
			if(not result["byline"]):
				#print(result)
				#print("No authors for this article")
				g.write(sentence + '\t' + '???')
				continue
			people = result["byline"]["person"]
			for person in people:
				if(not "firstname" in person):
					#print(person)
					continue
				first = person["firstname"].lower()
				first = first.encode('utf8')
				last = person["lastname"].lower()
				last = last.encode('utf8')
				if(first == "gina" and last == "kolata"):
					g.write(sentence + '\t1\n')
					#print("Hit a legit article")
				else:
					#print("Not legit, Written by: " + first + " " + last + "\n")

					g.write(sentence + '\t0\n')
	return True
def parse_train(author_dict):
	with open('../project_articles_train', 'rU') as f:
		i = 0
		for line in f:
			i+=1
			line = line.rsplit('\n', 1)[0]
			sent = line.split('\t')[0]
			author = line.split('\t')[1]
			author_dict[sent] = int(author)
			if(i==500):
				break

if __name__ == '__main__':
	#author_dict = {}
	#parse_train(author_dict)
	path = '/home1/c/cis530/project/data/project_articles_test'
 	i = 0
	with open(path, 'rU') as f:
		for line in f:
			i+=1
			sentence = line.rsplit('\n', 1)[0]	
			if(not times_call(sentence, api_key)):
				print(sentence)
			if(i == 500):
				break	
	
