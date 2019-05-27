from xml.dom import minidom
from sklearn import datasets
from sklearn import svm
from sklearn import neighbors
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import RegexpTokenizer
import json
import codecs
import sys
import unicodedata
import nltk
import math
import string

#Feature Flags :
USE_RELEVANCE_SCORES = False #Use and filter the words based on their relevance score
ADD_RELATED_WORDS = False #Fetch Synonyms, Hyponyms and Hypernyms from Wordnet
REMOVE_PUNCTUATION = True #Remove Punctuation
REMOVE_STOP_WORDS = True #Remove Stop Words
LOWER_CASE = True #Process the data stricly in lower case
HEAD = False	 #Include the word(lexelt) itself in the context
STEM = True #Perform stemming on tokens
CONTEXT_WINDOW = 10 #Window of words to generate context from

def parse_data(input_file):
	'''
	Parse the .xml data file to return a dictionary of list of contexts
	'''

	print '\tCONTEXT_WINDOW\t\t= ',CONTEXT_WINDOW
	print '\tADD_RELATED_WORDS\t= ',ADD_RELATED_WORDS
	print '\tHEAD\t\t\t= ',HEAD

	tokenizer = RegexpTokenizer(r'\w+')
	
	xmldoc = minidom.parse(input_file)
	data = {}
	lex_list = xmldoc.getElementsByTagName('lexelt')
	for node in lex_list:
		lexelt = node.getAttribute('item')
		data[lexelt] = []
		inst_list = node.getElementsByTagName('instance')
		for inst in inst_list:
			instance_id = inst.getAttribute('id')
			l = inst.getElementsByTagName('context')[0]

			tokens_left = tokenizer.tokenize((l.childNodes[0].nodeValue).replace('\n', ''))[-CONTEXT_WINDOW:]
			head = tokenizer.tokenize(l.childNodes[1].firstChild.nodeValue)
			tokens_right = tokenizer.tokenize((l.childNodes[2].nodeValue).replace('\n', ''))[:CONTEXT_WINDOW]

			if HEAD == True:
				context = tokens_left + head + tokens_right
			else:
				context = tokens_left + tokens_right
			

			if ADD_RELATED_WORDS == True:
					mid = len(context) // 2
					mid_five = [mid - 2, mid - 1, mid, mid + 1, mid + 2]
					for w in mid_five:
						context += related_nyms(context[w])

			# context = (l.childNodes[0].nodeValue + l.childNodes[1].firstChild.nodeValue + l.childNodes[2].nodeValue).replace('\n', '')
			data[lexelt].append((instance_id, context))
	
	return data

def relevance(word, sense, data):
	sense_list = sense[word]
	contexts = data[word]

	word_set = []
	temp = []

	for sense in sense_list:
		if sense not in temp:
			temp.append(sense)

	for sense in temp:
		sorted_relevance = {}
		relevance = {}
		for instance_id, context in contexts:
			for token in context:
				i = 0
				num_sc = 0
				num_c = 0
				for instance_id, context in contexts:
					if token in context and sense_list[i] == sense:
						num_sc += 1 #word frequency in context
					if token in context:
						num_c += 1 #word frequency in all contexts
					i += 1

				if 1 - num_sc/(num_c*1.0) == 0:
					relevance[token] = 32767
				elif num_sc/(num_c*1.0) == 0:
					relevance[token] = -32767
				else:
					relevance[token] = math.log((float)(num_sc/(num_c*1.0))/(1 - num_sc/(num_c*1.0)))
		
		sorted_relevance = sorted(relevance.items(), key = lambda d : d[1])
		cutoff=len(sorted_relevance) // 2

		for i in xrange(0, cutoff + 1):
			token = sorted_relevance[i][0]
			word_set.append(token)

		#for pair in sorted_relevance:
		#	if value >= 0 and token not in word_set:
		#		word_set.append(token)

	return word_set

def related_nyms(word):
	syn = []
	hyper = []
	hypo = []

	word_synsets = wn.synsets(word)
	for s in word_synsets:
		# Get synonyms
		for lemma in s.lemma_names():
			if lemma not in syn:
				syn.append(lemma)
		# Get hypernyms
		for s_hyper in s.hypernyms():
			hyper_w = s_hyper.name().split('.')[0]
			if hyper_w not in hyper:
				hyper.append(hyper_w)
		# Get hyponyms
		for s_hypo in s.hyponyms():
			hypo_w = s_hypo.name().split('.')[0]
			if hypo_w not in hypo:
				hypo.append(hypo_w)
	related_words = syn + hyper + hypo
	return related_words

def build_train_vector(data, sense):
	'''
	Build the context vector for each instance of a word in the training data
	'''

	stop_words = stopwords.words('english')
	stemmer = SnowballStemmer("english", ignore_stopwords = True)

	print '\tUSE_RELEVANCE_SCORES\t= ',USE_RELEVANCE_SCORES
	print '\tREMOVE_PUNCTUATION\t= ',REMOVE_PUNCTUATION
	print '\tREMOVE_STOP_WORDS\t= ',REMOVE_STOP_WORDS
	print '\tLOWER_CASE\t\t= ',LOWER_CASE
	print '\tSTEM\t\t\t= ',STEM
	

	vector = {}
	word_sets = {}
	for word, key in data.iteritems():
		word_set = []
		
		if USE_RELEVANCE_SCORES == True:
			word_set = relevance(word, sense, data)
			
		vector[word] = []
		for instance_id, context in key:
			for item in context:
				if REMOVE_PUNCTUATION == True:
						for c in string.punctuation:
							item = item.replace(c, '')
				
				if STEM == True:
					if LOWER_CASE == True:
						after = stemmer.stem(item.lower())
					else:
						after = stemmer.stem(item)
				else:
					if LOWER_CASE == True:
						after = item.lower()
					else:
						after = item

				if (after not in word_set):
					if REMOVE_STOP_WORDS == False:
						word_set.append(after)
					elif (after not in stop_words):
						word_set.append(after)

		word_sets[word] = word_set

		for instance_id, context in key:
			context_vector = [0] * len(word_set)
			for item in context:
				if REMOVE_PUNCTUATION == True:
						for c in string.punctuation:
							item = item.replace(c, '')
				
				if STEM == True:
					if LOWER_CASE == True:
						after = stemmer.stem(item.lower())
					else:
						after = stemmer.stem(item)
				else:
					if LOWER_CASE == True:
						after = item.lower()
					else:
						after = item

				if after in word_set:
					index = word_set.index(after)
					context_vector[index] += 1
			vector[word].append(context_vector)

	return vector, word_sets

def build_dev_vector(data, word_sets):
	'''
	Build the context vector for each instance of a word in the test data
	'''
	vector = {}
	id_lists = {}

	stemmer = SnowballStemmer("english", ignore_stopwords=True)

	for word, key in data.iteritems():
		vector[word] = []
		word_set = word_sets[word]

		id_list = []

		for instance_id, context in key:

			id_list.append(instance_id)

			context_vector = [0] * len(word_set)
			for item in context:
				if REMOVE_PUNCTUATION == True:
					for c in string.punctuation:
						item = item.replace(c, '')

				if STEM == True:
					if LOWER_CASE == True:
						after = stemmer.stem(item.lower())
					else:
						after = stemmer.stem(item)
				else:
					if LOWER_CASE == True:
						after = item.lower()
					else:
						after = item

				if after in word_set:
					index = word_set.index(after)
					context_vector[index] += 1
			vector[word].append(context_vector)

		id_lists[word] = id_list

	return vector, id_lists

def build_sense(input_file):
	'''
	Count the frequency of each sense
	'''

	xmldoc = minidom.parse(input_file)
	data = {}
	lex_list = xmldoc.getElementsByTagName('lexelt')
	sense_dict = []
	for node in lex_list:
		lexelt = node.getAttribute('item')
		data[lexelt] = {}
		inst_list = node.getElementsByTagName('instance')
		sense_list = []
		for inst in inst_list:
			sense_id = inst.getElementsByTagName('answer')[0].getAttribute('senseid')
			sense_list.append(sense_id)
			if sense_id not in sense_dict:
				sense_dict.append(sense_id)

		data[lexelt] = sense_list

	for lexelt, sense_list in data.iteritems():
		for i in range(len(sense_list)):
			if sense_list[i] is not "U":
				sense_list[i] = sense_dict.index(sense_list[i])
			else:
				sense_list[i] = -1

	return data, sense_dict

def SVC_predict(train_vector, sense, dev_vector):
	clfSVC = svm.LinearSVC()

	predict = {}

	for key, context_list in train_vector.iteritems():
		sense_list = sense[key]

		clfSVC.fit(context_list, sense_list)
		
		context_dev = dev_vector[key]
		predict_sense = clfSVC.predict(context_dev)

		predict[key] = predict_sense

	return predict

def KNN_predict(train_vector, sense, dev_vector):
	clfKNN = neighbors.KNeighborsClassifier()

	predict = {}

	for key, context_list in train_vector.iteritems():
		sense_list = sense[key]

		clfKNN.fit(context_list, sense_list)
		
		context_dev = dev_vector[key]
		predict_sense = clfKNN.predict(context_dev)

		predict[key] = predict_sense

	return predict

def replace_accented(input_str):
    nkfd_form = unicodedata.normalize('NFKD', input_str)
    return u"".join([c for c in nkfd_form if not unicodedata.combining(c)])

def output(predict_vector, id_lists, sense_dict, output_file):
	outfile = codecs.open(output_file, encoding = 'utf-8', mode = 'w')

	for key, sense_list in predict_vector.iteritems():
		id_list = id_lists[key]
		for i in range(len(sense_list)):
			string = key + ' ' + id_list[i] + ' ' + sense_dict[sense_list[i]]
			#string = replace_accented(string)
			outfile.write(string + '\n')

	outfile.close()

if __name__ == '__main__':
	if len(sys.argv) != 5:
		print 'Usage: python main.py <training_file> <test_file> <svm_output_file> <knn_output_file>'
		sys.exit(0)
	else:
		print '\nParsing training data...'
		train_data = parse_data(sys.argv[1])
		sense, sense_dict = build_sense(sys.argv[1])

		print '\nBuilding training vector based on extracted features...'
		train_vector, word_sets = build_train_vector(train_data, sense)

		print '\nParsing test data...'
		dev_data = parse_data(sys.argv[2])
		print '\nBuilding test vector...'
		dev_vector, id_lists = build_dev_vector(dev_data, word_sets)

		print '\nSVC in progress...'
		svc_predict_vector = SVC_predict(train_vector, sense, dev_vector)
		output(svc_predict_vector, id_lists, sense_dict, sys.argv[3])

		print '\nKNN in progress...'
		knn_predict_vector = KNN_predict(train_vector, sense, dev_vector)
		output(knn_predict_vector, id_lists, sense_dict, sys.argv[4])

		print '\nProcess Complete.\n'

		
