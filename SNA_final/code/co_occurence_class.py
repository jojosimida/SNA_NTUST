import os, sys
import jieba, codecs, math
import jieba.posseg as pseg
import pickle

class cooccurence():
	def __init__(self, path):
		self.path = path

		self.all_dict = "all_dict.txt"
		self.split_chapter_dir = 'split_chapter/'
		self.all_name = 'all_name.pkl'
		self.all_2ndname = 'all_2ndname.pkl'
		self.process_dir = 'process_data/'


		self.cut_dict = self.path+self.all_dict
		self.chapter_path = self.path+self.split_chapter_dir
		self.process_path = self.path+self.process_dir

		self.all_name_path = self.path+self.all_name
		self.all_2ndname_path = self.path+self.all_2ndname

		self.all_name = pickle.load(open(self.all_name_path, 'rb'))
		self.all_2ndname_dict = pickle.load(open(self.all_2ndname_path, 'rb'))

		self.all_2ndname = list(self.all_2ndname_dict.values())

		# load dictionary
		jieba.load_userdict(self.cut_dict)


	def search_word(self, word):
		the_word = list(self.all_2ndname_dict.keys())[list(self.all_2ndname_dict.\
			values()).index(word)]

		return the_word


	def check_the_word(self, word):
		if word in self.all_2ndname:
			word = self.search_word(word)

		return word


	def create_relation(self, doc):
		names = {}
		relationships = {}
		lineNames = []

		with codecs.open(doc.encode('utf-8'), "r", "utf8") as f:
			for line in f.readlines():
			    
				# cut the word and return the part of speech to the list
				poss = pseg.cut(line)
				lineNames.append([])
				for w in poss:
				# to determine whether it is a person's name
					if w.flag != "nr" or len(w.word) < 2:
						continue

					organized_word = self.check_the_word(w.word)
					lineNames[-1].append(organized_word)

					if names.get(organized_word) is None:
						names[organized_word] = 0
						relationships[organized_word] = {}
					names[organized_word] += 1

		return names, relationships, lineNames



	def to_cooccurence(self, doc):
		names, relationships, lineNames = self.create_relation(doc)

		# for each segment
		for line in lineNames:                    
			for name1 in line:   
				# any two people in each segment
				for name2 in line:  
					if name1 == name2:
						continue

					# create new items if they haven't appeared at the same time
					if relationships[name1].get(name2) is None:        
				   		relationships[name1][name2]= 1
					else:
					    # the number of co-occurrences of two people plus 1
						relationships[name1][name2] = relationships[name1][name2]+1

		return names, relationships


	def write_data(self, names, relationships, ind):
		with codecs.open(self.process_path+ind+"_node.txt", "w", "utf8") as f:
			f.write("Id Label Weight\r\n")
			for name, times in names.items():
				f.write(name + " " + name + " " + str(times) + "\r\n")
	        
		with codecs.open(self.process_path+ind+"_edge.txt", "w", "utf8") as f:
			f.write("Source Target Weight\r\n")
			for name, edges in relationships.items():
				for v, w in edges.items():
					if w > 5:
						f.write(name + " " + v + " " + str(w) + "\r\n")