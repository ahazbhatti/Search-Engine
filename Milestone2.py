from collections import defaultdict
from nltk.tokenize import RegexpTokenizer
from Milestone1 import Crawler
import json
import math
import pickle
import time

class search:

    def __init__(self, x):
        self.datalist = [] 
        self.database = {}   
        self.corpusCount = 0
        self.x = x
        self.start_time = 0
        
        with open("df_file.pickle", 'rb') as df_file:
            self.df_dict = pickle.load(df_file)

        with open("index.pickle",'rb') as data:   
            self.database = pickle.load(data)  

    def token_query(self):
        regex = RegexpTokenizer(r'\w+')
        query = input("Search: ")
        self.start_time = time.time()
        query_token = regex.tokenize(query.lower())
        query_dict = defaultdict(int)  

        for word in query_token: 
            query_dict[word.lower()] += 1 
            self.datalist.append(word.lower())

        return self.tf_idf(query_dict)

    def tf_idf(self, query_dict):
        for word in query_dict.keys():   
            tf = query_dict[word]
            log_freq = 1 + math.log10(tf)
            if self.df_dict[word] == set():
                idf = math.log10(len(self.x.valid_list))
            else:
                idf = math.log10(len(self.x.valid_list)/len(self.df_dict[word]))
            query_dict[word] = log_freq*idf
            
        return query_dict

    def cosine_similarity(self):
        query_dict = self.token_query()
        cos_dict = defaultdict(int)
        
        dot_product = 0
        query_squared = 0
        database_squared = 0
        for word in query_dict.keys():
            if word in self.database.keys(): 
                for file in self.database[word].keys():
                    for word2 in query_dict.keys():
                        if word2 == word:
                            dot_product += query_dict[word]*self.database[word][file][0]
                            query_squared += query_dict[word]**2
                            database_squared += self.database[word][file][0]**2

                    cos_dict[file] = dot_product / ( math.sqrt(query_squared) * math.sqrt(database_squared) )

        return cos_dict
    
    def printResult(self):
        result = sorted(self.cosine_similarity().items(), key = lambda x: x[1])
        top = result[:20]
        if len(result) == 0:
            print("\nNo results found")
        else:   
            with open("WEBPAGES_RAW/bookkeeping.json") as data_file: 
                data = json.load(data_file)

            print("\nAbout", len(result), "results in", time.time() - self.start_time, "seconds\n\n")
            for counter, file in enumerate(top):
                print(counter+1, data[file[0]])
                for word in set(self.datalist):
                    if file[0] in self.database[word].keys():
                        x = self.database[word][file[0]][1:]
                        print("  ",len(x), "occurrence(s) at position(s):", x, "in file", file[0])
                print()
                

if __name__ == "__main__":
    x = Crawler()
    y = search(x)
    y.printResult()

