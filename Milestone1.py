from collections import defaultdict
from bs4 import BeautifulSoup
from nltk.tokenize import RegexpTokenizer
from urllib.parse import urlparse, urljoin
import json
import math
import os
import re
import pickle

class Crawler:
    """
    This class is responsible for scraping urls from the next available link in frontier and adding the scraped links to
    the frontier
    """

    def __init__(self):
        self.found = defaultdict(int) #Counts occurrences of items
        self.valid_list = []
        
        with open("WEBPAGES_RAW/bookkeeping.json") as file:
            data = json.load(file)

        file.close()

        for key, value in data.items():
            if self.is_valid(value):
                self.valid_list.append(key)

        self.df_dict = defaultdict(set) #stores df values for every word     

    def is_valid(self, url):
        """
        Function returns True or False based on whether the url has to be fetched or not. This is a great place to
        filter out crawler traps. Duplicated urls will be taken care of by frontier. You don't need to check for duplication
        in this method
        """

        x = url.split("?")
        self.found[x[0]] += 1
        if self.found[x[0]] > 30:
            return False
                
        x = url.split("/")
        if len(x) > len(set(x)) + 5:
            return False            
        
        parsed = urlparse(url)
        
        
        try:
            return "ics.uci.edu" in url \
                   and not re.match(".*\.(css|js|bmp|gif|jpe?g|ico" + "|png|tiff?|mid|mp2|mp3|mp4" \
                                    + "|wav|avi|mov|mpeg|ram|m4v|mkv|ogg|ogv|pdf" \
                                    + "|ps|eps|tex|ppt|pptx|doc|docx|xls|xlsx|names|data|dat|exe|bz2|tar|msi|bin|7z|psd|dmg|iso|epub|dll|cnf|tgz|sha1" \
                                    + "|thmx|mso|arff|rtf|jar|csv" \
                                    + "|rm|smil|wmv|swf|wma|zip|rar|gz|pdf)$", parsed.path.lower())

        except TypeError:
            print("TypeError for ", parsed)
            return False

    def calc_df(self):
        """Populates a dictionary with the key as every word
           and the value as a list of files in which they appear"""
        regex = RegexpTokenizer(r'\w+')
        for file in self.valid_list:
            html = ""
            temp = file.split('/')
            base = "WEBPAGES_RAW/" + str(temp[0])
            data = open(os.path.join(base, temp[1]), encoding="utf8").read()
            for i in BeautifulSoup(data, "lxml").find_all("html"):
                html = html + " " + i.text + " "
            html_tag = regex.tokenize(html)
            for word in html_tag:
                self.df_dict[word.lower()].add(file)

        df_file = open('df_file.pickle', 'wb')
        pickle.dump(self.df_dict, df_file)
        df_file.close()

    def tokenize(self, file):
        """Reads every file and tokenizes the resulting html"""
        regex = RegexpTokenizer(r'\w+')

        #Extra credit
        word_position = 0

        html = ""
        bold = ""
        h1 = ""
        h2 = ""
        h3 = ""

        #https://stackoverflow.com/questions/9139897/how-to-set-default-value-to-all-keys-of-a-dict-object-in-python
        word_dict = {}
        word_dict = defaultdict(lambda: [0], word_dict)

        
        temp = file.split('/')
        base = "WEBPAGES_RAW/" + str(temp[0])
        data = open(os.path.join(base, temp[1]), encoding="utf8").read()

        for i in BeautifulSoup(data, "lxml").find_all("html"):
            html = html + " " + i.text + " "
        html_tag = regex.tokenize(html)
        for i in BeautifulSoup(data, "lxml").find_all("h1"):
            h1 = h1 + " " + i.text + " "
        h1_tag = regex.tokenize(h1)
        for i in BeautifulSoup(data, "lxml").find_all("h2"):
            h2 = h2 + " " + i.text + " "
        h2_tag = regex.tokenize(h2)
        for i in BeautifulSoup(data, "lxml").find_all("h3"):
            h3 = h3 + " " + i.text + " "
        h3_tag = regex.tokenize(h3)
        for i in BeautifulSoup(data, "lxml").find_all("b"):
            bold = bold + " " + i.text + " "
        bold_tag = regex.tokenize(bold)

        #print("Folder: ", folder)

        for word in html_tag:
            word_dict[word.lower()][0] += 1
            word_dict[word.lower()].append(word_position)
            word_position += 1
            
        if len(h1_tag) > 0:
            for word in h1_tag:
                word_dict[word.lower()][0] += 3
        if len(h2_tag) > 0:
            for word in h2_tag:
                word_dict[word.lower()][0] += 2
        if len(h3_tag) > 0:
            for word in h3_tag:
                word_dict[word.lower()][0] += 1
        if len(bold_tag) > 0:
            for word in bold_tag:
                word_dict[word.lower()][0] += 1

        return word_dict
        
    def tf_idf(self, word_dict):
        for word in word_dict.keys():   
            tf = word_dict[word][0]
            log_freq = 1 + math.log10(tf)
            if self.df_dict[word] == set():
                idf = math.log10(len(self.valid_list))
            else:
                idf = math.log10(len(self.valid_list)/len(self.df_dict[word]))
            word_dict[word][0] = log_freq*idf
            
        return word_dict

    def index(self):
        """Creates index for words using tf-idf scores as well as positional info"""
        info_holder = {}
        index = {}   

        #builds word dict for every file with first item in list as
        #tf-idf value and every item after is just positional values
        for file in self.valid_list:            
            info_holder[file] = self.tf_idf(self.tokenize(file))

        
        for file in info_holder.keys():
            for word in info_holder[file].keys():
                if word not in index.keys():
                    index[word] = {}
                    index[word][file] = info_holder[file][word]
                else:
                    index[word][file] = info_holder[file][word]
        
        return index


if __name__ == "__main__":
    y = Crawler()
    y.calc_df()

    with open ('index.pickle', 'wb') as x:
        pickle.dump(y.index(), x)
    x.close()
