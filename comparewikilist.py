import wikionly #script name is wikionly (no summary), class name is wiki
import re as re
import nltk
# nltk.download('wordnet')
from nltk.corpus import wordnet
import math

#Input two Wikipedia articles to compute similarity percentage
class similar:
    def __init__(self,text1,text2,verbose=1):
        """To start, assign var = comparewiki.similar('arg1','arg2', verbose=1). 
        arg1 and arg2 are names of the wikipedia articles.
        verbose=1 prints the probability score and mathematical calculation. 
        verbose=2 additionally prints array of words for each article
        verbose=0 disables any logs.
        To get values in a list for storage, use .ans(). 
        To iterate through arg1 with a list of articles in arg2, use .iterate()
        To get the 40 common words for comparison, use .words()"""

        self.wn = nltk.corpus.wordnet #the corpus reader
        self.verbose = verbose # Verbose/log level of detail

        #Error handling: check if both arguments input are string format
        checkstr = False
        if isinstance(text1, str) == True:
            if isinstance(text2, list) == True:
                self.text1 = text1
                self.text2 = text2
                checkstr = True
            else:
                print('Error! The second argument is not a list format!')
                return
        else:
            print('Error! The first argument is not a string format!')
            return
        
        #Run internal wikipedia python file for processing for both wiki titles
        if checkstr == True:
            self.wiki1 = wikionly.wiki(text1)
            # Get list of 40 common words for first article (string)
            # This means that article won't have to keep calling commonwords function in percent and iterate function
            self.dotn01 = ('.','n','.','0','1')
            self.wiki1list = []
            for key in self.wiki1.commonwords(40):
                self.wiki1slice = list(key)
                for letter in self.dotn01:
                    self.wiki1slice.append(letter)
                self.wiki1slice = ''.join(self.wiki1slice)
                self.wiki1list.append(self.wiki1slice)
            
    def iterate(self):
        # List that stores list of results
        self.matrix = []
        # Index for getting write article from self.text2 list
        self.index = 0
        
        for article in self.text2:
            self.wiki2 = wikionly.wiki(article)
            # Call function that calculates percentage
            pct = self.percent(self.wiki1,self.wiki2,self.verbose)
            # Call the function that shows list of words for both Wiki sites
            # Only can be used if self.percent has been called and list/arrays for articles are created
            if self.verbose == 2:
                print(self.words())
            self.matrix.append(self.ans(self.index))
            self.index += 1
            
        return self.matrix
                
    
    #Retrieve top 40 common words from wiki page, slice up and append .n01 for NLTK usage
    def percent(self,input1,input2,verbose):

        self.wiki2list = []
        for key in self.wiki2.commonwords(40):
            self.wiki2slice = list(key)
            for letter in self.dotn01:
                self.wiki2slice.append(letter)
            self.wiki2slice = ''.join(self.wiki2slice)
            self.wiki2list.append(self.wiki2slice)
        
        #count and sum for calculating similarity
        self.count = 0
        self.sum = 0
        #A count for the ranking of the word (how often it appears in both wiki passages)
        self.topten1 = 0
        self.topten2 = 0

        #For words that are 1-10th and 11-21st in popularity, if both wiki pages have the word, they get more points
        for word1 in self.wiki1list:
            #Reset self.topten2
            self.topten2 = 0
            self.topten1 += 1
            for word2 in self.wiki2list:
                self.topten2 += 1
                #reinitialize to zero to prevent old sums from going into maxsum
                self.sum1 = 0
                self.sum2 = 0
                self.sum3 = 0
                self.sum4 = 0
                self.maxsum = 0
                
                if self.topten1 < 11 and self.topten2 < 11:
                    self.expvalue = 4.5
                elif self.topten1 < 21 and self.topten2 < 21:
                    self.expvalue = 2.5
                else:
                    self.expvalue = 1.5
                
                #Main algorithm for calculating score of words
                try:
                    if re.findall(r"\d+.n.01", word1) == [] and re.findall(r"\d+.n.01", word2) == []: #check both words not numbers
                        #since words have many meanings, for every pair of words, use top two meanings n.01 and n.02 for comparison
                        #two for loops will check every permutation pair of words between wiki pages, two meanings for each word, 
                        #Take the max similarity value taken for computation of similarity index
                        #e.g. money.n.01 may have highest value with value.n.02 because value.n.01 has the obvious meaning of worth/significance and secondary for money
                        word11 = word1.replace('n.01','n.02')
                        word22 = word2.replace('n.01','n.02')
                        #print(word11,word22)
                        self.x = self.wn.synset(word1)
                        self.y = self.wn.synset(word2)
                        #get default similarity value of 1st definitions of word
                        self.sum1 = self.x.path_similarity(self.y) * math.exp(self.expvalue * self.x.path_similarity(self.y)) + 10 * math.log(0.885+self.x.path_similarity(self.y))
                        try: #get 2nd definitions of words and their similarity values, if it exist
                            self.xx = self.wn.synset(word11)
                            self.yy = self.wn.synset(word22)
                            self.sum2 = self.xx.path_similarity(self.y) * math.exp(self.expvalue * self.xx.path_similarity(self.y)) + 10 * math.log(0.89+self.xx.path_similarity(self.y))
                            self.sum3 = self.x.path_similarity(self.yy) * math.exp(self.expvalue * self.x.path_similarity(self.yy)) + 10 * math.log(0.89+self.x.path_similarity(self.yy))
                            self.sum4 = self.xx.path_similarity(self.yy) * math.exp(self.expvalue * self.xx.path_similarity(self.yy)) + 10 * math.log(0.89+self.xx.path_similarity(self.yy))
                        except:
                            continue
                        self.maxsum = max(self.sum1,self.sum2,self.sum3,self.sum4) #get the max similarity value between 2 words x 2 meanings = 4 comparisons
                        #print(word1, word2, self.maxsum)
                        self.sum += self.maxsum
                        self.count += 1
                except:
                    if word1 == word2 and re.findall(r"\d+.n.01", word1) == []: #remove years/numbers being counted as match yyyy.n.01
                        self.sum += math.exp(self.expvalue) + 10 * math.log(1.89)
                        self.count += 1
                    else:
                        continue

        #Print the results and implement ceiling if the percent exceeds 100% or drops below 0%
        if self.count != 0:
            self.pct = round(self.sum/self.count*100)
            if self.pct > 100:
                self.pct = 100
            elif self.pct < 0:
                self.pct = 0
            if self.verbose >= 1:
                print('Probability of topics being related is ' + str(self.pct) + '%')
                print('Count is ' + str(self.count) + ' and sum is ' + str(self.sum))
        else:
            if self.verbose >= 1:
                print('No relation index can be calculated as words are all foreign')
            
        return self.pct
        
    #Print out list of common words for both Wiki articles
    def words(self):
        print(self.wiki1list)
        print('\n')
        print(self.wiki2list)
        
    #Outputs list of results [Article 1, Article 2, Percentage, Yes/No] that can be put into a dataframe
    def ans(self, index=0):
        self.listans = [self.text1,self.text2[index],self.pct]
        if self.pct > 49:
            self.listans.append('Yes')
        else:
            self.listans.append('No')
        
        if self.verbose == 2:
            self.listans.append(self.wiki1list)
            self.listans.append(self.wiki2list)
        
        return self.listans
    
    def help(self):
        print("To start, assign var = comparewiki.similar('arg1','arg2', verbose=1). arg1 and arg2 are names of the wikipedia articles, while verbose=1 prints the probability score and mathematical calculation. verbose=2 additionally prints array of words for each article, and verbose=0 disables any logs. To get values in a list for storage, use .ans(). To iterate through arg1 with a list of articles in arg2, use .iterate(). To get the 40 common words for comparison, use .words()")