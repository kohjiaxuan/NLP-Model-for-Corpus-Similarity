#!/usr/bin/env python
# coding: utf-8

# In[1]:


import requests #Get the HTML code
from bs4 import BeautifulSoup #Tidy up the code
from collections import Counter #Counter to count occurances of each word
import matplotlib.pyplot as plt #graph plotting
import re #regular expression to check if language setting is exactly 2 letters (for non common langs) in the argument
import os #for plotwords to tell where file is saved

class wiki:

    #the main features of cleaning the wiki site and whether the site is valid is run in __init__
    
    def __init__(self,title,option='No',lang='en',checknltk='No'):
        #print("Page is loading...\n")

        if isinstance(title, str) == True:
            if str(option).lower() == 'yes' or str(option) == '':
                self.title = str(title.title()) #title on LHS is variable input, RHS is function for proper case/title case
                #print("Search text formatted to title/proper case by default. Set second argument as 'No' to disable formatting")
            elif str(option).lower() == 'no':
                #print("Search text has preserved the cases of each letter. Set second argument as 'Yes' to format to title/proper case")
                self.title = title
            else:
                self.title = title.title() #title on LHS is variable input, RHS is function for proper case/title case
                print('Invalid option for preserving case of search text, title/proper case will be used by default')
            #As long as title is string, regardless of option the replacement for title for URL can be done
            self.title = str(self.title.replace(" ", "_")) #Convert spaces to _ as is Wikipedia page format
        else:
            print('Error encountered, search text (first argument) is not written as a string with quotes. Please try again')
        
        
        #Checking if you should use NLTK library, default set to False
        self.nltkrun = False #default is don't use it for stoplist
        if isinstance(checknltk, str): #check for string yes, no and other permutations
            if checknltk.lower().strip() in {'yes','true','y','t'}:
                import nltk
                nltk.download('stopwords')
                nltk.download('wordnet')
                from nltk.corpus import stopwords
                from nltk.corpus import wordnet
                self.nltkrun = True
            elif checknltk.lower().strip() in {'no','false','n','f','na','n/a','nan'}:
                self.nltkrun = False
            else:
                self.nltkrun = False
        elif isinstance(checknltk, bool): #check for boolean yes/no
            if checknltk == True:
                import nltk
                nltk.download('stopwords')
                nltk.download('wordnet')
                from nltk.corpus import stopwords
                from nltk.corpus import wordnet
                self.nltkrun = True
            else:
                self.nltkrun = False
        else: #run default if options are invalid - don't run nltk stoplist
            self.nltkrun = False

        #Default: Stopword list obtained from nltk
        self.nltkstopword = []
        
        #Detect language settings in third argument
        self.lang = 'en'  
        
        
        self.url = 'https://' + self.lang + '.wikipedia.org/wiki/' + self.title #combine the two to get full URL

        try: 
            self.page = requests.get(self.url) #retrieve HTML info from site
        except:
            self.lang = 'en'
            self.url = 'https://' + self.lang + '.wikipedia.org/wiki/' + self.title
            self.page = requests.get(self.url)
            print('Error with language settings, English used as default\n')

        self.contents = self.page.content 
        self.soup = BeautifulSoup(self.contents, 'html.parser') #Parse the HTML nicely with formatting
        
        
        self.trancetext = self.soup.find_all('p') #obtain all paragraphs starting with tag <p>
        self.trancetext2 = self.soup.find_all('li') #obtain all paragraphs starting with tag <li>

        #get paragraphs from trancetext with special format into a list
        self.para=[]
        for paragraph in self.trancetext: #append paragraphs starting with <p>
            self.para.append(paragraph)

        self.relatedtopic = ",*RELATED WIKI TOPIC*" #to add to points with a link and are on sidebar
        for paragraph in self.trancetext2: #append paragraphs starting with <li>
            if str(paragraph).find('<li><a href=') != -1:
                if str(paragraph).find('</a></li>') != -1 or str(paragraph).find('</a></sup></li>') != -1: 
                    self.para.append(self.relatedtopic)
            if str(paragraph).find('toctext') == -1: #remove Wiki headers 1.2.3 with toctext as they can't be arranged properly
                self.para.append(paragraph)
    
        #REASON WHY WE HAVE TO DO TWO FOR LOOPS WITH TWO TRANCETEXT IS BECAUSE THE FIND_ALL FOR ARRAY IS NOT IN ORDER
        #COMMENCE CLEANING OF NONSENSE HTML <> and WIKI LINK [no]
        
        #For FIXING the summary function
        self.troubleshoot = self.para
        
        self.para = list(str(self.para)) #chop everything into letters for cleaning
        
        #This block of code removes the first letter [, removes any words with <> html tag or [] citation
        #When it detects a <li> it will create two blanks
        self.start = 0 #is letter currently inside tag <>
        self.end = 0 #has <> just ended, need to check for , if it just ended to not copy a comma after <>
        self.first = 1 #first letter is [, need to omit
        self.bracket = 0 #check if letter is inside bracket
        self.li = 0 #check for <li> to line break
        self.p = 0 #check for <p> to line break
        self.point = 0 #after <li>, puts a • before adding new letter
        self.para2 = []
        for letter in self.para:
            if self.first == 0:
                if letter == '<': #tells python to stop reading letters inside a bracket
                    self.start = 1
                elif letter == '>': #next letter can be read since its out of bracket, unless its another <
                    self.start = 0
                    self.end = 1
                elif self.end == 1 and letter == ',': #skip COMMA reading when it occurs like </p>, at end of para
                    self.end = 0
                    continue
                elif letter == '[':
                    self.bracket = 1
                    self.end = 0
                elif letter == ']':
                    self.bracket = 0
                    self.end = 0
                elif self.start == 0 and self.bracket == 0: #ALL CLEAR TO READ LETTER
                    self.end = 0
                    if self.point == 1:
                        self.para2.append('• ')
                        self.point = 0
                    self.para2.append(letter)
            if letter == '<':
                self.li = 1
            elif letter != 'l' and self.li == 1:
                self.li = 0
            elif letter == 'l' and self.li == 1:
                self.li = 2
            elif letter == 'i' and self.li == 2:
                self.li = 3
            elif letter != '>' and self.li == 3:
                self.li = 0
            elif letter == '>' and self.li == 3:
                self.para2.append('\n\n')
                self.li = 0
                self.point = 1
            if letter == '<':
                self.p = 1
            elif letter != 'p' and self.p == 1:
                self.p = 0
            elif letter == 'p' and self.p == 1:
                self.p = 2
            elif letter == '>' and self.p == 2:
                self.para2.append('\n')
                self.p = 0
            self.first = 0 #Had an issue with the first letter being [, after skipping this, the [number] checks can run

        self.para2=''.join(self.para2) #combine back all letters and spaces
        #REMOVE UNWANTED ARRAYS
        self.para1 = []
        
        
        #WORD COUNT (SELF.PARA3) AND COMMON WORDS (SELF.TRANCECOUNTER)
        
        self.para3 = self.para2.split() #split paragraphs into words again for counting
        self.niceword = ''
        self.punctuation = ('.',',','(',')','"',"'",'?','!','*','|',':',';')
        for index, word in enumerate(self.para3):
            self.niceword = word
            self.niceword = self.niceword.lower() #standardize all to lower case before counting
            for punctuation in self.punctuation:
                self.niceword = self.niceword.replace(punctuation,'') #clean up bad punctuation
            self.para3[index] = self.niceword 
        self.trancecounter = Counter(self.para3) 
        #counter solely used for word count, cannot be used as banlist not implemented yet. 
        #Make new trancecounter2+banlist for use
        self.allwords = dict(self.trancecounter.most_common()) 
        #convert to dictionary so that for loop can extract words + do unique word count + total word count
        
        self.trancelist = [] #full list of words to fill up, cannot be used yet as banlist not implemented
        
        #FIND OUT UNIQUE WORD COUNT AND TOTAL WORD COUNT BEFORE BANLIST
        self.fullcount = 0
        self.fullwords = 0
        for key in self.allwords:
            self.fullcount += self.allwords[key]
            self.fullwords += 1
            self.trancelist.append(key)
        
        #IMPLEMENT BAN LIST (FROM WIKIPEDIA) BY DEL FUNCTION FOR COUNTER TRANCECOUNT AND WORD LIST SELF.TRANCELIST
        #BAN YEARS AND NUMBERS
        banlist = ('the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'I', 'it', 'for', 'not', 'on', 'with', 
                   'he', 'as', 'you', 'do', 'at', 'this', 'but', 'his', 'by', 'from', 'they', 'we', 'say', 'her', 'she', 
                   'or', 'an', 'will', 'my', 'one', 'all', 'would', 'there', 'their', 'what', 'so', 'up', 'out', 'if', 
                   'about', 'who', 'get', 'which', 'go', 'me', 'when', 'make', 'can', 'like', 'time', 'no', 'just', 
                   'him', 'know', 'take', 'people', 'into', 'year', 'your', 'good', 'some', 'could', 'them', 'see', 
                   'other', 'than', 'then', 'now', 'look', 'only', 'come', 'its', 'over', 'think', 'also', 'back', 
                   'after', 'use', 'two', 'how', 'our', 'work', 'first', 'well', 'way', 'even', 'new', 'want','topic', 
                   'because', 'any', 'these', 'give', 'day', 'most', 'us','retrieved','^','archived',"•",'related',
                   "',*related","wiki","topic*',","is","are",'was','since','such','articles','has','&amp;','&amp',
                   'p','b','january','february','march','april','may','june','july','august','september','october',
                   'november','december','2019','2018','2017','2016','2015','2014','2013','2012','2011','2010','2009',
                   '&amp','1','2','3','4','5','2008','2007','2006','2005','2004','2003','2002','2001','2000',
                  '6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26',
                  '27','28','29','30','original','isbn','wikipedia')
        
        #English banlist from top 100 common words and some extra terms
        if self.lang == 'en':
            for word in banlist: #delete words in counter and list only if it's in english
                del self.trancecounter[word]
                self.allwords.pop(word, None)
            
        #If NLTK stoplist is used, delete words from their stoplist as well
        if self.nltkrun == True:
            for word in self.nltkstopword: #delete words in counter and list
                del self.trancecounter[word]
                self.allwords.pop(word, None)
            
        #DELETE '',.,·,•,↑,space,null,related,wiki,common from counter and dictionary
        del self.trancecounter["''"]
        self.allwords.pop("''", None)
        del self.trancecounter["."]
        self.allwords.pop(".", None)
        del self.trancecounter["·"]
        self.allwords.pop("·", None)
        del self.trancecounter["•"]
        self.allwords.pop("•", None)
        del self.trancecounter["↑"]
        self.allwords.pop("↑", None)
        del self.trancecounter[" "]
        self.allwords.pop(" ", None)
        del self.trancecounter[""]
        self.allwords.pop("", None)
        del self.trancecounter["-"]
        self.allwords.pop("-", None)
        del self.trancecounter["–"]
        self.allwords.pop("–", None)
        del self.trancecounter["related"]
        self.allwords.pop("related", None)
        del self.trancecounter["wiki"]
        self.allwords.pop("wiki", None)
        del self.trancecounter["common"]
        self.allwords.pop("common", None)
        
        #FIND OUT TOTAL WORD COUNT AFTER BANLIST FOR FUNCTION COMMONWORDPCT THAT USES PERCENTAGE OF THRESHOLD FOR WORD COUNT
        self.fullcount2 = 0
        self.fullwords2 = 0
        
        self.trancelist2 = [] #NOT USED!!
        
        for key in self.allwords: #SELF.ALLWORDS HAVE WORDS REMOVED ON ITSELF VIA BANLIST, DONT NEED NEW VARIABLE
            self.fullcount2 += self.allwords[key] #FULLCOUNT2 used for COMMONWORDPCT and TOTAL WORDS function to show total word count aft removing banlist
            self.fullwords2 += 1 #FULLWORDS2 used in TOTAL WORDS function to show unique word count aft removing banlist
            self.trancelist2.append(key) #NOT USED!!!
    
    
        #This section checks if the Wiki site was loaded successfully..
        self.missing = self.soup.find_all('b') 
        #Wikipedia does not have an article with this exact name.
        #This sentence that always appears for Error 404 pages, is bolded, so <b> tag can help to find it
                
        #Check for sentence that tells of Error 404 website using a counter.
        self.goodsite = 1
        self.offsite = 0
        
        for sentence in self.trancetext: #check if a site goes through but it is an ambiguous site (recommendations page)
            #refer to: phrase belongs in a <p> paragraph
            if str(sentence).find("refer to:") != -1:
                self.offsite = 1            
        
        for sentence in self.missing: #RUN THROUGH EVERY ELEMENT IN LIST
            if str(sentence) == "<b>Wikipedia does not have an article with this exact name.</b>": #CONVERT ELEMENT TO STRING TYPE BEFORE CHECK!!!
                self.goodsite = 0 #sentence exists, bad site means counter flips to 0
        
        if self.goodsite == 1 and self.offsite == 0:
            #print('Wikipedia page loaded successfully!! Type variablename.HELP() for documentation of functions.')
        
            #WHEN SEARCH IS SUCCESSFUL, PRINT 2 PARAGRAPHS FOR PREVIEW
            self.parashort = []
            self.noofpara = 0
            for paragraph in self.trancetext: #append ONLY 2 paragraphs starting with <p>
                if self.noofpara < 2 and str(paragraph) != '<p class="mw-empty-elt">\n</p>' and len(str(paragraph)) > 199: 
                    self.parashort.append(paragraph)
                    self.noofpara += 1

            self.parashort = list(str(self.parashort)) #chop everything into letters for usage
            self.start = 0 #is letter currently inside tag <>
            self.end = 0 #has <> just ended, need to check for , if it just ended to not copy a comma after <>
            self.first = 1 #first letter is [, need to omit
            self.bracket = 0 #check if letter is inside bracket
            self.li = 0 #check for <li> to line break
            self.p = 0 #check for <p> to line break
            self.parashort2 = []
            for letter in self.parashort:
                if self.first == 0:
                    if letter == '<': #tells python to stop reading letters inside a bracket
                        self.start = 1
                    elif letter == '>': #next letter can be read since its out of bracket, unless its another <
                        self.start = 0
                        self.end = 1
                    elif self.end == 1 and letter == ',': #skip COMMA reading when it occurs like </p>, at end of para
                        self.end = 0
                        continue
                    elif letter == '[':
                        self.bracket = 1
                        self.end = 0
                    elif letter == ']':
                        self.bracket = 0
                        self.end = 0
                    elif self.start == 0 and self.bracket == 0: #ALL CLEAR TO READ LETTER
                        self.end = 0
                        self.parashort2.append(letter)
                if letter == '<':
                    self.p = 1
                elif letter != 'p' and self.p == 1:
                    self.p = 0
                elif letter == 'p' and self.p == 1:
                    self.p = 2
                elif letter == '>' and self.p == 2:
                    self.parashort2.append('\n\n')
                    self.p = 0
                self.first = 0 #Had an issue with the first letter being [, after skipping this, the [number] checks can run

            self.parashort2=''.join(self.parashort2) #combine back all letters and spaces
            #REMOVE UNWANTED ARRAYS
            self.parashort = []        

            #END OF DATA PROCESSING FOR AUTOMATIC SUMMARY PRINTED WHEN SEARCHING

            #print(self.parashort2)
            #print two paragraphs when search is successful, disabled
            
        elif self.goodsite == 1 and self.offsite == 1:
            print('\nThe title "'+ self.title.replace("_", " ") + '" you specified is ambiguous. As a result, you are linked to a clarification page.\n\n')
            print('Here are some suggestions to use: \n')
            
            self.all_links = self.soup.find_all("a") #ALL HTML TAGS STARTING WITH <A, E.G. <A HREF, <A TITLE AND FULL PARAGRAPH
            self.wiktwords = []
            for link in self.all_links:
                self.wiktwords.append(link.get("title")) #TAG STARTING WITH A, CONTENT ENCLOSED INSIDE TITLE=""
                #print(link.get("title")) #shows list of items appended, common words all start with wikt:

            self.cleanlink = []
            for words in self.wiktwords: 
                self.words2 = str(words) #words are not string yet so need str function before saving into new var
                self.cleanlink.append(self.words2)
#                 if self.words2.find("wiktionary:") != -1:
#                     self.tempsliceword = list(str(self.words2))
#                     self.newword = []
#                     self.x = 0
#                     for letter in self.tempsliceword:
#                         if self.x > 10:
#                             self.newword.append(letter)
#                         self.x += 1
#                     self.cleanword = ''.join(self.newword)
#                     self.cleanlink.append(self.cleanword)
#                 elif self.words2 != 'None':
#                     self.cleanlink.append(self.words2)

            for link in self.cleanlink:
                if link.find("Help:") != -1:
                    break
                elif link.find("Edit section:") != -1:
                    continue
                else:
                    print(link)
            
        else:
            print('Wikipedia page could not be found for "' + str(self.title.replace("_", " ")) + '". Please try again!') 
            print('Other useful information: Enclose title argument with single quotes. Spaces are allowed, and title is case insensitive.')
        
    
    def gettext(self,outfull='No'): #comes after init
        if isinstance(outfull, str): #check for string yes, no and other permutations
            if outfull.lower().strip() in {'yes','true','y','t'}:
                print("Full text is output. To print it instead, put 'no' in argument")
                return self.para2
            elif outfull.lower().strip() in {'no','false','n','f','na','n/a','nan'}:
                print("Full text is printed. To set as output, put 'yes' in argument")
                print(self.para2)
            else:
                print("Full text is invalid, summary is printed by default. To set as output, put 'yes' in argument")
                print(self.para2)
        elif isinstance(outfull, bool): #check for boolean yes/no
            if outfull == True:
                print("Full text is output. To print it instead, put False or 'no' in argument")
                return self.para2
            else:
                print("Full text is printed. To set as output, put True or 'yes' in argument")
                print(self.para2)   
        else: #run default - print if invalid option
            print("2nd argument not a string, full text is printed by default. To set as output, put 'yes' in argument")
            print(self.para2)

        
    def commonwords(self,wordcount=100):
        self.wordcount = 100
        if wordcount != 100 and isinstance(wordcount, int) == True:
            self.wordcount = wordcount
        elif wordcount != 100 and isinstance(wordcount, int) == False:
            print('Word count specified is currently not an integer. Hence default of 100 words is used\n')
        #convert counter to list to dictionary then sum up total word count using for loop in word[key]
        self.topwords = dict(self.trancecounter.most_common(self.wordcount))
        return self.topwords

    def plotwords(self,wordcount2=20):
        self.wordcount2 = 20
        if wordcount2 != 20 and isinstance(wordcount2, int) == True:
            self.wordcount2 = wordcount2
        elif wordcount2 != 20 and isinstance(wordcount2, int) == False:
            print('Word count specified is currently not an integer. Hence default of 20 words is used for graph\n')
        #convert counter to list to dictionary then sum up total word count using for loop in word[key]
        self.topwords2 = dict(self.trancecounter.most_common(self.wordcount2))
        
        #matplotlib.pyplot.bar(range, height, tick_label)
        self.wordnames = list(self.topwords2.keys())
        self.wordvalues = list(self.topwords2.values())

        #tick_label does the some work as plt.xticks()
        plt.figure(figsize=(22, 12), dpi= 80, facecolor='w', edgecolor='k')
        plt.bar(range(len(self.topwords2)),self.wordvalues,tick_label=self.wordnames)
        plt.savefig('wordcount.png')
        plt.rcParams['figure.figsize'] = [22, 12]
        plt.show()
        
        self.cwd = os.getcwd()
        return 'Graph is saved as wordcount.png in directory: ' + str(self.cwd)


# In[ ]:



