#!/usr/bin/env python

from optparse import OptionParser
import os, logging
import utils_noun
import collections
import re

def create_model(sentences):
	word_count=collections.defaultdict(lambda : collections.defaultdict(int))  #house noun 10
	pos_count= collections.defaultdict(int) # total nouns: 2k
	word_frequency=collections.defaultdict(int)
	pos_seqcount=collections.defaultdict(lambda : collections.defaultdict(int))
	pos_seqcount_trigram=collections.defaultdict(lambda : collections.defaultdict(lambda : collections.defaultdict(int)))
	word_posprob=collections.defaultdict(lambda : collections.defaultdict(int))
	wordtypes=[]
	tagset=[]
	total_words=0
	single_words=0
	prob_single_word=0.0
	model=[pos_seqcount_trigram,word_posprob,tagset,word_count]
	for sentence in sentences:
		#print(sentence)
		pos_count["<s>"]+=1
		#print(sentence)
		pos_seqcount["<s>"]["<s>"] += 1
		pos_seqcount["<s>"][sentence[0].tag] += 1
		pos_seqcount_trigram["<s>"]["<s>"][sentence[0].tag] += 1
		pos_seqcount_trigram["<s>"][sentence[0].tag][sentence[1].tag] += 1
		#print(pos_seqcount["<s>"][sentence[0].tag])
		for token in sentence:
			token.word=token.word.lower()
			total_words+=1
			word_frequency[token.word]+=1
			#print(token.word,word_frequency[token.word] )
			word_count[token.word][token.tag]+=1
			if(token.word not in wordtypes):
				wordtypes.append(token.word)
			if(token.tag not in tagset):
				tagset.append(token.tag)
			pos_count[token.tag] += 1
		for index in range(0,len(sentence)-1):
			pos_seqcount[sentence[index].tag][sentence[index + 1].tag] += 1
		for index in range(0,len(sentence)-2):
			pos_seqcount_trigram[sentence[index].tag][sentence[index + 1].tag][sentence[index + 2].tag] += 1
	dummy_tagset=["<s>"] + tagset
	#print("Tagset:")
	#print(tagset)
	'''print("Dummytagset")
	print(dummy_tagset)'''
	'''print("wordtypes:")
	print(wordtypes)
	print("Prior prob values:")'''
	#print(len(tagset))
	#for tag1 in dummy_tagset:
	#	for tag2 in tagset:
	#		'''if(pos_seqcount[tag1][tag2]== 0):
	#			pos_seqcount[tag1][tag2] = float(1.0/(pos_count[tag1] + len(tagset)))
	#		else:'''
	#		pos_seqcount[tag1][tag2] = float((pos_seqcount[tag1][tag2] + 1.0)/(pos_count[tag1] + len(tagset)))
			#print(tag1,tag2,pos_seqcount[tag1][tag2])
	#print(len(tagset))
	#print("printing trigrams")
	for tag3 in tagset:
		#print(tag3,pos_seqcount_trigram["<s>"]["<s>"][tag3])
		pos_seqcount_trigram["<s>"]["<s>"][tag3]= float((pos_seqcount_trigram["<s>"]["<s>"][tag3] + 1.0)/(pos_seqcount["<s>"]["<s>"] + (len(tagset) * len(tagset))))
		#print(tag3,pos_seqcount_trigram["<s>"]["<s>"][tag3])
	for tag1 in dummy_tagset:
		for tag2 in tagset:
			for tag3 in tagset:
				#print("count:")
				#print(tag1,tag2,tag3,pos_seqcount_trigram[tag1][tag2][tag3])
				#print("value:")
				pos_seqcount_trigram[tag1][tag2][tag3]= float((pos_seqcount_trigram[tag1][tag2][tag3] + 1.0)/(pos_seqcount[tag1][tag2] + (len(tagset) * len(tagset))))
				#print(tag1,tag2,tag3,pos_seqcount_trigram[tag1][tag2][tag3])
	
	#print("2nd matrix")
	for tag in tagset:
		for word in wordtypes:
			word_posprob[tag][word]= float((word_count[word][tag] + 0.0) / pos_count[tag])
			#print(tag,word,word_posprob[tag][word])
	for wrd,freq in word_frequency.iteritems():
		#print(word,freq)
		if(freq==1):
			#print("incrementing by 1")
			single_words+=1
	
	prob_single_word= float(single_words)/total_words
	#print(prob_single_word)
			
	'''word_count=collections.defaultdict(lambda : collections.defaultdict(int))
	for sentence in sentences:
		for token in sentence:
			word_count[token.word][token.tag]+=1
	
	for word,tag_set in word_count.iteritems():
		count=0
		for pos,freq in tag_set.iteritems():
			if(count < freq):
				#print("updated")
				count=freq
				pos_name=pos
		model[word]=pos_name'''
	#print model
	model.append(prob_single_word)
	return model
    ## YOUR CODE GOES HERE: create a model

    #return model
def getPossbileTagForUnknownWord(word):
    if(word == "."):
        tag="."
    elif(re.compile("^[a-z]+ly$").match(word)):
        tag="RB"
    elif(re.compile("^[a-z]+ing$").match(word)):
        tag="VBG"
    elif(re.compile("^[a-z]+ed$").match(word)):
        tag="VBD"
    elif(re.compile("^[A-Z]+[a-z]*can$").match(word)):
        tag = "JJ"
    elif(re.compile("^[A-Z]+[a-z]*$").match(word)):
        tag = "NNP"
    elif(re.compile("^[A-Z]+[a-z]*s$").match(word)):
        tag = "NNPS"
    elif(re.compile("^[0-9]+$").match(word)):
        tag="CD"                                
    elif(re.compile("^[a-z]+ize$").match(word)):
        tag="VB"                                
    elif(re.compile("^[a-z]+ized").match(word)):
        tag="VBD"                                
    elif(re.compile("^[a-z]+al$").match(word)):
        tag="JJ"
    elif(re.compile("^[a-z]+-[a-z]+$").match(word)):
        tag="JJ"                                                                         
    elif(re.compile("^[a-z]+al$").match(word)):
        tag="JJ"
    elif(re.compile("^[a-z]+es$").match(word)):
        tag="NNS"
    elif(re.compile("^[a-z]+ation$").match(word)):
        tag="NN"
    elif(re.compile("^[a-z]+ations$").match(word)):
        tag="NNS"
    elif(re.compile("a|an|the").match(word)):
        tag="DT"
    else:
       if(str.startswith(word,"un")):
          newWord = re.sub("un","",word)
          subTag = getPossbileTagForUnknownWord(newWord)
          if(subTag == "VB"):
             tag="VB"
          else:
              tag="JJ"
       elif(str.startswith(word,"dis")):
          newWord = re.sub("dis","",word)
          subTag = getPossbileTagForUnknownWord(newWord)
          if(subTag == "VB"):
             tag="VB"
          else:
              tag="JJ"
       else:
           if(str.endswith(word,"s")):
                tag="NNS"
           else:
                tag="NN"
    return tag
def predict_tags(sentences, model):
    ## YOU CODE GOES HERE: use the model to predict tags for sentences
	prior=model[0]
	likehood= model[1]
	tagset=model[2]
	#print(tagset)
	word_count=model[3]
	goodtuned_value=model[4]
	#print(goodtuned_value)
	counts=0
	#viterbi2=[]
	for sentence in sentences:
		#print(counts)
		#counts+=1
		viterbi=[[0.0 for j in range(len(tagset))] for i in range(len(sentence))]
		viterbi2=[[[0 for z in range(2)] for j in range(len(tagset))] for i in range(len(sentence))]
		for word_index in range(0,len(sentence)):
			for tag_index in range(0,len(tagset)):
				sentence[word_index].word=sentence[word_index].word.lower()
				prsnt_tag= tagset[tag_index]
				#print("Present tag")
				#print(prsnt_tag)
				prsnt_word= sentence[word_index].word
				if(word_count.has_key(sentence[word_index].word)):
					likehood_value= likehood[prsnt_tag][prsnt_word]
				else:
					
					new_tag=getPossbileTagForUnknownWord(prsnt_word)
					#if(new_tag=="VB" or new_tag=="VBD" or new_tag=="VBG" or new_tag=="VBN" or new_tag=="VBP" or new_tag=="VBZ"):
					#	new_tag="VERB"
					if(new_tag == "NN" or new_tag=="NNS" or new_tag=="NNP" or new_tag=="NNPS"):
						new_tag="NOUN"
					if(prsnt_tag == new_tag):
						likehood_value=goodtuned_value
						#likehood_value=0.00000000001
						#likehood_value=0.0
					else:
						likehood_value=0.0
					#likehood_value=0.00000000001
					
				
				tag_winner="<s>"
				if(word_index == 0):
					viterbi[word_index][tag_index]= (likehood_value)*(prior["<s>"]["<s>"][prsnt_tag])
					#viterbi[word_index][tag_index]= (likehood[prsnt_tag][prsnt_word])*(prior["<s>"][prsnt_tag])
					viterbi2[word_index][tag_index]=[0,0]
				else:
					viterbi[word_index][tag_index]=0
					
					#print(word_index,prsnt_tag,tag_winner)
					for index_before in range(0,len(tagset)):
						tag_winner_loc=viterbi2[word_index -1][index_before]
						tag_winner=tagset[tag_winner_loc[1]]
						if(word_index == 1):
							current_viterbi_value = (viterbi[word_index - 1][index_before]) * ( likehood_value) * (prior["<s>"][tagset[index_before]][prsnt_tag])
						else:
							current_viterbi_value = (viterbi[word_index - 1][index_before]) * ( likehood_value) * (prior[tag_winner][tagset[index_before]][prsnt_tag])
						#current_viterbi_value = (viterbi[word_index - 1][index_before]) * ( likehood[prsnt_tag][prsnt_word]) * (prior[tagset[index_before]][prsnt_tag])
						if(current_viterbi_value >= viterbi[word_index][tag_index]):
							viterbi[word_index][tag_index]=current_viterbi_value
							prior_word_index=word_index - 1
							prior_tag_index= index_before
							viterbi2[word_index][tag_index]= [prior_word_index,prior_tag_index]
		max=0
		for last_tagindex in range(0,len(tagset)):
			if(viterbi[len(sentence)-1][last_tagindex] >= max):
				max=viterbi[len(sentence)-1][last_tagindex]
				last_tag=tagset[last_tagindex]
				temp=viterbi2[len(sentence)-1][last_tagindex]
		sentence[len(sentence)-1].tag=last_tag
		for token_index in range(len(sentence)-2,-1,-1):
			sentence[token_index].tag= tagset[temp[1]]
			temp= viterbi2[temp[0]][temp[1]]
		'''for row in range(len(sentence)):
			for column in range(len(tagset)):
				print(viterbi[row][column])'''
		'''for token in sentence:
			print(token.word,token.tag)'''
	return sentences

if __name__ == "__main__":
    usage = "usage: %prog [options] GOLD TEST"
    parser = OptionParser(usage=usage)

    parser.add_option("-d", "--debug", action="store_true",
                      help="turn on debug mode")

    (options, args) = parser.parse_args()
    if len(args) != 2:
        parser.error("Please provide required arguments")

    if options.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.CRITICAL)
	training_file = args[0]
	training_sents = utils_noun.read_tokens(training_file)
	test_file = args[1]
	test_sents = utils_noun.read_tokens(test_file)
	#create_model(training_sents)

    model = create_model(training_sents)
	

    ## read sentences again because predict_tags(...) rewrites the tags
    sents = utils_noun.read_tokens(training_file)
    predictions = predict_tags(sents, model)
    accuracy = utils_noun.calc_accuracy(training_sents, predictions)
    print "Accuracy in training [%s sentences]: %s" % (len(sents), accuracy)

    ## read sentences again because predict_tags(...) rewrites the tags
    sents = utils_noun.read_tokens(test_file)
    predictions = predict_tags(sents, model)
    accuracy = utils_noun.calc_accuracy(test_sents, predictions)
    print "Accuracy in training [%s sentences]: %s" % (len(sents), accuracy)
