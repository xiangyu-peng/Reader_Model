import operator
#import data_utils_miniSent_arg5 as data_utils
import numpy as np
import random
import jenkspy
import math
from collections import defaultdict
from nltk_gen import nltk_gen
import pickle
#reward words that occur in the three lines before marry
import sys
sys.path.append('../../semparse-core/')
from java_python import Verbnet_Parser

import os.path


nltk_gen_verb = nltk_gen()
verbnet_mapping = pickle.load(open("../data/mapping_v.p", "rb"))
# verbnet_parser = Verbnet_Parser()

def reward_word_statistics(input_file, train_set, inv_vocab, target_verb="admire-31.2"):
    print("reward_word_statistics function:",len(input_file),len(train_set[0]))
    reward_word_dict = defaultdict(int)
    i=0
    with open(input_file,'r') as input:
        for line in input:
            if "start_of_story" in line or "end_of_story" in line:
                i+=1
                continue
            words=line.strip().split(' ')
            flag, check_val = check_successor_lines(i,train_set,inv_vocab)
            if flag:
                reward_word_dict[words[1]] += check_val
            i+=1
    #print("reward_word_statistics",reward_word_dict)
    return reward_word_dict



#number of times each verb occurs before admire
def reward_word_statistics2(input_file,reward_word = 'admire-31.2'):
    reward_word_dict = defaultdict(int)
    story_verbs = []
    i=0
    with open(input_file,'r') as input:
        for line in input:
            # print('line', line)
            # break
            if "START" in line:
                continue
            if ".." in line:
                story_verbs = []
                continue
            else:
                words = line.strip().split(' ')
                print(words)
                if words[1] == reward_word:
                    for verb in story_verbs:
                        reward_word_dict[verb] += 1
                    story_verbs = []
                    continue
                else:
                    story_verbs.append(words[1])
    #print("reward_word_statistics2",reward_word_dict)
    return reward_word_dict

def word_distance_old(input_file,verbs,reward_word = 'admire-31.2'):
    reward_word_dict = defaultdict(int)        #total distance from admire for all verbs
    story_frequency = defaultdict(int)        #number of stories which ends in admire in which a verb occurs
    num_stories = 0
    word_num_stories = defaultdict(int)    #total number of occurances of all words
    final_reward_word_dict = defaultdict(int)
    story_verbs = []
    i=0
    with open(input_file,'r') as input:
        for line in input:
            if "start_of_story" in line:
                continue
            if "end_of_story" in line:
                story_verbs = []
                num_stories +=1
                continue
            else:
                words=line.strip().split(' ')
                if words == ["%%%%%%%%%%%%%%%%%%%%%%%%%%%%"]:
                    continue

                if words[1] == reward_word:            #If you need to change the target position, do that here.
                    i = len(story_verbs)
                    for verb in story_verbs:
                        if verb in verbs:
                            # reward_word_dict[verb].append(i)
                            reward_word_dict[verb] += i
                            story_frequency[verb] += 1
                        i-=1
                    story_verbs = []
                    num_stories+=1
                    continue
                else:
                    word_num_stories[words[1]]+=1
                    story_verbs.append(words[1])
    for key in reward_word_dict.keys():
        final_reward_word_dict[key] = math.log(reward_word_dict[key])
    for key in story_frequency.keys():
        story_frequency[key] = math.log(float(story_frequency[key])/word_num_stories[key])
    min_r = -1000000
    for key in reward_word_dict.keys():
        if (story_frequency[key]<=0):
            story_frequency[key] = 0.000001
        if (final_reward_word_dict[key]<=0):
            final_reward_word_dict[key] = 0.000001
        final_reward_word_dict[key] = float(final_reward_word_dict[key])*story_frequency[key]

    #print("word_distance",final_reward_word_dict)
    return final_reward_word_dict

def word_distance_all(input_file, verbs, reward_word='admire-31.2', location=None):
    if location and os.path.isfile(location):
        return pickle.load(open(location, "rb"))
    reward_word_dict = defaultdict(int)        #total distance from admire for all verbs
    story_frequency = defaultdict(int)        #number of stories which ends in admire in which a verb occurs
    word_num_stories = defaultdict(int)    #total number of occurances of all words
    final_reward_word_dict = defaultdict(int)
    story_verbs = []
    i = 0
    with open(input_file,'r') as input:
        for line in input:

            if "START" in line:
                continue
            if ".." in line:
                story_verbs = []  # Compute from the beginning for each story.
                continue
            else:
                verb = nltk_gen_verb.find_verb_lemma(line)
                if verb in verbnet_mapping:
                    verbclass = verbnet_mapping[verb]
                    # parsed_predicates = verbnet_parser.parse_in_batch([line])
                    # verbclass = verbnet_parser.find_verbnet_id(parsed_predicates)
                    for verb_c in verbclass:
                        if verb_c == reward_word:
                            # print(story_verbs)
                            i = len(story_verbs)
                            for verb_s in story_verbs:
                                # if verb in verbs:
                                # reward_word_dict[verb].append(i)
                                reward_word_dict[verb_s] += i
                                # if verb == "discover-84":
                                #    print(reward_word_dict[verb], " ", i)
                                story_frequency[verb_s] += 1
                                i -= 1
                            story_verbs = []
                            continue
                        else:
                            word_num_stories[verb_c] += 1
                            story_verbs.append(verb_c)

    for key in reward_word_dict.keys():
        final_reward_word_dict[key] = math.log(reward_word_dict[key])
    for key in story_frequency.keys():
        story_frequency[key] = math.log(float(story_frequency[key])/word_num_stories[key])
    min_r = -1000000
    for key in reward_word_dict.keys():
        if (story_frequency[key]<=0):
            story_frequency[key] = 0.000001
        if (final_reward_word_dict[key]<=0):
            final_reward_word_dict[key] = 0.000001
        final_reward_word_dict[key] = float(final_reward_word_dict[key])*story_frequency[key]

    #print("word_distance",final_reward_word_dict)
    pickle.dump(final_reward_word_dict, open(location, "wb"))
    return final_reward_word_dict

def word_distance(input_file, verb, reward_word='admire-31.2', location=None):
    final_reward_word_dict = word_distance_all(
                    input_file,
                    None,
                    reward_word,
                    location,
                )
    return float(final_reward_word_dict[verb])

def word_clustering(word_dict, num_classes = 6,reward_word='admire-31.2'):
    print("reward word in dict ", reward_word in word_dict)
    means = jenkspy.jenks_breaks(list(word_dict.values()), nb_class=num_classes)    # Jenks natural breaks optimization
    print("jenks means ", means)
    clusters = defaultdict(list)
    for word in word_dict.keys():
        for i, mean in enumerate(np.sort(np.array(means))):
            if word_dict[word]<=mean:
                #print(word, word_dict[word])
                clusters[i].append(word)
                break
    for i, c in enumerate(clusters.keys()):
        if reward_word in clusters[c]:
            print("FOUND REWARD WORD", i)
    #print(clusters)
    return clusters


def getGVF( dataList, numClass ):
  """
  The Goodness of Variance Fit (GVF) is found by taking the
  difference between the squared deviations
  from the array mean (SDAM) and the squared deviations from the
  class means (SDCM), and dividing by the SDAM
  """
  breaks = jenkspy.jenks_breaks(dataList, nb_class=numClass)
  dataList.sort()
  listMean = sum(dataList)/len(dataList)
  SDAM = 0.0
  for i in range(0,len(dataList)):
    sqDev = (dataList[i] - listMean)**2
    SDAM += sqDev
  SDCM = 0.0
  for i in range(0,numClass):
    if breaks[i] == 0:
      classStart = 0
    else:
      classStart = dataList.index(breaks[i])
      classStart += 1
    classEnd = dataList.index(breaks[i+1])
    classList = dataList[classStart:classEnd+1]
    classMean = sum(classList)/len(classList)
    preSDCM = 0.0
    for j in range(0,len(classList)):
      sqDev2 = (classList[j] - classMean)**2
      preSDCM += sqDev2
    SDCM += preSDCM
  return (SDAM - SDCM)/SDAM

"""
def check_successor_lines(pos,train_set,inv_vocab,num_lines=5,target_verb="admire-31.2"):
  for i in range(num_lines):
      train_set_size = np.array(train_set).shape
      if train_set_size[1]<=pos+i+1:
          return False, 0
      next = train_set[0][pos+i+1][0][1]

      if inv_vocab[next] == target_verb:
          return True, num_lines-i+1
      if inv_vocab[next] == data_utils.START_ID or inv_vocab[next] == data_utils.END_ID:
          return False, 0
  return False, 0
"""

if __name__ == "__main__":
    goal_verb = 'admire-31.2'
    verb_dict = word_distance("/media/ASTER/Story-GPT2/scifi_data_all.txt", None, goal_verb, location='../data/' + goal_verb + ".p")

    # scores = pickle.load(open("../data/verb_dict.p", "rb"))
    # print(verbnet_mapping['die'])
    # print(scores['meet-36.3-1'])
    # print(scores['die-42.4-1'])
