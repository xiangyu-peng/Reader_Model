import sys
import json
import argparse
import os
import pickle
import spacy
from collections import defaultdict

class verbnet_gen(object):
    """
    This class is used to generate the triples.
    """

    def __init__(self, args):
        """
        :param args:
        """
        self.args = args

        # The mapping file has the verb and its matching class id's
        f = open(args.mapping_file, 'rb')
        self.verb_class_mapping = pickle.load(f)

        # Get Verbnet
        f = open(args.verbnet_json, 'r')
        f_c = json.load(f)
        self.verbnet = dict()
        for v in f_c['VerbNet']:
            self.verbnet[v["class_id"]] = v

        self.nlp = spacy.load("en_core_web_sm")
        #####Define the dict for saving KG triples
        self.relations = defaultdict(list)

        # Load the verbsets
        with open(args.verbsets_path) as f:
            self.verbsets = json.load(f)

    def find_syntax(self, verb):
        """
        This function is used to generate syntax of a verb
        :param verb:
        :return:
        """
        tokens = self.nlp(verb)
        syntax = None
        for token in tokens:
            verb = token.head
            # Find in Verbnet.
            class_id = self.verb_class_mapping[verb.lemma_][0]
            if class_id in self.verbnet:
                syntax = self.verbnet[class_id]['frames'][0]["description"]["primary"]
        return syntax

    def find_verb(self, verb):
        """
        Given any verb, find the coresponding verb
        :param verb: str
        :return: verb in verbnet
        """
        tokens = self.nlp(verb)
        for token in tokens:
            verb = token.head
            # Find in Verbnet.
            return self.verb_class_mapping[verb.lemma_][0]

    def find_next_verb(self, verb):
        """
        Given a verb in verbnet, and predict next verb name in verbnet with prob
        :return: a verb name in verbnet.
        """
        print(self.verbsets[verb])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ignore_affector", type=bool, default=False, help="Whether to include the entity/person who "
                                                                            "did the relation")
    parser.add_argument("--mapping_file",
                        type=str,
                        default='../data/mapping_v.p',
                        help="The path of mapping file for verbnet")
    parser.add_argument("--verbnet_json",
                        type=str,
                        default='../data/verbnet.json',
                        help="The path of json file for verbnet")
    parser.add_argument("--verbsets_path",
                        type=str,
                        default='../data/verbsets/scifi-processed-stories-verb-pairs-percentages.json',
                        help="The path of json file for verb sets")

    working_dir = os.getcwd()
    parser.add_argument("--target_path", type=str, default=working_dir + "/KG", help="path to save kg visualization")
    args = parser.parse_args()

    verbnet_gen = verbnet_gen(args)
    verbnet_gen.find_next_verb(verbnet_gen.find_verb('mock'))
