import sys
import os
sys.path.append('../../genderComputer')
from allennlp_model_use import MC, Tagging, TE
from GPT_2_generate import GPT_2_gen
from verbatlas_kg import KG_RM, verbnet_gen, verbatlas_gen
# from genderComputer import GenderComputer
from nltk_gen import nltk_gen
from similarity import similarity_detect
from conceptNet import ConceptNet
from verb_generator import Verb_Generator
sys.path.append('../../KG_gen')
from KG_gen import COMET_gen
import random
from obj_generator import Object_Generator
# import logging
#
# logging.basicConfig(
#     format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
#     datefmt="%m/%d/%Y %H:%M:%S",
#     level=logging.CRITICAL,
# )
# logger = logging.getLogger(__name__)

class HiddenPrints:
    """
    To block outputs.
    """

    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

class Baseline_Trainer(object):
    """
    Reader Model Trainer.
    """
    def __init__(self, args, prompt_text=None):
        self.args = args
        self.GPT_2_gen = GPT_2_gen(self.args)
        self.prompt = prompt_text
        self.Tagging = Tagging(args.Tagging_model_id)

        self.roles = []  # roles names. EX. ['Jenny'] or ['Jenny', 'Linda']

        # read the file
        self.file_path = args.train_file_path
        self.save_path = args.save_path
        self.story_length = args.story_length

    def run_file(self):
        """
        Read the file and run forward.
        """
        f = open(self.file_path, "r")
        # if os.path.exists(file_name):
        #     f_save = open(self.save_path, "w")
        # else:
        f_save = open(self.save_path, "w")
        for sentence in f:
            self.clear()
            sentence = sentence.replace("\n", "")
            self.prompt = sentence
            self.define_roles()
            for idx in range(self.story_length - 1):
                self.forward()
            f_save.write(self.prompt + '\n')
            print('=' * 50)
            print('SAVE sentence is  ===> ', self.prompt)
            print('=' * 50)

        f_save.close()

    def clear(self):
        self.roles = []  # roles names. EX. ['Jenny'] or ['Jenny', 'Linda']
        self.prompt = ''

    def define_roles(self):
        """
        Find the sub., # of roles in this story and tense.
        """
        self.roles = self.Tagging.forward(self.prompt)
        print("Roles in this story are ===> " + ' and '.join(self.roles))

    def forward(self):
        """
        Pipeline - 1 KG-transition
        """
        # generate relation triples with verbatlas
        prompt_KG = self.prompt.split('.')[-2] + '.' + ' and '.join(self.roles)
        print('prompt in KG is ', prompt_KG)

        # genertate remaining sentence
        obj = self.GPT_2_gen.gen_multiple_obj(prompt_text=prompt_KG, num=1)[0]
        self.prompt += ' ' + ' and '.join(self.roles) + ' ' + obj.strip() + '.'

        print('The generate sentence is ===> ', self.prompt)
        print('='*50)

