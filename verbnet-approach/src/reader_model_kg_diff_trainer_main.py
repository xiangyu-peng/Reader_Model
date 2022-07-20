import sys
import os
import pickle
import gc  # clear cache

sys.path.append("../../genderComputer")
from allennlp_model_use import MC, Tagging, TE, CR
from GPT_2_generate import GPT_2_gen
from collections import defaultdict

# from genderComputer import GenderComputer
from nltk_gen import nltk_gen
from similarity import similarity_detect, similarity_score
from conceptNet import ConceptNet
from verb_generator import Verb_Generator
from compute_distance import word_distance
from stopwords_filter import Stopwords_Filter
import comet_new_usage
sys.path.append("../../KG_gen")
from KG_gen import COMET_gen
import random
from obj_generator import Object_Generator

sys.path.append("../../entailment")
from RocNLI import RocNLI

sys.path.append("../../fill_in_mask")
from inference import Roberta_Model

from verbatlas_kg import KG_RM, verbnet_gen, verbatlas_gen

sys.path.append("../../semparse-core")
from java_python import Verbnet_Parser

sys.path.append("../../comet-atomic-2020/models/comet_atomic2020_bart")
from generation_example import Comet

import copy
import time

def cal_time():
    global start_time
    process_time, start_time =  (time.time() - start_time) / 60, time.time()
    return process_time

def update_time():
    global start_time
    start_time = time.time()

start_time = time.time()

verbnet_mapping = pickle.load(open("../data/mapping_v.p", "rb"))

import logging
logger = logging.getLogger('logging_info')

class HiddenPrints:
    """
    To block outputs.
    """

    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


class Reader_Model_Trainer_KG_diff(object):
    """
    Reader Model Trainer.
    """

    def __init__(self, args, prompt_text=None):
        self.args = args
        self.MC = MC(args.MC_model_id)
        self.GPT_2_gen = GPT_2_gen(self.args)
        self.verbnet_gen = verbnet_gen(args)
        self.verbatlas_gen = verbatlas_gen(args)
        self.Tagging = Tagging(args.Tagging_model_id)
        self.nltk_gen = nltk_gen()
        self.TE = TE(args.TE_model_id)
        self.conceptNet = ConceptNet()
        self.verb_generator = Verb_Generator(self.args)
        self.object_generator = Object_Generator(self.args)
        self.comet = COMET_gen(args)
        self.Comet = Comet("../../comet-atomic-2020/models/comet_atomic2020_bart/comet-atomic_2020_BART")  # new atomic comet 2020
        self.Comet.model.zero_grad()
        self.rocnli = RocNLI(args.rocnli_path, args.device)
        self.roberta = Roberta_Model(args.roberta_path)
        self.tense = "present"  # verb tense in the prompt
        self.number = "singular"  # how many roles in the prompt
        self.cr = CR(args.CR_model_id)

        # read the file
        self.file_path = args.train_file_path
        self.save_path = args.save_path
        self.story_length = args.story_length

        # whether we use verb as action or the sentence.
        self.standard = args.action_standard

        self.topk = args.topk

        # stopwords
        self.Stopwords_Filter = Stopwords_Filter(args.stopwords_path)

        # verbnet
        self.Verbnet_Parser = Verbnet_Parser()

        # change!!!
        # goal
        self.goal_word = None
        self.goal_inference_set = defaultdict(list)

        # change all the time
        self.prompt = [prompt_text] * args.topk
        self.node_history_set = [set()] * args.topk  # nodes in all the sentences
        self.inference_set = [set()] * args.topk
        self.node_history = [set()] * args.topk  # nodes in one sentence
        self.premise_inferences = [set()] * args.topk  # a set to record all the inferences on the previous sentence.
        self.roles = set() # roles names. EX. ['Jenny'] or ['Jenny', 'Linda']
        # whether to turn on KG?
        if args.KG_use:
            self.KG_RM = [KG_RM(args)] * args.topk
            self.KG_goal = KG_RM(args)
        else:
            self.KG_RM = []
        self.rel_dict_goal = None  # goal's rel dict
        self.goal_rel_len = 0
        self.goal_story = None

        # save story
        self.file = open(args.save_path, "w")

        # remain more diversity
        self.remain_div = args.remain_div
        self.diff_prompt = 0  # #of difference prompt
        if args.remain_div:
            self.diff_prompt = args.topk // 2
        self.early_stop_dis = args.early_stop_dis  # distance_of_node stop 0.95
        self.scores = []
        self.score_patience = args.score_patience  # the distance increase
        self.score_patience_step = args.score_patience_step  # how many rounds to wait to check

    def clear(self):
        self.goal_word = None
        self.goal_inference_set = defaultdict(list)

        # change all the time
        self.prompt = [] * self.topk
        self.node_history_set = [set()] * self.topk  # nodes in all the sentences
        self.inference_set = [set()] * self.topk
        self.node_history = [set()] * self.topk  # nodes in one sentence
        self.premise_inferences = [set()] * self.topk  # a set to record all the inferences on the previous sentence.
        self.roles = set()  # roles names. EX. ['Jenny'] or ['Jenny', 'Linda']
        self.scores = []
        # whether to turn on KG?
        if self.args.KG_use:
            self.KG_RM = [KG_RM(self.args)] * self.topk
            self.KG_goal = KG_RM(self.args)
        else:
            self.KG_RM = []
        self.rel_dict_goal = None  # goal's rel dict
        self.goal_rel_len = 0
        self.goal_story = None

    def save_file(self, stop_sig_patience):
        self.file.write(self.goal_story + '\n')
        if stop_sig_patience:
            self.file.write('Early stop for the score patience. \n')
            self.file.write('. '.join(self.prompt[0].split('.')[:-1 * self.score_patience_step]) + '\n')
            self.file.write('\n')
        for i, prompt in enumerate(self.prompt):
            self.file.write(str(self.scores[-1][i]) + '\n')
            self.file.write(prompt + '\n')
        self.file.write('=' * 10 + '\n')

        self.file.flush()  # save file

    def forward(self, round=0):
        """

        :param round: int
        :return:
        """
        stop_sig_patience = False
        if round == 0:  # We are in the 1st round
            # build the goal first
            update_time()
            self.prepare_next_step(goal_or_build='build')
            logger.info("Build target KG takes %.2f mins" % cal_time())

            # generate candidates
            action, cs_entity_KG_dict = self.generate_actions()
            logger.info("Generate candidates in 1st round takes %.2f mins" % cal_time())

            actions_lst = self.top_k_prunning(actions_dicts=action,
                                              goal_word='',
                                              distance_type='graph')
            logger.info("Top_k_prunning takes %.2f mins" % cal_time())

            # truncanate the action candidates
            if len(actions_lst) < self.topk:
                actions_chosen = actions_lst
            else:
                actions_chosen = actions_lst[:self.topk]

            print('actions_chosen =>', actions_chosen)
            # prepare next step
            outputs, nodes_consider, rel_consider, stop_sig = self.prepare_next_step(
                obj=actions_chosen[0][0], picked_action=actions_chosen[0][1][0], cs_entity_KG_dict=cs_entity_KG_dict, actions_chosen=actions_chosen
            )
            logger.info("Prepare after 1st round takes %.2f mins" % cal_time())

        else:  # After 1st round.
            '''
            Step 1:
                For each topk story trainer, we generate the action candidates and rank them by its scores
                We can get _trainers_next_, a list with its possible action and trainer idx            
            '''
            round += 1
            update_time()
            # with HiddenPrints():
            action_lst, cs_entity_KG_dict_lst = self.generate_actions()
            logger.info("Generate candidates in %d round takes %.2f mins" % (round, cal_time()))

            actions_lst = self.top_k_prunning(actions_dicts=action_lst, goal_word='', distance_type='graph')  # [(obj, sentence, distance, idx),...]
            logger.info("Top_k_prunning in %d round takes %.2f mins" % (round, cal_time()))

            '''
            Step 2:
                Find the topk action and their trainer
                New list: _next_trainers_
            '''
            # with HiddenPrints():
            # actions_lst_all = list(sorted(actions_lst_all, key=lambda x: x[1][-1]))[:min(self.topk, len(actions_lst_all))]
            # for trainer_idx, actions_lst, cs_entity_KG_dict in trainers_next:
            #     for action in actions_lst_all:
            #         if action in actions_lst:  # in case same trainer have more than 1 topk actions -> deepcopy
            #             next_trainers.append([copy.deepcopy(self.trainer[trainer_idx][0]), action, cs_entity_KG_dict, actions_lst[1][-1]])
            # print('actions_lst_all', actions_lst_all)
            # print('next_trainers', next_trainers)
            # del actions_lst_all
            # del trainers_next
            # gc.collect()  # clear cache

            '''
            Step 3:
                For each topk action candidate and its trainer, prepare next step
            '''
            with HiddenPrints():
                outputs, nodes_consider, rel_consider, stop_sig = self.prepare_next_step(
                                                                    obj=actions_lst[0][0],
                                                                    picked_action=actions_lst[0][1][0],
                                                                    cs_entity_KG_dict=cs_entity_KG_dict_lst[0],
                                                                    actions_chosen=actions_lst,
                                                                )
            logger.info("Prepare next step in %d round takes %.2f mins" % (round, cal_time()))

            if len(self.scores) >= self.score_patience_step:
                print('self.scores', self.scores,  -1 * self.score_patience_step)
                print(self.scores[-1 * self.score_patience_step])
                print(self.scores[-1])
                if max(self.scores[-1 * self.score_patience_step]) >= max(self.scores[-1]) - self.score_patience:
                    stop_sig_patience = True

        return outputs, nodes_consider, rel_consider, stop_sig, actions_lst, stop_sig_patience

    def graph_distance(self, obj=None, sentence=None, idx=0):
        """
        Check the graph diff between the action w/ exsiting state and the goal state
        :param obj:
        :param sentence:
        :return:
        """
        # print('+' * 30)
        # print(obj, sentence)
        # nodes we have
        old_nodes = set(self.KG_RM[idx].graph_state.nodes)
        # relationships
        edges_old = list(self.KG_RM[idx].graph_state.edges)
        rel_set = set()
        for edge in edges_old:
            rel_set.add(self.KG_RM[idx].graph_state[edge[0]][edge[1]]['rel'])
        #relation dict
        rel_dict = defaultdict(list)
        for edge in edges_old:
            rel_dict[edge[0]].append(self.KG_RM[idx].graph_state[edge[0]][edge[1]]['rel'])
            rel_dict[edge[1]].append(self.KG_RM[idx].graph_state[edge[0]][edge[1]]['rel'])

        # nodes we will have after having this action?
        print('>sentence', sentence)
        sentence = self.cr.forward(self.prompt[idx] + '>' + sentence).split('>')[-1]
        print('>>sentence', sentence)
        outputs, nodes_consider, rel_consider = self.verbatlas_gen.add_relation_local(sentence)  # dict[rel] = [node_1, node_2]
        old_nodes = old_nodes.union(nodes_consider)
        # rel_set = rel_set.union(rel_consider)
        for rel in outputs:  # add new rel to rel_dict
            nodes_lst = outputs[rel]
            for nodes in nodes_lst:
                rel_dict[nodes[0]].append(rel)
                rel_dict[nodes[1]].append(rel)

        # goal nodes
        goal_nodes = set(self.KG_goal.graph_state.nodes)
        print('goal_nodes', goal_nodes)

        # nodes overlapping
        overlapping_nodes = old_nodes.intersection(goal_nodes)
        distance = len(overlapping_nodes) / len(goal_nodes)  # larger is better
        print('overlapping_nodes', overlapping_nodes)
        # rel overlapping
        distance_rel = 0
        for node in overlapping_nodes:
            dis = len(set(rel_dict[node]).intersection(set(self.rel_dict_goal[node]))) / len(self.rel_dict_goal[node])  # list
            distance_rel += dis
            # print("distance", node, dis, rel_dict[node])

        # calculate the inference nodes weights
        inference_nodes = set()
        for rel in self.goal_inference_set:
            for nodes in self.goal_inference_set[rel]:
                inference_nodes.add(nodes[0])
                inference_nodes.add(nodes[1])
        distance_inf = len(old_nodes.intersection(inference_nodes)) / len(inference_nodes)  # larger is better

        print('>>>', obj, sentence, distance, distance_inf, distance_rel/self.goal_rel_len)

        return distance + 0.5 * distance_inf + distance_rel/self.goal_rel_len, distance

    def entailment_distance(self, prompt, sentence):
        """
        This function is used to add a score for those who should ve entail the prompt
        :param prompt: str. one sentence
        :param sentence: str. candidate sentence/action
        :return: float. score
        """
        label_map = {0: "contradiction", 1: "neutral", 2: "entailment"}
        # print(f"premise = {prompt}")
        # print(f"hypothesis = {sentence}")
        outputs, prediction = self.rocnli.forward(premise=prompt, hypothesis=sentence)
        # print(f"===== prediction: {label_map[prediction]} with {outputs.max()} probability")
        return (
            outputs.max() if prediction == 2 else 0
        )  # return [0.5, 1] when entailment else 0

    def top_k_prunning(self, actions_dicts, goal_word=None, distance_type="vebnet"):
        """
        Given a list of dicts of actions [dict, dict, ...]; dict = {'obj': sentence, ...]
        Return top k obj and sentence
        :param actions_dicts: list of action_dict; action_dict's key is obj, and value is list of sentence
        :param goal_word:
        :param distance_type: 'verbnet' : Lara's one; 'graph': graph difference
        :return:
        """

        print('actions_dicts', actions_dicts)
        distance_lst = []
        distance_dict = defaultdict(float)
        for idx, action_dict in enumerate(actions_dicts):
            # calculate the distance of all the sentences
            for obj, sentence in action_dict.items():

                # find the verbnet id here
                # parsed_predicates = self.Verbnet_Parser.parse_in_batch(sentences=sentence)
                # verb_lst = self.Verbnet_Parser.find_verbnet_id(parsed_predicates=parsed_predicates)
                # print('sentence', sentence, self.nltk_gen.find_verb_lemma(sentence[0]))

                # use mapping way to find verbnet id.
                if (
                    self.nltk_gen.find_verb_lemma(sentence[0])
                    and distance_type == "verbnet"
                ):
                    print('topk - sentence[0]', sentence[0])
                    verb_lst = verbnet_mapping[
                        self.nltk_gen.find_verb_lemma(sentence[0])
                    ]
                    distance = word_distance(
                        "/media/ASTER/Story-GPT2/scifi_data_all.txt",
                        verb_lst[0],
                        goal_word,
                        location="../data/" + goal_word + ".p",
                    )
                elif distance_type == "graph":
                    # compared with the goal graph
                    distance, distance_node = self.graph_distance(obj, sentence[0], idx)
                    # distance = self.graph_distance(
                    #     obj, sentence[0]
                    # ) * (5 + self.entailment_distance(
                    #     self.prompt.split(".")[-2], sentence[0]
                    # )) / 5
                    # print(obj, sentence[0])
                    # print('Distance =>', self.graph_distance(
                    #     obj, sentence[0]
                    # ),(5 + self.entailment_distance(
                    #     self.prompt.split(".")[-2], sentence[0]
                    # )) / 5)
                else:
                    distance = 0
                # print('verb_lst', obj, sentence, '=>', distance, self.entailment_distance(
                #         self.prompt.split(".")[-2], sentence[0]
                #     ))
                if distance_node >= self.early_stop_dis:
                    stop_sig = False  # true
                else:
                    stop_sig = False
                distance_lst.append((obj, sentence, distance, idx, stop_sig))

        # only remain the one with topk prob
        distance_lst = list(reversed(sorted(distance_lst, key=lambda x: x[-3])))

        # consider diversity
        remain_dict = defaultdict(list)
        if self.remain_div:
            total_num = 0
            for _obj, _sentence, _distance, _idx, _stop_sig in distance_lst:
                remain_dict[_idx].append((_obj, _sentence, _distance, _idx, _stop_sig))
                total_num += 1
                if len(list(remain_dict.keys())) >= self.diff_prompt and total_num >= self.topk:
                    break
            res = []
            res_set = set()
            for idx in remain_dict:
                res.append(remain_dict[idx][0])
                res_set.add(remain_dict[idx][0][1][0])
            id = 0
            print('distance_lst', distance_lst)
            while len(res) < self.topk:
                if distance_lst[id][1][0] not in res_set:
                    res.append(distance_lst[id])
                    res_set.add(distance_lst[id][1][0])
                id += 1
            print('!!!!!', res)
            return res

        # print('topk', min(len(distance_lst)-1, self.topk), distance_lst[:min(len(distance_lst)-1, self.topk)])
        return distance_lst[: min(len(distance_lst), self.topk)]

    def run_file(self):
        """
        Read the file and run forward.
        """
        f = open(self.file_path, "r")
        f_save = open(self.save_path, "w")
        for sentence in f:
            self.clear()
            sentence = sentence.replace("\n", "")
            sentence = sentence.strip()  # delete all the blank space
            if "." != sentence[-1]:  # In case some sentence has no period.
                sentence += "."
            self.prompt = sentence.strip()
            for idx in range(self.story_length - 1):
                self.define_roles()
                self.forward()
            f_save.write(self.prompt + "\n")

        f_save.close()

    def replace_coreference(self, sentence):
        """
        return sentences with only real names instead of she or he
        :param sentence:
        :return:
        """

    def different_verbs(self, first_sen, second_sen):
        if self.nltk_gen.find_verb_lemma(first_sen) == self.nltk_gen.find_verb_lemma(
            second_sen
        ):
            return False
        return True

    def action_gen(self, last_verb=None, fixed_actions=False, prompt="."):
        """
        TODO: Generate actions
        :return:
        """
        print("prompt for action genration is =>", self.prompt)
        fixed_actions_list = ["begin", "decide"]
        if fixed_actions:
            return fixed_actions_list[self.prompt.count(".") - 1]
        else:
            return self.verb_generator.next_verb(last_verb)
            # outputs_comet = self.comet.forward(input_event=prompt,
            #                                    relation=['xWant','xEffect'],
            #                                    sampling_algorithm='beam-5')
            # verb_phrases = []
            # for rel in outputs_comet:
            #     for inference in outputs_comet[rel]:
            #         if inference != 'none':
            #             verb_phrases.append(self.nltk_gen.find_verb_phrase(inference))
            # return random.choices(verb_phrases, [1/len(verb_phrases)] * len(verb_phrases))[0]

    def define_roles(self, sentence):
        """
        Find the sub., # of roles in this story and tense.
        """
        # time
        update_time()

        roles = self.Tagging.forward(sentence)
        print("self.roles =>", sentence, roles)
        for role in roles:
            self.roles.add(role)
            # self.roles = [self.roles[-1]]  # only consider the last one.
        self.tense = self.nltk_gen.tense_detect(sentence)
        self.number = "singular" if len(self.roles) == 1 else "plural"
        print("Roles in this story are ===> ", self.roles)
        self.prompt = [sentence] * self.topk

        logger.info("Define roles takes %.2f mins" % cal_time())

    def TE_MC_GPT_gen(self, prompt, num=1, length=15, patience=10):
        """
        There are 2 ways to generate the obj with GPT-2
        TE: generate sentence until this sentence is entailing the previous one.
        MC: generate alternatives of obj and ask MC model to choose one.
        :param prompt: The prompt for GPT-2. story history.
        :return: obj, noun.
        """
        neural_sets = []
        if self.args.TEorMC == "TE":
            for idx in range(self.args.TE_loop_limit):
                # generate one
                with HiddenPrints():
                    gen_text = self.GPT_2_gen.gen_text(
                        prompt_text=prompt,
                        order_remain=prompt.count("."),
                        output_obj=False,
                    )
                # print("premise is ===>", prompt.split(".")[-2])
                # print("hypothesis is ===>", gen_text)

                # generate whether hyp entails premise. 1-> yes, 0-> neural, -1->no
                te_res = self.rocnli.forward(
                    premise=prompt.split(".")[-2], hypothesis=gen_text
                )
                # print("TE result => ", te_res)

                #
                if te_res == 1:
                    return gen_text[
                        len(prompt.split(".")[-1]) + 1 : -1
                    ]  # only return the obj
                elif te_res == 0:
                    neural_sets.append(gen_text)

            # use sentence transformer here to choose the neural set.
            for sentence in neural_sets:
                if similarity_detect(
                    corpus=prompt.split(".")[-2],
                    query=sentence,
                    threshold_upper=self.args.TE_st_threshold_upper,
                    threshold_limit=self.args.TE_st_threshold_limit,
                ):
                    return sentence[len(prompt.split(".")[-1]) + 1 : -1]
            # print("neural_sets", neural_sets)
            if neural_sets:
                return neural_sets[0][len(prompt.split(".")[-1]) + 1 : -1]
            else:
                for i in range(patience):
                    obj = self.GPT_2_gen.gen_multiple_obj(prompt_text=prompt, num=1)
                    if len(obj[0].split(" ")) < length:  # consider the length of obj
                        break
                return obj

        elif self.args.TEorMC == "MC":  # self.args.TEorMC == 'MC'
            # Generate obj lists.
            with HiddenPrints():
                alternatives = self.GPT_2_gen.gen_multiple_obj(
                    prompt_text=prompt, num=self.args.number_GPT_obj
                )
            # print("The alternatives GPT generates are ===>", alternatives)
            # Use MC model to decide which option we should choose.
            obj = self.MC.forward(prefix=prompt, alternatives=alternatives)
            # print("MC model chooses ===> ", obj)

        else:
            for i in range(patience):
                obj = self.GPT_2_gen.gen_multiple_obj(prompt_text=prompt, num=1)
                # print("generate =>", obj[0])
                if len(obj[0].split(" ")) < length:  # consider the length of obj
                    break
        return obj

    def obj_filter(self, obj_list):
        """
        Given a obj list, find which obj's noun is in inference set.
        :param obj_list:
        :return:
        """
        for obj_long in obj_list:
            objs = self.nltk_gen.nounFind(obj_long)
            for obj in objs:
                if obj in self.inference_set:
                    # print("FIND obj part in inference => ", obj, "!" * 3)
                    return obj_long
        return None

    def simplify_dict(self, dict_sims):
        """
        Given a dict_sim, return a dict, key = rel, value -> list of list [[node1, node2],...]
        """
        dict_res = defaultdict(list)
        for dict_sim in dict_sims:
            for rel in dict_sim:
                res = []
                for i in range(len(dict_sim[rel])):
                    node_1 = self.nltk_gen.nounFind(dict_sim[rel][i][0])
                    node_2 = self.nltk_gen.nounFind(dict_sim[rel][i][1])
                    for node in node_1:
                        node = node.lower()
                        for entity in node_2:
                            entity = entity.lower()
                            res.append([node, entity])
                dict_res[rel] = res
        return dict_res

    def action_extract(self, sentences_generated, standard="verb", idx=0):
        """
        From sentence_lst to extract actions and construct a dict, action is key, value is the best sent.
        :param sentences_generated: list of string
        :return: dict, and best generation
        """
        if standard == "verb":
            action_dict = defaultdict(list)
            best_one, prob_best = "", 100000
            for sent in sentences_generated:
                action = self.nltk_gen.find_verb_lemma(sent)
                if action:
                    action_dict[action].append(sent)
            for action in action_dict:
                sent_lst = action_dict[action]
                probs = self.GPT_2_gen.calculate_prob(sentences=sent_lst)
                idx_candidates = probs.index(min(probs))
                if min(probs) < prob_best:
                    prob_best = min(probs)
                    best_one = sent_lst[idx_candidates]
                action_dict[action] = [sent_lst[idx_candidates]]
            return action_dict, best_one, prob_best
        else:
            """
            When consider which sentence can be used for this object, we should also consider the verb.
            """
            best_one, prob_best = "", 100000
            logger.info("[action_extract] %d objects in %dth round" % (len(sentences_generated), idx))
            for obj in sentences_generated:
                distance_lst = []
                sent_lst = sentences_generated[obj][0]
                update_time()
                probs = self.GPT_2_gen.calculate_prob(
                    sentences=[
                        self.prompt[idx].split(".")[-2] + ". " + s.strip() for s in sent_lst
                    ]
                )
                logger.info("GPT-2 sentence ranking takes %.2f mins" % cal_time())

                for i, sen in enumerate(sent_lst):
                    distance = 1

                    # use verbnet to choose sentence?

                    # verbs = self.nltk_gen.find_verb_lemma(sen)
                    # if verbs and verbs in verbnet_mapping:
                    #     verb_lst = verbnet_mapping[
                    #        verbs
                    #     ]
                    #     # print('self.goal_word', self.goal_word)
                    #     if verb_lst and self.goal_word:
                    #         distance = word_distance(
                    #             "/media/ASTER/Story-GPT2/scifi_data_all.txt",
                    #             verb_lst[0],
                    #             self.goal_word,
                    #             location="../data/" + self.goal_word + ".p",
                    #         )

                    distance_lst.append(distance)
                # modify the prob
                distance_lst = [dis / max(min(distance_lst), 0.00000000001) for dis in distance_lst]
                # print('<<<probs', probs)
                # print(len(distance_lst), len(probs))
                for i, prob in enumerate(probs):
                    probs[i] = prob - (5 - distance_lst[i])
                # print('>>>probs', probs)
                #########

                idx_candidates = probs.index(min(probs))
                # print('sent_lst', sent_lst, idx_candidates, probs)
                if min(probs) < prob_best:
                    prob_best = min(probs)
                    best_one = sent_lst[idx_candidates]
                sentences_generated[obj] = [sent_lst[idx_candidates]]

            logger.info("Sentences ranked in roberta are %s" % str(sentences_generated))
            return sentences_generated, best_one, prob_best

    def roberta_action_obj_gen(self, GPT_gen_sig=False, idx=0):
        """
        Generate actions from RoBerta.
        :return:
        """
        if GPT_gen_sig:
            sentence = self.TE_MC_GPT_gen(self.prompt[idx])[0]
            return sentence, {"": [sentence]}
        sentences_generated = []
        obj_entity_candidates = defaultdict(list)
        actions_candidates = []
        ret_sen, obj = None, None
        # generate obj first.
        obj_entity_list = self.object_generator.forward_all(
            inferences=self.inference_set[idx]
        )
        sentences_generated = []
        logger.info("There are %d entity need to be considered when roberta in %d prompt" % (len(obj_entity_list), idx))

        for obj_entity in obj_entity_list:
            # logger.info("Entity in roberta is %s" % str(obj_entity))
            sent_lst = self.roberta.prepare_sent(
                sub=self.prompt[idx].split(".")[-2] + ". " + " and ".join(self.roles),
                obj=obj_entity,
                k_action=3,
                k_after_obj=4,
            )

            if len(self.roles) > 1:
                for role in self.roles:
                    sent_lst += self.roberta.prepare_sent(
                        sub=self.prompt[idx].split(".")[-2] + ". " + role,
                        obj=obj_entity,
                        k_action=3,
                        k_after_obj=4,
                    )

            sent_lst += self.roberta.prepare_sent(
                sub=self.prompt[idx].split(".")[-2] + ". ",
                obj=obj_entity,
                k_action=3,
                k_after_obj=4,
            )
            sent_gen = self.roberta.get_prediction(sent_lst)  # no period
            sentences_generated += sent_gen
            obj_entity_candidates[obj_entity].append(sent_gen)

        # logger.info("Fill in all mask in %dth prompt takes %.2f mins" % (idx, cal_time()))
        # consider which one makes more sense.
        if self.standard == "verb":
            action_dict, best_one, prob_best = self.action_extract(sentences_generated, idx=idx)
            return best_one, action_dict
        else:
            obj_dict, best_one, prob_best = self.action_extract(
                obj_entity_candidates, standard="sentence", idx=idx
            )
            # print("best_one, obj_dict", best_one, obj_dict)
            return best_one, obj_dict

    def bert_action_obj_gen(self, num_sentence=3, return_sentence=False):
        """
        Generate actions from BERT.
        :return:
        """
        sentences_generated = []
        obj_entity_candidates = []
        actions_candidates = []
        ret_sen, obj = None, None
        # generate obj first.
        obj_entity_list = self.object_generator.forward_all(
            inferences=self.inference_set
        )
        # print("obj_entity_list =>", obj_entity_list)
        for obj_entity in obj_entity_list:
            # print("=" * 20)
            # print("entity is =>", obj_entity)
            # # generate obj first.
            # obj_entity = self.object_generator.forward(inferences=self.inference_set)
            action_lst = []
            if return_sentence:
                # return a sentence with new verb and obj.    + ' and '.join(self.roles) + ' '
                ret_sen_list = self.verb_generator.next_verb(
                    prompt=self.prompt.split(".")[-2]
                    + ". "
                    + " and ".join(self.roles)
                    + " ",
                    obj=obj_entity,
                    return_sentence=return_sentence,
                )
                # print("Returned is =>", ret_sen_list)

                if ret_sen_list:
                    ret_sen_list = [ret_sen.strip() for ret_sen in ret_sen_list]
                    probs_ret_sen = self.GPT_2_gen.calculate_prob(
                        sentences=ret_sen_list
                    )

                    # only use the verb which is not the same with the previous sentence.
                    while min(probs_ret_sen) < 100:
                        idx_candidates_ret_sen = probs_ret_sen.index(min(probs_ret_sen))
                        if self.different_verbs(
                            first_sen=self.prompt.split(".")[-2],
                            second_sen=ret_sen_list[idx_candidates_ret_sen],
                        ):
                            break
                        else:
                            probs_ret_sen[idx_candidates_ret_sen] = 100
                            idx_candidates_ret_sen = probs_ret_sen.index(
                                min(probs_ret_sen)
                            )

                    ret_sen = ret_sen_list[idx_candidates_ret_sen]
                    if obj_entity in ret_sen and self.different_verbs(
                        first_sen=self.prompt.split(".")[-2], second_sen=ret_sen
                    ):
                        # generate more
                        prompt = str(self.prompt) + " " + ret_sen
                        obj = self.TE_MC_GPT_gen(prompt)[0].strip()

                        # comet is here
                        # if self.TE.forward_inferencesp(self.premise_inferences, ret_sen + ' ' + obj) and
                        if similarity_detect(
                            corpus=ret_sen + " " + obj,
                            query=self.prompt.split(".")[-2],
                            threshold_upper=0.9,
                            threshold_limit=0,
                        ):
                            # print(
                            #     "generate one =>",
                            #     self.prompt + " " + ret_sen + " " + obj,
                            # )
                            sentences_generated.append(
                                self.prompt + " " + ret_sen + " " + obj
                            )
                            obj_entity_candidates.append(obj)
                            actions_candidates.append(ret_sen)

            else:
                # generate verb
                action = self.verb_generator.next_verb(
                    prompt=self.prompt + " " + " and ".join(self.roles) + " ",
                    obj=obj_entity,
                )[0]

                action = self.nltk_gen.verb_tense_gen(
                    verb=action,
                    tense=self.tense,
                    person=3,  # We prefix the story are all the 3rd person.
                    number=self.number,
                )
                # print("Action from BERT is => ", action)

                # Get the new sentencep
                prompt = self.prompt + " " + " and ".join(self.roles) + " " + action
                # Use GPT-2 to generate the remaining part of the sentence
                # Then check the similarity with the obj entity
                obj_filtered = None
                for idx in range(self.args.obj_num):
                    # Generate obj.
                    obj = self.TE_MC_GPT_gen(prompt)[0]
                    # print("obj is ===> ", obj)
                    # filter obj
                    if similarity_detect(
                        corpus=obj,
                        query=obj_entity,
                        threshold_upper=1,
                        threshold_limit=0.3,
                    ):
                        obj_filtered = obj
                        sentences_generated.append(
                            " and ".join(self.roles) + " " + action + " " + obj
                        )
                        obj_entity_candidates.append(obj)
                        actions_candidates.append(action)
                        break

        # consider which one makes more sense.
        if sentences_generated:
            probs = self.GPT_2_gen.calculate_prob(sentences=sentences_generated)
            idx_candidates = probs.index(min(probs))
            # print(
            #     "What returns => obj=>",
            #     actions_candidates[idx_candidates],
            #     "|",
            #     obj_entity_candidates[idx_candidates],
            # )
            return (
                actions_candidates[idx_candidates],
                obj_entity_candidates[idx_candidates],
                actions_candidates,
                obj_entity_candidates,
            )
        else:
            prompt = str(self.prompt) + " " + " and ".join(self.roles)
            # print("prompt for last one is =>", prompt)
            obj = self.TE_MC_GPT_gen(prompt)
            # print("generated obj is =>", obj)
            # print("=> What returns => obj=>", " and ".join(self.roles), "|", obj[0])
            return " and ".join(self.roles), obj[0]

    def action_obj_gen(self):
        """
        Generate actions with verb set then generate obj from GPT-2
        :return:
        """
        # Generate action.
        action = self.action_gen(
            last_verb=self.nltk_gen.find_verb_lemma(prompt=self.prompt.split(".")[-2]),
            prompt=self.prompt.split(".")[-2],
        )

        # Use this action to generate obj.
        # print("verb => ", action, "tense =>", self.tense, "number => ", self.number)
        action = self.nltk_gen.verb_tense_gen(
            verb=action,
            tense=self.tense,
            person=3,  # We prefix the story are all the 3rd person.
            number=self.number,
        )
        # print("action will be => ", action)

        # Form the prompt of the GPT-2 to generate obj.
        prompt = self.prompt + " " + " and ".join(self.roles) + " " + action
        # print("The prompt we feed into GPT-2 is ===>", prompt)

        for idx in range(self.args.obj_num):
            # Generate obj.
            obj = self.TE_MC_GPT_gen(prompt)
            # print("obj is ===> ", obj)
            # filter obj
            obj_filtered = self.obj_filter(obj)
            if obj_filtered:
                break
        if not obj_filtered:
            obj_filtered = obj[0]
        # print("Filtered obj is ===> ", obj_filtered)
        return action, obj_filtered

    def prepare_next_step(
        self,
        obj=None,
        picked_action=None,
        cs_entity_KG_dict=None,
        goal_or_build="build",
        goal_word=None,
        actions_chosen=[None]
    ):
        """
        Pipeline - 1 KG-transition
        """
        stop_sig = False
        prompt_next = []
        KG_lst = []
        self.node_history = []
        self.inference_set = []
        # print('actions_chosen', actions_chosen)
        # outputs = None
        # nodes_consider = None
        # rel_consider = None
        if goal_or_build == "build":
            if actions_chosen[0]:
                self.scores.append([])
            for i, action in enumerate(actions_chosen):
                if action:  # obj, sentence, distance, idx, stop_sig
                    obj = action[0]
                    picked_action = action[1][0].strip()
                    idx = action[3]
                    stop_sig = stop_sig or action[4]  # reach the whole story
                    score = action[2]
                    self.scores[-1].append(score)
                else:
                    idx = 0
                if goal_word:
                    self.goal_word = goal_word

                print('picked_action, =>', picked_action)
                if picked_action:  # update prompt
                    prompt_next.append(self.prompt[idx] + " " + picked_action.strip() + ".")
                else:
                    prompt_next.append(self.prompt[idx])

                # generate relation triples with verbatlas
                prompt_this_round = self.cr.forward(prompt_next[-1])
                prompt_KG = prompt_this_round.split(".")[-2].strip()
                print(">>> prompt in KG is ", prompt_KG)
                print('>>> action is ', action)

                # get inferences for this prompt
                # with HiddenPrints():
                #     premise_inferences = self.comet.forward(
                #         input_event=prompt_KG,
                #         relation=["xWant", "xEffect", 'xAttr', 'xNeed', 'xIntent', 'xReact'],
                #         sampling_algorithm="beam-5",
                #     )

                outputs, nodes_consider, rel_consider = self.verbatlas_gen.add_relation_local(prompt_KG)
                verb_find = self.nltk_gen.find_all_verb_lemma(prompt_KG)

                # simplify the nodes names. 'in the morning' -> ['morning']
                node_history = set()  # clear nodes history in this sentence.
                inference_set = set()  # clear nodes inference in this sentence.
                for rel in outputs:
                    res = []  # delete the old one and add the new one.
                    for i in range(len(outputs[rel])):
                        node_1 = self.nltk_gen.nounFind(outputs[rel][i][0])
                        node_2 = [" ".join(self.nltk_gen.nounFind(outputs[rel][i][1]))]
                        for node in node_1:
                            node = node.lower()
                            if (
                                node.title() not in self.roles
                                and not self.Stopwords_Filter.filter_word(node.lower())
                            ):  # remove those words same w/ roles
                                node_history.add(node)
                            for entity in node_2:
                                entity = entity.lower()
                                res.append([node, entity])
                                if (
                                    entity.title() not in self.roles
                                    and not self.Stopwords_Filter.filter_word(
                                        entity.lower()
                                    )
                                ):  # remove those words same w/ roles
                                    node_history.add(entity)
                    outputs[rel] = res
                # print("self.node_history", self.node_history)

                # add wordnet entity to the KG
                if cs_entity_KG_dict and obj and obj in cs_entity_KG_dict[idx]:
                    outputs[cs_entity_KG_dict[idx][obj][-1]] = [cs_entity_KG_dict[idx][obj][0]]

                # build KG - add branches
                # print("inference add to KG =>", outputs)
                KG_RM = copy.deepcopy(self.KG_RM[idx])
                if KG_RM:
                    KG_RM.add_branches(outputs)
                    KG_RM.visualize(order=self.prompt.count("."))

                # update
                KG_lst.append(KG_RM)
                self.inference_set.append(inference_set)
                self.node_history.append(node_history)

            # update
            self.prompt = prompt_next
            self.KG_RM = KG_lst


        else:  # for the target 'goal'
            if goal_word:
                self.goal_word = goal_word
            self.goal_story = picked_action
            picked_action = self.cr.forward(picked_action)
            sentences = [sen for sen in picked_action.split(".") if sen.strip()]
            print('Goal sentences ->', sentences)
            inference_set = defaultdict(list)
            for sentence in sentences:

                outputs, nodes_consider, rel_consider = self.verbatlas_gen.add_relation_local(
                    sentence.strip()
                )
                print('--- outputs', sentence, outputs)
                # simplify the nodes names. 'in the morning' -> ['morning']

                # consider exact verb matching

                # verb_find = self.nltk_gen.find_all_verb_lemma(sentence)
                # if verb_find:
                #     outputs['verb'] = []
                # for verb in verb_find:  # only consider single char for now.
                #     for role in self.roles:
                #         outputs['verb'].append([role, verb])

                for rel in outputs:
                    res = []  # delete the old one and add the new one.
                    for i in range(len(outputs[rel])):
                        node_1 = self.nltk_gen.nounFind(outputs[rel][i][0])
                        node_2 = [" ".join(self.nltk_gen.nounFind(outputs[rel][i][1]))]
                        for node in node_1:
                            node = node.lower()
                            # if (
                            #     node.title() not in self.roles
                            #     and not self.Stopwords_Filter.filter_word(node.lower())
                            # ):  # remove those words same w/ roles
                            #     self.node_history.add(node)
                            for entity in node_2:
                                cs_dict = self.simplify_dict(
                                    self.conceptNet.relationGenerator(entity, filter=False)
                                )

                                entity = entity.lower()
                                res.append([node, entity])
                                # if (
                                #     entity.title() not in self.roles
                                #     and not self.Stopwords_Filter.filter_word(
                                #         entity.lower()
                                #     )
                                # ):  # remove those words same w/ roles
                                #     self.node_history.add(entity)

                                for _rel in cs_dict:
                                    inference_set[_rel] += cs_dict[_rel]
                    outputs[rel] = res


                # print('!!!inference_set ===》', inference_set)
                # for rel in inference_set:
                #     outputs[rel] += inference_set[rel]

                if self.KG_RM:
                    self.KG_goal.add_branches(outputs)
                    self.KG_goal.visualize(order=self.prompt.count("."))
            self.goal_inference_set = copy.deepcopy(inference_set)


            # relationships
            edges_goal = list(self.KG_goal.graph_state.edges)
            rel_set_goal = set()
            for edge in edges_goal:
                rel_set_goal.add(self.KG_goal.graph_state[edge[0]][edge[1]]['rel'])
            self.rel_dict_goal = defaultdict(list)
            for edge in edges_goal:
                self.rel_dict_goal[edge[0]].append(self.KG_goal.graph_state[edge[0]][edge[1]]['rel'])
                self.rel_dict_goal[edge[1]].append(self.KG_goal.graph_state[edge[0]][edge[1]]['rel'])
                self.goal_rel_len += 2
            print('self.rel_dict_goal', self.rel_dict_goal)
        print('outputs, nodes_consider, rel_consider, stop_sig', outputs, nodes_consider, rel_consider, stop_sig)
        return outputs, nodes_consider, rel_consider, stop_sig

    def generate_actions(self):
        action_dict_lst = []
        cs_entity_KG_dict_lst = []
        # print('self.node_history =>', self.node_history)
        # print('self.inference_set', self.inference_set)

        for i in range(len(self.node_history)):
            # add commonsense conceptnet to KG dict here
            cs_entity_KG_dict = defaultdict(list)
            GPT_gen_sig = False  # True will trigger GPT-2 generation.

            # add new comet inference
            update_time()
            new_inference = comet_new_usage.filter_obj_comet(list_of_objects=self.prompt[i],
                                                             comet_model=self.Comet,
                                                             nltk_model=self.nltk_gen,
                                                             prompt=self.prompt[i])
            new_inference.union(comet_new_usage.filter_obj_comet(list_of_objects=self.prompt[i].split('.')[-2],
                                                                 comet_model=self.Comet,
                                                                 nltk_model=self.nltk_gen,
                                                                 prompt=self.prompt[i]))
            logger.info("NEW Comet takes %.2f mins to generate one inference" % cal_time())

            self.inference_set[i] = self.inference_set[i].union(new_inference)
            logger.info("There are %d nodes in node_history in %dth prompt" % (len(self.node_history[i]), i))

            for entity in self.node_history[i]:
                if entity not in self.node_history_set[i] and entity not in [
                    role.lower() for role in self.roles
                ]:
                    cs_dict = self.simplify_dict(
                        self.conceptNet.relationGenerator(entity, filter=False)
                    )
                    print("cs_dict", cs_dict, self.roles)

                    # comet
                    print('Prompt is ', self.prompt[i].split(".")[-2], '; Entity is', entity)
                    # add new comet inference
                    new_inference = comet_new_usage.filter_obj_comet(list_of_objects=entity,
                                                                     comet_model=self.Comet,
                                                                     nltk_model=self.nltk_gen,
                                                                     prompt=self.prompt[i])
                    self.inference_set[i] = self.inference_set[i].union(new_inference)
                    print('New inference is ', new_inference)

                    # record all the inference in a set: self.inference_set
                    for rel in cs_dict:
                        for nodes in cs_dict[rel]:
                            # if nodes[0] not in self.KG_RM.graph_state.nodes() and nodes[0] != entity:
                            #     self.inference_set.add(nodes[0].lower())
                            for j in range(2):
                                if (
                                    self.KG_RM[i]
                                    and nodes[j] not in self.KG_RM[i].graph_state.nodes()
                                    and nodes[j] not in entity
                                    # and nodes[0] in nodes_consider
                                ):
                                    self.inference_set[i].add(nodes[j].lower())
                                    cs_entity_KG_dict[nodes[j].lower()] = [nodes, rel]
                                if (
                                    not self.KG_RM[i]
                                    and nodes[j] not in self.node_history[i]
                                    and nodes[j] not in entity
                                    # and nodes[0] in nodes_consider
                                ):
                                    self.inference_set[i].add(nodes[j].lower())
                                    cs_entity_KG_dict[nodes[j].lower()] = [nodes, rel]

                    # outputs.update(cs_dict)  # the cs_dict need to be simplify as well
                self.node_history_set[i].add(entity)

            print('self.inference_set', self.inference_set)
            # self.inference_set[i] = self.inference_set[i].union(comet_new_usage.filter_obj_comet(list_of_objects=[self.prompt[i].split(".")[-2]],
            #                                                                                      comet_model=self.Comet,
            #                                                                                      nltk_model=self.nltk_gen,
            #                                                                                      prompt=self.prompt[i].split(".")[-2]))
            # self.inference_set[i] = self.inference_set[i].union(
            #     comet_new_usage.filter_obj_comet(list_of_objects=[self.prompt[i]],
            #                                      comet_model=self.Comet,
            #                                      nltk_model=self.nltk_gen,
            #                                      prompt=self.prompt[i]))
            if not self.inference_set[i]:
                GPT_gen_sig = True
            else:
                # remove all the nodes here in self.inference_set
                filtered_inference = []
                update_time()
                for node in self.node_history_set[i]:
                    for inference in self.inference_set[i]:
                        if similarity_detect(
                            corpus=node,
                            query=inference,
                            threshold_upper=100,
                            threshold_limit=0.8,
                        ):
                            filtered_inference.append(inference)
                logger.info("Filter too similar words takes %.2f mins" % cal_time())
                self.inference_set[i] = set(
                    filter(lambda x: x not in filtered_inference, self.inference_set[i])
                )

            new_prompt, action_dict = self.roberta_action_obj_gen(GPT_gen_sig, idx=i)
            # logger.info("Generate candidates through roberta in %dth prompt takes %.2f mins" % (i, cal_time()))
            action_dict_lst.append(action_dict)
            cs_entity_KG_dict_lst.append(cs_entity_KG_dict)

        return action_dict_lst, cs_entity_KG_dict_lst
