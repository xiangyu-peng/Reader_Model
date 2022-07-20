from reader_model_kg_gen import Reader_Model_Trainer_KG
import copy
import sys
import os
import gc

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

class KG_diff_multiple_KG_trainer(object):
    """
    This object is used to duplicate graphs and remain multiple sentences at once.
    """
    def __init__(self, trainer, topk=2):
        self.topk = topk
        self.trainer = [trainer]


    def reset(self):
        pass

    def forward(self, trainers=[]):

        if len(self.trainer) == 1:  # We are in the 1st round
            outputs, nodes_consider, rel_consider = self.trainer[0].prepare_next_step(goal_or_build='build')
            # print('nodes_consider>>>', nodes_consider)
            action, cs_entity_KG_dict = self.trainer[0].generate_actions(outputs, nodes_consider)
            actions_lst = self.trainer[0].top_k_prunning(actions_dicts=[action],
                                                      goal_word='',
                                                      distance_type='graph')
            # print('actions_lst >>>', actions_lst)
            if len(actions_lst) < self.topk:
                actions_chosen = actions_lst
            else:
                actions_chosen = actions_lst[:self.topk]

            # prepare next step
            trainers = []
            for action in actions_chosen:
                # print('!!!action =>', action)
                with HiddenPrints():
                    trainer = copy.deepcopy(self.trainer[0])
                    outputs, nodes_consider, rel_consider = trainer.prepare_next_step(
                        action[0], action[1][0], cs_entity_KG_dict
                    )
                trainers.append([trainer, copy.deepcopy(outputs),  copy.deepcopy(nodes_consider), copy.deepcopy(rel_consider), 0])

        else:  # After 1st round.
            actions_lst_all = [] # find all the new actions.
            trainers_next = []
            '''
            Step 1:
                For each topk story trainer, we generate the action candidates and rank them by its scores
                We can get _trainers_next_, a list with its possible action and trainer idx            
            '''
            with HiddenPrints():
                for i, [trainer, outputs, nodes_consider, rel_consider, distance] in enumerate(self.trainer):
                    action, cs_entity_KG_dict = trainer.generate_actions(outputs, nodes_consider)
                    actions_lst = trainer.top_k_prunning(actions_dicts=[action], goal_word='',
                                                         distance_type='graph')
                    trainers_next.append([i, actions_lst, cs_entity_KG_dict])
                    for action in actions_lst:
                        actions_lst_all.append(action)
            print('actions_lst_all', actions_lst_all)
            '''
            Step 2:
                Find the topk action and their trainer
                New list: _next_trainers_
            '''
            print('list(reversed(sorted(actions_lst_all, key=lambda x: x[1][-1])))', list(reversed(sorted(actions_lst_all, key=lambda x: x[1][-1]))))
            with HiddenPrints():
                next_trainers = []
                actions_lst_all = list(sorted(actions_lst_all, key=lambda x: x[1][-1]))[:min(self.topk, len(actions_lst_all))]
                for trainer_idx, actions_lst, cs_entity_KG_dict in trainers_next:
                    for action in actions_lst_all:
                        if action in actions_lst:  # in case same trainer have more than 1 topk actions -> deepcopy
                            next_trainers.append([copy.deepcopy(self.trainer[trainer_idx][0]), action, cs_entity_KG_dict, actions_lst[1][-1]])
            print('actions_lst_all', actions_lst_all)
            print('next_trainers', next_trainers)
            del actions_lst_all
            del trainers_next
            gc.collect()  # clear cache

            '''
            Step 3:
                For each topk action candidate and its trainer, prepare next step
            '''
            with HiddenPrints():
                trainers = []
                for trainer, action, cs_entity_KG_dict, distance in next_trainers:
                    # new_trainer = copy.deepcopy(trainer)
                    outputs, nodes_consider, rel_consider = trainer.prepare_next_step(
                        action[0], action[1][0], cs_entity_KG_dict
                    )
                    trainers.append([trainer, outputs, nodes_consider, rel_consider, distance])
                    # print('prompt>>>', new_trainer.prompt, action[0], action[1][0])
                    #     del new_trainer
                del next_trainers
                gc.collect()

        self.trainer = trainers
        # del trainers
        # gc.collect()
