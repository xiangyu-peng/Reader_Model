import sys
sys.path.append("../../comet-atomic-2020/models/comet_atomic2020_bart")
from generation_example import Comet
import argparse
from nltk_gen import nltk_gen
import itertools
rels_consider = ['xNeed', 'xWant', 'xIntent', 'xEffect', 'Desires', 'xReact', 'CausesDesire', 'xAttr']

def filter_obj_comet(list_of_objects, comet_model, nltk_model, prompt):
    set_of_entities = set()
    for obj in list_of_objects:
        queries = ["{} {} [GEN]".format(obj, rel) for rel in comet_model.all_relations]
        results = comet_model.generate(queries, decode_method="beam", num_generate=5)

        for i, result in enumerate(results):
            print('result', comet_model.all_relations[i], '===>', result)
            for entity in result:
                entity = nltk_model.comet_to_delete(words=entity, sentence=prompt)
                if entity and entity != 'none':
                    set_of_entities.add(entity.replace('.', '').strip().lower())
        return set_of_entities

def filter_obj_comet_entity(list_of_objects, comet_model, nltk_model, goal=False):
    set_of_entities, set_of_phrases = set(), set()
    for obj in list_of_objects:
        queries = ["{} {} [GEN]".format(obj, rel) for rel in rels_consider]
        results = comet_model.generate(queries, decode_method="beam", num_generate=5)

        for i, result in enumerate(results):
            # print(result)
            if i < 1:
                for r in result:
                    if r.strip() != 'none':
                        set_of_phrases.add(r.strip())

            for entity in result:
                entity = nltk_model.comet_to_entity(words=entity)
                for e in entity:
                    if e.strip().lower() != 'none':
                        set_of_entities.add(e.replace('.', '').strip().lower())
    return set_of_entities, set_of_phrases

def filter_obj_comet_want(list_of_objects, comet_model, nltk_model, goal=False):
    set_of_phrases = set()
    for obj in list_of_objects:
        queries = ["{} {} [GEN]".format(obj, rel) for rel in ['xWant', 'xEffect']]
        results = comet_model.generate(queries, decode_method="beam", num_generate=5)

        for i, result in enumerate(results):
            for r in result:
                if r.strip() != 'none':
                    set_of_phrases.add(r.strip())

        return set_of_phrases

def filter_obj_comet_need(list_of_objects, comet_model, nltk_model, goal=False):
    set_of_phrases = set()
    for obj in list_of_objects:
        queries = ["{} {} [GEN]".format(obj, rel) for rel in ['xNeed']]
        results = comet_model.generate(queries, decode_method="beam", num_generate=5)

        for i, result in enumerate(results):
            for r in result:
                if r.strip() != 'none':
                    set_of_phrases.add(r.strip())

        return set_of_phrases

if __name__ == '__main__':
    comet = Comet("../../comet-atomic-2020/models/comet_atomic2020_bart/comet-atomic_2020_BART")  # new atomic comet 2020
    comet.model.zero_grad()
    # head = 'Char love ice cream'
    # queries = ["{} {} [GEN]".format(head, rel) for rel in comet.all_relations]
    # results = comet.generate(queries, decode_method="beam", num_generate=5)
    # print('-' * 20)
    # print('results', results)
    # print('*' * 50)

    parser = argparse.ArgumentParser()
    parser.add_argument("--example", type=str, default="apple", help="a obj")
    args = parser.parse_args()
    nltk_gen = nltk_gen()

    # print(filter_obj_comet(list_of_objects=['eat cake'], comet_model=comet, nltk_model=nltk_gen))
    queries = ["{} {} [GEN]".format('Alice enjoyed this beautiful view.', rel) for rel in comet.all_relations]
    results = comet.generate(queries, decode_method="beam", num_generate=10)

    for i, result in enumerate(results):
        print('result', comet.all_relations[i], '===>', result)

    # wants = filter_obj_comet_want(list_of_objects=['graduate from college'], comet_model=comet, nltk_model=nltk_gen)
    # print('wants', wants)
    # res = set()
    # for w in wants:
    #     vs, ns = nltk_gen.find_v_N(w)
    #     product = list(itertools.product(vs, ns))
    #     for p in product:
    #         res.add(p[0] + ' ' + p[1])
    # print(res)
