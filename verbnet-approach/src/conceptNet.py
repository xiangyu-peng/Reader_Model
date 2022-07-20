import requests
from collections import defaultdict
import argparse
class ConceptNet(object):
    def __init__(self):
        # self.request_url = 'http://api.conceptnet.io/c/en/'
        self.request_url = 'http://api.conceptnet.io/query?start=/c/en/'
        # self.request_url = 'http://3.235.107.151/c/en/'
        self.need_rels = ['RelatedTo', 'HasA', 'AtLocation', 'Causes', 'HasSubevent', 'HasFirstSubevent', 'HasLastSubevent',
                          'Desires', 'LocatedNear', 'CausesDesire', 'UsedFor']

    def relationGenerator(self, object, filter=True, nltk_model=None):
        if nltk_model:
            objects = [object, nltk_model.word_lemma(object)]
        else:
            objects = [object]
        res = []
        for object in objects:
            res.append(self.relationGenerator_single(object, filter))
        return res

    def relationGenerator_single(self, object, filter=True):
        """
        Given an node/str, outputs relations.
        :param obj: str
        :return: dict. key = rel, value = list of list. [[node, node],[node, node]]
        """
        # print('OOOBBBBJJJJ', object, filter)
        res = defaultdict(list)
        obj = requests.get(self.request_url + object.lower()).json()
        # print('url',self.request_url + object.lower())
        # print('obj', obj)
        # if 'edges' in obj:
        for i in range(len(obj['edges'])):
            # print('--->', obj['edges'][i])
            if 'language' in obj['edges'][i]['start'] and \
                'language' in obj['edges'][i]['end'] and \
                    obj['edges'][i]['start']['language'] == 'en' and \
                    obj['edges'][i]['end']['language'] == 'en':

                rel = obj['edges'][i]['rel']['label']
                if filter and rel in self.need_rels:
                    res[rel].append([])
                    res[rel][-1].append(obj['edges'][i]['start']['label'])
                    res[rel][-1].append(obj['edges'][i]['end']['label'])
                elif not filter:
                    # print('<--->', obj['edges'][i])
                    res[rel].append([])
                    res[rel][-1].append(obj['edges'][i]['start']['label'])
                    res[rel][-1].append(obj['edges'][i]['end']['label'])
                else:
                    pass

            elif 'language' not in obj['edges'][i]['start']:
                rel = obj['edges'][i]['rel']['label']
                if rel in self.need_rels:
                    res[rel].append([])
                    res[rel][-1].append(obj['edges'][i]['start']['label'])
                    res[rel][-1].append(obj['edges'][i]['end']['label'])
                elif not filter:
                    res[rel].append([])
                    res[rel][-1].append(obj['edges'][i]['start']['label'])
                    res[rel][-1].append(obj['edges'][i]['end']['label'])
                else:
                    pass
            elif 'language' not in obj['edges'][i]['end']:
                rel = obj['edges'][i]['rel']['label']
                if rel in self.need_rels:
                    res[rel].append([])
                    res[rel][-1].append(obj['edges'][i]['start']['label'])
                    res[rel][-1].append(obj['edges'][i]['end']['label'])
                elif not filter:
                    res[rel].append([])
                    res[rel][-1].append(obj['edges'][i]['start']['label'])
                    res[rel][-1].append(obj['edges'][i]['end']['label'])
                else:
                    pass
            else:
                pass
        if not res and len(object.split()) > 1:
            res_lst = []
            for obj_single in object.split():
                res_lst.append(self.relationGenerator(obj_single, filter=filter))
            for res_dicts in res_lst:
                for res_dict in res_dicts:
                    for key in res_dict:
                        # print('res_dict', res_dict)
                        # print('res_lst', res_lst)
                        # print('res_dict[key]', res_dict[key])
                        # print('res[key]', res[key])
                        res[key] += res_dict[key]
        return res

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--example",
                        type=str,
                        default='cake',
                        help="a obj")
    args = parser.parse_args()
    obj = args.example
    conceptNet = ConceptNet()
    print(conceptNet.relationGenerator(obj, filter=False))

