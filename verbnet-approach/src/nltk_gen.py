import spacy
import argparse
import itertools
import re
from pattern.en import conjugate, SG, PL, INFINITIVE, PRESENT, PAST, FUTURE, pluralize, singularize, comparative, superlative, referenced
useless_verbs = ['got', 'did', 'wanted', 'went', 'had', 'made', 'found', 'ate']
useless_verbs += ['liked', 'loved', 'felt']

class nltk_gen(object):
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")

    def find_v_N(self, text):
        '''
        find v and noun
        :param text: phrases
        :return:
        '''
        res_v = set()
        res_n = self.noun_phrase_finder(words=text, sentence=text)
        res_adj = self.adj_finder_entity(words=text)
        tokens = self.nlp(text)
        for token in tokens:
            if 'V' in token.tag_:
                res_v.add(self.verb_tense_gen(verb=token.text, tense=PAST, person=3, number=SG))
        return list(res_v), list(res_adj), list(res_n)

    def adj_finder_entity(self, words):
        res = set()
        tokens = self.nlp(words)
        for token in tokens:

            if 'JJ' in token.tag_ or 'RB' in token.tag_:
                res.add(token.text)

        return res

    def read_tags(self, words):
        """
        print tags
        :param words:
        :return:
        """
        res = dict()
        tokens = self.nlp(words)
        for token in tokens:
            print(token, token.tag_)
            res[token.text] = token.tag_
        return res

    def tense_detect(self, prompt):
        """
        Check the tense of verb
        """
        tokens = self.nlp(prompt)
        for token in tokens:
            if "VBD" == token.tag_:
                return "past"
        return "present"

    def verb_tense_gen(self, verb, tense, person, number):
        """

        :param verb: does not care whether it is lemma.
        :param tense: 'present', "past"
        :param person: 1 or 2 or 3
        :param number: "plural" or "singular"
        :return:
        """
        if len(verb.split(" ")) == 1:
            return conjugate(
                verb=str(verb),
                tense=tense,  # INFINITIVE, PRESENT, PAST, FUTURE
                person=int(person),  # 1, 2, 3 or None
                number=number,
            )  # SG, PL
        else:
            tokens = self.nlp(verb)
            res = ""
            for token in tokens:
                res = res + " " if res != "" else ""
                if "V" in token.tag_:
                    try:
                        res += conjugate(
                            verb=token.lemma_,
                            tense=tense,  # INFINITIVE, PRESENT, PAST, FUTURE
                            person=int(person),  # 1, 2, 3 or None
                            number=number,
                        )  # SG, PL
                    except StopIteration:
                        res += token.lemma_
                else:
                    res += token.text
            return res

    def nounFind(self, prompt):
        tokens = self.nlp(prompt)
        res = []
        for token in tokens:
            if "NN" in token.tag_:
                res.append(token.lemma_)
        return res if res else [prompt]


    def nounFindNone(self, prompt):
        tokens = self.nlp(prompt)
        res = []
        for token in tokens:
            if "NN" in token.tag_ and token.lemma_ not in ["personx", "none"]:
                res.append(token.lemma_)
        return res

    def find_verb_lemma(self, prompt):
        tokens = self.nlp(prompt)
        for token in tokens:
            if "V" in token.tag_:
                return token.lemma_

    def find_all_verb_lemma(self, prompt):
        tokens = self.nlp(prompt)
        res= []
        for token in tokens:
            if "V" in token.tag_:
                res.append(token.lemma_)
        return res

    def find_verb_phrase(self, prompt):
        if prompt.split(" ")[0] == "to":
            prompt = " ".join(prompt.split(" ")[1:])
        return prompt

    def find_verb_adj_noun_phrase(self, prompt, roles):
        '''
        sentence -> verb adj noun
        :param prompt:
        :return:
        '''
        res_v = dict()
        roles_c = roles.copy()
        roles_c += ['she', 'he', 'him', 'her', 'they', 'it']
        tokens = self.nlp(prompt)
        v = ''
        for i, token in enumerate(tokens):
            # print(token.text, token.tag_)
            if 'V' in token.tag_:
                v = self.verb_tense_gen(verb=token.text, tense=PAST, person=3, number=SG)
                res_v[v] = dict()
                res_v[v]['a'] = []
                res_v[v]['n'] = []
            elif v and 'NN' in token.tag_ and token.text not in roles_c:
                res_v[v]['n'].append((token.text, i))
            elif v and ('RB' in token.tag_ or 'JJ' in token.tag_):
                res_v[v]['a'].append((token.text, i))

        res = set()
        for v in res_v:
            if res_v[v]['n']:
                nouns = [n[0] for n in res_v[v]['n']]
                # res.add([' '.join(list(x)) for x in list(itertools.product([v], nouns))])
                res = res.union([' '.join(list(x)) for x in list(itertools.product([v], nouns))])
            if res_v[v]['a'] and not res_v[v]['n']:
                adjs = [n[0] for n in res_v[v]['a']]
                # print(adjs, [x for x in list(itertools.product([v], adjs))])
                res = res.union([' '.join(list(x)) for x in list(itertools.product([v], adjs))])
            if not res_v[v]['a'] and not res_v[v]['n']:
                if v not in useless_verbs:
                    res.add(v)

        # res = set()
        # for p in list(itertools.product(res_v, res_a, res_n)):
        #     p = [x for x in p if x]
        #     p = ' '.join(p)
        #     res.add(p)
        return res

    def noun_checker(self, phrase=None):
        tokens = self.nlp(phrase)
        for token in tokens:
            if "NN" in token.tag_:
                return True
        return False

    def verb_checker(self, phrase=None):
        tokens = self.nlp(phrase)
        for token in tokens:
            if "V" in token.tag_:
                return True
        return False

    def find_noun_lemma(self, words):
        tokens = self.nlp(words)
        for token in tokens:
            if "NN" in token.tag_:
                return token.lemma_
        return None

    def word_lemma(self, words):
        tokens = self.nlp(words)
        res = []
        for token in tokens:
            res.append(token.lemma_)
        return ' '.join(res) if res else ''

    def noun_phrase_finder(self, words, sentence):
        """
        Given a sentence, return phrases
        :param sentence: a hammer and a saw
        :return: [hammer, saw]
        """
        tokens = self.nlp(sentence)
        res = []
        res_single = ''
        for i, token in enumerate(tokens):
            # print(token, token.tag_)
            if token.text in words or token.lemma_ in words:
                if "NN" in token.tag_:
                    res_single += token.lemma_ + ' '
                else:
                    if res_single:
                        res.append(res_single.strip())
                        res_single = ''
                    else:
                        pass
                if res_single:
                    res.append(res_single.strip())
        return res if res else []

    def noun_phrase_finder(self, words, sentence):
        """
        Given a sentence, return phrases
        :param sentence: a hammer and a saw
        :return: [hammer, saw]
        """
        tokens = self.nlp(sentence)
        res = set()
        res_single = ''
        for i, token in enumerate(tokens):
            # print(token, token.tag_)
            if token.text in words or token.lemma_ in words:
                if "NN" in token.tag_:
                    res_single += token.lemma_ + ' '
                else:
                    if res_single:
                        res.add(res_single.strip())
                        res_single = ''
                    else:
                        pass
                if res_single:
                    res.add(res_single.strip())
        return list(res) if res else []

    def adj_finder(self, words, sentence):
        """
        find adj and adv in the words and also in the sentence
        :param words:
        :param sentence:
        :return:
        """
        tokens = self.nlp(sentence)
        res = []
        res_single = ''
        for i, token in enumerate(tokens):
            # print(token, token.tag_)
            if token.text in words or token.lemma_ in words:
                if "RB" in token.tag_ or "JJ" in token.tag_:
                    res_single += token.lemma_ + ' '
                else:
                    if res_single:
                        res.append(res_single.strip())
                        res_single = ''
                    else:
                        pass
                if res_single:
                    res.append(res_single.strip())
        return res if res else []

    def comet_to_delete(self, words, sentence):
        # print('COMET to delete', words, sentence)
        words = words.strip()
        if 'to ' == words[:3]:
            words = words[3:]
        words = words.replace('personx', '')
        words = words.replace('person x', '')
        words = words.replace('persony', 'others')
        words = words.replace('person y', 'others')
        words = words.strip()
        tense = self.tense_detect(sentence)
        tokens = self.nlp(words)
        res = ''
        for token in tokens:
            if 'V' in token.tag_:
                verb = self.verb_tense_gen(token.lemma_, tense, 3, 'singular')
                if verb == 'is':
                    pass
                else:
                    res += verb + ' '
            else:
                if 'PRP' in token.tag_ or 'NNP' in token.tag_:
                    pass
                else:
                    res += token.text + ' '
        return res.strip()

    def comet_to_delete_all_possible(self, words):
        words = words.strip()
        # if len(words.split(' ')) == 1:
        #     return words
        if 'to ' == words[:3]:
            words = words[3:]
        # words = words.replace('personx', '')
        # words = words.replace('persony', '')
        words = words.strip()
        tokens = self.nlp(words)
        res = []
        for token in tokens:
            # print('token', token, token.tag_)
            if 'V' in token.tag_:
                verb = self.verbs_all_tense(token.text)
                res.append(list(verb))
            # elif 'NN' in token.tag_ and 'char' not in token.text.lower():
            #     nouns = self.noun_all_tense(token.text)
            #     res.append(list(nouns))
            else:
                if 'PRP' in token.tag_: #or 'NNP' in token.tag_:
                    res.append([token.text, ''])
                else:
                    res.append([token.text])
        # print('res', res)
        all_words = []
        for tuple in [p for p in itertools.product(*res)]:
            words = ' '.join(tuple)
            if re.sub(r'[^\w]', ' ', words.strip()).strip() == 'none':
                pass
            else:
                all_words.append(re.sub(r'[^\w]', ' ', words.strip()).strip())
        return all_words

    def comet_to_entity(self, words):
        # print('COMET to delete', words, sentence)
        words = words.strip()
        if 'to ' == words[:3]:
            words = words[3:]
        words = words.lower().replace('personx', '')
        words = words.lower().replace('person x', '')
        words = words.lower().replace('persony', '')
        words = words.lower().replace('person y', '')
        words = words.strip()

        tokens = self.nlp(words)
        res = []
        previous = False
        for token in tokens:
            if 'NN' in token.tag_:
                if previous:
                    res[-1] += ' ' + token.text
                else:
                    res.append(token.text)
                previous = True
            else:
                previous = False
        return res

    def verbs_all_tense(self, verb):
        res = set()
        for tense in [INFINITIVE, PRESENT, PAST, FUTURE]:
            for person in [1,2,3]:
                for number in [SG, PL]:
                    v = conjugate(
                        verb=str(verb),
                        tense=tense,  # INFINITIVE, PRESENT, PAST, FUTURE
                        person=int(person),  # 1, 2, 3 or None
                        number=number,
                    )
                    if v:
                        res.add(v)  # SG, PL
        print(res)
        return res

    def noun_all_tense(self, noun):
        res = set()
        noun_p = pluralize(noun)
        if noun_p:
            res.add(noun_p)
        noun_s = singularize(noun)
        if noun_s:
            res.add(noun_s)
            res.add(referenced(noun_s))
        print(res)
        return res

    def noun_adj_adv_find(self, words, prompt):
        """
        Try to find the noun, adj and adv for verbatlas outputs
        :param prompt: sentence
        :return:
        """
        tokens = self.nlp(prompt)
        res = []
        prev = ('', '')
        outputs = []
        for token in tokens:
            if 'NN' in token.tag_:
                if 'NN' not in prev[-1] and token.lemma_ in words:  # noun 前面不是noun 比如 beautiful girl
                    res.append(token.lemma_)
                elif 'NN' in prev[-1] and token.lemma_ in words:  # noun phrase
                    if res:
                        res[-1] += ' ' + token.lemma_
                    else:
                        res.append(token.lemma_)
                else:
                    pass
            if 'JJ' in prev[-1] or 'RB' in prev[-1]:
                if 'NN' in token.tag_:
                    outputs.append((token.lemma_, prev[0]))
                elif 'NN' not in token.tag_ and token.lemma_ in words:
                    res.append(token.lemma_)

            prev = (token.lemma_, token.tag_)

        return res, outputs

    def read_pos(self, line):
        tokens = self.nlp(line)
        for t in tokens:
            print(t, t.pos_)

    def find_pron(self, line):
        tokens = self.nlp(line)
        res = []
        for t in tokens:
            if 'PRO' in t.pos_:
                res.append(t.text)
        return res

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--example", type=str, default="apple", help="a obj")
    args = parser.parse_args()
    nltk_gen = nltk_gen()
    # prompt = args.example
    words = "saw"
    prompt = ' [Char_1] loves ice cream.'
    words = '[Char_1] is learning maths, a good girl of his'
    # print(nltk_gen.verb_tense_gen('loves', "past", 3, "singular"))
    # print(nltk_gen.noun_adj_adv_find(words, prompt))
    # print(nltk_gen.comet_to_delete(words, prompt))
    # print(nltk_gen.read_pos(prompt))
    # print(nltk_gen.find_pron(prompt))
    # print(nltk_gen.comet_to_entity(words='ice cream'))
    # print(nltk_gen.noun_phrase_finder(words='to get ice cream and cake', sentence='to get ice cream and cake'))
    print(nltk_gen.find_v_N(text='go to the pharmacy'))
    # print(nltk_gen.find_verb_adj_noun_phrase(prompt='She got married', roles=['Jenny', 'she']))

