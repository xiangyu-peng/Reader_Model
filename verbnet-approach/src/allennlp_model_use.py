import argparse
from allennlp_models.pretrained import load_predictor
from collections import defaultdict
class SRL(object):
    """
    SRL labelling
    structured-prediction-srl, structured-prediction-srl-bert
    """
    def __init__(self, model_id):
        self.predictor = load_predictor(model_id)

    def forward(self, sentence):
        """
        Return
        :param sentence:
        :return:
        """
        outputs = self.predictor.predict(sentence=sentence)
        return outputs


class Tagging(object):
    """
    Named Entity Recognition with Transformer
    Find the person/char/role names
    model_id = [tagging-fine-grained-transformer-crf-tagger, tagging-fine-grained-crf-tagger, tagging-elmo-crf-tagger]
    """
    def __init__(self, model_id):
        self.predictor = load_predictor(model_id)

    def forward(self, sentence):
        """
        Return the Person names.
        :param sentence:
        :return:
        """
        print('sentence', sentence)
        outputs = self.predictor.predict(sentence=sentence)
        tags = outputs['tags']
        print(tags)
        words = outputs['words']
        # Only remain the one with tag - PERSON
        res = []
        for idx, word in enumerate(words):
            if 'PER' in tags[idx]:
                res.append(word)
        return res

class MC(object):
    """
    Multiple Choice.
    model_id = [mc-roberta-commonsenseqa, mc-roberta-piqa, mc-roberta-swag]
    """
    def __init__(self, model_id):
        self.predictor = load_predictor(model_id)

    def forward(self, prefix, alternatives):
        outputs = self.predictor.predict(prefix=prefix, alternatives=alternatives)
        return alternatives[outputs['best_alternative']]

    def forward_output(self, prefix, alternatives):
        return self.predictor.predict(prefix=prefix, alternatives=alternatives)

class QA(object):
    """
    QA.
    model_id = [rc-bidaf-elmo, rc-bidaf, rc-transformer-qa]
    """
    def __init__(self, model_id):
        self.predictor = load_predictor(model_id)

    def forward(self, passage, question):
        outputs = self.predictor.predict(passage=passage, question=question)
        return outputs['best_span_str']

class CR(object):
    """
    Coreference.
    model_id = coref-spanbert
    """
    def __init__(self, model_id):
        self.predictor = load_predictor(model_id)

    def forward(self, sentence):
        outputs = self.predictor.predict(document=sentence)
        # print(outputs)
        clusters = outputs['clusters']
        document = outputs['document']

        if not clusters:
            return sentence
        n = 0
        doc = {}
        for i, obj in enumerate(document):
            doc.update({n: obj})  # what I'm doing here is creating a dictionary of each word with its respective index, making it easier later.
            n = n + 1
        print(doc)
        for cluster in clusters:
            cluster = [i[0] for i in cluster]
            for c in cluster[1:]:
                doc[c] = doc[cluster[0]]
        return ' '.join([doc[i] for i in range(len(doc))]).replace(' .', '.')
        # clus_all = []
        # cluster = []
        # clus_one = {}
        # for i in range(0, len(clusters)):
        #     one_cl = clusters[i]
        #     for count in range(0, len(one_cl)):
        #         obj = one_cl[count]
        #         for num in range((obj[0]), (obj[1] + 1)):
        #             for n in doc:
        #                 if num == n:
        #                     cluster.append(doc[n])
        #     clus_all.append(cluster)
        #     cluster = []
        # print(clus_all)  # And finally, this shows all coreferences
        # return outputs['best_span_str']
    def check_char_num(self, story):
        outputs = self.predictor.predict(document=story)
        clusters = outputs['clusters'] = outputs['clusters']
        return len(clusters)


class TE(object):
    """
    Textual Entailment.
    model_id = [pair-classification-roberta-mnli, pair-classification-roberta-snli]
    """
    def __init__(self, model_id):
        self.predictor = load_predictor(model_id)

    def forward_inferences(self, premise_inferences, hypothesis):
        for rel in premise_inferences:
            for premise in premise_inferences[rel]:
                # print('p => ', premise, 'h => ', hypothesis)
                if premise != 'none' and self.forward(premise, hypothesis) == -1:
                    # print('Contradict')
                    return False
        return True

    def forward(self, premise, hypothesis):
        """
        Check whether hypothesis follows premise.
        :param premise: sentence.
        :param hypothesis: sentence.
        :return: int. -1 means contradiction; 0 means neural, and 1 means entailment.
        """
        outputs = self.predictor.predict(premise=premise, hypothesis=hypothesis)

        # According to outputs, produce the label with [-1, 0 ,1]
        # -1 means contradiction; 0 means neural, and 1 means entailment.
        if outputs['label'] == 'contradiction':
            return -1
        elif outputs['label'] == 'neutral':
            return 0
        else:  # entailment
            return 1

    def forward_output(self, premise, hypothesis):
        """
        Generate the original output from model.
        :param premise: sentence.
        :param hypothesis: sentence.
        :return: a dict. keys ='logits' /'probs'....
        """
        outputs = self.predictor.predict(premise=premise, hypothesis=hypothesis)
        return outputs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id",
                        type=str,
                        help="model id: [pair-classification-roberta-mnli, pair-classification-roberta-snli]",
                        default="tagging-elmo-crf-tagger")
    args = parser.parse_args()

    # 1. Uncomment to use textual entailment.

    # te_model = TE(args.model_id)
    #
    # premise = "to work."
    # hypothesis = "Jenny is now a retiree."
    #
    # print("="*50)
    # print(f"premise = {premise}")
    # print(f"hypothesis = {hypothesis}")
    # print("="*50)
    # outputs = te_model.forward_output(premise=premise, hypothesis=hypothesis)
    # print(outputs)
    # print("="*50)
    # print(f"===== {args.model_id} prediction: {outputs['label']} with {max(outputs['probs'])} probability")

    # 2. Uncomment to use QA model.
    # qa_model = QA(args.model_id)
    #
    # passage = "John studied very hard for the test. He drove to his school with his things. He went to the test room. He finished the test."
    # question = "Where is John?"
    #
    # print("=" * 50)
    # print(f"passage = {passage}")
    # print(f"question = {question}")
    # print("=" * 50)
    # outputs = qa_model.forward(passage=passage, question=question)
    # print(outputs)
    # print("=" * 50)
    # print(f"===== {args.model_id} prediction: {outputs['best_span_str']}")

    # 3. Uncomment to use a MC model.
    # mc_model = MC(args.model_id)
    #
    # prefix = "Jenny lived in Florida. Jenny hear?"
    # alternatives = ["earthquake", "noises of snakes", 'the floors shook']
    #
    # print("=" * 50)
    # print(f"prefix = {prefix}")
    # print(f"alternatives = {alternatives}")
    # print("=" * 50)
    # outputs = mc_model.forward_output(prefix=prefix, alternatives=alternatives)
    # print(outputs)
    # print("=" * 50)
    # print(f"===== {args.model_id} prediction: {mc_model.forward(prefix=prefix, alternatives=alternatives)}")

    # 4. Uncomment to use a Tagging model
    # tag_model = Tagging(args.model_id)
    #
    # sentence = "I love swim."
    #
    # print("=" * 50)
    # print(f"sentence = {sentence}")
    # print("=" * 50)
    # outputs = tag_model.forward(sentence=sentence)
    # print(outputs)
    # print("=" * 50)

    # 5. Uncomment to use a SRL.
    # srl_model = SRL(args.model_id)
    # sentence = "Karen was assigned a roommate in her first year of college."
    # print("=" * 50)
    # print(f"sentence = {sentence}")
    # print("=" * 50)
    # outputs = srl_model.forward(sentence=sentence)
    # print(outputs)
    # print("=" * 50)

    # 6. Uncomment to use a SRL.
    args.model_id = 'coref-spanbert'
    cr_model = CR(args.model_id)
    sentence = "Karen falls in love with Susan. She loves her."
    print("=" * 50)
    print(f"sentence = {sentence}")
    print("=" * 50)
    outputs = cr_model.check_char_num(story=sentence)
    print(outputs)
    print("=" * 50)

