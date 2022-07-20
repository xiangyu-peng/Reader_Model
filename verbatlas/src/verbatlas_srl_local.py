import argparse
import csv
import json
import sys
from allennlp_models.pretrained import load_predictor
from transformer_srl import dataset_readers, models, predictors

class SRL(object):
    """
    SRL labelling
    structured-prediction-srl, structured-prediction-srl-bert
    """

    def __init__(self, model_id):
        # self.predictor = load_predictor(model_id)
        self.predictor = predictors.SrlTransformersPredictor.from_path(model_id)

    def forward(self, sentence):
        """
        Return
        :param sentence:
        :return:
        """
        srl_outputs = self.predictor.predict(sentence=sentence)
        return srl_outputs


class VerbClassExtractor(object):
    """
    Verb extraction from VerbAtlas using an SRL model that generated propbank tags
    """
    def __init__(self, pb_v_path, frame_id_vatlas_path=None, vt='verbatlas'):
        """

        """
        self.vt = vt
        if self.vt == 'verbatlas':
            self.pb_to_v = self._read_pk_to_vatlas(pb_v_path, frame_id_vatlas_path)
        else:
            self.pb_to_v = json.load(open(pb_v_path))

    def _read_pk_to_vatlas(self, filename, frame_id_va_path):
        tsv_file = open(frame_id_va_path)
        read_tsv = csv.reader(tsv_file, delimiter='\t')
        frames_id = {}
        line_count = 0
        for row in read_tsv:
            if line_count == 0:
                line_count += 1
                continue
            else:
                frames_id[row[0]] = row[1]
        tsv_file = open(filename)
        read_tsv = csv.reader(tsv_file, delimiter='\t')
        line_count = 0
        rows = {}
        for row in read_tsv:
            if line_count == 0:
                line_count += 1
                continue
            else:
                ids = row[0].split('>')
                vargs = []
                for i in range(1, len(row)):
                    arg = row[i].split('>')
                    vargs.append(arg)
                va_dict = {'vatlas_id': ids[1], 'vaclass': frames_id[ids[1]], 'args': vargs}
                if ids[0] in rows:
                    rows[ids[0]].append(va_dict)  # This line doesn't run with VerbAtlas,
                    # for every VerbAtlas class there is only one PropBank frame
                else:
                    rows[ids[0]] = [va_dict]
                line_count += 1
        # print(f'Processed {line_count} lines.')
        return rows

    def convert_verbs(self, ex_srl):
        vatlas_frames = []
        for verb in ex_srl['verbs']:
            # print('verb', verb)
            v_obj = {}
            frame = verb['frame']
            v_obj['lemma'] = verb['lemma']
            try:
                tags = verb['tags']
                arg_tags = []
                tmp_b = None
                tmp_i = None
                for t in range(len(tags)):
                    if tags[t] != 'B-V' and 'B-' in tags[t]:
                        tmp_b = t
                        tmp_i = t + 1
                    if 'I-' in tags[t]:
                        tmp_i = t + 1
                    if not (tmp_i is None) and not (tmp_b is None) and (t+1 == len(tags) or ('I-' not in tags[t + 1])):
                        arg_tags.append((tmp_b, tmp_i))
                        tmp_i = None
                        tmp_b = None
                for vat in self.pb_to_v[frame]:
                    if self.vt == "verbatlas":
                        v_obj[self.vt] = vat['vaclass']
                        v_obj['vatlas_id'] = vat['vatlas_id']
                    else:
                        v_obj[self.vt] = vat['vnclass']
                    args_words = {t[0][1]: " ".join(ex_srl['words'][t[1][0]:t[1][1]]) for t in
                                  zip(vat['args'], arg_tags)}
                    v_obj['args_words'] = args_words
                vatlas_frames.append(v_obj)
            except KeyError:
                print(f"{frame}' PropBank frame is unknown to {self.vt}.")

        return vatlas_frames


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path",
                        type=str,
                        help="model path: fine tuned structured-prediction-srl-bert path",
                        default="../models/srl-bert-verbatlas")
    parser.add_argument("--verb_source",
                        type=str,
                        help="va_fi: [verbAtlass , verbnet]",
                        default="verbatlas")
    parser.add_argument("--pb_va",
                        type=str,
                        help="pb_va: PropBank VerbAtlas Mapping file path",
                        default="../data/verbatlas/pb2va.tsv")
    parser.add_argument("--pb_vn",
                        type=str,
                        help="pb_vn: PropBank VerbNet Mapping file path",
                        default="../data/pb-vn2.json")
    parser.add_argument("--va_fi",
                        type=str,
                        help="va_fi: VerbAtlass class names to ids file path",
                        default="../data/verbatlas/VA_frame_ids.tsv")
    args = parser.parse_args()

    # 5. Uncomment to use a SRL.
    srl_model = SRL(args.model_path)
    sentence = "[Char_1] drove home at night."  # Change to the desired sentence.
    print("=" * 50)
    print(f"sentence = {sentence}")
    print("=" * 50)
    outputs = srl_model.forward(sentence=sentence)
    print(outputs)
    print("=" * 50)
    pb_va_path = ""
    verbatlas_extractor = VerbClassExtractor(args.pb_va, args.va_fi, args.verb_source)
    vatlas_parse = verbatlas_extractor.convert_verbs(outputs)
    print('---' * 20)
    print(f"VerbAtlas = {vatlas_parse}")
    print("=" * 50)

    # Experimental: Trying with Verbnet.
    # verbnet_extractor = VerbClassExtractor(args.pb_vn, vt='verbnet')
    # verbnet_parse = verbnet_extractor.convert_verbs(outputs)
    # print(f"VerbNet = {verbnet_parse}")
    # print("=" * 50)