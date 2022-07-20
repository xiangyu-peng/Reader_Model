from flask import Flask, request, jsonify
import time
from verbatlas_srl_local import SRL, VerbClassExtractor
app = Flask(__name__)
app.debug = True
# SRL_args = {'model_id' : '/media/becky/ReaderModel/verbatlas/models/srl-bert-verbatlas'}
# VerbClassExtractor_args = {'pb_v_path' : '../../verbatlas/data/verbatlas/pb2va.tsv',
#                            'frame_id_vatlas_path' : '../../verbatlas/data/verbatlas/VA_frame_ids.tsv'}
srl_model = SRL('/media/becky/ReaderModel/verbatlas/models/srl-bert-verbatlas')
verbatlas_extractor = VerbClassExtractor(pb_v_path='/media/becky/ReaderModel/verbatlas/data/verbatlas/pb2va.tsv',
                                         frame_id_vatlas_path='/media/becky/ReaderModel/verbatlas/data/verbatlas/VA_frame_ids.tsv',
                                         )

@app.route('/', methods=['POST'])
def result():
    if request.method == 'POST':
        data = request.data.decode('utf-8')
        # print('data =>', request.data.decode('utf-8'))
        outputs = srl_model.forward(sentence=data)
        # print('outputs', outputs)
        vatlas_parse = verbatlas_extractor.convert_verbs(outputs)
        # print('vatlas_parse ===>', vatlas_parse)
        ov = {'vatlas_parse': vatlas_parse}
        return jsonify(ov), 201

if __name__ == '__main__':
    app.run(host='0.0.0.0')