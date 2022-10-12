# Reader Model

This code accompanies the paper [Guiding Neural Story Generation with Reader Models](https://arxiv.org/abs/2112.08596)
<img src="README_images/RM_figure1.png" width="300">

## :boom: Installation
1. Virtual env: `conda env create -f environment.yml python=3.6`
2. Install the [COMeT](https://github.com/atcbosselut/comet-commonsense)
    - Follow the instruction in the COMET original repo.
    - There are some extra packages you need to install here.
    ```ruby
    conda install -c pytorch pytorch
    pip install ftfy
    ```
    - First, download the pretrained models from [here](https://drive.google.com/open?id=1FccEsYPUHnjzmX-Y5vjCBeyRt1pLo8FB).

    - Then untar the file:
            
    ```ruby
    tar -xvzf pretrained_models.tar.gz
    ```

    - Then run the following script to interactively generate arbitrary ATOMIC event effects:
        
    ```ruby
      python scripts/interactive/atomic_single_example.py --model_file pretrained_models/atomic_pretrained_model.pickle
    ```
 
    - Some absolute paths have to be changed after you clone the repo `./comet/comet-commonsense/src/interactive/functions.py`.    

    - How to use COMeT to convert sentence as knowledge graph?   

    ```ruby
    python KG_gen/KG_gen.py
    ``` 
        
3. Install verbatlas SRL
    - The file is located at `./verbatlas/src`
    - Installation follows [here](https://github.com/xiangyu-peng/Reader_Model/tree/master/verbatlas)
4. No need to prepare for verbnet parser. Location: `./semparse-core/`
5. Install allennlp and allennlp-model.
6. Install new comet-atomic 2020 from AI2.

### Possible Bugs!!!
1. Pattern cannot used in python 3.7
[Solution](https://github.com/clips/pattern/issues/282)

2. No module named '_ctypes'

```ruby
sudo apt-get install libffi-dev
```

3. When run SRL, get ` __init__() got an unexpected keyword argument 'max_n'`

```ruby
pip uninstall py-rouge
pip install py-rouge
```

## Pipeline for RM in KG-gen

<img src="README_images/RM-pipeline.png" width="750">

```ruby
cd verbatlas/src&&gunicorn --workers 4 --bind 0.0.0.0:5000 wsgi:app
cd verbnet-approach/src&&python main.py --trainer_type KG_diff_file --topk 2 --story_length 3 --KG_use --remain_div
cd verbnet-approach/src&&python main.py --trainer_type KG_diff_step_file --topk 5 --story_length 10 --KG_use --remain_div --look_ahead 1 --seed 3
```

### :heavy_heart_exclamation: KG / State Transition
Given a state, RM will output action space and then human choose an action, and RM will generate a new state.
This is the code you need to run in the command line. However, there might need some modification for the absolute path.
I have tried my best to convert all the absolute path to relative ones. Pls LMK if there still exist any absolute path. Thanks.

#### :sweat_drops: Functions

For generating action space and new state, the main file is [here](verbnet-approach/src/reader_model_kg_step.py).     
- `define_roles()`: Find role/ character names in the sentence.
- `prepare_next_step()`: Use the parser, construct or add nodes in KG
- `generate_actions()`: Given the KG and then generate all the possible action space.
- `forward()`: Different round will trigger different functions. It will be called in [main.py](verbnet-approach/src/main.py) and move one sentence forward.

#### :star_struck: Features
* `trainer_type`: `QA` or `generation` or `KG` or `KG_diff` or `KG_diff_file`, We are using `KG_diff_file` as default.
* `model_name_or_path`: GPT-2's pretrained model path
* `mapping_file` and `verbnet_json`: path of verbnet files.
* `target_path`: path to save kg visualization
* `model_file`: COMeT path
* [BART](https://huggingface.co/transformers/v2.11.0/model_doc/bart.html) is used to fill in the verbs. The model is `facebook/bart-large`
* GPT-2 finetuned on ROC is used to generate the sentence after we have verb and noun.
* `--KG_use`: whether to use KG to update the game state
* `--topk`: int, # of sentence for each round we remain.
* `--action_standard`: Use verb or sentence as actions. Now we are using `sentence` as default.
* `--remain_div`: half of the candidates have different prompts
* `--story_length`: length of the generated story
* `--look_ahead`: only apply to `trainer_type=KG_diff_step_file` or `trainer_type=KG_diff_step`, `m_1` in the paper.
