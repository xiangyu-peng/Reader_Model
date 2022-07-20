#python main.py --trainer_type KG_diff_file --remain_div --KG_use --topk 6 --story_length 6
from reader_model_trainer import Reader_Model_Trainer
from reader_model_kg_gen import Reader_Model_Trainer_KG
from reader_model_QA import Reader_Model_QA_Trainer
from baseline_trainer import Baseline_Trainer
from reader_model_kg_diff_trainer_main import Reader_Model_Trainer_KG_diff
from reader_model_kg_diff_decode import Reader_Model_Trainer_KG_diff_decode
from reader_model_kg_step import Reader_Model_Trainer_KG_Step
import argparse
import torch
import numpy as np
import sys
import os
import gc  # clear cache
import time
# os.environ["CUDA_VISIBLE_DEVICES"] = '2'

# # log info
# handler = logging.StreamHandler()
# formatter = OneLineExceptionFormatter(logging.BASIC_FORMAT)
# handler.setFormatter(formatter)
# root = logging.getLogger()
# root.setLevel(os.environ.get("LOGLEVEL", "INFO"))
# root.addHandler(handler)
from logging_info import log_file_create
logger = log_file_create('logs/logging_info.log')

from transformers import (
    CTRLLMHeadModel,
    CTRLTokenizer,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    OpenAIGPTLMHeadModel,
    OpenAIGPTTokenizer,
    TransfoXLLMHeadModel,
    TransfoXLTokenizer,
    XLMTokenizer,
    XLMWithLMHeadModel,
    XLNetLMHeadModel,
    XLNetTokenizer,
)

MODEL_CLASSES = {
    "gpt2": (GPT2LMHeadModel, GPT2Tokenizer),
    "ctrl": (CTRLLMHeadModel, CTRLTokenizer),
    "openai-gpt": (OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    "xlnet": (XLNetLMHeadModel, XLNetTokenizer),
    "transfo-xl": (TransfoXLLMHeadModel, TransfoXLTokenizer),
    "xlm": (XLMWithLMHeadModel, XLMTokenizer),
}


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


def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # GPT-2-generate
    parser.add_argument(
        "--model_type",
        default="gpt2",
        type=str,
        required=False,
        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--model_name_or_path",
        default="../gpt-2/roc_char",
        type=str,
        required=False,
        help="Path to pre-trained model or shortcut name selected in the list: "
        + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--length", type=int, default=20)
    parser.add_argument(
        "--stop_token",
        type=str,
        default=None,
        help="Token at which text generation is stopped",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="temperature of 1.0 has no effect, lower tend toward greedy sampling",
    )
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.0,
        help="primarily useful for CTRL model; in that case, use 1.2",
    )
    parser.add_argument("--k", type=int, default=0)
    parser.add_argument("--p", type=float, default=0.9)
    parser.add_argument(
        "--prefix", type=str, default="", help="Text added prior to input."
    )
    parser.add_argument(
        "--padding_text",
        type=str,
        default="",
        help="Deprecated, the use of `--prefix` is preferred.",
    )
    parser.add_argument(
        "--xlm_language",
        type=str,
        default="",
        help="Optional language when used with the XLM model.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed for initialization"
    )
    parser.add_argument(
        "--no_cuda", action="store_true", help="Avoid using CUDA when available"
    )
    parser.add_argument(
        "--num_return_sequences",
        type=int,
        default=1,
        help="The number of samples to generate.",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument("--device_id", type=int, default=2, help="gpu id")

    # allennlp
    parser.add_argument(
        "--Tagging_model_id",
        type=str,
        choices=[
            "tagging-fine-grained-transformer-crf-tagger",
            "tagging-fine-grained-crf-tagger",
            "tagging-elmo-crf-tagger",
        ],
        default="tagging-elmo-crf-tagger",
        help="allenlp model id - tagging.",
    )
    parser.add_argument(
        "--CR_model_id",
        type=str,
        choices=[
            "coref-spanbert"
        ],
        default="coref-spanbert",
        help="allenlp model id - Coreference.",
    )
    parser.add_argument(
        "--MC_model_id",
        type=str,
        choices=["mc-roberta-commonsenseqa", "mc-roberta-piqa", "mc-roberta-swag"],
        default="mc-roberta-commonsenseqa",
        help="allenlp model id - Mutliple choice.",
    )
    parser.add_argument(
        "--TE_model_id",
        type=str,
        choices=[
            "pair-classification-roberta-mnli",
            "pair-classification-roberta-snli",
        ],
        default="pair-classification-roberta-mnli",
        help="allenlp model id - Textual Entailment.",
    )
    # verbnet
    parser.add_argument(
        "--ignore_affector",
        type=bool,
        default=False,
        help="Whether to include the entity/person who did the relation",
    )
    parser.add_argument(
        "--mapping_file",
        type=str,
        default="../data/mapping_v.p",
        help="The path of mapping file for verbnet",
    )
    parser.add_argument(
        "--verbnet_json",
        type=str,
        default="../data/verbnet.json",
        help="The path of json file for verbnet",
    )
    parser.add_argument(
        "--target_path",
        type=str,
        default="../KG_plot/KG",
        help="path to save kg visualization",
    )

    # Comet
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--model_file_conceptnet",
        type=str,
        default="../../comet-commonsense/pretrained_models/conceptnet_pretrained_model.pickle",
    )
    parser.add_argument(
        "--model_file_atomic",
        type=str,
        default="../../comet-commonsense/pretrained_models/atomic_pretrained_model.pickle",
    )
    parser.add_argument("--sampling_algorithm", type=str, default="greedy")
    parser.add_argument(
        "--relations_needed",
        type=str,
        default="all",
        help="# of relations needed in KG.",
    )
    parser.add_argument(
        "--comet_type", type=str, default="atomic", help="Use atomic or conceptnet"
    )

    # Trainer
    parser.add_argument(
        "--trainer_type", type=str, default="gen", help="QA or gen or baseline or KG or KG_diff"
    )
    parser.add_argument(
        "--number_GPT_obj",
        type=int,
        default=4,
        help="# of obj generated by GPT-2 for choosen by MC model.",
    )
    parser.add_argument(
        "--rocnli_path",
        type=str,
        default="../../entailment/epoch_3_lr_5e-06_#_631980_0406-203629.pt",
        help="RocNLI trained path",
    )
    parser.add_argument(
        "--TEorMC", type=str, default="", help="use TE or MC or S to do the generation."
    )
    parser.add_argument(
        "--TE_loop_limit",
        type=int,
        default=5,
        help="limit of sentences generated to find TE.",
    )
    parser.add_argument(
        "--TE_st_threshold_upper",
        type=float,
        default=0.7,
        help="smaller than this threshold, use sentence transformer to find obj in neural sets after TE. ",
    )
    parser.add_argument(
        "--TE_st_threshold_limit",
        type=float,
        default=0.3,
        help="larger than this threshold, use sentence transformer to find obj in neural sets after TE. ",
    )
    parser.add_argument(
        "--obj_num",
        type=int,
        default=50,
        help="number of obj generated from GPT-2 to comprise obj candidate set",
    )

    # verb generator
    parser.add_argument(
        "--verb_gen_mode",
        default="roc",
        type=str,
        help="which dataset we should use, roc / scifi / combine",
    )
    parser.add_argument(
        "--roc_pairs_path",
        default="../data/roc_verb_pairs.json",
        type=str,
        help="verb pair prob json file path -- roc",
    )
    parser.add_argument(
        "--scifi_pairs_path",
        default="../data/scifi_verb_pairs.json",
        type=str,
        help="verb pair prob json file path -- scifi",
    )
    parser.add_argument(
        "--combine_pairs_path",
        default="../data/combine_verb_pairs.json",
        type=str,
        help="verb pair prob json file path -- scifi and roc",
    )
    parser.add_argument(
        "--combine_pairs_count_path",
        default="../data/combine_verb_counts.json",
        type=str,
        help="verb pair counts json file path -- scifi and roc",
    )
    parser.add_argument(
        "--train_file_path",
        default="../data/rocstory_edit.txt",
        type=str,
        help="the path for rocstory file.",
    )
    parser.add_argument(
        "--save_path",
        default="../data/rocstory_generate_plot.txt",
        type=str,
        help="the path for saving rocstory file.",
    )
    parser.add_argument(
        "--story_length", default=5, type=int, help="the length of story."
    )
    parser.add_argument(
        "--roberta_path",
        default="../../fill_in_mask/roberta-finetuned-on-fairytale.pt",
        type=str,
        help="path of roberta fill in mask pt.",
    )
    parser.add_argument(
        "--action_standard",
        default="sentence",
        type=str,
        help="verb or sentence",
    )

    # local verbatatlas setting
    parser.add_argument(
        "--SRL_local_path",
        type=str,
        help="model path: fine tuned structured-prediction-srl-bert path",
        default="../../verbatlas/models/srl-bert-verbatlas",
    )
    parser.add_argument(
        "--verb_source",
        type=str,
        help="va_fi: [verbAtlass , verbnet]",
        default="verbatlas",
    )
    parser.add_argument(
        "--pb_va",
        type=str,
        help="pb_va: PropBank VerbAtlas Mapping file path",
        default="../../verbatlas/data/verbatlas/pb2va.tsv",
    )
    parser.add_argument(
        "--pb_vn",
        type=str,
        help="pb_vn: PropBank VerbNet Mapping file path",
        default="../../verbatlas/data/pb-vn2.json",
    )
    parser.add_argument(
        "--va_fi",
        type=str,
        help="va_fi: VerbAtlass class names to ids file path",
        default="../../verbatlas/data/verbatlas/VA_frame_ids.tsv",
    )

    # whether to use KG in topk prunning
    parser.add_argument(
        "--KG_use", action="store_true", help="Whether to use KG track the state"
    )
    parser.add_argument(
        "--topk", type=int, help="# of the sentences we remain each time", default=2
    )

    # stopwords path
    parser.add_argument(
        "--stopwords_path",
        default="../data/stopwords.txt",
        type=str,
        required=False,
        help="path of stopwords",
    )

    # file
    parser.add_argument(
        "--story_file_path",
        default="../data/survey/prompt_ending.txt",
        type=str,
        required=False,
        help="path of stopwords"
    )

    # diversity
    parser.add_argument(
        "--remain_div",
        action="store_true",
        help="whether to control div",
    )
    parser.add_argument(
        "--early_stop_dis",
        type=float,
        default=0.9,
        help="whether to control div",
    )
    parser.add_argument(
        "--score_patience",
        type=float,
        default=0.02,
        help="after score_patience_step, the score should at least go patience up",
    )
    parser.add_argument(
        "--score_patience_step",
        type=int,
        default=3,
        help="after score_patience_step, the score should at least go patience up",
    )

    # new decode system
    parser.add_argument(
        "--look_ahead",
        type=int,
        default=2,
        help="# of nodes ahead",
    )

    #scripts
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="# of gpu",
    )
    parser.add_argument(
        "--prompt_text",
        type=str,
        default=None,
        help="natural language sentence",
    )
    parser.add_argument(
        "--goal_text",
        type=str,
        default=None,
        help="natural language sentences goal",
    )

    args = parser.parse_args()
    # args.device = torch.device(
    #     "cuda:" + str(args.device_id) if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    set_seed(args)
    print(args)

    if args.trainer_type == "gen":
        prompt_text = "Anna went to the salon."
        rm_trainer = Reader_Model_Trainer(args=args, prompt_text=prompt_text)
        rm_trainer.define_roles()
        print(rm_trainer.KG_RM.graph_state.edges)
        action, obj, prompt = rm_trainer.forward()
        print("action is =>", action)
        print("obj is =>", obj)
        print("prompt is =>", prompt)
        print(rm_trainer.KG_RM.graph_state.edges)
        print("=" * 30)
        action, obj, prompt = rm_trainer.forward()
        print("action is =>", action)
        print("obj is =>", obj)
        print("prompt is =>", prompt)
        print(rm_trainer.KG_RM.graph_state.edges)
        # print('action is =>', rm_trainer.forward())
        # print('action is =>', rm_trainer.forward())
        # rm_trainer.run_file()

    elif args.trainer_type == "QA":  # QA
        rm_trainer = Reader_Model_QA_Trainer(args=args)
        prompt_text = [
            "Jenny woke up in the morning at home.",
            "Jenny brushed her teeth.",
        ]
        for sentence in prompt_text:
            rm_trainer.forward(sentence=sentence)

    elif args.trainer_type == "KG":
        prompt_text = "Judy went to school."
        rm_trainer = Reader_Model_Trainer_KG(args=args, prompt_text=prompt_text)
        rm_trainer.define_roles(prompt_text)

        # define the goal
        goal_text = """
        Judy went to school.
        Judy took a test.
        She answered all the questions.        
        """
        outputs = rm_trainer.prepare_next_step(picked_action=goal_text, goal_or_build='goal')
        print("000. GOAL state", rm_trainer.KG_goal.visualize(order=0))
        #prepare for generation
        outputs, nodes_consider = rm_trainer.prepare_next_step(goal_or_build='build', goal_word='reciprocate-112-1')

        if args.KG_use:
            print("111. initial state", rm_trainer.KG_RM.visualize(order=0))
            rm_trainer.graph_distance(sentence='Eric ate way too much ice cream growing up.')

        round_num = 1
        for num in range(1, round_num + 1):
            print("=" * 5 + str(num) + "=" * 5)
            action, cs_entity_KG_dict = rm_trainer.generate_actions(outputs, nodes_consider)
            print("CONCEPTNET", cs_entity_KG_dict)
            print('action =>', action)
            actions_lst = rm_trainer.top_k_prunning(actions_dicts = [action], goal_word='admire-31.2', distance_type='graph')
            # action_chosen = [(key, [action[key][0]]) for key in action][0]
            print("Action list is =>", actions_lst)
            action_chosen = actions_lst[0]
            print("Action is =>", action_chosen)
            outputs, nodes_consider = rm_trainer.prepare_next_step(
                action_chosen[0], action_chosen[1][0], cs_entity_KG_dict
            )
            if args.KG_use:
                print("New state", rm_trainer.KG_RM.visualize(order=num))

        print(rm_trainer.prompt)

    elif args.trainer_type == 'KG_diff_decode':
        # with HiddenPrints():
        prompt_text = "Char_1 loved ice cream."
        # prompt_text = 'John is sleepy.'
        rm_trainer = Reader_Model_Trainer_KG_diff_decode(args=args, prompt_text=prompt_text)
        rm_trainer.define_roles(prompt_text)

        # define the goal
        goal_text = """
                    Char_1 loved ice cream.
                    Char_1 got fat.
                    """
        # goal_text = """
        #             John is sleepy.
        #             He starts a pot of coffee.
        #             John puts cream and sugar in his cup.
        #             He then goes to work.
        #             """
        outputs = rm_trainer.prepare_next_step(picked_action=goal_text, goal_or_build='goal', goal_word='run-51.3.2-1')

        print("000. GOAL state", rm_trainer.KG_goal.visualize(order=0))
        print('rm_trainer', rm_trainer.goal_word)
        # with HiddenPrints():
        text_lst = None
        for i in range(3):
            _, text_lst = rm_trainer.forward(round=i, text_lst=text_lst)


    elif args.trainer_type == 'KG_diff_step_file':
        with open(args.story_file_path, "r") as f:
            for i, story in enumerate(f):
                roles = story.split('**')[0].split(',')
                story = story.split('**')[1]
                prompt_text = story.split('.')[0] + '.'
                if i == 0:
                    rm_trainer = Reader_Model_Trainer_KG_Step(args=args, prompt_text=prompt_text)
                rm_trainer.define_roles(prompt_text)
                rm_trainer.roles = roles
                goal_text = story.split('.')[-2] + '.'
                rm_trainer.forward(goal=True, goal_sentence=goal_text)
                print("000. GOAL state", rm_trainer.KG_goal.visualize(order=0))
                # print('rm_trainer', rm_trainer.goal_word)

                # with HiddenPrints():
                text_lst = None
                for j in range(args.story_length):
                    final_chance = True if j == args.story_length -1 else False
                    text_lst = rm_trainer.forward(round=j, text_lst=text_lst, final_chance=final_chance)
                    dis_lst_max = [x for x in text_lst if x[-2] == 1 and not x[-1]]
                    if dis_lst_max:
                        break

                rm_trainer.save_file()
                # clear everything!
                rm_trainer.clear()


    elif args.trainer_type == 'KG_diff_step':
        # with HiddenPrints():
        prompt_text = args.prompt_text.replace('_', ' ')
        # prompt_text = 'John is sleepy.'
        rm_trainer = Reader_Model_Trainer_KG_Step(args=args, prompt_text=prompt_text)
        rm_trainer.define_roles(prompt_text)
        rm_trainer.roles = ['David', 'he']

        # define the goal
        goal_text = args.goal_text.replace('_', ' ')
        # goal_words = ['weight', 'lose']
        # outputs = rm_trainer.prepare_next_step(picked_action=goal_text, goal_or_build='goal', goal_word='run-51.3.2-1')

        rm_trainer.forward(goal=True, goal_sentence=goal_text)

        print("000. GOAL state", rm_trainer.KG_goal.visualize(order=0))
        print('rm_trainer', rm_trainer.goal_word)

        # with HiddenPrints():
        text_lst = None
        text_lst = rm_trainer.forward(round=0, text_lst=text_lst)
        # for i in range(1, args.story_length):
        #     text_lst = rm_trainer.forward(round=i, text_lst=text_lst)
        #     print('text_lst===>', text_lst)
        #     print('rm_trainer.prompt', rm_trainer.prompt)
        #     dis_lst_max = [x for x in text_lst if x[-2] == 1 and not x[-1]]
        #     if dis_lst_max:
        #
        #         break



        # dis = [(x,i) for i, x in enumerate(dis)]
        # dis = list(reversed(sorted(dis, key=lambda x: (x[0][0], x[0][1]))))
        # print('DDDIIISSS', dis)
        # print('dis_lst', rm_trainer.distance)
        # for idx, dis_t in enumerate(rm_trainer.distance):
        #     dis_t_0 = [s[0] for s in dis_t]
        #     dis_t_1= [s[1] for s in dis_t]
        #     idx_0 = dis_t_0.index(max(dis_t_0))
        #     idx_1 = dis_t_1.index(max(dis_t_1))
        #     idx_last = max(idx_0, idx_1)
        #     print(idx_last, idx_0, idx_1 )
        #     rm_trainer.prompt[idx] = '.'.join(rm_trainer.prompt[idx].split('.')[:idx_last+1])
        #
        # idx = dis[0][1]
        # print('BEST', rm_trainer.prompt[idx])
        # rm_trainer.forward(goal=True, goal_sentence='Bob lose weight.')

    elif args.trainer_type == 'KG_diff':
        # with HiddenPrints():
        prompt_text = "I hid under the tiny bed of my room , terrified."
        # prompt_text = 'John is sleepy.'
        rm_trainer = Reader_Model_Trainer_KG_diff(args=args, prompt_text=prompt_text)
        rm_trainer.define_roles(prompt_text)

        # define the goal
        goal_text = """
                    I hid under the tiny bed of my room , terrified.
                    I covered my mouth , trying to muffle my heavy breathing. 
                    I heard its footsteps in the distance , dragging it 's feet as it walked . I heard it sniff the air , grunting , as it caught on to a scent . It started walking in a different direction , before suddenly breaking out into a sprint . I gasped instinctively , before realizing it was moving away from me . <newline> <newline> I lay there , in complete darkness , filled with fear . It had broken the main generators , and the backups only provided power to the main hallways . In the distance , I heard a faint scream , which was soon replaced by a deafening silence . <newline> <newline> I slowly moved out from under my bed . My hands were trembling , tears blurring my vision . I stepped out of my room , into the crew quarters . Light from a distant hallway dimly illuminated the passageway . I started walking forward , unsure of where to go . I felt dizzy , and it hurt to think . <newline> <newline> Then , in the distance I saw a man , darkened by the shadows cast on him . He walked in a very strange manner , almost as if he were limping , but it seemed incredibly unnatural . I was about to call out to him , when I felt a hand cover my mouth . <newline> <newline> `` Are you trying to fucking kill us ? '' <newline> <newline> -- -- - <newline> <newline> I sat in the room of Father John . I 'd seen him a couple of times before , in his black robes . A priest , recommended by The Pope himself . I nervously fidgeted with my fingers as the Father drew some kind of symbol on the wall . <newline> <newline> The entire room was covered with symbols like the one he was drawing . Scribbles of Latin often accompanied them . The man was either insane , or knew exactly what he was doing . <newline> <newline> `` What is that out there ? '' I asked timidly . <newline> <newline> `` It 's a demon . '' He answered nonchalantly , as he continued to draw . <newline> <newline> `` A demon ? '' I scoffed . I was a man of science , not faith . Science had gotten us to the Moon , and soon Mars . Not faith . <newline> <newline> `` They do n't exist . '' I continued . <newline> <newline> `` You can tell it that when you see it . '' He replied . <newline> <newline> `` We 're the sacrificial lambs Musk has offered this creature . It 's going to pick us apart , one by one . There is n't going to be anyone left to step on to Mars , when we finally reach . '' He said coldly . <newline> <newline> It was so easy to dismiss what John was saying , yet I believed him . <newline> <newline> `` Is there anything we can do ? '' I asked , as the possibility of death suddenly seemed imminent . <newline> <newline> `` We can fly back home . '' <newline> <newline> -- -- - <newline> Hope you liked it OP . Not exactly what the prompt asked for , but hope it was n't too bad . Would you like a part 2 ? <newline> <newline> Would love for your feedback . Super cool prompt by the way . <newline> <newline> Enjoyed my writing ? I 'm actually writing a small series on /r/fallenwings and would love for you to check it out ! <newline>

                    """
        # goal_text = """
        #             John is sleepy.
        #             He starts a pot of coffee.
        #             John puts cream and sugar in his cup.
        #             He then goes to work.
        #             """
        outputs = rm_trainer.prepare_next_step(picked_action=goal_text, goal_or_build='goal', goal_word='run-51.3.2-1')

        print("000. GOAL state", rm_trainer.KG_goal.visualize(order=0))
        print('rm_trainer', rm_trainer.goal_word)
        with HiddenPrints():
            rm_trainer.forward()
        for prompt in rm_trainer.prompt:
            print('1>>>', prompt)
        # print('self.node_history', rm_trainer.node_history)
        for i in range(2, 10):
            # with HiddenPrints():
            rm_trainer.forward(round=i)
            for prompt in rm_trainer.prompt:
                print(i, '>>>', prompt)
        # for i in range(2, 10):
        #     with HiddenPrints():
        #         kg_diff_trainer.forward()
        #     for trainer in kg_diff_trainer.trainer:
        #         print(i, '>>>', trainer[0].prompt,  trainer[-1])

    elif args.trainer_type == 'KG_diff_file':
        start_time = time.time()
        with open(args.story_file_path, "r") as f:
            for i, story in enumerate(f):
                prompt_text = story.split('.')[0] + '.'
                process_time, start_time = (time.time() - start_time) / 60, time.time()
                if i == 0:
                    rm_trainer = Reader_Model_Trainer_KG_diff(args=args, prompt_text=prompt_text)
                with HiddenPrints():
                    rm_trainer.define_roles(prompt_text)
                    process_time, start_time = (time.time() - start_time) / 60, time.time()
                # with HiddenPrints():
                outputs = rm_trainer.prepare_next_step(picked_action=story,
                                                       goal_or_build='goal',
                                                       goal_word='')
                process_time, start_time = (time.time() - start_time) / 60, time.time()
                print("build target KG takes %.2f mins" % process_time)
                with HiddenPrints():
                    rm_trainer.forward()
                    process_time, start_time = (time.time() - start_time) / 60, time.time()
                    print("whole 1st forward takes %.2f mins" % process_time)


                for prompt in rm_trainer.prompt:
                    print('1>>>', prompt)

                for j in range(1, args.story_length):
                    # with HiddenPrints():
                    _, _, _, stop_sig, _, stop_sig_patience = rm_trainer.forward(round=j)
                    print(rm_trainer.prompt)
                    for prompt in rm_trainer.prompt:
                        print(j+1, '>>>', prompt)
                    if stop_sig or stop_sig_patience:
                        print('EARLY STOP!!!')
                        break

                rm_trainer.save_file(stop_sig_patience)
                # clear everything!
                rm_trainer.clear()

    else:  # baseline
        prompt_text = "Judy went to school."
        rm_trainer = Baseline_Trainer(args=args, prompt_text=prompt_text)
        rm_trainer.define_roles()
        print("action is =>", rm_trainer.forward())
        print("action is =>", rm_trainer.forward())
        print("action is =>", rm_trainer.forward())


        # rm_trainer.run_file()


