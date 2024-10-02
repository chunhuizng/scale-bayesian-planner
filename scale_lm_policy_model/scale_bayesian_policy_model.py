import os
import json
from cv2 import exp
from numpy import save
import torch
import argparse
from tqdm import tqdm
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import pickle
import os


norm_dict = {
    'housework_qa': 'un',
    'neg_housework_qa': 'un',
    'act_infer': 'un',
    'act_recog': 'ln',
    'count': 'ln',
    'obj_move': 'ln',
    'eval_dataset_48': 'ln'
}

SURFACE_LIST = [
    "coffeetable",
    "desk"
    "kitchentable",
    "sofa",
]

CONTAINER_LIST = [
    "kitchencabinet",
    "cabinet",
    "bathroomcabinet",
    "dishwasher",
    "fridge",
    "microwave",
    "stove",
]

NEW_CONTAINER_LIST = [
    "1st kitchencabinet from left to right",
    "2nd kitchencabinet from left to right",
    "3rd kitchencabinet from left to right",
    "4th kitchencabinet from left to right",
    "5th kitchencabinet from left to right",
    "6th kitchencabinet from left to right",
    "7th kitchencabinet from left to right",
    "8th kitchencabinet from left to right",
    "cabinet",
    "bathroomcabinet",
    "dishwasher",
    "fridge",
    "microwave",
    "stove",
]

OBJECT_LIST = [ 
    "apple",
    "book",
    "chips",
    "condimentbottle",
    "cupcake",
    "dishbowl",
    "plate",
    "remotecontrol",
    "salmon",
    "waterglass",
    "wine",
    "wineglass",
]

POSSIBLE_BELIEF = [
    "1st kitchencabinet",
    "2nd kitchencabinet",
    "3rd kitchencabinet",
    "4th kitchencabinet",
    "5th kitchencabinet",
    "6th kitchencabinet",
    "7th kitchencabinet",
    "8th kitchencabinet",
    "cabinet",
    "bathroomcabinet",
    "dishwasher",
    "fridge",
    "microwave",
    "stove",
    "coffeetable",
    "desk"
    "kitchentable",
    "sofa"
]


def remove_item(s, item_to_remove):
    items = [item.strip() for item in s.split(',')]
    if item_to_remove in items:
        items.remove(item_to_remove)
    return ', '.join(items)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default='cyj/files/multimodal_representations.json')
    parser.add_argument('--load_lora', type=int, default=1)
    parser.add_argument('--output_log', action="store_true")
    parser.add_argument('--output_path', type=str, default='n_planner.txt')

    # device for map: str = "auto"
    parser.add_argument("--device_map", type=str, default="auto") # or auto balanced_low_0
    parser.add_argument('--model_name_or_path', type=str, default='/data/01/public_models/Meta-Llama-3-8B/')#/data/01/public_models/LLaMA-2-7B-32K/')
    parser.add_argument('--large_base_name_or_path', type=str, default='/data/01/public_models/Meta-Llama-3-70B')#/data/01/public_models/LLaMA-2-7B-32K/')
    parser.add_argument('--use_lora_post_trained_lm', type=bool, default=True)
    parser.add_argument('--lora_name_or_path', type=str, default='zch/llama3_8b_ft')#llama2_7b_ft/')

    # setup for the likelihood estimator
    parser.add_argument('--alpha', type=float, default=1.0, help='weighting factor for the post_trained_lm logits') # also can consider normalize the post_trained_lm logits
    parser.add_argument('--load_in_4bit', default=False, help='load the model in 8bit mode') # keep the default value False
    parser.add_argument('--use_fast_tokenizer', default=True, help='use fast tokenizer') # keep the default value True
    args = parser.parse_args()
    return args

def main(args):
    # show all info of the args clearly
    print(args)
    acc_list = []

    # Prepare tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, add_bos_token = False)#, token="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") # add_bos_token means add the bos token to the beginning of the prompt
    tokenizer.pad_token_id = tokenizer.eos_token_id # set the pad token id to the eos token id
    # padding side default is right
    # define the base and post_trained_lm in same scale
    print("Loading base from ", args.model_name_or_path)
    base = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, load_in_4bit=args.load_in_4bit, device_map=args.device_map)#, token="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")
    if args.load_lora:
        # re-create an post_trained_lm_base like the base for peftmodel wrap: 
        post_trained_lm_base = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, load_in_4bit=args.load_in_4bit, device_map=args.device_map)
        print("Loading LORA post_trained_lm from ", args.lora_name_or_path)
        post_trained_lm = PeftModel.from_pretrained(post_trained_lm_base, args.lora_name_or_path, device_map=args.device_map)
    # define the model_large_base in larger scale
    print("Loading large_base from ", args.large_base_name_or_path)
    large_base = AutoModelForCausalLM.from_pretrained(args.large_base_name_or_path, load_in_4bit=args.load_in_4bit, device_map=args.device_map)#, token="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")
    
    # set the model to eval mode
    base.eval()
    post_trained_lm.eval()
    large_base.eval()

    # Directory to store the logits
    logits_dir = args.logits_dir
    os.makedirs(logits_dir, exist_ok=True)

    def compute_prob(inp, contxt_len, answer_tokens):
        # Generate the key for storing/loading logits
        key = (inp, contxt_len)
        # Compute the log probability of the answer given the input
        inputs = tokenizer(inp, return_tensors='pt', add_special_tokens=False)
        inputs = {k: v.cuda() for k, v in inputs.items()}

        base_logits = base(**inputs).logits
        post_trained_lm_logits = post_trained_lm(**inputs).logits
        large_base_logits = large_base(**inputs).logits

        base_logits = base_logits[:, contxt_len - 1:inputs['attention_mask'].sum()]
        post_trained_lm_logits = post_trained_lm_logits[:, contxt_len - 1:inputs['attention_mask'].sum()]
        large_base_logits = large_base_logits[:, contxt_len - 1:inputs['attention_mask'].sum()]

        delta_logits = post_trained_lm_logits - base_logits

        logits = large_base_logits + args.alpha * delta_logits

        vocab_log_probs = torch.log_softmax(logits, dim=-1)

        token_log_probs = torch.gather(
            vocab_log_probs, dim=2, index=answer_tokens[:, :, None] # answer_tokens is the tokenized answer indices, here it only used to get the log probability of the answer
        )
 
        log_prob = token_log_probs.sum()
        return log_prob.cpu().item()

    answer_contxt_len = tokenizer('Answer:', return_tensors="pt").input_ids.size(1)
    with torch.no_grad():
        with open(args.data_path, "r") as f:
            d = {}
            acc_list = []
            correct = {}
            all = {}
            lines = list(f)
            count = 0
            for line in tqdm(lines):
                count += 1
                sample = json.loads(line)
                print(f"\n\n\nQuestion {count}")
                print(f"Type: {sample['question_type']}")
                actions = sample['actions']
                actions[0] = 'None'
                actions[-1] = actions[-1].replace("about to open", "walktowards").replace("prepare to open", "walktowards").replace("ready to open", "walktowards")
                T = len(actions) - 1


                if sample['question_type'] in [1.1, 1.2, 1.3]:
                    # Belief inference
                    if sample['question_type'] == 1.3:
                        sample['answer'] = 'b'
                    else:
                        sample['answer'] = 'a'
                    print('answer', sample['answer'])
                    
                    hypo_belief = sample['hypo_belief']
                    hypo_goal = sample['hypo_goal']
                    print(f'hypo_belief: {hypo_belief}')
                    print(f'hypo_goal: {hypo_goal}')
                    # whether the goal is in the belief

                    beliefs = ['init_belief']
                    for i in range(1, T+1):
                        beliefs.append(sample[f'belief_{i}'])


                    last_belief = [b.strip() for b in beliefs[-1].split(",")]
                    hypo_belief_choice = beliefs.copy()
                    hypo_belief_choice[i] = remove_item(hypo_belief_choice[i], hypo_belief)
                    choices = [hypo_belief_choice]
                    for belief in last_belief:
                        if belief == hypo_belief:
                            continue
                        
                        other_belief_choice = beliefs.copy()
                        other_belief_choice[i] = remove_item(other_belief_choice[i], belief)
                        choices.append(other_belief_choice)

                    prob_choices = []
                    for beliefs in choices: # relation between beliefs and i is: beliefs[i] is the belief at time i
                        prob_list = []
                        for i in range(T, T+1):
                            state = sample[f'state_{i}']
                            prompt = f'goal: {hypo_goal} \nstate: {state} \nbelief (possible locations the person suspects the {hypo_goal} could be): {beliefs[i]} \naction: '
                            answer = actions[i]
                            prompt_len = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.size(1)
                            answer_tokens = tokenizer(f'{prompt} {answer}', return_tensors='pt', add_special_tokens=False).input_ids.cuda()[:,prompt_len:]
                            # prompt_len is the length of the prompt, so the input_ids.size(1) is the length of the prompt, answer_tokens is the tokenized actions which only contains the answer instead of the prompt
                            prob = compute_prob(f'{prompt} {answer}', prompt_len, answer_tokens)
                            final_prob = prob / answer_tokens.size(1)
                            print(f'belief: {prompt}')
                            print(f'belief: {beliefs[i]}')
                            print(f'action: {answer}')
                            print(f"P(A_{i} | G, S_{i}, B_{i}): {final_prob}\n")
                            prob_list.append(final_prob)
                        total_prob = sum(prob_list)
                        prob_choices.append(total_prob)
                    
                    if (sample['answer'] == 'a' and prob_choices[0] == min(prob_choices)) or (sample['answer'] == 'b' and prob_choices[0] != min(prob_choices)):
                        # while the answer is a, the probability of the first choice should be the smallest
                        print("correct")
                        acc_list.append(1)
                        correct[sample['question_type']] = correct.get(sample['question_type'], 0) + 1
                    else:
                        # while the answer is b, the probability of the first choice should be the largest
                        print("wrong")
                        acc_list.append(0)
                    all[sample['question_type']] = all.get(sample['question_type'], 0) + 1

                else:
                    # Goal inference
                    prob_choices = []
                    for num in ["1", "2"]: # means goal1 and goal2
                        prob_list = []
                        for i in range(1, T+1):
                            goal = sample[f'hypo_goal{num}']
                            state = sample[f'state_{i}']
                            belief = sample[f'goal{num}_belief_{i}']
                            if f'goal{num}_remove_belief' in sample.keys():
                                belief = remove_item(belief, sample[f'goal{num}_remove_belief'])

                            prompt = f'goal: {goal} \nstate: {state} \nbelief (possible locations the person suspects the {goal} could be): {belief} \naction: '

                            answer = actions[i]
                            prompt_len = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.size(1)
                            answer_tokens = tokenizer(f'{prompt} {answer}', return_tensors='pt', add_special_tokens=False).input_ids.cuda()[:,prompt_len:]
                            prob = compute_prob(f'{prompt} {answer}', prompt_len, answer_tokens)
                            final_prob = prob / answer_tokens.size(1)
                            print(f'prompt: {prompt}')
                            print(f'answer: {answer}')
                            print(f"P(A_{i} | G, S_{i}, B_{i}): {final_prob}")
                            prob_list.append(final_prob)
                        total_prob = sum(prob_list)
                        print(f"total probabilities: {total_prob}")
                        prob_choices.append(total_prob)
                    
                    if (prob_choices[0] >= prob_choices[1] and sample['answer'] == 'a') or (prob_choices[0] < prob_choices[1] and sample['answer'] == 'b'):
                        print("correct")
                        acc_list.append(1)
                        correct[sample['question_type']] = correct.get(sample['question_type'], 0) + 1
                    else:
                        print("wrong")
                        acc_list.append(0)
                    all[sample['question_type']] = all.get(sample['question_type'], 0) + 1


    for question_type in [1.1, 1.2, 1.3, 2.1, 2.2, 2.3, 2.4]:
        print(f"type {question_type}: {correct.get(question_type, 0)} / {all.get(question_type, 0)}")      
    acc = sum(acc_list) / len(acc_list)
    return acc

if __name__ == "__main__":
    args = parse_args()
    args.task_name = os.path.basename(args.data_path).split('.')[0]
    acc = main(args)

    task_name = args.task_name.replace('_', ' ')
    output_str = f"{task_name}: {acc:.4f}"
    print(output_str)
    if args.output_log:
        with open(args.output_path, 'a') as f:
            f.write(output_str + '\n')
