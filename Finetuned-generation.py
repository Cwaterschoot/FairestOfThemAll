import torch
import json
import re
import os
import pandas as pd
import numpy as np
import time
import random
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig
from peft import PeftModel


random.seed(time.time())


DRIVE_PATH = ''
LOCAL_PATH = '/content/temp_modules/'

if not os.path.exists(LOCAL_PATH):
    os.makedirs(LOCAL_PATH)


! cp -r "$DRIVE_PATH"config_gen.py "$LOCAL_PATH"
! cp -r "$DRIVE_PATH"strats.py "$LOCAL_PATH"

if DRIVE_PATH in sys.path:
    sys.path.remove(DRIVE_PATH)

if LOCAL_PATH not in sys.path:
    sys.path.insert(0, LOCAL_PATH)

try:
    from config_gen import generate_coalitional_conf, generate_divergent_conf, generate_minorty_conf, generate_uniform_conf
    from strats import ADD, MPL, APP, LMS, MAJ, FAI
    print("SUCCESS: Modules imported from local runtime.")
except ImportError as e:
    print(f"FATAL WARNING: Import failed even after copying: {e}")





OUTPUT_DIR = "" 
BASE_MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct" 
FILE_PATH = 'results2.csv'
ITERATIONS = 1
llm_name = "" 


######
tester = True
#####



SYSTEM_PROMPT ="""You are a participant in a scientific study. You will be presented with scenarios where a software system makes recommendations based on group preferences. Evaluate the fairness, consensus, and satisfaction using a 7-point Likert scale (-3 to 3).
**Important: Think efficiently—focus on 2-3 key observations per dimension. Avoid verbose calculations or repetition. Keep thinking concise**

The statements are to be answered using a 7-point Likert scale:
-3: strongly disagree
-2: disagree
-1: somewhat disagree
0: neither agree, neither disagree
1: somewhat agree
2: agree
3: strongly agree

Use these scores to answer the statements.

**Instructions:**
- For each recommended restaurant, provide reasoning for fairness, consensus, and satisfaction.
- Your reasoning should refer to the ratings given by individual members, highlighting trends, disagreements, or patterns, but you may describe them in natural language—there is no need for a strict template.
- The provided ratings are on a 1-5 star scale. (3 is average, above 3 is good, below 3 is bad.)
- Discuss what aspects of the ratings or group history influence your judgment, and optionally mention what would improve fairness, consensus, or satisfaction.
- Finally, provide integer scores for each dimension ranging from (and including) -3 (strongly disagree) to 3 (strongly agree).
- **Only respond with the format below. Keep your responses concise. Avoid unnecessary calculations or thinking.**

For example, if you strongly disagree that the recommended restaurant is fair, give -3 for fairness.

**Output format:**

**Scores:**
Fairness: score
Consensus: score
Satisfaction: score





"""


import re
import json

def parse_output(output_string: str) -> dict:
    """
    Parse model output that may contain reasoning followed by ratings.
    """
    output_string = output_string.strip()


    try:
        cleaned = re.sub(r'```json\s*|\s*```', '', output_string)
        parsed = json.loads(cleaned)

        result = {}
        for key in ['fairness', 'Fairness']:
            if key in parsed:
                result['Fairness'] = parsed[key]
                break

        for key in ['consensus', 'Consensus']:
            if key in parsed:
                result['Consensus'] = parsed[key]
                break

        for key in ['satisfaction', 'Satisfaction']:
            if key in parsed:
                result['Satisfaction'] = parsed[key]
                break

        if len(result) == 3 and all(isinstance(v, (int, float)) for v in result.values()):

            result = {k: int(v) for k, v in result.items()}
            if all(-3 <= v <= 3 for v in result.values()):
                return result
    except (json.JSONDecodeError, AttributeError, ValueError, KeyError):
        pass


    pattern = re.compile(
    r'\*{0,2}(Fairness|Consensus|Satisfaction)\*{0,2}\s*\*{0,2}:\*{0,2}\s*(-?\d+)',
    re.IGNORECASE
)


    results = {}
    matches = list(pattern.finditer(output_string))

    for match in reversed(matches):
        key = match.group(1).capitalize()
        if key not in results:
            try:
                value = int(match.group(2))
                if -3 <= value <= 3:
                    results[key] = value
                else:
                    print(f"WARNING: Value {value} for {key} out of range [-3, 3]")
            except ValueError:
                print(f"WARNING: Could not convert '{match.group(2)}' to int for {key}")


    required_keys = ['Fairness', 'Consensus', 'Satisfaction']
    if len(results) != 3:
        missing = [k for k in required_keys if k not in results]
        print(f"WARNING: Found only {len(results)}/3 metrics. Missing: {missing}")
        print(f"   Input: {output_string[:150]}...")

    return results

#
# MODEL LOADING
#


if 'model' not in globals() or 'tokenizer' not in globals():
    print("="*80)
    print("LOADING MODEL (FIRST TIME)")
    print("="*80)

    bnb_4bit_compute_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16


    tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR,fix_mistral_regex=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"✓ Tokenizer loaded (vocab size: {len(tokenizer)})")


    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=bnb_4bit_compute_dtype,
        bnb_4bit_use_double_quant=True,
    )




    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        dtype=bnb_4bit_compute_dtype,
        device_map="auto",
        quantization_config=bnb_config,
        #attn_implementation="flash_attention_2",
    )



    model = PeftModel.from_pretrained(base_model, OUTPUT_DIR)
    model = model.merge_and_unload()
    model.eval()

    print("✓ Model loaded and ready for inference")


    terminators = [tokenizer.eos_token_id]

    if "<|eot_id|>" in tokenizer.get_vocab():  # Llama
        eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
        if eot_id is not None and eot_id != tokenizer.unk_token_id:
            terminators.append(eot_id)
    elif "<|im_end|>" in tokenizer.get_vocab():  # Qwen
        im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        if im_end_id is not None and im_end_id != tokenizer.unk_token_id:
            terminators.append(im_end_id)
else:
    print("="*80)
    print("MODEL ALREADY LOADED - SKIPPING")
    print("="*80)
    print("✓ Using cached model and tokenizer")


# TEST

print("\n" + "="*80)
print("TESTING SINGLE EXAMPLE")
print("="*80)

messages_test = [
    {'role': 'system', 'content': SYSTEM_PROMPT},
    {'role': 'user', 'content': """
Assume that there is a group of friends. Every month, a group decision is made by these friends to decide on a restaurant to have dinner together.
To select a restaurant for the dinner next month, the group again has to take the same decision.
In this decision, each group member explicitly rated ten possible restaurants using a 5-star rating scale (1: the worst, 5: the best).
The ratings given by group members are shown in the table below:

### BEGIN TABLE ###
{'rest_1': [4, 2, 4, 2, 4], 'rest_2': [5, 5, 3, 5, 4], 'rest_3': [1, 1, 2, 2, 1], 'rest_4': [3, 5, 5, 4, 4], 'rest_5': [1, 4, 1, 2, 3], 'rest_6': [2, 1, 1, 1, 2], 'rest_7': [4, 2, 2, 5, 5], 'rest_8': [5, 4, 5, 5, 5], 'rest_9': [3, 3, 2, 1, 1], 'rest_10': [4, 3, 4, 4, 3]}
### END TABLE ###

The group decided to avoid going in the same restaurant too often; hence, after a restaurant has been selected, it cannot be chosen again for the next 4 dinners.
**The last 3 restaurants visited are: rest_2, rest_4, rest_8**.
Using the provided ratings, the system made a suggestion for the group on the basis of the preferences of all the group members.
**rest_7** has been recommended to the group.

Indicate your agreement to the following statements (7-point Likert scale ranging from -3 to 3):

** 1. Fairness **
The group recommendation is fair to all group members.

** 2. Consensus **
The group members will agree on the group recommendation.

** 3. Satisfaction **
The group members will be satisfied with regard to the group recommendation.

**Instructions:**
1. Very briefly make some observations
2. Provide your scores in the output format
3. Write a very short reasoning for each dimension
4. **Keep it concise.**

**Output format:**
**Scores:**
Fairness: score
Consensus: score
Satisfaction: score


"""}
]
if tester == True:
    prompt_text = tokenizer.apply_chat_template(
        messages_test,
        add_generation_prompt=False,
        tokenize=False,
        #enable_thinking=False
    )


    inputs = tokenizer(
        prompt_text,
        return_tensors="pt",
        padding=False,
    )

    input_ids = inputs['input_ids'].to(model.device)
    attention_mask = inputs['attention_mask'].to(model.device)


    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=1500,
            eos_token_id=terminators,
            temperature=0.5,
            do_sample=True,
            repetition_penalty=1.15,
        )

    response = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)

    print("--- RAW MODEL RESPONSE ---")
    print(response)
    print("\n--- PARSED OUTPUT ---")
    parsed = parse_output(response)
    print(parsed)




print("\n" + "="*80)
print("STARTING FULL INFERENCE LOOP")
print("="*80)

configurations = {
    "coalitional": lambda: generate_coalitional_conf(n=5, m=10, r=5),
    "divergent":   lambda: generate_divergent_conf(n=5, m=10, r=5),
    "minority":    lambda: generate_minorty_conf(n=5, m=10, r=5),
    "uniform":     lambda: generate_uniform_conf(n=5, m=10, r=5),
}

strategies = {
    "ADD": lambda: ADD(rating_df),
    "MAJ": lambda: MAJ(result),
    "LMS": lambda: LMS(rating_df),
    "MPL": lambda: MPL(rating_df),
    "APP": lambda: APP(rating_df, threshold=3),
    "FAI": lambda: FAI(result),
}

explanations = {
    "ADD": "since it achieves the highest total rating among the available options",
    "MAJ": "since most group members like it among the available options",
    "LMS": "since no group members has a real problem with it among the available options",
    "MPL": "since it achieves the highest of all individual group members among the available options",
    "APP": "since it achieves the highest number of ratings which are above 3 among the available options",
}

count = 0
errorcount = 0
ers = []
goal = 150
configuration_list = ['divergent', 'minority' , 'uniform', 'coalitional'] #['divergent', 'minority' , 'uniform', 'coalitional']
names = ['Alex', 'Bob', 'Carl', 'David', 'Elle']


if os.path.exists(FILE_PATH):
    df_results = pd.read_csv(FILE_PATH)
    print(f"Loaded existing results with {len(df_results)} rows.")
else:
    columns = ['iteration', 'configuration', 'strategy', 'expl',
               'fairness', 'consensus', 'satisfaction',
               'scenario', 'llm']
    df_results = pd.DataFrame(columns=columns)
    print("Initialized new results DataFrame.")

system_message = {'role': 'system', 'content': SYSTEM_PROMPT}

while count < goal:
    count += 1
    configuration = random.choice(configuration_list)
    print(configuration)
    matrix = configurations[configuration]()

    restaurants = matrix[0].keys()
    result = {r: [entry[r] for entry in matrix] for r in restaurants}

    person_four = names[3]  # David
    rating_df = pd.DataFrame([
        {"item": item, "rating": r}
        for item, ratings in dict(result).items()
        for r in ratings
    ])

    for strategy in ['ADD', 'APP', 'LMS', 'MPL', 'MAJ', 'FAI']:
        if strategy == 'FAI':
            result['names'] = names

        for expl in [0, 1]:
            try:
                order = strategies[strategy]()
                already_visited = ", ".join(map(str, order[0:3]))
                recommendation = str(order[3])
            except Exception as e:
                print(f"Error executing strategy {strategy}: {e}")
                continue

            if expl == 1:
                if strategy == 'FAI':
                    explanation = f"since it is the user {person_four}'s turn and this is their favourite choice among the available options"
                else:
                    explanation = explanations[strategy]
            else:
                explanation = ""

            scenario_content = f"""
Assume that there is a group of friends. Every month, a group decision is made by these friends to decide on a restaurant to have dinner together.
To select a restaurant for the dinner next month, the group again has to take the same decision.
In this decision, each group member explicitly rated ten possible restaurants using a 5-star rating scale (1: the worst, 5: the best).
The ratings given by group members are shown in the table below:

### BEGIN TABLE ###
{result}
### END TABLE ###

The group decided to avoid going in the same restaurant too often; hence, after a restaurant has been selected, it cannot be chosen again for the next 4 dinners.
**The last 3 restaurants visited are: {already_visited}**.
Using the provided ratings, the system made a suggestion for the group on the basis of the preferences of all the group members.
**{recommendation}** has been recommended to the group {explanation}.

Indicate your agreement to the following statements (7-point Likert scale ranging from -3 to 3):

** 1. Fairness **
The group recommendation is fair to all group members.

** 2. Consensus **
The group members will agree on the group recommendation.

** 3. Satisfaction **
The group members will be satisfied with regard to the group recommendation.


Score these statements in the context of the recommendation and already visited restaurants, which are not options anymore.

**Instructions:**
1. Very briefly rating observations
2. Provide your scores in the output format
3. Write a very short reasoning for each dimension
4. **Keep it concise.**

**Output format:**

**Scores:**
Fairness: score
Consensus: score
Satisfaction: score



"""

            messages_loop = [system_message, {'role': 'user', 'content': scenario_content}]


            i = 1
            while i <= ITERATIONS:
                try:

                    prompt_text = tokenizer.apply_chat_template(
                        messages_loop,
                        add_generation_prompt=False,
                        tokenize=False,
                        #enable_thinking=False
                    )

                    inputs = tokenizer(
                        prompt_text,
                        return_tensors="pt",
                        padding=False,
                    )

                    input_ids = inputs['input_ids'].to(model.device)
                    attention_mask = inputs['attention_mask'].to(model.device)

                    with torch.no_grad():
                        outputs = model.generate(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            max_new_tokens=3000,
                            eos_token_id=terminators,
                            temperature=0.5,
                            do_sample=True,
                            repetition_penalty=1.15,
                        )

                    response = tokenizer.decode(
                        outputs[0][input_ids.shape[-1]:],
                        skip_special_tokens=True
                    )

                    print(response)
                    out = parse_output(response)

                    required = ["Fairness", "Consensus", "Satisfaction"]
                    if not all(k in out and isinstance(out[k], int) for k in required):
                        raise ValueError(f"Missing or invalid keys. Got: {out}")

                    print(f"[{count}-{strategy}-{expl}] {out}")


                    df_results.loc[len(df_results)] = [
                        i, configuration, strategy, expl,
                        out['Fairness'], out['Consensus'], out['Satisfaction'],
                        f"{result} | visited: {already_visited} | rec: {recommendation} | expl: {explanation}",
                        llm_name
                    ]

                except ValueError as e:
                    print(f"Skipped malformed response: {str(e)}")
                    print(f"   Raw: {response[:80]}...")
                    errorcount += 1
                    ers.append((response, str(e)))

                except torch.cuda.OutOfMemoryError:
                    print("CUDA OOM: skipping iteration")
                    torch.cuda.empty_cache()
                    time.sleep(2)
                    break

                except Exception as e:
                    print(f"Unexpected error: {e}")
                    errorcount += 1
                    break

                i += 1



    df_results.to_csv(FILE_PATH, index=False)



df_results.to_csv(FILE_PATH, index=False)
print(f"\n{'='*80}")
print(f"✓ COMPLETE! Processed {count} scenarios")
print(f"  Total errors: {errorcount}")
print(f"  Success rate: {(count*12 - errorcount)/(count*12)*100:.1f}%")
print(f"  Results saved to: {FILE_PATH}")
print(f"{'='*80}")
