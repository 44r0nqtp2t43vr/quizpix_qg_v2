from fastapi import FastAPI
from pydantic import BaseModel
from pathlib import Path
import os
from typing import List
import re
import nltk
import torch
import requests
import random
import numpy as np
import tensorflow as tf
from nltk import tokenize
from rake_nltk import Rake
from unidecode import unidecode
from nltk.tokenize import sent_tokenize, word_tokenize
from transformers import AutoTokenizer, T5ForConditionalGeneration, T5Tokenizer, TFGPT2LMHeadModel, GPT2Tokenizer
from fastT5 import get_onnx_model, get_onnx_runtime_sessions, OnnxT5

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

app = FastAPI()


class QuestionRequest(BaseModel):
    text: str


class QuestionResponse(BaseModel):
    quiz: List[dict]

def extract_keywords(text, num_keywords):
    rake = Rake(include_repeated_phrases=False)
    keywords = rake.extract_keywords_from_text(text)
    keywords = rake.get_ranked_phrases()[0:num_keywords]
    return keywords

def tokenize_sentences(text):
    text = text.strip().replace("\n"," ")
    sentences = [sentence.strip() for sentence in sent_tokenize(text)]
    return sentences

def get_sentences_for_keyword(sentences, keywords):
    keyword_sentences = {}
    for keyword in keywords:
        keyword_sentences[keyword] = []
        for sentence in sentences:
            if re.search(r"\b{}\b".format(keyword), sentence.lower().strip()):
                keyword_sentences[keyword].append(sentence)
    return keyword_sentences

def get_sentences_with_keywords(sentences, keywords):
    keyword_sentences = []
    for keyword in keywords:
        for sentence in sentences:
            if re.search(r"\b{}\b".format(keyword), sentence.lower().strip()):
                keyword_sentences.append(sentence)
    return list(set(keyword_sentences))

# defining the api-endpoint 
API_ENDPOINT = "https://8cy7cvl9mi.execute-api.ap-southeast-1.amazonaws.com/dev/entities"

grammar = r"""
    THING:
        {<DT>?<NN|NNS>+<VB|VBD|MD>}
        {<VB|VBD|MD><DT>?<NN|NNS>+}
        {<IN><DT>?<NN|NNS>+}
        }<DT|VB|VBD|MD|IN>{
    DIGIT:
        {<CD>}
"""

def chunk(sentences):
    answers_sentences = []
    keywords_dict = {
        'name': [],
        'year': [],
        'location': [],
        'org': [],
        'other': [],
    }

    for sentence in sentences:
        text = nltk.tokenize.word_tokenize(sentence)
        tagged_text = nltk.pos_tag(text)
        cp = nltk.RegexpParser(grammar)
        chunk_tree = cp.parse(tagged_text)

        answer_dict = {
            'name': [],
            'year': [],
            'location': [],
            'org': [],
            'other': [],
            'answer': [],
        }

        # data to be sent to api
        data = {
            "text": sentence,
        }

        # sending post request and saving response as response object
        r = requests.post(url=API_ENDPOINT, json=data)

        # extracting data in json format
        res = r.json()

        if 'PERSON' in res.keys():
            answer_dict['name'] = [item['name'] for item in res['PERSON'] if len(item['name'].split()) >= 2]
        if 'ORGANIZATION' in res.keys():
            answer_dict['org'] = [item['name'].replace(" 's", "'s") if " 's" in item['name'] else item['name'] for item in res['ORGANIZATION']]
        if 'LOCATION' in res.keys():
            answer_dict['location'] = [item['name'].replace(" 's", "'s") if " 's" in item['name'] else item['name'] for item in res['LOCATION']]

        for n in chunk_tree:
            if isinstance(n, nltk.tree.Tree):               
                if n.label() == 'DIGIT':
                    years = [word[0] for word in n.leaves() if word[0].isdigit() and len(word[0]) == 4 and int(word[0]) <= 2025]
                    if len(years) > 0:
                        answer_dict['year'].append(" ".join(years))
                else:
                    others = [word[0] for word in n.leaves() if len(word[0]) >= 3]
                    answer_dict['other'].append(" ".join(others))
                
        
        if len(answer_dict['name']) > 0:
            answer_dict['answer'].append(answer_dict['name'])
            keywords_dict['name'].append(answer_dict['name'])
        if len(answer_dict['year']) > 0:
            answer_dict['answer'].append(answer_dict['year'])
            keywords_dict['year'].append(answer_dict['year'])
        if len(answer_dict['location']) > 0:
            if len(answer_dict['answer']) == 0:
                answer_dict['answer'] = [[answer_dict['location'][random.randint(0, len(answer_dict['location'])-1)]]]
            keywords_dict['location'].append(answer_dict['location'])
        if len(answer_dict['org']) > 0:
            if len(answer_dict['answer']) == 0:
                answer_dict['answer'] = [[answer_dict['org'][random.randint(0, len(answer_dict['org'])-1)]]]
            keywords_dict['org'].append(answer_dict['org'])
        if len(answer_dict['other']) > 0:
            if len(answer_dict['answer']) == 0:
                answer_dict['answer'] = [[answer_dict['other'][random.randint(0, len(answer_dict['other'])-1)]]]
            keywords_dict['other'].append(answer_dict['other'])
        answer_dict['answer'] = [item for sublist in answer_dict['answer'] for item in sublist]
        
        for answer in answer_dict['answer']:
            answers_sentences.append((answer, sentence))

    for key in keywords_dict:
        keywords_dict[key] = list(set([item for sublist in keywords_dict[key] for item in sublist]))

    random.shuffle(answers_sentences)
    return answers_sentences, keywords_dict

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# question_model = T5ForConditionalGeneration.from_pretrained('ramsrigouthamg/t5_squad_v1')
# question_tokenizer = T5Tokenizer.from_pretrained('ramsrigouthamg/t5_squad_v1')
# question_model = question_model.to(device)

trained_model_path = './t5_squad_v1/'
pretrained_model_name = Path(trained_model_path).stem
encoder_path = os.path.join(trained_model_path,f"{pretrained_model_name}-encoder-quantized.onnx")
decoder_path = os.path.join(trained_model_path,f"{pretrained_model_name}-decoder-quantized.onnx")
init_decoder_path = os.path.join(trained_model_path,f"{pretrained_model_name}-init-decoder-quantized.onnx")
model_paths = encoder_path, decoder_path, init_decoder_path
model_sessions = get_onnx_runtime_sessions(model_paths)
question_model = OnnxT5(trained_model_path, model_sessions)
question_tokenizer = AutoTokenizer.from_pretrained(trained_model_path)

def get_question(context, answer, model, tokenizer):
    text = "context: {} answer: {}".format(context, answer)
    encoding = tokenizer.encode_plus(
        text, max_length=384, pad_to_max_length=False, truncation=True, return_tensors="pt")
    input_ids, attention_mask = encoding["input_ids"], encoding["attention_mask"]

    outs = model.generate(input_ids=input_ids,
                          attention_mask=attention_mask,
                          early_stopping=True,
                          num_beams=5,
                          num_return_sequences=1,
                          no_repeat_ngram_size=2,
                          max_length=72)

    dec = [tokenizer.decode(ids, skip_special_tokens=True) for ids in outs]

    Question = dec[0].replace("question:", "")
    Question = Question.strip()
    return Question

def get_levenshtein_dists(sources, target):
    dists = []

    for source in sources:
        matrix = np.zeros((len(source) + 1, len(target) + 1), dtype=int)
        matrix[0, :] = [x for x in range(len(target) + 1)]
        matrix[:, 0] = [x for x in range(len(source) + 1)]

        for i in range(1, len(source) + 1):
            for j in range(1, len(target) + 1):
                if source[i - 1] == target[j - 1]:
                    matrix[i][j] = min([matrix[i - 1][j] + 1, matrix[i][j - 1] + 1, matrix[i - 1][j - 1]])
                else:
                    matrix[i][j] = min([matrix[i - 1][j] + 1, matrix[i][j - 1] + 1, matrix[i - 1][j - 1] + 2])
        dists.append(matrix[len(source)][len(target)])
    return dists

def get_distractors(question, answer, keywords_dict):
    distractors = []
    random.shuffle(keywords_dict['name'])
    random.shuffle(keywords_dict['year'])
    random.shuffle(keywords_dict['location'])
    random.shuffle(keywords_dict['org'])
    random.shuffle(keywords_dict['other'])
    if answer in keywords_dict['name']:
        for name in keywords_dict['name']:
            if name != answer:
                distractors.append(name)
            if len(distractors) >= 3:
                return distractors
    elif answer in keywords_dict['year']:
        for year in keywords_dict['year']:
            if int(year) != int(answer):
                distractors.append(year)
            if len(distractors) >= 3:
                return distractors
        while len(distractors) < 3:
            rand_year = str(int(keywords_dict['year'][random.randint(0, len(keywords_dict['year'])-1)]) - random.randrange(-10, 10))
            if rand_year not in distractors:
                distractors.append(rand_year)
        return distractors
    if answer in keywords_dict['location']:
        for location in keywords_dict['location']:
            if location != answer and not any(dist < 5 for dist in get_levenshtein_dists(distractors, location)) and location not in question.rstrip('?:!.,;').split():
                distractors.append(location)
            if len(distractors) >= 3:
                return distractors
    if answer in keywords_dict['org']:
        for org in keywords_dict['org']:
            if org != answer and not any(dist < 5 for dist in get_levenshtein_dists(distractors, answer)) and org not in question.rstrip('?:!.,;').split():
                distractors.append(org)
            if len(distractors) >= 3:
                return distractors
    if answer in keywords_dict['other']:
        for other in keywords_dict['other']:
            if other != answer and not any(dist < 5 for dist in get_levenshtein_dists(distractors, other)) and other not in question.rstrip('?:!.,;').split():
                distractors.append(other)
            if len(distractors) >= 3:
                return distractors
    for other in keywords_dict['org'] + keywords_dict['name'] + keywords_dict['location']:
        if other != answer and not any(dist < 5 for dist in get_levenshtein_dists(distractors, other)) and other not in question.rstrip('?:!.,;').split():
            distractors.append(other)
        if len(distractors) >= 3:
            return distractors
    for other in keywords_dict['other']:
        if other != answer and not any(dist < 5 for dist in get_levenshtein_dists(distractors, other)) and other not in question.rstrip('?:!.,;').split():
            distractors.append(other)
        if len(distractors) >= 3:
            return distractors
    return distractors

def get_fill_in_the_blanks(answers_sentences):
    out = {}
    blank_sentences = []
    processed = []
    answers = []
    for answer_sentence in answers_sentences:
        sentence = answer_sentence[1]
        if sentence in processed:
            continue
        insensitive_sentence = re.compile(re.escape(answer_sentence[0]), re.IGNORECASE)
        blank_sentence = insensitive_sentence.sub(' _________ ', sentence)
        blank_sentences.append(blank_sentence)
        processed.append(sentence)
        answers.append(unidecode(answer_sentence[0]))
    out['sentences'] = blank_sentences
    out['answers'] = answers
    return out

#initialize GPT2 tokenizer and model for generating sentences
GPT2tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
GPT2model = TFGPT2LMHeadModel.from_pretrained("distilgpt2", pad_token_id = GPT2tokenizer.eos_token_id)

def remove_from_string(main_string, sub_string):
    combined_sub_string = sub_string.replace(" ", "")
    main_string_list = main_string.split()
    last_index = len(main_string_list)
    for i in range(last_index):
        check_string_list = main_string_list[i:]
        check_string = "".join(check_string_list)
        check_string = check_string.replace(" ", "")
        if check_string == combined_sub_string:
            return " ".join(main_string_list[:i])
    ind = 0
    first_word = sub_string.split()[0]
    while first_word not in main_string_list and ind < last_index - 1:
      first_word = sub_string.split()[ind + 1]
      ind = ind + 1
    if first_word in main_string_list:
      return " ".join(main_string_list[:main_string_list.index(first_word)])
    else:
      return " ".join(main_string_list[:last_index-1])
    
def falsify_sentence(sentence):
    sentence = sentence.rstrip('?:!.,;')
    text = word_tokenize(sentence)
    tagged_text = nltk.pos_tag(text)
    words = [word[0] for word in tagged_text]
    tags = [word[1] for word in tagged_text]
    if ('CC' in tags) and ((len(tags) - tags[::-1].index('CC') - 1) >= len(tags)/2):
        ind = len(tags) - tags[::-1].index('CC') - 1
    elif ('IN' in tags) and ((len(tags) - tags[::-1].index('IN') - 1) >= len(tags)/2):
        ind = len(tags) - tags[::-1].index('IN') - 1
    elif ('VB' in tags) and ((len(tags) - tags[::-1].index('VB') - 1) >= len(tags)/2):
        ind = len(tags) - tags[::-1].index('VB') - 1
    elif ('VBD' in tags) and ((len(tags) - tags[::-1].index('VBD') - 1) >= len(tags)/2):
        ind = len(tags) - tags[::-1].index('VBD') - 1
    elif ('NN' in tags) and ((len(tags) - tags[::-1].index('NN') - 1) >= len(tags)/2):
        ind = len(tags) - tags[::-1].index('NN') - 1
    else:
        ind = -1
    if (tags[ind - 1] == 'NNP') or (tags[ind - 2] == 'NNP'):
        ind = ind - 3
    substring = " ".join(words[ind:])
    substring = re.sub(r"-LRB- ", "(", substring)
    substring = re.sub(r" -RRB-", ")", substring)
    split_sentence = remove_from_string(sentence, substring)

    #encode split_sentence and generate sentence using words with >= 80% probability
    input_ids = GPT2tokenizer.encode(split_sentence, return_tensors='tf')

    maximum_length = len(input_ids[0]) + 16
    sample_outputs = GPT2model.generate(
        input_ids, 
        do_sample = True, 
        max_length = maximum_length, 
        top_p = 0.80,
        top_k = 30,
        repetition_penalty = 10.0,
        num_return_sequences = 4
    )

    #decode generated sentences
    gen_sentences = []
    for sample_output in sample_outputs:
        decoded_sentence = GPT2tokenizer.decode(sample_output, skip_special_tokens = True)
        final_sentence = tokenize.sent_tokenize(decoded_sentence)[0]
        tagged_final_sentence = nltk.pos_tag(word_tokenize(final_sentence))
        fs_tags = [word[1] for word in tagged_final_sentence]
        if (fs_tags[-1] == 'NN' or fs_tags[-1] == 'NNS') and len(final_sentence.split()) > len(sentence.split()):
            return final_sentence
        gen_sentences.append(final_sentence)
    return gen_sentences[0]

def generate_quiz(text):
    text = text.replace(',', '')
    num_words = len(text.split())
    sentences = tokenize_sentences(text)
    keywords = extract_keywords(text, int(num_words/10))
    keywords_sentences = get_sentences_with_keywords(sentences, keywords)
    tf_sentences = [sentence for sentence in sentences if sentence not in keywords_sentences]
    if len(tf_sentences) > len(keywords_sentences):
        tf_sentences = tf_sentences[:len(keywords_sentences)]
    answers_sentences, keywords_dict = chunk(keywords_sentences)
    answers_sentences_p1 = answers_sentences[:int(len(answers_sentences)/2)]
    answers_sentences_p2 = answers_sentences[int(len(answers_sentences)/2) + 1:]

    generated_question_list = []

    for answer_sentence in answers_sentences_p1:
        generated_question = {}
        generated_question['type'] = 'multiple_choice'
        generated_question['question'] = get_question(answer_sentence[1], answer_sentence[0], question_model, question_tokenizer)
        generated_question['answer'] = answer_sentence[0]
        generated_question['choices'] = get_distractors(generated_question['question'], answer_sentence[0], keywords_dict) + [answer_sentence[0]]
        random.shuffle(generated_question['choices'])
        generated_question_list.append(generated_question)
    
    fill_in_the_blanks = get_fill_in_the_blanks(answers_sentences_p2)
    for ind in range(len(fill_in_the_blanks['sentences'])):
        generated_question = {}
        generated_question['type'] = 'identification'
        generated_question['question'] = fill_in_the_blanks['sentences'][ind]
        generated_question['answer'] = fill_in_the_blanks['answers'][ind]
        generated_question['choices'] = []
        generated_question_list.append(generated_question)
    
    for sentence in tf_sentences:
        generated_question = {}
        generated_question['type'] = 'true_or_false'
        randNum = random.randint(0, 1)
        if randNum == 0:
            generated_question['question'] = falsify_sentence(sentence)
            generated_question['answer'] = 'false'
        else:
            generated_question['question'] = sentence
            generated_question['answer'] = 'true'
            generated_question['choices'] = []
            generated_question_list.append(generated_question)

    return generated_question_list

@app.get('/')
def index():
    return {'message': 'hello world'}


@app.post("/generatequiz", response_model=QuestionResponse)
def getquestion(request: QuestionRequest):
    text = request.text
    quiz = generate_quiz(unidecode(text))

    return QuestionResponse(quiz=quiz)
