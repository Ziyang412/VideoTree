from string import Template
import re

def first_char_as_answer(res):
    mapping = {'A':0, 'B':1, 'C':2, 'D':3, 'E':4}
    if res is None:
        return -1
    if res[0] in mapping:
        return mapping[res[0]]
    return -1

def identity(res):
    return res

def first_char_after_anchor(anchor):
    def f(res):
        mapping = {'A':0, 'B':1, 'C':2, 'D':3, 'E':4}
        anchor_index = res.find(anchor)
        pred = -1  # if decoding failed, return -1
        if anchor_index >= 0:
            pred_letter = res[anchor_index+len(anchor)]
            if pred_letter in mapping:
                pred = mapping[pred_letter]
        return pred
    return f

def get_intervals_as_list(text):
    text = text.split('.')[0]
    text = text.strip()
    if text[-1] != ']':
        index = text.rfind(']')
        assert index > 0
        text = text[:index+1]
    interval_list_text = text.split('and')
    intervals = []
    for interval_text in interval_list_text:
        if ',' not in interval_text:
            intervals.append([0, 0])
            continue
        start_text, end_text = interval_text.split(',')
        start_text, end_text = start_text.strip(' []'), end_text.strip(' []')
        if start_text == 'None':
            start_text = '0'
        if end_text == 'None':
            end_text = '1'
        start, end = int(start_text), int(end_text)
        intervals.append([start, end])
    return intervals


class PromptTemplate(object):
    def __init__(self, head, template, post_process_fn):
        self.head = head
        self.prompt_template = template
        self.post_process_fn = post_process_fn

    def get_num_stages(self):
        return len(self.template)

    def get_template_str(self):
        template = []
        for temp in self.prompt_template:
            template.append(temp.safe_substitute())
        return template

    def fill(self, **kwargs):
        # match variable names: duration, narration, question, optionA, optionB, optionC, optionD, optionE, num_words
        prompt_filled = []


        if 'loc_pred' in kwargs and 'narration' in kwargs and kwargs['loc_pred'] is not None and kwargs['narration'] is not None:
            narration = kwargs['narration']

            # Find all occurrences of separators and maintain their positions
            # Use regex to keep the separators with the split parts
            parts = re.split(r'(#C|#O)', narration)
            
            # Recombine parts with their separators
            captions = []
            for i in range(1, len(parts), 2):
                if i + 1 < len(parts):
                    captions.append(parts[i] + parts[i + 1])

            # Extract relevant captions based on loc_pred indices
            loc_caption = [captions[i - 1] for i in kwargs['loc_pred'] if i > 0 and i <= len(captions)]

            # Join the relevant captions with "narration" label
            kwargs['narration'] = "narration " + "".join(loc_caption)
            print("kwargs['narration']", kwargs['narration'])

        for temp in self.prompt_template:
            prompt_filled.append(temp.substitute(kwargs))
        return prompt_filled


class PromptFactory(object):
    def __init__(self):
        self.prompt_templates = self.build()
    
    def build(self):
        prompt_templates = {}


        # egoschema relevance
        prompt_templates['cap_score'] = PromptTemplate(
            head = "You are presented with a textual description of a first view video clip, it consists of about 8 frame captions (might be slightly fewer) sparsely sampled from the video (#C means the first person view, and #O indicates another). The ultimate goal is to answer a question related to this video, choosing the correct option out of five possible answers. Please provide the answer with a single-letter (A, B, C, D, E)" + \
                        "It is crucial that you imagine the visual scene as vividly as possible to enhance the accuracy of your response. After selecting your answer, rate your confidence level in this choice on a scale from 1 to 100, where 1 indicates low confidence and 100 signifies high confidence. " + \
                        "Please provide a concise one-sentence explanation for your chosen answer. If you are uncertain about the correct option, select the one that seems closest to being correct. " + \
                        "Meanwhile, could you provide a relevance score for each frame caption to evaluate their relevance with the query-answering process. The score is between 1,2,3, where 1 indicates low relevance and 3 signifies high relevance. Please return the relevance score in the format of a list of 8 scores (if the caption is not enough of 8, then output the score of exact number of captions in the list).",
            template = [
                Template("Description: $narration \n\n###\n\n Questions: $question \n Options: \n A: $optionA \n B: $optionB \n C: $optionC \n D: $optionD \n E: $optionE \n\n###\n\n The prediction, explanation, confidence is (please response in the format of 'prediction: \n explanation: \n confidence: \n frame relevance: \n'):"),
            ],
            post_process_fn = first_char_as_answer
        )

        # egoschema QA
        prompt_templates['qa_standard'] = PromptTemplate(
            head = "You are presented with a textual description of a video clip, it consists of frame captions sparsely sampled from the video. Your task is to answer a question related to this video, choosing the correct option out of five possible answers. Please provide the answer with a single-letter (A, B, C, D, E)" + \
                        "It is crucial that you imagine the visual scene as vividly as possible to enhance the accuracy of your response. After selecting your answer, rate your confidence level in this choice on a scale from 1 to 100, where 1 indicates low confidence and 100 signifies high confidence. " + \
                        "Please provide a concise one-sentence explanation for your chosen answer. If you are uncertain about the correct option, select the one that seems closest to being correct. ",
            template = [
                Template("Here are a few examples. \n${examplars} \n\n###\n\n  Description: $narration \n\n###\n\n Questions: $question \n Options: \n A: $optionA \n B: $optionB \n C: $optionC \n D: $optionD \n E: $optionE \n\n###\n\n The prediction, explanation, confidence is (please response in the format of 'prediction: \n explanation: \n confidence: \n '):"),
            ],
            post_process_fn = first_char_as_answer
        )


    def get(self, prompt_type):
        return self.prompt_templates[prompt_type]
