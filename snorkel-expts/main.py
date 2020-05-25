import jsonlines
import sys
import copy
from nltk.tree import ParentedTree
from nltk.tree import Tree
import nltk
import json
from wiser.eval import score_linking_rules
from wiser.rules import ElmoLinkingRule
from wiser.eval import score_tagging_rules
from wiser.rules import TaggingRule
from wiser.data.dataset_readers import StoriesDatasetReader
from wiser.rules import DictionaryMatcher

dataset_reader = StoriesDatasetReader()
train_data = dataset_reader.read(
    'cd_connlu/data/unlabelled_conandoyle_train.csv')
dev_data = dataset_reader.read('cd_connlu/data/labelled_conandoyle_dev.csv')
test_data = dataset_reader.read('cd_connlu/data/labelled_conandoyle_test.csv')

# We must merge all partitions to apply the rules
data = train_data + dev_data + test_data

# TODO remove cd_connlu when moving to notebook
nooj_tagged_file = "nooj-{0}.json"

# Opening JSON file
with open(nooj_tagged_file.format("train")) as file:
    nooj_train_data = json.load(file)
with open(nooj_tagged_file.format("dev")) as file:
    nooj_dev_data = json.load(file)
with open(nooj_tagged_file.format("test")) as file:
    nooj_test_data = json.load(file)

# RULE 1. MARK ALL NEGATION CUES.
common_true_positives_implicits_cue = [['not'], ['no'], ['n\'t'],
                                       ['never'], ['absence'], ['without'],
                                       ['nobody'], ['nowhere'], ['nothing'],
                                       ["except"], ["fail"], ['none'],
                                       ['neither'], ['nor']
                                       ]

tr = DictionaryMatcher("CommonTruePositivesImplicitsCue",
                       terms=common_true_positives_implicits_cue,
                       i_label="I-cue",
                       uncased=True)
tr.apply(data)


def scope_candidate(parse_strings, neg_cue_word=None):
    candidates = []
    for parse_string in parse_strings:
        ptree = ParentedTree.fromstring(parse_string)
        leaf_values = ptree.leaves()
        print(leaf_values)
        if neg_cue_word in leaf_values:
            leaf_index = leaf_values.index(neg_cue_word)
            tree_location = ptree.leaf_treeposition(leaf_index)
            print(tree_location)
            print(ptree[tree_location])
            candidates.append(set(ptree[tree_location[:-3]].leaves()))

    return candidates


class NoojLabels(TaggingRule):
    def apply_instance(self, instance):

        tokens = [t.text for t in instance['tokens']]
        labels = ['ABS'] * len(tokens)

        nooj_data = []
        key = " ".join(tokens)

        if key in nooj_train_data:
            nooj_data = nooj_train_data[key]
        elif key in nooj_dev_data:
            nooj_data = nooj_dev_data[key]
        elif key in nooj_test_data:
            nooj_data = nooj_test_data[key]
        if nooj_data:
            for i in range(len(tokens)):
                labels[i] = nooj_data[i]
        return labels


tr = NoojLabels()
# tr.apply(data)

print(score_tagging_rules(dev_data))

# Links tokens whose cosine similarity is larger than 0.8
# lr = ElmoLinkingRule(0.8)
# lr.apply(data)

# print(score_linking_rules(dev_data))


class ScopeConstituencyKeywords(TaggingRule):
    def __init__(self):
        super().__init__()
        # TODO change this once we run full scale
        self.parsed_data = {}
        with jsonlines.open('nooj-wiser-expts/spacy_features.jsonl') as reader:
            for obj in reader:
                for key, value in (obj.items()):
                    self.parsed_data[key] = value

    def apply_instance(self, instance):

        tokens = [t.text for t in instance['tokens']]
        labels = ['ABS'] * len(tokens)

        implicit_cue_positives = [
            t for t in instance['WISER_LABELS']['CommonTruePositivesImplicitsCue']]
        # a.extend(['foo', 'bar'])
        parse_string = None
        sentence = " ".join(tokens)
        if sentence in self.parsed_data:
            parse_strings = self.parsed_data[sentence]["parse_string"]
        else:
            sys.exit("parse string not found in the feature store")

        implicit_cue_indices = [index for index, value in enumerate(
            implicit_cue_positives) if value == "I-cue"]
        for implicit_cue_index in implicit_cue_indices:
            # pass the parse strings list and the neg_cue token
            scope_candidates = scope_candidate(
                parse_strings, tokens[implicit_cue_index])
            # check if you got something
            if scope_candidates:
                # loop through the candidates
                for candidates in scope_candidates:
                    # get the window length
                    candidate_len = len(candidates)
                    # create a window around the index where the neg-cue is found
                    start_index = implicit_cue_index-candidate_len
                    end_index = implicit_cue_index+candidate_len
                    if implicit_cue_index-candidate_len < 0:
                        star_index = 0
                    if implicit_cue_index+candidate_len > len(tokens):
                        end_index = len(tokens)
                    c_tokens = copy.deepcopy(tokens[start_index:end_index])
                    for c_index, c in enumerate(c_tokens):
                        # check if the c_tokens have any punctuations stop the process if any
                        if c in candidates and implicit_cue_positives[start_index+c_index] != "I-cue":
                            labels[start_index+c_index] = "I-scope"

        # for i in range(len(tokens)):
        #     if implicit_cue_positives[i] == "I-cue":

        #         candidates = scope_candidate(parse_strings, tokens[i])
        #         if candidates:
        #             if len(candidates)>1:
        #                 input("hello")
        #             # get the window length
        #             candidates = candidates[0]
        #             candidate_len = len(candidates)
        #             # create a window around the index where the neg-cue is found
        #             c_tokens = copy.deepcopy(
        #                 tokens[i-candidate_len:i+candidate_len])

        #             for c_index, c in enumerate(c_tokens):
        #                 # check if the c_tokens have any punctuations stop the process if any
        #                 if c in candidates and implicit_cue_positives[i-candidate_len+c_index] != "I-cue":
        #                     labels[i-candidate_len+c_index] = "I-scope"
        return labels


tr = ScopeConstituencyKeywords()
tr.apply(data)

print(score_tagging_rules(dev_data))
