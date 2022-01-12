import numpy as np
from scipy.sparse import csr_matrix
from scipy.optimize import fmin_l_bfgs_b
from collections import defaultdict
import pandas as pd


class MEMM:
    "this class creates an memm object"
    def __init__(self, data_set):
        self.data = data_set
        self.train = []
        self.test = []

        self.feature_vector_len = 0
        self.v = np.ones(self.feature_vector_len, dtype=np.float16)
        self.regular = 3  #lambda

        self.training_feature_matrix = csr_matrix([0])

        self.all_sentences_and_tags = []
        self.all_sentences = []
        self.all_tags = []
        self.all_tags_train = []
        self.tags_seen_on_train_for_word = {}
        self.all_sentences_train = []
        self.tags_we_saw_in_train = []
        self.all_ones_positions_in_the_feature_vector = {}

        self.all_tags_test = []
        self.all_sentences_test = []

        self.feature_100 = {}
        self.feature_101 = {}
        self.feature_102 = {}
        self.feature_103 = {}
        self.feature_104 = {}
        self.feature_105 = {}
        self.feature_106 = {}
        self.feature_107 = {}
        self.feature_108 = {}
        self.feature_109 = {}
        self.feature_first_word = {}
        self.feature_last_word = {}
        self.feature_start_upper = {}
        self.feature_middle_upper = {}
        self.feature_everything_upper = {}
        self.feature_digits = {}

        self.feature_100_mapping = {}
        self.feature_101_mapping = {}
        self.feature_102_mapping = {}
        self.feature_103_mapping = {}
        self.feature_104_mapping = {}
        self.feature_105_mapping = {}
        self.feature_106_mapping = {}
        self.feature_107_mapping = {}
        self.feature_108_mapping = {}
        self.feature_109_mapping = {}
        self.feature_first_word_mapping = {}
        self.feature_last_word_mapping = {}
        self.feature_start_upper_mapping = {}
        self.feature_middle_upper_mapping = {}
        self.feature_everything_upper_mapping = {}
        self.feature_digits_mapping = {}

        self.feature_mapping = {}

        self.test_predictions = []
        self.comp_predictions = []
        self.all_sentences_comp = []

    def initialize_train(self):
        "builds the data for train"
        for line in self.data:
            sentence = []
            tags = []
            splitted = line.split()  # split after space
            for word in splitted:
                word_and_Tag = word.split("_")
                sentence.append(word_and_Tag[0])
                tags.append(word_and_Tag[1])
            self.all_sentences_and_tags.append([sentence, tags])

        self.train = self.all_sentences_and_tags

        for words_and_tags in self.train:
            sentence = words_and_tags[0]
            tags = words_and_tags[1]
            for i in range(len(sentence)):
                "saving all tags observed in train"
                if tags[i] not in self.tags_we_saw_in_train:
                    self.tags_we_saw_in_train.append(tags[i])
                "we save all the tags we saw for each word in a dict - key: word, value: list of observed tags"
                if sentence[i].lower() not in self.tags_seen_on_train_for_word:
                    self.tags_seen_on_train_for_word[sentence[i].lower()] = []
                    self.tags_seen_on_train_for_word[sentence[i].lower()].append(tags[i])
                else:
                    if tags[i] not in self.tags_seen_on_train_for_word[sentence[i].lower()]:
                        self.tags_seen_on_train_for_word[sentence[i].lower()].append(tags[i])
            self.all_sentences_train.append(sentence)
            self.all_tags_train.append(tags)

        self.features_count()

    def features_count(self):
        "counting and saving the features from each type"
        for i in range(len(self.all_sentences_train)):
            for j in range(len(self.all_sentences_train[i])):
                word = self.all_sentences_train[i][j]
                word_lower = self.all_sentences_train[i][j].lower()
                y = self.all_tags_train[i][j]
                "starts with upper"
                if str.isupper(word[0]):
                    if y not in self.feature_start_upper:
                        self.feature_start_upper[y] = 1
                    else:
                        self.feature_start_upper[y] += 1
                "any upper without first lette"
                if any(str.isupper(w) for w in word[1:]):
                    if y not in self.feature_middle_upper:
                        self.feature_middle_upper[y] = 1
                    else:
                        self.feature_middle_upper[y] += 1
                "everything upper"
                if str.isupper(word):
                    if y not in self.feature_everything_upper:
                        self.feature_everything_upper[y] = 1
                    else:
                        self.feature_everything_upper[y] += 1
                "is digits"
                if str.isdigit(word):
                    if y not in self.feature_digits:
                        self.feature_digits[y] = 1
                    else:
                        self.feature_digits[y] += 1
                "f106"
                if j > 0:
                    w_prev = self.all_sentences_train[i][j-1].lower()
                    if (w_prev, y) not in self.feature_106:
                        self.feature_106[(w_prev, y)] = 1
                    else:
                        self.feature_106[(w_prev, y)] += 1
                "f107"
                if j < (len(self.all_sentences_train[i])-1):
                    w_next = self.all_sentences_train[i][j+1].lower()
                    if (w_next, y) not in self.feature_107:
                        self.feature_107[(w_next, y)] = 1
                    else:
                        self.feature_107[(w_next, y)] += 1
                "f108"
                if j > 1:
                    w_2_prev = self.all_sentences_train[i][j-2].lower()
                    if (w_2_prev, y) not in self.feature_108:
                        self.feature_108[(w_2_prev, y)] = 1
                    else:
                        self.feature_108[(w_2_prev, y)] += 1
                "f109"
                if j < (len(self.all_sentences_train[i])-2):
                    w_2_next = self.all_sentences_train[i][j+2].lower()
                    if (w_2_next, y) not in self.feature_109:
                        self.feature_109[(w_2_next, y)] = 1
                    else:
                        self.feature_109[(w_2_next, y)] += 1
                "first word + tag"
                if j == 0:
                    if y not in self.feature_first_word:
                        self.feature_first_word[y] = 1
                    else:
                        self.feature_first_word[y] += 1
                "last word + tag"
                if j == (len(self.all_sentences_train[i])-1):
                    if y not in self.feature_last_word:
                        self.feature_last_word[y] = 1
                    else:
                        self.feature_last_word[y] += 1

                "f100"
                if (word_lower, y) not in self.feature_100:  # see if the (word,tag) combination already exist, and if not add it
                    self.feature_100[(word_lower, y)] = 1
                else:
                    self.feature_100[(word_lower, y)] += 1
                "f101 suffix"
                for k in range(1, 5):
                    if len(word_lower) >= k:
                        if (word_lower[-k:], y) not in self.feature_101:
                            self.feature_101[(word_lower[-k:], y)] = 1
                        else:
                            self.feature_101[(word_lower[-k:], y)] += 1

                "f102 prefix"
                for k in range(1, 5):
                    if len(word_lower) >= k:
                        if (word_lower[:k], y) not in self.feature_102:
                            self.feature_102[(word_lower[:k], y)] = 1
                        else:
                            self.feature_102[(word_lower[:k], y)] += 1

        for i in range(len(self.all_tags_train)):  # one sentence tags
            for j in range(len(self.all_tags_train[i])):
                "f105 unigrams"
                if self.all_tags_train[i][j] not in self.feature_105:
                    self.feature_105[self.all_tags_train[i][j]] = 1
                else:
                    self.feature_105[self.all_tags_train[i][j]] += 1
                "f104 bigrams"
                if j > 0:
                    if (self.all_tags_train[i][j - 1], self.all_tags_train[i][j]) not in self.feature_104:
                        self.feature_104[(self.all_tags_train[i][j - 1], self.all_tags_train[i][j])] = 1
                    else:
                        self.feature_104[(self.all_tags_train[i][j - 1], self.all_tags_train[i][j])] += 1
                else:
                    if ('*', self.all_tags_train[i][j]) not in self.feature_104:
                        self.feature_104[('*', self.all_tags_train[i][j])] = 1
                    else:
                        self.feature_104[('*', self.all_tags_train[i][j])] += 1

                "f103 trigrams"
                if j > 1:
                    if (self.all_tags_train[i][j - 2], self.all_tags_train[i][j - 1], self.all_tags_train[i][j]) not in self.feature_103:
                        self.feature_103[(self.all_tags_train[i][j - 2], self.all_tags_train[i][j - 1], self.all_tags_train[i][j])] = 1
                    else:
                        self.feature_103[(self.all_tags_train[i][j - 2], self.all_tags_train[i][j - 1], self.all_tags_train[i][j])] += 1
                if j == 0:
                    if ('*', '*', self.all_tags_train[i][j]) not in self.feature_103:
                        self.feature_103[('*', '*', self.all_tags_train[i][j])] = 1
                    else:
                        self.feature_103[('*', '*', self.all_tags_train[i][j])] += 1
                if j == 1:
                    if ('*', self.all_tags_train[i][j - 1], self.all_tags_train[i][j]) not in self.feature_103:
                        self.feature_103[('*', self.all_tags_train[i][j - 1], self.all_tags_train[i][j])] = 1
                    else:
                        self.feature_103[('*', self.all_tags_train[i][j - 1], self.all_tags_train[i][j])] += 1
        self.define_features_mapping()

    def define_features_mapping(self):
        "cutting the number of features from each type according to threshold, defines mapping between features and numbers"
        i = 0
        f100_size = 0
        for key, value in self.feature_100.items():
            if value >= 1:
                self.feature_100_mapping[key] = i
                i += 1
                f100_size += 1
        print('f_100 size :' + str(f100_size))
        f101_size = 0
        for key, value in self.feature_101.items():
            if value >= 1:
                self.feature_101_mapping[key] = i
                f101_size += 1
        print('f_101 size :' + str(f101_size))
        f102_size = 0
        for key, value in self.feature_102.items():
            if value >= 1:
                self.feature_102_mapping[key] = i
                i += 1
                f102_size += 1
        print('f_102 size :' + str(f102_size))
        f103_size = 0
        for key, value in self.feature_103.items():
            if value >= 1:
                self.feature_103_mapping[key] = i
                i += 1
                f103_size += 1
        print('f_103 size :' + str(f103_size))
        f104_size = 0
        for key, value in self.feature_104.items():
            if value >= 1:
                self.feature_104_mapping[key] = i
                i += 1
                f104_size += 1
        print('f_104 size :' + str(f104_size))
        f105_size = 0
        for key, value in self.feature_105.items():
            if value >= 1:
                self.feature_105_mapping[key] = i
                i += 1
                f105_size += 1
        print('f_105 size :' + str(f105_size))
        f106_size = 0
        for key, value in self.feature_106.items():
            if value >= 1:
                self.feature_106_mapping[key] = i
                i += 1
                f106_size += 1
        print('f_106 size :' + str(f106_size))
        f107_size = 0
        for key, value in self.feature_107.items():
            if value >= 1:
                self.feature_107_mapping[key] = i
                i += 1
                f107_size += 1
        print('f_107 size :' + str(f107_size))
        f108_size = 0
        for key, value in self.feature_108.items():
            if value >= 1:
                self.feature_108_mapping[key] = i
                i += 1
                f108_size += 1
        print('f_108 size :' + str(f108_size))
        f109_size = 0
        for key, value in self.feature_109.items():
            if value >= 1:
                self.feature_109_mapping[key] = i
                i += 1
                f109_size += 1
        print('f_109 size :' + str(f109_size))
        first_word = 0
        for key, value in self.feature_first_word.items():
            if value >= 1:
                self.feature_first_word_mapping[key] = i
                i += 1
                first_word += 1
        print('first word size :' + str(first_word))
        last_word = 0
        for key, value in self.feature_last_word.items():
            if value >= 1:
                self.feature_last_word_mapping[key] = i
                i += 1
                last_word += 1
        print('last word size :' + str(last_word))
        start_up = 0
        for key, value in self.feature_start_upper.items():
            if value >= 1:
                self.feature_start_upper_mapping[key] = i
                i += 1
                start_up += 1
        print('start with upper size :' + str(start_up))
        middle_up = 0
        for key, value in self.feature_middle_upper.items():
            if value >= 1:
                self.feature_middle_upper_mapping[key] = i
                i += 1
                middle_up += 1
        print('middle upper size :' + str(middle_up))
        any_up = 0
        for key, value in self.feature_everything_upper.items():
            if value >= 1:
                self.feature_everything_upper_mapping[key] = i
                i += 1
                any_up += 1
        print('any upper size :' + str(any_up))
        digits_size = 0
        for key, value in self.feature_digits.items():
            if value >= 1:
                self.feature_digits[key] = i
                i += 1
                digits_size += 1
        print('digits size :' + str(digits_size))
        self.feature_vector_len = i
        #print('number of features: ' + str(i))
        self.build_features_matrix_train()

    def build_features_matrix_train(self):
        "builds feature matrix for each word with each tag"
        training_matrix_rows_index_counter = 0
        training_matrix_rows_index = []
        training_matrix_columns_index = []
        for i in range(len(self.all_sentences_train)):  #for each sentence in train
            sentence = self.all_sentences_train[i]
            tags = self.all_tags_train[i]

            for word_index in range(len(sentence)):     #for each word in sentence
                word_matrix_rows_index_counter = 0
                word_matrix_rows_index = []
                word_matrix_columns_index = []

                origin_tag = tags[word_index]
                origin_word = sentence[word_index]
                word = origin_word.lower()
                for label in self.tags_we_saw_in_train: #for each tag we saw in the train set
                    columns = []
                    if word_index == 0:
                        label_1 = '*'
                        label_2 = '*'
                        "first word + tag"
                        if label in self.feature_first_word_mapping:
                            word_matrix_rows_index.append(word_matrix_rows_index_counter)
                            columns.append(self.feature_first_word_mapping[label])
                    "last word + tag"
                    if word_index == (len(sentence)-1):
                        if label in self.feature_last_word_mapping:
                            word_matrix_rows_index.append(word_matrix_rows_index_counter)
                            columns.append(self.feature_last_word_mapping[label])
                    "f108"
                    if word_index > 1:
                        if (sentence[word_index-2].lower(), label) in self.feature_108_mapping:
                            word_matrix_rows_index.append(word_matrix_rows_index_counter)
                            columns.append(self.feature_108_mapping[(sentence[word_index-2].lower(), label)])
                    "f109"
                    if word_index < (len(sentence)-2):
                        if (sentence[word_index+2].lower(), label) in self.feature_109_mapping:
                            word_matrix_rows_index.append(word_matrix_rows_index_counter)
                            columns.append(self.feature_109_mapping[(sentence[word_index+2].lower(), label)])
                    "f100"
                    if word in self.tags_seen_on_train_for_word:
                        if (word, label) in self.feature_100_mapping:
                            word_matrix_rows_index.append(word_matrix_rows_index_counter)
                            columns.append(self.feature_100_mapping[(word, label)])
                    "f103"
                    if (label_2, label_1, label) in self.feature_103_mapping:
                        word_matrix_rows_index.append(word_matrix_rows_index_counter)
                        columns.append(self.feature_103_mapping[(label_2, label_1, label)])
                    "f104"
                    if (label_1, label) in self.feature_104_mapping:
                        word_matrix_rows_index.append(word_matrix_rows_index_counter)
                        columns.append(self.feature_104_mapping[(label_1, label)])
                    "f105"
                    if label in self.feature_105_mapping:
                        word_matrix_rows_index.append(word_matrix_rows_index_counter)
                        columns.append(self.feature_105_mapping[label])
                    "f101"
                    prefix_to_check = [word[:i] for i in range(1, 5)]
                    prefix_to_check = list(set(prefix_to_check))
                    for prefix in prefix_to_check:
                        if (prefix, label) in self.feature_101_mapping:
                            word_matrix_rows_index.append(word_matrix_rows_index_counter)
                            columns.append(self.feature_101_mapping[(prefix, label)])
                    "f102"
                    suffix_to_check = [word[i:] for i in range(-4, 0)]
                    suffix_to_check = list(set(suffix_to_check))
                    for suffix in suffix_to_check:
                        if suffix in self.feature_102_mapping:
                            word_matrix_rows_index.append(word_matrix_rows_index_counter)
                            columns.append(self.features_position[suffix])

                    "f106"
                    if word_index != 0:
                        if (sentence[word_index-1].lower(), label) in self.feature_106_mapping:
                            word_matrix_rows_index.append(word_matrix_rows_index_counter)
                            columns.append(self.feature_106_mapping[(sentence[word_index-1].lower(), label)])
                    "f107"
                    if word_index != (len(sentence)-1):
                        if (sentence[word_index+1].lower(), label) in self.feature_107_mapping:
                            word_matrix_rows_index.append(word_matrix_rows_index_counter)
                            columns.append(self.feature_107_mapping[(sentence[word_index+1].lower(), label)])
                    "is digits"
                    if str.isdigit(word):
                        if label in self.feature_digits_mapping:
                            word_matrix_rows_index.append(word_matrix_rows_index_counter)
                            columns.append(self.feature_digits_mapping[label])
                    "starts with capital"
                    if str.isupper(origin_word[0]):
                        if label in self.feature_start_upper_mapping:
                            word_matrix_rows_index.append(word_matrix_rows_index_counter)
                            columns.append(self.feature_start_upper_mapping[label])
                    "middle upper"
                    if any(str.isupper(w) for w in origin_word[1:]):
                        if label in self.feature_middle_upper_mapping:
                            word_matrix_rows_index.append(word_matrix_rows_index_counter)
                            columns.append(self.feature_middle_upper_mapping[label])
                    "everything upper"
                    if str.isupper(origin_word):
                        if label in self.feature_everything_upper_mapping:
                            word_matrix_rows_index.append(word_matrix_rows_index_counter)
                            columns.append(self.feature_everything_upper_mapping[label])
                    word_matrix_columns_index += columns
                    word_matrix_rows_index_counter += 1
                    if label == origin_tag:
                        temp = list(np.ones(len(columns), dtype=np.int32) * training_matrix_rows_index_counter)
                        training_matrix_rows_index += temp
                        training_matrix_columns_index += columns
                        training_matrix_rows_index_counter += 1
                rows_index = np.asarray(a=word_matrix_rows_index, dtype=np.int32)
                cols_index = np.asarray(a=word_matrix_columns_index, dtype=np.int32)
                data_to_insert = np.ones(len(word_matrix_rows_index), dtype=np.int8)
                word_matrix = csr_matrix((data_to_insert, (rows_index, cols_index)),
                                         shape=(len(self.tags_we_saw_in_train), self.feature_vector_len))

                # Initialize the dict where key : (sentence,word), value : word_matrix
                self.all_ones_positions_in_the_feature_vector[(i, word_index)] = word_matrix
                label_2 = label_1
                label_1 = origin_tag

        rows_index = np.asarray(a=training_matrix_rows_index, dtype=np.int32)
        cols_index = np.asarray(a=training_matrix_columns_index, dtype=np.int32)
        data = np.ones(len(rows_index), dtype=np.int8)
        "creating the sparse matrix for training - for each word with each possible tag"
        self.training_feature_matrix = csr_matrix((data, (rows_index, cols_index)),
                                          shape=(training_matrix_rows_index_counter, self.feature_vector_len))
        self.find_max()

    def log_likelihood(self, v):
        "builds the log likelihood L(v)"
        l_func_linear_part = np.sum(self.training_feature_matrix.dot(v.transpose()))
        sum_log_vector = []
        for (i, j) in self.all_ones_positions_in_the_feature_vector:
            matrix = self.all_ones_positions_in_the_feature_vector[(i, j)]
            sum_log_vector.append(np.log(np.sum(np.exp(matrix.dot(v)))))
        l_func_exp_part = np.sum(sum_log_vector)
        regularization = self.regular * 0.5 * np.dot(v, v)
        return -(l_func_linear_part - l_func_exp_part - regularization)

    def log_likelihood_derivative(self, v):
        "builds the derivation for L(v)"
        empirical_counts = np.squeeze(np.asarray(csr_matrix.sum(self.training_feature_matrix, axis=0)))
        "expected counts"
        expected_counts = np.zeros(self.feature_vector_len)
        for (i, j) in self.all_ones_positions_in_the_feature_vector:
            mat = self.all_ones_positions_in_the_feature_vector[(i, j)]
            nominators = np.exp(mat.dot(v))
            denominator = np.sum(nominators)
            prob = nominators / denominator
            expected_counts += mat.transpose().dot(prob)
        return -(empirical_counts - expected_counts - self.regular * v)

    def find_max(self):
        "runs all procedure to find the best v"
        result = fmin_l_bfgs_b(func=self.log_likelihood,
                               x0=np.ones(shape=self.feature_vector_len, dtype=np.float16) * 0.1,
                               fprime=self.log_likelihood_derivative,
                               factr=1e12,
                               pgtol=1e-3)
        self.v = result[0]

    def initialize_test(self):
        "builds the test data for test"
        for words_and_tags in self.test:
            sentence = words_and_tags[0]
            tags = words_and_tags[1]
            self.all_sentences_test.append(sentence)
            self.all_tags_test.append(tags)
        self.inference_procedure()

    def find_feature_vector_test_set(self, sentence, k, t, u, v):
        "for each word we will find the feature vector"
        curr_word = sentence[k - 1]
        curr_word_lower = curr_word.lower()
        columns = []
        # f100
        if (curr_word_lower, v) in self.feature_100_mapping:
            columns.append(self.feature_100_mapping[(curr_word_lower, v)])
        for j in range(1, 5):
            if len(curr_word_lower) >= j:
                # f101
                if (curr_word_lower[-j:], v) in self.feature_101_mapping:
                    columns.append(self.feature_101_mapping[(curr_word_lower[-j:], v)])
                # f102
                if (curr_word_lower[:j], v) in self.feature_102_mapping:
                    columns.append(self.feature_102_mapping[(curr_word_lower[:j], v)])
        # f103
        if (t, u, v) in self.feature_103_mapping:
            columns.append(self.feature_103_mapping[(t, u, v)])
        # f104
        if (u, v) in self.feature_104_mapping:
            columns.append(self.feature_104_mapping[(u, v)])
        # f105
        if v in self.feature_105_mapping:
            columns.append(self.feature_105_mapping[v])
        #f106
        if k > 1:
            if (sentence[k-2].lower(), v) in self.feature_106_mapping:
                columns.append(self.feature_106_mapping[(sentence[k-2].lower(), v)])
        #f107
        if k < len(sentence):
            if (sentence[k].lower(), v) in self.feature_107_mapping:
                columns.append(self.feature_107_mapping[(sentence[k].lower(), v)])
        #f108
        if k > 2:
            if (sentence[k-3].lower(), v) in self.feature_108_mapping:
                columns.append(self.feature_108_mapping[(sentence[k-3].lower(), v)])
        #f109
        if k < (len(sentence)-1):
            if (sentence[k+1].lower(), v) in self.feature_109_mapping:
                columns.append(self.feature_109_mapping[(sentence[k+1].lower(), v)])
        #first word and tag
        if k == 1:
            if v in self.feature_first_word_mapping:
                columns.append(self.feature_first_word_mapping[v])
        #last word and tag
        if k == len(sentence):
            if v in self.feature_last_word_mapping:
                columns.append(self.feature_last_word_mapping[v])

        # rest of the features
        if str.isupper(curr_word[0]):
            if v in self.feature_start_upper_mapping:
                columns.append(self.feature_start_upper_mapping[v])
        if any(str.isupper(w) for w in curr_word[1:]):
            if v in self.feature_middle_upper_mapping:
                columns.append(self.feature_middle_upper_mapping[v])
        if str.isupper(curr_word):
            if v in self.feature_everything_upper_mapping:
                columns.append(self.feature_everything_upper_mapping[v])
        if str.isdigit(curr_word_lower):
            if v in self.feature_digits_mapping:
                columns.append(self.feature_digits_mapping[v])
        return columns

    def probability(self, S_k_2, S_k_1, S_k, sentence, k):
        "calculates the probability for each word position and all the trigram options"
        probability_table = defaultdict(tuple)
        weights = self.v
        for t in S_k_2:
            for u in S_k_1:
                for v in S_k:
                    probability_table[(t, u, v)] = np.exp(sum(weights[self.find_feature_vector_test_set(sentence, k, t, u, v)]))

                # Constant Denominator
                denominator = np.sum(probability_table[(t, u, v)] for v in S_k)
                for v in S_k:
                    probability_table[(t, u, v)] /= denominator
        return probability_table

    def viterbi(self, sentence):
        "for each sentence we will find the most reasonable POS tags"
        pie = {}
        bp = {}  # back pointers
        pie[(0, "*", "*")] = 1.0    # initialization
        for k in range(1, len(sentence) + 1):
            current_word = sentence[k - 1]
            S_k_1 = self.tags_we_saw_in_train
            S_k_2 = self.tags_we_saw_in_train
            "we will check if we saw the word with any tags in the train, if we saw we will take the group of tags we saw"
            if current_word.lower in self.tags_seen_on_train_for_word:
                S_k = self.tags_seen_on_train_for_word[current_word.lower()]
            else:
                S_k = self.tags_we_saw_in_train
            if k == 1:
                S_k_1, S_k_2 = ["*"], ["*"]
            elif sentence[k - 2].lower() in self.tags_seen_on_train_for_word:
                S_k_1 = self.tags_seen_on_train_for_word[sentence[k - 2].lower()]
            if k == 2:
                S_k_2 = ["*"]
            elif k > 2 and sentence[k - 3].lower() in self.tags_seen_on_train_for_word:
                S_k_2 = self.tags_seen_on_train_for_word[sentence[k - 3].lower()]

            "we will build the probabilities"
            probabilities = self.probability(S_k_2, S_k_1, S_k, sentence, k)

            "now we run the Viterbi algorithm with back pointers"
            for u in S_k_1:
                for v in S_k:
                    pie_max = 0
                    bp_max = None
                    for t in S_k_2:
                        pie_temp = pie[(k - 1, t, u)] * probabilities[(t, u, v)]

                        if pie_temp > pie_max:
                            pie_max = pie_temp
                            bp_max = t
                    pie[(k, u, v)] = pie_max
                    bp[(k, u, v)] = bp_max
        t = {}
        n = len(sentence)
        pie_max = 0
        for u in S_k_1:
            for v in S_k:
                curr_pie = pie[(n, u, v)]
                if curr_pie > pie_max:
                    pie_max = curr_pie
                    t[n] = v
                    t[n - 1] = u
        for k in range(n - 2, 0, -1):
            t[k] = bp[k + 2, t[k + 1], t[k + 2]]
        tag_sequence_1 = []
        for i in t:
            tag_sequence_1.append(t[i])
        if n == 1:
            tag_sequence_1 = [tag_sequence_1[n]]
        tag_sequence = list(reversed(tag_sequence_1))
        return tag_sequence

    def calculate_accuracy(self, true_tags_list, pred_tags_list):
        counter_true = 0
        counter = 0
        for i in range(len(true_tags_list)):
            for j in range(len(true_tags_list[i])):
                counter += 1
                if true_tags_list[i][j] == pred_tags_list[i][j]:
                    counter_true += 1
        acc_chen = counter_true/counter
        print('accuracy: ' + str(acc_chen))

    def make_confusion_matrix(self, x_true, x_pred):
        "printing confusion matrix"
        x_true_all = []
        x_pred_all = []
        for line in x_true:
            for tag in line:
                x_true_all.append(tag)
        for line in x_pred:
            for tag in line:
                x_pred_all.append(tag)
        tags = []
        tags = list(set(x_true_all + x_pred_all))
        tag2inx = {tag: i for (i, tag) in enumerate(tags)}
        mat = np.zeros((len(tags), len(tags)))
        mat_new = {}
        for i in range(len(x_true_all)):
            true_inx = tag2inx[x_true_all[i]]
            pred_inx = tag2inx[x_pred_all[i]]
            mat[true_inx, pred_inx] += 1
            if x_true_all[i] != x_pred_all[i]:
                if (x_true_all[i], x_pred_all[i]) not in mat_new:
                    mat_new[(x_true_all[i], x_pred_all[i])] = 1
                else:
                    mat_new[(x_true_all[i], x_pred_all[i])] += 1

        tags_print = tag2inx.keys()
        print(pd.DataFrame(mat, columns=tags_print, index=tags_print))
        pd.DataFrame(mat, columns=tags_print, index=tags_print).to_csv('conf_matrix_model_2', sep='\t')
        n = 10
        {key: mat_new[key] for key in sorted(mat_new, key=mat_new.get, reverse=True)[:n]}
        for key, value in mat_new.items():
            print(str(key) + ': ' + str(value))


    def write_predictions_to_file(self):
        "writes the predictions for comp. file to wtag file"
        file_name = 'comp_m2_303091698.wtag'
        with open(file_name, "a") as new_file:
            for sentence_index in range(len(self.all_sentences_comp)):
                text_to_write = ""
                for word_index in range(len(self.all_sentences_comp[sentence_index])):
                    pair = self.all_sentences_comp[sentence_index][word_index] + "_" + \
                           self.comp_predictions[sentence_index][word_index]
                    if word_index != len(self.all_sentences_comp[sentence_index]) - 1:
                        text_to_write += pair + " "
                    else:
                        text_to_write += pair + "\n"
                        new_file.write(text_to_write)

    def inference_procedure(self):
        "runs inference for test"
        test_tags = []
        for sentence in self.all_sentences_test:
            tags_sequence = self.viterbi(sentence)
            test_tags.append(tags_sequence)
        self.test_predictions = test_tags
        self.calculate_accuracy(self.all_tags_test, test_tags)
        #self.make_confusion_matrix(self.all_tags_test, test_tags)

    def inference_for_comp(self, comp_data):
        "runs inference for the comp. file"
        comp_tags = []
        for line in comp_data:
            words = line.split()
            new_sentence = []
            for word in words:
                new_sentence.append(word)
            self.all_sentences_comp.append(new_sentence)

        for sentence_to_tag in self.all_sentences_comp:
            tags_sequence = self.viterbi(sentence_to_tag)
            comp_tags.append(tags_sequence)

        self.comp_predictions = comp_tags
        self.write_predictions_to_file()






