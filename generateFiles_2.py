from chu_liu import Digraph
import numpy as np
from scipy.sparse import csr_matrix
from collections import defaultdict
import time


class Depndency_Parser:
    "creates dependency parser object"
    def __init__(self, mode):
        self.mode = mode
        self.feature_vec_len = 0
        self.w_vector = np.zeros(shape=self.feature_vec_len, dtype=np.float16)
        self.mst_dict = defaultdict(int)
        self.mst_dict_comp = defaultdict(int)
        self.train_feature_matrix = defaultdict(tuple)
        self.features_matrix = defaultdict(tuple)

        self.all_sentences_train = []
        self.all_sentences_test = []
        self.all_sentences_comp = []

        self.feature_1 = {}
        self.feature_2 = {}
        self.feature_3 = {}
        self.feature_4 = {}
        self.feature_5 = {}
        self.feature_6 = {}
        self.feature_7 = {}
        self.feature_8 = {}
        self.feature_9 = {}
        self.feature_10 = {}
        self.feature_11 = {}
        self.feature_12 = {}
        self.feature_13 = {}
        self.feature_14 = {}
        self.feature_15 = {}
        self.feature_16 = {}
        self.feature_17 = {}
        self.feature_18 = {}
        self.feature_19 = {}
        self.feature_20 = {}
        self.feature_21 = {}
        self.feature_22 = {}
        self.feature_23 = {}
        self.feature_24 = {}
        self.feature_25 = {}
        self.feature_26 = {}
        self.feature_27 = {}
        self.feature_28 = {}
        self.feature_29 = {}
        self.feature_30 = {}
        self.feature_31 = {}
        self.feature_32 = {}
        self.feature_33 = {}
        self.feature_34 = {}
        self.feature_35 = {}
        self.feature_36 = {}
        self.feature_37 = {}
        self.feature_38 = {}
        self.feature_39 = {}
        self.feature_40 = {}
        self.feature_41 = {}
        self.feature_42 = {}

        self.feature_1_mapping = {}
        self.feature_2_mapping = {}
        self.feature_3_mapping = {}
        self.feature_4_mapping = {}
        self.feature_5_mapping = {}
        self.feature_6_mapping = {}
        self.feature_7_mapping = {}
        self.feature_8_mapping = {}
        self.feature_9_mapping = {}
        self.feature_10_mapping = {}
        self.feature_11_mapping = {}
        self.feature_12_mapping = {}
        self.feature_13_mapping = {}
        self.feature_14_mapping = {}
        self.feature_15_mapping = {}
        self.feature_16_mapping = {}
        self.feature_17_mapping = {}
        self.feature_18_mapping = {}
        self.feature_19_mapping = {}
        self.feature_20_mapping = {}
        self.feature_21_mapping = {}
        self.feature_22_mapping = {}
        self.feature_23_mapping = {}
        self.feature_24_mapping = {}
        self.feature_25_mapping = {}
        self.feature_26_mapping = {}
        self.feature_27_mapping = {}
        self.feature_28_mapping = {}
        self.feature_29_mapping = {}
        self.feature_30_mapping = {}
        self.feature_31_mapping = {}
        self.feature_32_mapping = {}
        self.feature_33_mapping = {}
        self.feature_34_mapping = {}
        self.feature_35_mapping = {}
        self.feature_36_mapping = {}
        self.feature_37_mapping = {}
        self.feature_38_mapping = {}
        self.feature_39_mapping = {}
        self.feature_40_mapping = {}
        self.feature_41_mapping = {}
        self.feature_42_mapping = {}


    def build_data(self):
        "create lists for the sentences in each file"
        with open('train.labeled', 'r') as file:
            # isreting root
            root_node = [0, 'Root', '*', 0]
            new_sentence = [root_node]
            for line in file:   #new line
                if line in ['\r\n', '\n']:
                    self.all_sentences_train.append(new_sentence)
                    new_sentence = [root_node]
                else:
                    word_data = line.split('\t')
                    temp_word = [int(word_data[0]), word_data[1], word_data[3], int(word_data[6])]  #[token num, token name, token POS, token head]
                    new_sentence.append(temp_word)

        with open('test.labeled', 'r') as file:
            # inserting root
            root_node = [0, 'Root', '*', 0]
            new_sentence = [root_node]
            for line in file:   #new line
                if line in ['\r\n', '\n']:
                    self.all_sentences_train.append(new_sentence)
                    new_sentence = [root_node]
                else:
                    word_data = line.split('\t')
                    temp_word = [int(word_data[0]), word_data[1], word_data[3], int(word_data[6])]  #[token name, token POS, token head]
                    new_sentence.append(temp_word)

        with open('comp.unlabeled', 'r') as file:
            # inserting root
            root_node = [0, 'Root', '*', 0]
            new_sentence = [root_node]
            for line in file:   #new line
                if line in ['\r\n', '\n']:
                    self.all_sentences_comp.append(new_sentence)
                    new_sentence = [root_node]
                else:
                    word_data = line.split('\t')
                    temp_word = [int(word_data[0]), word_data[1], word_data[3]]  #[token name, token POS]
                    new_sentence.append(temp_word)

    def build_features_dicts(self):
        "builds feature mapping"
        #first we count them
        for sentence in self.all_sentences_train:
            for word_list in sentence:
                modifier = word_list[0]
                #if modifier is the root continue
                if modifier == 0:
                    continue
                c_word = word_list[1].lower()
                c_pos = word_list[2]
                head = word_list[3]
                p_word = sentence[head][1].lower()
                p_pos = sentence[head][2]

                distance_sign = np.sign(modifier - head)
                distance = abs(modifier - head)

                between = ''
                if distance_sign == 1:
                    for j in range(1, distance - 1):
                        between += "_" + sentence[head + j][2]
                else:
                    for j in range(1, distance - 1):
                        between += "_" + sentence[modifier + j][2]

                c_r_pos = 'STOP'
                c_2r_pos = 'STOP'
                c_l_pos = '**'
                c_2l_pos = '**'
                p_r_pos = 'STOP'
                p_2r_pos = 'STOP'
                p_l_pos = '**'
                p_2l_pos = '**'

                # left word POS of modifier in the sentence
                if modifier > 1:
                    c_l_pos = sentence[modifier - 1][2]
                    if modifier > 2:
                        c_2l_pos = sentence[modifier - 2][2]
                # right word POS of modifier in the sentence
                if modifier < len(sentence) - 1:
                    c_r_pos = sentence[modifier + 1][2]
                    if modifier < len(sentence) - 2:
                        c_2r_pos = sentence[modifier + 2][2]

                # left word POS of head in the sentence
                if head > 1:
                    p_l_pos = sentence[head - 1][2]
                    if head > 2:
                        p_2l_pos = sentence[head - 2][2]
                # right word POS of head in the sentence
                if head < len(sentence) - 1:
                    p_r_pos = sentence[head + 1][2]
                    if head < len(sentence) - 2:
                        p_2r_pos = sentence[head + 2][2]

                if (p_word, p_pos) not in self.feature_1:
                    self.feature_1[(p_word, p_pos)] = 1
                else:
                    self.feature_1[(p_word, p_pos)] += 1

                if p_word not in self.feature_2:
                    self.feature_2[p_word] = 1
                else:
                    self.feature_2[p_word] += 1

                if p_pos not in self.feature_3:
                    self.feature_3[p_pos] = 1
                else:
                    self.feature_3[p_pos] += 1

                if (c_word, c_pos) not in self.feature_4:
                    self.feature_4[(c_word, c_pos)] = 1
                else:
                    self.feature_4[(c_word, c_pos)] += 1

                if c_word not in self.feature_5:
                    self.feature_5[c_word] = 1
                else:
                    self.feature_5[c_word] += 1

                if c_pos not in self.feature_6:
                    self.feature_6[c_pos] = 1
                else:
                    self.feature_6[c_pos] += 1

                if (p_pos, c_word, c_pos) not in self.feature_8:
                    self.feature_8[(p_pos, c_word, c_pos)] = 1
                else:
                    self.feature_8[(p_pos, c_word, c_pos)] += 1

                if (p_word, p_pos, c_pos) not in self.feature_10:
                    self.feature_10[(p_word, p_pos, c_pos)] = 1
                else:
                    self.feature_10[(p_word, p_pos, c_pos)] += 1

                if (p_pos, c_pos) not in self.feature_13:
                    self.feature_13[(p_pos, c_pos)] = 1
                else:
                    self.feature_13[(p_pos, c_pos)] += 1

                if self.mode == 'advanced':
                    if (p_word, p_pos, c_word, c_pos) not in self.feature_7:
                        self.feature_7[(p_word, p_pos, c_word, c_pos)] = 1
                    else:
                        self.feature_7[(p_word, p_pos, c_word, c_pos)] += 1

                    if (p_word, c_word, c_pos) not in self.feature_9:
                        self.feature_9[(p_word, c_word, c_pos)] = 1
                    else:
                        self.feature_9[(p_word, c_word, c_pos)] += 1

                    if (p_word, p_pos, c_word) not in self.feature_11:
                        self.feature_11[(p_word, p_pos, c_word)] = 1
                    else:
                        self.feature_11[(p_word, p_pos, c_word)] += 1

                    if (p_word, c_word) not in self.feature_12:
                        self.feature_12[(p_word, c_word)] = 1
                    else:
                        self.feature_12[(p_word, c_word)] += 1

                    if (p_word, p_pos, p_r_pos) not in self.feature_14:
                        self.feature_14[(p_word, p_pos, p_r_pos)] = 1
                    else:
                        self.feature_14[(p_word, p_pos, p_r_pos)] += 1

                    if (p_word, p_pos, p_l_pos) not in self.feature_15:
                        self.feature_15[(p_word, p_pos, p_l_pos)] = 1
                    else:
                        self.feature_15[(p_word, p_pos, p_l_pos)] += 1

                    if (p_pos, p_r_pos, p_l_pos) not in self.feature_16:
                        self.feature_16[(p_pos, p_r_pos, p_l_pos)] = 1
                    else:
                        self.feature_16[(p_pos, p_r_pos, p_l_pos)] += 1

                    if (c_word, c_pos, c_r_pos) not in self.feature_17:
                        self.feature_17[(c_word, c_pos, c_r_pos)] = 1
                    else:
                        self.feature_17[(c_word, c_pos, c_r_pos)] += 1

                    if (c_pos, c_r_pos, c_l_pos) not in self.feature_19:
                        self.feature_19[(c_pos, c_r_pos, c_l_pos)] = 1
                    else:
                        self.feature_19[(c_pos, c_r_pos, c_l_pos)] += 1

                    if (c_pos, p_pos, p_r_pos) not in self.feature_20:
                        self.feature_20[(c_pos, p_pos, p_r_pos)] = 1
                    else:
                        self.feature_20[(c_pos, p_pos, p_r_pos)] += 1

                    if (c_pos, p_pos, p_l_pos) not in self.feature_21:
                        self.feature_21[(c_pos, p_pos, p_l_pos)] = 1
                    else:
                        self.feature_21[(c_pos, p_pos, p_l_pos)] += 1

                    if (c_pos, p_pos, c_l_pos) not in self.feature_22:
                        self.feature_22[(c_pos, p_pos, c_l_pos)] = 1
                    else:
                        self.feature_22[(c_pos, p_pos, c_l_pos)] += 1

                    if (c_pos, p_pos, c_r_pos) not in self.feature_23:
                        self.feature_23[(c_pos, p_pos, c_r_pos)] = 1
                    else:
                        self.feature_23[(c_pos, p_pos, c_r_pos)] += 1

                    if (p_pos, c_r_pos, c_l_pos) not in self.feature_24:
                        self.feature_24[(p_pos, c_r_pos, c_l_pos)] = 1
                    else:
                        self.feature_24[(p_pos, c_r_pos, c_l_pos)] += 1

                    if (c_pos, p_r_pos, p_l_pos) not in self.feature_25:
                        self.feature_25[(c_pos, p_r_pos, p_l_pos)] = 1
                    else:
                        self.feature_25[(c_pos, p_r_pos, p_l_pos)] += 1

                    if (c_pos, p_pos, c_r_pos, c_l_pos) not in self.feature_26:
                        self.feature_26[(c_pos, p_pos, c_r_pos, c_l_pos)] = 1
                    else:
                        self.feature_26[(c_pos, p_pos, c_r_pos, c_l_pos)] += 1

                    if (c_pos, p_pos, p_r_pos, p_l_pos) not in self.feature_27:
                        self.feature_27[(c_pos, p_pos, p_r_pos, p_l_pos)] = 1
                    else:
                        self.feature_27[(c_pos, p_pos, p_r_pos, p_l_pos)] += 1

                    if (c_pos, p_pos, c_r_pos, p_r_pos) not in self.feature_28:
                        self.feature_28[(c_pos, p_pos, c_r_pos, p_r_pos)] = 1
                    else:
                        self.feature_28[(c_pos, p_pos, c_r_pos, p_r_pos)] += 1

                    if (c_pos, p_pos, c_l_pos, p_l_pos) not in self.feature_29:
                        self.feature_29[(c_pos, p_pos, c_l_pos, p_l_pos)] = 1
                    else:
                        self.feature_29[(c_pos, p_pos, c_l_pos, p_l_pos)] += 1

                    if (c_pos, p_pos, c_l_pos, p_r_pos) not in self.feature_30:
                        self.feature_30[(c_pos, p_pos, c_l_pos, p_r_pos)] = 1
                    else:
                        self.feature_30[(c_pos, p_pos, c_l_pos, p_r_pos)] += 1

                    if (c_pos, p_pos, c_r_pos, p_l_pos) not in self.feature_31:
                        self.feature_31[(c_pos, p_pos, c_r_pos, p_l_pos)] = 1
                    else:
                        self.feature_31[(c_pos, p_pos, c_r_pos, p_l_pos)] += 1

                    if (c_pos, p_pos, distance) not in self.feature_32:
                        self.feature_32[(c_pos, p_pos, distance)] = 1
                    else:
                        self.feature_32[(c_pos, p_pos, distance)] += 1

                    if (c_pos, distance) not in self.feature_33:
                        self.feature_33[(c_pos, distance)] = 1
                    else:
                        self.feature_33[(c_pos, distance)] += 1

                    if (p_pos, distance) not in self.feature_34:
                        self.feature_34[(p_pos, distance)] = 1
                    else:
                        self.feature_34[(p_pos, distance)] += 1

                    if (c_pos, p_pos, distance_sign) not in self.feature_35:
                        self.feature_35[(c_pos, p_pos, distance_sign)] = 1
                    else:
                        self.feature_35[(c_pos, p_pos, distance_sign)] += 1

                    if (c_pos, c_l_pos, c_2l_pos, p_pos) not in self.feature_36:
                        self.feature_36[(c_pos, c_l_pos, c_2l_pos, p_pos)] = 1
                    else:
                        self.feature_36[(c_pos, c_l_pos, c_2l_pos, p_pos)] += 1

                    if (c_pos, c_r_pos, c_2r_pos, p_pos) not in self.feature_37:
                        self.feature_37[(c_pos, c_r_pos, c_2r_pos, p_pos)] = 1
                    else:
                        self.feature_37[(c_pos, c_r_pos, c_2r_pos, p_pos)] += 1

                    if (p_pos, p_l_pos, p_2l_pos, c_pos) not in self.feature_38:
                        self.feature_38[(p_pos, p_l_pos, p_2l_pos, c_pos)] = 1
                    else:
                        self.feature_38[(p_pos, p_l_pos, p_2l_pos, c_pos)] += 1

                    if (p_pos, p_r_pos, p_2r_pos, c_pos) not in self.feature_39:
                        self.feature_39[(p_pos, p_r_pos, p_2r_pos, c_pos)] = 1
                    else:
                        self.feature_39[(p_pos, p_r_pos, p_2r_pos, c_pos)] += 1

                    if (c_pos, c_l_pos, c_2l_pos, c_r_pos, c_2r_pos, p_pos) not in self.feature_40:
                        self.feature_40[(c_pos, c_l_pos, c_2l_pos, c_r_pos, c_2r_pos, p_pos)] = 1
                    else:
                        self.feature_40[(c_pos, c_l_pos, c_2l_pos, c_r_pos, c_2r_pos, p_pos)] += 1

                    if between not in self.feature_41:
                        self.feature_41[between] = 1
                    else:
                        self.feature_41[between] += 1

                    if str(distance_sign) not in self.feature_42:
                        self.feature_42[str(distance_sign)] = 1
                    else:
                        self.feature_42[str(distance_sign)] += 1

        "creating one to one mapping from the features to numbers"
        if self.mode == 'basic':
            i = 0
            for key, value in self.feature_1.items():
                if value > 0:
                    self.feature_1_mapping[key] = i
                    i += 1
            print('f_1 size :' + str(len(self.feature_1_mapping)))

            for key, value in self.feature_2.items():
                if value > 0:
                    self.feature_2_mapping[key] = i
                    i += 1
            print('f_2 size :' + str(len(self.feature_2_mapping)))

            for key, value in self.feature_3.items():
                if value > 0:
                    self.feature_3_mapping[key] = i
                    i += 1
            print('f_3 size :' + str(len(self.feature_3_mapping)))

            for key, value in self.feature_4.items():
                if value > 0:
                    self.feature_4_mapping[key] = i
                    i += 1
            print('f_4 size :' + str(len(self.feature_4_mapping)))

            for key, value in self.feature_5.items():
                if value > 0:
                    self.feature_5_mapping[key] = i
                    i += 1
            print('f_5 size :' + str(len(self.feature_5_mapping)))

            for key, value in self.feature_6.items():
                if value > 0:
                    self.feature_6_mapping[key] = i
                    i += 1
            print('f_6 size :' + str(len(self.feature_6_mapping)))

            for key, value in self.feature_8.items():
                if value > 0:
                    self.feature_8_mapping[key] = i
                    i += 1
            print('f_8 size :' + str(len(self.feature_8_mapping)))

            for key, value in self.feature_10.items():
                if value > 0:
                    self.feature_10_mapping[key] = i
                    i += 1
            print('f_10 size :' + str(len(self.feature_10_mapping)))

            for key, value in self.feature_13.items():
                if value > 0:
                    self.feature_13_mapping[key] = i
                    i += 1
            print('f_13 size :' + str(len(self.feature_13_mapping)))

        elif self.mode == 'advanced':
            i = 0
            """for key, value in self.feature_7.items():
                if value > 1:
                    self.feature_7_mapping[key] = i
                    i += 1
            print('f_7 size :' + str(len(self.feature_7_mapping)))"""

            for key, value in self.feature_8.items():
                if value > 0:
                    self.feature_8_mapping[key] = i
                    i += 1
            print('f_8 size :' + str(len(self.feature_8_mapping)))

            """for key, value in self.feature_9.items():
                if value > 1:
                    self.feature_9_mapping[key] = i
                    i += 1
            print('f_9 size :' + str(len(self.feature_9_mapping)))"""

            for key, value in self.feature_10.items():
                if value > 0:
                    self.feature_10_mapping[key] = i
                    i += 1
            print('f_10 size :' + str(len(self.feature_10_mapping)))

            """for key, value in self.feature_11.items():
                if value > 1:
                    self.feature_11_mapping[key] = i
                    i += 1
            print('f_11 size :' + str(len(self.feature_11_mapping)))

            for key, value in self.feature_12.items():
                if value > 1:
                    self.feature_12_mapping[key] = i
                    i += 1
            print('f_12 size :' + str(len(self.feature_12_mapping)))"""

            for key, value in self.feature_13.items():
                if value > 0:
                    self.feature_13_mapping[key] = i
                    i += 1
            print('f_13 size :' + str(len(self.feature_13_mapping)))

            for key, value in self.feature_14.items():
                if value > 0:
                    self.feature_14_mapping[key] = i
                    i += 1
            print('f_14 size :' + str(len(self.feature_14_mapping)))

            for key, value in self.feature_15.items():
                if value > 0:
                    self.feature_15_mapping[key] = i
                    i += 1
            print('f_15 size :' + str(len(self.feature_15_mapping)))

            for key, value in self.feature_16.items():
                if value > 0:
                    self.feature_16_mapping[key] = i
                    i += 1
            print('f_16 size :' + str(len(self.feature_16_mapping)))

            for key, value in self.feature_17.items():
                if value > 0:
                    self.feature_17_mapping[key] = i
                    i += 1
            print('f_17 size :' + str(len(self.feature_17_mapping)))

            for key, value in self.feature_19.items():
                if value > 0:
                    self.feature_19_mapping[key] = i
                    i += 1
            print('f_19 size :' + str(len(self.feature_19_mapping)))

            for key, value in self.feature_20.items():
                if value > 0:
                    self.feature_20_mapping[key] = i
                    i += 1
            print('f_20 size :' + str(len(self.feature_20_mapping)))

            for key, value in self.feature_22.items():
                if value > 0:
                    self.feature_22_mapping[key] = i
                    i += 1
            print('f_22 size :' + str(len(self.feature_22_mapping)))

            for key, value in self.feature_23.items():
                if value > 0:
                    self.feature_23_mapping[key] = i
                    i += 1
            print('f_23 size :' + str(len(self.feature_23_mapping)))

            for key, value in self.feature_24.items():
                if value > 0:
                    self.feature_24_mapping[key] = i
                    i += 1
            print('f_24 size :' + str(len(self.feature_24_mapping)))

            for key, value in self.feature_25.items():
                if value > 0:
                    self.feature_25_mapping[key] = i
                    i += 1
            print('f_25 size :' + str(len(self.feature_25_mapping)))

            for key, value in self.feature_26.items():
                if value > 0:
                    self.feature_26_mapping[key] = i
                    i += 1
            print('f_26 size :' + str(len(self.feature_26_mapping)))

            for key, value in self.feature_27.items():
                if value > 0:
                    self.feature_27_mapping[key] = i
                    i += 1
            print('f_27 size :' + str(len(self.feature_27_mapping)))

            for key, value in self.feature_28.items():
                if value > 0:
                    self.feature_28_mapping[key] = i
                    i += 1
            print('f_28 size :' + str(len(self.feature_28_mapping)))

            for key, value in self.feature_29.items():
                if value > 0:
                    self.feature_29_mapping[key] = i
                    i += 1
            print('f_29 size :' + str(len(self.feature_29_mapping)))

            for key, value in self.feature_30.items():
                if value > 0:
                    self.feature_30_mapping[key] = i
                    i += 1
            print('f_30 size :' + str(len(self.feature_30_mapping)))

            for key, value in self.feature_31.items():
                if value > 0:
                    self.feature_31_mapping[key] = i
                    i += 1
            print('f_31 size :' + str(len(self.feature_31_mapping)))

            for key, value in self.feature_32.items():
                if value > 0:
                    self.feature_32_mapping[key] = i
                    i += 1
            print('f_32 size :' + str(len(self.feature_32_mapping)))

            for key, value in self.feature_33.items():
                if value > 0:
                    self.feature_33_mapping[key] = i
                    i += 1
            print('f_33 size :' + str(len(self.feature_33_mapping)))

            for key, value in self.feature_34.items():
                if value > 0:
                    self.feature_34_mapping[key] = i
                    i += 1
            print('f_34 size :' + str(len(self.feature_34_mapping)))

            for key, value in self.feature_35.items():
                if value > 0:
                    self.feature_35_mapping[key] = i
                    i += 1
            print('f_35 size :' + str(len(self.feature_35_mapping)))

            for key, value in self.feature_36.items():
                if value > 0:
                    self.feature_36_mapping[key] = i
                    i += 1
            print('f_36 size :' + str(len(self.feature_36_mapping)))

            for key, value in self.feature_37.items():
                if value > 0:
                    self.feature_37_mapping[key] = i
                    i += 1
            print('f_37 size :' + str(len(self.feature_37_mapping)))

            for key, value in self.feature_38.items():
                if value > 0:
                    self.feature_38_mapping[key] = i
                    i += 1
            print('f_38 size :' + str(len(self.feature_38_mapping)))

            for key, value in self.feature_39.items():
                if value > 0:
                    self.feature_39_mapping[key] = i
                    i += 1
            print('f_39 size :' + str(len(self.feature_39_mapping)))

            for key, value in self.feature_40.items():
                if value > 0:
                    self.feature_40_mapping[key] = i
                    i += 1
            print('f_40 size :' + str(len(self.feature_40_mapping)))

            for key, value in self.feature_41.items():
                if value > 0:
                    self.feature_41_mapping[key] = i
                    i += 1
            print('f_41 size :' + str(len(self.feature_41_mapping)))

            for key, value in self.feature_42.items():
                if value > 0:
                    self.feature_42_mapping[key] = i
                    i += 1
            print('f_42 size :' + str(len(self.feature_42_mapping)))

        self.feature_vec_len = i+1

    def create_feature_vector_for_edge(self, sentence, head_index, modifier_index):
        "returns a lists of features index vector for each edge (head, modifier)"
        c_word = sentence[modifier_index][1].lower()
        c_pos = sentence[modifier_index][2]
        p_word = sentence[head_index][1].lower()
        p_pos = sentence[head_index][2]

        distance_sign = np.sign(modifier_index - head_index)
        distance = abs(modifier_index - head_index)

        between = ''
        if distance_sign == 1:
            for j in range(1, distance - 1):
                between += "_" + sentence[head_index + j][2]
        else:
            for j in range(1, distance - 1):
                between += "_" + sentence[modifier_index + j][2]

        c_r_pos = 'STOP'
        c_2r_pos = 'STOP'
        c_l_pos = '**'
        c_2l_pos = '**'
        p_r_pos = 'STOP'
        p_2r_pos = 'STOP'
        p_l_pos = '**'
        p_2l_pos = '**'

        # left word POS of modifier in the sentence
        if modifier_index > 1:
            c_l_pos = sentence[modifier_index - 1][2]
            if modifier_index > 2:
                c_2l_pos = sentence[modifier_index - 2][2]
        # right word POS of modifier in the sentence
        if modifier_index < len(sentence) - 1:
            c_r_pos = sentence[modifier_index + 1][2]
            if modifier_index < len(sentence) - 2:
                c_2r_pos = sentence[modifier_index + 2][2]

        # left word POS of head in the sentence
        if head_index > 1:
            p_l_pos = sentence[head_index - 1][2]
            if head_index > 2:
                p_2l_pos = sentence[head_index - 2][2]
        # right word POS of head in the sentence
        if head_index < len(sentence) - 1:
            p_r_pos = sentence[head_index + 1][2]
            if head_index < len(sentence) - 2:
                p_2r_pos = sentence[head_index + 2][2]

        vector_columns_index = []

        "if a combination exist in features dict we append its index"
        if self.mode == 'basic':
            if (p_word, p_pos) in self.feature_1_mapping:
                vector_columns_index.append(self.feature_1_mapping[(p_word, p_pos)])

            if p_word in self.feature_2_mapping:
                vector_columns_index.append(self.feature_2_mapping[p_word])

            if p_pos in self.feature_3_mapping:
                vector_columns_index.append(self.feature_3_mapping[p_pos])

            if (c_word, c_pos) in self.feature_4_mapping:
                vector_columns_index.append(self.feature_4_mapping[(c_word, c_pos)])

            if c_word in self.feature_5_mapping:
                vector_columns_index.append(self.feature_5_mapping[c_word])

            if c_pos in self.feature_6_mapping:
                vector_columns_index.append(self.feature_6_mapping[c_pos])

            if (p_pos, c_word, c_pos) in self.feature_8_mapping:
                vector_columns_index.append(self.feature_8_mapping[(p_pos, c_word, c_pos)])

            if (p_word, p_pos, c_pos) in self.feature_10_mapping:
                vector_columns_index.append(self.feature_10_mapping[(p_word, p_pos, c_pos)])

            if (p_pos, c_pos) in self.feature_13_mapping:
                vector_columns_index.append(self.feature_13_mapping[(p_pos, c_pos)])

        if self.mode == 'advanced':
            """if (p_word, p_pos, c_word, c_pos) in self.feature_7_mapping:
                vector_columns_index.append(self.feature_7_mapping[(p_word, p_pos, c_word, c_pos)])"""

            if (p_pos, c_word, c_pos) in self.feature_8_mapping:
                vector_columns_index.append(self.feature_8_mapping[(p_pos, c_word, c_pos)])


            """if (p_word, c_word, c_pos) in self.feature_9_mapping:
                vector_columns_index.append(self.feature_9_mapping[(p_word, c_word, c_pos)])"""

            if (p_word, p_pos, c_pos) in self.feature_10_mapping:
                vector_columns_index.append(self.feature_10_mapping[(p_word, p_pos, c_pos)])


            """if (p_word, p_pos, c_word) in self.feature_11_mapping:
                vector_columns_index.append(self.feature_11_mapping[(p_word, p_pos, c_word)])


            if (p_word, c_word) in self.feature_12_mapping:
                vector_columns_index.append(self.feature_12_mapping[(p_word, c_word)])"""

            if (p_pos, c_pos) in self.feature_13_mapping:
                vector_columns_index.append(self.feature_13_mapping[(p_pos, c_pos)])

            if (p_word, p_pos, p_r_pos) in self.feature_14_mapping:
                vector_columns_index.append(self.feature_14_mapping[(p_word, p_pos, p_r_pos)])

            if (p_word, p_pos, p_l_pos) in self.feature_15_mapping:
                vector_columns_index.append(self.feature_15_mapping[(p_word, p_pos, p_l_pos)])

            if (p_pos, p_r_pos, p_l_pos) in self.feature_16_mapping:
                vector_columns_index.append(self.feature_16_mapping[(p_pos, p_r_pos, p_l_pos)])

            if (c_word, c_pos, c_r_pos) in self.feature_17_mapping:
                vector_columns_index.append(self.feature_17_mapping[(c_word, c_pos, c_r_pos)])

            if (c_pos, c_r_pos, c_l_pos) in self.feature_19_mapping:
                vector_columns_index.append(self.feature_19_mapping[(c_pos, c_r_pos, c_l_pos)])

            if (c_pos, p_pos, p_r_pos) in self.feature_20_mapping:
                vector_columns_index.append(self.feature_20_mapping[(c_pos, p_pos, p_r_pos)])

            if (c_pos, p_pos, p_l_pos) in self.feature_21_mapping:
                vector_columns_index.append(self.feature_21_mapping[(c_pos, p_pos, p_l_pos)])

            if (c_pos, p_pos, c_l_pos) in self.feature_22_mapping:
                vector_columns_index.append(self.feature_22_mapping[(c_pos, p_pos, c_l_pos)])

            if (c_pos, p_pos, c_r_pos) in self.feature_23_mapping:
                vector_columns_index.append(self.feature_23_mapping[(c_pos, p_pos, c_r_pos)])

            if (p_pos, c_r_pos, c_l_pos) in self.feature_24_mapping:
                vector_columns_index.append(self.feature_24_mapping[(p_pos, c_r_pos, c_l_pos)])

            if (c_pos, p_r_pos, p_l_pos) in self.feature_25_mapping:
                vector_columns_index.append(self.feature_25_mapping[(c_pos, p_r_pos, p_l_pos)])

            if (c_pos, p_pos, c_r_pos, c_l_pos) in self.feature_26_mapping:
                vector_columns_index.append(self.feature_26_mapping[(c_pos, p_pos, c_r_pos, c_l_pos)])

            if (c_pos, p_pos, p_r_pos, p_l_pos) in self.feature_27_mapping:
                vector_columns_index.append(self.feature_27_mapping[(c_pos, p_pos, p_r_pos, p_l_pos)])

            if (c_pos, p_pos, c_r_pos, p_r_pos) in self.feature_28_mapping:
                vector_columns_index.append(self.feature_28_mapping[(c_pos, p_pos, c_r_pos, p_r_pos)])

            if (c_pos, p_pos, c_l_pos, p_l_pos) in self.feature_29_mapping:
                vector_columns_index.append(self.feature_29_mapping[(c_pos, p_pos, c_l_pos, p_l_pos)])

            if (c_pos, p_pos, c_l_pos, p_r_pos) in self.feature_30_mapping:
                vector_columns_index.append(self.feature_30_mapping[(c_pos, p_pos, c_l_pos, p_r_pos)])

            if (c_pos, p_pos, c_r_pos, p_l_pos) in self.feature_31_mapping:
                vector_columns_index.append(self.feature_31_mapping[(c_pos, p_pos, c_r_pos, p_l_pos)])

            if (c_pos, p_pos, distance) in self.feature_32_mapping:
                vector_columns_index.append(self.feature_32_mapping[(c_pos, p_pos, distance)])

            if (c_pos, distance) in self.feature_33_mapping:
                vector_columns_index.append(self.feature_33_mapping[(c_pos, distance)])

            if (p_pos, distance) in self.feature_34_mapping:
                vector_columns_index.append(self.feature_34_mapping[(p_pos, distance)])

            if (c_pos, p_pos, distance_sign) in self.feature_35_mapping:
                vector_columns_index.append(self.feature_35_mapping[(c_pos, p_pos, distance_sign)])

            if (c_pos, c_l_pos, c_2l_pos, p_pos) in self.feature_36_mapping:
                vector_columns_index.append(self.feature_36_mapping[(c_pos, c_l_pos, c_2l_pos, p_pos)])

            if (c_pos, c_r_pos, c_2r_pos, p_pos) in self.feature_37_mapping:
                vector_columns_index.append(self.feature_37_mapping[(c_pos, c_r_pos, c_2r_pos, p_pos)])

            if (p_pos, p_l_pos, p_2l_pos, c_pos) in self.feature_38_mapping:
                vector_columns_index.append(self.feature_38_mapping[(p_pos, p_l_pos, p_2l_pos, c_pos)])

            if (p_pos, p_r_pos, p_2r_pos, c_pos) in self.feature_39_mapping:
                vector_columns_index.append(self.feature_39_mapping[(p_pos, p_r_pos, p_2r_pos, c_pos)])

            if (c_pos, c_l_pos, c_2l_pos, c_r_pos, c_2r_pos, p_pos) in self.feature_40_mapping:
                vector_columns_index.append(self.feature_40_mapping[(c_pos, c_l_pos, c_2l_pos, c_r_pos, c_2r_pos, p_pos)])

            if between in self.feature_41_mapping:
                vector_columns_index.append(self.feature_41_mapping[between])

            if str(distance_sign) in self.feature_42_mapping:
                vector_columns_index.append(self.feature_42_mapping[str(distance_sign)])

        return vector_columns_index

    def build_train_matrix(self):
        sentence_num = 0
        for sentence in self.all_sentences_train:
            for word in sentence:
                if word[0] == 0:
                    continue
                matrix_rows_index = []
                matrix_columns_index = []
                for temp_head in sentence:
                    if temp_head[0] == word[0]:
                        continue
                    columns_index = []

                    current_features = self.create_feature_vector_for_edge(sentence,temp_head[0], word[0])
                    for feature in current_features:
                        matrix_rows_index.append(temp_head[0])
                        columns_index.append(feature)
                    matrix_columns_index += columns_index

                    if temp_head[0] == word[3]:
                        rows_index = np.zeros(shape=len(columns_index), dtype=np.int32)
                        cols_index = np.asarray(a=columns_index, dtype=np.int32)
                        data = np.ones(len(columns_index), dtype=np.int8)
                        train_word_vector = csr_matrix((data, (rows_index, cols_index)), shape=(1, self.feature_vec_len))
                        self.train_feature_matrix[(sentence_num, word[0])] = train_word_vector

                rows_index = np.asarray(a=matrix_rows_index, dtype=np.int32)
                cols_index = np.asarray(a=matrix_columns_index, dtype=np.int32)
                data = np.ones(len(matrix_rows_index), dtype=np.int8)
                matrix = csr_matrix((data, (rows_index, cols_index)), shape=(len(sentence), self.feature_vec_len))

                self.features_matrix[(sentence_num, word[0])] = matrix
            sentence_num += 1

    def perceptron(self, iterations):
        w_temp = np.zeros(shape=self.feature_vec_len, dtype=np.float16)
        w = np.zeros(shape=self.feature_vec_len, dtype=np.float16)
        for i in range(iterations):
            sentence_num = 0
            for sentence in self.all_sentences_train:
                for word in sentence:
                    if word[0] != 0:
                        head = word[3]
                        argmax_head = np.argmax(self.features_matrix[sentence_num, word[0]].dot(w_temp))
                        if head != argmax_head:
                            w_temp += (self.train_feature_matrix[sentence_num, word[0]].toarray()[0] -
                                self.features_matrix[sentence_num, word[0]][argmax_head, :].toarray()[0])
                            w += ((iterations - i) * (self.train_feature_matrix[sentence_num, word[0]].toarray()[0] -
                                                   self.features_matrix[sentence_num, word[0]][argmax_head, :].toarray()[0]))

                sentence_num += 1
        w = w / (iterations * len(self.all_sentences_train))
        self.w_vector = w

    def training_procedure(self):
        start_1 = time.time()
        self.build_data()
        self.build_features_dicts()
        self.build_train_matrix()
        end_1 = time.time()
        print('build data time: ' + str((end_1 - start_1)/60))
        start_2 = time.time()
        if self.mode == 'basic':
            self.perceptron(50)
        else:
            self.perceptron(80)
        end_2 = time.time()
        print('training time: ' + str((end_2 - start_2)/60))

    def calc_weights_for_edges(self, sentence):
        weights = defaultdict(tuple)
        for modifier in sentence:
            # If root
            if modifier[0] == 0:
                continue
            word_matrix_rows_index = []
            word_matrix_columns_index = []
            # Every possible head
            for possible_head in sentence:
                # node cannot be head of itself
                if possible_head[0] == modifier[0]:
                    continue
                columns_index = []
                current_features = self.create_feature_vector_for_edge(sentence, possible_head[0], modifier[0])
                for feature in current_features:
                    word_matrix_rows_index.append(possible_head[0])
                    columns_index.append(feature)
                word_matrix_columns_index += columns_index
            rows_index = np.asarray(a=word_matrix_rows_index, dtype=np.int32)
            cols_index = np.asarray(a=word_matrix_columns_index, dtype=np.int32)
            data = np.ones(len(word_matrix_rows_index), dtype=np.int8)
            word_matrix = csr_matrix((data, (rows_index, cols_index)), shape=(len(sentence), self.feature_vec_len))

            weights[modifier[0]] = word_matrix.dot(self.w_vector)
        return weights

    def build_full_graph(self, sentence):
        full_graph = defaultdict(tuple)
        for target in sentence:
            # The root can't be modifier only head
            if target[0] == 0:
                continue
            for source in sentence:
                # Node can't be a head of itself
                if source[0] == target[0]:
                    continue
                if source[0] not in full_graph:
                    full_graph[source[0]] = [target[0]]
                else:
                    full_graph[source[0]].append(target[0])
        return full_graph

    def make_list_of_pred_and_true(self):
        gold_head_list = []
        for sentence in self.all_sentences_test:
            dict = {}
            for node in sentence:
                if node[0] == 0:
                    continue
                dict[node[0]] = node[3]
            for item in sorted(dict.items()):
                gold_head_list.append(item[1])

        pred_head_list = []
        for succ in self.mst_dict:
            dict = {}
            for key in self.mst_dict[succ]:
                for node in self.mst_dict[succ][key]:
                    dict[node] = key
            for item in sorted(dict.items()):
                pred_head_list.append(item[1])

        return gold_head_list, pred_head_list

    def calculate_accuracy(self, gold_list, prediction_list):
        count_correct = 0
        total = len(gold_list)
        for i in range(0, total):
            if gold_list[i] == prediction_list[i]:
                count_correct += 1
        return float(count_correct) / total

    def write_comp_file(self, head_prediction_list):
        if self.mode == 'basic':
            file_name = 'comp_m1.wtag'
        else:
            file_name = 'comp_m2.wtag'
        with open(file_name, "a") as new_file:
            sentence_len = 0
            for sentence in self.all_sentences_comp:
                for modifier in sentence:
                    if modifier[0] == 0:
                        continue
                    text_to_write = str(modifier[0]) + '\t' + modifier[1] + '\t' + '_' + '\t' + modifier[2] + '\t' + '_' + '\t' + \
                                    '_' + '\t' + str(head_prediction_list[sentence_len + modifier[0] - 1]) + '\t' + '_' + '\t' + '_' + '\t' + '_'
                    new_file.write(text_to_write)
                    new_file.write('\n')
                new_file.write('\n')
                sentence_len += len(sentence) - 1

    def make_inference_test(self):
        start = time.time()
        num_tree = 1
        for sentence in self.all_sentences_test:
            weights = self.calc_weights_for_edges(sentence)
            digraph = Digraph(self.build_full_graph(sentence), lambda source, target: weights[target][source])
            mst = digraph.mst()
            self.mst_dict[num_tree] = mst.successors
            num_tree += 1
        end = time.time()
        print('inference test time: ' + str((end - start)/60))
        gold_list, prediction_list = self.make_list_of_pred_and_true()
        accuracy = self.calculate_accuracy(gold_list, prediction_list)
        print("Accuracy: " + str(accuracy * 100))

    def make_inference_comp(self):
        start = time.time()
        num_tree = 1
        for sentence in self.all_sentences_comp:
            weights = self.calc_weights_for_edges(sentence)
            digraph = Digraph(self.build_full_graph(sentence), lambda source, target: weights[target][source])
            mst = digraph.mst()
            self.mst_dict_comp[num_tree] = mst.successors
            num_tree += 1
        end = time.time()
        print('inference comp time: ' + str((end - start)/60))

        pred_head_list = []
        for succ in self.mst_dict_comp:
            dict = {}
            for key in self.mst_dict_comp[succ]:
                for node in self.mst_dict_comp[succ][key]:
                    dict[node] = key
            for item in sorted(dict.items()):
                pred_head_list.append(item[1])
        self.write_comp_file(pred_head_list)


DP_basic = Depndency_Parser('basic')
DP_basic.training_procedure()
#DP_basic.make_inference_test()
DP_basic.make_inference_comp()

DP_advanced = Depndency_Parser('advanced')
DP_advanced.training_procedure()
#DP_advanced.make_inference_test()
DP_advanced.make_inference_comp()


