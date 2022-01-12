from datetime import datetime
import Model_1
import Model_2


def run_model_1():
    print('runs model 1')
    time_1 = datetime.now()
    train = open("train.wtag", "r")
    test = open("test.wtag", "r")
    model = Model_1.MEMM(train, test)
    model.initialize_train()
    time_2 = datetime.now()
    print('train time: ' + str(time_2 - time_1))

    time_3 = datetime.now()
    model.initialize_test()
    time_4 = datetime.now()
    print('inference time for test: ' + str(time_4 - time_3))
    time_5 = datetime.now()
    comp = open("comp.words", "r")
    model.inference_for_comp(comp)
    time_6 = datetime.now()
    print('inference time for comp: ' + str(time_6 - time_5))


def run_model_2():
    data = open("train2.wtag", "r")
    comp = open("comp2.words", "r")
    time_1 = datetime.now()
    model = Model_2.MEMM(data)
    model.initialize_train()
    time_2 = datetime.now()
    print('train time: ' + str(time_2 - time_1))
    time_5 = datetime.now()
    model.inference_for_comp(comp)
    time_6 = datetime.now()
    print('inference time for comp: ' + str(time_6 - time_5))


run_model_1()
run_model_2()
