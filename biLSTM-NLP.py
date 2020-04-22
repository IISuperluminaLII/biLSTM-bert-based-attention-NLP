import datetime
import os
import random
import string
import re
import numpy as np
from keras.utils import Sequence
from nltk import tokenize as tk
import tensorflow as tf
from tensorflow import keras
os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
tf.logging.set_verbosity(tf.logging.ERROR)
from tensorflow.keras.regularizers import l2, l1_l2
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Embedding, CuDNNLSTM, Dense, Input, Bidirectional, concatenate, Dropout, LayerNormalization, BatchNormalization, LeakyReLU, add, subtract
from tensorflow.keras.activations import softmax, relu
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop, Adam, Nadam
from tensorflow.keras.losses import BinaryCrossentropy, CategoricalCrossentropy, SparseCategoricalCrossentropy

#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'



def load_doc():
    openedFile = open("speeches.txt", 'r')
    of = open("trmp.txt", 'r')
    oof = open("fuck.txt", 'r')
    allLines = openedFile.readlines()
    allL = of.readlines()
    fupa = oof.readlines()
    giantTrump = ""
    printable = set(string.printable)
    counter = 0
    for line in allLines:
        line = re.sub(r"http\S+", "", line)
        filter(lambda x: x in printable, line)
        giantTrump += line[5:]
    for le in allL:
        le = re.sub(r"http\S+", "", le)
        filter(lambda x: x in printable, le)
        giantTrump += le
    for keloo in fupa:
        filter(lambda x: x in printable, keloo)
        giantTrump += keloo

    return giantTrump


def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        #print("Checking Index "+str(index)+"against input index"+str(integer))
        if index == integer:
            return word
    return "liberal"

def filtersdffew(arr1, filterarr):
    arr1_copy = arr1.copy()
    for index, value in zip(range(len(arr1_copy)), arr1_copy):
        if value in filterarr:
            arr1_copy[index] = pad_idx
    return arr1_copy

def generate_desc_v2(model, tokenizer, epoch):
    # seed the generation process
    outfiledata = ""
    in_text = '<SOS>'
    # iterate over the whole length of the sequence
    counter = 0
    for i in range(0, 99):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence_l1 = pad_sequences([sequence[:i]], maxlen=50, padding='post', value=pad_idx)
        in_rseq = pad_sequences([filtersdffew(sequence, [sequence[i]])], maxlen=50, padding='post', value=pad_idx)
        yhat = model.predict([sequence_l1, in_rseq], verbose=0)
        yhat = np.argmax(yhat)
        word = word_for_id(yhat, tokenizer)
        if word is None:
            break
        in_text += ' ' + word
        #print(' ' + word, end='')
        outfiledata += ' ' + word
        if counter % 15 == 0 and counter != 0:
            #print("\n")
            outfiledata += "\n"
        counter += 1
    testes = open("seq2seq/"+"output_epoch_"+str(epoch)+".txt", "w+")
    testes.write(outfiledata)
    print(outfiledata)


def generate_desc(model, tokenizer, epoch):
    output = ""
    select = holder[random.randrange(2, len(holder) - 2)]
    sequence = tokenizer.texts_to_sequences([select])[0]
    output += "\n============================rand word bag generation===================================\n"
    output += "Seed Word:-" + str(select)
    output += "\n________________________________________________________________________________________\n"
    counter = 0
    for itrs in range(100-len(sequence)):
        if itrs == 0:
            print(sequence)
            for i in range(1, len(sequence)):
                random_select_1 = pad_sequences([sequence[:i]], maxlen=50, padding='post', value=pad_idx)
                random_select_2 = pad_sequences([filtersdffew(sequence, [sequence[i]])], maxlen=50, padding='post', value=pad_idx)
                yhat = model.predict([random_select_1, random_select_2], verbose=0)
                yhat = np.argmax(yhat)
                word = word_for_id(yhat, tokenizer)
                output += ' ' + word
                if counter % 15 == 0 and counter != 0:
                    output += "\n"
                counter += 1
            output += "|"
        else:
            sequence = tokenizer.texts_to_sequences([output])[0]
            random_select_1 = pad_sequences([sequence[:itrs]], maxlen=50, padding='post', value=pad_idx)
            random_select_2 = pad_sequences([filtersdffew(sequence, [sequence[itrs]])], maxlen=50, padding='post', value=pad_idx)
            yhat = model.predict([random_select_1, random_select_2], verbose=0)
            yhat = np.argmax(yhat)
            word = word_for_id(yhat, tokenizer)
            output += ' ' + word
            if counter % 15 == 0:
                output += "\n"
            counter += 1
    output += "\n________________________________________________________________________________________\n"
    output += "=========================================================================================\n"
    print(output)
    text_file = open("wordbagseq/output_epoch_"+str(epoch)+".txt", "w+")
    text_file.write(output)
    text_file.close()
    print("SINGLE GENERATION TEST (Prev sequence dependent)")
    print("_______________________________________________________________________________________\n")
    generate_desc_v2(model, tokenizer, epoch)
    print("_______________________________________________________________________________________\n")
    print("--------------------------------------------------------------------------\n")


speech_loc = load_doc()
print(len(speech_loc))
holder = tk.sent_tokenize(speech_loc)
temp_holder = list()
for tok in holder:
    position = '<SOS> ' + str(tok) + ' <EOS>'
    temp_holder.append(position)
temp_holder.append('<PAD>')
holder = temp_holder.copy()

tokenizer = Tokenizer(split=" ", oov_token=0, lower=False)
tokenizer.fit_on_texts(holder)
pad_idx = 0
for index, word in tokenizer.index_word.items():
    if word == 'PAD':
        pad_idx = index
sequences = tokenizer.texts_to_sequences(holder)#random.sample(tokenizer.texts_to_sequences(holder), 50000)
max_length = 50
print("Max sequence length: " + str(max_length))
len_vocab = len(tokenizer.word_index) + 1
print("vocab Size: " + str(len_vocab))
print("total sequence length: " + str(len(sequences)))
#z = input("\nContinue?\n")
#n = input("happy with the sizes?")


class PredictionCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        #if epoch % 50 == 0:
        generate_desc(self.model, tokenizer, epoch)


def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return "liberal"


def generate_training_setss(batch_size, train, test):
    selectedSequence = random.sample(sequences, 10000)
    if train > 0:
        selectedSequence = random.sample(sequences, train)
    elif test > 0:
        selectedSequence = random.sample(sequences, test)
    xTrainLocal, x2TrainLocal, yTrainLocal = list(), list(), list()
    cnt = 0
    for idx, dat in enumerate(selectedSequence):
        #print("parsing idx: " + str(idx))
        for i in range(1, len(dat) - 2):
            in_seq, rand_pad = dat[:], filtersdffew(dat.copy(), [dat[i], dat[i + 1]])
            out_seq = to_categorical([dat[i]], num_classes=len_vocab)[0]
            in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
            rand_pad = pad_sequences([rand_pad], maxlen=max_length)[0]
            xTrainLocal.append(in_seq)
            x2TrainLocal.append(rand_pad)
            yTrainLocal.append(out_seq)
    print("\n________Generated newnew________")
    # exit(0)668561,
    xTrainLocal = np.array(xTrainLocal)
    print("xTrain shape: " + str(xTrainLocal.shape))
    x2TrainLocal = np.array(x2TrainLocal)
    print("x2Train shape: " + str(x2TrainLocal.shape))
    yTrainLocal = np.array(yTrainLocal)
    print("yTrain shape: " + str(yTrainLocal.shape))
    print("__________________________________\n")
    return [xTrainLocal, x2TrainLocal], yTrainLocal

class TrainingSequence(keras.utils.Sequence):
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.xtrain = np.empty((None, 100))
        self.x2train = np.empty((None, 100))
        self.ytrain = np.empty((None, len_vocab))
    def __len__(self):
        return self.batch_size
    def __getitem__(self, idx):
        selectedSequence = sequences[idx * self.batch_size: (idx + 1) * self.batch_size]
        xTrainLocal, x2TrainLocal, yTrainLocal = list(), list(), list()
        cnt = 0
        for idxs, dat in enumerate(selectedSequence):
            # print("parsing idx: " + str(idx))
            for i in range(1, len(dat) - 2):
                in_seq, rand_pad = dat[:], filtersdffew(dat.copy(), [dat[i], dat[i + 1]])
                #out_seq = to_categorical([dat[i]], num_classes=len_vocab)[0]
                in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                rand_pad = pad_sequences([rand_pad], maxlen=max_length)[0]
                xTrainLocal.append(in_seq)
                x2TrainLocal.append(rand_pad)
                yTrainLocal.append(dat[i])
        print("\n________Generated newnew from printable thread________"+str(idx))
        # exit(0)668561,
        if idx == 0:
            self.xtrain = np.array(xTrainLocal)
            print("xTrain shape: " + str(self.xtrain.shape))
            self.x2train = np.array(x2TrainLocal)
            print("x2Train shape: " + str(self.x2train.shape))
            self.ytrain = np.array(yTrainLocal)
            print("yTrain shape: " + str(self.ytrain.shape))
            print("__________________________________\n")
            return [self.xtrain, self.x2train], self.ytrain
        else:
            np.append(self.xtrain, )


xTrain, x2Train, yTrain = None, None, None
generate = True
if generate:
    load_saved = False
    print(np.shape(1))
    if not load_saved:
        xTrainLocal, x2TrainLocal, yTrainLocal = list(), list(), list()
        cnt = 0
        for idx, dat in enumerate(sequences):
            if len(dat) < 30 or len(dat) > 50:
                #print("skipping: "+str(len(dat)))
                continue
            cnt+=1
            #print("parsing idx: "+str(idx))
            #print("parsing count: " + str(cnt))
            for i in range(1, len(dat) - 1):
                in_seq, rand_pad = dat[:i], filtersdffew(dat.copy(), [dat[i]])
                #out_seq = to_categorical([dat[i]], num_classes=len_vocab)[0].astype('float16')
                in_seq = pad_sequences([in_seq], maxlen=max_length, padding='post', value=pad_idx)[0]
                rand_pad = pad_sequences([rand_pad], maxlen=max_length, padding='post', value=pad_idx)[0]
                xTrainLocal.append(in_seq)
                x2TrainLocal.append(rand_pad)
                yTrainLocal.append(dat[i])
        print("\n________TRAINING SET SHAPE________")
        xTrainLocal = np.array(xTrainLocal)
        print("xTrain shape: " + str(xTrainLocal.shape))
        x2TrainLocal = np.array(x2TrainLocal)
        print("x2Train shape: " + str(x2TrainLocal.shape))
        yTrainLocal = np.array(yTrainLocal)
        print("yTrain shape: " + str(yTrainLocal.shape))
        print("saved new word bag encoding")
        print("__________________________________\n")

        np.save("raw_text_dat_vbincross234", xTrainLocal)
        np.save("raw_sequence_data_vbincross234", x2TrainLocal)
        np.save("all_words_vbincross234", yTrainLocal)
    else:
        # fpath = "tokenSeperatorYJoin/ramsaverYTrain"
        # npyfilespath = "tokenSeperatorY"
        # os.chdir(npyfilespath)
        # npfiles = glob.glob("*.npy")
        # npfiles.sort()
        # all_arrays = []
        # # for i, npfile in enumerate(npfiles):
        # #     print("processing file: "+str(i))
        # #     all_arrays.append(np.load(npfile).astype('float16'))
        # np.save(fpath, np.concatenate([np.load(npfiles[4]).astype('float16'),
        #                                np.load(npfiles[1]).astype('float16'),
        #                                np.load(npfiles[2]).astype('float16'),
        #                                np.load(npfiles[3]).astype('float16'),
        #                                np.load(npfiles[0]).astype('float16')]))
        # all_arrays.clear()
        # j = input("can the ram handle or what?")
        # fpath = "/tokenSeperatortXJoin/ramsaverXTrain"
        # npyfilespath = "tokenSeperatorX"
        # os.chdir(npyfilespath)
        # npfiles = glob.glob("*.npy")
        # npfiles.sort()
        # all_arrays = []
        # # for i, npfile in enumerate(npfiles):
        # #     all_arrays.append(np.load(npfile).astype('float16'))
        # # np.save(fpath, np.concatenate(all_arrays))
        # np.save(fpath, np.concatenate([np.load(npfiles[4]).astype('float16'),
        #                                np.load(npfiles[1]).astype('float16'),
        #                                np.load(npfiles[2]).astype('float16'),
        #                                np.load(npfiles[3]).astype('float16'),
        #                                np.load(npfiles[0]).astype('float16')]))
        # all_arrays.clear()
        # j = input("bruh how the ram doing, idk how this fucking shit working. probably has to to with float16")
        # fpath = "tokenSeperatorX2Join/ramsaverX2Train"
        # npyfilespath = "/tokenSeperatorX2"
        # os.chdir(npyfilespath)
        # npfiles = glob.glob("*.npy")
        # npfiles.sort()
        # all_arrays = []
        # # for i, npfile in enumerate(npfiles):
        # #     all_arrays.append(np.load(npfile).astype('float16'))
        # # np.save(fpath, np.concatenate(all_arrays))
        # np.save(fpath, np.concatenate([np.load(npfiles[4]).astype('float16'),
        #                                np.load(npfiles[1]).astype('float16'),
        #                                np.load(npfiles[2]).astype('float16'),
        #                                np.load(npfiles[3]).astype('float16'),
        #                                np.load(npfiles[0]).astype('float16')]))
        # all_arrays.clear()
        xTrainLocal = np.load("raw_text_dat_vbincross234.npy")
        print("Loaded xTrain")
        x2TrainLocal = np.load("raw_sequence_data_vbincross234.npy")
        print("Loaded yTrain")
        yTrainLocal = np.load("all_words_vbincross234.npy")#.astype('float16')#yTrainLocal
        print("Loaded one-hot-encoded-wordbag\n")
    print("Shapes")
    print(xTrainLocal.shape)
    print(x2TrainLocal.shape)
    print(yTrainLocal.shape)
    print("_______")

    print("Generating lagged training matericies with multi-encoder-decoder")

    xTrain, x2Train, yTrain = xTrainLocal, x2TrainLocal, yTrainLocal#getTrainingBatch(xTrainLocal, yTrainLocal, tokenizer)
    print("\n________TRAINING SET SHAPE________")
    print("xTrain shape: "+str(xTrain.shape))
    print("x2Train shape: "+str(x2Train.shape))
    print("yTrain shape: "+str(yTrain.shape))
    print(keras.backend.floatx())
    #policy = tf.keras.mixed_precision.experimental.Policy('infer_float32_vars')
    print(keras.backend.floatx())
    #print(policy.should_cast_variables)
    print("xTrain dtype: ", str(xTrain.dtype))
    print("x2Train dtype: ", str(x2Train.dtype))
    print("yTrain dtype: ", str(yTrain.dtype))
    print("__________________________________\n")
else:
    print("Whoops hahahahah")

with tf.device('/gpu:1'):
    language_input = Input(shape=(None,), name="langInput")
    langauge_model_embed = Embedding(len_vocab, 50, input_length=50, name="langEmbedding")(language_input)
    #langnorm = LayerNormalization()(langauge_model_embed)
    lang_dropout = Dropout(0.3)(langauge_model_embed)
    langauge_model_LSTM1 = Bidirectional(CuDNNLSTM(128, return_sequences=True),
                                         name="langLSTM_1")(lang_dropout)

    context_input = Input(shape=(None,), name="contextInput")
    context_model = Embedding(len_vocab, 50, input_length=50, name="contextEmbedding")(context_input)
    cont_dropout = Dropout(0.3)(context_model)
    #embedlnorm = LayerNormalization()(context_model)
    context_model_LSTM1 = Bidirectional(CuDNNLSTM(128, return_sequences=True),
                                        name="contextLSTM_1")(cont_dropout)

    decoder = concatenate([langauge_model_LSTM1, context_model_LSTM1])
    decoder2 = Bidirectional(CuDNNLSTM(128, return_sequences=True), name="decLSTM_1")(decoder)
    decoder3 = Bidirectional(CuDNNLSTM(128, return_sequences=False), name="decLSTM_2")(decoder2)
    outputs = Dense(len_vocab, activation='softmax', name="decOutput")(decoder3)

    # compile model
    model = Model(inputs=[language_input, context_input], outputs=outputs)
    print(model.summary())

    opts = 2#str(input("\nWhich optimizer (Adam: 1) (RMSProp: 2)"))
    if opts == 1:
        optimizer = Adam(lr=3e-4, clipvalue=1.0)
        print("picked Adam")
    elif opts == 2:
        optimizer = Nadam(lr=3e-4)
        print("picked NAdam")
    else:
        optimizer = RMSprop(lr=1e-4, clipvalue=1.0)
        print("picked rmsprop")

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])

    filepath = "Model\/weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='auto')
    frq = 0#int(input("\HistoGram Frequncy?"))
    write = False
    if frq > 0:
        write = True
    n = input("go?")
    tensorboard = TensorBoard(log_dir=("Tensorboard/"+"logs-date-("+datetime.datetime.now().strftime("%d-%Hh:%mm:%s")+")-graph/"),
                              update_freq='batch',
                              profile_batch=0)

    classic_reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=1, mode='auto',
                                       min_delta=0.0001, cooldown=10)

    bsiz = 256
    #input("\nBatchSize?")
    callbacks_list = [classic_reduce, tensorboard, PredictionCallback()]

    model.fit([xTrain, x2Train], yTrain,
              batch_size=int(bsiz),
              epochs=1000,
              validation_split=0.30,
              shuffle=True,
              callbacks=callbacks_list)


def get_seed(sequence, randomizer):
    print("Seed Word")
    for g in sequence:
        if len(g) == 0:
            continue
        randomizer.append(g[0])
        what = word_for_id(g[0], tokenizer)
        print(' ' + str(what), end='')
    print("\n")
    print("End seed word")
    print("\n")
    return randomizer

while True:
    n = input("\n\nKeep generating with test data?: ")
    if n=="n":
        break
    generate_desc(model, tokenizer, "Final")
