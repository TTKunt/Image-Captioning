from __future__ import print_function
from __future__ import division
from keras.models import Sequential
from keras.layers.core import Dense, RepeatVector, Reshape
from keras.preprocessing import sequence
from keras.optimizers import Adagrad
import keras.layers as layers
from keras.layers.core import TimeDistributedDense, Merge
from keras.layers import Embedding, GRU
from keras.models import model_from_json
from scipy.ndimage import imread
import cPickle as pickle
from scipy.misc import imresize
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
import urllib
import cStringIO
import json
import scipy.io
import os.path
import hickle
import numpy as np

# controlling variables
test_size = 100
max_caption_len = 16

dataname = 'MSCOCO'

image_caption_model = 1  # 1: Feed image at the beginning only (dim = 300),
# 2: Feed image at the beginning (dim=300) and at each time (dim = 150)

if dataname == 'MSCOCO':
    batch_size = 200  # multiple of max_caption_len
    nb_epoch = 50
    if image_caption_model == 1:
        iteration = 20
    else:
        iteration = 24
elif dataname == 'Flickr_30k':
    batch_size = 1000
    nb_epoch = 30
    if image_caption_model == 1:
        iteration = 17
    else:
        iteration = 24
elif dataname == 'MSCOCO':
    batch_size = 1000
    nb_epoch = 20
    if image_caption_model == 1:
        iteration = 20
    else:
        iteration = 26

rnn_model_name = 'LSTM'  # or LSTM, GRU, SimpleRNN
output_rnn_dim = 512
img_dim = 4096
word_vec_dim = 300
image_caption_dict = {}
image_path = 'dataset/' + dataname + '_Dataset/'
VGG_model = []
Generating_model_1 = []
Generating_model_2 = []
Generating_model = []

vocab = []
vocab_size = 0
index_to_word = {i: c for i, c in enumerate(vocab)}
word_to_index = {c: i for i, c in enumerate(vocab)}


def update_vocab(word):
    global vocab, vocab_size, index_to_word, word_to_index
    vocab.append(word)
    index_to_word[vocab_size] = word
    word_to_index[word] = vocab_size
    vocab_size = len(vocab)

update_vocab('FIRST')
update_vocab('START')


def to_categorical(y, nb_classes=None):
    '''Convert class vector (integers from 0 to nb_classes)
    to binary class matrix, for use with categorical_crossentropy.
    '''
    y = np.asarray(y, dtype='int32')
    if not nb_classes:
        nb_classes = np.max(y)+1
    Y = np.zeros((len(y), nb_classes), dtype='bool')
    for i in range(len(y)):
        Y[i, y[i]] = 1
    return Y


def get_extract_data():
    global image_caption_dict
    print("Get extracted Data")

    file_save = '/scratch/nguyenkh/final_project/dataset/'+dataname+'/'+'data_processed.hkl'

    # run for the first time to generate result file, if not it takes longer to run
    if os.path.isfile(file_save):
        print("Loading result file")
        with open(file_save, 'r') as f:
            result = hickle.load(f)
            [(image_train, partial_caption_train, next_word_train), (image_test, partial_caption_test, next_word_test)] = result

        print("Loading Vocab file")
        with open('models/vocab_list_'+dataname+'.pkl', 'rb') as f:
            vocabs = pickle.load(f)
            for v in vocabs:
                if v not in vocab:
                    update_vocab(v)

        if dataname != 'MSCOCO':
            next_word_train = to_categorical(next_word_train,vocab_size)
            next_word_test = to_categorical(next_word_test,vocab_size)
            result = [(image_train, partial_caption_train, next_word_train), (image_test, partial_caption_test, next_word_test)]

        print("Loading Image Caption dict")
        with open('dataset/'+dataname+'/image_caption_dict.pkl', 'rb') as f:
            image_caption_dict = pickle.load(f)

    else:

        with open('dataset/'+dataname+'/dataset.json', 'rb') as f:
            dataset = json.load(f)

        with open('dataset/'+dataname+'/vgg_feats.mat', 'rb') as f:
            vgg_feats = np.transpose(scipy.io.loadmat(f)['feats'])

        train_data_list = [element for element in dataset['images'] if element['split'] == 'train']
        test_data_list = [element for element in dataset['images'] if element['split'] == 'test']

        image_train_list = []
        partial_caption_train_list = []
        next_word_train_list = []
        image_test_list = []
        partial_caption_test_list = []
        next_word_test_list = []

        for i, element in enumerate(train_data_list):
            print(str(i)+'/'+str(len(train_data_list)))
            image_id = element['imgid']
            image_data = vgg_feats[image_id,:]
            sentences = element['sentences']

            for sentence in sentences:
                words = sentence['tokens'][:]
                words.insert(0, 'START')

                list_words = []

                if words[-1] != '.':
                    words.append('.')

                for i in range(len(words)):
                    image_train_list.append(image_data)

                    if i < len(words)-1:
                        word = str(words[i]).lower() if i is not 0 else 'START'
                        next_word = str(words[i+1]).lower()

                        if word not in word_to_index.keys():
                            update_vocab(word)
                        if next_word not in word_to_index.keys():
                            update_vocab(next_word)

                        list_words.append(word_to_index[word])

                    partial_caption_train_list.append(list_words[:])

                    next_word_train_list.append(word_to_index[next_word])

        for i, element in enumerate(test_data_list):
            print(str(i)+'/'+str(len(test_data_list)))
            image_id = element['imgid']
            image_data = vgg_feats[image_id,:]
            sentences = element['sentences']
            temp_dict = {'image_data': image_data, 'sentences': sentences}

            image_caption_dict[element['filename']] = temp_dict

            for sentence in sentences:
                words = sentence['tokens'][:]
                words.insert(0, 'START')

                list_words = []

                if words[-1] != '.':
                    words.append('.')

                for i in range(len(words)):
                    image_test_list.append(image_data)

                    if i < len(words)-1:
                        word = str(words[i]).lower() if i is not 0 else 'START'
                        next_word = str(words[i+1]).lower()

                        if word not in word_to_index.keys():
                            update_vocab(word)
                        if next_word not in word_to_index.keys():
                            update_vocab(next_word)

                        list_words.append(word_to_index[word])

                    partial_caption_test_list.append(list_words[:])

                    next_word_test_list.append(word_to_index[next_word])

        print("Dump Vocab List, vocab size = " + str(vocab_size))
        with open('models/vocab_list_'+dataname+'.pkl', 'wb') as f:
            pickle.dump(vocab, f, pickle.HIGHEST_PROTOCOL)

        print("Dump Image Caption Dict")
        with open('dataset/'+dataname+'/image_caption_dict.pkl', 'wb') as f:
            pickle.dump(image_caption_dict, f, pickle.HIGHEST_PROTOCOL)

        image_train = np.array(image_train_list)
        image_test = np.array(image_test_list)

        partial_caption_train = sequence.pad_sequences(partial_caption_train_list, maxlen=max_caption_len, padding='post', truncating='post')
        partial_caption_test = sequence.pad_sequences(partial_caption_test_list, maxlen=max_caption_len, padding='post', truncating='post')

        if dataname == 'MSCOCO':
            next_word_train = np.array(next_word_train_list)
            next_word_test = np.array(next_word_test_list)
        else:
            next_word_train = to_categorical(next_word_train_list, vocab_size)
            next_word_test = to_categorical(next_word_test_list, vocab_size)

        result = [(image_train, partial_caption_train, next_word_train), (image_test, partial_caption_test, next_word_test)]

        print("Dump image_train and others!")
        with open(file_save, 'w') as f:
            hickle.dump(result, f)

    if dataname == 'MSCOCO':
        n_train_examples = image_train.shape[0]

        n_batches = int(np.ceil(n_train_examples/batch_size))

        train_chunks = []

        for i in range(0, n_batches):
            start = i*batch_size
            end = (i+1)*batch_size
            if (i+1)*batch_size > n_train_examples:
                end = n_train_examples

            index_range = range(start,end)
            train_chunks.append((image_train[index_range,:], partial_caption_train[index_range,:], next_word_train[index_range]))

        result = (train_chunks, (image_test, partial_caption_test, next_word_test))

    return result


def load_model(model_name, weights_path=None, model_caption=None):
    print("Loading model")

    if not model_caption:
        model_caption = image_caption_model

    image_caption_json = 'models/'+dataname+'/image_captioning_' + model_name + 'model_'+str(model_caption)+\
                         '_output_rnn_'+str(output_rnn_dim)+'.json'

    if weights_path:  # Second tim run or predict mode
        model = model_from_json(open(image_caption_json).read())
        model.load_weights(weights_path)
    else:

        if image_caption_model == 1:
            image_model = Sequential()
            image_model.add(Dense(word_vec_dim, input_dim=img_dim))
            image_model.add(Reshape((1,word_vec_dim)))

            language_model = Sequential()
            language_model.add(Embedding(vocab_size, word_vec_dim, input_length=max_caption_len))
            language_model.add(GRU(output_dim=word_vec_dim, return_sequences=True))
            language_model.add(TimeDistributedDense(word_vec_dim))

            model = Sequential()
            model.add(Merge([image_model, language_model], mode='concat', concat_axis=1))
            model.add(getattr(layers, model_name)(output_rnn_dim, return_sequences=False, input_shape=(max_caption_len+1, word_vec_dim),stateful=False))
            model.add(Dense(vocab_size, activation='softmax'))

        elif image_caption_model == 2:

            image_model1 = Sequential()
            image_model1.add(Dense(150, input_dim=img_dim))
            image_model1.add(RepeatVector(max_caption_len))

            image_model2 = Sequential()
            image_model2.add(Dense(450, input_dim=img_dim))
            image_model2.add(Reshape((1,450)))

            language_model = Sequential()
            language_model.add(Embedding(vocab_size, 300, input_length=max_caption_len))
            language_model.add(GRU(output_dim=300, return_sequences=True))
            language_model.add(TimeDistributedDense(300))

            model1 = Sequential()
            model1.add(Merge([image_model1, language_model],mode='concat', concat_axis=-1))

            model = Sequential()
            model.add(Merge([image_model2, model1], mode='concat', concat_axis=1))
            model.add(getattr(layers, model_name)(output_rnn_dim, input_shape=(max_caption_len+1, 450),
                                                  return_sequences=False, stateful=False))
            model.add(Dense(vocab_size, activation='softmax'))

        json_string = model.to_json()
        open(image_caption_json, 'w').write(json_string)

    opt = Adagrad()
    model.compile(loss='categorical_crossentropy', optimizer=opt)
    return model


def get_caption(model, filename, file_data, model_caption=None):

    if not model_caption:
        model_caption = image_caption_model

    print(filename)

    result = []

    partial_caption = [word_to_index['START']]

    while True:
        partial_caption_padding = sequence.pad_sequences([partial_caption[:]],maxlen=max_caption_len,padding='post')

        if model_caption != 2:
            next_word_predict = model.predict([file_data.reshape((1,4096)), partial_caption_padding])[0]
        else:
            next_word_predict = model.predict([file_data.reshape((1,4096)), file_data.reshape((1,4096)), partial_caption_padding])[0]

        next_word_predict = int(np.argmax(next_word_predict))

        partial_caption.append(next_word_predict)

        next_word = index_to_word[next_word_predict]
        result.append(next_word)

        if next_word == '.' or len(result) == max_caption_len:
            break

    print(result)

    return result


def calculate_bleu_score(model, number):
    print("Calculate BLUE score")

    current_image = 0

    results = []

    for key, value in image_caption_dict.iteritems():
        if current_image == number:
            break
        current_image += 1

        image_path_new = image_path+key

        result = get_caption(model, image_path_new, value['image_data'])

        results.append(result)

    return results


def train():
    iteration = 0

    model_file = 'models/'+dataname+'/' + rnn_model_name + '_model_'+str(image_caption_model)+'_output_rnn_'+\
                 str(output_rnn_dim)+'_weights_iteration_' + str(iteration) + '.h5'

    if dataname != 'MSCOCO':
        [(image_train, partial_caption_train, next_word_train), (image_test, partial_caption_test, next_word_test)] = get_extract_data()
    else:
        (train_chunks, (image_test, partial_caption_test, next_word_test)) = get_extract_data()
        next_word_test = to_categorical(next_word_test, vocab_size)

    if iteration == 0:
        model = load_model(rnn_model_name)
    else:
        model = load_model(rnn_model_name, model_file)

    best_acc = 0
    best_iter = 0

    while iteration <= nb_epoch:
        iteration += 1
        print()
        print('-' * 50)
        print('Iteration', iteration)

        print("Start fitting model")

        if dataname != 'MSCOCO':
            if image_caption_model != 2:
                results = model.fit([image_train, partial_caption_train], next_word_train, batch_size, nb_epoch=1,
                          validation_data=([image_test, partial_caption_test], next_word_test),
                          show_accuracy=True)
            else:
                results = model.fit([image_train, image_train, partial_caption_train], next_word_train, batch_size, nb_epoch=1,
                          validation_data=([image_test, image_test, partial_caption_test], next_word_test),
                          show_accuracy=True)

            val_acc, val_loss, acc, loss = results.history['val_acc'][0], results.history['val_loss'][0], results.history['acc'][0], results.history['loss'][0]
        else:
            train_results = np.zeros((len(train_chunks),2))

            print("fit train data with chunks")
            for i,chunk in enumerate(train_chunks):
                print('\t\t'+str(i)+'/'+str(len(train_chunks)))

                (image_train, partial_caption_train, next_word_train) = chunk

                next_word_train = to_categorical(next_word_train, vocab_size)

                if image_caption_model != 2:
                    temp_result = model.fit([image_train, partial_caption_train], next_word_train, batch_size, nb_epoch=1, show_accuracy=True)
                    train_results[i,0] = temp_result.history['loss'][0]
                    train_results[i,1] = temp_result.history['acc'][0]
                else:
                    temp_result = model.fit([image_train, image_train, partial_caption_train], next_word_train, batch_size, nb_epoch=1, show_accuracy=True)
                    train_results[i,0] = temp_result.history['loss'][0]
                    train_results[i,1] = temp_result.history['acc'][0]

            if image_caption_model != 2:
                test_results = model.evaluate([image_test, partial_caption_test], next_word_test, show_accuracy=True)
            else:
                test_results = model.evaluate([image_test, image_test, partial_caption_test], next_word_test, show_accuracy=True)

            acc, loss, val_acc, val_loss = np.mean(train_results[:,1]), np.mean(train_results[:,0]), test_results[1], test_results[0]

        # check the result of current epoch
        results = calculate_bleu_score(model, 1)

        # save model check point
        # save the best model
        if val_acc > best_acc:
            previous_file = 'models/'+dataname+'/best_' + rnn_model_name + '_model_'+str(image_caption_model)+\
                            '_output_rnn_'+str(output_rnn_dim)+'_weights_iteration_' + str(best_iter) + '.h5'
            if os.path.isfile(previous_file):
                os.remove(previous_file)

            best_acc = val_acc
            best_iter = iteration
            model.save_weights('models/'+dataname+'/best_' + rnn_model_name + '_model_'+str(image_caption_model)+
                               '_output_rnn_'+str(output_rnn_dim)+'_weights_iteration_' + str(best_iter) + '.h5',
                           overwrite=True)

        previous_file = 'models/'+dataname+'/' + rnn_model_name + '_model_'+str(image_caption_model)+'_output_rnn_'+\
                        str(output_rnn_dim)+'_weights_iteration_' + str(iteration-1) + '.h5'

        if os.path.isfile(previous_file):
            os.remove(previous_file)

        model.save_weights('models/'+dataname + '/' + rnn_model_name + '_model_'+str(image_caption_model)+
                           '_output_rnn_'+str(output_rnn_dim)+'_weights_iteration_' + str(iteration) + '.h5',
                           overwrite=True)

        # write to history file
        with open('models/'+dataname+'/history_'+rnn_model_name+'_model_'+ str(image_caption_model)+'_output_rnn_'+
                          str(output_rnn_dim)+".txt", "ab") as text_file:
            text_file.write('Iteration: %s\tloss: %s\tacc: %s\tval_loss: %s\tval_acc: %s\texample: %s\n' % (iteration, loss, acc, val_loss, val_acc, results))


def load_VGG_model():
    print("Loading VGG model")
    global Generating_model_1, Generating_model_2

    global VGG_model, Generating_model
    pretrain_model = model_from_json(open('models/CNN_pretrained_model.json').read())

    pretrain_model.load_weights('models/vgg16_weights.h5')

    VGG_model = model_from_json(open('models/CNN_model.json').read())

    for k in range(len(pretrain_model.layers)):
        weights_loaded = pretrain_model.layers[k].get_weights()
        if k < 35:
            VGG_model.layers[k].set_weights(weights_loaded)

    VGG_model.compile(loss='categorical_crossentropy', optimizer='sgd')

    with open('models/vocab_list_'+dataname+'.pkl', 'rb') as f:
        vocabs = pickle.load(f)
        for v in vocabs:
            if v not in vocab:
                update_vocab(v)

    Generating_model_1 = load_model(rnn_model_name, 'models/'+dataname+'/best_' + rnn_model_name + '_model_1_output_rnn_'
                                    +str(output_rnn_dim)+'_weights_iteration_20.h5', 1)

    Generating_model_2 = load_model(rnn_model_name, 'models/' + dataname + '/best_' + rnn_model_name + '_model_4_output_rnn_'
                                    + str(output_rnn_dim) +'_weights_iteration_24.h5', 4)


def demo(file_name):

    file = cStringIO.StringIO(urllib.urlopen(file_name).read())

    original_image = imread(file)

    X = np.array(np.transpose(imresize(original_image, (224, 224)),
                                 (2, 0, 1)).astype('float32')).reshape((1, 3, 224, 224))

    features = VGG_model.predict(X)[0]

    temp_result1= get_caption(Generating_model_1, file_name, features, 1)
    temp_result2 = get_caption(Generating_model_2, file_name, features, 4)

    result1 = (' '.join(temp_result1[:-1])).capitalize() + temp_result1[-1]
    result2 = (' '.join(temp_result2[:-1])).capitalize() + temp_result2[-1]

    print(result1)
    print(result2)

    return result1, result2


def compute_BLEU_score_corpus():
    print("Generate Captions")

    print("Loading Vocab")

    with open('models/vocab_list_'+dataname+'.pkl', 'rb') as f:
        vocabs = pickle.load(f)
        for v in vocabs:
            if v not in vocab:
                update_vocab(v)

    generating_model = load_model(rnn_model_name, 'models/'+dataname+'/best_' + rnn_model_name + '_model_'+
                                  str(image_caption_model)+'_output_rnn_'+str(output_rnn_dim)+'_weights_iteration_' +
                                  str(iteration) + '.h5')

    print("Loading Image Caption dict")
    with open('dataset/'+dataname+'/image_caption_dict.pkl', 'rb') as f:
        image_caption_dict = pickle.load(f)

    sentences = []
    references = []

    i = 0

    # Calculate Bleu-n
    weights = [0.25,0.25,0.25,0.25]

    for key, value in image_caption_dict.iteritems():
        print(str(i)+'/'+str(len(image_caption_dict.keys())))
        i += 1

        image_path_new = image_path+key

        result = get_caption(generating_model, image_path_new, value['image_data'])

        sentences.append(result)

        reference = [[str(word).lower() for word in x['tokens']] for x in value['sentences']]

        references.append(reference)

    corpus_score = corpus_bleu(references, sentences, weights) * 100

    print(corpus_score)

    return corpus_score


def generate_captions():
    print("Generate Captions")

    print("Loading Vocab")

    with open('models/vocab_list_'+dataname+'.pkl', 'rb') as f:
        vocabs = pickle.load(f)
        for v in vocabs:
            if v not in vocab:
                update_vocab(v)

    generating_model = load_model(rnn_model_name, 'models/'+dataname+'/best_' + rnn_model_name + '_model_'+
                                  str(image_caption_model)+'_output_rnn_'+str(output_rnn_dim)+'_weights_iteration_' +
                                  str(iteration) + '.h5')

    print("Loading Image Caption dict")
    with open('dataset/'+dataname+'/image_caption_dict.pkl', 'rb') as f:
        image_caption_dict = pickle.load(f)

    results = []

    i = 0

    for key, value in image_caption_dict.iteritems():
        print(str(i)+'/'+str(len(image_caption_dict.keys())))
        i += 1

        image_path_new = image_path+key

        temp_result = get_caption(generating_model, image_path_new, value['image_data'])

        result = (' '.join(temp_result[:-1])).capitalize() + temp_result[-1]

        print(result)

        temp_dict = {'image_id': int(key[13:-4]), 'caption': result}

        results.append(temp_dict)

    with open('models/'+dataname+'/results_' + rnn_model_name + '_model_' + str(image_caption_model) +\
                      '_evaluation.json', 'w') as f:
        json.dump(results, f)

    with open('eval/results/captions_val2014_fakecap_results.json', 'w') as f:
        json.dump(results, f)

    import eval.evaluate as evaluate
    evaluate.eval()


if __name__ == "__main__":

    print("Using model:" + str(image_caption_model))

    # # if you want to train, run the following code
    # train()

    # # if you want to calculate BLEU score for Flickr_8k, Flickr_30k, MSCOCO
    # compute_BLEU_score_corpus()

    # # if you want to generate captions for MSCOCO dataset and evaluate with MSCOCO evaluate code
    # generate_captions()

    # # if you want to demo instantly:
    # load_VGG_model()
    # image_link = ""
    # demo(image_link)

    # # if you want to run server for anyone can use in your own server: open server.py and edit host_name and port
    # and run in the command line: python server.py