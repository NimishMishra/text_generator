import numpy as np
import tensorflow as tf
import glob
import codecs
import pickle

book_path = 'King_Arthur.txt.utf-8.txt'

# HYPERPARAMETERS
num_epochs = 10000
batch_size = 512
rnn_size = 512
num_layers = 3
keep_prob = 0.7
embed_dim = 512
seq_length = 30
learning_rate = 0.001
save_dir = './save'
with codecs.open(filename = book_path, mode = 'r', encoding='utf-8') as file:
    corpus = file.read() # Load the book into something tangible

# Let's get a lookup table working

def lookup_table(text):
    vocab = set(text)
    int_to_vocab = {key: word for key, word in enumerate(vocab)}
    vocab_to_int = {word: key for key, word in enumerate(vocab)}
    return vocab_to_int, int_to_vocab

# Punctuations need to recognized well

def recognize_token():
    return{
        '.': '||period||',
        ',': '||comma||',
        '"': '||quotes||',
        ';': '||semicolon||',
        '!': '||exclamation-mark||',
        '?': '||question-mark||',
        '(': '||left-parentheses||',
        ')': '||right-parentheses||',
        '--': '||emm-dash||',
        '\n': '||return||'
    }

import pickle

dictionary = recognize_token() # contains replacements for tokens
for token, replacement in dictionary.items():
    corpus = corpus.replace(token, ' {} '.format(replacement)) # Replaces a token with its name
corpus = corpus.lower()  # Removing any possible dependencies on capital case
corpus = corpus.split()

vocab_to_int, int_to_vocab = lookup_table(corpus)
corpus_int = [vocab_to_int[word] for word in corpus]
pickle.dump((corpus_int, vocab_to_int, int_to_vocab, dictionary), open('preprocess.p', 'wb'))


# We'll train on batches of input data

def batch_data(text, batch_size, length):
    words_per_batch = batch_size * length
    total_batches = len(text)//words_per_batch
    text = text [:total_batches * words_per_batch]
    y = np.array(text[1:] + [text[0]])
    x = np.array(text)

    x_batches = np.split(x.reshape(batch_size, -1), total_batches, axis = 1)
    y_batches = np.split(y.reshape(batch_size, -1), total_batches, axis=1)

    data = list(zip(x_batches, y_batches))
    return np.array(data)


train_graph = tf.Graph()
with train_graph.as_default():    
    
    # Initialize input placeholders
    input_text = tf.placeholder(tf.int32, [None, None], name='input')
    targets = tf.placeholder(tf.int32, [None, None], name='targets')
    lr = tf.placeholder(tf.float32, name='learning_rate')
    
    # Calculate text attributes
    vocab_size = len(int_to_vocab)
    input_text_shape = tf.shape(input_text)
    
    # Build the RNN cell
    lstm = tf.contrib.rnn.BasicLSTMCell(num_units=rnn_size)
    drop_cell = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob)
    cell = tf.contrib.rnn.MultiRNNCell([drop_cell] * num_layers)
    
    # Set the initial state
    initial_state = cell.zero_state(input_text_shape[0], tf.float32)
    initial_state = tf.identity(initial_state, name='initial_state')
    
    # Create word embedding as input to RNN
    embed = tf.contrib.layers.embed_sequence(input_text, vocab_size, embed_dim)
    
    # Build RNN
    outputs, final_state = tf.nn.dynamic_rnn(cell, embed, dtype=tf.float32)
    final_state = tf.identity(final_state, name='final_state')
    
    # Take RNN output and make logits
    logits = tf.contrib.layers.fully_connected(outputs, vocab_size, activation_fn=None)
    
    # Calculate the probability of generating each word
    probs = tf.nn.softmax(logits, name='probs')
    
    # Define loss function
    cost = tf.contrib.seq2seq.sequence_loss(
        logits,
        targets,
        tf.ones([input_text_shape[0], input_text_shape[1]])
    )
    
    # Learning rate optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate)
    
    # Gradient clipping to avoid exploding gradients
    gradients = optimizer.compute_gradients(cost)
    capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients if grad is not None]
    train_op = optimizer.apply_gradients(capped_gradients)


    import time

pickle.dump((seq_length, save_dir), open('params.p', 'wb'))
batches = batch_data(corpus_int, batch_size, seq_length)
num_batches = len(batches)
start_time = time.time()

with tf.Session(graph=train_graph) as sess:
    sess.run(tf.global_variables_initializer())
    
    for epoch in range(num_epochs):
        state = sess.run(initial_state, {input_text: batches[0][0]})
        
        for batch_index, (x, y) in enumerate(batches):
            feed_dict = {
                input_text: x,
                targets: y,
                initial_state: state,
                lr: learning_rate
            }
            train_loss, state, _ = sess.run([cost, final_state, train_op], feed_dict)
            
        time_elapsed = time.time() - start_time
        print('Epoch {:>3} Batch {:>4}/{}   train_loss = {:.3f}   time_elapsed = {:.3f}   time_remaining = {:.0f}'.format(
            epoch + 1,
            batch_index + 1,
            len(batches),
            train_loss,
            time_elapsed,
            ((num_batches * num_epochs)/((epoch + 1) * (batch_index + 1))) * time_elapsed - time_elapsed))

        # save model every 10 epochs
        if epoch % 10 == 0:
            saver = tf.train.Saver()
            saver.save(sess, save_dir)
            print('Model Trained and Saved')


corpus_int, vocab_to_int, int_to_vocab, token_dict = pickle.load(open('preprocess.p', mode='rb'))
seq_length, save_dir = pickle.load(open('params.p', mode='rb'))