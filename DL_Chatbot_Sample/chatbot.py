# importing the Natural Language Tool Kit for natural language
import nltk
# Firt time if you don't have the nltk packages
#nltk.download()

# Importing the general purpose array processing package
import numpy
# Importing tflearn to provide higher level API to Tensorflow frame work
import tflearn
import tensorflow
# for randomizing
import random
# Used to stem our words
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import pickle

# To read a json file
import json
with open('intents.json') as file:
    data = json.load(file)

#print(data["intents"])

try:
	with open("data.pickle", "rb") as f:
		words, labels, training, output = pickle.load(f)
except:
	words = []
	labels = []
	# For each pattern, to put another element in doc_y that stands what intent its part of which tag
	# Each entry in docs_x corresponds to the docs_y
	# docs_x will be the pattern and docs_y will be intents, important for training the model
	docs_x = []
	docs_y = []

	# Loop through all the dictionaries of the intents data
	for intent in data['intents']:
	    for pattern in intent['patterns']:
	    	# We are doing something specific called stemming
	    	# Stemming: take each word from pattern and bring it down to the root word
	        # Tokenizing, to get all the words in patterns
	        wrds = nltk.word_tokenize(pattern)
	        # Rather than looping and then appending each one in we can just extend the list and add all the words
	        words.extend(wrds)
	        # Add the pattern of words to the docs
	        docs_x.append(wrds)
	        docs_y.append(intent["tag"])
	        
	    if intent['tag'] not in labels:
	        labels.append(intent['tag'])

	# Stem all the words that we have in words list and remove any duplicate elements to figure the vocabulary size of the model
	# Lower to convert into lowercase
	words = [stemmer.stem(w.lower()) for w in words if w != "?"]
	# Remove all the duplicates and converts back to sorted list
	words = sorted(list(set(words)))
	# Sorted labels
	labels = sorted(labels)

	# Till now we have only strings, neural net only understand numbers

	# Creating training and testing dataset and making our data ready to feed into our model
	training = []
	output = []

	out_empty = [0 for _ in range(len(labels))]

	# Gives us bunch of bags of words
	for x, doc in enumerate(docs_x):
	    bag = []
	    # Stemming them to docs
	    wrds = [stemmer.stem(w.lower()) for w in doc]
	    # We are going to see where the tag is in the list and set that value to 1
	    for w in words:
	        if w in wrds:
	            bag.append(1)
	        else:
	            bag.append(0)

	    output_row = out_empty[:]
	    output_row[labels.index(docs_y[x])] = 1
	    # Appanding the list that is bagged to training list
	    # Both one hot encoded
	    training.append(bag)
	    output.append(output_row)

	# Train these into arrays
	# Changing our output into numpy arrays
	# As the model understands this form
	training = numpy.array(training)
	output = numpy.array(output)

	with open("data.pickle", "wb") as f:
		pickle.dump((words, labels, training, output), f)

# To visualize and classify the data use tensorflow
# To get rid of all the previous settings
from tensorflow.python.framework import ops
ops.reset_default_graph()

# Define the input layer with shape that we are expecting for our model
# Training length 0 adn each training length gonna be same
net = tflearn.input_data(shape=[None, len(training[0])])
# Adding this fully connected layer to the 
# For another hidden layer has 8 neurons
net = tflearn.fully_connected(net, 8)
# Finally the output layer
# Softmax allows us to get the probabilities for each output
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

# Training the model
model = tflearn.DNN(net)

try:
	model.load("model.tflearn")
except:
	# Fit the model
	# Passing the training, sees the data 1000 times
	model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
	model.save("model.tflearn")

# Now time for predictions
def bag_of_words(s, words):
	# Create a blank bag of words list, change the element if the element exist
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
    	# The current word that we are looking in the words list
        for i, w in enumerate(words):
        	# Which is equal to the word in the sentence
            if w == se:
                bag[i] = 1
    # Take the bag of words convert into numpy array and return it     
    return numpy.array(bag)


def chat():
    print("Start talking with 'ROCKY' the bot (type quit to stop)!")
    while True:
        inp = input("You: ")
        # To get out of the program
        if inp.lower() == "quit":
            break

        # Feeding the model
        results = model.predict([bag_of_words(inp, words)])[0]

        # Gives the probabilities
        #print(results)

        # Gives the index to the greatest value on the list
        # Then with that index we use it to respond to actual display
        results_index = numpy.argmax(results)
        # print(results)

        # Gives us the label that it thinks our message
        tag = labels[results_index]
        # To print the tag
        #print(tag)

        # Threshold
        if (results[results_index]>0.7):
        	# Instead of tags we are giving the responses in the intents
        	for tg in data["intents"]:
        		if tg['tag'] == tag:
        			responses = tg['responses']
        	print(random.choice(responses))
        else:
        	print("I didn't get that, try again")
        

chat()