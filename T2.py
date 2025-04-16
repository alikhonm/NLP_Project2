# CSC 427: NLP
# Project 2
# Due Apr 22

from collections import Counter
import argparse
import re
import random

def uni_bi_gram(unigrams, bigrams):
    # Get the unigram and bigram counts
    unigram_counts = Counter(unigrams)
    bigram_counts = Counter(bigrams)

    # Calculate the total number of tokens
    total_num_tokens = sum(unigram_counts.values())

    # Calculate the unigram probs
    unigram_probs = {
        unigram: count / total_num_tokens
        for unigram, count in unigram_counts.items()
    }

    # Calculate the bigram probs
    bigram_probs = {
        bigram: count / unigram_counts[bigram[0]]
        for bigram, count in bigram_counts.items()
    }
    
    return unigram_probs,bigram_probs


def generate_sentence(bigram_model):
    # A variable that control the number of words in the generated sentence  
    sentence_length = 15
    # Get the random first word 
    start_words = [w1 for (w1, w2) in bigram_model.keys()]
    first_word = random.choice(start_words)
    sentence = first_word
    current_word = first_word

    # Getting and adding a new random word based on probability gained by bigram model
    for i in range(sentence_length - 1):
        # Get a list of the word succeeding the current word and its bigram prob
        candidates = [(w2, prob) for (w1, w2), prob in bigram_model.items() if w1 == current_word]
        if not candidates:
            break 
        # Divide the list into the succeeding words and its probabilities, and choosing a random word with respect to weights, decided by bigram probs
        words, probs = zip(*candidates)
        next_word = random.choices(words, weights=probs)[0]
        # Append the choosen random word to sentence 
        sentence = sentence + ' ' + next_word
        current_word = next_word
    
    return sentence

def print_grams(unigram_probs,bigram_probs):
    print("\nUnigram Probabilities (just the first 5):")
    for unigram, prob in list(unigram_probs.items())[:5]:
        print(f"P({unigram}) = {prob}")
    print("\nBigram Probabilities (just the first 5):")
    for bigram, prob in list(bigram_probs.items())[:5]:
        print(f"P({bigram[1]} | {bigram[0]}) = {prob}")
    
# Get the corpus file name as an argument from the command line
parser = argparse.ArgumentParser(description="Enter your corpus file path")
parser.add_argument("corpus_path", help="File path of your corpus.")
args = parser.parse_args()

filename = args.corpus_path

# Read the text file and normalize (make every word lowercase)
with open(filename, 'r') as file:
    text = file.read().lower()

# Tokenize using whitespace and punctuation
tokens = re.findall(r'\b\w+\b', text)

# Create lists of unigrams and bigrams
unigrams = tokens
bigrams = list(zip(tokens[:-1], tokens[1:])) # Pair every word with its succeeding word

unigram_probs, bigram_probs = uni_bi_gram(unigrams,bigrams)

print_grams(unigram_probs, bigram_probs)

# Getting a user input for optional actions
userInput = True
while (userInput):
    userInput = input("\nIf you want you can: \n1: Look at the unigram and bigram probabilities again\n2: Generate a Sentence. \n3: Do Add-1 Smoothing. \n4: Compute the perplexity of a test set \n5: Quit\n")
    
    # If not an integer ask for input again
    try:
        choice = int(userInput)
    except ValueError:
        print("\nPleace enter a number!")
        continue

    if (choice > 5 or choice < 1): 
        print("Please enter a valid number!") 
        continue

    # If the input is 1, print the unigram and bigram probabilities again
    if (choice == 1):
        print_grams(unigram_probs, bigram_probs)

    # If the input is 2, probabilistically generate a sentence
    elif (choice == 2):
        sentence = generate_sentence(bigram_probs)
        print("\n" + sentence)

    # FIXME: ADD OPTIONS FOR T3 AND T4 HERE.

    #If the input is 5, terminate the current session
    elif (choice == 5):
        print("\nHave a good day!\n")
        exit()