# CSC 427: NLP
# Project 2
# Due Apr 22

from collections import Counter
import argparse
import re

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

# Example outputs
print("Unigram Probabilities (just the first 5):")
for unigram, prob in list(unigram_probs.items())[:5]:
    print(f"P({unigram}) = {prob}")
print("\nBigram Probabilities (just the first 5):")
for bigram, prob in list(bigram_probs.items())[:5]:
    print(f"P({bigram[1]} | {bigram[0]}) = {prob}")
