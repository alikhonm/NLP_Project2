import argparse
import random
import math
from collections import Counter

# Function to determine if a character alphanumeric or underscore
def is_word_char(c):
    return c.isalnum() or c == '_'

# Function to tokenize text into words
def word_tokenizer(text):
    """
    Tokenizes the input text into a list of words by identifying alphanumeric characters and underscores.
    Args:
        text (str): Input text to tokenize.
    Returns:
        tokens (list): A list of tokens.
    """
    tokens = []         # List to store the tokens
    word = ""           # Temporary holder to build words

    for char in text:           # Iterate over each character in the text
        if char.isalnum() or char == '_' or char =='\'':    # Check if the word is alphanumerical, underscore, or '
            word += char                                    # Append the character to the current word
        else:  
            if word:                                        # If a non-word character is encountered and the word is non emptry
                tokens.append(word)                         # Store the word in list of tokens
                word = ""                                   # Reset the word to continue building
    
    # Catch the last word if the text ends with a word character
    if word:
        tokens.append(word)

    return tokens

# Function to calculate Maximum Likelihood Estimate (MLE) for unigrams and bigrams
def uni_bi_gram(unigrams, bigrams):
    """
    Computes the Maximum Likelihood Estimate (MLE) probabilities for unigrams and bigrams.
    Args:
        unigrams (list): List of unigram tokens.
        bigrams (list): List of bigram pairs (tuples).
    Returns:
        unigram_probs, bigram_probs (dict, dict): Unigram probabilities and bigram probabilities.
    """
    unigram_counts = Counter(unigrams)  # Count the frequency of each unigram
    bigram_counts = Counter(bigrams)    # Count the frequency of each bigram

    total_tokens = sum(unigram_counts.values())  # Total number of tokens in the corpus

    # MLE for unigrams: P(w) = C(w) / N
    unigram_probs = {w: c / total_tokens 
                     for w, c in unigram_counts.items()}
    
    # MLE for bigrams: P(w2 | w1) = C(w1, w2) / C(w1)
    bigram_probs = {bg: c / unigram_counts[bg[0]] 
                    for bg, c in bigram_counts.items()}
    
    return unigram_probs, bigram_probs

# Function to apply Add-1 smoothing to unigrams and bigrams
def add_one_smoothing(unigram_counts, bigram_counts):
    """
    Applies Add-1 smoothing to the unigram and bigram probabilities.
    Args:
        unigram_counts (Counter): Counter object containing unigram counts.
        bigram_counts (Counter): Counter object containing bigram counts.
    Returns:
        sm_unigrams, sm_bigrams (dict, dict): Smoothed unigram probabilities and smoothed bigram probabilities.
    """
    vocab = set(unigram_counts.keys())  # Vocabulary (unique words)
    V = len(vocab)  # Size of the vocabulary
    total = sum(unigram_counts.values())  # Total occurrences of all unigrams

    # Smoothed unigram probabilities: (C(w) + 1) / (N + V)
    sm_unigrams = {w: (unigram_counts[w] + 1) / (total + V) for w in vocab}

    # Smoothed bigram probabilities: (C(w1, w2) + 1) / (C(w1) + V)
    sm_bigrams = {}
    # for w1 in vocab:
    #     denom = unigram_counts[w1] + V  # Denominator for bigram smoothing (count of unigram in the text + size of the vocabulary)
    #     for w2 in vocab:
    #         sm_bigrams[(w1, w2)] = (bigram_counts.get((w1, w2), 0) + 1) / denom

    # Iterate over all the bigrams instead of all |V| * |V| words to save memory
    for (w1, w2), count in bigram_counts.items():   
        denom = unigram_counts[w1] + V
        sm_bigrams[(w1, w2)] = (count + 1) / denom
    
    return sm_unigrams, sm_bigrams

# Function to compute the perplexity of a test set given bigram probabilities
def compute_perplexity(test_tokens, bigram_probs):
    """
    Computes the perplexity of a test set using the bigram model.
    Args:
        test_tokens (list): List of tokens from the test text.
        bigram_probs (dict): Dictionary containing bigram probabilities.
    Returns:
        float: Perplexity score.
    """
    test_bigrams = list(zip(test_tokens[:-1], test_tokens[1:]))  # Create bigrams from the test tokens
    log_sum = 0.0  # Initialize the log probability sum

    # Calculate the log probability of each bigram
    for bg in test_bigrams:
        p = bigram_probs.get(bg, 0.0)  # Get the probability of the bigram
        if p <= 0.0:                   # If the probability is zero, return infinite perplexity
            return float('inf')
        log_sum += math.log2(p)        # Add the log2 of the probability to the sum
    
    M = len(test_bigrams)  # Number of bigrams in the test set
    return 2 ** (-log_sum / M)  # Return perplexity as 2 raised to the negative average log-probability

# Function to print the first five unigram and bigram probabilities
def print_grams(unigram_probs, bigram_probs):
    """
    Prints the first five unigram and bigram probabilities.
    Args:
        unigram_probs (dict): Unigram probabilities.
        bigram_probs (dict): Bigram probabilities.
    """
    print("\nUnigram Probabilities (first 5):")
    for w, p in list(unigram_probs.items())[:5]:  # Display first 5 unigram probabilities
        print(f"P({w}) = {p}")
    
    print("\nBigram Probabilities (first 5):")
    for (w1, w2), p in list(bigram_probs.items())[:5]:  # Display first 5 bigram probabilities
        print(f"P({w2}|{w1}) = {p}")

# Function to print the top ten unigram and bigram probabilities
def print_top_grams(unigram_probs, bigram_probs):   
    """
    Prints the top ten unigram and bigram probabilities.
    Args:
        unigram_probs (dict): Unigram probabilities
        bigram_probs (dict): Bigram probabilities.
    """
    print("\nTop 10 Unigram Probabilities:")
    for w, p in sorted(unigram_probs.items(), key=lambda x: x[1], reverse=True)[:10]:   # Display the top 10 unigram probabilities
        print(f"P({w}) = {p}")

    print("\nTop 10 Bigram Probabilities:")
    for (w1, w2), p in sorted(bigram_probs.items(), key=lambda x: x[1], reverse=True)[:10]:   # Display the top 10 bigram probabilities
        print(f"P({w2}|{w1}) = {p}")

# Function to create a successor table for bigram-based sampling
def build_bigram_successors(bigram_probs):
    """
    Builds a successor table for bigrams for efficient sampling.
    Args:
        bigram_probs (dict): Dictionary containing bigram probabilities.
    Returns:
        succ (dict): Successor table for bigram-based sampling.
    """
    succ = {}  # Initialize an empty dictionary for the successor table
    for (w1, w2), p in bigram_probs.items():  # Iterate through all bigrams
        if w1 not in succ:
            succ[w1] = ([], [])  # If w1 not in the successor table, initialize an entry
        succ[w1][0].append(w2)  # Append w2 as a possible successor to w1
        succ[w1][1].append(p)   # Append the probability of w2 given w1
    
    return succ  # Return the successor table

# Function to generate a sentence based on unigram and bigram probabilities
def generate_sentence(unigram_probs, bigram_succ, sentence_length=15):
    """
    Generates a random sentence based on unigram and bigram probabilities.
    Args:
        unigram_probs (dict): Unigram probabilities for the start of the sentence.
        bigram_succ (dict): Successor table for bigram sampling.
        sentence_length (int): Desired length of the generated sentence, set to 15 by default.
    Returns:
        sentence (str): A generated sentence.
    """
    words, weights = zip(*unigram_probs.items())  # Get words and their corresponding probabilities
    current = random.choices(words, weights=weights)[0]  # Sample a starting word using unigram probabilities
    sentence = [current]  # Initialize sentence with the starting word
    
    # Generate the rest of the sentence using bigram probabilities
    for _ in range(sentence_length - 1):
        if current not in bigram_succ:  # If no successor exists for the current word, stop
            break
        next_words, next_weights = bigram_succ[current]  # Get possible next words and their probabilities
        current = random.choices(next_words, weights=next_weights)[0]  # Sample the next word
        sentence.append(current)  # Append the sampled word to the sentence
    
    return " ".join(sentence)  # Return the sentence as a string

def main():
    # Argument parsing for the input corpus path
    parser = argparse.ArgumentParser(description="N-Grams, Language Modeling & Perplexity")
    parser.add_argument("corpus_path", help="Path to the training corpus.")
    args = parser.parse_args()

    # Load and tokenize the training corpus
    with open(args.corpus_path, 'r') as file:
        text = file.read().lower()  # Read and convert to lowercase
    tokens = word_tokenizer(text)  # Tokenize the text

    # Ensure there are enough tokens in the corpus
    if len(tokens) < 2:
        print("Corpus is too small for modeling.")
        return
    
    # Prepare unigrams and bigrams for further processing
    unigrams = tokens
    bigrams = list(zip(tokens[:-1], tokens[1:]))
    unigram_counts = Counter(unigrams)
    bigram_counts = Counter(bigrams)

    # Compute MLE probabilities for unigrams and bigrams
    unigram_probs, bigram_probs = uni_bi_gram(unigrams, bigrams)

    # Build the bigram successor table for efficient sampling
    bigram_succ = build_bigram_successors(bigram_probs)

    # Print initial unigram and bigram probabilities
    print_grams(unigram_probs, bigram_probs)

    # Interactive loop for user commands
    while True:
        choice = input("\n1: Reprint probabilities  2: Generate sentence  3: Apply Add-1 smoothing  "
                       "4: Calculate Perplexity on test data  5: Print top 10 unigram and bigram probs 6: Exit\n> ")
        
        # Reprint probabilities
        if choice == '1':
            print_grams(unigram_probs, bigram_probs)
        
        # Generate sentences (if Add-1 Smoothing applied, generate based on new probabilities)
        elif choice == '2':
            sentence = generate_sentence(unigram_probs, bigram_succ)
            print(f"\n{sentence}")
        
        # Applies Add-1 Smoothing
        elif choice == '3':
            uni_sm, bi_sm = add_one_smoothing(unigram_counts, bigram_counts)
            unigram_probs, bigram_probs = uni_sm, bi_sm
            bigram_succ = build_bigram_successors(bigram_probs)
            print("\nAdd-1 smoothing applied.")
            print_grams(unigram_probs, bigram_probs)
        
        # Computes perplexity 
        elif choice == '4':
            test_file_path = input("Enter the test file path: ")
            with open(test_file_path, 'r') as test_file:
                test_text = test_file.read().lower()
            test_tokens = word_tokenizer(test_text)
            
            # Ensure the test tokens are sufficient for perplexity calculation
            if len(test_tokens) < 2:
                print("Test text is too small")
                continue
            
            # Calculate and display the perplexity
            perp = compute_perplexity(test_tokens, bigram_probs)
            if perp == float('inf'):
                print("\nPerplexity: Infinite (zero-prob bigram encountered).")
            else:
                print(f"\nPerplexity: {perp}")
        
        # Print top 10 unigram and bigram probabilities
        elif choice == '5':
            print_top_grams(unigram_probs, bigram_probs)


        elif choice == '6':
            print("\nExiting. Have a good day!")
            break

        else:
            print("Please choose a valid option (1–6).")

if __name__ == "__main__":
    main()
