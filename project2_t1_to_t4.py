# CSC 427: NLP
# Project 2 (with T1–T4 and optimized T2)
# Due Apr 22

from collections import Counter
import argparse
import re
import random
import math


def uni_bi_gram(unigrams, bigrams):
    """
    Compute MLE unigram and bigram probabilities.
    Args:
      unigrams (list of str): all tokens in corpus
      bigrams  (list of tuple): list of (w1, w2) pairs
    Returns:
      (dict, dict): (unigram_probs, bigram_probs)
    """
    unigram_counts = Counter(unigrams)
    bigram_counts  = Counter(bigrams)

    total_tokens = sum(unigram_counts.values())
    # MLE unigram: P(w) = C(w)/N
    unigram_probs = {w: c / total_tokens for w, c in unigram_counts.items()}
    # MLE bigram: P(w2|w1) = C(w1,w2)/C(w1)
    bigram_probs = {bg: c / unigram_counts[bg[0]]
                    for bg, c in bigram_counts.items()}
    return unigram_probs, bigram_probs


def add_one_smoothing(unigram_counts, bigram_counts):
    """
    Compute Add-1 smoothed unigram and bigram probabilities.
    Returns (unigram_probs, bigram_probs).
    """
    vocab = set(unigram_counts.keys())
    V = len(vocab)
    total = sum(unigram_counts.values())

    # Smoothed unigram: (C(w)+1)/(N+V)
    sm_unigrams = {w: (unigram_counts[w] + 1) / (total + V)
                   for w in vocab}

    # Smoothed bigram: (C(w1,w2)+1)/(C(w1)+V)
    sm_bigrams = {}
    for w1 in vocab:
        denom = unigram_counts[w1] + V
        for w2 in vocab:
            sm_bigrams[(w1, w2)] = (bigram_counts.get((w1, w2), 0) + 1) / denom
    return sm_unigrams, sm_bigrams


def compute_perplexity(test_tokens, bigram_probs):
    """
    Compute bigram-model perplexity on test_tokens.
    Returns float('inf') if any bigram has zero probability.
    """
    test_bigrams = list(zip(test_tokens[:-1], test_tokens[1:]))
    log_sum = 0.0
    for bg in test_bigrams:
        p = bigram_probs.get(bg, 0.0)
        if p <= 0.0:
            return float('inf')
        log_sum += math.log2(p)
    M = len(test_bigrams)
    return 2 ** (-log_sum / M)


def print_grams(unigram_probs, bigram_probs):
    """Print first five unigram and bigram probabilities."""
    print("\nUnigram Probabilities (first 5):")
    for w, p in list(unigram_probs.items())[:5]:
        print(f"P({w}) = {p}")
    print("\nBigram Probabilities (first 5):")
    for (w1, w2), p in list(bigram_probs.items())[:5]:
        print(f"P({w2}|{w1}) = {p}")


def build_bigram_successors(bigram_probs):
    """
    Build a successor table for O(1) sampling: w1 -> ([w2,...], [p,...])
    """
    succ = {}
    for (w1, w2), p in bigram_probs.items():
        if w1 not in succ:
            succ[w1] = ([], [])
        succ[w1][0].append(w2)
        succ[w1][1].append(p)
    return succ


def generate_sentence(unigram_probs, bigram_succ, sentence_length=15):
    """Generate a sentence using unigram start and O(1) bigram sampling."""
    words, weights = zip(*unigram_probs.items())
    current = random.choices(words, weights=weights)[0]
    sentence = [current]
    for _ in range(sentence_length - 1):
        if current not in bigram_succ:
            break
        next_words, next_weights = bigram_succ[current]
        current = random.choices(next_words, weights=next_weights)[0]
        sentence.append(current)
    return " ".join(sentence)


def main():
    parser = argparse.ArgumentParser(
        description="N-Grams, Language Modeling & Perplexity (Optimized)"
    )
    parser.add_argument("corpus_path", help="Path to training corpus.")
    args = parser.parse_args()

    # Load text and tokenize
    text = open(args.corpus_path, 'r').read().lower()
    tokens = re.findall(r'\b\w+\b', text)
    if len(tokens) < 2:
        print("Corpus too small.")
        return

    # Prepare data
    unigrams = tokens
    bigrams = list(zip(tokens[:-1], tokens[1:]))
    unigram_counts = Counter(unigrams)
    bigram_counts  = Counter(bigrams)

    # Compute MLE probabilities
    unigram_probs, bigram_probs = uni_bi_gram(unigrams, bigrams)

    # Build structures for generation
    bigram_succ = build_bigram_successors(bigram_probs)
    print_grams(unigram_probs, bigram_probs)

    while True:
        choice = input("\n1: Reprint probs  2: Generate  3: Add-1 smooth  "
                       "4: Perplexity on test  5: Quit\n> ")
        if choice == '1':
            print_grams(unigram_probs, bigram_probs)
        elif choice == '2':
            sentence = generate_sentence(unigram_probs, bigram_succ)
            print(f"\n{sentence}")
        elif choice == '3':
            uni_sm, bi_sm = add_one_smoothing(unigram_counts, bigram_counts)
            unigram_probs, bigram_probs = uni_sm, bi_sm
            bigram_succ = build_bigram_successors(bigram_probs)
            print("\nAdd-1 smoothing applied.")
            print_grams(unigram_probs, bigram_probs)
        elif choice == '4':
            path = input("Enter test file path: ")
            test_text = open(path, 'r').read().lower()
            test_tokens = re.findall(r'\b\w+\b', test_text)
            if len(test_tokens) < 2:
                print("Test text too small.")
                continue
            perp = compute_perplexity(test_tokens, bigram_probs)
            if perp == float('inf'):
                print("\nPerplexity: infinite (zero-prob bigram encountered)")
            else:
                print(f"\nPerplexity: {perp}")
        elif choice == '5':
            print("\nExiting. Have a good day!")
            break
        else:
            print("Please enter a valid option (1–5).")

if __name__ == "__main__":
    main()
