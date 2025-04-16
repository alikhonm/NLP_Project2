# CSC 427: NLP
# Project 2 (with T3: Add-1 smoothing & T4: Perplexity)
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
    # Count occurrences
    unigram_counts = Counter(unigrams)
    bigram_counts  = Counter(bigrams)

    # Total tokens
    total_num_tokens = sum(unigram_counts.values())

    # MLE unigram probabilities
    unigram_probs = {
        w: count / total_num_tokens
        for w, count in unigram_counts.items()
    }

    # MLE bigram probabilities:  P(w₂|w₁) = C(w₁,w₂) / C(w₁)
    bigram_probs = {
        bg: count / unigram_counts[bg[0]]
        for bg, count in bigram_counts.items()
    }

    return unigram_probs, bigram_probs


def print_grams(unigram_probs, bigram_probs):
    """Print first five unigram and bigram probabilities."""
    print("\nUnigram Probabilities (first 5):")
    for w, p in list(unigram_probs.items())[:5]:
        print(f"P({w}) = {p}")

    print("\nBigram Probabilities (first 5):")
    for (w1, w2), p in list(bigram_probs.items())[:5]:
        print(f"P({w2}|{w1}) = {p}")


def generate_sentence(bigram_model, sentence_length=15):
    """Generate a sentence of up to `sentence_length` words using the bigram model."""
    # Pick a random start word
    start_words = [w1 for (w1, _) in bigram_model.keys()]
    current = random.choice(start_words)
    sentence = [current]

    for _ in range(sentence_length - 1):
        # Collect successors and weights
        candidates = [
            (w2, prob)
            for (w1, w2), prob in bigram_model.items()
            if w1 == current
        ]
        if not candidates:
            break
        words, weights = zip(*candidates)
        current = random.choices(words, weights=weights)[0]
        sentence.append(current)

    return " ".join(sentence)


def add_one_smoothing(unigram_counts, bigram_counts):
    """
    Compute Add-1 smoothed unigram and bigram probabilities.
    Returns (unigram_probs, bigram_probs).
    """
    vocab = set(unigram_counts.keys())
    V = len(vocab)
    total = sum(unigram_counts.values())

    # Smoothed unigram: (count+1) / (N + V)
    sm_unigrams = {
        w: (unigram_counts[w] + 1) / (total + V)
        for w in vocab
    }

    # Smoothed bigram: for *all* pairs
    sm_bigrams = {}
    for w1 in vocab:
        denom = unigram_counts[w1] + V
        for w2 in vocab:
            sm_bigrams[(w1, w2)] = (
                (bigram_counts.get((w1, w2), 0) + 1) / denom
            )
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
    return 2 ** (- log_sum / M)


def main():
    parser = argparse.ArgumentParser(
        description="N-Grams, Language Modeling, Perplexity (T1–T4)"
    )
    parser.add_argument("corpus_path", help="Path to training corpus.")
    args = parser.parse_args()

    # Load & tokenize
    text = open(args.corpus_path, 'r').read().lower()
    tokens = re.findall(r'\b\w+\b', text)
    if len(tokens) < 2:
        print("Corpus too small.")
        return

    # Prepare counts & initial MLE model
    unigrams = tokens
    bigrams = list(zip(tokens[:-1], tokens[1:]))
    unigram_counts = Counter(unigrams)
    bigram_counts  = Counter(bigrams)
    vocab = set(unigram_counts.keys())

    unigram_probs, bigram_probs = uni_bi_gram(unigrams, bigrams)
    print_grams(unigram_probs, bigram_probs)

    # Interactive menu
    while True:
        choice = input("\n1: Reprint probs  2: Generate  3: Add-1 smooth  "
                       "4: Perplexity on test  5: Quit\n> ")
        if choice == '1':
            print_grams(unigram_probs, bigram_probs)
        elif choice == '2':
            sent = generate_sentence(bigram_probs)
            print(f"\n{sent}")
        elif choice == '3':
            # T3: apply Add-1 smoothing
            uni_sm, bi_sm = add_one_smoothing(unigram_counts, bigram_counts)
            unigram_probs, bigram_probs = uni_sm, bi_sm
            print("\nAdd-1 smoothing applied.")
            print_grams(unigram_probs, bigram_probs)
        elif choice == '4':
            # T4: compute perplexity
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
