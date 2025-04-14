import argparse
import subprocess

# Get the corpus file name as an argument from the command line
parser = argparse.ArgumentParser(description="Enter your corpus file path")
parser.add_argument("corpus_path", help="File path of your corpus.")
args = parser.parse_args()

filename = args.corpus_path

# Run Ken Church's Unix for Poets on the corpus file
try:
    # Define the shell command
    command = f"tr -sc 'A-Za-z' '\n' < {filename} | sort | uniq -c | sort -n -r > unigram_counts.txt"

    # Run the command in shell
    result = subprocess.run(command, shell=True, text=True, capture_output=True)

except Exception as e:
    print(f"Error: {e}")

# Open the unigram_counts.txt file and read line by line. 
# Each line has the form: (count) (word). Example: 3133 the
with open('unigram_counts.txt', 'r') as file:
    total_num_words = 0
    count_word_matrix = []

    for line in file:
        # Strip any leading or trailing whitespace
        line = line.strip()

        # Split line into count and word based on whitespace
        parts = line.split(maxsplit=1)

        # Account for weird cases when the word section is just whitespace
        # For example, line 13283 is just: "1 "
        if len(parts) == 1:
            continue    # If the word is empty, just skip
        else:
            count, word = parts
        
        # Sum the total number of words in the unigram_counts.txt file
        total_num_words = total_num_words + int(count)

        # Add the count and word to matrix  /// FIXME: SHOULD PROBABLY BE DICTIONARY
        count_word_matrix.append([count, word])
    
    print(f"Total # of words: {total_num_words}")
    
    prob = int(count_word_matrix[0][0]) / total_num_words
    print(f"Unigram probability of 'the': {prob}")
