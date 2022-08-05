import os
import re
from Post import Posting
from bs4 import BeautifulSoup
import json
from collections import defaultdict
from nltk.stem import PorterStemmer
from math import log
from ast import literal_eval as make_tuple


def getBatch(file_count, all_file_list):
    batch = []  # List of Batch Files
    mb_sum = 0  # Current count of Megabytes
    for path_to_file in all_file_list:
        file_size = os.path.getsize(path_to_file) * 10**-6  # Get size of current file in Megabytes
        if mb_sum + file_size <= 1024:  # Checks if current file can be added to batch
            batch.append(path_to_file)  # Adds file path to batch list
            mb_sum += file_size  # Increments current Megabyte size by file just appended to batch
            file_count += 1  # Increment point to next file = new current file
        else:  # files list is not empty yet, but exceeds batch maximum
            return batch, file_count
    # files list is empty and batch maximum is not exceeded
    return batch, file_count


def buildIndex():
    # {Token : List of Posting Objects}, this will be reset after each partial index is written.
    hash_table = dict()

    # {token: doc count} Document Frequency Dictionary
    token_count = defaultdict(int)

    # {Doc ID: URL}
    aux_dict = dict()

    # Doc ID initialized to 0.
    doc_id = 0

    # Measure the size of indices
    total_kb = 0

    # Count for which partial index is to be created
    partial_index_count = 1

    # Path to Directory that holds sub-directories (URL Domains)
    current_dir = os.path.join(str(os.getcwd()), "DEV")

    # All file list similar to files, but contains path to the file
    all_file_list = list()
    for root, dirs, files in os.walk(current_dir):
        for file in files:
            all_file_list.append(os.path.join(root, file))

    # Count of which document we are on, used for slicing the all file list
    file_count = 0

    # Create the next batch of files to be indexed
    while file_count < len(all_file_list):
        # Slice the all file list to continue to the next document after a batch has been completed
        new_list = all_file_list[file_count:]

        # Get the batch of files to index return in a tuple of ([list of files], updated file count)
        result = getBatch(file_count, new_list)

        # Assign batch to the list of file paths
        batch = result[0]

        # Update file count to last file seen
        file_count = result[1]

        # For each file in the batch, tokenize it and index them
        for path in batch:
            # Create a document id to assign to each file.
            doc_id += 1
            print(doc_id)

            # Read json file and load its corresponding data such as the url and html content
            f = open(path)
            data = json.load(f)

            url = data["url"]
            content = data["content"]

            # Dictionary to remember url to document id
            aux_dict[doc_id] = url
            # Catch any exceptions that occur while parsing the html content
            try:
                # Use BeautifulSoup to parse the html content of the current document
                soup = BeautifulSoup(content, features="html.parser")

                # Get the text as strip and remove newlines/begin tokenization process
                text = soup.get_text().lower().strip().rstrip()

                # Use PorterStemmer to fix stemming of words, words with different tenses will become the same
                ps = PorterStemmer()

                # Tokenize the text string split alphanumeric characters.
                tokens = [ps.stem(x.lower()) for x in re.split(r'[^a-zA-Z]+', text) if len(ps.stem(x.lower())) >= 1]

                # Initialize a term frequency dictionary
                tokens_dict = defaultdict(int)

                # Begin count for term frequency of how many times a single word appears in a document
                for t in tokens:
                    tokens_dict[t] += 1

                # Remove the duplicates, to reduce times to loop over entire tokens list for a document
                tokens = list(set(tokens))

                # Update the Document Frequency of a term, if a term appears in a document update its df by 1
                for tok in tokens:
                    token_count[tok] += 1

                # Begin adding tokens to the Hash Table with its corresponding Posting
                for j in range(len(tokens)):
                    # Grab next available token to be entered into Hash Table
                    t = tokens[j]

                    # If the token is not in the Hash Table, create its key and append to its value list its Posting
                    if t not in hash_table:
                        # Initialize {token: [empty list]}
                        hash_table[t] = list()

                        # create new Posting object
                        p = Posting(doc_id, tokens_dict[t])

                        # append Posting object to list for token
                        hash_table[t].append(p)

                    # If the token is in the Hash Table, create a posting object for it and append it to the Hash Table
                    elif t in hash_table:
                        p = Posting(doc_id, tokens_dict[t])
                        hash_table[t].append(p)

                # Free memory of the count for term frequency for that specific document
                tokens_dict.clear()
            except Exception:
                pass
        # Sort Hash Table by Key (Token)
        hash_table = dict(sorted(hash_table.items(), key=lambda x: x[0]))

        # Create and open a partial index file to write the Hash Table content acquired from the documents in batch
        text_file = f"partialIndex{partial_index_count}.txt"
        f = open(text_file, "w")

        # Begin looping over tokens and its list of postings to write to the partial index
        for key, values in hash_table.items():
            # Write the Token
            f.write(str(key) + ": ")

            # Loop through that tokens list of postings
            for i in range(len(values)):
                # For all elements that are not the last one in values, write to file in this format
                if i != len(values) - 1:
                    f.write("(" + str(values[i].getDocId()) + "," + str(values[i].gettf()) + "), ")

                # For the element that is the last one in values, write to file in this format
                else:
                    f.write("(" + str(values[i].getDocId()) + "," + str(values[i].gettf()) + ")")

            # Once a list of postings for a token has been written, end that line by writing a newline
            f.write("\n")

        # Close the partial index that has just been written to
        f.close()

        # Free memory of the {token: [list of postings]} Hash Table
        hash_table.clear()

        # Increment Partial Index Count to signify which next partial index is to be created.
        partial_index_count += 1

        # Add the size of partial index just written to Total Kilobytes
        total_kb += os.path.getsize(os.path.join(os.getcwd(), text_file)) * 10**-3

    # Return (Total Files, Total Kilobytes of Partial Indices, Dictionary containing the document frequency of terms)
    return file_count, total_kb, token_count, aux_dict


def final_index(file_count, token_count):
    f = ["partialIndex1.txt", "partialIndex2.txt", "partialIndex3.txt"]
    output = open("output.txt", "w")

    # Open all partial index files simultaneously
    with open("partialIndex1.txt", "r") as f1, open("partialIndex2.txt", "r") as f2, open("partialIndex3.txt", "r") as f3:
        # Get ONE line for each text file
        line1 = f1.readline()
        line2 = f2.readline()
        line3 = f3.readline()
        while line1 != "" and line2 != "" and line3 != "":  # When all files reach EOF, then STOP
            minimum = min(line1, line2, line3)  # Get LINE with minimum token among lines retrieved --> minimum

            if minimum.split(":")[0] == line1.split(":")[0]:
                try:
                    output.write(line1)
                    line1 = f1.readline()
                except EOFError:
                    pass
            elif minimum.split(":")[0] == line2.split(":")[0]:
                try:
                    output.write(line2)
                    line2 = f2.readline()
                except EOFError:
                    pass
            else:
                try:
                    output.write(line3)
                    line3 = f3.readline()
                except EOFError:
                    pass
        output.close()

    # Open and read file to be rewritten as the final inverted index
    outputcurr = open("output.txt", "r")

    # Open and write to a file to be the final inverted index
    final = open("final.txt", "w")

    # Previous line, none if no line has been written to final inverted index
    prev = None

    # Current line to be checked to previous if previous is not none and written to final inverted index
    curr = outputcurr.readline().rstrip().split(":")

    # Bool variable for when the end of file from Output.txt has been met
    checker = True

    # Begin Iteration
    while checker is not False:
        # End case when we reach EOF
        if len(curr) == 1:
            checker = False
        # When not EOF
        else:
            # Skip line where the token is ""
            if curr[0] != "":
                # If previous is none, write the current line to final inverted index
                if prev is None:
                    # Turn the string of posting tuples to a list of tuples
                    post_list = [list(make_tuple(x.strip())) for x in curr[1].rstrip().split(", ")]

                    # Write the token to final inverted index
                    final.write("{}:".format(curr[0]))

                    # Loop until the end of the list of tuples and write each tuple to final inverted index
                    # and replace tf with tf-idf score
                    i = 0
                    while i < len(post_list):
                        tf = post_list[i][1]
                        df = token_count[curr[0]]
                        post_list[i][1] = round((1 + log(tf, 10)) * log(file_count / df, 10), 3)
                        post = str(tuple(post_list[i])).strip().replace(" ", "")
                        if i < len(post_list) - 1:
                            final.write("{}, ".format(post))
                        else:
                            final.write("{}".format(post))
                        i += 1
                    # Set previous line to what current line is
                    prev = curr

                    # Move to next line and set current to that
                    curr = outputcurr.readline().rstrip().split(":")
                # If previous is not none, compare previous and current line to check for token similarity
                else:
                    # if current token and previous token are the same
                    if curr[0] == prev[0]:
                        # Turn the string of posting tuples to a list of tuples
                        post_list = [list(make_tuple(x.strip())) for x in curr[1].rstrip().split(", ")]
                        final.write(", ")
                        i = 0

                        # Loop until the end of the list of tuples and write each tuple to final inverted index
                        # and replace tf with tf-idf score
                        while i < len(post_list):
                            tf = post_list[i][1]
                            df = token_count[curr[0]]
                            post_list[i][1] = round((1 + log(tf, 10)) * log(file_count / df, 10), 3)
                            post = str(tuple(post_list[i])).strip().replace(" ", "")
                            if i < len(post_list) - 1:
                                final.write("{}, ".format(post))
                            else:
                                final.write("{}".format(post))
                            i += 1
                        # Set previous line to what current line is
                        prev = curr

                        # Move to next line and set current to that
                        curr = outputcurr.readline().rstrip().split(":")
                    # if current token and previous token are not the same
                    else:
                        # Write newline to know to begin a new token
                        final.write("\n")

                        # Turn the string of posting tuples to a list of tuples
                        post_list = [list(make_tuple(x.strip())) for x in curr[1].rstrip().split(", ")]

                        # Write key to final inverted index
                        final.write("{}:".format(curr[0]))
                        i = 0
                        while i < len(post_list):
                            tf = post_list[i][1]
                            df = token_count[curr[0]]
                            post_list[i][1] = round((1 + log(tf, 10)) * log(file_count / df, 10), 3)
                            post = str(tuple(post_list[i])).strip().replace(" ", "")
                            if i < len(post_list) - 1:
                                final.write("{}, ".format(post))
                            else:
                                final.write("{}".format(post))
                            i += 1
                        # Set previous line to what current line is
                        prev = curr

                        # Move to next line and set current to that
                        curr = outputcurr.readline().rstrip().split(":")
            # If current token == ""
            else:
                # Keep previous line none
                prev = None

                # Move to next line and set current to that
                curr = outputcurr.readline().rstrip().split(":")

    # Close the files that have been read and written to
    outputcurr.close()
    final.close()
    return None


if __name__ == "__main__":
    # Begin Building Partial Indices
    file_count, total_kb, token_count, aux_dict = buildIndex()

    # Mini Report for M1:
    f = open("Report.txt", "w")
    f.write("Index Documents: " + str(file_count) + '\n')
    f.write(f"Total Kilobytes of Partial Indexes based on n = {file_count} documents: " + str(total_kb) + '\n')
    f.close()

    # Function to combine partial indices and create one final Inverted Index
    final_index(file_count, token_count)

    # Write aux-dict
    f = open("Url.txt", "w")
    for key, values in aux_dict.items():
        f.write("Document ID: {}, URL: {}\n".format(key, values))
