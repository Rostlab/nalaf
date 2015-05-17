from bs4 import BeautifulSoup
import re

filename = "test.html"
# TODO iterate through all files
# TODO randomly select 5-10 documents

# simple method - inclusive
minimum_spaces = 2
maximum_spaces = 5
minimum_lettres = 12
maximum_lettres = 100
indicatives = ["substitution", "deletion",
               "insertion", "mutation", "point mutation"]
# TODO indicatives long list
connecting = ["at", "off", "placed"]  # TODO incomplete connecting list
positions = ["position", r'^\d+$']

# pseudocode simple method
# annotations[start, length]
# for each sentence
#   for each word
#       if in indicatives
#           for each next word
#               if max_spaces < cur_spaces !! max_lettres < cur_lettres
#                   break next word for-loop
#               endif
#               if in positions
#                   save as nl mention in annotations[start, length]
#                   break next word for-loop and continue each word-loop at cur-word
#               endif
#           endloop
#       endif
#   endloop
# endloop


def simple_inclusive(sentences):
    isen = 0
    itotal = 0
    iword = 0
    found = []
    for sentence in sentences:
        isen += 1
        iword = 0
        words = sentence.split(" ")
        for word in words:
            iword += 1
            itotal += 1
            if word in indicatives:
                # print (sentence, isen, iword, itotal)
                for i in xrange(iword - 1, len(words) - 1):
                    pos = i - iword + 1
                    # print iword, '"' + words[i] + '"', i, pos
                    if pos > maximum_spaces:
                        break
                    if regex_array(words[i], positions):
                        # print (words[i], "found")
                        # higher border
                        if len(found) > 0:
                            if found[len(found)-1][0] == itotal:
                                found[len(found)-1] = [itotal, i - iword + 1]
                            else:
                                found.append([itotal, i - iword + 1])
                        else:
                            found.append([itotal, i - iword + 1])
                    elif words[i] == words[len(words) - 1]:
                        print ("not found")
    return found
# print (words[iword - 1])


def print_annotated(raw_text, annotation_array):
    words = raw_text.split(" ")
    for x in annotation_array:
        print (words[x[0] - 1:x[0] + x[1]])


def regex_array(string, regex_array):
    # searches a string with multiple regex
    # TODO regex tree? search to increase performance if needed (profiling)
    for x in regex_array:
        if re.search(x, string):
            return True
    return False

# exclusive
# minimum_spaces = 2
# minimum_lettres = 12
conventions = ["c.[0-9]+[ACTG]>[ACTG]"]
# list comprehension for "p.Lys76Asn" e.g. [(p.X[0-9]+Y) with X in aa, Y in aa]
# V232fs --> frameshift
# delta Phe581
# Arg-199-->Cys delta
# D3.49(164)
# del/del

# TODO pre-processing: strip new lines, just paragraphs

# TODO mapping annotation to text
# documents[pubmedid,text,annotation_array]
# annotation_array[position,length]

# pseudocode exclusive method
# for each annotation in annotation_array
#   if is_annotation(cur-annotation)
#       save as nl mention in annotations[start, length]
#   endif
# endloop
#
# is_annotation(words from start to start+length) { # TODO precise pseudocode
#   count spaces and count lettres
#   check for conventions patterns for each word
# return true if conditions ok
# }


# operation on simple method currently
# SIMPLE METHOD
# TODO mode select inclusive/exclusive
with open(filename, "r") as f:
    html_doc = f.read().replace("\n", "")
    soup = BeautifulSoup(html_doc)
    raw_text = soup.p.string
    sentences = raw_text.split(". ")
    an_array = simple_inclusive(sentences)
    print_annotated(raw_text, an_array)

    # TODO sentences not via ". ", but take care of e.g. "E. coli"
    # print(sentences)
