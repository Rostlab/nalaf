from bs4 import BeautifulSoup
filename = "test.html"
# TODO iterate through all files
# TODO randomly select 5-10 documents

# simple method - inclusive
minimum_spaces = 2
maximum_spaces = 5
minimum_lettres = 12
maximum_lettres = 100
indicatives = ["substitution", "deletion", "insertion", "mutation", "point mutation"]
# TODO indicatives long list
# TODO connecting long list
positions = ["position", "[0-9]+"]

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
    sentences = soup.p.string.split(". ")

    # TODO sentences not via ". ", but take care of e.g. "E. coli"
    print(sentences)
