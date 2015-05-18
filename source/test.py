# imports
from bs4 import BeautifulSoup
import re
import glob
import json

# constants
# TODO import through parameters
filename = "test.html"
filelist = glob.glob("../IDP4_plain_html/pool/*.plain.html")
tempfile = "test.json"

##############
# interfaces #
##############

# documents[(pubmedid,text,annotation_array)]
documents = []
# annotation_array[(position,length)]
annotation_array = []


#     d888888b d8b   db  .o88b. db      db    db .d8888. d888888b db    db d88888b
#       `88'   888o  88 d8P  Y8 88      88    88 88'  YP   `88'   88    88 88'
#        88    88V8o 88 8P      88      88    88 `8bo.      88    Y8    8P 88ooooo
#        88    88 V8o88 8b      88      88    88   `Y8b.    88    `8b  d8' 88~~~~~
#       .88.   88  V888 Y8b  d8 88booo. 88b  d88 db   8D   .88.    `8bd8'  88.
#     Y888888P VP   V8P  `Y88P' Y88888P ~Y8888P' `8888Y' Y888888P    YP    Y88888P

# abstract class with dictionaries
# required dicts (dict start, end, limits, exclude)
# dicts start (indicatives, strong indicatives)
# dicts end (positions, helping indicatives)
# dict limits (start, stop for spaces and lettres)
# dicts inside (connecting)
# dicts exclude (conventions, common hints for standard mentions)


# simple method
minimum_spaces = 2
maximum_spaces = 5
minimum_lettres = 12
maximum_lettres = 100
indicatives = ["point", "substitution", "deletion",
               "insertion", "mutation", "point mutation"]
positions = ["position", r'^\d+$', "entire gene"]

# pseudocode
# annotations[start, length]
# for each sentence
#   for each word
#       if in indicatives
#           for each next word
#               if max_spaces < cur_spaces !! max_lettres < cur_lettres
#                   break next word for-loop
#               endif
#               if in positions8
#                   save as nl mention in annotations[start, length]
#                   break next word for-loop and continue each word-loop at cur-word
#               endif
#           endloop
#       endif
#   endloop
# endloop


def whole_filelist_test_inclusive(flist):
    for f in flist:
        with open(f, 'r') as f:
            raw = f.read()
            html_doc = raw.replace("\n", "")
            soup = BeautifulSoup(html_doc)
            raw_text = soup.p.string
            sentences = phrasing(raw_text)
            an_array = simple_inclusive(sentences)
            documents.append((soup.html.attrs['data-origid'], raw_text, an_array))
            # print_annotated(raw_text, an_array)
    log_to_file(documents)


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
                            if found[len(found) - 1][0] == itotal:
                                found[len(found) - 1] = (itotal, i - iword + 1)
                            else:
                                found.append((itotal, i - iword + 1))
                        else:
                            found.append((itotal, i - iword + 1))
    return found
# print (words[iword - 1])

# sophisticated
strong_indicatives = []
helping_indicatives = []
# TODO indicatives long list
connecting = ["at", "off", "placed"]  # TODO incomplete connecting list

# TODO Sophisticated method


#     d88888b db    db  .o88b. db      db    db .d8888. d888888b db    db d88888b
#     88'     `8b  d8' d8P  Y8 88      88    88 88'  YP   `88'   88    88 88'
#     88ooooo  `8bd8'  8P      88      88    88 `8bo.      88    Y8    8P 88ooooo
#     88~~~~~  .dPYb.  8b      88      88    88   `Y8b.    88    `8b  d8' 88~~~~~
#     88.     .8P  Y8. Y8b  d8 88booo. 88b  d88 db   8D   .88.    `8bd8'  88.
#     Y88888P YP    YP  `Y88P' Y88888P ~Y8888P' `8888Y' Y888888P    YP    Y88888P
#
#
# minimum_spaces = 2
# minimum_lettres = 12
conventions = ["c.[0-9]+[ACTG]>[ACTG]"]
# list comprehension for "p.Lys76Asn" e.g. [(p.X[0-9]+Y) with X in aa, Y in aa]
# V232fs --> frameshift
# delta Phe581
# Arg-199-->Cys delta
# D3.49(164)
# del/del

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


#     db    db d888888b d888888b db      d888888b d888888b db    db
#     88    88 `~~88~~'   `88'   88        `88'   `~~88~~' `8b  d8'
#     88    88    88       88    88         88       88     `8bd8'
#     88    88    88       88    88         88       88       88
#     88b  d88    88      .88.   88booo.   .88.      88       88
#     ~Y8888P'    YP    Y888888P Y88888P Y888888P    YP       YP
#
#

def print_annotated(raw_text, annotation_array):
    words = raw_text.split(" ")
    for x in annotation_array:
        print "position:", x[0], "with length", x[1]
        # TODO annotation options (map information to stuff)
        print (words[x[0] - 1:x[0] + x[1]])


def regex_array(string, regex_array):
    # searches a string with multiple regex
    # TODO regex tree? search to increase performance if needed @profiling
    for x in regex_array:
        if re.search(x, string):
            return True
    return False


def print_info(pubmedid):
    for x in filelist:
        if pubmedid in x:
            with open(x, "r") as f:
                html_doc = f.read()
                soup = BeautifulSoup(html_doc)
                print "PubMed-ID:", soup.html.attrs['data-origid']
                print "Title:", soup.find(attrs={"data-type": "title"}).h2.string
                print "Abstract:", soup.find(attrs={"data-type": "abstract"}).p.string
            break


def phrasing(text):
    REG_PHRASE_SPLIT = r'\. '
    return text.split(REG_PHRASE_SPLIT)
    # TODO sentences not via ". ", but take care of e.g. "E. coli"


def log_to_file(obj):
    with open(tempfile, 'w') as f:
        f.write(json.dumps(obj))

# operation on simple method currently
# SIMPLE METHOD
# TODO mode select inclusive/exclusive


def print_statistics_documents():
    n = 0.0  # number of documents with found nl mentions
    t = 0.0  # number of found nl mentions
    for x in documents:
        if len(x[2]) > 0:
            n += 1
            t += len(x[2])
    print "statistics\n"
    print "ratios\n"
    print "documents with found mentions"
    print "vs"
    print "total number of documents"
    print "{:.4f}".format(n / len(documents))
    print ""
    print "found mentions"
    print "vs"
    print "total number of documents"
    print "{:.4f}".format(t / len(documents))
    print "\n"


#     .88b  d88.  .d8b.  d888888b d8b   db
#     88'YbdP`88 d8' `8b   `88'   888o  88
#     88  88  88 88ooo88    88    88V8o 88
#     88  88  88 88~~~88    88    88 V8o88
#     88  88  88 88   88   .88.   88  V888
#     YP  YP  YP YP   YP Y888888P VP   V8P
#
#


whole_filelist_test_inclusive(filelist)
print_statistics_documents()
# print_info("127")
# with open(filename, "r") as f:
#     html_doc = f.read().replace("\n", "")
#     soup = BeautifulSoup(html_doc)
# print soup.html.attrs['data-origid']
#     raw_text = soup.p.string
#     sentences = phrasing(raw_text)
#     an_array = simple_inclusive(sentences)
# print_annotated(raw_text, an_array)

    # print(sentences)
