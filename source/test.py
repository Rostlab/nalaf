# imports
from bs4 import BeautifulSoup
import re
import glob
import json

# constants
# TODO import through parameters
filename = "test.html"
tempfile = "test.json"

resources = "../resources/"
filelist = glob.glob(resources + "IDP4_plain_html/pool/*.plain.html")
jsonlist = glob.glob(resources + "IDP4_members_json/pool/aboj*/*.ann.json")

##############
# interfaces #
##############

# documents[(pubmedid,text,annotation_array)]
# documents = { pubmedid : { part_id : { text: raw_text, annotations : annotation_array } } }
documents = {}
# annotation_array = { start : start_char_part, end : end_char_part, prob : pred_or_1}
annotation_array = []
# stats = { iteration_id : [(stat, value)]}
stats = []

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
        with open(f, 'rb') as f:
            # raw = f.read()
            # html_doc = raw.replace("\n", "")
            soup = BeautifulSoup(f)
            raw_text = soup.p.string  # FIXME check for whole document
            # TODO just abstracts or whole documents?
            # TODO opt. parameter for whole documents
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
                for i in range(iword - 1, len(words) - 1):
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



# Ankit's Algorithm
def ankit_algorithm():
    total_mentions = 0
    nl_mentions = 0
    docs_nlmentions = 0
    for pubmedid, doc in documents.items():
        if has_annotations(doc):
            for part_id, part in doc.items():
                if len(part['annotations']) > 0:
                    for annotation in part['annotations']:
                        # FILTER
                        if len(annotation['text'].split(" ")) > 2 and len(annotation['text']) > 24:
                            print(annotation['text'])
                            nl_mentions += 1
                        total_mentions += 1
    print("nlmentions:", nl_mentions, "Total", total_mentions, "nl/total:", nl_mentions/total_mentions)


# Finally come up with:
# * [ ] #NL / #Total Number
# * [ ] #NL
# * [ ] Ratio of docs that have at least 1
# * [ ] Abstract vs. Full-Text


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

#######################################
# legend for ann.json files
# e_1 = protein (entity)
# e_2 = mutation (entity)
# e_3 = organism (entity)
# m_4 = figstabs_with_mutations (meta)
# r_5 = e_1|e_2 (relation)
# r_6 = e_1|e_3 (relation)
#######################################
test_db = {
    "PMC123322": {
        "s2s1p1": {
            "text": "raw text of paragraph",
            "annotations": [
                {
                    "start": 113,
                    "text": "blabal",
                    "end": 119,
                    "prob": 1
                },
                {
                    "start": 113,
                    "text": "blabal",
                    "end": 119,
                    "prob": 1
                }
            ],
        },
    },
    "123323": {
        "s2s1p1": {
            "text": "raw text of paragraph",
            "annotations": [
                {
                    "start": 113,
                    "text": "blabal",
                    "end": 119,
                    "prob": 1
                },
                {
                    "start": 113,
                    "text": "blabal",
                    "end": 119,
                    "prob": 1
                }
            ],
        },
    },
}


def test_json_import(fname):
    for x in jsonlist:
        if fname in x:
            with open(x, 'r') as f:
                raw_text = f.read()
                json_object = json.loads(raw_text)
                entities = json_object['entities']

                for ent in entities:
                    if is_mutation_entity(ent):
                        print(ent)


def is_mutation_entity(entity):
    for key, value in entity.items():
        if key == "classId" and value == "e_2":
            return True
    return False


def print_annotated(raw_text, annotation_array):
    words = raw_text.split(" ")
    for x in annotation_array:
        print("position:", x[0], "with length", x[1])
    # TODO annotation options (map information to stuff)
        print(words[x[0] - 1:x[0] + x[1]])


def regex_array(string, regex_array):
    # searches a string with multiple regex
    # TODO regex tree? search to increase performance if needed @profiling
    for x in regex_array:
        if re.search(x, string):
            return True
    return False


def print_info(pubmedid):
    if len(documents) > 0:
        if pubmedid in documents:
            doc = documents[pubmedid]
            # print("Title:", '"' + doc['s1h1']['text'] + '"')
            list_doc = sorted(doc.items(), key=lambda c: c[0])
            for part in list_doc:
                print(part[1]['text'])
            # for key, part in doc.items():
            # print(part)
            #     print(part['text'])
        else:
            print("not found")
    # else:
    #     for x in filelist:
    #         if pubmedid in x:
    #             with open(x, "r") as f:
    #                 html_doc = f.read()
    #                 soup = BeautifulSoup(html_doc)
    #                 print("PubMed-ID:", soup.html.attrs['data-origid'])
    #                 print("Title:", soup.find(attrs={"data-type": "title"}).h2.string)
    #                 abstract_full = ""

    # abstract_part = soup.find(attrs={"data-type": "abstract"})
    # print abstract_part.find_all("p")
    #                 abstract_parts = soup.find_all("p", id=re.compile("^s2"))
    #                 for tag in abstract_parts:
    #                     print(tag['id'])
    #                     abstract_full += "\n" + tag.string


def phrasing(text):
    REG_PHRASE_SPLIT = r'\. '
    return text.split(REG_PHRASE_SPLIT)
    # TODO sentences not via ". ", but take care of e.g. "E. coli"


def log_to_file(obj):
    with open(tempfile, 'w') as f:
        f.write(json.dumps(obj))


def import_json_to_db():
    for j in jsonlist:
        with open(j, 'r') as f:
            json_object = json.loads(f.read())
            pubmedid = json_object['sources'][0]['id']
            doc = documents[pubmedid]
            entities = json_object['entities']
            for entity in entities:
                if entity['classId'] == 'e_2':
                    an_array = doc[entity['part']]['annotations']
                    if an_array is None:
                        print("entity"['part'])
                    start_char_part = entity['offsets'][0]['start']
                    text = entity['offsets'][0]['text']
                    end_char_part = start_char_part + len(text)
                    an_array.append(
                        {'start': start_char_part, 'end': end_char_part, 'prob': 1, 'text': text})
                else:
                    next


def import_html_to_db():
    for x in filelist:
        with open(x, "rb") as f:
            doc = {}
            counter = 0
            soup = BeautifulSoup(f)
            pubmedid = soup.html.attrs['data-origid']
            # print("PubMed-ID:", pubmedid)
            # title = soup.find(attrs={"data-type": "title"}).h2.string
            # print "Title:", title

            # basic information input
            # doc['title'] = title

            # abstract_part = soup.find(attrs={"data-type": "abstract"})
            # print abstract_part.find_all("p")
            abstract_parts = soup.find_all(id=re.compile("^s"))
            for tag in abstract_parts:
                counter += 1
                doc[tag['id']] = {'text': tag.string, 'annotations': [], 'counter': counter}
            documents[str(pubmedid)] = doc


def check_db_integrity():
    """
    Check documents-object for offsets annotations.
    Logs some information if Errors exist.
    """
    for pubmedid, doc in documents.items():
        if has_annotations(doc):
            # print("----------------------")
            # print("whole document")
            # print(json.dumps(doc, indent=4, sort_keys=True))
            # iterate through parts with annotation
            for part_id in doc:
                if len(doc[part_id]['annotations']) > 0:
                    part = doc[part_id]
                    # print("--------")
                    # print("--------")
                    # print(part_id)
                    for annotation in sorted(part['annotations'], key=lambda ann: ann['start']):
                        # print("-------")
                        # print(annotation['text'])
                        orig_string = annotation['text']
                        start = annotation['start']
                        end = annotation['end']
                        cut_string = part['text'][start:end]
                        # print(part['text'][annotation['start']:annotation['end']])
                        if orig_string != cut_string:
                            print("ID:", pubmedid, ", part_id:", part_id)
                            print("org_string:", orig_string)
                            print("cut_string:", cut_string)
                            print("")
                            print("cut_start:", start)
                            print("cut_end:  ", end)
                            print("TEXT\n", part['text'])
                            print("------\n\n")
                            print("WHOLE DOCUMENT:")
                            print(json.dumps(doc, indent=4, sort_keys=True))

            # return


def has_annotations(doc):
    for part in doc.values():
        if len(part['annotations']) > 0:
            # print("----------------------")
            # print("part with annotation")
            # print(json.dumps(part, indent=4))
            return True
    return False
# operation on simple method currently
# SIMPLE METHOD

# OPTIONAL: mode select inclusive/exclusive


def print_statistics_documents():
    n = 0.0  # number of documents with found nl mentions
    t = 0.0  # number of found nl mentions
    for x in documents:
        if len(x[2]) > 0:
            n += 1
            t += len(x[2])
    print("statistics\n")
    print("ratios\n")
    print("documents with found mentions")
    print("vs")
    print("total number of documents")
    print("{:.4f}".format(n / len(documents)))
    print("")
    print("found mentions")
    print("vs")
    print("total number of documents")
    print("{:.4f}".format(t / len(documents)))
    print("\n")


#     .88b  d88.  .d8b.  d888888b d8b   db
#     88'YbdP`88 d8' `8b   `88'   888o  88
#     88  88  88 88ooo88    88    88V8o 88
#     88  88  88 88~~~88    88    88 V8o88
#     88  88  88 88   88   .88.   88  V888
#     YP  YP  YP YP   YP Y888888P VP   V8P


# whole_filelist_test_inclusive(filelist)
# print_statistics_documents()

# test_json_import("17327381")
import_html_to_db()
# test_doc = list(documents.items())

# print(type(test_doc))
# print(json.dumps(test_doc, indent=4))
# print_info("17327381")
import_json_to_db()
# ankit_algorithm()
# check_db_integrity()
print(json.dumps(list(documents.items())[0:1], indent=4))
# print(documents)

# print_info("127")
# with open(filename, "r") as f:
#     html_doc = f.read().replace("\n", "")
#     soup = BeautifulSoup(html_doc)
# print soup.html.attrs['data-origid']
#     raw_text = soup.p.string
#     sentences = phrasing(raw_text)
#     an_array = simple_inclusive(sentences)
# print_annotated(raw_text, an_array)
