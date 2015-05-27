# imports
from bs4 import BeautifulSoup
import re
import glob
import json

# constants
# TODO import through parameters
filename = "test.html"
tempfile = "test.json"
statsfile = "stats.json"

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
    """ easy_predictor on the whole dataset. """
    for f in flist:
        with open(f, 'rb') as f:
            soup = BeautifulSoup(f)
            raw_parts = soup.find_all(id=re.compile('^s'))
            raw_text = [part.string for part in raw_parts].join(' ')

            sentences = phrasing(raw_text)
            an_array = easy_predictor(sentences)
            pubmedid = soup.html.attrs['data-origid']
            documents.append((pubmedid, raw_text, an_array))

            # OPTIONAL abstracts vs documents

    log_to_file(documents)


def easy_predictor(sentences):
    """ Very simple state machine algorithm that does de-novo prediction of nl mutation mentions
    by using dictionaries in conjunction with patterns.
    """
    # TODO Export into other file. (20)
    isen = 0
    itotal = 0
    iword = 0
    found = []

    # iterate over sentences
    for sentence in sentences:
        isen += 1
        iword = 0
        words = sentence.split(" ")

        # iterate over words in sentence
        for word in words:
            iword += 1
            itotal += 1

            # check for word in dict
            if word in indicatives:
                for i in range(iword - 1, len(words) - 1):
                    pos = i - iword + 1
                    if pos > maximum_spaces:
                        break

                    # check for regexs
                    if regex_array(words[i], positions):

                        # filter max_spaces
                        if len(found) > 0:
                            if found[len(found) - 1][0] == itotal:
                                found[len(found) - 1] = (itotal, i - iword + 1)
                            else:
                                found.append((itotal, i - iword + 1))
                        else:
                            found.append((itotal, i - iword + 1))
            # cleaner idea:
            # status "object" (can be simple vars)
            # that contains information about next, previous, current word like:
            # - length
            # - capital letter
            # - current nr of spaces
            # - current nr of lettres
            # - ... [attributes that can be filtered for]
            # iterator that gets next word
            # filter that checks for attributes from current parameters (settings of filters)

    return found
# print (words[iword - 1])

# sophisticated
strong_indicatives = []
helping_indicatives = []
# OPTIONAL use statistics from nl mentions of annotated dataset to find important words

connecting = ["at", "off", "placed"]  # TODO incomplete connecting list (30)

# TODO Sophisticated method (30)

#     d88888b db    db  .o88b. db      db    db .d8888. d888888b db    db d88888b
#     88'     `8b  d8' d8P  Y8 88      88    88 88'  YP   `88'   88    88 88'
#     88ooooo  `8bd8'  8P      88      88    88 `8bo.      88    Y8    8P 88ooooo
#     88~~~~~  .dPYb.  8b      88      88    88   `Y8b.    88    `8b  d8' 88~~~~~
#     88.     .8P  Y8. Y8b  d8 88booo. 88b  d88 db   8D   .88.    `8bd8'  88.
#     Y88888P YP    YP  `Y88P' Y88888P ~Y8888P' `8888Y' Y888888P    YP    Y88888P

# minimum_spaces = 2
# minimum_lettres = 12
# TODO complete conventions according to HGVS and set of regexs by tmVar (3)
conventions = [r'c.[0-9]+[ACTG]>[ACTG]', r'p.[A-Z][a-z]{2}\d+[A-Z][a-z]{2}',
               r'[A-Z][a-z]{2}-->[A-Z][a-z]{2}']
# list comprehension for "p.Lys76Asn" e.g. [(p.X[0-9]+Y) with X in aa, Y in aa]
# V232fs --> frameshift
# delta Phe581
# Arg-199-->Cys delta
# D3.49(164)
# del/del


# Ankit's Algorithm
def ankit_algorithm():
    """ Ankit s version of the algorithm.
    Obsolete, because of the general_algorithm which incorporates Ankit's algorithm. """
    total_mentions = 0
    nl_mentions = 0

    # for each document
    for pubmedid, doc in documents.items():

        if has_annotations(doc):

            # for each part
            for part_id, part in doc.items():
                if len(part['annotations']) > 0:
                    for annotation in part['annotations']:

                        # filter attributes
                        if len(annotation['text'].split(" ")) > 2 and len(annotation['text']) > 24:
                            print(annotation['text'])
                            nl_mentions += 1
                        total_mentions += 1

    print("nlmentions:", nl_mentions, "Total", total_mentions,
          "nl/total:", nl_mentions / total_mentions)


def general_algorithm(is_export=False, is_exclusive=False,
                      minimum_spaces=None, minimum_lettres=None,
                      maximum_spaces=None,
                      maximum_lettres=None, indicatives_array=None,
                      connecting_array=None, positions_array=None,
                      conventions_array=None, is_export_mentions=False):
    # parameters
    total_mentions = 0
    nl_mentions = 0
    docs_nlmentions = 0
    docs_total = len(documents.keys())
    nl_mentions_string = []  # nl mentions saved in string array for later examination
    st_mentions_string = []  # st mentions saved in string array for later examination
    docs_nlmentions_status = False
    full_document = 0
    abstract_document = 0

    # for each document
    for pubmedid, doc in documents.items():

        if has_annotations(doc):
            docs_nlmentions_status = False
            # for each part
            for part_id, part in doc.items():

                # check for no annotations
                if len(part['annotations']) == 0:
                    next

                for annotation in part['annotations']:
                    # extractable attributes
                    text = annotation['text']
                    current_lettres = len(text)
                    text_array = text.split(" ")
                    current_spaces = len(text_array) - 1

                    # in case params are not defined
                    cond_max_spaces = True
                    cond_min_spaces = True
                    cond_min_lettres = True
                    cond_max_lettres = True

                    # TODO convention filtering (3)
                    cond_conventions = True

                    # filter attributes
                    # spaces/wordcount
                    if minimum_spaces is not None:
                        cond_min_spaces = (current_spaces >= minimum_spaces)
                    if maximum_spaces is not None:
                        cond_max_spaces = (current_spaces <= maximum_spaces)

                    # lettres
                    if maximum_lettres is not None:
                        cond_max_lettres = (current_lettres <= maximum_lettres)
                    if minimum_lettres is not None:
                        cond_min_lettres = (current_lettres >= minimum_lettres)

                    # convention filtering
                    if conventions_array is not None:
                        for word in text_array:
                            if regex_array(word, conventions_array):
                                if is_exclusive:
                                    cond_conventions = True
                                else:
                                    cond_conventions = False

                    # combine filters
                    cond_spaces = cond_min_spaces and cond_max_spaces
                    cond_lettres = cond_min_lettres and cond_max_lettres

                    if is_exclusive:
                        cond_all = not(cond_spaces and cond_lettres and cond_conventions)
                    else:
                        cond_all = cond_spaces and cond_lettres and cond_conventions

                    if text == 'Ser280D':
                        print(cond_all, cond_lettres, cond_spaces, cond_conventions)

                    # if all filters satisfy, then is nl mention
                    if cond_all:
                        # print(annotation['text'])
                        nl_mentions += 1
                        if not docs_nlmentions_status:
                            docs_nlmentions += 1
                            if is_full_document(doc):
                                full_document += 1
                            else:
                                abstract_document += 1
                            docs_nlmentions_status = True
                        # nl mention saved in array
                        nl_mentions_string.append(text)
                    else:
                        # st mention saved in array
                        st_mentions_string.append(text)
                    total_mentions += 1

                    # inclusive: all conditions that satisfy to be a nl mention
                    #   data
                    # exclusive: everything is nl mention that is not standard mention
                    #               means: all conditions for standard mention
    # Params print
    print("Params:")
    print("minimum_spaces:", minimum_spaces, "| minimum_lettres:", minimum_lettres)
    print("maximum_lettres:", maximum_lettres, "| maximum_spaces:", maximum_spaces)
    print("export?:", is_export)
    if conventions is not None:
        print("conventions:", " | ".join(conventions))

    # Stats calculation
    nl_total_ratio = nl_mentions / total_mentions
    docs_total_ratio = docs_nlmentions / docs_total
    abstract_full_ratio = abstract_document / full_document
    export_dict = {
        'abs_nl_mention': nl_mentions,
        'abs_total_mention': total_mentions,
        'ratio_nl_to_total': nl_total_ratio,
        'docs_with_nl_mentions': docs_nlmentions,
        'ratio_docs_with_nl_mentions_to_total': docs_total_ratio,
        'ratio_abstract_full': abstract_full_ratio,
        'abs_abstract': abstract_document,
        'abs_full': full_document,
        'docs_total': docs_total,
        'min_space': minimum_spaces,
        'max_space': maximum_spaces,
        'min_lettres': minimum_lettres,
        'max_lettres': maximum_lettres,
        'exclusive': is_exclusive
    }

    if is_export_mentions:
        export_dict['mentions'] = nl_mentions_string

    # Stats print
    print("Stats:")
    print("nlmentions:", nl_mentions, "| Total:", total_mentions,
          "| nl/total:", nl_total_ratio,
          "\nDocs with min #1:", docs_nlmentions,
          "| DocsNL vs DocsTotal:", docs_total_ratio,
          "\nAbstract vs Full:", abstract_full_ratio,
          "| Abs abstract:", abstract_document, "| Abs full:", full_document)

    # Print most interesting
    print(json.dumps(get_interesting_mentions(nl_mentions_string), indent=4, sort_keys=True))

    print("--------------------------------------------------------------")

    if is_export:
        return export_dict

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
# is_annotation(words from start to start+length) { # TODO precise pseudocode (15)
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


def stats_run():
    """ Does one whole run with various parameters.
    Saves those values and exports them in a CVS-file.
    """
    # inclusive
    min_space_start = 2
    min_lettres_start = 12
    min_space_end = 6
    min_lettres_end = 40

    for min_space in range(min_space_start, min_space_end):
        for min_lettres in range(min_lettres_start, min_lettres_end):
            stats.append(general_algorithm(minimum_spaces=min_space, minimum_lettres=min_lettres,
                                           is_exclusive=False, is_export=True))

    # exclusive
    max_space_start = 1
    max_lettres_start = 6
    max_space_end = 4
    max_lettres_end = 28

    for max_space in range(max_space_start, max_space_end):
        for max_lettres in range(max_lettres_start, max_lettres_end):
            stats.append(general_algorithm(maximum_spaces=max_space, maximum_lettres=max_lettres,
                                           conventions_array=conventions, is_exclusive=True,
                                           is_export=True))

    with open(statsfile, 'w') as f:
        f.write(json.dumps(stats, indent=4, sort_keys=True))


def get_interesting_mentions(mention_array):
    top_words = {}
    min_word = "------------------------------------------------------------------------------------------------------------------------------------------------"
    max_word = ""
    for mention in mention_array:
        if mention in top_words:
            top_words[mention] += 1
        else:
            top_words[mention] = 1

        if len(min_word) > len(mention):
            min_word = mention
        if len(max_word) < len(mention):
            max_word = mention

    # TODO (3) get top 5 words

    return_dict = {
        'min_word': min_word,
        'max_word': max_word,
    }

    return return_dict


def test_json_import(fname):
    """ Test method.
    ann.json print of entities
    """
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
    """ Confirms whether provided object of
    annotation array in ann.json is e_2 (mutation mention).
    """
    for key, value in entity.items():
        if key == "classId" and value == "e_2":
            return True
    return False


def print_annotated(raw_text, annotation_array):
    """ Obsolete (only meant for easy_predictor.) """
    words = raw_text.split(" ")
    for x in annotation_array:
        print("position:", x[0], "with length", x[1])
        print(words[x[0] - 1:x[0] + x[1]])


def regex_array(string, regex_array):
    """
        Search through a string for a match with multiple regexs'
        t = string
        S = regexs in []
    """
    # OPTIONAL regex tree? search to increase performance if needed @profiling
    for regex in regex_array:
        if re.search(regex, string):
            return True
    return False


def print_info(pubmedid):
    """ Print Information about provided pubmedid. """
    if len(documents) > 0:
        if pubmedid in documents:
            doc = documents[pubmedid]
            # print("Title:", '"' + doc['s1h1']['text'] + '"')
            list_doc = sorted(doc.items(), key=lambda c: c[0])
            for part in list_doc:
                print(part[1]['text'])
        else:
            print("not found")
    # OPTIONAL options for extended report/info


def phrasing(text):
    """ Very easy sentence splitter.
        Can be extended.
    """
    REG_PHRASE_SPLIT = r'\. '
    return text.split(REG_PHRASE_SPLIT)
    # OPTIONAL sophisticated sentence splitting defined through REG_PHRASE_SPLIT var


def log_to_file(obj):
    """
    Write object to logfile in JSON format.
    Logfile is defined through: tempfile.
    """
    with open(tempfile, 'w') as f:
        f.write(json.dumps(obj))


def is_full_document(doc):
    for part in doc:
        if re.search("^s[3-9]", part):
            return True
    return False


def import_json_to_db():
    """ Import ann.json files to documents object. """
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
    """
    Import the raw html files imported from tagtog.net.
    Format is with parts that have part-ids e.g. "s1s2" or "s1h3".
    """
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


def has_annotations(doc):
    """ Check if document has any mutation mention saved. """
    for part in doc.values():
        if len(part['annotations']) > 0:
            return True
    return False
# OPTIONAL: mode select inclusive/exclusive


def print_statistics_documents():
    """ Obsolete (only meant for documents easy_predictor) """
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

# FIXME Add __main__ function so this can be imported as module
# FIXME rename
# FIXME replace to other folder

import_html_to_db()
import_json_to_db()

return_obj = general_algorithm(minimum_lettres=28, minimum_spaces=4,
                               is_export=True, is_export_mentions=True)

with open('mentions_list_ankit.json', 'w') as f:
    f.write(json.dumps(return_obj, indent=2, sort_keys=True))

# stats_run()

# whole_filelist_test_inclusive(filelist)
# print_statistics_documents()

# test_json_import("17327381")
# test_doc = list(documents.items())

# print(type(test_doc))
# print(json.dumps(test_doc, indent=4))
# print_info("17327381")
# ankit_algorithm()
# print("ä")
# for min_l in range(12, 36):
#     general_algorithm(2, minimum_lettres=min_l)
#     general_algorithm(3, minimum_lettres=min_l)
#     general_algorithm(4, minimum_lettres=min_l)
# check_db_integrity()
# print(json.dumps(list(documents.items())[0:1], indent=4))
# print(documents)
