#!/bin/env python3
import json
import glob
import re
from bs4 import BeautifulSoup
import sys
from nalaf.utils import MUT_CLASS_ID


def main(html_path='', ann_path=''):
    documents = {}

    # get files from folders
    filelist = glob.glob(html_path + "/*.plain.html")
    jsonlist = glob.glob(ann_path + "/*.ann.json")

    # import html files
    import_html_to_db(documents, filelist)
    # import json files
    import_json_to_db(documents, jsonlist)
    # check database
    check_db_integrity(documents)


def import_html_to_db(documents, filelist):
    """
    Import the raw html files imported from tagtog.net.
    Format is with parts that have part-ids e.g. "s1s2" or "s1h3".
    """
    for x in filelist:
        with open(x, "rb") as f:
            doc = {}
            counter = 0
            soup = BeautifulSoup(f, "html.parser")
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


def import_json_to_db(documents, jsonlist):
    """ Import ann.json files to documents object. """
    for j in jsonlist:
        with open(j, 'r', encoding='utf-8') as f:
            json_object = json.loads(f.read())
            pubmedid = json_object['sources'][0]['id']
            doc = documents[pubmedid]
            entities = json_object['entities']
            for entity in entities:
                if entity['classId'] == MUT_CLASS_ID:
                    an_array = doc[entity['part']]['annotations']
                    if an_array is None:
                        print("entity"['part'])
                    start_char_part = entity['offsets'][0]['start']
                    text = entity['offsets'][0]['text']
                    end_char_part = start_char_part + len(text)
                    an_array.append(
                        {'start': start_char_part, 'end': end_char_part, 'prob': 1, 'text': text})


def check_db_integrity(documents):
    """
    Check documents-object for offsets annotations.
    Logs some information if Errors exist.
    """
    counter = 0
    wrong_pmid_list = []
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
                            if pubmedid not in wrong_pmid_list:
                                wrong_pmid_list.append(pubmedid)
                            counter += 1
                            print("ID:", pubmedid, ", part_id:", part_id)
                            print("org_string:", orig_string)
                            print("cut_string:", cut_string)
                            print("")
                            print("cut_start:", start)
                            print("cut_end:  ", end)
                            print("TEXT\n", part['text'])
                            print("------\n")

    if counter > 0:
        print("\n\n")
        print("WHOLE DOCUMENTS:")
        for pubmedid in wrong_pmid_list:
            print("PubMed-ID:", pubmedid + "\n")
            print(json.dumps(documents[pubmedid], indent=4, sort_keys=True))
            print("\n")
    else:
        print("Integrity at 100 % !")


def has_annotations(doc):
    """ Check if document has any mutation mention saved. """
    for part in doc.values():
        if len(part['annotations']) > 0:
            return True
    return False


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
