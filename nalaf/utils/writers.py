import csv
from itertools import chain
import random
import re
import math
import xml.etree.ElementTree as ET
import json
import sys
import os
from nalaf import print_verbose


class StatsWriter:
    """ Is able to be populated by stats and then be exported into various formats.
            file is the csvfile saved into
            data is the stats object
    """

    def __init__(self, csvfile, graphfile, init_counter=15):
        self.csvfile = csvfile
        self.graphfile = graphfile
        self.data = []
        self.init_counter = init_counter
        """ internal constant being defined """
        self.ylim_max_nl_total = 1
        """ upper ylim for plot y-axis for nl vs total mentions """

    def addrow(self, dictstats, mode):
        """
        adds one dataset stats object as dictionary into the data-array
        """
        dictstats['mode'] = mode

        # generate stub array for representing the True, False nls
        stub_arr = [True] * dictstats['nl_mention_nr'] + [False] * (dictstats['tot_mention_nr'] - dictstats['nl_mention_nr'])
        # stub_arr.extend([False] * (dictstats['tot_mention_nr'] - dictstats['nl_mention_nr']))

        # add xerror
        stv, err = self.calc_dev_error(stub_arr)
        dictstats['error'] = err

        # append to stats-data
        self.data.append(dictstats)

    def writecsv(self):
        """
        write the stats into a csv file
        """
        with open(self.csvfile, "w", encoding='utf-8') as f:
            fieldnames = ['mode',
                          'nl_mention_nr',
                          'tot_mention_nr',
                          'nl_token_nr',
                          'tot_token_nr',
                          'abstract_nl_mention_nr',
                          'abstract_nl_token_nr',
                          'abstract_tot_token_nr',
                          'full_nl_mention_nr',
                          'full_nl_token_nr',
                          'full_tot_token_nr',
                          'nl_mention_array',
                          'abstract_nr',
                          'full_nr',
                          'abstract_nl_mention_array',
                          'full_nl_mention_array']
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for row in self.data:
                w.writerow(row)


    def makegraph(self):
        import matplotlib.pyplot as plt
        """
        make the graph
        """
        # xposition as int and label as string
        x_pos = []
        label = []
        bar_color = []

        # param arrays to be shown in graph
        simple_array = []  # nl absolute nr
        nl_total_ratio_array = []  # nl vs total ratio

        abstract_token_ratio_array = []  # abstract_nl_token / abstract_tot_token
        full_token_ratio_array = []  # full_nl_token / full_tot_token
        abstract_full_ratio_array = []  # abstract_token_ratio / full_token_ratio

        # error
        error_nl_ratio_array = []

        # helping vars
        re_compiled_param = re.compile(r'_(\d+)$')
        simple_counter = self.init_counter
        total_counter = 0

        for row in self.data:
            total_counter += 1

            error_nl_ratio_array.append(row['error'] * 4)  # NOTE 4 might be too few

            nl_total_ratio = row['nl_mention_nr'] / float(row['tot_mention_nr'])
            # abstract = abstract_tokens/tokens in abstract
            # full = full_tokens/tokens in full
            # abstract full ratio = abstract/full

            is_not_ok = row['abstract_tot_token_nr'] == 0 or row['full_tot_token_nr'] == 0 or \
                        row['abstract_nl_token_nr'] == 0 or row['full_nl_token_nr'] == 0

            if is_not_ok:
                abstract_full_ratio = 0
                abstract_token_ratio = 0
                full_token_ratio = 0
            else:
                abstract_token_ratio = row['abstract_nl_token_nr'] / float(row['abstract_tot_token_nr'])
                full_token_ratio = row['full_nl_token_nr'] / float(row['full_tot_token_nr'])
                abstract_full_ratio = abstract_token_ratio / full_token_ratio

            match = re.search(re_compiled_param, row['mode'])
            if match:
                x_pos.append(int(match.group(1)))
                label.append(row['mode'])
                # match.group(1) = min_length param as int
            else:
                x_pos.append(simple_counter)
                label.append(row['mode'])
                simple_counter += 1

            # OPTIONAL could make this param not hardcoded
            if row['mode'] == "Carsten" or row['mode'] == "Inclusive_18":
                bar_color.append('orange')
            else:
                bar_color.append('green')

            # print(nl_total_ratio)
            # nl total ratio
            if nl_total_ratio > 0:
                nl_total_ratio_array.append(nl_total_ratio)
            else:
                nl_total_ratio_array.append(0)

            # print(abstract_full_ratio)
            # abstract vs full ratio
            if abstract_full_ratio > 0:
                abstract_full_ratio_array.append(math.log(abstract_full_ratio))
            else:
                abstract_full_ratio_array.append(0)

            # abstract_token_ratio
            abstract_token_ratio_array.append(abstract_token_ratio)

            # full_token_ratio
            full_token_ratio_array.append(full_token_ratio)

            # # abstract_nl_nr / abstract_tot_token_nr
            # if row['abstract_tot_token_nr'] > 0:
            #     simple_abstract_ratio = row['abstract_nl_mention_nr'] / float(row['abstract_tot_token_nr'])
            # else:
            #     simple_abstract_ratio = 0

            # nl mention nr
            simple_array.append(row['nl_mention_nr'])

        # plot for nl total ratio array
        fig1 = plt.figure()
        fig1.add_axes([0.1, 0.24, 0.88, 0.72])
        plt.bar(x_pos, nl_total_ratio_array, color=bar_color, yerr=error_nl_ratio_array)
        plt.xticks([x + 0.3 for x in x_pos], label, rotation=90)
        plt.ylabel("NL vs Total mentions")
        plt.xlim(min(x_pos), max(x_pos) * 1.05)
        plt.ylim(0, self.ylim_max_nl_total)
        plt.show()

        # subplot for abstract vs full ratio
        # only if the array contains non zeros
        if set(abstract_full_ratio_array) != {0}:
            fig2 = plt.figure()
            fig2.add_axes([0.1, 0.24, 0.88, 0.72])
            plt.bar(x_pos, abstract_full_ratio_array, color=bar_color, yerr=error_nl_ratio_array)
            xticks_pos = list(map(lambda x: x + 0.4, x_pos))
            plt.xticks(xticks_pos, label, rotation=90)
            plt.ylabel("Abstract vs Full documents")
            plt.xlim(min(x_pos) * 0.95, max(x_pos) * 1.05)
            plt.ylim(0, 3)

        plt.show()

    def calc_dev_error(self, total_set):
        sample_size = int(len(total_set)*0.15)
        sample_results = []

        for _ in range(1,1000):
            sample = random.sample(total_set, sample_size)
            sample_results.append(sample.count(True)/sample_size)

        expected_val = total_set.count(True)/len(total_set)
        standard_deviation = sum((x - expected_val)**2 for x in sample_results)/len(sample_results)**(1/2)
        standard_error = standard_deviation/(len(sample_results)-1)**(1/2)
        return standard_deviation, standard_error


class PubTatorFormat:
    """
    Export into PubTatorFormat with or without annotations
    """
    def __init__(self, dataset, location="resources/corpora/idp4/pubtator.txt", no_annotations=True):
        self.location = location
        self.dataset = dataset
        self.no_annotations = no_annotations

    def export(self):
        with open(self.location, 'w', encoding='utf-8') as f:
            for pid, doc in self.dataset.documents.items():
                offset = 0
                f.write("{0}|t|{title}\n".format(pid, title=doc.get_title()))
                f.write("{0}|a|{text}\n".format(pid, text=doc.get_body()))
                for part in doc:
                    for ann in chain(part.annotations, part.predicted_annotations):
                        f.write("{0}\t{start}\t{end}\t{text}\t{classid}\n".format(pid, start=ann.offset,
                                                                                  end=ann.offset + len(ann.text),
                                                                                  text=ann.text, classid=ann.class_id))
                    offset += len(part.text)
                f.write("\n")


class TagTogFormat:
    """
    Ability to Export the dataset as Html + Ann.json database.
    """
    def __init__(self, dataset, use_predicted, to_save_to="resources/corpora/sample/anndoc", who="ml:nalaf", _annjson_folder="annjson", _html_folder="html", use_original_partids = True):
        """
        init function that does prepare annjson folder and html folder
        :param to_save_to: usually resources/corpora/[name of corpus]/anndoc/
        :type dataset: nalaf.structures.data.Dataset
        :param who:
        :param _annjson_folder:
        :param _html_folder:
        :return:
        """
        self.location = to_save_to
        """ root folder, that documents are saved into """
        self.data = dataset
        """ dataset param """
        self.who = who
        """ who parameter """
        self.use_original_partids = use_original_partids
        self.use_predicted = use_predicted

        # Possibility to use instance without writing files to disk
        if to_save_to:
            # check for root folder for files to save to
            if not os.path.isdir(self.location):
                print_verbose("mkdir", os.path.abspath(self.location))
                try:
                    os.makedirs(self.location)
                except FileExistsError:
                    pass

            # create subfolders if not existent
            # annjson folder
            self.annjson_path = os.path.join(self.location, _annjson_folder)
            """ subfolder where ann.json files are saved into """
            if not os.path.isdir(self.annjson_path):
                os.mkdir(self.annjson_path)
            # html folder
            self.html_folder = os.path.join(self.location, _html_folder)
            """ subfolder where html files are saved into """
            if not os.path.isdir(self.html_folder):
                os.mkdir(self.html_folder)

    def export(self, threshold_val):
        self.export_html()
        self.export_ann_json(threshold_val=threshold_val)

    def export_html(self):
        """
        Exporting Html files into folder with each html file being a document itself.
        Html files have sections and everything as if document was exported from tagtog.net itself.
        :return:
        """
        for docid, doc in self.data.documents.items():
            fname = os.path.join(self.html_folder, docid + ".html")

            with open(fname, 'wb') as f:

                # "tag" or "tag_attr" for their attributes

                html_attr = {
                    'id' : docid,
                    'data-origid' : docid,
                    'class' : 'anndoc',
                    'data-anndoc-version' : '2.0',
                    'lang' : '',
                    'xml:lang' : '',
                    'xmlns' : 'http://www.w3.org/1999/xhtml'
                }
                html = ET.Element('html', html_attr)

                head = ET.SubElement(html, 'head')

                # meta1 = ET.SubElement(head, 'meta', { 'charset' : 'UTF-8'} )
                # meta2 = ET.SubElement(head, 'meta', { 'name' : 'generator', 'content' : 'nalaf.utils.writers.TagTogFormat'} )
                # meta3 = ET.SubElement(head, 'meta', { 'name': 'dcterms.source', 'content' : 'http://www.ncbi.nlm.nih.gov/pubmed/' + docid } )  # deprecated maybe different sources

                title = ET.SubElement(head, 'title')
                title.text = docid

                body = ET.SubElement(html, 'body')

                article = ET.SubElement(body, 'article')

                section = ET.SubElement(article, 'section', { 'data-type' : 'article' } )
                div = ET.SubElement(section, 'div', { 'class' : 'content' } )

                for i, (partid, part) in enumerate(doc.parts.items()):
                    if not self.use_original_partids:
                        partid = "s1p{}".format(i + 1)
                    p = ET.SubElement(div, 'p', { 'id' : partid } )
                    p.text = part.text

                # print(ET.dump(html))
                # output = ET.tostring(html, encoding='UTF-8')
                f.write(ET.tostring(html, encoding='utf-8', method='html'))


    def get_single_ann_json(self, threshold_val=None, docid=None):
        if docid:
            doc = self.data.documents[docid]
        else:
            docid, doc = next(iter(self.data.documents.items()))

        partids = list(doc.parts.keys())
        if not self.use_original_partids:
            partids = []
            for i, (partid, part) in enumerate(doc.parts.items()):
                partid = "s1p{}".format(i + 1)
                partids.append(partid)

        # init empty json-object
        json_obj = {
            "annotatable": {
                "parts" : partids
            },
            "anncomplete": False,
            "sources": [
                {
                    "name": "ORIG",
                    "id": docid,
                    "url": ""
                }
                # each entry is a dict with "name", "id", "url"
            ],
            "metas": {
                # nothing important -->  # OPTIONAL add meta information from project
            },
            "entities": [
                # dict with "classId", "part", "offsets" (being [{"start","text"},...], confidence
            ],
            "relations": [
                # not important for here  # OPTIONAL get relations as well
            ]
        }

        for i, (partid, part) in enumerate(doc.parts.items()):
            if not self.use_original_partids:
                partid = "s1p{}".format(i + 1)

            for ann in (part.predicted_annotations if self.use_predicted else part.annotations):
                if threshold_val:
                    if ann.confidence >= threshold_val:
                        state = 'selected'
                    else:
                        state = 'pre-added'
                else:
                    state = ''

                fields = {}
                normalizations = {}

                for key, value in ann.norms.items():
                    if isinstance(value, str):
                        normalizations[key] = {
                            # "source": 'http://uniprot.org',
                            "recName": value,
                            # "confidence": 1  # todo discussion confidence from GNormPlus is not provided so just putting in here 1
                        }
                    else:
                        for substring in value:
                            normalizations[key + '_' + substring[0]] = {
                                # "source": 'http://uniprot.org',
                                "recName": substring,
                                # "confidence": 1  # todo discussion confidence from GNormPlus is not provided so just putting in here 1
                            }

                ent = {
                    "classId": ann.class_id,
                    "part": partid,
                    "offsets": [{"start": ann.offset, "text": ann.text}],
                    "confidence": {
                        "state": state,
                        "who": [
                            self.who
                        ],
                        "prob": ann.confidence
                    },
                    "fields": fields,
                    "normalizations": normalizations
                }
                json_obj['entities'].append(ent)

            for rel in doc.relations():
                ent_string_format = '{}|{},{}'
                ent_string1 = ent_string_format.format(partid, rel.start1, rel.start1 + len(rel.text1))
                ent_string2 = ent_string_format.format(partid, rel.start2, rel.start2 + len(rel.text2))
                relation = {
                    "classId": rel.class_id,
                    "directed": False,
                    "entities": [ent_string1, ent_string2],
                    "confidence": 1
                }
                json_obj['relations'].append(relation)

        return json_obj


    def export_ann_json(self, threshold_val=None):
        """
        Creates all Annotation files in the corresponding ann.json format.
        Description of ann.json-format: "https://github.com/tagtog/tagtog-doc/wiki/ann.json"
        :return:
        """
        for docid in self.data.documents.keys():
            fname = os.path.join(self.annjson_path, docid + ".ann.json")
            print_verbose(fname)
            with open(fname, 'w', encoding='utf-8') as f:
                json_obj = self.get_single_ann_json(threshold_val, docid)
                json.dump(json_obj, f)


class ConsoleWriter:
    """
    Writes the predicted annotations onto the console.
    """
    def __init__(self, ent1_class_id, ent2_class_id, color=True):
        self.color = self.__supports_color() if color else color
        self.protein_color_start = '\033[42m'
        self.mutation_color_start = '\033[41m'
        self.end_color = '\033[0m'
        self.ent1_class_id = ent1_class_id
        self.ent2_class_id = ent2_class_id


    def write(self, dataset):
        """
        :type dataset: nalaf.structures.data.Dataset()
        """
        for doc_id, doc in dataset.documents.items():
            print('DOCUMENT: {}'.format(doc_id))
            for part_id, part in doc.parts.items():
                print('PART {}'.format(part_id))
                self.___print_part(part)

    def ___print_part(self, part):
        if self.color:
            text = part.text
            total_offset = 0
            for ann in sorted(part.predicted_annotations, key=lambda x: x.offset):
                color = self.mutation_color_start if ann.class_id == self.ent2_class_id else self.protein_color_start
                text = text[:ann.offset+total_offset] + color + text[ann.offset+total_offset:]
                total_offset += 5
                text = text[:ann.offset+len(ann.text)+total_offset] + self.end_color + text[ann.offset+len(ann.text)+total_offset:]
                total_offset += 4
            print(text)
            print()
        else:
            padding = len(str(len(part.text)))
            print(part.text)
            print('ANNOTATIONS')
            for ann in sorted(part.predicted_annotations, key=lambda x: x.offset):
                if ann.class_id == self.ent2_class_id:
                    print('Mutation {0: <{pad}} {1: <{pad}} {2}'
                          .format(ann.offset, ann.offset+len(ann.text), ann.text, pad=padding))
                elif ann.class_id == self.ent1_class_id:
                    print('GGP      {0: <{pad}} {1: <{pad}} {2} {3}'
                          .format(ann.offset, ann.offset+len(ann.text), ann.text, ann.norms, pad=padding))
        print('RELATIONS')
        for rel in part.relations:
            print('{} ---> {}'.format(rel.text1, rel.text2))
        print()

    @staticmethod
    def __supports_color():
        """
        Returns True if the running system's terminal supports color, and False
        otherwise.
        """
        plat = sys.platform
        supported_platform = plat != 'Pocket PC' and (plat != 'win32' or 'ANSICON' in os.environ)
        is_a_tty = hasattr(sys.stdout, 'isatty') and sys.stdout.isatty()

        if not supported_platform or not is_a_tty:
            return False
        return True
