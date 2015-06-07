import csv
import re
import matplotlib.pyplot as plt
import math


class StatsWriter:
    def __init__(self, csvfile, graphfile, init_counter=15):
        self.csvfile = csvfile
        self.graphfile = graphfile
        self.data = []
        self.init_counter = init_counter
        """ Is able to be populated by stats and then be exported into various formats.
            file is the csvfile saved into
            data is the stats object
        """

    def addrow(self, dictstats, mode):
        """
        adds one dataset stats object as dictionary into the data-array
        """
        dictstats['mode'] = mode
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
                          'abstract_nl_mention_array',
                          'full_nl_mention_array']
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for row in self.data:
                w.writerow(row)

    def makegraph(self):
        """
        make the graph
        """
        # TODO (3) matplotlib create graph
        simple_array = []
        annotate_array = []
        nl_total_ratio_array = []
        abstract_full_ratio_array = []
        re_compiled_param = re.compile(r'_(\d+)$')
        simple_counter = self.init_counter
        total_counter = 0

        for row in self.data:
            total_counter += 1
            # TODO subplots for params
            # TODO plt.axhan or sth like that for highlighting area of inclusive method with param
            nl_total_ratio = row['nl_mention_nr'] / float(row['tot_mention_nr'])
            # abstract = abstract_tokens/tokens in abstract
            # full = full_tokens/tokens in full
            # abstract full ratio = abstract/full


            is_not_ok = row['abstract_tot_token_nr'] == 0 or row['full_tot_token_nr'] == 0 or \
                    row['abstract_nl_token_nr'] == 0 or row['full_nl_token_nr'] == 0
            if is_not_ok:
                abstract_full_ratio = 0
            else:
                abstract_token_ratio = row['abstract_nl_token_nr'] / float(row['abstract_tot_token_nr'])
                full_token_ratio = row['full_nl_token_nr'] / float(row['full_tot_token_nr'])
                abstract_full_ratio = abstract_token_ratio / full_token_ratio

            match = re.search(re_compiled_param, row['mode'])
            if match:
                annotate_array.append(match.group(1))
                # print(row['mode'], match.group(1))
            else:
                annotate_array.append(simple_counter)
                # print(row['mode'], simple_counter)
                simple_counter += 1

            print(nl_total_ratio)
            if nl_total_ratio > 0:
                nl_total_ratio_array.append(nl_total_ratio)
            else:
                nl_total_ratio_array.append(0)

            print(abstract_full_ratio)
            if abstract_full_ratio > 0:
                abstract_full_ratio_array.append(math.log(abstract_full_ratio))
            else:
                abstract_full_ratio_array.append(0)

            simple_array.append(row['nl_mention_nr'])
        plt.plot(annotate_array, nl_total_ratio_array, 'rs', annotate_array, abstract_full_ratio_array, 'bs')
        plt.axis([self.init_counter, self.init_counter + total_counter - 1, 0, 3])
        plt.show()
