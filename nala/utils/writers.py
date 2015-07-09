import csv
import re
import matplotlib.pyplot as plt
import math


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
        # xposition as int and label as string
        x_pos = []
        label = []

        # param arrays to be shown in graph
        simple_array = []  # nl absolute nr
        nl_total_ratio_array = []  # nl vs total ratio

        abstract_token_ratio_array = []  # abstract_nl_token / abstract_tot_token
        full_token_ratio_array = []  # full_nl_token / full_tot_token
        abstract_full_ratio_array = []  # abstract_token_ratio / full_token_ratio

        # helping vars
        re_compiled_param = re.compile(r'_(\d+)$')
        simple_counter = self.init_counter
        total_counter = 0

        for row in self.data:
            total_counter += 1
            # TODO plt.axhan or sth like that for highlighting area of inclusive method with param
            # TODO add standard error
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

            # abstract_nl_nr / abstract_tot_token_nr
            if row['abstract_tot_token_nr'] > 0:
                simple_abstract_ratio = row['abstract_nl_mention_nr'] / float(row['abstract_tot_token_nr'])
            else:
                simple_abstract_ratio = 0

            # nl mention nr
            simple_array.append(row['nl_mention_nr'])

        # subplot for nl total ratio array
        fig1 = plt.figure()
        fig1.add_axes([0.1, 0.22, 0.88, 0.74])
        # plt.subplot(121)
        plt.bar(x_pos, nl_total_ratio_array)
        plt.xticks([x + 0.3 for x in x_pos], label, rotation=90)
        print([x + 0.3 for x in x_pos])
        # fig.ylabel("NL vs Total mention ratio")
        plt.xlim(min(x_pos), max(x_pos) * 1.05)
        plt.show()

        # subplot for abstract vs full ratio
        # only if the array contains non zeros
        if set(abstract_full_ratio_array) != {0}:
            # plt.subplot(122)
            fig2 = plt.figure()
            fig2.add_axes([0.1, 0.22, 0.88, 0.74])
            plt.bar(x_pos, abstract_full_ratio_array)
            plt.xticks(x_pos, label, rotation=90)  # TODO shift position to get correct labeling on bars
            plt.ylabel("Abstract vs Full document ratio")
            plt.xlim(min(x_pos) * 0.95, max(x_pos) * 1.05)
            # plt.ylim(min([x for x in abstract_full_ratio_array if x > 0]) * 0.95, max(abstract_full_ratio_array) * 1.05)
            plt.ylim(1, 3)
            # OPTIONAL define better border for ylim

        # subplot for abstract token ratio
        # plt.subplot(223)
        # plt.bar(x_pos, abstract_token_ratio_array)
        # plt.xticks(x_pos, label, rotation=90)
        # plt.ylabel("Abstract: NL Tokens / Tot Tokens")

        # subplot for full token ratio
        # plt.subplot(224)
        # plt.bar(x_pos, full_token_ratio_array)
        # plt.xticks(x_pos, label, rotation=90)
        # plt.ylabel("Full: Nl Tokens / Tot Tokens")

        # subplot minimum one abstract/full
        # OPTIONAL subplot minimum one abstract/full

        # plt.plot(annotate_array, nl_total_ratio_array, 'rs', annotate_array, abstract_full_ratio_array, 'bs')
        # plt.axis([self.init_counter, self.init_counter + total_counter - 1, 0, 3])

        plt.show()
