import csv


class StatsWriter:
    def __init__(self, csvfile, graphfile):
        self.csvfile = csvfile
        self.graphfile = graphfile
        self.data = []
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
        pass