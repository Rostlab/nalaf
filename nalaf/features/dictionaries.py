import os
import glob
import traceback
import codecs

from nalaf.features import FeatureGenerator
from nalaf.utils.hdfs import maybe_get_hdfs_client, walk_hdfs_directory
from nalaf import print_verbose, print_debug


class DictionaryFeatureGenerator(FeatureGenerator):

    def __init__(self, name, words_set, case_sensitive=False):
        self.name = name
        self.words_set = words_set
        self.key = "dics." + name
        self.case_sensitive = case_sensitive

    def __repr__(self):
        return "{} (size: {})".format(self.name, len(self.words_set))

    def generate(self, dataset):
        for token in dataset.tokens():
            normalized_token = token if self.case_sensitive else token.word.lower()
            token.features[self.key] = normalized_token in self.words_set

    @staticmethod
    def construct_words_set(file_reader, string_tokenizer, case_sensitive, stop_words):
        """
        Note, the file_reader descriptor is not closed. The client is responsible for this.
        """
        ret = set()
        for name in file_reader:
            tokens = string_tokenizer(name)
            normalized_tokens = tokens if case_sensitive else (x.lower() for x in tokens)
            filtered_normalized_tokens = (x for x in normalized_tokens if ((x not in stop_words) and DictionaryFeatureGenerator.default_stop_rules(x)))

            ret.update(filtered_normalized_tokens)

        return ret

    @staticmethod
    def default_stop_rules(token):
        return len(token) > 1

    @staticmethod
    def __normalize_stop_words(stop_words):
        if stop_words is None:
            return set()
        elif type(stop_words) is str:
            return set(stop_words.split())
        else:
            return stop_words

    @staticmethod
    def __get_filename(path):
        return os.path.splitext(os.path.basename(path))[0]

    @staticmethod
    def __read_dictionaries(dic_paths, read_function, string_tokenizer, case_sensitive, stop_words):
        stop_words = DictionaryFeatureGenerator.__normalize_stop_words(stop_words)

        ret = []

        for dic_path in dic_paths:
            try:
                reader = read_function(dic_path)
                try:
                    name = DictionaryFeatureGenerator.__get_filename(dic_path)
                    words_set = DictionaryFeatureGenerator.construct_words_set(reader, string_tokenizer, case_sensitive, stop_words)
                    generator = DictionaryFeatureGenerator(name, words_set, case_sensitive)
                    ret.append(generator)
                finally:
                    reader.close()
            except Exception as e:
                traceback.print_exc()
                print_debug("Could not read dictionary: {}".format(dic_path), e)
                continue

        print_verbose("Using dictionaries: {}".format(", ".join((repr(x) for x in ret))))

        return ret

    @staticmethod
    def __localfs_read_function(dic_path):
        return open(dic_path, "r", encoding="utf-8")  # closed later

    @staticmethod
    def __hdfs_read_function(hdfs_client):
        def ret(dic_path):
            # here, if we use read(), the connection is closed immediately if not in a "with" context
            # Thus we use _open(), see: https://github.com/mtth/hdfs/blob/2.5.8/hdfs/client.py#L231
            res = hdfs_client._open(dic_path)
            # res.encoding = "utf-8"
            # return res
            return codecs.getreader("utf-8")(res.raw)  # closed later

        return ret


    @staticmethod
    def construct_all_from_paths(dictionaries_paths, string_tokenizer=(lambda x: x.split()), case_sensitive=False, hdfs_url=None, hdfs_user=None, stop_words=None, accepted_extensions=[".tsv", ".csv", ".dic", ".dict", ".txt"]):
        if type(dictionaries_paths) is str:
            dictionaries_paths = dictionaries_paths.split()

        hdfs_client = maybe_get_hdfs_client(hdfs_url, hdfs_user)

        if hdfs_client:
            read_function = DictionaryFeatureGenerator.__hdfs_read_function(hdfs_client)

        else:
            read_function = DictionaryFeatureGenerator.__localfs_read_function

        #

        return DictionaryFeatureGenerator.__read_dictionaries(dictionaries_paths, read_function, string_tokenizer, case_sensitive, stop_words)

    @staticmethod
    def construct_all_from_folder(dictionaries_folder, string_tokenizer=(lambda x: x.split()), case_sensitive=False, hdfs_url=None, hdfs_user=None, stop_words=None, accepted_extensions=[".tsv", ".csv", ".dic", ".dict", ".txt"]):

        def accept_filename_fun(filename: str):
            return any(filename.endswith(accepted_extension) for accepted_extension in accepted_extensions)

        hdfs_client = maybe_get_hdfs_client(hdfs_url, hdfs_user)

        if hdfs_client:
            # hdfs
            dic_paths = walk_hdfs_directory(hdfs_client, dictionaries_folder, accept_filename_fun)
            read_function = DictionaryFeatureGenerator.__hdfs_read_function(hdfs_client)

        else:
            # local file system
            dic_paths = (path for path in glob.glob(os.path.join(dictionaries_folder, "*"), recursive=True) if accept_filename_fun(path))
            read_function = DictionaryFeatureGenerator.__localfs_read_function

        #

        return DictionaryFeatureGenerator.__read_dictionaries(dic_paths, read_function, string_tokenizer, case_sensitive, stop_words)
