from nalaf.features import FeatureGenerator
from nalaf.utils.hdfs import maybe_get_hdfs_client, walk_hdfs_directory
import os


class DictionaryFeatureGenerator(FeatureGenerator):

    def __init__(self, name, words_set, case_sensitive=False):
        self.name = name
        self.words_set = words_set
        self.key = "dics." + name
        self.case_sensitive = case_sensitive

    def generate(self, dataset):
        for token in dataset.tokens():
            normalized_token = token if self.case_sensitive else token.lower()
            token.features[self.key] = normalized_token in self.words_set

    @staticmethod
    def construct_words_set(file_reader, tokenizer, case_sensitive):
        """
        Note, the file_reader descriptor is not closed. The client is responsible for this.
        """
        ret = set()
        for name in file_reader:
            tokens = tokenizer.tokenize_string(name)
            normalized_tokens = tokens if case_sensitive else (x.lower() for x in tokens)
            ret.update(normalized_tokens)

        return ret

    @staticmethod
    def construct_all_from_folder(tokenizer, case_sensitive, dictionaries_folder, hdfs_url, hdfs_user, accepted_extensions=[".dic", "dict", ".txt", ".tsv", ".csv"]):

        def accept_filename_fun(filename):
            return any(filename.endswith(accepted_extension) for accepted_extension in accepted_extensions)

        def get_filename(path):
            return os.path.splitext(os.path.basename(path))[0]

        def read_dictionaries(dic_paths, read_function):
            ret = []

            for dic_path in dic_paths:
                reader = read_function(dic_path)
                try:
                    name = get_filename(dic_path)
                    words_set = DictionaryFeatureGenerator.construct_words_set(reader, tokenizer, case_sensitive)
                    generator = DictionaryFeatureGenerator(name, words_set, case_sensitive)
                    ret.append(generator)
                finally:
                    reader.close()

            return ret

        #

        hdfs_client = maybe_get_hdfs_client(hdfs_url, hdfs_user)

        if hdfs_client:
            # hdfs
            dic_paths = walk_hdfs_directory(hdfs_client, dictionaries_folder, accept_filename_fun)
            read_function = lambda dic_path: hdfs_client.read(dic_path)

        else:
            # local file system
            dic_paths = (for path in glob.glob(str(dictionaries_folder), recursive=True) if accept_filename_fun(path))
            read_function = lambda dic_path: open(dic_path, "f")

        #

        return read_dictionaries(dic_paths, read_function)
