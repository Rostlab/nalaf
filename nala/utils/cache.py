import os
import json
from nala import print_verbose


class Cacheable:
    """
    Utility class that would allow subclasses to cache anything they would like to disk.

    They can add or remove anything from the cache by adding or removing elements
    from the self.cache dictionary.

    For every subclass the following will happen:
    1. Cache gets read in at initialization into self.cache
    2. Any manipulation of self.cache is allowed
    3. Cache gets rewritten into a json file on disk

    Separate files (caches) will be created for each subclass
    with the format [Subclass Name]_cache.json

    If you want to manipulate the cache:
        You have to use the subclass with a context manager like so:
        with SomeSubclass() as instance:
            instance.do_something()
    Otherwise for a regular call like so:
        instance = SomeSubclass()
        instance.do_something()
    the cache will not be read/written.

    Subclasses that implement __init__ have to call super().__init__() first.
    """
    def __init__(self):
        self.cache = {}

    def __enter__(self):
        self.cache_filename = '{}_cache.json'.format(self.__class__.__name__)
        print_verbose('reading from cache {}'.format(self.cache_filename))
        if os.path.exists(self.cache_filename):
            self.cache = json.load(open(self.cache_filename))
        else:
            self.cache = {}
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.cache:
            print_verbose('writing the cache {}'.format(self.cache_filename))
            with open(self.cache_filename, 'w') as file:
                json.dump(self.cache, file)