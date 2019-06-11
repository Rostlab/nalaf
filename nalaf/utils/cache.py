import os
import json
from nalaf import print_verbose
import time


class Cacheable:
    """
    Utility class that allows subclasses to cache results to disk.

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
    They can also set the max time for the cache to live.
    """
    # 604800 = 7 days in seconds
    def __init__(self, max_time_in_seconds=604800):
        self.cache = {}
        self.max_time_in_seconds = max_time_in_seconds
        self.is_timed = True

    def __enter__(self):
        self.cache_directory = os.path.join(os.path.expanduser('~'), '.nalaf')
        self.cache_filename = '{}_cache.json'.format(os.path.join(self.cache_directory, self.__class__.__name__))
        if os.path.exists(self.cache_filename):

            # if the file is too old reset the cache
            if self.is_timed and (time.time() - os.path.getctime(self.cache_filename)) > self.max_time_in_seconds:
                print_verbose('resetting the cache {}'.format(self.cache_filename))
                os.remove(self.cache_filename)
                self.cache = {}
            else:
                print_verbose('reading from cache {}'.format(self.cache_filename))
                with open(self.cache_filename) as f:
                    self.cache = json.load(f)
        else:
            print_verbose('no cache found {}'.format(self.cache_filename))
            self.cache = {}
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.cache:
            print_verbose('writing the cache {}'.format(self.cache_filename))
            if not os.path.exists(self.cache_directory):
                os.makedirs(self.cache_directory)
            with open(self.cache_filename, 'w') as file:
                json.dump(self.cache, file)
