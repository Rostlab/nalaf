import os
import json


class Cacheable:
    """
    Utility class that would allow subclasses to cache anything they would like to disk.

    They can add or remove anything from the cache by adding or removing elements
    from the self.cache dictionary.

    For every subclass the following will happen:
    1. Cache gets read in at initialization into self.cache
    2. Any manipulation of self.cache is allowed
    3. Cache gets rewritten into a json file on disk

    Separate files (caches) will be create for each subclass
    with the format [Subclass Name]_cache.json

    If subclass implement their own __init__ or __del__ functions then they have to call first
    super().__init__()
    super().__del__()
    """
    def __init__(self):
        self.cache_filename = '{}_cache.json'.format(self.__class__.__name__)
        if os.path.exists(self.cache_filename):
            self.cache = json.load(open(self.cache_filename))
        else:
            self.cache = {}

    def __del__(self):
        if self.cache:
            with open(self.cache_filename, 'w') as file:
                json.dump(self.cache, file)