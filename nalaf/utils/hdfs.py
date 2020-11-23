from hdfs import InsecureClient, Client
from typing import Callable

import os


def maybe_get_hdfs_client(hdfs_url: str, hdfs_user: str) -> Client:
    if hdfs_url is None:
        return None
    else:
        assert hdfs_user is not None
        return InsecureClient(hdfs_url, user=hdfs_user)


def is_hdfs_directory(hdfs_client: Client, path: str):
    return hdfs_client.status(path)["type"] == "DIRECTORY"


def walk_hdfs_directory(hdfs_client: Client, path: str, accept_filename_fun: Callable[[str], bool]):
    return (os.path.join(dpath, fname) for dpath, _, fnames in hdfs_client.walk(path) for fname in fnames if accept_filename_fun(fname))
