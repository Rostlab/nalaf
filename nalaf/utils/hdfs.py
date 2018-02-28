from hdfs import InsecureClient

import os


def maybe_get_hdfs_client(hdfs_url, hdfs_user):
    if hdfs_url is None:
        return None
    else:
        assert hdfs_user is not None
        return InsecureClient(hdfs_url, user=hdfs_user)


def is_hdfs_directory(hdfs_client, path):
    return hdfs_client.status(path)["type"] == "DIRECTORY"


def walk_hdfs_directory(hdfs_client, path, accept_filename_fun):
    return (os.path.join(dpath, fname) for dpath, _, fnames in hdfs_client.walk(path) for fname in fnames if accept_filename_fun(fname))
