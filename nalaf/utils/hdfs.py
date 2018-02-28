from hdfs import InsecureClient


def maybe_get_hdfs_client(hdfs_url, hdfs_user):
    if hdfs_url is None:
        return None
    else:
        assert hdfs_user is not None
        return InsecureClient(hdfs_url, user=hdfs_user)


def is_hdfs_directory(hdfs_client, path):
    return hdfs_client.status(path)["type"] == "DIRECTORY"
