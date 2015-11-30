def find_current_git_ref():
    """
    Tries to find the current version of the project we are working on under git.
    If successful returns the hash of the latest commit.
    """
    import os
    import nalaf
    import re
    git_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(nalaf.__file__))), '.git')
    if os.path.exists(git_dir):
        if os.path.exists(os.path.join(git_dir, 'HEAD')):
            with open(os.path.join(git_dir, 'HEAD')) as file:
                ref = re.search('ref: ?(.*)', file.read())
                current_ref = os.path.join(git_dir, ref.group(1))
                if os.path.exists(current_ref):
                    with open(current_ref) as ref_file:
                        return ref_file.read()
    return None
