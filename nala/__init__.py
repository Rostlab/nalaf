import configparser
import pkg_resources
import os

config = configparser.ConfigParser()

# if there is a local config.ini file use that
if os.path.exists('config.ini'):
    config.read('config.ini')
# otherwise use the default config.ini file
else:
    config.read(pkg_resources.resource_filename('nala.data', 'default_config.ini'))

print_verbose = print if config.getboolean('print', 'verbose', fallback=False) else lambda *a, **k: None
print_debug = print if config.getboolean('print', 'debug', fallback=False) else lambda *a, **k: None
