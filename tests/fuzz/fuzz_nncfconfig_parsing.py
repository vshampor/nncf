import atheris


from nncf import NNCFConfig
import jstyleson
import json
import sys
import jsonschema
from nncf import set_log_level
import logging
set_log_level(logging.CRITICAL)

@atheris.instrument_func
def TestNNCFConfigParsing(input_bytes):
    try:
        unicode_str = input_bytes.decode('utf8')
    except:
        return

    try:
        parsed_dict = jstyleson.loads(unicode_str)
    except json.JSONDecodeError:
        return

    try:
        NNCFConfig.from_dict(parsed_dict)
    except jsonschema.ValidationError:
        return

# def json_mutator(data, max_size, seed):
#     fdp = atheris.FuzzedDataProvider(data)
#     unicode_str = fdp.ConsumeUnicode(sys.maxsize)
#     try:
#         decompressed = jstyleson.loads(unicode_str)
#     except json.JSONDecodeError:
#         parsed_dict = {}
#     else:
#         decompressed = atheris.Mutate(decompressed, len(decompressed))
#     return zlib.compress(decompressed)


atheris.Setup(sys.argv, TestNNCFConfigParsing)
atheris.Fuzz()
