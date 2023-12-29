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
    fdp = atheris.FuzzedDataProvider(input_bytes)
    unicode_str = fdp.ConsumeUnicode(sys.maxsize)
    try:
        parsed_dict = jstyleson.loads(unicode_str)
    except json.JSONDecodeError:
        return

    try:
        NNCFConfig.from_dict(parsed_dict)
    except jsonschema.ValidationError:
        return


atheris.Setup(sys.argv, TestNNCFConfigParsing)
atheris.Fuzz()
