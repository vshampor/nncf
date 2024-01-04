from hypothesis import given
from hypothesis_jsonschema import from_schema

from nncf import NNCFConfig
import jstyleson
import json
import sys
import jsonschema
from nncf import set_log_level
import logging

from nncf.config.schema import NNCF_CONFIG_SCHEMA

set_log_level(logging.CRITICAL)

@given(from_schema(NNCF_CONFIG_SCHEMA))
def test_nncfconfig_parses_schema_conforming_config(parsed_dict):
    _ = NNCFConfig.from_dict(parsed_dict)
    pass