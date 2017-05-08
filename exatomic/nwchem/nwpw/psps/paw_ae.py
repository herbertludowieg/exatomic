# -*- coding: utf-8 -*-
# Copyright (c) 2015-2017, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
"""
PAW Specific AE Output Editors
##############################
These auxiliary output parsers can be used when ``print debug`` is set in a plane
wave NWChem calculation.
"""
import six
from exa.tex import text_value_cleaner
from exa.special import LazyFunction
from exa.core import Meta, Parser, DataFrame
from exatomic.nwchem.nwpw.psps.paw_ps import PAWOutput


class AEOutputMeta(Meta):
    """
    Defines data objects parsed from an output associated with an all-electron
    calculation performed as part of pseudopotential generation.
    """
    info = dict
    data = DataFrame
    grid = LazyFunction
    _descriptions = {'data': "All electron data",
                     'info': "Energy data",
                     'grid': "Grid information"}


class AEOutput(six.with_metaclass(AEOutputMeta, Parser)):
    """
    Parser for debug output files generated by NWChem's on-the-fly pseudopotential
    generation code.

    Note:
        These files are only generated if the user adds ``print debug`` to their
        input file(s).
    """
    description = "Parser for debug output files of the form H_out, C_out, U_out, etc."
    _key_delim0 = "*******"
    _key_delim1 = ":"
    _key_delim2 = "="
    _key_symbol = 5
    _key_charge = 6


    def _parse(self):
        """Identify and process the separate sections of the output file."""
        key0, value0 = str(self[self._key_symbol]).split(self._key_delim1)
        key1, value1 = str(self[self._key_charge]).split(self._key_delim1)
        self.info = {key0.strip(): text_value_cleaner(value0),
                     key1.strip(): text_value_cleaner(value1)}
        sections = self.find(self._key_delim0, text=False)[self._key_delim0]
        self.grid = PAWOutput(self[:sections[0]]).grid
        self.data = self[sections[0]+4:sections[1]-2].to_data(delim_whitespace=True)
        self.data['l'] = self.data['l'].str.upper()
        for line in self[sections[1]:]:
            if self._key_delim2 in line:
                key, value = line.split(self._key_delim2)
                self.info[key.strip()] = text_value_cleaner(value)
