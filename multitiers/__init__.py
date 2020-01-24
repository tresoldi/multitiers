# __init__.py

"""
__init__ file for the MultiTiers liberary.
"""

# Version and other package information
__version__ = "0.1"
__author__ = "Tiago Tresoldi"
__email__ = "tresoldi@shh.mpg.de"

# Build the namespace
from multitiers.multitiers import MultiTiers
from multitiers.utils import clts_object, read_wordlist_data
