# multitiers

A library for multi-tiered sequence representation of linguistic data.

## TODO

- have utils.get_orders() accept a list of orders (mostly for testing
  purposes) or override the call if the user already provides a list
  of orders (better solution?)
- reimplement utils.sc_mapper() to a generic mapper, in order not use
  CLTS mapper; this implies building the manual conversion from CLTS
  into the generic format
- implement initialization from saved JSON
- modify MultiTiers.as_list() in order to return a matrix suitable for
  numeric analysis (or just have a different function)
- Deal with morphological markers (and others if possible, including
  syllable and lexeme)
- Think about renaming cogid, doculect, etc. to something more general
- Implement vectors of presence/absence
- Implement vectors of phonological features from CLTS
- Review automatic alignments, especially  in Latin2Spanish
- Have a CLDF reading function, aking to wordlist2mt
