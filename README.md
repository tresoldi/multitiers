# multitiers

A library for multi-tiered sequence representation of linguistic data.

## How to use

### Loading data

Data must be a list of dictionaries including, at least, a `DOCULECT`,
a `COGID`, and an `ALIGNMENT` field. Key/column names can be different
from default, and alignment can be either lists or strings (but data must
have been aligned beforehand).

A typical source is the following:

```bash
ID      DOCULECT        PARAMETER       VALUE   IPA     TOKENS  ALIGNMENT       COGID
1       Proto-Germanic  *wīban  *wīban  wiːban  w iː b a n      w iː b ( a n )  538
2       German  *wīban  Weib    vaip    v ai p  v ai b ( - - )  538
3       English *wīban  wife    ʋaɪf    ʋ aɪ f  ʋ aɪ f ( - - )  538
4       Dutch   *wīban  wijf    ʋɛɪf    ʋ ɛɪ f  ʋ ɛɪ f ( - - )  538
5       Proto-Germanic  *kurnan *kurnan kurnan  k u r n a n     k u r n ( a n ) 533
6       German  *kurnan Korn    kɔrn    k ɔ r n k ɔ r n ( - - ) 533
7       English *kurnan corn    kɔːn    k ɔː n  k ɔː - n ( - - )        533
8       Dutch   *kurnan koren   koːrə   k oː r ə        k oː r - ( ə - )        533
9       Proto-Germanic  *xaimaz *xaimaz xaimaz  x ai m a z      x ai m ( a z )  532
```

Data can be loaded and manipulated in any preferred way, including the
simple wrapper provided by the library:

```python
data = multitiers.read_wordlist_data(filename)
```

### Building a multitiers object

A `MultiTiers` object can be initialized with just the data, which will
compute the tiers for the alignments of all doculects and the basic
positional tiers. Syllable structures and suprasegmentals are currently
missing.

```python
mt = multitiers.MultiTiers(data)
```

If shifted alignment tiers will be used, they have to be specified when
building the object. For example, for considering the two following sounds
and the preceding one:

```python
mt = multitiers.MultiTiers(data, left=1, right=2)
```

Other tiers, such as for sound classes and features, are computed on-the-fly
whenever necessary and cached for future reuse within the object.

### Running studies

There are two main types of study currently implemented, correspondence
studies and classifiers. They will serve as building blocks for more elaborate
ones, including for automatic or assisted feature selection.

Correspondence studies require the specification of "known" or "fixed" tiers,
those which are observed, and of "unknown" or "free" ones, those we want
to study the behavior in terms of the observed data. For both types, it
is possible to include and/or exclude values from the tiers, including
individual words (referred by their ID). The study specifications must be
provided in two dictionaries, such as in the following example that
investigates the correspondence of initial (`index=1`) /s/ in Proto-Germanic
into German (excluding the cases where the reflex has /r/, and thus with
German `exclude=["r"]`), using the `cv` class of the following sound (`R1`):

```python
# run a correspondence study
known = {
    "index": {"include": [1]},  # first position in word...
    "Proto-Germanic": {"include": ["s"]},  # when PG has /s/
    "German": {"exclude": ["r"]},  # and G doesn't have /r/
}
unknown = {"Proto-Germanic_cv_R1": {}}

mt.correspondence_study(known, unknown)
```

The computed results are intended for later computer-consumption and might be
a bit harder to interpret at present, but the results show that in all the 78
cases where `Proto-Germanic_cv_R1` is a consonant, the third known tier
(`German`) is `'ʃ'`. When the following sound is a vowel (`V`), in 30 out of 31
cases the `German` tier has `'z'`: a single case has `None`, a gap, which
would need investigation.

```bash
{(1, 's', None): {('V',): 1},
 (1, 's', 'z'): {('V',): 30},
 (1, 's', 'ʃ'): {('C',): 78}}
```

Passing study specifications as a dictionary allows more flexibility, but the
library also provides an auxiliary function for parsing specifications
according to a simple language. The specification can use tabs and
multiple spaces, allows more than one item in includes and excludes (separated
by commas), and takes care of converting strings to integers in the case
of indexes.

```python
demo_study = """
KNOWN index INCLUDE 1
KNOWN Proto-Germanic INCLUDE s
KNOWN German EXCLUDE r
UNKNOWN Proto-Germanic_cv_R1
"""
known2, unknown2 = multitiers.utils.parse_correspondence_study(demo_study)
study_result2 = mt.correspondence_study(known2, unknown2)
```

The second kind of study currently possible are classifications and
predictions using `sklearn`; while currently only decision trees are
implement, nothing forbids using other classifiers or even exporting X/y
matrices for usage in other systems (such as in R or Julia).

## TODO

- Allow shifted alignment tiers at any moment (not only upon loading)
