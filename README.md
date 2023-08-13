# MultiTiers

Multitiered alignment is a formalism for the representation of diachronically
related lexical data. It builds upon the concept of "alignment", usually
segments or sound classes, allowing a wide range of information to be
considered, mostly of phonological or phonetic nature. Facilitating the
manipulation and exploration of such data, multitiers are organized as
common dataframes, that is, sets of parallel data series that follow the
relational model of data.

This library is an experimentation of such formalism, building upon
frequently used Python libraries such as `pandas`, `sklearn`, and
`numpy`, aligning the manipulation of linguistic data with the same tools
and discussions used for machine learning in general, where linguistic data
has mostly been confined to Natural Language Processing.

The starting point of tiers are collection of entries that include, at
least, a unique form ID (usually but not necessarily numeric), a
doculect identifier (usually a language name, but "doculect" is used to
guarantee reproducibility in terms of sources), a cognate identifier
(again, usually but not necessarily numeric), and an alignment, built as
a sequence of segments (usually IPA graphemes), potentially with gaps and
morpheme marks. The number of elements in alignments must be the same for
all cognate IDs; the `multitier` library does not perform alignment, for
which different tools such as `lingpy` can be used. Note that it is not
necessary for each cognate_id to be expressed in all doculects.

  | ID  | DOCULECT       | COGNATE_ID | ALIGNMENT    |
  |-----|----------------|------------|--------------|
  | 1   | Proto-Germanic | 538        | w iː b a n   |
  | 2   | German         | 538        | v ai b - -   |
  | 3   | English        | 538        | ʋ aɪ f - -   |
  | 4   | Dutch          | 538        | ʋ ɛɪ f - -   |
  | 5   | Proto-Germanic | 533        | k u r n a n  |
  | 6   | German         | 533        | k ɔ r n - -  |
  | 7   | English        | 533        | k ɔː - n - - |
  | 8   | Dutch          | 533        | k oː r - ə - |
  | ... | ...            | ...        | ...          |

[explain internally how it works]

## How to use

The library is loaded as expected

```
import multitiers
```

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

Other tiers, such as for sound classes and features (including shifted ones),
are computed on-the-fly
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
would need investigation. More complex studies could consider other tiers,
and they are not limited to a single unknown tier: we can, for example,
investigate tuples of correspondences between different doculects, such as
the relation of German/Dutch against English.

```bash
{(1, 's', None): {('V',): 1},
 (1, 's', 'z'): {('V',): 30},
 (1, 's', 'ʃ'): {('C',): 78}}
```

As said, these results are mostly intended for computer-consumption: their
immediate usage is for proposing sound changes (or, in a more appropriate way,
correspondence laws, as they would allow to involve multiple languages and
tiers in the output). Restricting to traditional `a > b / c _ d`
notation, the output above could be mapped to a pair of changes
`s > z / _ V` and `s > ʃ / _ C`, but our focus is in correspondence
descriptors and not properly in sound changes (also considering how the
traditional notation would not be adequate for a multitiered system
with potentially multiple correspondences). Note that a system for
searching for correspondence laws will require different pruning strategies
not yet implemented.

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
known2, unknown2 = multitiers.utils.parse_study(demo_study)
study_result2 = mt.correspondence_study(known2, unknown2)
```

The second kind of study currently possible are classifications and
predictions using `sklearn`; while currently only decision trees are
implement, nothing forbids using other classifiers or even exporting X/y
matrices for usage in other systems (such as in R or Julia).

Classifiers are used through a `Classifier` object that will build a
multitier collection internally, and accept the same kind of data.

```python
clf = multitiers.Classifier(data)
```

Classifiers need to be trained with a tier specification like the one
for correspondence studies, as in the following example. The parser for
specifying studies in a single string can be used here as well, but it is
recommended to follow the nomenclature of machine learning studies and
call the observed data `X_tiers` and the target ones `y_tiers`.

```python
# Study
X_tiers = {
    "index": {"include": [1]},  # first position in word...
    "Proto-Germanic": {"include": ["s"]},  # when PG has /s/
    "Proto-Germanic_cv_R1" : {}, # any following class
}
y_tiers = {
    "German": {"exclude": ["r"]},  # and G doesn't have /r/
}
clf.train(X_tiers, y_tiers)
```

After training, the results can be exported to a graph via `graphviz`. Using
the same example of above, we can visualize the decision tree for the
development of initial Proto-Germanic \*s in German:

```python
clf.to_graphviz("docs/germanic.png")
```

![germanic](http://www.tiagotresoldi.com/images/tiers/germanic.png)

The visualization of the decision tree confirms what we observed in the
other experiment. At the first, top node, we are informed that the most
likely value for the `German` tier, before any restriction (i.e., in
this case, when `index==1` and `Proto-Germanic=s`), is `'ʃ'`. The Gini
index is 0.401, indicating that the cases with a different sound are
numerous enough. The first test (and, in this case, the only one) is at
the top the node: it is decided by `Proto-Germanic_cv_R1` (that is, the CV
class of the following sound in Proto-Germanic, one position to the right)
being `'V'`. If True, we move to the right of the tree, and find out that
in this case all observed tiers (`gini = 0.0`) are `'z'`. If the test
fails (that is, the value is not `'V'`), all cases are `'ʃ'`.

More complex examples can involve, as in the case of the correspondence
studies, multiple tiers and restrictions both in `X` and `y`. For example,
we can use our small language to investigate the same context as above,
but this time considering both the German and English reflexes and the
class of the Proto-Germanic sounds according to the more complex SCA
model. For exploration, we limit the tree depth to three nodes.

```python
clf2 = multitiers.Classifier(data)

study = """
X_tier index INCLUDE 1
X_tier Proto-Germanic INCLUDE s
X_tier Proto-Germanic_sca_L1
X_tier Proto-Germanic_sca_R1
y_tier German EXCLUDE r
y_tier English
"""

X_tiers2, y_tiers2 = multitiers.utils.parse_study(study)
clf2.train(X_tiers2, y_tiers2, max_depth=3)
clf2.to_graphviz("docs/germanic2")
```

![germanic2](http://www.tiagotresoldi.com/images/tiers/germanic2.png)

From the graph, we are informed that the most common correspondence set
between the `German` and `English` tiers (again, with the study restrictions,
in initial position with an aligned \*s in Proto-Germanic) are respectively
`'ʃ'` and `'s'`. The first test is on whether the following Proto-Germanic
sound (`R1`) is of class `K` (dorsal plosives and affricates), which, if true,
indicates that all German and English correspondences have `'ʃ'`. If the
test fails, the most informative test is whether the following sound is
of class `T`, when the `'ʃ/s'` is again found in all samples. The following
test is similar, and ends (as we limited the expansion of the decision tree)
with a Gini index of 0.457, for cases where the most likely correspondence
is with /z/ in German and /s/ in English. Note that this is the most likely
*correspondence*, not (necessarily) the most frequent reflex for each
language.

It is possible to investigate many other correspondences. We can for example
test how much we can predict of Dutch vowels given German and English reflexes,
combining information from different languages, with no reconstructions
involved. To limit the tree for easiness in exploration, we only
perform a split if it will decrease the impurity by at least 0.03333.

```python
clf3 = multitiers.Classifier(data)

study3 = """
X_tier German
X_tier English
X_tier Dutch_cv INCLUDE V
y_tier Dutch
"""

X_tiers3, y_tiers3 = multitiers.utils.parse_study(study3)
clf3.train(X_tiers3, y_tiers3, min_impurity_decrease=0.0333)
clf3.to_graphviz("docs/dutch_pred")
```

![dutch_pred](http://www.tiagotresoldi.com/images/tiers/dutch_pred.png)

The visualization shows that the most common Dutch vowel is /ə/, which
will be correct almost all the time (Gini index of 0.015) if the
corresponding German vowel is /ə/ is as well. If the German vowel is /a/,
we expect /ɑ/ in Dutch (but note the high impurity), if it is /aː/ we
can expect a corresponding /aː/, and so on. Note that, in this tree,
the algorithm didn't pick
any tier from English: at least with the restrictions we set, the German
vowel is always more informative. However, if we build a tree without
restrictions, it will end up using English for cases when our data has no
German/Dutch cognates: it is behavior is not unlike a linguist
having to use a more distant language for reconstruction when the closest
is missing the cognate which is needed.

At last, the classifier, even if currently limited to decision trees, can
be used to predict features or bundle of features. It can also report the
confidence on each prediction, coming from the score of each class. If, for
example, we train on a more complex model for predicting all Dutch consonants,
we can verify how well it matches the actual data.

```python
clf4 = multitiers.Classifier(data)

study4 = """
X_tier German
X_tier English
X_tier German_sca
X_tier German_sca_L1
X_tier German_sca_R1
X_tier English_sca
X_tier English_sca_L1
X_tier English_sca_R1
X_tier Dutch_cv INCLUDE C
y_tier Dutch
"""

X_tiers4, y_tiers4 = multitiers.utils.parse_study(study4)
clf4.train(X_tiers4, y_tiers4, max_depth=15)
clf4.show_pred(max_lines=10)
```

Among the first ten occurrences, this will miss one case, predicting
an /x/ when an /ŋ/ is observed (as we carry the IDs in the tiers, we could
either check or exclude this word from the analysis):

```bash
#0: v/v -- True
#1: r/r -- True
#2: s/s -- True
#3: p/p -- True
#4: r/r -- True
#5: l/l -- True
#6: x/x -- True
#7: h/h -- True
#8: n/n -- True
#9: ŋ/x -- False
#10: b/b -- True
```

The same experiment can be performed with the probabilities:

```python
clf4.show_pred_prob(max_lines=10)
```

Which will print the three most likely classes for each position, highlighting,
among others, how the case above is an outlier in this simple model or how the
model is totally confident in most cases. The first entry is interesting,
showing that, while the correct sound /v/ was predicted with 97% confidence,
the corresponding voiceless sound would not be totally expected.

```bash
#0: v/(v|0.973,f|0.027)
#1: r/(r|1.000)
#2: s/(s|1.000)
#3: p/(p|1.000)
#4: r/(r|1.000)
#5: l/(l|1.000)
#6: x/(x|0.519,ɣ|0.160,ʋ|0.086)
#7: h/(h|1.000)
#8: n/(n|1.000)
#9: ŋ/(x|0.519,ɣ|0.160,ʋ|0.086)
#10: b/(b|1.000)
```

Note that these experiments were conducted on the training data, with no
splitting of training and test, cross-validation, or similar. While these
can be performed, in most cases we will want to use all available data to
understand how it can be described, and not for actual predictions.

## TODO

- Allow to strip material between parentheses and/or morphology markers
  in `utils.parse_alignment()`
- Allow to pass also string or integers (i.e., using `get_orders()`) to
  `utils.shift_tier()`

# Draft

The multitiered representation of alignments is constructed through a systematic process that facilitates the analysis of cross-linguistic phonological data. Given the alignments in the form of a dictionary with tuples of (parameter, doculect) as keys and lists of phonemes as values, the process initializes by identifying the unique parameters and doculects.

For each parameter, a sub-dataframe is created with the phonemes from the corresponding doculects as columns, where each alignment site becomes a row. If an alignment for a given doculect is missing, the values are filled with NaN. This ensures that a uniform structure is maintained across different alignments.

Additional columns, referred to as tiers, are then added using a set of specified functions (contained in the function_dict). These functions allow the mapping of phonemes to other values, such as sound classes or numerical features. The application of these functions creates a parallel tier for each phoneme column, enabling a multifaceted view of the data.

The next step involves the addition of left and right contexts to the alignments, as specified by the left and right parameters. The process constructs these contexts on a per-alignment basis, ensuring that the context does not extend beyond the boundaries of the specific alignment. Context elements extending outside the alignment are represented by a placeholder symbol '∅'. Contextual extensions can also be applied to the additional tiers specified in the context_tiers argument, allowing the user to analyze not only phonemes but also other aspects like sound classes within a broader contextual framework.

Once the individual sub-dataframes for each parameter are constructed with the phonemes, additional tiers, and contextual extensions, they are concatenated into a final DataFrame. An ID column is added to uniquely identify each alignment site across different parameters, providing a coherent and unified representation suitable for various types of linguistic analyses.

In summary, this method provides a versatile and robust way to represent cross-linguistic phonological alignments in a structured and extensible form. By encapsulating phonemes, additional tiers, and contextual information within a unified framework, it lays the groundwork for in-depth explorations and comparisons across different languages and linguistic features.

AND indexes