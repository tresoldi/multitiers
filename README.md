# multitiers

A library for multi-tiered sequence representation of linguistic data.

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
would need investigation. More complex studies could consider other tiers,
and they are not limited to a single unknown tier: we can, for example,
investigate tuples of correspondences between different doculects, such as
the relation of German/Dutch against English.

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

[add image]

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

[add image]

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

[add image]

The visualization shows that the most common Dutch vowel is /ə/, which
will be correct almost all the time (Gini index of 0.015) if the
corresponding German vowel is /ə/ is as well. If the German vowel is /a/,
we expect /ɑ/ in Dutch (but note the high impurity), if it is /aː/ we
can expect a corresponding /aː/, and so on. Note that the tree didn't pick
any tier from English: at least with the restrictions we set, the German
vowel is always more informative.

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
#0: v/[(0.972972972972973, 'v'), (0.02702702702702703, 'f')]
#1: r/[(1.0, 'r')]
#2: s/[(1.0, 's')]
#3: p/[(1.0, 'p')]
#4: r/[(1.0, 'r')]
#5: l/[(1.0, 'l')]
#6: x/[(0.5185185185185185, 'x'), (0.16049382716049382, 'ɣ'), (0.08641975308641975, 'ʋ')]
#7: h/[(1.0, 'h')]
#8: n/[(1.0, 'n')]
#9: ŋ/[(0.5185185185185185, 'x'), (0.16049382716049382, 'ɣ'), (0.08641975308641975, 'ʋ')]
#10: b/[(1.0, 'b')]
```

Note that these experiments were conducted on the training data, with no
splitting of training and test, cross-validation, or similar. While these
can be performed, in most cases we will want to use all available data to
understand how it can be described, and not for actual predictions.

## TODO

- Allow shifted alignment tiers at any moment (not only upon loading)
