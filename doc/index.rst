.. DiDip Handwriting Datasets documentation master file, created by
   sphinx-quickstart on Sun Nov  3 09:57:03 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

***************************
HTR Mini-apps
***************************

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


How to use 
====================================

VGSL
=================

Note: by convention, VGSL specifies the dimensions in NHWC order, while Torch uses NCHW. The example
below uses a NHWC = (1, 64, 2048, 3) image as an input.

+-------------+------------------------------------------+-------------------------+--------------------+
| VGSL        | DESCRIPTION                              | Output size (NHWC)      | Output size (NCHW) |
+=============+==========================================+=========================+====================+
| Cr3,13,32   | kernel filter 3x13, 32 activations relu  | 1, 64, 2048, 32         | 1, 32, 64, 2048    |
+-------------+------------------------------------------+-------------------------+--------------------+
| Do0.1,2     | dropout prob 0.1 dims 2                  | -                       | -                  |
+-------------+------------------------------------------+-------------------------+--------------------+
| Mp2,2       | Max Pool kernel 2x2 stride 2x2           | 1, 32, 1024, 32         | 1, 32, 32, 1024    | 
+-------------+------------------------------------------+-------------------------+--------------------+
| ...         | (same)                                   | 1, 16,  512, 32         | 1, 32, 16, 512     |
| Cr3,9,64    | kernel filter 3x9, 64 activations relu   | 1, 16,  512, 64         | 1, 64, 16, 512     |
+-------------+------------------------------------------+-------------------------+--------------------+
| ...         |                                          |                         |                    |
| Mp2,2       |                                          | 1, 8, 256, 64           | 1, 64, 8, 256      |
+-------------+------------------------------------------+-------------------------+--------------------+
| Cr3,9,64    |                                          | 1, 8, 256, 64           | 1, 64, 8, 256      |
| Do0.1,2     |                                          |                         |                    |
+-------------+------------------------------------------+-------------------------+--------------------+
| S1(1x0)1,3  | reshape (N,H,W,C) into N, 1, W,C*H       | 1, 1, 256, 64x8=512     | 1, 1024, 1, 256    |
+-------------+------------------------------------------+-------------------------+--------------------+
| Lbx200      | RNN b[irectional] on width-dimension (x) | 1, 1, 256, 400          | 1, 400, 1, 256     |
|             | with 200 output channels                 | (either forward (f) or  |                    |
|             |                                          | reverse (r) would yield |                    |
|             |                                          | 200-sized output)       |                    |
+-------------+------------------------------------------+-------------------------+--------------------+
| ...         | (same)                                   |                         |                    |
| Lbx200      | RNN b[irectional] on width-dimension (x) | 1, 1, 256, 400          | 1, 400, 1, 256     |
+-------------+------------------------------------------+-------------------------+--------------------+



Alphabets: notes for myself
===========================

-----------------
Desired features
-----------------


* An alphabet is a glorified dictionary; if it is many-to-one (*n* symbols map to 1 code), the operation
  that maps a code to a symbol (for decoding) is consistently returns the same symbol, no matter how the
  dictionary was created.

* No matter how it has been created (from a in-memory dictionary or a file), the alphabet 
  is stored and serialized with the model.

* In its serialized form, the alphabet is easy to read and easy to modify, as well as foolproof: ordering should not matter.

----------------------------
Input/Serialization format
----------------------------

Although the internal storage of an alphabet is a pair of dictionaries (symbol-to-code and code-to-symbol),
the preferred format for input and serialization is a list of lists of this form:

```python
['0', '1', '2', ..., ['A', 'À', 'Á', 'Â', ...], 'B', ['C', 'Ç', 'Ć', ...], ..., ['Z', 'Ź', 'Ż', 'Ž'], ... ]
```

* A list made only of atoms (no sublist) defines a one-to-one alphabet; atoms may single-character symbols or compound symbols (eg. `'ae'`).
* All symbols in the same sublist map to the same code.
* Order does not matter, except in one case: the code assigned to a sublist maps to the first character in the list (in the decoding stage)
* Special characters (EOS, SOS, epsilon, default) are handled automatically, whether they are already in the input list or not.

There is an option to dump the alphabet's dictionary into a TSV file, for easy checking of the mapping. This is a one-way operation: in order to avoid redundancy, no method is provided to construct an alphabet from a TSV file.

-----------------------------------------
How actual data fit to a model
-----------------------------------------

Distinguish between:

1. GT symbols we want to ignore/discard, because they're junk or
   irrelevant for any HTR purpose - they typically do not have
   any counterpart in the input image; one option is to explicitly skip them at encoding time
   (instead of defaulting to the unknown code), but that complicates both the 
   encoding function and the alphabet definition (which has to store the list of banned 
   characters); the other option-the right one-is to filter the characters
   out of the sample, using a target transform, for instance.

2. GT symbols we don't need to interpret but typically have their
   counterpart in the image and need to to be part of the output
   somehow (if just for being able to say: there is something
   here that needs interpreting. Eg. abbreviations); the alphabet
   map them all to a single, default code, that stands for 'unknown'.

Examples: how to deal with... ?

* diacritics? Depending on the corpus, some accented letters such as 'Ê' may be rendered either as
  a 1-byte character (from the extended Latin charset: in this case 0x00CA) or as the combination 
  of a letter and a diacritic (in this case 0x0045 (E) + 0x0302 (̂ )).
  
  + at the alphabet level, the accented 1-byte glyph (say 'Ê') maps on its class representative's code (eg. 'E')
  + at encoding time, the diacritic mark is either removed from the input string (simpler); 
  
  If we wanted the alphabet to handle the accented character as such, one could replace the 1-byte combination by its extended Latin 1-byte equivalent, but it is not always possible. A step further would be to include the combinations themselves into the alphabet(as compound symbols), but that is probably an overkill.
 



.. toctree::
   :maxdepth: 3
   :caption: Contents:


Alphabet
===========

.. automodule:: character_classes
   :members:

.. automodule:: libs.alphabet
   :members:


