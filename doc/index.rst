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

-----------------------------------------
Fitting data actual charset to a model
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

.. toctree::
   :maxdepth: 3
   :caption: Contents:


Alphabet
===========

.. automodule:: character_classes
   :members:

.. automodule:: libs.alphabet
   :members:


