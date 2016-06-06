"""
By: Akhila Ananthram (asa225)
"""
import numpy as np
import os
import json
import pandas

from Bio import SeqIO, AlignIO
from constants import *
from string import digits
from node import Node


def generate_leaves(sequences):
  """
  Converts DNA sequences to numerical representations and makes nodes
  @param sequences     (list)    list of list of DNA bases
  @return              (list)    list of Nodes
  """
  return [Node(np.array([MAPPING[b] for b in seq])) for seq in sequences]


def generate_trees(sequences):
  """
  tree topologies: HC1, HC2, HG, CG
  @param sequences (np.array)   list of list of DNA bases
  @return          (tuple)      tuple of Nodes representing the trees
  """
  # Tree HC1
  H, C, G, O = generate_leaves(sequences)
  len_seq = H.len_seq

  a = 0.006591
  b = 0.009411 - a
  c = 0.036199
  H.time = a
  C.time = a
  G.time = a + b
  O.time = c
  n1 = Node(len_seq, b, H, C)
  n2 = Node(len_seq, c - a - b, n1, G)
  HC1 = Node(len_seq, left=n2, right=O, name="HC1")

  # converts from paper's numbers to ours
  factor = c / 28.53

  # Tree HC2
  H, C, G, O = generate_leaves(sequences)
  len_seq = H.len_seq

  a_tilda = 6.11 * factor
  b_tilda = 2.03 * factor
  c_tilda = 27.85 * factor
  H.time = a_tilda
  C.time = a_tilda
  G.time = a_tilda + b_tilda
  O.time = c_tilda
  n1 = Node(len_seq, b_tilda, H, C)
  n2 = Node(len_seq, c_tilda - a_tilda - b_tilda, n1, G)
  HC2 = Node(len_seq, left=n2, right=O, name="HC2")

  # Tree HG
  H, C, G, O = generate_leaves(sequences)
  len_seq = H.len_seq

  a_tilda = (3.85 + 1.58) * factor
  b_tilda = (6.11 + 2.03 - 3.85 - 1.58) * factor
  c_tilda = 27.85 * factor
  H.time = a_tilda
  C.time = a_tilda + b_tilda
  G.time = a_tilda
  O.time = c_tilda
  n1 = Node(len_seq, b_tilda, H, G)
  n2 = Node(len_seq, c_tilda - a_tilda - b_tilda, n1, C)
  HG = Node(len_seq, left=n2, right=O, name="HG")

  # Tree CG
  H, C, G, O = generate_leaves(sequences)
  len_seq = H.len_seq

  a_tilda = (3.85 + 1.58) * factor
  b_tilda = (6.11 + 2.03 - 3.85 - 1.58) * factor
  c_tilda = 27.85 * factor
  H.time = a_tilda + b_tilda
  C.time = a_tilda
  G.time = a_tilda
  O.time = c_tilda
  n1 = Node(len_seq, b_tilda, C, G)
  n2 = Node(len_seq, c_tilda - a_tilda - b_tilda, n1, H)
  CG = Node(len_seq, left=n2, right=O, name="CG")

  return [HC1, HC2, HG, CG]

def read_maf(filename, save=False):
  """
  Reads the maf file and returns the alignments for human, chimpanzee, gorilla,
  and orangutan
  @param filename (str)     Name of maf file or json file to read
  @param save     (bool)    Whether or not to save a json file of a subset
  @return         (list)    list of list of (name, bases) for the alignments
  """
  basename, ext = os.path.splitext(filename)
  if ext == ".json":
    with open(filename) as f:
      return json.load(f)

  alignments = []
  current = [None, None, None, None]

  # TODO: eventually add gibbon as well
  # hg38, panTro4, gorGor3, ponAbe2, nomLeu3
  # human, chimpanzee, gorilla, orangutan, gibbon
  names = ["hg", "panTro", "gorGor", "ponAbe"]
  with open(filename) as f:
    for line in f:
      # comment
      if line.startswith('#'):
        continue

      # new alignment
      if line.startswith('a'):
        current = [None, None, None, None]

      # new sequence for alignment
      if line.startswith('s'):
        s = line.split()
        seq = list(s[-1].upper())
        name = (s[1].split('.')[0]).translate(None, digits)

        if name in names and set(seq) == set(['A','C', 'T', 'G', '-']):
          current[names.index(name)] = seq

      # end of alignment
      if not line.strip():
        alignments.append(current)

  alignments = [a for a in alignments if all(x is not None for x in a)]

  if save:
    print "Saving"
    with open("{}.json".format(basename), 'w') as f:
      json.dump(alignments, f)
    print "done"

  return alignments
