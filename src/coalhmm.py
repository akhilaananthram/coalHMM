"""
By: Akhila Ananthram (asa225)
"""
import argparse
import math
import numpy as np
from collections import Counter

import hmm
import felsenstein
import utils

def coalhmm(args):
  """
  Trains and tests a Coal-HMM
  @param args   (argparse.Namespace)   Arguments provided by user: filename,
                                       sample, rounds
  """
  # from table 2 in Hobolth et al.
  # mean_fragment_length_HC1 = 1684
  # mean_fragment_length_others = 65
  # probability_leaving_HC1 = 3 * s = 1 / 1684
  s = 1.0 / (1684 * 3)
  # probability_leaving_others = 1 / 65 = u + 2 * v
  # u + 2 * v = 1 / 65
  stationary = (0.49, 0.17, 0.17, 0.17)
  # stationary = np.array([psi, (1 - psi) / 3, (1 - psi) / 3, (1 - psi) / 3])
  psi = 0.49
  # psi = 1 / (1 + 3 * s / u)
  # 1 + 3 * s / u = 1 / psi
  # u + 3 * s = u / psi
  # 3 * s = (1 / psi - 1) * u
  u = 3 * s / (1 / psi - 1)
  v = (1 / 65.0 - u) / 2

  # Transition probability: HC1, HC2, HG, CG
  transition = np.array([[1 - 3 * s, s, s, s],
                         [u, 1 - (u + 2 * v), v, v],
                         [u, v, 1 - (u + 2 * v), v],
                         [u, v, v, 1 - (u + 2 * v)]])

  print "Reading alignments"
  original_alignments = [np.array(a) for a in utils.read_maf(args.filename)]
  print "done"
  for j in range(args.rounds):
    print "ROUND {}".format(j)

    print "sampling"
    if args.sample is not None:
      alignments = [original_alignments[i] for i in np.random.choice(np.arange(len(original_alignments)), args.sample, False)]
    else:
      alignments = original_alignments
    print "Number of alignments: {}".format(len(alignments))
    print "done"

    print "felsenstein"
    groupings = {}
    emission = np.zeros((4, 5 ** 4))
    for alignment in alignments:
      _, len_alignment = alignment.shape
      for i in range(len_alignment):
        column = "".join(alignment[:, i])
        if column not in groupings:
          groupings[column] = len(groupings)

          trees = utils.generate_trees(alignment[:, i])
          
          # Felsenstein to get emission
          for i, t in enumerate(trees):
            emission[i, groupings[column]] = math.exp(felsenstein.felsensteins(t))
    print "done"

    print "BW"
    initial = np.array([0.25, 0.25, 0.25, 0.25])
    # Baum welsh to update matrices
    emission, transition = hmm.baum_welch(initial, emission, transition, alignments, groupings)
    print "done"

    print "viterbi"
    # use viterbi to see which state we are in the longest
    hidden_states = []
    for alignment in alignments:
      hidden_states.append(hmm.viterbi(initial, emission, transition, alignment, groupings))
    print "done"

    # calculate time spent in a state
    counts = Counter([s for states in hidden_states for s in states])
    print "Number of bases in each state: ", counts


if __name__ == "__main__":
  parser = argparse.ArgumentParser("Coal-HMM")
  parser.add_argument("filename", help="name of maf file to train on")
  parser.add_argument("--sample", default=None, type=int, help="number to sample")
  parser.add_argument("--rounds", default=1, type=int, help="number to rounds to run")
  args = parser.parse_args()
  coalhmm(args)
