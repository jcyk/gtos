#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
This script computes smatch score between two AMRs.
For detailed description of smatch, see http://www.isi.edu/natural-language/amr/smatch-13.pdf

"""

from __future__ import print_function
from __future__ import division

import amr
import os
import random
import sys

# total number of iteration in smatch computation
iteration_num = 5

# verbose output switch.
# Default false (no verbose output)
verbose = False
veryVerbose = False

# single score output switch.
# Default true (compute a single score for all AMRs in two files)
single_score = True

# precision and recall output switch.
# Default false (do not output precision and recall, just output F score)
pr_flag = False

# Error log location
ERROR_LOG = sys.stderr

# Debug log location
DEBUG_LOG = sys.stderr

# dictionary to save pre-computed node mapping and its resulting triple match count
# key: tuples of node mapping
# value: the matching triple count
match_triple_dict = {}


def build_arg_parser():
    """
    Build an argument parser using argparse. Use it when python version is 2.7 or later.

    """
    parser = argparse.ArgumentParser(description="Smatch calculator -- arguments")
    parser.add_argument('-f', nargs=2, required=True, type=argparse.FileType('r'),
                        help='Two files containing AMR pairs. AMRs in each file are separated by a single blank line')
    parser.add_argument('-r', type=int, default=4, help='Restart number (Default:4)')
    parser.add_argument('--significant', type=int, default=3, help='significant digits to output (default: 2)')
    parser.add_argument('-v', action='store_true', help='Verbose output (Default:false)')
    parser.add_argument('--vv', action='store_true', help='Very Verbose output (Default:false)')
    parser.add_argument('--ms', action='store_true', default=False,
                        help='Output multiple scores (one AMR pair a score)'
                             'instead of a single document-level smatch score (Default: false)')
    parser.add_argument('--pr', action='store_true', default=False,
                        help="Output precision and recall as well as the f-score. Default: false")
    parser.add_argument('--justinstance', action='store_true', default=False,
                        help="just pay attention to matching instances")
    parser.add_argument('--justattribute', action='store_true', default=False,
                        help="just pay attention to matching attributes")
    parser.add_argument('--justrelation', action='store_true', default=False,
                        help="just pay attention to matching relations")

    return parser


def build_arg_parser2():
    """
    Build an argument parser using optparse. Use it when python version is 2.5 or 2.6.

    """
    usage_str = "Smatch calculator -- arguments"
    parser = optparse.OptionParser(usage=usage_str)
    parser.add_option("-f", "--files", nargs=2, dest="f", type="string",
                      help='Two files containing AMR pairs. AMRs in each file are ' \
                           'separated by a single blank line. This option is required.')
    parser.add_option("-r", "--restart", dest="r", type="int", help='Restart number (Default: 4)')
    parser.add_option('--significant', dest="significant", type="int", default=2,
                      help='significant digits to output (default: 2)')
    parser.add_option("-v", "--verbose", action='store_true', dest="v", help='Verbose output (Default:False)')
    parser.add_option("--vv", "--veryverbose", action='store_true', dest="vv",
                      help='Very Verbose output (Default:False)')
    parser.add_option("--ms", "--multiple_score", action='store_true', dest="ms",
                      help='Output multiple scores (one AMR pair a score) instead of ' \
                           'a single document-level smatch score (Default: False)')
    parser.add_option('--pr', "--precision_recall", action='store_true', dest="pr",
                      help="Output precision and recall as well as the f-score. Default: false")
    parser.add_option('--justinstance', action='store_true', default=False,
                      help="just pay attention to matching instances")
    parser.add_option('--justattribute', action='store_true', default=False,
                      help="just pay attention to matching attributes")
    parser.add_option('--justrelation', action='store_true', default=False,
                      help="just pay attention to matching relations")
    parser.set_defaults(r=4, v=False, ms=False, pr=False)
    return parser


def get_best_match(instance1, attribute1, relation1,
                   instance2, attribute2, relation2,
                   prefix1, prefix2, doinstance=True, doattribute=True, dorelation=True):
    """
    Get the highest triple match number between two sets of triples via hill-climbing.
    Arguments:
        instance1: instance triples of AMR 1 ("instance", node name, node value)
        attribute1: attribute triples of AMR 1 (attribute name, node name, attribute value)
        relation1: relation triples of AMR 1 (relation name, node 1 name, node 2 name)
        instance2: instance triples of AMR 2 ("instance", node name, node value)
        attribute2: attribute triples of AMR 2 (attribute name, node name, attribute value)
        relation2: relation triples of AMR 2 (relation name, node 1 name, node 2 name)
        prefix1: prefix label for AMR 1
        prefix2: prefix label for AMR 2
    Returns:
        best_match: the node mapping that results in the highest triple matching number
        best_match_num: the highest triple matching number

    """
    # Compute candidate pool - all possible node match candidates.
    # In the hill-climbing, we only consider candidate in this pool to save computing time.
    # weight_dict is a dictionary that maps a pair of node
    (candidate_mappings, weight_dict) = compute_pool(instance1, attribute1, relation1,
                                                     instance2, attribute2, relation2,
                                                     prefix1, prefix2, doinstance=doinstance, doattribute=doattribute,
                                                     dorelation=dorelation)
    if veryVerbose:
        print("Candidate mappings:", file=DEBUG_LOG)
        print(candidate_mappings, file=DEBUG_LOG)
        print("Weight dictionary", file=DEBUG_LOG)
        print(weight_dict, file=DEBUG_LOG)

    best_match_num = 0
    # initialize best match mapping
    # the ith entry is the node index in AMR 2 which maps to the ith node in AMR 1
    best_mapping = [-1] * len(instance1)
    for i in range(iteration_num):
        if veryVerbose:
            print("Iteration", i, file=DEBUG_LOG)
        if i == 0:
            # smart initialization used for the first round
            cur_mapping = smart_init_mapping(candidate_mappings, instance1, instance2)
        else:
            # random initialization for the other round
            cur_mapping = random_init_mapping(candidate_mappings)
        # compute current triple match number
        match_num = compute_match(cur_mapping, weight_dict)
        if veryVerbose:
            print("Node mapping at start", cur_mapping, file=DEBUG_LOG)
            print("Triple match number at start:", match_num, file=DEBUG_LOG)
        while True:
            # get best gain
            (gain, new_mapping) = get_best_gain(cur_mapping, candidate_mappings, weight_dict,
                                                len(instance2), match_num)
            if veryVerbose:
                print("Gain after the hill-climbing", gain, file=DEBUG_LOG)
            # hill-climbing until there will be no gain for new node mapping
            if gain <= 0:
                break
            # otherwise update match_num and mapping
            match_num += gain
            cur_mapping = new_mapping[:]
            if veryVerbose:
                print("Update triple match number to:", match_num, file=DEBUG_LOG)
                print("Current mapping:", cur_mapping, file=DEBUG_LOG)
        if match_num > best_match_num:
            best_mapping = cur_mapping[:]
            best_match_num = match_num
    return best_mapping, best_match_num


def normalize(item):
    """
    lowercase and remove quote signifiers from items that are about to be compared
    """
    return item.lower().rstrip('_')


def compute_pool(instance1, attribute1, relation1,
                 instance2, attribute2, relation2,
                 prefix1, prefix2, doinstance=True, doattribute=True, dorelation=True):
    """
    compute all possible node mapping candidates and their weights (the triple matching number gain resulting from
    mapping one node in AMR 1 to another node in AMR2)

    Arguments:
        instance1: instance triples of AMR 1
        attribute1: attribute triples of AMR 1 (attribute name, node name, attribute value)
        relation1: relation triples of AMR 1 (relation name, node 1 name, node 2 name)
        instance2: instance triples of AMR 2
        attribute2: attribute triples of AMR 2 (attribute name, node name, attribute value)
        relation2: relation triples of AMR 2 (relation name, node 1 name, node 2 name
        prefix1: prefix label for AMR 1
        prefix2: prefix label for AMR 2
    Returns:
      candidate_mapping: a list of candidate nodes.
                       The ith element contains the node indices (in AMR 2) the ith node (in AMR 1) can map to.
                       (resulting in non-zero triple match)
      weight_dict: a dictionary which contains the matching triple number for every pair of node mapping. The key
                   is a node pair. The value is another dictionary. key {-1} is triple match resulting from this node
                   pair alone (instance triples and attribute triples), and other keys are node pairs that can result
                   in relation triple match together with the first node pair.


    """
    candidate_mapping = []
    weight_dict = {}
    for instance1_item in instance1:
        # each candidate mapping is a set of node indices
        candidate_mapping.append(set())
        if doinstance:
            for instance2_item in instance2:
                # if both triples are instance triples and have the same value
                if normalize(instance1_item[0]) == normalize(instance2_item[0]) and \
                        normalize(instance1_item[2]) == normalize(instance2_item[2]):
                    # get node index by stripping the prefix
                    node1_index = int(instance1_item[1][len(prefix1):])
                    node2_index = int(instance2_item[1][len(prefix2):])
                    candidate_mapping[node1_index].add(node2_index)
                    node_pair = (node1_index, node2_index)
                    # use -1 as key in weight_dict for instance triples and attribute triples
                    if node_pair in weight_dict:
                        weight_dict[node_pair][-1] += 1
                    else:
                        weight_dict[node_pair] = {}
                        weight_dict[node_pair][-1] = 1
    if doattribute:
        for attribute1_item in attribute1:
            for attribute2_item in attribute2:
                # if both attribute relation triple have the same relation name and value
                if normalize(attribute1_item[0]) == normalize(attribute2_item[0]) \
                        and normalize(attribute1_item[2]) == normalize(attribute2_item[2]):
                    node1_index = int(attribute1_item[1][len(prefix1):])
                    node2_index = int(attribute2_item[1][len(prefix2):])
                    candidate_mapping[node1_index].add(node2_index)
                    node_pair = (node1_index, node2_index)
                    # use -1 as key in weight_dict for instance triples and attribute triples
                    if node_pair in weight_dict:
                        weight_dict[node_pair][-1] += 1
                    else:
                        weight_dict[node_pair] = {}
                        weight_dict[node_pair][-1] = 1
    if dorelation:
        for relation1_item in relation1:
            for relation2_item in relation2:
                # if both relation share the same name
                if normalize(relation1_item[0]) == normalize(relation2_item[0]):
                    node1_index_amr1 = int(relation1_item[1][len(prefix1):])
                    node1_index_amr2 = int(relation2_item[1][len(prefix2):])
                    node2_index_amr1 = int(relation1_item[2][len(prefix1):])
                    node2_index_amr2 = int(relation2_item[2][len(prefix2):])
                    # add mapping between two nodes
                    candidate_mapping[node1_index_amr1].add(node1_index_amr2)
                    candidate_mapping[node2_index_amr1].add(node2_index_amr2)
                    node_pair1 = (node1_index_amr1, node1_index_amr2)
                    node_pair2 = (node2_index_amr1, node2_index_amr2)
                    if node_pair2 != node_pair1:
                        # update weight_dict weight. Note that we need to update both entries for future search
                        # i.e weight_dict[node_pair1][node_pair2]
                        #     weight_dict[node_pair2][node_pair1]
                        if node1_index_amr1 > node2_index_amr1:
                            # swap node_pair1 and node_pair2
                            node_pair1 = (node2_index_amr1, node2_index_amr2)
                            node_pair2 = (node1_index_amr1, node1_index_amr2)
                        if node_pair1 in weight_dict:
                            if node_pair2 in weight_dict[node_pair1]:
                                weight_dict[node_pair1][node_pair2] += 1
                            else:
                                weight_dict[node_pair1][node_pair2] = 1
                        else:
                            weight_dict[node_pair1] = {-1: 0, node_pair2: 1}
                        if node_pair2 in weight_dict:
                            if node_pair1 in weight_dict[node_pair2]:
                                weight_dict[node_pair2][node_pair1] += 1
                            else:
                                weight_dict[node_pair2][node_pair1] = 1
                        else:
                            weight_dict[node_pair2] = {-1: 0, node_pair1: 1}
                    else:
                        # two node pairs are the same. So we only update weight_dict once.
                        # this generally should not happen.
                        if node_pair1 in weight_dict:
                            weight_dict[node_pair1][-1] += 1
                        else:
                            weight_dict[node_pair1] = {-1: 1}
    return candidate_mapping, weight_dict


def smart_init_mapping(candidate_mapping, instance1, instance2):
    """
    Initialize mapping based on the concept mapping (smart initialization)
    Arguments:
        candidate_mapping: candidate node match list
        instance1: instance triples of AMR 1
        instance2: instance triples of AMR 2
    Returns:
        initialized node mapping between two AMRs

    """
    random.seed()
    matched_dict = {}
    result = []
    # list to store node indices that have no concept match
    no_word_match = []
    for i, candidates in enumerate(candidate_mapping):
        if not candidates:
            # no possible mapping
            result.append(-1)
            continue
        # node value in instance triples of AMR 1
        value1 = instance1[i][2]
        for node_index in candidates:
            value2 = instance2[node_index][2]
            # find the first instance triple match in the candidates
            # instance triple match is having the same concept value
            if value1 == value2:
                if node_index not in matched_dict:
                    result.append(node_index)
                    matched_dict[node_index] = 1
                    break
        if len(result) == i:
            no_word_match.append(i)
            result.append(-1)
    # if no concept match, generate a random mapping
    for i in no_word_match:
        candidates = list(candidate_mapping[i])
        while candidates:
            # get a random node index from candidates
            rid = random.randint(0, len(candidates) - 1)
            candidate = candidates[rid]
            if candidate in matched_dict:
                candidates.pop(rid)
            else:
                matched_dict[candidate] = 1
                result[i] = candidate
                break
    return result


def random_init_mapping(candidate_mapping):
    """
    Generate a random node mapping.
    Args:
        candidate_mapping: candidate_mapping: candidate node match list
    Returns:
        randomly-generated node mapping between two AMRs

    """
    # if needed, a fixed seed could be passed here to generate same random (to help debugging)
    random.seed()
    matched_dict = {}
    result = []
    for c in candidate_mapping:
        candidates = list(c)
        if not candidates:
            # -1 indicates no possible mapping
            result.append(-1)
            continue
        found = False
        while candidates:
            # randomly generate an index in [0, length of candidates)
            rid = random.randint(0, len(candidates) - 1)
            candidate = candidates[rid]
            # check if it has already been matched
            if candidate in matched_dict:
                candidates.pop(rid)
            else:
                matched_dict[candidate] = 1
                result.append(candidate)
                found = True
                break
        if not found:
            result.append(-1)
    return result


def compute_match(mapping, weight_dict):
    """
    Given a node mapping, compute match number based on weight_dict.
    Args:
    mappings: a list of node index in AMR 2. The ith element (value j) means node i in AMR 1 maps to node j in AMR 2.
    Returns:
    matching triple number
    Complexity: O(m*n) , m is the node number of AMR 1, n is the node number of AMR 2

    """
    # If this mapping has been investigated before, retrieve the value instead of re-computing.
    if veryVerbose:
        print("Computing match for mapping", file=DEBUG_LOG)
        print(mapping, file=DEBUG_LOG)
    if tuple(mapping) in match_triple_dict:
        if veryVerbose:
            print("saved value", match_triple_dict[tuple(mapping)], file=DEBUG_LOG)
        return match_triple_dict[tuple(mapping)]
    match_num = 0
    # i is node index in AMR 1, m is node index in AMR 2
    for i, m in enumerate(mapping):
        if m == -1:
            # no node maps to this node
            continue
        # node i in AMR 1 maps to node m in AMR 2
        current_node_pair = (i, m)
        if current_node_pair not in weight_dict:
            continue
        if veryVerbose:
            print("node_pair", current_node_pair, file=DEBUG_LOG)
        for key in weight_dict[current_node_pair]:
            if key == -1:
                # matching triple resulting from instance/attribute triples
                match_num += weight_dict[current_node_pair][key]
                if veryVerbose:
                    print("instance/attribute match", weight_dict[current_node_pair][key], file=DEBUG_LOG)
            # only consider node index larger than i to avoid duplicates
            # as we store both weight_dict[node_pair1][node_pair2] and
            #     weight_dict[node_pair2][node_pair1] for a relation
            elif key[0] < i:
                continue
            elif mapping[key[0]] == key[1]:
                match_num += weight_dict[current_node_pair][key]
                if veryVerbose:
                    print("relation match with", key, weight_dict[current_node_pair][key], file=DEBUG_LOG)
    if veryVerbose:
        print("match computing complete, result:", match_num, file=DEBUG_LOG)
    # update match_triple_dict
    match_triple_dict[tuple(mapping)] = match_num
    return match_num


def move_gain(mapping, node_id, old_id, new_id, weight_dict, match_num):
    """
    Compute the triple match number gain from the move operation
    Arguments:
        mapping: current node mapping
        node_id: remapped node in AMR 1
        old_id: original node id in AMR 2 to which node_id is mapped
        new_id: new node in to which node_id is mapped
        weight_dict: weight dictionary
        match_num: the original triple matching number
    Returns:
        the triple match gain number (might be negative)

    """
    # new node mapping after moving
    new_mapping = (node_id, new_id)
    # node mapping before moving
    old_mapping = (node_id, old_id)
    # new nodes mapping list (all node pairs)
    new_mapping_list = mapping[:]
    new_mapping_list[node_id] = new_id
    # if this mapping is already been investigated, use saved one to avoid duplicate computing
    if tuple(new_mapping_list) in match_triple_dict:
        return match_triple_dict[tuple(new_mapping_list)] - match_num
    gain = 0
    # add the triple match incurred by new_mapping to gain
    if new_mapping in weight_dict:
        for key in weight_dict[new_mapping]:
            if key == -1:
                # instance/attribute triple match
                gain += weight_dict[new_mapping][-1]
            elif new_mapping_list[key[0]] == key[1]:
                # relation gain incurred by new_mapping and another node pair in new_mapping_list
                gain += weight_dict[new_mapping][key]
    # deduct the triple match incurred by old_mapping from gain
    if old_mapping in weight_dict:
        for k in weight_dict[old_mapping]:
            if k == -1:
                gain -= weight_dict[old_mapping][-1]
            elif mapping[k[0]] == k[1]:
                gain -= weight_dict[old_mapping][k]
    # update match number dictionary
    match_triple_dict[tuple(new_mapping_list)] = match_num + gain
    return gain


def swap_gain(mapping, node_id1, mapping_id1, node_id2, mapping_id2, weight_dict, match_num):
    """
    Compute the triple match number gain from the swapping
    Arguments:
    mapping: current node mapping list
    node_id1: node 1 index in AMR 1
    mapping_id1: the node index in AMR 2 node 1 maps to (in the current mapping)
    node_id2: node 2 index in AMR 1
    mapping_id2: the node index in AMR 2 node 2 maps to (in the current mapping)
    weight_dict: weight dictionary
    match_num: the original matching triple number
    Returns:
    the gain number (might be negative)

    """
    new_mapping_list = mapping[:]
    # Before swapping, node_id1 maps to mapping_id1, and node_id2 maps to mapping_id2
    # After swapping, node_id1 maps to mapping_id2 and node_id2 maps to mapping_id1
    new_mapping_list[node_id1] = mapping_id2
    new_mapping_list[node_id2] = mapping_id1
    if tuple(new_mapping_list) in match_triple_dict:
        return match_triple_dict[tuple(new_mapping_list)] - match_num
    gain = 0
    new_mapping1 = (node_id1, mapping_id2)
    new_mapping2 = (node_id2, mapping_id1)
    old_mapping1 = (node_id1, mapping_id1)
    old_mapping2 = (node_id2, mapping_id2)
    if node_id1 > node_id2:
        new_mapping2 = (node_id1, mapping_id2)
        new_mapping1 = (node_id2, mapping_id1)
        old_mapping1 = (node_id2, mapping_id2)
        old_mapping2 = (node_id1, mapping_id1)
    if new_mapping1 in weight_dict:
        for key in weight_dict[new_mapping1]:
            if key == -1:
                gain += weight_dict[new_mapping1][-1]
            elif new_mapping_list[key[0]] == key[1]:
                gain += weight_dict[new_mapping1][key]
    if new_mapping2 in weight_dict:
        for key in weight_dict[new_mapping2]:
            if key == -1:
                gain += weight_dict[new_mapping2][-1]
            # to avoid duplicate
            elif key[0] == node_id1:
                continue
            elif new_mapping_list[key[0]] == key[1]:
                gain += weight_dict[new_mapping2][key]
    if old_mapping1 in weight_dict:
        for key in weight_dict[old_mapping1]:
            if key == -1:
                gain -= weight_dict[old_mapping1][-1]
            elif mapping[key[0]] == key[1]:
                gain -= weight_dict[old_mapping1][key]
    if old_mapping2 in weight_dict:
        for key in weight_dict[old_mapping2]:
            if key == -1:
                gain -= weight_dict[old_mapping2][-1]
            # to avoid duplicate
            elif key[0] == node_id1:
                continue
            elif mapping[key[0]] == key[1]:
                gain -= weight_dict[old_mapping2][key]
    match_triple_dict[tuple(new_mapping_list)] = match_num + gain
    return gain


def get_best_gain(mapping, candidate_mappings, weight_dict, instance_len, cur_match_num):
    """
    Hill-climbing method to return the best gain swap/move can get
    Arguments:
    mapping: current node mapping
    candidate_mappings: the candidates mapping list
    weight_dict: the weight dictionary
    instance_len: the number of the nodes in AMR 2
    cur_match_num: current triple match number
    Returns:
    the best gain we can get via swap/move operation

    """
    largest_gain = 0
    # True: using swap; False: using move
    use_swap = True
    # the node to be moved/swapped
    node1 = None
    # store the other node affected. In swap, this other node is the node swapping with node1. In move, this other
    # node is the node node1 will move to.
    node2 = None
    # unmatched nodes in AMR 2
    unmatched = set(range(instance_len))
    # exclude nodes in current mapping
    # get unmatched nodes
    for nid in mapping:
        if nid in unmatched:
            unmatched.remove(nid)
    for i, nid in enumerate(mapping):
        # current node i in AMR 1 maps to node nid in AMR 2
        for nm in unmatched:
            if nm in candidate_mappings[i]:
                # remap i to another unmatched node (move)
                # (i, m) -> (i, nm)
                if veryVerbose:
                    print("Remap node", i, "from ", nid, "to", nm, file=DEBUG_LOG)
                mv_gain = move_gain(mapping, i, nid, nm, weight_dict, cur_match_num)
                if veryVerbose:
                    print("Move gain:", mv_gain, file=DEBUG_LOG)
                    new_mapping = mapping[:]
                    new_mapping[i] = nm
                    new_match_num = compute_match(new_mapping, weight_dict)
                    if new_match_num != cur_match_num + mv_gain:
                        print(mapping, new_mapping, file=ERROR_LOG)
                        print("Inconsistency in computing: move gain", cur_match_num, mv_gain, new_match_num,
                              file=ERROR_LOG)
                if mv_gain > largest_gain:
                    largest_gain = mv_gain
                    node1 = i
                    node2 = nm
                    use_swap = False
    # compute swap gain
    for i, m in enumerate(mapping):
        for j in range(i + 1, len(mapping)):
            m2 = mapping[j]
            # swap operation (i, m) (j, m2) -> (i, m2) (j, m)
            # j starts from i+1, to avoid duplicate swap
            if veryVerbose:
                print("Swap node", i, "and", j, file=DEBUG_LOG)
                print("Before swapping:", i, "-", m, ",", j, "-", m2, file=DEBUG_LOG)
                print(mapping, file=DEBUG_LOG)
                print("After swapping:", i, "-", m2, ",", j, "-", m, file=DEBUG_LOG)
            sw_gain = swap_gain(mapping, i, m, j, m2, weight_dict, cur_match_num)
            if veryVerbose:
                print("Swap gain:", sw_gain, file=DEBUG_LOG)
                new_mapping = mapping[:]
                new_mapping[i] = m2
                new_mapping[j] = m
                print(new_mapping, file=DEBUG_LOG)
                new_match_num = compute_match(new_mapping, weight_dict)
                if new_match_num != cur_match_num + sw_gain:
                    print(mapping, new_mapping, file=ERROR_LOG)
                    print("Inconsistency in computing: swap gain", cur_match_num, sw_gain, new_match_num,
                          file=ERROR_LOG)
            if sw_gain > largest_gain:
                largest_gain = sw_gain
                node1 = i
                node2 = j
                use_swap = True
    # generate a new mapping based on swap/move
    cur_mapping = mapping[:]
    if node1 is not None:
        if use_swap:
            if veryVerbose:
                print("Use swap gain", file=DEBUG_LOG)
            temp = cur_mapping[node1]
            cur_mapping[node1] = cur_mapping[node2]
            cur_mapping[node2] = temp
        else:
            if veryVerbose:
                print("Use move gain", file=DEBUG_LOG)
            cur_mapping[node1] = node2
    else:
        if veryVerbose:
            print("no move/swap gain found", file=DEBUG_LOG)
    if veryVerbose:
        print("Original mapping", mapping, file=DEBUG_LOG)
        print("Current mapping", cur_mapping, file=DEBUG_LOG)
    return largest_gain, cur_mapping


def print_alignment(mapping, instance1, instance2):
    """
    print the alignment based on a node mapping
    Args:
        mapping: current node mapping list
        instance1: nodes of AMR 1
        instance2: nodes of AMR 2

    """
    result = []
    for instance1_item, m in zip(instance1, mapping):
        r = instance1_item[1] + "(" + instance1_item[2] + ")"
        if m == -1:
            r += "-Null"
        else:
            instance2_item = instance2[m]
            r += "-" + instance2_item[1] + "(" + instance2_item[2] + ")"
        result.append(r)
    return " ".join(result)


def compute_f(match_num, test_num, gold_num):
    """
    Compute the f-score based on the matching triple number,
                                 triple number of AMR set 1,
                                 triple number of AMR set 2
    Args:
        match_num: matching triple number
        test_num:  triple number of AMR 1 (test file)
        gold_num:  triple number of AMR 2 (gold file)
    Returns:
        precision: match_num/test_num
        recall: match_num/gold_num
        f_score: 2*precision*recall/(precision+recall)
    """
    if test_num == 0 or gold_num == 0:
        return 0.00, 0.00, 0.00
    precision = float(match_num) / float(test_num)
    recall = float(match_num) / float(gold_num)
    if (precision + recall) != 0:
        f_score = 2 * precision * recall / (precision + recall)
        if veryVerbose:
            print("F-score:", f_score, file=DEBUG_LOG)
        return precision, recall, f_score
    else:
        if veryVerbose:
            print("F-score:", "0.0", file=DEBUG_LOG)
        return precision, recall, 0.00


def generate_amr_lines(f1, f2):
    """
    Read one AMR line at a time from each file handle
    :param f1: file handle (or any iterable of strings) to read AMR 1 lines from
    :param f2: file handle (or any iterable of strings) to read AMR 2 lines from
    :return: generator of cur_amr1, cur_amr2 pairs: one-line AMR strings
    """
    while True:
        cur_amr1 = amr.AMR.get_amr_line(f1)
        cur_amr2 = amr.AMR.get_amr_line(f2)
        if not cur_amr1 and not cur_amr2:
            pass
        elif not cur_amr1:
            print("Error: File 1 has less AMRs than file 2", file=ERROR_LOG)
            print("Ignoring remaining AMRs", file=ERROR_LOG)
        elif not cur_amr2:
            print("Error: File 2 has less AMRs than file 1", file=ERROR_LOG)
            print("Ignoring remaining AMRs", file=ERROR_LOG)
        else:
            yield cur_amr1, cur_amr2
            continue
        break


def get_amr_match(cur_amr1, cur_amr2, sent_num=1, justinstance=False, justattribute=False, justrelation=False):
    amr_pair = []
    for i, cur_amr in (1, cur_amr1), (2, cur_amr2):
        try:
            amr_pair.append(amr.AMR.parse_AMR_line(cur_amr))
        except Exception as e:
            print("Error in parsing amr %d: %s" % (i, cur_amr), file=ERROR_LOG)
            print("Please check if the AMR is ill-formatted. Ignoring remaining AMRs", file=ERROR_LOG)
            print("Error message: %s" % e, file=ERROR_LOG)
    amr1, amr2 = amr_pair
    prefix1 = "a"
    prefix2 = "b"
    # Rename node to "a1", "a2", .etc
    amr1.rename_node(prefix1)
    # Renaming node to "b1", "b2", .etc
    amr2.rename_node(prefix2)
    (instance1, attributes1, relation1) = amr1.get_triples()
    (instance2, attributes2, relation2) = amr2.get_triples()
    if verbose:
        print("AMR pair", sent_num, file=DEBUG_LOG)
        print("============================================", file=DEBUG_LOG)
        print("AMR 1 (one-line):", cur_amr1, file=DEBUG_LOG)
        print("AMR 2 (one-line):", cur_amr2, file=DEBUG_LOG)
        print("Instance triples of AMR 1:", len(instance1), file=DEBUG_LOG)
        print(instance1, file=DEBUG_LOG)
        print("Attribute triples of AMR 1:", len(attributes1), file=DEBUG_LOG)
        print(attributes1, file=DEBUG_LOG)
        print("Relation triples of AMR 1:", len(relation1), file=DEBUG_LOG)
        print(relation1, file=DEBUG_LOG)
        print("Instance triples of AMR 2:", len(instance2), file=DEBUG_LOG)
        print(instance2, file=DEBUG_LOG)
        print("Attribute triples of AMR 2:", len(attributes2), file=DEBUG_LOG)
        print(attributes2, file=DEBUG_LOG)
        print("Relation triples of AMR 2:", len(relation2), file=DEBUG_LOG)
        print(relation2, file=DEBUG_LOG)
    # optionally turn off some of the node comparison
    doinstance = doattribute = dorelation = True
    if justinstance:
        doattribute = dorelation = False
    if justattribute:
        doinstance = dorelation = False
    if justrelation:
        doinstance = doattribute = False
    (best_mapping, best_match_num) = get_best_match(instance1, attributes1, relation1,
                                                    instance2, attributes2, relation2,
                                                    prefix1, prefix2, doinstance=doinstance,
                                                    doattribute=doattribute, dorelation=dorelation)
    if verbose:
        print("best match number", best_match_num, file=DEBUG_LOG)
        print("best node mapping", best_mapping, file=DEBUG_LOG)
        print("Best node mapping alignment:", print_alignment(best_mapping, instance1, instance2), file=DEBUG_LOG)
    if justinstance:
        test_triple_num = len(instance1)
        gold_triple_num = len(instance2)
    elif justattribute:
        test_triple_num = len(attributes1)
        gold_triple_num = len(attributes2)
    elif justrelation:
        test_triple_num = len(relation1)
        gold_triple_num = len(relation2)
    else:
        test_triple_num = len(instance1) + len(attributes1) + len(relation1)
        gold_triple_num = len(instance2) + len(attributes2) + len(relation2)
    return best_match_num, test_triple_num, gold_triple_num


def score_amr_pairs(f1, f2, justinstance=False, justattribute=False, justrelation=False):
    """
    Score one pair of AMR lines at a time from each file handle
    :param f1: file handle (or any iterable of strings) to read AMR 1 lines from
    :param f2: file handle (or any iterable of strings) to read AMR 2 lines from
    :param justinstance: just pay attention to matching instances
    :param justattribute: just pay attention to matching attributes
    :param justrelation: just pay attention to matching relations
    :return: generator of cur_amr1, cur_amr2 pairs: one-line AMR strings
    """
    # matching triple number, triple number in test file, triple number in gold file
    total_match_num = total_test_num = total_gold_num = 0
    # Read amr pairs from two files
    for sent_num, (cur_amr1, cur_amr2) in enumerate(generate_amr_lines(f1, f2), start=1):
        best_match_num, test_triple_num, gold_triple_num = get_amr_match(cur_amr1, cur_amr2,
                                                                         sent_num=sent_num,  # sentence number
                                                                         justinstance=justinstance,
                                                                         justattribute=justattribute,
                                                                         justrelation=justrelation)
        total_match_num += best_match_num
        total_test_num += test_triple_num
        total_gold_num += gold_triple_num
        # clear the matching triple dictionary for the next AMR pair
        match_triple_dict.clear()
        if not single_score:  # if each AMR pair should have a score, compute and output it here
            yield compute_f(best_match_num, test_triple_num, gold_triple_num)
    if verbose:
        print("Total match number, total triple number in AMR 1, and total triple number in AMR 2:", file=DEBUG_LOG)
        print(total_match_num, total_test_num, total_gold_num, file=DEBUG_LOG)
        print("---------------------------------------------------------------------------------", file=DEBUG_LOG)
    if single_score:  # output document-level smatch score (a single f-score for all AMR pairs in two files)
        yield compute_f(total_match_num, total_test_num, total_gold_num)


def main(arguments):
    """
    Main function of smatch score calculation
    """
    global verbose
    global veryVerbose
    global iteration_num
    global single_score
    global pr_flag
    global match_triple_dict
    # set the iteration number
    # total iteration number = restart number + 1
    iteration_num = arguments.r + 1
    if arguments.ms:
        single_score = False
    if arguments.v:
        verbose = True
    if arguments.vv:
        veryVerbose = True
    if arguments.pr:
        pr_flag = True
    # significant digits to print out
    floatdisplay = "%%.%df" % arguments.significant
    for (precision, recall, best_f_score) in score_amr_pairs(args.f[0], args.f[1],
                                                             justinstance=arguments.justinstance,
                                                             justattribute=arguments.justattribute,
                                                             justrelation=arguments.justrelation):
        # print("Sentence", sent_num)
        if pr_flag:
            print("Precision: " + floatdisplay % precision)
            print("Recall: " + floatdisplay % recall)
        print("F-score: " + floatdisplay % best_f_score)
    args.f[0].close()
    args.f[1].close()


if __name__ == "__main__":
    parser = None
    args = None
    # use optparse if python version is 2.5 or 2.6
    if sys.version_info[0] == 2 and sys.version_info[1] < 7:
        import optparse

        if len(sys.argv) == 1:
            print("No argument given. Please run smatch.py -h to see the argument description.", file=ERROR_LOG)
            exit(1)
        parser = build_arg_parser2()
        (args, opts) = parser.parse_args()
        file_handle = []
        if args.f is None:
            print("smatch.py requires -f option to indicate two files \
                                             containing AMR as input. Please run smatch.py -h to  \
                                             see the argument description.", file=ERROR_LOG)
            exit(1)
        # assert there are 2 file names following -f.
        assert (len(args.f) == 2)
        for file_path in args.f:
            if not os.path.exists(file_path):
                print("Given file", args.f[0], "does not exist", file=ERROR_LOG)
                exit(1)
            file_handle.append(open(file_path))
        # use opened files
        args.f = tuple(file_handle)
    #  use argparse if python version is 2.7 or later
    else:
        import argparse

        parser = build_arg_parser()
        args = parser.parse_args()
    main(args)
