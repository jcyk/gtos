#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
AMR (Abstract Meaning Representation) structure
For detailed description of AMR, see http://www.isi.edu/natural-language/amr/a.pdf

"""

from __future__ import print_function
from collections import defaultdict
import sys

# change this if needed
ERROR_LOG = sys.stderr

# change this if needed
DEBUG_LOG = sys.stderr


class AMR(object):
    """
    AMR is a rooted, labeled graph to represent semantics.
    This class has the following members:
    nodes: list of node in the graph. Its ith element is the name of the ith node. For example, a node name
           could be "a1", "b", "g2", .etc
    node_values: list of node labels (values) of the graph. Its ith element is the value associated with node i in
                 nodes list. In AMR, such value is usually a semantic concept (e.g. "boy", "want-01")
    root: root node name
    relations: list of edges connecting two nodes in the graph. Each entry is a link between two nodes, i.e. a triple
               <relation name, node1 name, node 2 name>. In AMR, such link denotes the relation between two semantic
               concepts. For example, "arg0" means that one of the concepts is the 0th argument of the other.
    attributes: list of edges connecting a node to an attribute name and its value. For example, if the polarity of
               some node is negative, there should be an edge connecting this node and "-". A triple < attribute name,
               node name, attribute value> is used to represent such attribute. It can also be viewed as a relation.

    """
    def __init__(self, node_list=None, node_value_list=None, relation_list=None, attribute_list=None):
        """
        node_list: names of nodes in AMR graph, e.g. "a11", "n"
        node_value_list: values of nodes in AMR graph, e.g. "group" for a node named "g"
        relation_list: list of relations between two nodes
        attribute_list: list of attributes (links between one node and one constant value)

        """
        # initialize AMR graph nodes using list of nodes name
        # root, by default, is the first in var_list

        if node_list is None:
            self.nodes = []
            self.root = None
        else:
            self.nodes = node_list[:]
            if len(node_list) != 0:
                self.root = node_list[0]
            else:
                self.root = None
        if node_value_list is None:
            self.node_values = []
        else:
            self.node_values = node_value_list[:]
        if relation_list is None:
            self.relations = []
        else:
            self.relations = relation_list[:]
        if attribute_list is None:
            self.attributes = []
        else:
            self.attributes = attribute_list[:]

    def rename_node(self, prefix):
        """
        Rename AMR graph nodes to prefix + node_index to avoid nodes with the same name in two different AMRs.

        """
        node_map_dict = {}
        # map each node to its new name (e.g. "a1")
        for i in range(0, len(self.nodes)):
            node_map_dict[self.nodes[i]] = prefix + str(i)
        # update node name
        for i, v in enumerate(self.nodes):
            self.nodes[i] = node_map_dict[v]
        # update node name in relations
        for node_relations in self.relations:
            for i, l in enumerate(node_relations):
                node_relations[i][1] = node_map_dict[l[1]]
  
    def get_triples(self):
        """
        Get the triples in three lists.
        instance_triple: a triple representing an instance. E.g. instance(w, want-01)
        attribute triple: relation of attributes, e.g. polarity(w, - )
        and relation triple, e.g. arg0 (w, b)

        """
        instance_triple = []
        relation_triple = []
        attribute_triple = []
        for i in range(len(self.nodes)):
            instance_triple.append(("instance", self.nodes[i], self.node_values[i]))
            # l[0] is relation name
            # l[1] is the other node this node has relation with
            for l in self.relations[i]:
                relation_triple.append((l[0], self.nodes[i], l[1]))
            # l[0] is the attribute name
            # l[1] is the attribute value
            for l in self.attributes[i]:
                attribute_triple.append((l[0], self.nodes[i], l[1]))
        return instance_triple, attribute_triple, relation_triple


    def get_triples2(self):
        """
        Get the triples in two lists:
        instance_triple: a triple representing an instance. E.g. instance(w, want-01)
        relation_triple: a triple representing all relations. E.g arg0 (w, b) or E.g. polarity(w, - )
        Note that we do not differentiate between attribute triple and relation triple. Both are considered as relation
        triples.
        All triples are represented by (triple_type, argument 1 of the triple, argument 2 of the triple)

        """
        instance_triple = []
        relation_triple = []
        for i in range(len(self.nodes)):
            # an instance triple is instance(node name, node value).
            # For example, instance(b, boy).
            instance_triple.append(("instance", self.nodes[i], self.node_values[i]))
            # l[0] is relation name
            # l[1] is the other node this node has relation with
            for l in self.relations[i]:
                relation_triple.append((l[0], self.nodes[i], l[1]))
            # l[0] is the attribute name
            # l[1] is the attribute value
            for l in self.attributes[i]:
                relation_triple.append((l[0], self.nodes[i], l[1]))
        return instance_triple, relation_triple


    def __str__(self):
        """
        Generate AMR string for better readability

        """
        lines = []
        for i in range(len(self.nodes)):
            lines.append("Node "+ str(i) + " " + self.nodes[i])
            lines.append("Value: " + self.node_values[i])
            lines.append("Relations:")
            for relation in self.relations[i]:
                lines.append("Node " + relation[1] + " via " + relation[0])
            for attribute in self.attributes[i]:
                lines.append("Attribute: " + attribute[0] + " value " + attribute[1])
        return "\n".join(lines)

    def __repr__(self):
        return self.__str__()

    def output_amr(self):
        """
        Output AMR string

        """
        print(self.__str__(), file=DEBUG_LOG)

    @staticmethod
    def get_amr_line(input_f):
        """
        Read the file containing AMRs. AMRs are separated by a blank line.
        Each call of get_amr_line() returns the next available AMR (in one-line form).
        Note: this function does not verify if the AMR is valid

        """
        cur_amr = []
        has_content = False
        for line in input_f:
            line = line.strip()
            if line == "":
                if not has_content:
                    # empty lines before current AMR
                    continue
                else:
                    # end of current AMR
                    break
            if line.strip().startswith("#"):
                # ignore the comment line (starting with "#") in the AMR file
                continue
            else:
                has_content = True
                cur_amr.append(line.strip())
        return "".join(cur_amr)

    @staticmethod
    def parse_AMR_line(line):
        """
        Parse a AMR from line representation to an AMR object.
        This parsing algorithm scans the line once and process each character, in a shift-reduce style.

        """
        # Current state. It denotes the last significant symbol encountered. 1 for (, 2 for :, 3 for /,
        # and 0 for start state or ')'
        # Last significant symbol is ( --- start processing node name
        # Last significant symbol is : --- start processing relation name
        # Last significant symbol is / --- start processing node value (concept name)
        # Last significant symbol is ) --- current node processing is complete
        # Note that if these symbols are inside parenthesis, they are not significant symbols.
        
        exceptions =set(["prep-on-behalf-of", "prep-out-of", "consist-of"])
        def update_triple(node_relation_dict, triple):
            # we detect a relation (r) between u and v, with direction u to v.
            # in most cases, if relation name ends with "-of", e.g."arg0-of",
            # it is reverse of some relation. For example, if a is "arg0-of" b,
            # we can also say b is "arg0" a.
            # If the relation name ends with "-of", we store the reverse relation.
            # but note some exceptions like "prep-on-behalf-of" and "prep-out-of"
            # also note relation "mod" is the reverse of "domain"
            u, r, v = triple 
            if r.endswith("-of") and not r in exceptions:
                node_relation_dict[v].append((r[:-3], u))
            elif r=="mod":
                node_relation_dict[v].append(("domain", u))
            else:
                node_relation_dict[u].append((r, v))

        state = 0
        # node stack for parsing
        stack = []
        # current not-yet-reduced character sequence
        cur_charseq = []
        # key: node name value: node value
        node_dict = {}
        # node name list (order: occurrence of the node)
        node_name_list = []
        # key: node name:  value: list of (relation name, the other node name)
        node_relation_dict1 = defaultdict(list)
        # key: node name, value: list of (attribute name, const value) or (relation name, unseen node name)
        node_relation_dict2 = defaultdict(list)
        # current relation name
        cur_relation_name = ""
        # having unmatched quote string
        in_quote = False
        for i, c in enumerate(line.strip()):
            if c == " ":
                # allow space in relation name
                if state == 2:
                    cur_charseq.append(c)
                continue
            if c == "\"":
                # flip in_quote value when a quote symbol is encountered
                # insert placeholder if in_quote from last symbol
                if in_quote:
                    cur_charseq.append('_')
                in_quote = not in_quote
            elif c == "(":
                # not significant symbol if inside quote
                if in_quote:
                    cur_charseq.append(c)
                    continue
                # get the attribute name
                # e.g :arg0 (x ...
                # at this point we get "arg0"
                if state == 2:
                    # in this state, current relation name should be empty
                    if cur_relation_name != "":
                        print("Format error when processing ", line[0:i + 1], file=ERROR_LOG)
                        return None
                    # update current relation name for future use
                    cur_relation_name = "".join(cur_charseq).strip()
                    cur_charseq[:] = []
                state = 1
            elif c == ":":
                # not significant symbol if inside quote
                if in_quote:
                    cur_charseq.append(c)
                    continue
                # Last significant symbol is "/". Now we encounter ":"
                # Example:
                # :OR (o2 / *OR*
                #    :mod (o3 / official)
                #  gets node value "*OR*" at this point
                if state == 3:
                    node_value = "".join(cur_charseq)
                    # clear current char sequence
                    cur_charseq[:] = []
                    # pop node name ("o2" in the above example)
                    cur_node_name = stack[-1]
                    # update node name/value map
                    node_dict[cur_node_name] = node_value
                # Last significant symbol is ":". Now we encounter ":"
                # Example:
                # :op1 w :quant 30
                # or :day 14 :month 3
                # the problem is that we cannot decide if node value is attribute value (constant)
                # or node value (variable) at this moment
                elif state == 2:
                    temp_attr_value = "".join(cur_charseq)
                    cur_charseq[:] = []
                    parts = temp_attr_value.split()
                    if len(parts) < 2:
                        print("Error in processing; part len < 2", line[0:i + 1], file=ERROR_LOG)
                        return None
                    # For the above example, node name is "op1", and node value is "w"
                    # Note that this node name might not be encountered before
                    relation_name = parts[0].strip()
                    relation_value = parts[1].strip()
                    # We need to link upper level node to the current
                    # top of stack is upper level node
                    if len(stack) == 0:
                        print("Error in processing", line[:i], relation_name, relation_value, file=ERROR_LOG)
                        return None
                    # if we have not seen this node name before
                    if relation_value not in node_dict:
                        update_triple(node_relation_dict2, (stack[-1], relation_name, relation_value))
                    else:
                        update_triple(node_relation_dict1, (stack[-1], relation_name, relation_value))
                state = 2
            elif c == "/":
                if in_quote:
                    cur_charseq.append(c)
                    continue
                # Last significant symbol is "(". Now we encounter "/"
                # Example:
                # (d / default-01
                # get "d" here
                if state == 1:
                    node_name = "".join(cur_charseq)
                    cur_charseq[:] = []
                    # if this node name is already in node_dict, it is duplicate
                    if node_name in node_dict:
                        print("Duplicate node name ", node_name, " in parsing AMR", file=ERROR_LOG)
                        return None
                    # push the node name to stack
                    stack.append(node_name)
                    # add it to node name list
                    node_name_list.append(node_name)
                    # if this node is part of the relation
                    # Example:
                    # :arg1 (n / nation)
                    # cur_relation_name is arg1
                    # node name is n
                    # we have a relation arg1(upper level node, n)
                    if cur_relation_name != "":
                        update_triple( node_relation_dict1, (stack[-2], cur_relation_name, node_name))
                        cur_relation_name = ""
                else:
                    # error if in other state
                    print("Error in parsing AMR", line[0:i + 1], file=ERROR_LOG)
                    return None
                state = 3
            elif c == ")":
                if in_quote:
                    cur_charseq.append(c)
                    continue
                # stack should be non-empty to find upper level node
                if len(stack) == 0:
                    print("Unmatched parenthesis at position", i, "in processing", line[0:i + 1], file=ERROR_LOG)
                    return None
                # Last significant symbol is ":". Now we encounter ")"
                # Example:
                # :op2 "Brown") or :op2 w)
                # get \"Brown\" or w here
                if state == 2:
                    temp_attr_value = "".join(cur_charseq)
                    cur_charseq[:] = []
                    parts = temp_attr_value.split()
                    if len(parts) < 2:
                        print("Error processing", line[:i + 1], temp_attr_value, file=ERROR_LOG)
                        return None
                    relation_name = parts[0].strip()
                    relation_value = parts[1].strip()
                    # attribute value not seen before
                    # Note that it might be a constant attribute value, or an unseen node
                    # process this after we have seen all the node names
                    if relation_value not in node_dict:
                        update_triple(node_relation_dict2, (stack[-1], relation_name, relation_value))
                    else:
                        update_triple(node_relation_dict1, (stack[-1], relation_name, relation_value))
                # Last significant symbol is "/". Now we encounter ")"
                # Example:
                # :arg1 (n / nation)
                # we get "nation" here
                elif state == 3:
                    node_value = "".join(cur_charseq)
                    cur_charseq[:] = []
                    cur_node_name = stack[-1]
                    # map node name to its value
                    node_dict[cur_node_name] = node_value
                # pop from stack, as the current node has been processed
                stack.pop()
                cur_relation_name = ""
                state = 0
            else:
                # not significant symbols, so we just shift.
                cur_charseq.append(c)
        #create data structures to initialize an AMR
        node_value_list = []
        relation_list = []
        attribute_list = []
        for v in node_name_list:
            if v not in node_dict:
                print("Error: Node name not found", v, file=ERROR_LOG)
                return None
            else:
                node_value_list.append(node_dict[v])
            # build relation list and attribute list for this node
            node_rel_list = []
            node_attr_list = []
            if v in node_relation_dict1:
                for v1 in node_relation_dict1[v]:
                    node_rel_list.append([v1[0], v1[1]])
            if v in node_relation_dict2:
                for v2 in node_relation_dict2[v]:
                    # if value is in quote, it is a constant value
                    # strip the quote and put it in attribute map
                    if v2[1][0] == "\"" and v2[1][-1] == "\"":
                        assert True == False
                        node_attr_list.append([[v2[0]], v2[1][1:-1]])
                    # if value is a node name
                    elif v2[1] in node_dict:
                        node_rel_list.append([v2[0], v2[1]])
                    else:
                        node_attr_list.append([v2[0], v2[1]])
            # each node has a relation list and attribute list
            relation_list.append(node_rel_list)
            attribute_list.append(node_attr_list)
        # add TOP as an attribute. The attribute value is the top node value
        attribute_list[0].append(["TOP", node_value_list[0]])
        result_amr = AMR(node_name_list, node_value_list, relation_list, attribute_list)
        return result_amr

# test AMR parsing
# run by amr.py [file containing AMR]
# a unittest can also be used.
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("No file given", file=ERROR_LOG)
        exit(1)
    amr_count = 1
    for line in open(sys.argv[1]):
        cur_line = line.strip()
        if cur_line == "" or cur_line.startswith("#"):
            continue
        print("AMR", amr_count, file=DEBUG_LOG)
        current = AMR.parse_AMR_line(cur_line)
        current.output_amr()
        amr_count += 1
