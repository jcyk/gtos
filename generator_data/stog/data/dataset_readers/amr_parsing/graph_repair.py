from collections import defaultdict

from stog.utils import logging


logger = logging.init_logger()


def is_similar(instances1, instances2):
    if len(instances1) < len(instances2):
        small = instances1
        large = instances2
    else:
        small = instances2
        large = instances1
    coverage1 = sum(1 for x in small if x in large) / len(small)
    coverage2 = sum(1 for x in large if x in small) / len(large)
    return coverage1 > .8 and coverage2 > .8


class GraphRepair:

    def __init__(self, graph):
        self.graph = graph
        self.repaired_items = set()

    @staticmethod
    def do(graph):
        gr = GraphRepair(graph)
        gr.remove_redundant_edges()
        gr.remove_unknown_nodes()

    def remove_unknown_nodes(self):
        graph = self.graph
        nodes = [node for node in graph.get_nodes()]
        for node in nodes:
            for attr, value in node.attributes:
                if value == '@@UNKNOWN@@' and attr != 'instance':
                    graph.remove_node_attribute(node, attr, value)
            if node.instance == '@@UNKNOWN@@':
                if len(list(graph._G.edges(node))) == 0:
                    for source, target in list(graph._G.in_edges(node)):
                        graph.remove_edge(source, target)
                    graph.remove_node(node)
                    self.repaired_items.add('remove-unknown-node')

    def remove_redundant_edges(self):
        """
        Edge labels such as ARGx, ARGx-of, and 'opx' should only appear at most once
        in each node's outgoing edges.
        """
        graph = self.graph
        nodes = [node for node in graph.get_nodes()]
        removed_nodes = set()
        for node in nodes:
            if node in removed_nodes:
                continue
            edges = list(graph._G.edges(node))
            edge_counter = defaultdict(list)
            for source, target in edges:
                label = graph._G[source][target]['label']
                # `name`, `ARGx`, and `ARGx-of` should only appear once.
                if label == 'name':  # or label.startswith('ARG'):
                    edge_counter[label].append(target)
                # the target of `opx' should only appear once.
                elif label.startswith('op') or label.startswith('snt'):
                    edge_counter[str(target.instance)].append(target)
                else:
                    edge_counter[label + str(target.instance)].append(target)
            for label, children in edge_counter.items():
                if len(children) == 1:
                    continue
                if label == 'name':
                    # remove redundant edges.
                    for target in children[1:]:
                        if len(list(graph._G.in_edges(target))) == 1 and len(list(graph._G.edges(target))) == 0:
                            graph.remove_edge(node, target)
                            graph.remove_node(target)
                            removed_nodes.add(target)
                            self.repaired_items.add('remove-redundant-edge')
                    continue
                visited_children = set()
                groups = []
                for i, target in enumerate(children):
                    if target in visited_children:
                        continue
                    subtree_instances1 = [n.instance for n in graph.get_subtree(target, 5)]
                    group = [(target, subtree_instances1)]
                    visited_children.add(target)
                    for _t in children[i + 1:]:
                        if _t in visited_children or target.instance != _t.instance:
                            continue
                        subtree_instances2 = [n.instance for n in graph.get_subtree(_t, 5)]
                        if is_similar(subtree_instances1, subtree_instances2):
                            group.append((_t, subtree_instances2))
                            visited_children.add(_t)
                    groups.append(group)
                for group in groups:
                    if len(group) == 1:
                        continue
                    kept_target, _ = max(group, key=lambda x: len(x[1]))
                    for target, _ in group:
                        if target == kept_target:
                            continue
                        graph.remove_edge(node, target)
                        removed_nodes.update(graph.remove_subtree(target))
