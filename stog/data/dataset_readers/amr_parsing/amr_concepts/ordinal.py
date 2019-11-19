import re


class Alignment:

    def __init__(self, token_index, op_index, score):
        self.token_index = token_index
        self.op_index = op_index
        self.score = score


class Ordinal:

    DIGIT_MAP = {
        '1': 'one',
        '3': 'three',
        '-2': 'second',
        '-4': 'four',
        '-6': 'six'
    }

    ORDINAL_MAP = {
        'first': '1',
        'firstly': '1',
        'second': '2',
        'secondly': '2',
        'twice': '2',
        'ii': '2',
        'third': '3',
        'fourth': '4',
        'fifth': '5',
        'sixth': '6',
        'seventh': '7',
        'eighth': '8',
        'ninth': '9',
        'tenth': '10',
        'twelfth': '12',
        'centennial': '100'
    }

    def __init__(self, node, amr, align=True):
        self.node = node
        self.amr = amr
        self.value_node = None
        self.ops = self._get_ops()
        if align:
            self.alignment = self._get_alignment()
            self.span = self._get_best_span(self.alignment)
        self.ner_type = 'ORDINAL_ENTITY'

    def to_dict(self, amr, span):
        return {
            'type': 'ordinal-entity',
            'span': ' '.join(map(amr.tokens.__getitem__, span)),
            'ops': self.ops}

    def _get_ops(self):
        for attr, value in self.node.attributes:
            if attr == 'value':
                return [str(value)]
        # The ordinal value is not an attribute, try to find it in the children.
        value = None
        graph = self.amr.graph
        edges = list(graph._G.edges(self.node))
        for source, target in edges:
            label = graph._G[source][target]['label']
            if label == 'value':
                value = target
                break
        if value is None:
            return []
        self.value_node = value
        return list(map(str, value.ops))

    def _get_alignment(self):
        alignment = {}
        for i, op in enumerate(self.ops):
            if re.search(r'^".*"$', op):
                op = op[1: -1]
            for j in range(len(self.amr.tokens)):
                alignment_score = self._maybe_align(op, j)
                if alignment_score == 0:
                    continue
                coherence_score = self._get_coherence(j)
                score = (alignment_score, coherence_score)
                if j not in alignment or alignment[j].score < score:
                    alignment[j] = Alignment(j, i, score)
        return alignment

    def _get_coherence(self, i):
        return 0

    def _maybe_align(self, op, index):
        lemma = self.amr.lemmas[index].lower().replace(',', '')
        if op == lemma:
            return 10
        if op + 'th' == lemma or op + 'rd' == lemma or op + 'nd' == lemma or op + 'st' == lemma:
            return 10
        if op == '-1' and lemma in ('last', 'mast', 'final', 'lastly'):
            return 10
        if op == '-4' and lemma == 'preantepenultimate':
            return 10
        if op == '2' and lemma == 'latter':
            return 8
        if lemma in self.ORDINAL_MAP and self.ORDINAL_MAP[lemma] == op:
            return 10
        if lemma.startswith('-') and lemma[1:] == op:
            return 8
        if op in self.DIGIT_MAP and self.DIGIT_MAP[op] == lemma:
            return 8
        return 0

    def _get_best_span(self, alignment):
        indexes = list(alignment.keys())
        indexes.sort()
        spans = []
        last_index = None
        for index in indexes:
            if last_index is None:
                spans.append([])
            elif index - last_index > 2:
                spans.append([])
            else:
                for i in range(last_index + 1, index):
                    if self.amr.lemmas[i] in ('-', 'to'):
                        spans[-1].append(index - 1)
                    else:
                        spans.append([])
                        break
            last_index = index
            spans[-1].append(index)
        if len(spans):
            return max(spans, key=lambda x: (
                sum([alignment[j].score[0] for j in x if j in alignment],
                    sum([alignment[j].score[1] for j in x if j in alignment]))))
        else:
            return None

    @staticmethod
    def collapse_ordinal_nodes(ordinals, amr):
        node_count = 0
        ordinals.sort(key=lambda x: x.span[-1] if x.span is not None else float('inf'))
        offset = 0
        for ordinal in ordinals:
            if ordinal.span is not None:
                node_count += 1
                abstract = '{}_{}'.format(ordinal.ner_type, node_count)
                span = [index - offset for index in ordinal.span]
                amr.abstract_map[abstract] = ordinal.to_dict(amr, span)
                amr.replace_span(span, [abstract], ['JJ'], [ordinal.ner_type])
                amr.stems = amr.stems[:span[0]] + [abstract] + amr.stems[span[-1] + 1:]
                for attr, value in ordinal.node.attributes:
                    if attr == 'value':
                        amr.graph.remove_node_attribute(ordinal.node, attr, value)
                        break
                amr.graph.replace_node_attribute(
                    ordinal.node, 'instance', ordinal.node.instance, abstract)
                if ordinal.value_node:
                    # Remove the value node.
                    amr.graph.remove_edge(ordinal.node, ordinal.value_node)
                    amr.graph.remove_subtree(ordinal.value_node)
                offset += len(ordinal.span) - 1
            else:
                edges = list(amr.graph._G.in_edges(ordinal.node))
                for source, target in edges:
                    amr.graph.remove_edge(source, target)
                    amr.graph.remove_subtree(target)
