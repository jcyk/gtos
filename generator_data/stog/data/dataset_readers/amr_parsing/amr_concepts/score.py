class Score:
    def __init__(self, node, amr):
        self.node = node
        self.amr = amr
        self.alignment = self._get_alignment()
        self.span = self._get_best_span(self.alignment)
        self.ner_type = 'SCORE_ENTITY'

    def to_dict(self, amr, span):
        return {
            'type': 'score-entity',
            'span': ' '.join(map(amr.tokens.__getitem__, span)),
            'ops': self.node.ops}

    def _get_alignment(self):
        alignment = {}
        for i, op in enumerate(self.node.ops):
            for j, token in enumerate(self.amr.tokens):
                confidence = self._maybe_align(op, j)
                if confidence > 0:
                    if j not in alignment or alignment[j][1] < confidence:
                        alignment[j] = (i, confidence)
        return alignment

    def _maybe_align(self, op, index):
        op = str(op)
        if self.amr.lemmas[index] in (op, '-' + op):
            return 10
        return 0

    def _get_best_span(self, alignment):
        indexes = list(alignment.keys())
        indexes.sort()
        spans = []
        last_index = None
        for index in indexes:
            if last_index is None:
                spans.append([])
            elif index - last_index > 3:
                spans.append([])
            else:
                for i in range(last_index + 1, index):
                    if self.amr.lemmas[i] in ('-', 'to', ':', 'vote'):
                        spans[-1].append(index - 1)
                    else:
                        spans.append([])
                        break
            last_index = index
            spans[-1].append(index)
        if len(spans):
            return max(spans, key=lambda x: sum([alignment[i][1] for i in x if i in alignment]))
        else:
            return None

    @staticmethod
    def collapse_score_nodes(scores, amr):
        score_node_count = 0
        scores.sort(key=lambda score: score.span[-1] if score.span is not None else float('inf'))
        offset = 0
        for score in scores:
            if score.span is not None:
                score_node_count += 1
                abstract = '{}_{}'.format(score.ner_type, score_node_count)
                span = [index - offset for index in score.span]
                amr.abstract_map[abstract] = score.to_dict(amr, span)
                amr.replace_span(span, [abstract], ['NNP'], [score.ner_type])
                amr.stems = amr.stems[:span[0]] + [abstract] + amr.stems[span[-1] + 1:]
                amr.graph.remove_node_ops(score.node)
                amr.graph.replace_node_attribute(
                    score.node, 'instance', score.node.instance, abstract)
                offset += len(score.span) - 1
            else:
                amr.graph.remove_node(score.node)
