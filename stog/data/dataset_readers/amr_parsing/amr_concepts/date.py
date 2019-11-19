import re
from collections import defaultdict


class Date:

    attribute_list = ['year', 'month', 'day', 'decade', 'time', 'century', 'era', 'timezone',
                      'quant', 'value', 'quarter', 'year2']

    edge_list = ['dayperiod', 'weekday']

    month_map = [
        ('January', 'Jan.', 'Jan'),
        ('February', 'Feb.', 'Feb', 'Febuary'),
        ('March', 'Mar.', 'Mar'),
        ('April', 'Apr.', 'Apr', 'Aril'),
        ('May',),
        ('June', 'Jun.', 'Jun'),
        ('July', 'Jul.', 'Jul'),
        ('August', 'Aug.', 'Aug'),
        ('September', 'Sep.', 'Sep', 'Sept.'),
        ('October', 'Oct.', 'Oct'),
        ('November', 'Nov.', 'Nov', 'Novmber'),
        ('December', 'Dec.', 'Dec')
    ]

    era_map = {
        'BC': ['BCE', 'bce'],
        'AD': ['CE', 'ce']
    }

    def __init__(self, node, graph):
        self.node = node
        self.attributes, self.edges = self._get_attributes_and_edges(node, graph)
        self.span = None
        self.confidence = 0
        self.ner_type = 'DATE_ATTRS'

    @staticmethod
    def collapsable(node, graph):
        if any(attr in Date.attribute_list for attr, _ in node.attributes):
            return True
        edges = list(graph._G.edges(node))
        for source, target in edges:
            if graph._G[source][target]['label'] in Date.edge_list:
                return True

    @staticmethod
    def collapse_date_nodes(dates, amr):
        if amr.abstract_map is None:
            amr.abstract_map = {}
        dates.sort(key=lambda date: date.span[-1] if date.span else float('inf'))
        offset = 0
        align_count = 0
        for date in dates:
            if date.span:
                align_count += 1
                abstract = '{}_{}'.format(date.ner_type, align_count)
                span_with_offset = [index - offset for index in date.span]
                amr.abstract_map[abstract] = Date.save_collapsed_date_node(
                    date, span_with_offset, amr)
                amr.replace_span(span_with_offset, [abstract], ['NNP'], [date.ner_type])
                # Remove edges
                for source, target in list(amr.graph._G.edges(date.node)):
                    edge_label = amr.graph._G[source][target]['label']
                    if edge_label in Date.edge_list:
                        amr.graph.remove_edge(source, target)
                        amr.graph.remove_subtree(target)
                # Update instance
                amr.graph.replace_node_attribute(date.node, 'instance', 'date-entity', abstract)
                # Remove attributes
                for attr, value in date.attributes.items():
                    amr.graph.remove_node_attribute(date.node, attr, value)
                offset += len(date.span) - 1
            else:
                for attr, value in date.attributes.items():
                    amr.graph.remove_node_attribute(date.node, attr, value)

    @staticmethod
    def save_collapsed_date_node(date, span, amr):
        return dict(
            type='date-entity',
            span=' '.join(map(amr.tokens.__getitem__, span)),
            attrs=date.attributes,
            edges=date.edges
        )

    def _get_attributes_and_edges(self, node, graph):
        attributes = {attr: value for attr, value in node.attributes if attr in self.attribute_list}
        edges = {}
        for source, target in graph._G.edges(node):
            label = graph._G[source][target]['label']
            if label in self.edge_list:
                edges[label] = target.instance
        return attributes, edges

    def _is_covered(self, alignment):
        attributes = self.attributes.copy()
        for index in self.span:
            if index in alignment:
                for item, _ in alignment[index]:
                    if item[0] in attributes:
                        attributes.pop(item[0])
        return len(attributes) == 0

    def _get_alignment(self, amr):
        alignment = defaultdict(list)
        for item in list(self.attributes.items()) + list(self.edges.items()):
            attr, value = item
            for i in range(len(amr.tokens)):
                confidence = self._maybe_align(attr, value, i, amr)
                if confidence != 0:
                    alignment[i].append((item, confidence))
        return alignment

    def _get_span(self, alignment, amr):
        spans = self.group_indexes_to_spans(alignment.keys(), amr)
        span_scores = []
        for span in spans:
            attr_set = set()
            for index in span:
                if index in alignment:
                    for item, _ in alignment[index]:
                        attr_set.add(item[0])
            span_scores.append((span, len(span), len(attr_set)))
        span_scores.sort(key=lambda x: (x[1], x[2]), reverse=True)
        spans = [span for span, _, _ in span_scores]
        if len(spans):
            self.span = max(spans, key=lambda span: sum(
                max(alignment[i], key=lambda x: x[1])[1] for i in span if i in alignment))
            self.confidence = sum(max(alignment[i], key=lambda x: x[1])[1] for i in self.span if i in alignment)

    def _clean_span(self, spans, alignment, amr):
        # Make sure each op only appears once in a span.
        clean_spans = []
        for span in spans:
            _align = {}
            trivial_indexes = []
            for index in span:
                if index not in alignment:
                    trivial_indexes.append(index)
                    continue
                for item, confidence in alignment[index]:
                    if item not in _align or _align[item][1] < confidence:
                        _align[item] = (index, confidence)
            indexes = [i for i, _ in _align.values()] + trivial_indexes
            _spans = self.group_indexes_to_spans(indexes, amr)
            clean_spans.append(max(_spans, key=lambda s: len(s)))
        return clean_spans

    def _maybe_align(self, attr, value, index, amr):
        if attr == 'year':
            return self._maybe_align_year(value, index, amr)
        elif attr == 'month':
            return self._maybe_align_month(value, index, amr)
        elif attr == 'day':
            return self._maybe_align_day(value, index, amr)
        elif attr == 'decade':
            return self._maybe_align_decade(value, index, amr)
        elif attr == 'time':
            return self._maybe_align_time(value, index, amr)
        elif attr == 'century':
            return self._maybe_align_century(value, index, amr)
        elif attr == 'era':
            return self._maybe_align_era(value, index, amr)
        elif attr == 'quant':
            return self._maybe_align_quant(value, index, amr)
        elif attr == 'quarter':
            return self._maybe_align_quarter(value, index, amr)
        elif attr == 'weekday':
            return self._maybe_align_weekday(value, index, amr)
        else:
            return self._maybe_align_basic(value, index, amr)

    def _maybe_align_basic(self, value, index, amr):
        value = str(value).lower()
        if re.search(r'^".*"$', value):
            value = value[1:-1]
        if amr.tokens[index].lower() == value or amr.lemmas[index].lower() == value:
            return 10
        stripped_lemma = self._strip_date_lemma(amr.lemmas[index])
        if stripped_lemma == value:
            return 10
        elif value.endswith(stripped_lemma):
            return 8
        return 0

    def _strip_date_lemma(self, lemma):
        # Remove '-'.
        if len(lemma) and lemma[0] == '-':
            lemma = lemma[1:]
        if len(lemma) and lemma[-1] == '-':
            lemma = lemma[:-1]
        return lemma

    def _maybe_align_year(self, value, index, amr):
        basic_confidence = self._maybe_align_basic(value, index, amr)
        if basic_confidence != 0:
            return basic_confidence
        lemma = amr.lemmas[index]
        stripped_lemma = self._strip_date_lemma(lemma)
        year_short = str(value)[-2:]
        if ':' not in lemma and (lemma.startswith(year_short) or stripped_lemma.startswith(year_short)):
            return 10
        year_with_s = str(value) + 's'
        if year_with_s == lemma:
            return 10
        year_with_stroke = "'" + year_short
        if year_with_stroke == lemma:
            return 10
        return 0

    def _maybe_align_month(self, value, index, amr):
        lemma = amr.lemmas[index]
        if 0 < value < 13:
            for month in self.month_map[value - 1]:
                if month.lower() == lemma.lower():
                    return 15
        basic_confidence = self._maybe_align_basic(value, index, amr)
        if basic_confidence != 0:
            return basic_confidence
        month_fixed_length = '{:02d}'.format(value)
        if month_fixed_length == lemma:
            return 10
        return 0

    def _maybe_align_day(self, value, index, amr):
        basic_confidence = self._maybe_align_basic(value, index, amr)
        if basic_confidence != 0:
            return basic_confidence
        lemma = amr.lemmas[index]
        day_fixed_length = '{:02d}'.format(value)
        if day_fixed_length == lemma:
            return 10
        day = str(value)
        if (day + 'th' == lemma or
                day + 'st' == lemma or
                day + 'nd' == lemma or
                day + 'rd' == lemma or
                day + 'sr' == lemma
        ):
            return 10
        if value == 1 and lemma == 'first':
            return 8
        if value == 2 and lemma == 'second':
            return 8
        if value == 3 and lemma == 'third':
            return 8
        return 0

    def _maybe_align_decade(self, value, index, amr):
        basic_confidence = self._maybe_align_basic(value, index, amr)
        if basic_confidence != 0:
            return basic_confidence
        year_confidence = self._maybe_align_year(value, index, amr)
        if year_confidence != 0:
            return year_confidence
        try:
            w2n_number = str(w2n.word_to_num(amr.lemmas[index]))
        except:
            w2n_number = None
        decade_short = str(value)[-2:]
        lemma = amr.lemmas[index]
        if lemma.endswith(decade_short + 's'):
            return 10
        if decade_short == w2n_number:
            return 10
        return 0

    def _maybe_align_time(self, value, index, amr):
        if re.search(r'^".*"$', value):
            value = value[1:-1]
        basic_confidence = self._maybe_align_basic(value, index, amr)
        if basic_confidence != 0:
            return basic_confidence
        lemma = amr.lemmas[index]
        m = re.search(r'^(\d{2})(\d{2})(\d{0,2})$', lemma)
        if m is None or m.group(1) == '':
            m = re.search(r'-?(\d{1,2})[:.]?(\d{0,2})[:.]?(\d{0,2}).*$', lemma)
            if m is None or m.group(1) == '':
                return 0
        _hour, _min, _sec = m.group(1), m.group(2), m.group(3)
        m = re.search(r'(\d*)[:.]?(\d*)[:.]?(\d*)$', value)
        hour, min, sec = m.group(1), m.group(2), m.group(3)
        hour_in_12 = str(int(hour) - 12)
        if ((_sec == '00' or _sec == '') and (sec == '00' or sec == '')) or _sec == sec:
            if ((_min == '00' or _min == '') and (min == '00' or min == '')) or _min == min:
                if _hour == hour or int(_hour) == int(hour):
                    return 10
                elif _hour == hour_in_12 or int(_hour) == int(hour_in_12):
                    return 8
                elif _hour == '12' and hour == '0':
                    return 8
        return 0

    def _maybe_align_century(self, value, index, amr):
        value = str(value)
        basic_confidence = self._maybe_align_basic(value, index, amr)
        if basic_confidence != 0:
            return basic_confidence
        lemma = amr.lemmas[index]
        m = re.search(r'^(\d+)(st|nd|rd|th)?(century)?$', lemma)
        if m and m.group(1) == value:
            return 10
        return 0

    def _maybe_align_era(self, value, index, amr):
        value = str(value)
        basic_confidence = self._maybe_align_basic(value, index, amr)
        if basic_confidence != 0:
            return basic_confidence
        lemma = amr.lemmas[index].replace('.', '')
        if re.search(r'^".*"$', value):
            value = value[1:-1]
        if value == lemma or lemma in self.era_map.get(value, []):
            return 10
        return 0

    def _maybe_align_quant(self, value, index, amr):
        value = str(value)
        basic_confidence = self._maybe_align_basic(value, index, amr)
        if basic_confidence != 0:
            return basic_confidence
        lemma = amr.lemmas[index]
        if value == '1' and (lemma == 'one' or lemma == 'a'):
            return 8
        return 0

    def _maybe_align_quarter(self, value, index, amr):
        value = str(value)
        basic_confidence = self._maybe_align_basic(value, index, amr)
        if basic_confidence != 0:
            return basic_confidence
        lemma = amr.lemmas[index]
        try:
            w2n_number = str(w2n.word_to_num(lemma))
        except:
            w2n_number = None
        if w2n_number == lemma:
            return 8
        return 0

    def _maybe_align_weekday(self, value, index, amr):
        value = str(value).lower()
        basic_confidence = self._maybe_align_basic(value, index, amr)
        if basic_confidence != 0:
            return basic_confidence
        lemma = amr.lemmas[index].lower()
        if value.startswith(lemma) or lemma.startswith(value) or lemma.endswith(value):
            return 10
        return 0

    def group_indexes_to_spans(self, indexes, amr):
        indexes = list(indexes)
        indexes.sort()
        spans = []
        last_index = None
        for idx in indexes:
            if last_index is None or idx - last_index > 3:
                spans.append([])
            elif idx - last_index <= 3:
                for i in range(last_index + 1, idx):
                    if re.search(r"(,|'s|of|'|-|in|at|on|about|the|every|\(|\))", amr.tokens[idx - 1]):
                        continue
                    else:
                        spans.append([])
                        break
                else:
                    spans[-1] += list(range(last_index + 1, idx))
            last_index = idx
            spans[-1].append(idx)
        return spans

