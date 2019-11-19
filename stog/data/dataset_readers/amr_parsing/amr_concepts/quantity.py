import re


def quantify(x):
    if isinstance(x, int) or isinstance(x, float):
        return str(x)
    else:
        if re.search(r"^[0-9]+/[0-9]+$", x):
            numerator, denominator = x.split('/')
            return str(int(numerator) / int(denominator))
        elif re.search(r"^[0-9]+$", x):
            return x
        else:
            return None


class AlignedPairs:

    def __init__(self, quant_tokens, x, y, amr, score=0, near=-float('inf')):
        self.quant_tokens = quant_tokens
        self.quant_token_index = x
        self.snt_token_index = y
        self.amr = amr
        self.score = score
        self.near = near

    def __str__(self):
        return '{}: {}({})'.format(
            self.quant_tokens[self.quant_token_index],
            self.amr.tokens[self.snt_token_index],
            self.snt_token_index
        )


class Alignment:

    def __init__(self, node, attr, value):
        self.node = node
        self.attr = attr
        self.value = value
        self.aligned_pairs = []
        self.backup = []

    def __str__(self):
        return str(list(map(str, self.aligned_pairs)))

    @property
    def score(self):
        if len(self.aligned_pairs) == 0:
            return 0
        return sum(p.score for p in self.aligned_pairs)

    @property
    def near(self):
        if len(self.aligned_pairs) == 0:
            return -float('inf')
        return sum(p.near for p in self.aligned_pairs)

    @property
    def begin(self):
        if len(self.aligned_pairs) == 0:
            return -1
        return min(p.snt_token_index for p in self.aligned_pairs)

    @property
    def end(self):
        if len(self.aligned_pairs) == 0:
            return -1
        return max(p.snt_token_index for p in self.aligned_pairs) + 1

    @property
    def span(self):
        return range(self.begin, self.end)

    def has_overlap_with(self, other):
        index_list1 = [p.snt_token_index for p in self.aligned_pairs]
        return any(p.snt_token_index in index_list1 for p in other.aligned_pairs)


class QuantityCounter:

    def __init__(self):
        self.one = 0
        self.ten = 0
        self.hundred = 0
        self.thousand = 0

    def get_count(self, value):
        value = float(quantify(value))
        if 0 <= value < 10:
            self.one += 1
            return self.one
        elif 10 <= value < 100:
            self.ten += 10
            return self.ten
        elif 100 <= value < 1000:
            self.hundred += 100
            return self.hundred
        elif value >= 1000:
            self.thousand += 1000
            return self.thousand
        else:
            value


class Quantity:

    normalize_dict = {
        '1': ['one'],
        '2': ['two'],
        '3': ['three'],
        '4': ['four'],
        '5': ['five'],
        '6': ['six'],
        '7': ['seven'],
        '8': ['eight'],
        '9': ['nine'],
        '10': ['ten'],
        '12': ['dozen'],
        '20': ['twenty'],
        '30': ['thirty'],
        '50': ['fifty'],
        '100': ['hundred'],
        '200': ['two', 'hundred'],
        '1000': ['thousand'],
        '10000': ['ten', 'of', 'thousand'],
        '100000': ['hundred', 'of', 'thousand'],
        '10000000': ['ten', 'million'],
        '1000000000': ['billion'],
        '2.5': ['2', 'and', 'a', 'half'],
        '7.5': ['seven', 'and', 'a', 'half'],
        '6.5': ['six', 'and', 'a', 'half'],
    }

    def __init__(self, amr, dry=False):
        self.amr = amr
        self.dry = dry
        self.alignments = []
        self.ordered_node_list = [n for n, _, _ in amr.graph.get_list_node()]
        self.quant_count = 0

    def abstract(self, align=True):
        if not align:
            return self._abstract_without_alignment()
        graph = self.amr.graph
        for node in graph.get_nodes():
            if node.copy_of is not None:
                continue
            self.align_node_attrs(node)
        groups = self.group_alignment()
        return self.abstract_group(groups)

    def _abstract_without_alignment(self):
        graph = self.amr.graph
        
        count = 0
        counter = QuantityCounter()
        for node in graph.get_nodes():
            if node.copy_of is not None:
                continue
            for attr, value in node.attributes:
                q = quantify(value)
                if q is None:
                    continue
                span = []
                abstract = str(counter.get_count(value))
                self.amr.abstract_map[abstract] = dict(
                    type='quantity',
                    span=' '.join(map(self.amr.tokens.__getitem__, span)),
                    value=value)
                self.amr.graph.replace_node_attribute(node, attr, value, abstract)
                count += 1
        return count


    def align_node_attrs(self, node):
        node_position = self.get_node_position(node)
        for attr, value in node.attributes:
            q = quantify(value)
            if q is None:
                continue
            self.quant_count += 1
            alignment, backup = self.get_alignment([q], node_position, node, attr, value)
            quant_tokens = self.normalize_quant(q)
            if quant_tokens is not None:
                alignment2, backup2 = self.get_alignment(
                    quant_tokens, node_position, node, attr, value)
                if (alignment2.score, alignment2.near) > (alignment.score, alignment.near):
                    backup = [alignment] + backup + backup2
                    alignment = alignment2
                else:
                    backup = [alignment2] + backup + backup2
                backup.sort(key=lambda x: (-x.score, -x.near))
            alignment.backup = backup
            if alignment.score > 0:
                self.alignments.append(alignment)
            else:
                continue

    def group_alignment(self):
        groups = []
        visited = set()
        for i, x in enumerate(self.alignments):
            if i in visited:
                continue
            visited.add(i)
            group = [x]
            for j, y in enumerate(self.alignments[i + 1:], i + 1):
                if j in visited:
                    continue
                if x.has_overlap_with(y):
                    visited.add(j)
                    group.append(y)
            groups.append(group)
            if len(group) > 1:
                continue
        return groups

    def abstract_group(self, groups):
        if len(groups) == 0:
            return 0
        count, offset = 0, 0
        counter = QuantityCounter()
        representatives = [max(g, key=lambda x: x.end - x.begin) for g in groups]
        groups, representatives = zip(*sorted(
            zip(groups, representatives), key=lambda x: (x[1].begin, x[1].end)))
        for i, alignment in enumerate(representatives):
            abstract = str(counter.get_count(alignment.value))
            span = [index - offset for index in alignment.span]
            offset += len(span) - 1
            self.amr.abstract_map[abstract] = dict(
                type='quantity',
                span=' '.join(map(self.amr.tokens.__getitem__, span)),
                value=alignment.value)
            pos_tag = self.amr.pos_tags[span[0]]
            if pos_tag in ('0', 'O'):
                pos_tag = 'CD'
            self.amr.replace_span(span, [abstract], [pos_tag], ['NUMBER'])
            for a in groups[i]:
                count += 1
                self.amr.graph.replace_node_attribute(a.node, a.attr, a.value, abstract)#
        return count

    def get_alignment(self, tokens, node_position, node, attr, value):
        candidate_alignments = []
        for start in range(len(self.amr.tokens) - len(tokens) + 1):
            alignment = Alignment(node, attr, value)
            for i, index in enumerate(range(start, start + len(tokens))):
                score = self.maybe_align(index, tokens[i])
                near = -float('inf')
                if node_position != -1:
                    near = -abs(node_position - index)
                if score > 0:
                    alignment.aligned_pairs.append(
                        AlignedPairs(tokens, i, index, self.amr, score, near))
            candidate_alignments.append(alignment)
        candidate_alignments.sort(key=lambda x: (-x.score, -x.near))
        return candidate_alignments[0], candidate_alignments[1:]

    def get_node_position(self, node):
        lemmas = self.amr.lemmas
        node_lemma = re.sub(r'-\d+$', '', str(node.instance))
        position = -1
        if node_lemma in lemmas:
            position = lemmas.index(node_lemma)
        if position == -1:
            for _, child in self.amr.graph._G.edges(node):
                if self.amr.graph._G[node][child]['label'] == 'unit':
                    instance = child.instance
                    if instance in lemmas:
                        position = lemmas.index(instance)
                        break
        if position == -1:
            position = self.ordered_node_list.index(node)
        return position

    def normalize_quant(self, q):
        quant_tokens = self.normalize_dict.get(q, None)
        if quant_tokens is None and not re.search(r'[./]', q):
            billion = int(q) / 1000000000
            if int(q) % 1000000000 == 0:
                billion = int(billion)
            million = int(q) / 1000000
            if int(q) % 1000000 == 0:
                million = int(million)
            thousand = 0
            if int(q) % 1000 == 0:
                thousand = int(int(q) / 1000)
            hundred = 0
            if int(q) % 100 == 0:
                hundred = int(int(q) / 100)
            if billion >= 1:
                quant_tokens = [str(billion), 'billion']
            elif million >= 1:
                quant_tokens = [str(million), 'million']
            elif 10 > thousand > 0:
                quant_tokens = [str(thousand), 'thousand']
            elif 10 <= thousand < 1000:
                quant_tokens = [str(thousand) + 'k']
            elif 10 > hundred > 0:
                quant_tokens = [str(hundred), 'hundred']

        if quant_tokens is None and q.startswith('-'):
            quant_tokens = [q[0], q[1:]]
        return quant_tokens

    def maybe_align(self, index, token):
        lemma = self.amr.lemmas[index].replace(',', '').lower()
        if lemma == token:
            return 10
        if lemma == token + 's' or lemma == token + 'th':
            return 8
        if lemma == 'per' and token == '1':
            return 8
        if lemma in ('firstly',) and token == '1':
            return 5
        if lemma == 'minus' and token == '-':
            return 8
        if re.search(r'^\d+\.\d+$', lemma) and re.search(r'^\d+\.\d+$', token) and float(lemma) == float(token):
            return 10
        if lemma == 'secondly' and token == '2':
            return 5
        if lemma == 'lastly' and token == '-1':
            return 5
        if lemma == '.' + token:
            return 5
        return 0

