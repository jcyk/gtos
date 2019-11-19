import re


class Alignment:

    def __init__(self, node, url, amr, indexes, score):
        self.node = node
        self.url = url
        self.amr = amr
        self.aligned_token_indexes = indexes
        self.score = score

    def __str__(self):
        return 'node: {}\nurl: {}\naligned_token_indexes: {}\naligned_tokens: {}'.format(
            self.node, self.url, self.aligned_token_indexes,
            ' '.join([self.amr.lemmas[i] for i in self.aligned_token_indexes]))

    @property
    def begin(self):
        if len(self.aligned_token_indexes) == 0:
            return -1
        return min(self.aligned_token_indexes)

    @property
    def end(self):
        if len(self.aligned_token_indexes) == 0:
            return -1
        return max(self.aligned_token_indexes)


class URL:

    def __init__(self, amr, dry=False):
        self.amr = amr
        self.dry = dry
        self.alignments = []

    def abstract(self, align=True):
        url_count = 0
        graph = self.amr.graph
        for node, _, _ in graph.get_list_node(replace_copy=False):
            if node.copy_of is not None:
                continue
            if node.instance == 'url-entity':
                url_count += 1
                url_value = self.get_url_value(node)
                if url_value is None:
                    continue
                if align:
                    alignment = self.align_url(node, url_value)
                else:
                    alignment = Alignment(node, url_value, self.amr, [], 1) 
                if alignment is not None and alignment.score > 0:
                    self.alignments.append(alignment)
        if align:
            abstract_count = self.abstract_url()
            self.remove_redundant_url()
        else:
            abstract_count = self.abstract_url_without_alignment()
        return url_count, abstract_count

    def abstract_url_without_alignment(self):
        count = 0
        for i, alignment in enumerate(self.alignments):
            count += 1
            abstract = 'URL_{}'.format(i + 1)
            span = []
            self.amr.abstract_map[abstract] = dict(
                type='url-entity',
                span=' '.join(map(self.amr.tokens.__getitem__, span)),
                value=alignment.url)
            self.amr.graph.replace_node_attribute(alignment.node, 'value', alignment.url, abstract)
        return count

    def get_url_value(self, node):
        for attr, value in node.attributes:
            if attr == 'value':
                assert re.search(r'^".*"$', value)
                return value
        try:
            return self.fix_url_node(node)
        except:
            return None

    def fix_url_node(self, node):
        name_node = list(self.amr.graph._G[node].items())[0][0]
        url = name_node.ops[0]
        self.amr.graph.remove_edge(node, name_node)
        self.amr.graph.remove_subtree(name_node)
        self.amr.graph.add_node_attribute(node, 'value', url)
        return url

    def align_url(self, node, url):
        candidate_alignments = []
        for index in range(len(self.amr.tokens)):
            score = self.maybe_align(index, url)
            if score > 0:
                alignment = Alignment(node, url, self.amr, [index], score)
                candidate_alignments.append(alignment)
        if len(candidate_alignments) == 0:
            return None
        candidate_alignments.sort(key=lambda x: -x.score)
        return candidate_alignments[0]

    def maybe_align(self, index, url):
        url = re.sub(r'http:', 'https:', url[1:-1].lower().replace(' ', '-'))
        lemma = self.amr.lemmas[index]
        lemma = re.sub(r'http:', 'https:', lemma.lower())
        if url == lemma:
            return 10
        if url in lemma:
            return 9
        elif url[:20] in lemma or url[-20:] in lemma:
            return 8
        elif url[:10] in lemma or url[-10:] in lemma:
            return 7
        elif url[:5] in lemma or url[-5:] in lemma:
            return 6
        elif url == 'https://www.christianforums.com' and lemma in ('cf', 'cfer'):
            return 10
        return 0

    def abstract_url(self):
        count, offset = 0, 0
        self.alignments.sort(key=lambda x: x.end)
        for i, alignment in enumerate(self.alignments):
            count += 1
            abstract = 'URL_{}'.format(i + 1)
            span = [index - offset for index in alignment.aligned_token_indexes]
            offset += len(span) - 1
            self.amr.abstract_map[abstract] = dict(
                type='url-entity',
                span=' '.join(map(self.amr.tokens.__getitem__, span)),
                value=alignment.url)
            self.amr.replace_span(span, [abstract], ['NN'], ['URL'])
            self.amr.graph.replace_node_attribute(alignment.node, 'value', alignment.url, abstract)
        return count

    def remove_redundant_url(self):
        while True:
            for i in range(len(self.amr.lemmas)):
                lemma = self.amr.lemmas[i]
                if re.search('(https?:|<a.*href=|^</a>$)', lemma):
                    self.amr.remove_span([i])
                    break
            else:
                break

