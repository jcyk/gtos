import re


class Polarity:

    lemma_map = {
        'alignment': 'align',
        'afraid': 'fear',
        'can': 'possible',
        'cant': 'possible',
        'could': 'possible',
        'have-to': 'obligate',
        'will': 'refuse',
        'as': 'equal',
        'strength': 'strong',
        'acceptable': 'possible',
        'disputable': 'possible',
        'to': 'have-purpose',
        'less-than': 'less',
        'without': 'have',
        'business': 'concern',
        'because': 'cause',
        'realise': 'realize',
        'able': 'capable',
        'work': 'work-out',
        'open': 'open-up',
        'shut': 'shut-off',
        'go-over': 'go',
        'ability': 'capable',
        'beat': 'beat-up',
        'moany': 'moan',
        'dependent': 'depend',
        'known': 'know',
        'agreement': 'agree',
        'applicable': 'apply',
        'proof': 'prove',
        'way': 'possible',
        'explanation': 'explain',
        'aware': 'realize',
        'legally': 'law',
        'conclusive': 'conclude',
        '-permanent': 'permanence',
        'occur': 'accident',
    }

    strict_lemma_map = {
        'month': 'possible',
        'evidence': 'thing',
        'member': 'have-org-role',
        'think': 'recommend',
        'traffic': 'have-polarity',
    }

    def __init__(self, amr, dry=False):
        self.amr = amr
        self.dry = dry
        self.nodes = self.get_negated_nodes()
        self.negations = []
        self.special_negations = []
        self.true_positive = 0
        self.false_positive = 0

    def remove_polarity(self):
        count = 0
        for node in self.get_negated_nodes():
            for attr, value in node.attributes:
                if attr == 'polarity':
                    if not self.dry:
                        self.amr.graph.remove_node_attribute(node, attr, value)
                    count += 1
        for node in self.amr.graph.get_nodes():
            if node.instance == 'have-polarity-91':
                for attr, value in node.attributes:
                    if value == '-':
                        if not self.dry:
                            self.amr.graph.remove_node_attribute(node, attr, value)
                        count += 1
        return count

    def predict_polarity(self):
        """
        Use rules to predict polarity and its head.
        """
        for i in range(len(self.amr.tokens)):
            if self.is_negation(i):
                head = self.get_head(i)
                if head is not None:
                    self.negations.append((i, head))
            else:
                self.add_special_negation(i)

    def restore_polarity(self):
        negations = self.negations
        special_negations = self.special_negations
        instance_map = self.get_node_instances()
        remaining_nodes = list(self.amr.graph.get_nodes())
        for neg_index, head_index in negations:
            for instance, nodes in instance_map.items():
                if self.is_match(head_index, instance):
                    nodes = self.sort_node_by_distance(head_index, nodes)
                    for node in nodes:
                        if node not in remaining_nodes:
                            continue
                        remaining_nodes.remove(node)
                        self.restore_node_polarity(neg_index, head_index, node)
                        break
                    break

        for neg_index, head in special_negations:
            head_lemma = self.amr.lemmas[neg_index]
            for instance, nodes in instance_map.items():
                if head_lemma == instance:
                    continue
                if self.is_match(head, instance):
                    nodes = self.sort_node_by_distance(neg_index, nodes)
                    for node in nodes:
                        if node not in remaining_nodes:
                            continue
                        remaining_nodes.remove(node)
                        self.restore_node_polarity(neg_index, head, node)
                        break
                    break

        for node in self.amr.graph.get_nodes():
            if node.instance == 'have-polarity':
                for attr, value in node.attributes:
                    if attr == 'ARG2' and value == '-':
                        self.true_positive += 1
                        break
                else:
                    self.amr.graph.add_node_attribute(node, 'ARG2', '-')

    def restore_node_polarity(self, neg_index, head, node):
        if self.special_restoration(neg_index, head, node):
            return
        if not self.dry:
            self.amr.graph.add_node_attribute(node, 'polarity', '-')
        else:
            for attr, value in node.attributes:
                if attr == 'polarity':
                    self.true_positive += 1
                    break
            else:
                self.false_positive += 1

    def to_dict(self):
        return dict(Negations=self.negations, SpecialNegations=self.special_negations)

    def special_restoration(self, neg_index, head, node):
        head_lemma = self.amr.lemmas[head] if isinstance(head, int) else head
        if node.instance == 'have-polarity':
            if self.dry:
                for attr, value in node.attributes:
                    if attr == 'ARG2' and value == '-':
                        self.true_positive += 1
                        break
            else:
                self.amr.graph.add_node_attribute(node, 'ARG2', '-')
            return True

        if head_lemma == 'face' and node.instance == 'face':
            for source, target in self.amr.graph._G.in_edges(node):
                if source.instance == 'want':
                    if self.dry:
                        for attr, value in source.attributes:
                            if attr == 'polarity':
                                self.true_positive += 1
                                break
                    else:
                        self.amr.graph.add_node_attribute(source, 'polarity', '-')
                    break
            return True
        return False

    def is_negation(self, index):
        lemma = self.amr.lemmas[index]
        next_lemma = self.amr.lemmas[index + 1] if index + 1 < len(self.amr.lemmas) else None
        if lemma in ('not', 'never', 'without', 'no', 'dont', 'nowhere', 'none',
                     'neither', 'havent', 'didnt', 'wont', 'cant', 'doesnt'):
            return True
        if lemma == 'no-one' and next_lemma == 'can':
            return True
        return False

    def add_special_negation(self, index):
        lemma = self.amr.lemmas[index]
        last_lemma = self.amr.lemmas[index - 1] if index - 1 >= 0 else None
        if re.search(r'.+less$', lemma):
            head = re.sub(r'less$', '', lemma)
            self.special_negations.append((index, head))
        elif re.search(r'^(non|un|il|irr|in|Non).+', lemma):
            head = re.sub(r'^(non|un|il|ir|in|Non)', '', lemma)
            if head not in ('i', 'common') and last_lemma != 'not':
                self.special_negations.append((index, head))
        elif lemma == 'asymmetrical':
            head = 'symmetrical'
            self.special_negations.append((index, head))

    def is_head(self, index):
        lemma = self.amr.lemmas[index]
        token = self.amr.tokens[index]
        pos_tag = self.amr.pos_tags[index]
        next_lemma = self.amr.lemmas[index + 1] if index + 1 < len(self.amr.lemmas) else None
        if re.search(r'^(^[^a-zA-Z0-9]+|be|the|a|so|that|any|person|this|stage|people|near|health)$', lemma):
            return False
        if token in ('remaining',):
            return False
        if pos_tag in ('PRP', '.', 'IN', 'PRP$', 'POS', 'DT'):
            return False
        if pos_tag in ('JJ',) and next_lemma in ['measure']:
            return False
        if pos_tag == 'RB' and lemma not in ('there',):
            return False
        if lemma == 'take' and next_lemma == 'long':
            return False
        return True

    def get_head(self, index):
        head = self.get_special_head(index)
        if head is None:
            i = index + 1
            head = None
            while i < len(self.amr.tokens):
                if self.is_head(i):
                    head = i
                    break
                i += 1
        if self.is_false_positive(index, head):
            return None
        return head

    def is_false_positive(self, index, head):
        if head is None:
            return True
        last_lemma = self.amr.lemmas[index - 1] if index - 1 >= 0 else None
        next_lemma = self.amr.lemmas[index + 1] if index + 1 < len(self.amr.lemmas) else None
        third_lemma = self.amr.lemmas[index + 2] if index + 2 < len(self.amr.lemmas) else None
        head_lemma = self.amr.lemmas[head]
        if next_lemma and next_lemma in ('matter', 'address', 'other', 'only', ',', '.'):
            return True
        if next_lemma == '-' and third_lemma != 'alignment':
            return True
        if next_lemma == 'like' and self.amr.pos_tags[index + 1] == 'IN':
            return True
        if last_lemma and last_lemma in ('will',) and next_lemma not in ('fade', 'get', 'make', 'succumb'):
            return True
        if head_lemma in ('behave',):
            return True
        return False

    def get_special_head(self, index):
        lemma = self.amr.lemmas[index]
        last_lemma = self.amr.lemmas[index - 1] if index - 1 >= 0 else None
        llast_lemma = self.amr.lemmas[index - 2] if index - 2 >= 0 else None
        next_lemma = self.amr.lemmas[index + 1] if index + 1 < len(self.amr.lemmas) else None
        third_lemma = self.amr.lemmas[index + 2] if index + 2 < len(self.amr.lemmas) else None
        fourth_lemma = self.amr.lemmas[index + 3] if index + 3 < len(self.amr.lemmas) else None
        if lemma in ('cant',):
            return index
        if last_lemma in ('can', 'want', 'add', 'feel', 'strike', 'could'):
            return index - 1
        if last_lemma == 'would':
            if next_lemma in ('tell', 'read', 'worry'):
                return index + 1
            if next_lemma == 'be' and third_lemma in ('know', 'interested', 'arrest'):
                return index + 2
            return index - 1
        if last_lemma and next_lemma and last_lemma in ('have',) and next_lemma in (
                'cohesion', 'access', 'more', 'enforcement'):
            return index - 1
        if next_lemma in ('because', 'alone'):
            return index + 1
        if next_lemma in ('as',):
            if third_lemma in ('much', 'severely'):
                return index + 2
            if third_lemma and self.amr.pos_tags[index + 2] in ('RB',):
                return None
            return index + 1
        if next_lemma == 'new' and third_lemma == 'initiative':
            return index + 8
        if next_lemma == '-' and third_lemma == 'alignment':
            return index + 2
        if next_lemma == 'take' and third_lemma == 'long':
            return index + 2
        if next_lemma == 'make' and fourth_lemma == 'sense':
            return index + 3
        if next_lemma == 'have' and fourth_lemma == 'right':
            return index + 3
        if next_lemma == 'to' and third_lemma in ('keep', 'worry'):
            return index + 2
        if next_lemma == 'to' and third_lemma in ('be',):
            return index + 3
        if next_lemma == 'on' and fourth_lemma == 'way':
            return index + 5
        if next_lemma == 'to' and third_lemma == 'way':
            return index + 2
        if next_lemma == 'something' and third_lemma == 'you':
            return index + 5
        if next_lemma == 'negotiation' and third_lemma == 'can':
            return index + 2
        if next_lemma == 'diplomatic' and third_lemma == 'factor':
            return index + 2
        if third_lemma == 'diplomat' and fourth_lemma == 'nor':
            return index + 9
        if third_lemma == 'hacking' and fourth_lemma == 'claim':
            return index + 16
        if last_lemma == 'off' and llast_lemma == 'better':
            return index
        if last_lemma == 'would' and next_lemma in 'like':
            return index + 1
        if next_lemma == 'go' and third_lemma == 'to':
            return index + 3
        if lemma == 'neither' and fourth_lemma == 'counseller':
            return index + 5
        return None

    def get_negated_nodes(self):
        graph = self.amr.graph
        nodes = []
        for node in graph.get_nodes():
            for attr, value in node.attributes:
                if attr == 'polarity':
                    nodes.append(node)
                    break
        return nodes

    def get_node_instances(self):
        instances = {}
        for node in self.amr.graph.get_nodes():
            instance = re.sub(r'-\d\d$', '', str(node.instance))
            if instance not in instances:
                instances[instance] = []
            instances[instance].append(node)
        return instances

    def validate(self):
        p1, p2 = self.get_precision()
        r1, r2 = self.get_recall()
        return p1, p2, r1, r2

    def sort_node_by_distance(self, head, nodes):
        if len(nodes) <= 1:
            return nodes
        lemmas = self.amr.lemmas
        scores = []
        alignment = {}
        for node in nodes:
            aligned_indexes = set()
            for n in self.amr.graph.get_subtree(node, max_depth=1)[1:]:
                if n.instance in lemmas:
                    aligned_indexes.add(lemmas.index(n.instance))
            if len(aligned_indexes) == 0:
                for n in self.amr.graph.get_subtree(node, max_depth=5)[1:]:
                    if n.instance in lemmas:
                        aligned_indexes.add(lemmas.index(n.instance))
            for n, _ in self.amr.graph._G.in_edges(node):
                if n.instance in lemmas:
                    aligned_indexes.add(lemmas.index(n.instance))
            alignment[node] = aligned_indexes
        for node in nodes:
            aligned_indexes = alignment[node]
            valid_indexes = []
            for index in aligned_indexes:
                if sum(1 for indexes in alignment.values() if index in indexes) == 1:
                    valid_indexes.append(index)
            if len(valid_indexes):
                scores.append(min([abs(i - head) for i in valid_indexes]))
            else:
                scores.append(float('inf'))
        nodes, scores = zip(*sorted(zip(nodes, scores), key=lambda x: x[1]))
        return nodes

    def get_precision(self):
        neg_nodes = self.get_negated_nodes()
        rest_nodes = list(self.amr.graph.get_nodes())
        match_count, correct_count = 0, 0
        instances = self.get_node_instances()
        for neg, head in self.negations:
            match, correct = False, False
            for instance, nodes in instances.items():
                if self.is_match(head, instance):
                    match = True
                    nodes = self.sort_node_by_distance(head, nodes)
                    for node in nodes:
                        if node not in rest_nodes:
                            continue
                        if node in neg_nodes:
                            correct = True
                        rest_nodes.remove(node)
                        break
                    match_count += int(match)
                    correct_count += int(correct)
                    break

        for index, head in self.special_negations:
            match, correct = False, False
            head_lemma = self.amr.lemmas[index]
            for instance, nodes in instances.items():
                if head_lemma == instance:
                    continue
                if self.is_match(head, instance):
                    match = True
                    nodes = self.sort_node_by_distance(index, nodes)
                    for node in nodes:
                        if node not in rest_nodes:
                            continue
                        if node in neg_nodes:
                            correct = True
                        rest_nodes.remove(node)
                        break
                    match_count += int(match)
                    correct_count += int(correct)
                    break
        return correct_count, match_count

    def get_recall(self):
        recall_count = 0
        for node in self.nodes:
            recall = False
            for neg, head in self.negations:
                if self.is_match(head, node.instance):
                    recall = True
                    break
            if not recall:
                for index, head in self.special_negations:
                    head_lemma = self.amr.lemmas[index]
                    if head_lemma == node.instance:
                        continue
                    if self.is_match(head, node.instance):
                        recall = True
                        break
            if recall:
                recall_count += 1
            else:
                continue
        return recall_count, len(self.nodes)

    def is_match(self, index, instance):
        instance_lemma = re.sub('-\d\d$', '', instance)
        lemma = self.amr.lemmas[index] if isinstance(index, int) else index
        lemma = self.strict_lemma_map.get(lemma, lemma)
        if self.lemma_map.get(lemma, None) == instance_lemma:
            return True
        if lemma == instance_lemma:
            return True
        if instance_lemma + 'ed' == lemma or instance_lemma + 'd' == lemma:
            return True
        if '-' + instance_lemma == lemma:
            return True
        if re.sub('ly$', 'le', lemma) == instance_lemma:
            return True
        if re.sub('tive$', 'te', lemma) == instance_lemma:
            return True
        if re.sub('tion$', 'te', lemma) == instance_lemma:
            return True
        if re.sub('ied$', 'y', lemma) == instance_lemma:
            return True
        if re.sub('ly$', '', lemma) == instance_lemma:
            return True
        return False

    def __str__(self):
        lemmas = self.amr.lemmas
        negations = [(lemmas[i], lemmas[j]) for i, j in self.negations]
        return 'Negations: {}\nSpecials:{}\n'.format(str(negations), str(self.special_negations))

