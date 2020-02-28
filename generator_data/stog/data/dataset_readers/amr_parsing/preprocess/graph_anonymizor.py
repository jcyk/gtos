import os
import json
from collections import defaultdict

import nltk

from stog.data.dataset_readers.amr_parsing.io import AMRIO
from stog.data.dataset_readers.amr_parsing.amr_concepts import Entity, Date, Score, Quantity, Ordinal, Polarity, Polite, URL
from stog.utils import logging


logger = logging.init_logger()

class GraphAnonymizor:

    def __init__(self, train_data=None, build_utils=False, util_dir=None):
        self.stemmer = nltk.stem.SnowballStemmer('english').stem
        self.train_data = train_data
        self.build_utils = build_utils
        self.named_entity_count = 0
        self.recat_named_entity_count = 0
        self.date_entity_count = 0
        self.recat_date_entity_count = 0
        self.score_entity_count = 0
        self.recat_score_entity_count = 0
        self.ordinal_entity_count = 0
        self.recat_ordinal_entity_count = 0
        self.quantity_count = 0
        self.recat_quantity_count = 0
        self.url_count = 0
        self.recat_url_count = 0
        self.removed_wiki_count = 0

        self.name_type_cooccur_counter = defaultdict(lambda: defaultdict(int))
        self.name_op_cooccur_counter = defaultdict(lambda: defaultdict(int))
        self.wiki_span_cooccur_counter = defaultdict(lambda: defaultdict(int))
        self.build_entity_map = False
        self.entity_type_cooccur_counter = defaultdict(lambda: defaultdict(int))
        if build_utils:
            self._build_utils()
            self._dump_utils(util_dir)
        else:
            self._load_utils(util_dir)

    def _print_statistics(self):
        if self.named_entity_count != 0:
            logger.info('Named entity collapse rate: {} ({}/{})'.format(
                self.recat_named_entity_count / self.named_entity_count,
                self.recat_named_entity_count, self.named_entity_count))
        if self.date_entity_count != 0:
            logger.info('Dated entity collapse rate: {} ({}/{})'.format(
                self.recat_date_entity_count / self.date_entity_count,
                self.recat_date_entity_count, self.date_entity_count))
        if self.score_entity_count != 0:
            logger.info('Score entity collapse rate: {} ({}/{})'.format(
                self.recat_score_entity_count / self.score_entity_count,
                self.recat_score_entity_count, self.score_entity_count))
        if self.ordinal_entity_count != 0:
            logger.info('Ordinal entity collapse rate: {} ({}/{})'.format(
                self.recat_ordinal_entity_count / self.ordinal_entity_count,
                self.recat_ordinal_entity_count, self.ordinal_entity_count))
        if self.quantity_count != 0:
            logger.info('Quantity collapse rate: {} ({}/{})'.format(
                self.recat_quantity_count / self.quantity_count,
                self.recat_quantity_count, self.quantity_count))
        if self.url_count != 0:
            logger.info('URL collapse rate: {} ({}/{})'.format(
                self.recat_url_count / self.url_count,
                self.recat_url_count, self.url_count))
        logger.info('Removed {} wikis.'.format(self.removed_wiki_count))

    def reset_statistics(self):
        self.named_entity_count = 0
        self.recat_named_entity_count = 0
        self.date_entity_count = 0
        self.recat_date_entity_count = 0
        self.score_entity_count = 0
        self.recat_score_entity_count = 0
        self.ordinal_entity_count = 0
        self.recat_ordinal_entity_count = 0
        self.quantity_count = 0
        self.recat_quantity_count = 0
        self.url_count = 0
        self.recat_url_count = 0
        self.removed_wiki_count = 0

    def _build_utils(self):
        logger.info('Building name_type_cooccur_counter and wiki_span_cooccur_counter...')
        for _ in self.recategorize_file(self.train_data):
            pass
        self.build_entity_map = True
        logger.info('Done.\n')
        logger.info('Building entity_type_cooccur_counter...')
        self.reset_statistics()
        for _ in self.recategorize_file(self.train_data):
            pass
        logger.info('Done.\n')

    def _dump_utils(self, directory):
        with open(os.path.join(directory, 'name_type_cooccur_counter.json'), 'w', encoding='utf-8') as f:
            json.dump(self.name_type_cooccur_counter, f, indent=4)
        with open(os.path.join(directory, 'name_op_cooccur_counter.json'), 'w', encoding='utf-8') as f:
            json.dump(self.name_op_cooccur_counter, f, indent=4)
        with open(os.path.join(directory, 'wiki_span_cooccur_counter.json'), 'w', encoding='utf-8') as f:
            json.dump(self.wiki_span_cooccur_counter, f, indent=4)
        with open(os.path.join(directory, 'entity_type_cooccur_counter.json'), 'w', encoding='utf-8') as f:
            json.dump(self.entity_type_cooccur_counter, f, indent=4)

    def _load_utils(self, directory):
        with open(os.path.join(directory, 'name_type_cooccur_counter.json'), encoding='utf-8') as f:
            self.name_type_cooccur_counter = json.load(f)
        with open(os.path.join(directory, 'name_op_cooccur_counter.json'), encoding='utf-8') as f:
            self.name_op_cooccur_counter = json.load(f)
        with open(os.path.join(directory, 'wiki_span_cooccur_counter.json'), encoding='utf-8') as f:
            self.wiki_span_cooccur_counter = json.load(f)
        with open(os.path.join(directory, 'entity_type_cooccur_counter.json'), encoding='utf-8') as f:
            self.entity_type_cooccur_counter = json.load(f)

    def _map_name_node_type(self, name_node_type):
        if not self.build_utils and name_node_type in self.name_type_cooccur_counter:
            ner_type = max(self.name_type_cooccur_counter[name_node_type].keys(),
                       key=lambda ner_type: self.name_type_cooccur_counter[name_node_type][ner_type])
            if ner_type in ('0', 'O'):
                return Entity.unknown_entity_type
            else:
                return ner_type
        else:
            return Entity.unknown_entity_type

    def recategorize_file(self, file_path):
        for i, amr in enumerate(AMRIO.read(file_path), 1):
            self.recategorize_graph(amr)
            yield amr
            if i % 1000 == 0:
                logger.info('Processed {} examples.'.format(i))
        logger.info('Done.\n')

    def recategorize_graph(self, amr):
        #amr.stems = [self.stemmer(l) for l in amr.lemmas]
        self.resolve_name_node_reentrancy(amr)
        self.recategorize_name_nodes(amr)
        if self.build_utils:
            return
        self.remove_wiki(amr)
        #self.remove_negation(amr)
        self.recategorize_date_nodes(amr)
        self.recategorize_score_nodes(amr)
        self.recategorize_ordinal_nodes(amr)
        self.recategorize_quantities(amr)
        self.recategorize_urls(amr)

    def resolve_name_node_reentrancy(self, amr):
        graph = amr.graph
        for node in graph.get_nodes():
            if graph.is_name_node(node):
                edges = list(graph._G.in_edges(node))
                name_head = None
                for source, target in edges:
                    if graph._G[source][target]['label'] == 'name':
                        name_head = source
                        break
                for source, target in edges:
                    label = graph._G[source][target]['label']
                    if label != 'name':
                        graph.remove_edge(source, target)
                        graph.add_edge(source, name_head, label)

    def remove_wiki(self, amr):
        graph = amr.graph
        for node in graph.get_nodes():
            for attr, value in node.attributes.copy():
                if attr == 'wiki':
                    self.removed_wiki_count += 1
                    graph.remove_node_attribute(node, attr, value)

    def remove_negation(self, amr):
        polarity = Polarity(amr)
        polarity.remove_polarity()
        polite = Polite(amr)
        polite.remove_polite()

    def recategorize_name_nodes(self, amr):
        graph = amr.graph
        entities = []
        for node, _, _ in graph.get_list_node(replace_copy=False):
            if node.copy_of is not None:
                continue
            if graph.is_name_node(node):
                edges = list(graph._G.in_edges(node))
                assert all(graph._G[s][t]['label'] == 'name' for s, t in edges)
                self.named_entity_count += 1
                amr_type = amr.graph.get_name_node_type(node)
                backup_ner_type = self._map_name_node_type(amr_type)
                entity = Entity(node=node, ner_type=backup_ner_type)
                entities.append(entity)
        type_counter = self._collapse_name_nodes(entities, amr)

    def _collapse_name_nodes(self, entities, amr, type_counter=None):
        if amr.abstract_map is None:
            amr.abstract_map = {}
        if len(entities) == 0:
            return
        if type_counter is None:
            type_counter = defaultdict(int)

        for entity in entities:
            type_counter[entity.ner_type] += 1
            abstract = '{}_{}'.format(
                entity.ner_type, type_counter[entity.ner_type])
            span_with_offset = []
            amr.abstract_map[abstract] = Entity.save_collapsed_name_node(
                entity, span_with_offset, amr)
            amr.graph.remove_node_ops(entity.node)
            amr.graph.replace_node_attribute(
                entity.node, 'instance', entity.node.instance, abstract)

        return type_counter

    def recategorize_date_nodes(self, amr):
        graph = amr.graph
        dates = []
        for node, _, _ in graph.get_list_node(replace_copy=False):
            if node.copy_of is not None:
                continue
            if graph.is_date_node(node) and Date.collapsable(node, graph):
                self.date_entity_count += 1
                date = Date(node=node, graph=graph)
                dates.append(date)
        self._collapse_date_nodes(dates, amr)
    
    def _collapse_date_nodes(self, dates, amr):
        if amr.abstract_map is None:
            amr.abstract_map = {}
        
        align_count = 0
        for date in dates:
            align_count += 1
            abstract = '{}_{}'.format(date.ner_type, align_count)
            span_with_offset = []
            amr.abstract_map[abstract] = Date.save_collapsed_date_node(
                date, span_with_offset, amr)

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

    def recategorize_score_nodes(self, amr):
        graph = amr.graph
        scores = []
        for node, _, _ in graph.get_list_node(replace_copy=False):
            if node.copy_of is not None:
                continue
            if node.instance == 'score-entity':
                scores.append(node)
        self._collapse_score_nodes(scores, amr)
    
    def _collapse_score_nodes(self, scores, amr):
        score_node_count = 0
        for score in scores:
            score_node_count += 1
            abstract = '{}_{}'.format('SCORE_ENTITY', score_node_count)
            amr.abstract_map[abstract] = {  'type': 'score-entity',
                                            'span': '',
                                            'ops': score.ops}
            amr.graph.remove_node_ops(score)
            amr.graph.replace_node_attribute(
                score, 'instance', score.instance, abstract)

    def recategorize_ordinal_nodes(self, amr):
        graph = amr.graph
        ordinals = []
        for node, _, _ in graph.get_list_node(replace_copy=False):
            if node.copy_of is not None:
                continue
            if node.instance == 'ordinal-entity':
                ordinal = Ordinal(node, amr, align=False)
                ordinals.append(ordinal)
        self._collapse_ordinal_nodes(ordinals, amr)
    
    def _collapse_ordinal_nodes(self, ordinals, amr):
        node_count = 0
        for ordinal in ordinals:
            node_count += 1
            abstract = '{}_{}'.format(ordinal.ner_type, node_count)
            span = []
            amr.abstract_map[abstract] = ordinal.to_dict(amr, span)
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

    def recategorize_quantities(self, amr):
        quantity = Quantity(amr)
        self.recat_quantity_count += quantity.abstract(align=False)
        self.quantity_count += quantity.quant_count

    def recategorize_urls(self, amr):
        url = URL(amr)
        url_count, recat_url_count = url.abstract(align=False)
        self.url_count += url_count
        self.recat_url_count += recat_url_count


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser("graph_anonymizor.py")

    parser.add_argument('--amr_file', required=True, help="File to anonymize.")
    parser.add_argument('--util_dir')
    args = parser.parse_args()

    graph_anonymizor = GraphAnonymizor(util_dir=args.util_dir)


    with open(args.amr_file + ".recategorize", "w", encoding="utf-8") as f:
        for amr in graph_anonymizor.recategorize_file(args.amr_file):
            f.write(str(amr) + '\n\n')