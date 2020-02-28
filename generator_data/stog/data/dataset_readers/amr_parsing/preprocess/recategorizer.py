import os
import json
from collections import defaultdict

import nltk

from stog.data.dataset_readers.amr_parsing.io import AMRIO
from stog.data.dataset_readers.amr_parsing.amr_concepts import Entity, Date, Score, Quantity, Ordinal, Polarity, Polite, URL
from stog.utils import logging


logger = logging.init_logger()


def resolve_conflict_entities(entities):
    # If there's overlap between any two entities,
    # remove the one that has lower confidence.
    index_entity_map = {}
    empty_entities = []
    for entity in entities:
        if not entity.span:
            empty_entities.append(entity)
            continue

        for index in entity.span:
            if index in index_entity_map:
                _entity = index_entity_map[index]
                if _entity.confidence < entity.confidence:
                    index_entity_map[index] = entity
            else:
                index_entity_map[index] = entity
    node_entity_map = {}
    for entity in index_entity_map.values():
        node_entity_map[entity.node] = entity

    removed_entities = []
    for entity in entities:
        if not entity.span:
            continue
        if entity.node in node_entity_map:
            continue
        removed_entities.append(entity)

    return list(node_entity_map.values()) + empty_entities, removed_entities


class Recategorizer:

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
        amr.stems = [self.stemmer(l) for l in amr.lemmas]
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
                entity = Entity.get_aligned_entity(
                    node, amr, backup_ner_type, self.entity_type_cooccur_counter)
                if len(entity.span):
                    self.recat_named_entity_count += 1
                entities.append(entity)
        entities, removed_entities = resolve_conflict_entities(entities)
        if not self.build_utils:
            type_counter = Entity.collapse_name_nodes(entities, amr)
            for entity in removed_entities:
                amr_type = amr.graph.get_name_node_type(entity.node)
                backup_ner_type = self._map_name_node_type(amr_type)
                entity = Entity.get_aligned_entity(
                    entity.node, amr, backup_ner_type, self.entity_type_cooccur_counter)
                Entity.collapse_name_nodes([entity], amr, type_counter)
        else:
            self._update_utils(entities, amr)

    def recategorize_date_nodes(self, amr):
        graph = amr.graph
        dates = []
        for node, _, _ in graph.get_list_node(replace_copy=False):
            if node.copy_of is not None:
                continue
            if graph.is_date_node(node) and Date.collapsable(node, graph):
                self.date_entity_count += 1
                date = self._get_aligned_date(node, amr)
                if date.span is not None:
                    self.recat_date_entity_count += 1
                dates.append(date)
        dates, removed_dates = resolve_conflict_entities(dates)
        Date.collapse_date_nodes(dates, amr)

    def recategorize_score_nodes(self, amr):
        graph = amr.graph
        scores = []
        for node, _, _ in graph.get_list_node(replace_copy=False):
            if node.copy_of is not None:
                continue
            if node.instance == 'score-entity':
                self.score_entity_count += 1
                score = Score(node, amr)
                if score.span is not None:
                    self.recat_score_entity_count += 1
                scores.append(score)
        Score.collapse_score_nodes(scores, amr)

    def recategorize_ordinal_nodes(self, amr):
        graph = amr.graph
        ordinals = []
        for node, _, _ in graph.get_list_node(replace_copy=False):
            if node.copy_of is not None:
                continue
            if node.instance == 'ordinal-entity':
                self.ordinal_entity_count += 1
                ordinal = Ordinal(node, amr)
                if ordinal.span is not None:
                    self.recat_ordinal_entity_count += 1
                ordinals.append(ordinal)
        Ordinal.collapse_ordinal_nodes(ordinals, amr)

    def recategorize_quantities(self, amr):
        quantity = Quantity(amr)
        self.recat_quantity_count += quantity.abstract()
        self.quantity_count += quantity.quant_count

    def recategorize_urls(self, amr):
        url = URL(amr)
        url_count, recat_url_count = url.abstract()
        self.url_count += url_count
        self.recat_url_count += recat_url_count

    def _get_aligned_date(self, node, amr):
        date = Date(node, amr.graph)
        if len(date.attributes) + len(date.edges) == 0:
            return date
        alignment = date._get_alignment(amr)
        date._get_span(alignment, amr)
        return date

    def _update_utils(self, entities, amr):
        if not self.build_entity_map:
            for entity in entities:
                wiki_title = amr.graph.get_name_node_wiki(entity.node)
                if wiki_title is None:
                    wiki_title = '-'
                for text_span in entity.get_text_spans(amr):
                    text_span = text_span.lower()
                    self.wiki_span_cooccur_counter[text_span][wiki_title] += 1
                    self.name_op_cooccur_counter[text_span][' '.join(entity.get_ops())] += 1

                if len(entity.span) == 0:
                    continue
                entity_text = ' '.join(amr.tokens[index] for index in entity.span).lower()
                self.entity_type_cooccur_counter[entity_text][entity.ner_type] += 1

        else:
            for entity in entities:
                if len(entity.span) == 0:
                    continue
                if entity.ner_type != Entity.unknown_entity_type:
                    self.name_type_cooccur_counter[entity.amr_type][entity.ner_type] += 1


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser('recategorizer.py')
    parser.add_argument('--amr_train_file')
    parser.add_argument('--amr_files', nargs='+')
    parser.add_argument('--dump_dir')
    parser.add_argument('--build_utils', action='store_true')

    args = parser.parse_args()

    recategorizer = Recategorizer(
        train_data=args.amr_train_file,
        build_utils=args.build_utils,
        util_dir=args.dump_dir)

    for file_path in args.amr_files:
        with open(file_path + '.recategorize', 'w', encoding='utf-8') as f:
            for amr in recategorizer.recategorize_file(file_path):
                f.write(str(amr) + '\n\n')

