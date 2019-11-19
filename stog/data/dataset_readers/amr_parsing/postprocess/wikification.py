import os
import json
import requests
from time import sleep
from collections import defaultdict

from bs4 import BeautifulSoup

from stog.data.dataset_readers.amr_parsing.io import AMRIO
from stog.utils import logging


logger = logging.init_logger()


def strip(text):
    start = 0
    while text[start] == '-' and start < len(text) - 1:
        start += 1
    end = len(text) - 1
    while text[end] == '-' and end > start:
        end -= 1
    return text[start:end + 1].strip()


def joint_dash(text):
    return text.replace(' - ', '-')


class Wikification:

    def __init__(self, util_dir):
        self.util_dir = util_dir
        self.wiki_span_cooccur_counter = None
        self.spotlight_cooccur_counter = defaultdict(lambda : defaultdict(int))
        self.nationality_map = {}
        self.name_node_count = 0
        self.correct_wikification_count = 0

    def reset_stats(self):
        self.name_node_count = 0
        self.correct_wikification_count = 0

    def print_stats(self):
        logger.info('Correctly wikify {}/{} name nodes.'.format(
            self.correct_wikification_count, self.name_node_count))

    def wikify_file(self, file_path):
        for i, amr in enumerate(AMRIO.read(file_path)):
            self.wikify_graph(amr)
            yield amr

    def wikify_graph(self, amr):
        graph = amr.graph
        abstract_map = amr.abstract_map
        for node in graph.get_nodes():
            instance = node.instance
            if instance in abstract_map:
                saved_dict = abstract_map[instance]
                instance_type = saved_dict['type']
                # amr_type = graph.get_name_node_type(node)
                #!!! the following line has been changed by Deng Cai
                cached_wiki = self._spotlight_wiki[amr.sentence] if amr.sentence in self._spotlight_wiki else None#self.spotlight_wiki(amr.sentence)
                if instance_type == 'named-entity':
                    self.name_node_count += 1
                    wiki = '-'
                    span = strip(saved_dict['span'])
                    if span.lower() in self.nationality_map:
                        country = self.nationality_map[span.lower()]
                        wiki = self.wikify(country, cached_wiki)
                    if wiki == '-':
                        wiki = self.wikify(span, cached_wiki)
                    if wiki == '-':
                        span_no_space = joint_dash(span)
                        wiki = self.wikify(span_no_space, cached_wiki)
                    graph.set_name_node_wiki(node, wiki)

    def wikify(self, text, cached_wiki=None):
        text = text.lower()
        if text in self.wiki_span_cooccur_counter:
            return max(self.wiki_span_cooccur_counter[text].items(),
                       key=lambda x: (x[1], abs(len(text) - len(x[0]))))[0]
        elif cached_wiki is not None:
            for wiki in cached_wiki.values():
                if text == ' '.join(wiki.lower().split('_')):
                    return wiki
        if text in self.spotlight_cooccur_counter:
            return max(self.spotlight_cooccur_counter[text].items(),
                       key=lambda x: (x[1], abs(len(text) - len(x[0]))))[0]
        else:
            tokens = text.split()
            if len(tokens) > 1:
                s_token, e_token = tokens[0], tokens[-1]
                for mention in self.spotlight_cooccur_counter:
                    m = mention.split()
                    if m[0] == s_token and m[-1] == e_token:
                        return max(self.spotlight_cooccur_counter[mention].items(),
                                   key=lambda x: (x[1], abs(len(mention) - len(x[0]))))[0]
            return '-'

    def load_utils(self):
        with open(os.path.join(self.util_dir, 'wiki_span_cooccur_counter.json'), encoding='utf-8') as f:
            self.wiki_span_cooccur_counter = json.load(f)

        with open(os.path.join(self.util_dir, 'spotlight_wiki.json'), encoding='utf-8') as f:
            spotlight_wiki = json.load(f)
        self._spotlight_wiki = spotlight_wiki
        for cached_wiki in spotlight_wiki.values():
            for mention, wiki in cached_wiki.items():
                if mention == 'xp':
                    continue
                self.spotlight_cooccur_counter[mention][wiki] += 1

        # The country list is downloaded from github:
        # https://github.com/Dinu/country-nationality-list
        with open(os.path.join(self.util_dir, 'countries.json'), encoding='utf-8') as f:
            countries = json.load(f)
            for country in countries:
                nationalities = [n.strip() for n in country['nationality'].split(',')]
                if len(nationalities) > 1 and 'Chinese' in nationalities:
                    nationalities.remove('Chinese')
                for nationality in nationalities:
                    self.nationality_map[nationality.lower()] = country['en_short_name']

    @staticmethod
    def spotlight_wiki(sent, confidence=0.5):
        success = False
        while not success:
            try:
                spotlight = requests.post(
                    "http://model.dbpedia-spotlight.org/en/annotate",
                    data={'text': sent, 'confidence': confidence}
                )
                spotlight.encoding = 'utf-8'
            except requests.exceptions.ConnectionError:
                logger.info('sleeping a bit (spotlight overload) - if this keeps happening server is down or changed')
                sleep(0.1)
                continue
            success = True
        parsed_spotlight = BeautifulSoup(spotlight.text, 'lxml')
        mention_map = {}
        for wiki_tag in parsed_spotlight.find_all('a'):
            mention_map[wiki_tag.string.lower()] = wiki_tag.get('href').split('/')[-1]
        logger.info(mention_map)
        return mention_map

    def dump_spotlight_wiki(self, file_path):
        sent_map = {}
        for i, amr in enumerate(AMRIO.read(file_path), 1):
            if i % 20 == 0:
                print('+', end='')
            sent = amr.sentence
            wiki = self.spotlight_wiki(sent)
            sent_map[sent] = wiki
            sleep(0.1)
        with open(os.path.join(self.util_dir, 'spotlight_wiki.json'), 'w', encoding='utf-8') as f:
            json.dump(sent_map, f)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser('wikification.py')
    parser.add_argument('--amr_files', nargs='+', default=[])
    parser.add_argument('--util_dir', required=True)
    parser.add_argument('--dump_spotlight_wiki', action='store_true',
                        help='Use the Spotlight API to do wikification, and dump the results.')

    args = parser.parse_args()

    wikification = Wikification(util_dir=args.util_dir)

    if args.dump_spotlight_wiki:
        for file_path in args.amr_files:
            wikification.dump_spotlight_wiki(file_path)

    else:
        wikification.load_utils()
        for file_path in args.amr_files:
            with open(file_path + '.wiki', 'w', encoding='utf-8') as f:
                for amr in wikification.wikify_file(file_path):
                    f.write(str(amr) + '\n\n')
