import os
import re
import json
from collections import Counter

import editdistance

from stog.data.dataset_readers.amr_parsing.io import AMRIO
from stog.data.dataset_readers.amr_parsing.propbank_reader import PropbankReader
from stog.utils import logging


logger = logging.init_logger()
WORDSENSE_RE = re.compile(r'-\d\d$')
QUOTED_RE = re.compile(r'^".*"$')


def is_sense_string(s):
    return isinstance(s, str) and WORDSENSE_RE.search(s)


def is_quoted_string(s):
    return isinstance(s, str) and QUOTED_RE.search(s)


class NodeUtilities:

    def __init__(self,
                 senseless_node_counter=None,
                 lemma_frame_counter=None,
                 frame_lemma_counter=None,
                 ):
        self.senseless_node_counter = senseless_node_counter
        self.lemma_frame_counter = lemma_frame_counter
        self.frame_lemma_counter = frame_lemma_counter

        self.frequent_senseless_nodes = None
        self.lemma_frame_map = None
        self.frame_lemma_map = None

    def get_lemmas(self, frame):
        """
        Given a frame, find the most likely lemmas for the frame.
        If no lemma is found, return a single element list [frame].
        """
        if frame not in self.frame_lemma_map:
            return [frame]
        else:
            frame_lemma = re.sub(r'-\d\d$', '', frame)
            lemmas = list(self.frame_lemma_map[frame])
            lemmas.sort(key=lambda lemma: editdistance.eval(frame_lemma, lemma), reverse=True)
            return lemmas

    def get_frames(self, lemma):
        """
        Given a lemma, find the most likely frames for the lemma.
        If no lemma is found or it should be a senseless node, return a single element list [lemma].
        """
        if lemma in self.frequent_senseless_nodes or lemma not in self.lemma_frame_map:
            return [lemma]
        else:
            frames = list(self.lemma_frame_map[lemma])
            frames.sort(
                key=lambda frame: (
                    editdistance.eval(re.sub(r'-\d\d$', '', frame), lemma),
                    -int(frame[-2:]) if re.search(r'-\d\d$', frame) else 0
                ),
                reverse=True
            )
            return frames

    @classmethod
    def from_json(cls, json_dir, frequent_threshold=50):
        nu = cls()
        with open(os.path.join(json_dir, 'senseless_node_counter.json')) as f:
            nu.senseless_node_counter = Counter(json.load(f))
        with open(os.path.join(json_dir, 'lemma_frame_counter.json')) as f:
            nu.lemma_frame_counter = json.load(f)
        with open(os.path.join(json_dir, 'frame_lemma_counter.json')) as f:
            nu.frame_lemma_counter = json.load(f)

        nu.frequent_senseless_nodes = nu._get_frequent_senseless_nodes(
            nu.senseless_node_counter, frequent_threshold
        )
        nu.lemma_frame_map = nu._get_map_from_counter(nu.lemma_frame_counter)
        nu.frame_lemma_map = nu._get_map_from_counter(nu.frame_lemma_counter)
        return nu

    @classmethod
    def from_raw(cls, amr_train_files, propbank_dir, verbalization_file, dump_dir,
                 # Hyperparameters
                 train_file_base_freq=1,
                 propbank_base_freq=1,
                 propbank_bonus=10,
                 verbalization_base_freq=1,
                 verbalize_freq=100,
                 maybe_verbalize_freq=1,
                 verbalize_bonus=10
                 ):

        # counter[senseless_node]: the occurrence of senseless_node
        senseless_node_counter = cls._get_senseless_node_counter(amr_train_files)
        # counter[lemma][frame]: the co-occurrence of (lemma, frame)
        lemma_frame_counter = dict()
        # counter[frame][lemma]: the co-occurrence of (frame, lemma)
        frame_lemma_counter = dict()

        nu = cls(
            senseless_node_counter=senseless_node_counter,
            lemma_frame_counter=lemma_frame_counter,
            frame_lemma_counter=frame_lemma_counter
        )
        nu._update_counter_from_train_files(amr_train_files, train_file_base_freq)
        propbank_reader = PropbankReader(propbank_dir)
        nu._update_counter_from_propbank(propbank_reader, propbank_base_freq, propbank_bonus)
        nu._update_counter_from_verbalization(
            verbalization_file,
            verbalization_base_freq,
            verbalize_freq,
            maybe_verbalize_freq,
            verbalize_bonus
        )

        nu.frequent_senseless_nodes = cls._get_frequent_senseless_nodes(senseless_node_counter)
        nu.frame_lemma_map = nu._get_map_from_counter(nu.frame_lemma_counter)
        nu.lemma_frame_map = nu._get_map_from_counter(nu.lemma_frame_counter)

        nu.dump(dump_dir)

        return nu

    @staticmethod
    def _get_senseless_node_counter(amr_train_files):
        logger.info('Building the senseless node counter.')
        sense_less_nodes = []
        for amr_file in amr_train_files:
            for amr in AMRIO.read(amr_file):
                for node in amr.graph.get_nodes():
                    for attr, value in node.get_senseless_attributes():
                        sense_less_nodes.append(value)
        return Counter(sense_less_nodes)

    @staticmethod
    def _get_frequent_senseless_nodes(senseless_node_counter, threshold=50):
        frequent_senseless_nodes = set()
        for node, count in senseless_node_counter.most_common():
            if count >= threshold:  # hard threshold
                frequent_senseless_nodes.add(node)
        return frequent_senseless_nodes

    def _update_counter_from_train_files(self, amr_train_files, base_freq=1):
        logger.info('Updating (lemma, frame) counter from AMR train files.')
        for file_path in amr_train_files:
            for amr in AMRIO.read(file_path):
                for node in amr.graph.get_nodes():
                    for _, frame in node.get_frame_attributes():
                        frame_lemma = re.sub(WORDSENSE_RE, '', frame)
                        self._update_counter(self.lemma_frame_counter, frame_lemma, frame, base_freq)
                        self._update_counter(self.frame_lemma_counter, frame, frame_lemma, base_freq)

    def _update_counter_from_propbank(self, propbank_reader, base_freq=1, bonus=10):
        logger.info('Updating (lemma, frame) counter from Propbank.')
        for lemma, frames in propbank_reader.lemma_map.items():
            for frame in frames:
                freq = base_freq
                if lemma == frame.lemma:  # bonus the frame lemma.
                    freq *= bonus
                self._update_counter(self.lemma_frame_counter, lemma, frame.frame, freq)
                self._update_counter(self.frame_lemma_counter, frame.frame, lemma, freq)

    def _update_counter_from_verbalization(
            self, verbalization_file,
            base_freq=1,
            verbalize_freq=100,
            maybe_verbalize_freq=1,
            bonus=10,
    ):
        logger.info('Updating (lemma, frame) counter from Verbalization.')
        with open(verbalization_file, encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split(' ')
                if len(parts) == 4 and parts[0] in ('VERBALIZE', 'MAYBE-VERBALIZE'):
                    lemma = parts[1]
                    frame = parts[3]
                    frame_lemma = re.sub(WORDSENSE_RE, '', frame)

                    freq = verbalize_freq if parts[0] == 'VERBALIZE' else maybe_verbalize_freq
                    if lemma == frame_lemma:  # bonus frame lemma
                        freq *= bonus
                    self._update_counter(self.lemma_frame_counter, lemma, frame, freq)

                    freq = base_freq
                    if lemma == frame_lemma:  # bonus frame lemma
                        freq *= bonus
                    self._update_counter(self.frame_lemma_counter, frame, lemma, freq)

    @staticmethod
    def _get_map_from_counter(counter):
        map_dict = {}
        for key1 in counter:
            freq_key2_set = set()
            highest_freq = 0
            for key2, freq in sorted(counter[key1].items(), key=lambda x: -x[1]):
                if freq >= highest_freq:
                    highest_freq = freq
                    freq_key2_set.add(key2)
                else:
                    break
            map_dict[key1] = freq_key2_set
        return map_dict

    @staticmethod
    def _update_counter(obj, key1, key2, value):
        if key1 not in obj:
            obj[key1] = dict()
        if key2 not in obj[key1]:
            obj[key1][key2] = 0
        obj[key1][key2] += value

    def dump(self, directory):
        logger.info('Dumping Node utilities to {}.'.format(directory))
        with open(os.path.join(directory, 'lemma_frame_counter'), 'w', encoding='utf-8') as f:
            for lemma in self.lemma_frame_counter:
                f.write(lemma + ':\n')
                for frame, freq in sorted(self.lemma_frame_counter[lemma].items(), key=lambda x: -x[1]):
                    f.write('\t{}\t{}\n'.format(frame, freq))
                f.write('\n')
        with open(os.path.join(directory, 'lemma_frame_counter.json'), 'w', encoding='utf-8') as f:
            json.dump(self.lemma_frame_counter, f)

        with open(os.path.join(directory, 'frame_lemma_counter'), 'w', encoding='utf-8') as f:
            for frame in self.frame_lemma_counter:
                f.write(frame + ':\n')
                for lemma, freq in sorted(self.frame_lemma_counter[frame].items(), key=lambda x: -x[1]):
                    f.write('\t{}\t{}\n'.format(lemma, freq))
                f.write('\n')
        with open(os.path.join(directory, 'frame_lemma_counter.json'), 'w', encoding='utf-8') as f:
            json.dump(self.frame_lemma_counter, f)

        with open(os.path.join(directory, 'senseless_node_counter'), 'w', encoding='utf-8') as f:
            for key, value in self.senseless_node_counter.most_common():
                f.write('{}\t{}\n'.format(key, value))
        with open(os.path.join(directory, 'senseless_node_counter.json'), 'w', encoding='utf-8') as f:
            json.dump(self.senseless_node_counter, f)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser('node_utils.py')
    parser.add_argument('--amr_train_files', nargs='+', default=['data/all_amr/train_amr.txt.features'])
    parser.add_argument('--propbank_dir', default='data/misc/propbank-frames-xml-2016-03-08/')
    parser.add_argument('--verbalization_file', default='data/misc/verbalization-list-v1.06.txt')
    parser.add_argument('--dump_dir', default='./temp/')
    parser.add_argument('--train_file_base_freq', type=float, default=1)
    parser.add_argument('--propbank_base_freq', type=float, default=1)
    parser.add_argument('--propbank_bonus', type=float, default=10)
    parser.add_argument('--verbalization_base_freq', type=float, default=1)
    parser.add_argument('--verbalize_freq', type=float, default=100)
    parser.add_argument('--maybe_verbalize_freq', type=float, default=100)
    parser.add_argument('--verbalize_bonus', type=float, default=10)

    args = parser.parse_args()

    nu = NodeUtilities.from_raw(
        args.amr_train_files,
        args.propbank_dir,
        args.verbalization_file,
        args.dump_dir,
        args.train_file_base_freq,
        args.propbank_base_freq,
        args.propbank_bonus,
        args.verbalization_base_freq,
        args.verbalize_freq,
        args.maybe_verbalize_freq,
        args.verbalize_bonus
    )
