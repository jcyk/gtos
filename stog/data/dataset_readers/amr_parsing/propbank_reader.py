import os
from collections import namedtuple
import xml.etree.ElementTree as ET


Frame = namedtuple('Frame', ['frame', 'lemma', 'sense'])


class PropbankReader:

    def __init__(self, directory):
        """
        Load Propbank frames from the given directory, and build two data structures:
            frame_lemma_set: each frame consists of two parts, frame lemma and frame sense,
                e.g., `run-01`. frame_lemma_set collects all frame lemmas.
            lemma_map: besides frame lemmas, a frame could be invoked by other lemmas.
                Here we build a dict that maps a lemma to a set of frames it could invoke.

        :param directory: string.
        """
        self.frame_lemma_set = set()
        self.lemma_map = dict()
        self.directory = directory
        self._load()

    def _load(self):
        """
        Load Propbank frame files (.xml).
        """
        for file_name in os.listdir(self.directory):
            if file_name.endswith('.xml'):
                file_path = os.path.join(self.directory, file_name)
                self._parse_file(file_path)

    def _parse_file(self, file_path):
        """
        Parse a propbank frame file.
        :param file_path: the frame file path.
        """
        tree = ET.parse(file_path)
        for child in tree.getroot():
            if child.tag == 'predicate':
                self._add_predicate(child)

    def _add_predicate(self, node):
        """
        Update frame_lemma_set and lemma_map given this predicate.
        """
        # Get the primary lemma of the frame.
        lemma = node.attrib['lemma'].replace('_', '-')  # AMR use dash.
        for child in node:
            if child.tag == 'roleset':
                # Split sense from frame id; get `frame_lemma` and `sense`.
                frame_id = child.attrib['id']
                if '.' not in frame_id:
                    parts = frame_id.split('-')
                    if len(parts) == 1:
                        frame_lemma = parts[0].replace('_', '-')
                        sense = None
                    else:
                        frame_lemma, sense = parts
                else:
                    frame_lemma, sense = frame_id.replace('_', '-').split('.')
                # Get frame id in AMR convention.
                frame = frame_id.replace('_', '-').replace('.', '-')    # AMR use dash
                # Put them together.
                frame_obj = Frame(frame, frame_lemma, sense)

                # Update
                self.frame_lemma_set.add(frame_lemma)
                self._update_lemma_map(self.lemma_map, lemma, frame_obj)

                aliases = child.find('aliases')
                if aliases:
                    for alias in aliases.findall('alias'):
                        alias_text = alias.text.replace('_', '-')
                        if alias_text != frame_lemma and alias_text not in self.lemma_map:
                            self._update_lemma_map(self.lemma_map, alias_text, frame_obj)

    def _update_lemma_map(self, obj, key, value):
        if key not in obj:
            obj[key] = set()
        obj[key].add(value)



