import re

from pycorenlp import StanfordCoreNLP

from stog.utils import logging


logger = logging.init_logger()


class FeatureAnnotator:

    NumberTexts = ('hundred', 'thousand', 'million', 'billion', 'trillion',
                   'hundreds', 'thousands', 'millions', 'billions', 'trillions')
    DashedNumbers = re.compile(r'-*\d+-\d+')

    def __init__(self, url, compound_map_file):
        self.nlp = StanfordCoreNLP(url)
        self.nlp_properties = {
            'annotators': "tokenize,ssplit,pos,lemma,ner",
            "tokenize.options": "splitHyphenated=true,normalizeParentheses=false",
            "tokenize.whitespace": False,
            'ssplit.isOneSentence': True,
            'outputFormat': 'json'
        }
        self.compound_map = self.load_compound_map(compound_map_file)

    @staticmethod
    def load_compound_map(file_path):
        """Load a compound map from partial compound word to a list of possible next token in the compound.

        :param file_path: the compound map file.
        "https://github.com/ChunchuanLv/AMR_AS_GRAPH_PREDICTION/blob/master/data/joints.txt"
        :return: a dict from string to list.
        """
        compound_map = {}
        with open(file_path) as f:
            for line in f:
                compounds = line.split()
                precedents = ''
                for token in compounds:
                    if len(precedents) > 0:
                        _precedents = precedents[:-1] # exclude dash
                    else:
                        _precedents = ''
                    if _precedents not in compound_map:
                        compound_map[_precedents] = []
                    compound_map[_precedents].append(token)
                    precedents += token + '-'
        return compound_map

    @staticmethod
    def assert_equal_length(annotation):
        tokens = annotation['tokens']
        for key in annotation:
            if key == 'tokens':
                continue
            value = annotation[key]
            assert len(tokens) == len(value), (
                len(tokens), len(value), '\n', list(zip(tokens, value)), tokens, value)

    def annotate(self, text):
        tokens = self.nlp.annotate(text.strip(), self.nlp_properties)['sentences'][0]['tokens']
        output = dict(
            tokens=[], lemmas=[], pos_tags=[], ner_tags=[]
        )
        for token in tokens:
            output['tokens'].append(token['word'])
            output['lemmas'].append(token['lemma'])
            output['pos_tags'].append(token['pos'])
            output['ner_tags'].append(token['ner'])
        return output

    def __call__(self, text):
        annotation = self.annotate(text)
        original = annotation['tokens']
        annotation = self._combine_compounds(annotation)
        annotation = self._combine_numbers(annotation)
        annotation = self._tag_url_and_split_number(annotation)
        annotation['original'] = original
        return annotation

    def _combine_compounds(self, annotation):
        # Combine tokens in compounds, e.g., 'make up' -> 'make-up'.
        tokens = []
        lemmas = []
        pos_tags = []
        ner_tags = []
        skip = False
        for i, lemma in enumerate(annotation['lemmas']):
            if skip:
                skip = False
            elif len(lemmas) > 0 and lemma in self.compound_map.get(lemmas[-1], []):
                # lemma belongs to a compound.
                lemmas[-1] = lemmas[-1] + '-' + lemma
                tokens[-1] = tokens[-1] + "-" + annotation['tokens'][i]
                pos_tags[-1] = "COMP"
                ner_tags[-1] = "0"
            elif len(lemmas) > 0 and lemma == "-" and i < len(annotation['lemmas']) - 1 \
                and annotation['lemmas'][i + 1] in self.compound_map.get(lemmas[-1], []):
                # lemma is a dash and the next lemma belongs to a compound.
                lemmas[-1] = lemmas[-1] + '-' + annotation['lemmas'][i + 1]
                tokens[-1] = tokens[-1] + '-' + annotation['tokens'][i + 1]
                pos_tags[-1] = "COMP"
                ner_tags[-1] = "0"
                skip = True # skip the next lemma.
            else:
                lemmas.append(lemma)
                tokens.append(annotation['tokens'][i])
                pos_tags.append(annotation['pos_tags'][i])
                ner_tags.append(annotation['ner_tags'][i])

        output = dict(
            tokens=tokens, lemmas=lemmas, pos_tags=pos_tags, ner_tags=ner_tags
        )
        self.assert_equal_length(output)
        return output

    def _combine_numbers(self, annotation):

        def two_combinable_numbers(x, y):
            return x in self.NumberTexts and y != "-"

        def combinable(i, tag):
            return len(lemmas) > 0 and tag == 'CD' and pos_tags[-1] == 'CD' and \
                   two_combinable_numbers(lemmas[-1], annotation['lemmas'][i])

        tokens = []
        lemmas = []
        pos_tags = []
        ner_tags = []

        for i, tag in enumerate(annotation['pos_tags']):
            if combinable(i, tag) :
                lemmas[-1] = lemmas[-1] + ',' + annotation['lemmas'][i]
                tokens[-1] = tokens[-1] + ',' + annotation['tokens'][i]
                pos_tags[-1] = "CD"
            else:
                lemmas.append(annotation['lemmas'][i])
                tokens.append(annotation['tokens'][i])
                pos_tags.append(annotation['pos_tags'][i])
                ner_tags.append(annotation['ner_tags'][i])

        output = dict(
            tokens=tokens, lemmas=lemmas, pos_tags=pos_tags, ner_tags=ner_tags
        )
        self.assert_equal_length(output)
        return output

    def _tag_url_and_split_number(self, annotation):
        tokens = []
        lemmas = []
        pos_tags = []
        ner_tags = []

        for i, lemma in enumerate(annotation['lemmas']):
            if 'http' in lemma or 'www.' in lemma:
                lemmas.append(lemma)
                tokens.append(annotation['tokens'][i])
                pos_tags.append(annotation['pos_tags'][i])
                ner_tags.append("URL")
            elif re.match(self.DashedNumbers, lemma) and annotation['ner_tags'][i] == 'DATE':
                _lemmas = lemma.replace('-', ' - ').split()
                _tokens = annotation['tokens'][i].replace('-', ' - ').split()
                assert len(_lemmas) == len(_tokens), annotation
                for l in _lemmas:
                    if l != '-':
                        pos_tags.append(annotation['pos_tags'][i])
                        ner_tags.append(annotation['ner_tags'][i])
                    else:
                        pos_tags.append(':')
                        ner_tags.append('0')
                lemmas = lemmas + _lemmas
                tokens = tokens + _tokens
            else:
                lemmas.append(annotation['lemmas'][i])
                tokens.append(annotation['tokens'][i])
                pos_tags.append(annotation['pos_tags'][i])
                ner_tags.append(annotation['ner_tags'][i])

        output = dict(
            tokens=tokens, lemmas=lemmas, pos_tags=pos_tags, ner_tags=ner_tags
        )
        self.assert_equal_length(output)
        return output


if __name__ == '__main__':
    import argparse

    from stog.data.dataset_readers.amr_parsing.io import AMRIO

    parser = argparse.ArgumentParser('feature_annotator.py')
    parser.add_argument('files', nargs='+', help='files to annotate.')
    parser.add_argument('--compound_file', default='')

    args = parser.parse_args()

    annotator = FeatureAnnotator('http://localhost:9000', args.compound_file)

    for file_path in args.files:
        logger.info('Processing {}'.format(file_path))
        with open(file_path + '.features', 'w', encoding='utf-8') as f:
            for i, amr in enumerate(AMRIO.read(file_path), 1):
                if i % 1000 == 0:
                    logger.info('{} processed.'.format(i))
                annotation = annotator(amr.sentence)
                amr.tokens = annotation['tokens']
                amr.lemmas = annotation['lemmas']
                amr.pos_tags = annotation['pos_tags']
                amr.ner_tags = annotation['ner_tags']
                amr.original = annotation['original']
                AMRIO.dump([amr], f)
    logger.info('Done!')
