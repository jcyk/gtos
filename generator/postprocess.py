import re
from num2words import num2words
from word2number.w2n import word_to_num
from pycorenlp import StanfordCoreNLP

class PostProcess:
    
    month_map = {
        1:'January',
        2:'February',
        3:'March',
        4:'April',
        5:'May',
        6:'June',
        7:'July',
        8:'August',
        9:'September',
        10:'October',
        11:'November',
        12:'December'
    }
    
    country_map = {
        'India': 'Indian',
        'North Korea': 'North Korean',
        'Germany': 'German',
        'Greece':'Greek',
        'Croatia':'Croatian',
        'Asia':'Asian',
        'Britain':'British',
        'Italy':'Italian',
        'Estonia': 'Estonian',
        'Russia':'Russian',
        'Afghanistan':'Afghan',
        'France':'French',
        'Europe':'European',
        'Iran':'Iranian',
        'Sweden':'Swedish',
        'Brazil':'Brazilian',
        'Mexico':'Mexican',
        'Taiwan':'Taiwanese',
        'Nigeria':'Nigerian',
        'Africa':'African',
        'China':'Chinese',
        'Japan':'japanese',
        'America':'American',
        'Netherlands':'Dutch',
        'Norway':'Norwegian',
        'Israel':'Israeli',
        'Ukraine':'Ukrainian'
    }
    def __init__ (self, retokenize=False, span=True, compound_map_file=None):
        """
        the defualt settings are for development only
        for testing, span must be set to False
        """
        if retokenize:
            nlp = StanfordCoreNLP('http://localhost:9000')
            nlp_properties = {
                'annotators': "tokenize,ssplit",
                "tokenize.options": "splitHyphenated=true,normalizeParentheses=false",
                "tokenize.whitespace": False,
                'ssplit.isOneSentence': True,
                'outputFormat': 'json'
            }
            self.stanford_tokenize = lambda text : [x['word'] for x in nlp.annotate(text, nlp_properties)['sentences'][0]['tokens']]
        self.retokenize = retokenize
        self.span = span
        self.compound_map = self.load_compound_map(compound_map_file)

    @staticmethod
    def load_compound_map(file_path):
        compound_map = dict()
        if file_path is None:
            return compound_map
        for line in  open(file_path).readlines():
            compound = line.strip().split()
            compound_map['-'.join(compound)] = ' '.join(compound)
        return compound_map

    def _find_node(self, abstract, graph):
        ret = []
        for name in graph.name2concept:
            value = graph.name2concept[name]
            if abstract == value:
                ret.append(name)
        #assert len(ret) == 1, (ret)
        if not ret:
            return None
        return ret[0]

    def _check(self, x, abstract, graph):
        """some speical cases where we map abstract symbols to strings, 
        will return None if not in any case
        """
        #China => Chinese
        if x.startswith('NATIONALITY') or x.startswith('COUNTRY'):
            node = self._find_node(x, graph)
            if not node:
                return None
            node1 = None
            for nxt in graph.graph[node]:
                if graph.graph[node][nxt]['label'] == "name_reverse_":
                    node1 = nxt
                    break
            if not node1:
                return None
            if graph.name2concept[node1] == 'country':
                do_transform = False
                for nxt in graph.graph[node1]:
                    if graph.graph[node1][nxt]['label'] == "domain":
                        #or graph.graph[node1][nxt]['label'] == "ARG1_reverse_":
                        do_transform = True
                if do_transform:
                    v = self.country_map.get(abstract['ops'], None)
                    if v is not None:
                        return [v]
            return None
        
        #100 => hundred 
        if re.search(r'^\d+$', x):
            node = self._find_node(x, graph)
            if node is None:
                return None
            for nxt in graph.graph[node]:
                if graph.graph[node][nxt]['label'] == "li_reverse_":
                    return [str(abstract['value'])]
            value = abstract['value']
            if value == 100000:
                return ['hundreds of thousands']
            if int(value) == value:
                if value >= 1000000000 and value % 1000 == 0:
                    v = value / 1000000000
                    if int(v) == v:
                        v = int(v)
                    return [str(v) + ' billion']
                if value >= 1000000 and value % 1000 == 0:
                    v = value / 1000000
                    if int(v) == v:
                        v = int(v)
                    return [str(v) + ' million']
            return None
        
        # 7 => July
        if x.startswith('DATE_ATTRS'):
            assert 'attrs' in abstract or 'edges' in abstract
            if len(abstract['attrs']) > 0:
                xmap = abstract['attrs']
                year = xmap.get('year', None)
                month = xmap.get('month', None)
                day = xmap.get('day', None)
                decade = xmap.get('decade', None)
                century = xmap.get('century', None)
                time = xmap.get('time', None)
                if year and month and day:
                    #30 July 2019
                    return [str(day), self.month_map[month], str(year)]
                if day and month:
                    #April 18th
                    return [self.month_map[month], num2words(day, to='ordinal_num')]
                if year and month:
                    #October 2008
                    return [ self.month_map[month], str(year)]
                if year:
                    #2020
                    return [str(year)]
                if month:
                    #October
                    return [self.month_map[month]]
                if day:
                    #21st
                    return [num2words(day, to='ordinal_num')]
                if decade:
                    #1980s
                    return [str(decade) + 's']
                if century:
                    # 21st
                    return [num2words(century, to='ordinal_num')]
                if time:
                    #return as it is
                    return [time.strip('"')]
            else:
                xmap = abstract['edges']
                weekday = xmap.get('weekday', None)
                dayperiod = xmap.get('dayperiod', None)
                if weekday and dayperiod:
                    return [weekday, dayperiod]
                if weekday:
                    return [weekday]
                if dayperiod:
                    return [dayperiod]
            assert False
            return None
        
        # 3 2 => 3:2
        if x.startswith('SCORE_ENTITY'):
            assert len(abstract['ops']) == 2
            return [str(abstract['ops'][0]), ':', str(abstract['ops'][1])]
            
        # 3 => 3rd
        if x.startswith('ORDINAL_ENTITY'):
            assert len(abstract['ops']) == 1
            return [num2words(int(abstract['ops'][0]), to='ordinal_num')]
                        
    def check(self, abstract, graph):
        """Get the abstract-to-string map"""
        ret = dict()
        for x in abstract:
            y = self._check(x, abstract[x], graph)
            if y is not None:
                ret[x] = y
                continue
                
            xmap = abstract[x]
            if 'ops' in xmap:
                assert 'value' not in xmap
                assert isinstance(xmap['ops'], str) or isinstance(xmap['ops'], list)
                if isinstance(xmap['ops'], list):
                    assert len(xmap['ops'])==1
                    ret[x] = [str(xmap['ops'][0])]
                else:
                    ret[x] = [xmap['ops']]
            elif 'value' in xmap:
                assert 'ops' not in xmap
                assert isinstance(xmap['value'], float) or \
                isinstance(xmap['value'], int) or \
                isinstance(xmap['value'], str)
                ret[x] = [str(xmap['value'])]
        return ret

    def post_process(self, sent, abstract, graph):
        """
        span is for development only
        """
        if self.span:
            _abstract = {}
            for x in abstract:
                _abstract[x] = [abstract[x]['span']]
            abstract = _abstract
        else:
            abstract = self.check(abstract, graph)
        ret = []
        for tok in sent:
            if tok in abstract:
                ret.extend(abstract[tok])
            else:
                tok = self.compound_map.get(tok, tok)
                ret.append(tok)
        ret = ' '.join(ret)

        if self.retokenize:
            ret = ' '.join(self.stanford_tokenize(ret)).lower()
        else:
            ret = ret.lower()
        return ret

def parse_config():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--golden_file', type=str, default='../data/AMR/amr_2.0/test.txt.features')
    parser.add_argument('--pred_file', type=str, default='./epoch718_batch137999_test_out')
    parser.add_argument('--retokenize', type=bool, default=True)
    parser.add_argument('--span', type=bool, default=False)
    parser.add_argument('--compound_map_file', type=str, default='../data/AMR/amr_2.0_utils/joints.txt')
    parser.add_argument('--output', action='store_true')
    
    return parser.parse_args()

if __name__ == '__main__':
    import json
    from extract import read_file
    import sacrebleu
    args = parse_config()
    pp = PostProcess(retokenize=args.retokenize, span=args.span, compound_map_file=args.compound_map_file)

    ref_stream = []
    for line in open(args.golden_file):
        if line.startswith('# ::original '):
            o = json.loads(line[len('# ::original '):].strip())
            ref_stream.append(' '.join(o).lower())
    # gold model output
    graph, gold_sys_stream, _, abstract = read_file(args.golden_file+'.preproc')
    ref_streams = [ref_stream]

    pred_sys_stream = []
    for line in open(args.pred_file):
        if line.startswith('#model output:'):
            ans = line[len('#model output:'):].strip().split()
            pred_sys_stream.append(ans)
            
    prev = [ ' '.join(o) for o in pred_sys_stream]
        
    # choose one (gold or pred) and postprocess
    sys_stream = pred_sys_stream
    sys_stream = [ pp.post_process(o, abstract[i], graph[i]) for i, o in enumerate(sys_stream)]

    bleu = sacrebleu.corpus_bleu(sys_stream, ref_streams, 
                          force=True, lowercase=True, 
                          tokenize='none').score
    chrf = sacrebleu.corpus_chrf(sys_stream, ref_stream)
    all_sent_chrf = [sacrebleu.sentence_chrf(x, y) for x, y in zip(sys_stream, ref_stream)]
    avg_sent_chrf = sum(all_sent_chrf) / len(all_sent_chrf)
    if args.output:
        with open(args.pred_file+'.final', 'w') as fo:
            for x in sys_stream:
                fo.write(x+'\n')

        with open(args.pred_file+'.ref', 'w') as fo:
            for x in ref_stream:
                fo.write(x+'\n')
    print (avg_sent_chrf)
    print (bleu, chrf)
