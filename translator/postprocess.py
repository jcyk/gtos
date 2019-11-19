import re

class PostProcess:
    
    def __init__ (self):
        pass

    def post_process(self, sent):
        return re.sub(r'(@@ )|(@@ ?$)', '', sent)

def parse_config():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--golden_file', type=str, default='../../data/en2de/newstest2015-ende-ref.tok.de')
    parser.add_argument('--pred_file', type=str, default='./check/epoch718_batch137999_test_out')
    parser.add_argument('--output', action='store_true')

    return parser.parse_args()

if __name__ == '__main__':
    import json
    from extract import read_file
    import sacrebleu
    args = parse_config()
    pp = PostProcess()

    ref_stream = []
    for line in open(args.golden_file):
        ref_stream.append(line.rstrip('\n'))
    ref_streams = [ref_stream]

    # gold model output
    pred_sys_stream = []
    for line in open(args.pred_file):
        if line.startswith('#model output:'):
            ans = line[len('#model output:'):].rstrip('\n')
            pred_sys_stream.append(ans)
            
    sys_stream = [ pp.post_process(o) for o in pred_sys_stream]

    if args.output:
        with open(args.pred_file+'.postproc', 'w') as fo:
            for x in sys_stream:
                fo.write(x + '\n')

        
    bleu = sacrebleu.corpus_bleu(sys_stream, ref_streams, 
                          force=True, lowercase=False, 
                          tokenize='none').score
    chrf = sacrebleu.corpus_chrf(sys_stream, ref_stream)
    all_sent_chrf = [sacrebleu.sentence_chrf(x, y) for x, y in zip(sys_stream, ref_stream)]
    avg_sent_chrf = sum(all_sent_chrf) / len(all_sent_chrf)
    print (avg_sent_chrf)
    print (bleu, chrf)
