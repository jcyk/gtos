def merge_files(dep_file, head_file, tok_file, tgt_file, out_file):
    drop = 0
    dep = open(dep_file).readlines()
    head = open(head_file).readlines()
    tok = open(tok_file).readlines()
    tgt = open(tgt_file).readlines()
    assert len(dep) == len(head) == len(tok) == len(tgt), (len(dep), len(head), len(tok), len(tgt))
    with open(out_file, 'w') as fo:
        for d, h, s, t in zip(dep, head, tok, tgt):
            dd = d.rstrip('\n').split(' ')
            hh = h.rstrip('\n').split(' ')
            ss = s.rstrip('\n').split(' ')
            tt = t.rstrip('\n').split(' ')
            assert len(dd) == len(hh) == len(ss), (len(dd), len(hh), len(ss), d, h, s, ss[-1])
            if not 0.5*len(dd) <= len(tt) <= 4*len(dd):
                drop += 1
            else:
                fo.write(d+h+s+t)
    print (drop)
import sys

if __name__ == "__main__":
    dep_file= sys.argv[1]
    head_file = sys.argv[2]
    tok_file = sys.argv[3]
    tgt_file = sys.argv[4]
    out_file = sys.argv[5]
    merge_files(dep_file, head_file, tok_file, tgt_file, out_file)