#!/usr/bin/env python3
"""
Analyze Julia vs C++ PRELEG/STEP/LEG logs for a chunk, find earliest state divergence,
and write a CSV suitable for plotting.

Usage:
  python3 scripts/analyze_divergence.py --chunk chunk00973 --root test/fixtures/chunks --out reports/divergence_chunk00973.csv
"""
import argparse
import json
from pathlib import Path

def load_log(path):
    events=[]
    splits=[]
    with open(path,'r') as f:
        for line in f:
            s=line.strip()
            if not s: continue
            if s.startswith('SPLIT '):
                try:
                    payload=json.loads(s[6:])
                    splits.append(payload)
                except Exception:
                    pass
                continue
            if s.startswith('CPPDBG '): s=s[7:]
            if s.startswith('JULIADB '): s=s[8:]
            try:
                ev=json.loads(s)
            except Exception:
                continue
            events.append(ev)
    return events, splits

def key_preleg(ev):
    return (int(ev.get('timestamp',-1)), ev.get('stage'), tuple(ev.get('pair',[None,None])))

def first_preleg_divergence(j_events,c_events, tol=1e-8):
    jp={key_preleg(e):e for e in j_events if e.get('event')=='PRELEG' and e.get('pair')==[0,1]}
    cp={key_preleg(e):e for e in c_events if e.get('event')=='PRELEG' and e.get('pair')==[0,1]}
    common=sorted(set(jp.keys()) & set(cp.keys()))
    for k in common:
        je=jp[k]; ce=cp[k]
        # compare reserves and price_before
        jr=je.get('reserves',[]); cr=ce.get('reserves',[])
        if len(jr)!=len(cr):
            return k,(je,ce)
        diff = sum(abs(jr[i]-cr[i]) for i in range(len(jr)))
        if abs(je.get('price_before',0)-ce.get('price_before',0))>tol or diff>tol:
            return k,(je,ce)
    return None,(None,None)

def write_csv(out_path, j_events, c_events):
    fields=["timestamp","stage","price_before_j","price_before_c","dx_j","dx_c","dy_j","dy_c","x0_j","x1_j","x0_c","x1_c"]
    with open(out_path,'w') as f:
        f.write(",".join(fields)+"\n")
        # iterate over union of PRELEG keys
        jpre=[e for e in j_events if e.get('event')=='PRELEG' and e.get('pair')==[0,1]]
        cpre=[e for e in c_events if e.get('event')=='PRELEG' and e.get('pair')==[0,1]]
        jm={key_preleg(e):e for e in jpre}; cm={key_preleg(e):e for e in cpre}
        keys=sorted(set(jm.keys()) | set(cm.keys()))
        # for each timestamp/stage, also find matching LEG to get dx/dy
        def leg_map(events):
            m={}
            for e in events:
                if e.get('event')=='LEG' and e.get('pair')==[0,1]:
                    k=(int(e.get('timestamp',-1)), e.get('stage'), tuple(e.get('pair')))
                    m[k]=e
            return m
        jl=leg_map(j_events); cl=leg_map(c_events)
        for k in keys:
            ts,st,p=k
            je=jm.get(k); ce=cm.get(k)
            jdx=jdy=None; cdx=cdy=None
            if (ts,st,p) in jl:
                jdx=jl[(ts,st,p)].get('dx'); jdy=jl[(ts,st,p)].get('dy')
            if (ts,st,p) in cl:
                cdx=cl[(ts,st,p)].get('dx'); cdy=cl[(ts,st,p)].get('dy')
            pbj=je.get('price_before') if je else None
            pbc=ce.get('price_before') if ce else None
            rj=je.get('reserves') if je else [None,None]
            rc=ce.get('reserves') if ce else [None,None]
            row=[ts,st,pbj,pbc,jdx,cdx,jdy,cdy,rj[0],rj[1],rc[0],rc[1]]
            f.write(",".join(str(x) for x in row)+"\n")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--chunk', required=True)
    ap.add_argument('--root', default='test/fixtures/chunks')
    ap.add_argument('--out', default=None)
    args=ap.parse_args()
    chunk_dir=Path(args.root)/args.chunk
    jlog=chunk_dir/(f'julia_log.{args.chunk}.jsonl')
    clog=chunk_dir/('cpp_log.jsonl')
    j_events, j_splits = load_log(jlog)
    c_events, c_splits = load_log(clog)
    k,(je,ce)=first_preleg_divergence(j_events,c_events)
    if k is None:
        print('No PRELEG divergence found')
    else:
        print('Earliest PRELEG divergence at',k)
        print('Julia pre reserves:', je.get('reserves'))
        print(' C++  pre reserves:', ce.get('reserves'))
        print('Julia price_before:', je.get('price_before'), ' C++:', ce.get('price_before'))
    if args.out:
        write_csv(args.out, j_events, c_events)
        print('Wrote CSV to', args.out)
    # Print first M SPLITs and PRELEGs side-by-side
    M = 20
    j_split01 = [s for s in j_splits if tuple(s.get('pair',[None,None]))==(0,1)][:M]
    c_pre01 = [e for e in c_events if e.get('event')=='PRELEG' and e.get('pair')==[0,1]][:M]
    print('\nFirst {} Julia SPLITs (pair 0-1):'.format(len(j_split01)))
    for s in j_split01:
        print('  J', s.get('timestamp'), s.get('leg_dir'), 'hi', s.get('high'), 'lo', s.get('low'), 'vol', s.get('volume'), 'is_last', s.get('is_last'))
    print('\nFirst {} C++ PRELEGs (pair 0-1):'.format(len(c_pre01)))
    for e in c_pre01:
        print('  C', e.get('timestamp'), e.get('stage'), 'pb', e.get('price_before'), 'v_before', e.get('volume_before'))

if __name__=='__main__':
    main()
