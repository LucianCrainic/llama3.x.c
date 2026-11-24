"""Check the placement evidence in one engine report (schema v2) against what the run asked for.

    check_placement.py report.json --weights-node 2 --embedding-node 0 --kv-node 2

Every weights/embedding region must be on its node, fully verified, with no mismatched or missing page. With
--kv-node the KV caches and the work region must be on that node too, the post-prompt and final KV snapshots must be
verified with no page elsewhere, and active <= resident <= reserved (plus one page per layer of slack, one per layer and KV
head with LLAMA_KV_LAYOUT=head, whose touched positions are a prefix of each head's span) must hold.
Prints one PASS or FAIL line per check and exits non-zero if any failed. Reads only; never touches the engine.
"""
import argparse
from collections import Counter
import json
import sys
from pathlib import Path

PAGE = 4096


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('report')
    parser.add_argument('--placement', help='expected runtime.placement (ddr or hbm)')
    parser.add_argument('--weights-node', type=int)
    parser.add_argument('--embedding-node', type=int, help='node of an untied embedding region, if the model has one')
    parser.add_argument('--kv-node', type=int)
    parser.add_argument('--planned', action='store_true', help='check every arena against the admitted placement plan')
    parser.add_argument('--allow-sampled', action='store_true', help='accept sampled verification with --planned')
    parser.add_argument('--layers', type=int, default=0, help='layer count, for the resident <= reserved slack')
    args = parser.parse_args(argv)
    report = json.loads(Path(args.report).read_text())
    failures = 0

    def check(name, ok, detail=''):
        nonlocal failures
        failures += not ok
        print(('PASS' if ok else 'FAIL'), name, detail)

    check('run completed', report.get('status') == 'completed', str(report.get('status')))
    runtime = report.get('runtime') or {}
    if args.placement:
        check(f'placement is {args.placement}', runtime.get('placement') == args.placement, str(runtime.get('placement')))
    regions = report.get('regions') or []
    by_role = {}
    for region in regions:
        by_role.setdefault(region['role'], []).append(region)

    def region_ok(region, node):
        return (region.get('node') == node and region.get('verified') == 'full' and (region.get('pages_queried') or 0) > 0
                and region.get('pages_mismatched') == 0 and region.get('pages_missing') == 0)

    if args.planned:
        plan = report.get('placement_plan') or {}
        check('physical plan admitted', plan.get('admitted') is True and plan.get('simulated') is False)

        def arena_key(arena):
            return tuple(arena.get(field) for field in ('socket', 'node', 'role', 'bytes'))

        expected = Counter(map(arena_key, plan.get('arenas', [])))
        actual = Counter(map(arena_key, regions))
        check('region inventory matches plan', bool(expected) and actual == expected,
              f'missing={list((expected - actual).elements())} unexpected={list((actual - expected).elements())}')
        verification = {'full', 'sampled'} if args.allow_sampled else {'full'}
        for region in regions:
            verified = (region.get('node') is not None and region.get('verified') in verification
                        and (region.get('pages_queried') or 0) > 0 and region.get('pages_mismatched') == 0
                        and region.get('pages_missing') == 0)
            check(f"socket {region.get('socket')} {region.get('role')} on node {region.get('node')}", verified)
        kv = report.get('kv') or {}
        for label, snapshot in (('final', kv), ('after prompt', kv.get('after_prompt') or {})):
            check(f'{label}: active KV pages verified',
                  snapshot.get('verified') is True and (args.allow_sampled or snapshot.get('full') is True)
                  and snapshot.get('pages_wrong_node') == 0 and snapshot.get('pages_missing') == 0
                  and (snapshot.get('pages_queried') or 0) > 0
                  and snapshot['active_bytes'] <= snapshot['resident_bytes'])
        copies = [a for a in plan.get('arenas', []) if a.get('role') == 'kv.bf16']
        if copies:  # LLAMA_ATTN=amx: the persistent BF16 K/V copy is planned, verified above and reported with the caches
            for label, snapshot in (('final', kv), ('after prompt', kv.get('after_prompt') or {})):
                copy = snapshot.get('bf16_copy') or {}
                check(f'{label}: BF16 K/V copy reported',
                      copy.get('reserved_bytes') == sum(a['bytes'] for a in copies)
                      and 0 < copy.get('active_bytes', 0) <= copy.get('reserved_bytes', 0)
                      and copy.get('active_bytes', 0) <= (copy.get('resident_bytes') or 0),
                      str(copy))
        teams = report.get('teams') or []
        check('worker affinity', len(teams) == runtime.get('sockets')
              and all(team.get('affinity_verified') is True for team in teams))

    for role, node in (('weights', args.weights_node), ('embedding', args.embedding_node)):
        if node is None:
            continue
        found = by_role.get(role, [])
        if role == 'embedding' and not found:
            print('INFO no separate embedding region (tied classifier or none requested)')
            continue
        check(f'{role} regions: {len(found)} found', bool(found))
        for region in found:
            check(f'{role} on node {node}, fully verified, nothing elsewhere or missing', region_ok(region, node),
                  f"node={region.get('node')} verified={region.get('verified')} queried={region.get('pages_queried')} "
                  f"mismatched={region.get('pages_mismatched')} missing={region.get('pages_missing')} "
                  f"thp_bytes={region.get('thp_bytes')} bytes={region.get('bytes')}")
    if args.kv_node is not None:
        for role in ('kv.k', 'kv.v', 'work'):
            found = by_role.get(role, [])
            check(f'{role} region reserved on node {args.kv_node}', bool(found) and all(r.get('node') == args.kv_node for r in found))
        kv = report.get('kv') or {}
        for label, snapshot in (('final', kv), ('after prompt', kv.get('after_prompt') or {})):
            check(f'kv {label}: verified on node {args.kv_node}, no page elsewhere',
                  snapshot.get('verified') is True and snapshot.get('node') == args.kv_node and
                  snapshot.get('pages_wrong_node') == 0 and (snapshot.get('pages_queried') or 0) > 0,
                  f"verified={snapshot.get('verified')} queried={snapshot.get('pages_queried')} "
                  f"elsewhere={snapshot.get('pages_wrong_node')} missing={snapshot.get('pages_missing')} full={snapshot.get('full')}")
            if snapshot.get('verified'):
                slack = PAGE * max(args.layers, 1)
                if (report.get('build') or {}).get('kv_layout') == 'head':  # one partly touched page per (layer, KV head)
                    slack *= max(((report.get('model') or {}).get('configuration') or {}).get('n_kv_heads') or 1, 1)
                check(f'kv {label}: active <= resident <= reserved',
                      snapshot['active_bytes'] <= snapshot['resident_bytes'] <= snapshot['reserved_bytes'] + slack,
                      f"active={snapshot['active_bytes']} resident={snapshot['resident_bytes']} reserved={snapshot['reserved_bytes']}")
    memory = report.get('memory_bytes') or {}
    print('INFO peak_resident=%s owned_weights_peak=%s mapped_checkpoint=%s' % (
        memory.get('peak_resident'), memory.get('owned_weights_peak'), memory.get('mapped_checkpoint')))
    timing = report.get('timing_seconds') or {}
    print('INFO checkpoint_load=%s placement_populate=%s placement_verify=%s' % (
        timing.get('checkpoint_load'), timing.get('placement_populate'), timing.get('placement_verify')))
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
