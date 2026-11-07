"""Topology discovery against synthetic sysfs trees (tests/sysfs_fixtures.py) through tests/numa_probe.c."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import sysfs_fixtures as fx

BUILD = Path(sys.argv.pop(1)).resolve()


class TopologyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        root = Path(cls.temp.name)
        cls.flat = fx.flat(root / 'flat')
        cls.no_numaperf = fx.flat(root / 'no-numaperf', numaperf=False)
        cls.cache = fx.cache(root / 'cache')
        cls.snc = fx.snc4(root / 'snc4')
        cls.hole = fx.flat(root / 'hole', possible='0-7')
        cls.smt = fx.build(root / 'smt', cores=4, smt=2)
        cls.slow = fx.flat(root / 'slow', hbm_bandwidth=100000)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def probe(self, sysfs, *args, success=True):
        result = subprocess.run([str(BUILD/'numa_probe'), '--sysfs', str(sysfs), *map(str, args)],
                                capture_output=True, text=True, timeout=30)
        if success:
            self.assertEqual(result.returncode, 0, result.stderr)
            return json.loads(result.stdout)
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn('[ERROR]', result.stderr)
        self.assertNotIn('runtime error', result.stderr)
        return result.stderr

    def test_flat_numaperf(self):
        t = self.probe(self.flat)
        self.assertEqual(t['classification'], 'numaperf')
        self.assertEqual((t['nsockets'], t['max_node'], t['mems_allowed']), (2, 3, [0, 1, 2, 3]))
        s0, s1 = t['sockets']
        self.assertEqual(s0['cpus'], list(range(0, 56)))
        self.assertEqual(s1['cpus'], list(range(56, 112)))
        self.assertEqual((s0['ddr_node'], s0['hbm_node'], s1['ddr_node'], s1['hbm_node']), (0, 2, 1, 3))
        self.assertEqual((s0['hbm_bytes'], s0['ddr_bytes']), (64 * fx.GIB * 1024, 256 * fx.GIB * 1024))
        self.assertTrue(s0['ddr_allowed'] and s0['hbm_allowed'] and s1['ddr_allowed'] and s1['hbm_allowed'])
        self.probe(self.flat, '--sockets', 2, '--placement', 'hbm')

    def test_cache_mode_has_no_hbm(self):
        t = self.probe(self.cache)
        self.assertEqual(t['classification'], 'ddr-only')
        self.assertEqual([s['hbm_node'] for s in t['sockets']], [-1, -1])
        self.probe(self.cache, '--placement', 'ddr', '--sockets', 2)
        message = self.probe(self.cache, '--placement', 'hbm', success=False)
        self.assertIn('no HBM node', message)

    def test_snc_and_more_than_two_sockets_are_rejected(self):
        self.assertIn('sub-NUMA', self.probe(self.snc, success=False))
        self.assertIn('more than two sockets', self.probe(fx.build(Path(self.temp.name)/'four', packages=4, cores=2,
                                                                  hbm=()), success=False))

    def test_missing_numaperf_needs_the_explicit_map(self):
        message = self.probe(self.no_numaperf, success=False)
        self.assertIn('explicit:C0/H0/D0,C1/H1/D1', message)
        t = self.probe(self.no_numaperf, '--topology', 'explicit:0/2/0,1/3/1')
        self.assertEqual(t['classification'], 'explicit')
        self.assertEqual([(s['ddr_node'], s['hbm_node']) for s in t['sockets']], [(0, 2), (1, 3)])

    def test_explicit_map_is_checked_against_the_machine(self):
        for spec, text in [('explicit:0/3/0,1/2/1', 'nearer'),   # HBM crossed: distance relation fails
                           ('explicit:0/2/1,1/3/0', 'not the CPU-bearing DDR node'),
                           ('explicit:0/1/0,1/3/1', 'not an online CPU-less'),
                           ('explicit:0/2/0', 'machine has 2'),
                           ('explicit:0/2/0,0/3/1', 'not the CPU-bearing DDR node'),
                           ('explicit:0/2/0,5/3/1', 'package 5 does not exist'),
                           ('explicit:0/2/0,1/2/1', 'share a node'),
                           ('explicit:nonsense', 'invalid explicit'),
                           ('bogus', 'invalid topology')]:
            with self.subTest(spec=spec):
                self.assertIn(text, self.probe(self.no_numaperf, '--topology', spec, success=False))

    def test_memory_only_node_that_is_not_faster_is_not_hbm(self):
        t = self.probe(self.slow)
        self.assertEqual([s['hbm_node'] for s in t['sockets']], [-1, -1])
        self.assertIn('no HBM node', self.probe(self.slow, '--placement', 'hbm', success=False))

    def test_offline_node_ids_size_the_mask(self):
        t = self.probe(self.hole)
        self.assertEqual((t['max_node'], t['mems_allowed']), (7, [0, 1, 2, 3]))
        self.assertEqual(t['mems_allowed_list'], '0-3')

    def test_restricted_memory_nodes(self):
        t = self.probe(self.flat, '--mems', '0,1,2')
        self.assertEqual(t['mems_allowed'], [0, 1, 2])
        s0, s1 = t['sockets']
        self.assertTrue(s0['hbm_allowed'] and s1['ddr_allowed'])
        self.assertFalse(s1['hbm_allowed'])
        self.assertEqual(s1['hbm_node'], 3)
        message = self.probe(self.flat, '--mems', '0,1,2', '--sockets', 2, '--placement', 'hbm', success=False)
        self.assertIn('node 3', message)
        self.assertIn('allowed memory nodes (0-2)', message)
        # One socket on node 2 is still fine; so is DDR placement on both.
        self.probe(self.flat, '--mems', '0,2', '--cpus', '0-55', '--sockets', 1, '--placement', 'hbm')
        self.probe(self.flat, '--mems', '0,1,2', '--sockets', 2, '--placement', 'ddr')

    def test_restricted_cpus(self):
        t = self.probe(self.flat, '--cpus', '0-27,56-60')
        self.assertEqual([s['cpus'] for s in t['sockets']], [list(range(28)), list(range(56, 61))])
        t = self.probe(self.flat, '--cpus', '0-55')
        self.assertEqual((t['nsockets'], t['sockets'][0]['package']), (1, 0))
        message = self.probe(self.flat, '--cpus', '0-55', '--sockets', 2, success=False)
        self.assertIn('only 1', message)
        t = self.probe(self.flat, '--cpus', '60-70')            # only the second socket is allowed
        self.assertEqual((t['nsockets'], t['sockets'][0]['package'], t['sockets'][0]['hbm_node']), (1, 1, 3))
        self.assertIn('no allowed CPU', self.probe(self.flat, '--cpus', '500', success=False))

    def test_smt_keeps_one_cpu_per_core(self):
        t = self.probe(self.smt, '--topology', 'auto')
        self.assertEqual([s['cpus'] for s in t['sockets']], [[0, 1, 2, 3], [4, 5, 6, 7]])
        # The lowest allowed sibling represents a core, so CPU 8 stands for core 0 (its sibling 0 is not allowed).
        t = self.probe(self.smt, '--cpus', '1-2,8-9')
        self.assertEqual([s['cpus'] for s in t['sockets']], [[1, 2, 8]])

    def test_virtual_topology(self):
        t = self.probe(self.flat, '--topology', 'virtual:2x3')
        self.assertEqual((t['classification'], t['nsockets']), ('virtual', 2))
        self.assertEqual([(s['ncpus'], s['cpus'], s['ddr_node']) for s in t['sockets']], [(3, None, -1)] * 2)
        self.probe(self.flat, '--topology', 'virtual:1x1', '--sockets', 1)
        self.assertIn('needs real NUMA nodes', self.probe(self.flat, '--topology', 'virtual:2x2', '--placement', 'hbm',
                                                          success=False))
        for bad in ('virtual:3x2', 'virtual:2', 'virtual:2x0', 'virtual:2x2x'):
            self.assertIn('invalid virtual topology', self.probe(self.flat, '--topology', bad, success=False))

    def test_broken_fixtures_fail_cleanly(self):
        broken = Path(self.temp.name)/'broken'
        shutil.copytree(self.flat, broken)
        (broken/'devices/system/node/node1/distance').write_text('10 21\n')
        self.assertIn('distances', self.probe(broken, success=False))
        (broken/'devices/system/node/node1/distance').unlink()
        self.assertIn('cannot read', self.probe(broken, success=False))
        self.assertIn('cannot read', self.probe(Path(self.temp.name)/'absent', success=False))


class RegionTests(unittest.TestCase):
    def run_regions(self, *args, env=None):
        import os
        return subprocess.run([str(BUILD/'test-regions'), *map(str, args)], capture_output=True, text=True, timeout=60,
                              env={**os.environ, **(env or {})})

    def test_unit_tests(self):
        result = self.run_regions()
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_failure_during_startup_leaves_nothing_behind(self):
        result = self.run_regions('fail3')
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn('[ERROR] injected startup failure after 3 regions', result.stderr)
        self.assertEqual(result.stdout.strip(), 'live=0 current=0')

    @unittest.skipUnless(Path('/sys/devices/system/node/node0').exists(), 'needs a NUMA Linux host')
    def test_bind_to_node0_and_injected_failures(self):
        result = self.run_regions('bind', '0', 4 << 20)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('node 0', result.stdout)
        for fault, text in [('EPERM', 'Operation not permitted'), ('ENOSYS', 'not implemented'), ('EINVAL', 'Invalid')]:
            with self.subTest(fault=fault):
                result = self.run_regions('bind', '0', 4 << 20, env={'LLAMA_NUMA_FAULT': fault})
                self.assertEqual(result.returncode, 1)
                self.assertIn('cannot bind', result.stderr)
                self.assertIn('NUMA node 0', result.stderr)
                self.assertIn('allowed memory nodes:', result.stderr)
        result = self.run_regions('bind', '999', 4096)
        self.assertEqual(result.returncode, 1)
        self.assertIn('not a possible node', result.stderr)

    def verify(self, *args):
        result = self.run_regions('verify', *map(str, args))
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout)

    def test_portable_region_has_nothing_to_verify(self):
        v = self.verify(-1, 1 << 20, 'half')
        self.assertEqual((v['verify_status'], v['queried'], v['populated_bytes']), (-1, 0, 1 << 19))

    @unittest.skipUnless(Path('/sys/devices/system/node/node0').exists(), 'needs a NUMA Linux host')
    def test_verify_counts_unpopulated_and_correct_pages(self):
        v = self.verify(0, 64 << 20, 'half')
        self.assertEqual(v['verify_status'], 0)
        self.assertTrue(v['full'])
        self.assertEqual(v['wrong_node'] + v['errors'], 0)
        self.assertAlmostEqual(v['missing'] / v['queried'], 0.5, delta=0.02)
        self.assertAlmostEqual(v['on_node'] / v['queried'], 0.5, delta=0.02)
        self.assertEqual(v['numa_maps_policy'], 'bind:0')
        v = self.verify(0, 64 << 20, 'all')
        self.assertEqual((v['on_node'], v['missing']), (v['queried'], 0))
        v = self.verify(0, 64 << 20, 'all', 'samples=64')
        self.assertFalse(v['full'])
        self.assertLessEqual(v['queried'], 64)
        self.assertEqual(v['scheme'], 'stride+extent-endpoints')
        self.assertEqual(v['on_node'], v['queried'])
        v = self.verify(0, 64 << 20, 'all', 'expect=1')            # wrong expectation is reported, not hidden
        self.assertEqual((v['on_node'], v['wrong_node']), (0, v['queried']))


class WorkerTests(unittest.TestCase):
    def run_workers(self, *args, timeout=120):
        return subprocess.run([str(BUILD/'test-workers'), *map(str, args)], capture_output=True, text=True,
                              timeout=timeout)

    def test_unit_tests(self):
        result = self.run_workers()
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_barrier_stress_with_random_delays(self):
        for threads in (2, 5, 8):
            with self.subTest(threads=threads):
                result = self.run_workers('barrier', 100000, threads)
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_fail_in_a_worker_ends_the_process(self):
        for mode, text in [('fail-worker', 'injected failure in worker 1'), ('fail-leader', 'in the leader')]:
            with self.subTest(mode=mode):
                result = self.run_workers(mode, timeout=30)
                self.assertEqual(result.returncode, 1, result.stderr)
                self.assertIn(text, result.stderr)

    @unittest.skipUnless(sys.platform.startswith('linux'), 'pinning is Linux-only')
    def test_pinned_workers_read_back_their_cpu(self):
        result = self.run_workers('affinity', 4)
        self.assertEqual(result.returncode, 0, result.stderr)
        data = json.loads(result.stdout)
        if data['supported']:
            self.assertEqual(data['verified'], 1)
            self.assertTrue(all(w['assigned'] == w['observed'] for w in data['workers']))


class RealMachineTests(unittest.TestCase):
    def run_probe(self, *args):
        return subprocess.run([str(BUILD/'numa_probe'), *args], capture_output=True, text=True, timeout=30)

    def test_virtual_topology_from_the_environment_path(self):
        result = self.run_probe('--topology', 'virtual:1x2')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)['classification'], 'virtual')

    @unittest.skipUnless(sys.platform.startswith('linux'), 'real /sys and affinity are Linux-only')
    def test_real_machine_is_classified_or_refused(self):
        result = self.run_probe()
        if result.returncode == 0:
            t = json.loads(result.stdout)
            self.assertGreaterEqual(t['nsockets'], 1)
            self.assertTrue(all(s['cpus'] for s in t['sockets']))
        else:
            self.assertIn('[ERROR] topology discovery failed', result.stderr)


if __name__ == '__main__':
    unittest.main()
