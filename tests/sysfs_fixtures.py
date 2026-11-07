"""Synthetic sysfs trees shaped like /sys on CRESCO8 hbm14 (contents are made up)."""
from pathlib import Path

GIB = 1024 * 1024


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + '\n')


def _rng(cpus):
    cpus = sorted(cpus)
    out, start = [], None
    for i, c in enumerate(cpus):
        if start is None:
            start = prev = c
        elif c == prev + 1:
            prev = c
        else:
            out.append((start, prev))
            start = prev = c
    if start is not None:
        out.append((start, prev))
    return ','.join(str(a) if a == b else f'{a}-{b}' for a, b in out)


def build(root, *, packages=2, cores=56, smt=1, ddr_per_package=1, hbm=(2, 3), numaperf=True,
          possible=None, hbm_bandwidth=550000, ddr_bandwidth=300000, mem_kib=256 * GIB):
    """Package p owns cores [p*cores, (p+1)*cores) and, with smt=2, siblings offset by packages*cores.

    DDR nodes are numbered first (package-major), then the CPU-less `hbm` nodes. HBM node hbm[p] is
    nearest to package p (distance 13 local, 23 remote, as on hbm14)."""
    root = Path(root)
    total_cores = packages * cores
    ncpu = total_cores * smt
    _write(root / 'devices/system/cpu/online', f'0-{ncpu - 1}')
    for cpu in range(ncpu):
        core = cpu % total_cores
        package = core // cores
        siblings = [core + t * total_cores for t in range(smt)]
        base = root / f'devices/system/cpu/cpu{cpu}/topology'
        _write(base / 'physical_package_id', str(package))
        _write(base / 'core_id', str(core % cores))
        _write(base / 'thread_siblings_list', _rng(siblings))
    ddr_nodes = packages * ddr_per_package
    hbm = tuple(hbm)[:packages] if hbm else ()
    nodes = list(range(ddr_nodes)) + list(hbm)
    online = sorted(nodes)
    _write(root / 'devices/system/node/online', _rng(online))
    _write(root / 'devices/system/node/possible', possible or _rng(online))
    per_node = cores // ddr_per_package
    for n in online:
        base = root / f'devices/system/node/node{n}'
        if n < ddr_nodes:
            package, slot = divmod(n, ddr_per_package)
            first = package * cores + slot * per_node
            cpus = [first + i + t * total_cores for i in range(per_node) for t in range(smt)]
            _write(base / 'cpulist', _rng(cpus))
            _write(base / 'meminfo', f'Node {n} MemTotal:       {mem_kib} kB')
        else:
            _write(base / 'cpulist', '')
            _write(base / 'meminfo', f'Node {n} MemTotal:       {64 * GIB} kB')
        row = []
        for m in online:
            if m == n:
                row.append(10)
            elif n < ddr_nodes and m < ddr_nodes:
                row.append(21 if n // ddr_per_package != m // ddr_per_package else 12)
            elif n >= ddr_nodes and m >= ddr_nodes:
                row.append(20)
            else:
                ddr, other = (n, m) if n < ddr_nodes else (m, n)
                package = ddr // ddr_per_package
                row.append(13 if hbm and hbm[package] == other else 23)
        _write(base / 'distance', ' '.join(map(str, row)))
        if numaperf:
            init = base / 'access0/initiators'
            if n < ddr_nodes:
                (init / f'node{n}').mkdir(parents=True, exist_ok=True)
                _write(init / 'read_bandwidth', str(ddr_bandwidth))
            else:
                package = hbm.index(n)
                for slot in range(ddr_per_package):
                    (init / f'node{package * ddr_per_package + slot}').mkdir(parents=True, exist_ok=True)
                _write(init / 'read_bandwidth', str(hbm_bandwidth))
    return root


def flat(root, **kw):
    """hbm14 in Flat mode: 4 nodes, 112 cores, DDR 0/1, HBM 2/3."""
    return build(root, **kw)


def cache(root, **kw):
    """Cache mode: HBM is invisible, two DDR nodes only."""
    return build(root, hbm=(), **kw)


def snc4(root, **kw):
    """Sub-NUMA clustering: four CPU-bearing nodes per package."""
    return build(root, ddr_per_package=4, hbm=(), **kw)
