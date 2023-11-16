import tracemalloc
from tracemalloc import Snapshot

tracemalloc.start()
sn = tracemalloc.take_snapshot()
s = tracemalloc.Snapshot.load('tests/mem_tests/snap.dat')
ts = s.statistics('lineno')