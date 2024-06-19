import hashlib
from scatterers import cyl, starfish, fat_starfish
import uuid

k = 1 + 0.45j
alpertorder = 6
Npoints = 64

c1 = cyl('d', [0, 0], 3)
c2 = cyl('d', [0, 0], 3)

def get_hash(s):
    key = (s.name, s.bc, s.sc_scale, k, alpertorder, Npoints, 20) # modes hardcoded for the moment
    m = hashlib.md5()
    m.update(str(key).encode('utf-8'))
    new_uuid = uuid.UUID(m.hexdigest())
    return new_uuid

print(get_hash(c1), get_hash(c2))
print(get_hash(c1)==get_hash(c2))