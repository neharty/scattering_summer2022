import numpy as np
import unittest
import sys

from Ncyl.tds import scattering, cyl
from precond import preconditioner

class TestCylSetGet(unittest.TestCase):
    def test_dirichlet(self):
        pos_d = np.random.random(size=2)
        rad_d = np.random.random() + 0.1
        testcyl = cyl('d', pos=pos_d, radius = rad_d)
        
        self.assertEqual(testcyl.get_label(), 0, "incorrect label")
        self.assertEqual(testcyl.get_bc(), 'd', 'incorrect bc')
        self.assertEqual(testcyl.get_pos().tolist(), pos_d.tolist(), 'incorrect position')
        self.assertEqual(testcyl.get_radius(), rad_d, 'incorrect radius')
        self.assertEqual(testcyl.get_sum_index(), None, 'incorrect sum index')
        self.assertEqual(testcyl.get_impedance(), None, 'incorrect impedance')

        self.assertEqual(testcyl.label, 0, "incorrect label")
        self.assertEqual(testcyl.bc, 'd', 'incorrect bc')
        self.assertEqual(testcyl.pos.tolist(), pos_d.tolist(), 'incorrect position')
        self.assertEqual(testcyl.radius, rad_d, 'incorrect radius')
        self.assertEqual(testcyl.sum_index, None, 'incorrect sum index')
        self.assertEqual(testcyl.impedance, None, 'incorrect impedance')

    def test_neumann(self):
        pos_n = np.random.random(size=2)
        rad_n = np.random.random() + 0.1
        testcyl = cyl('n', pos=pos_n, radius = rad_n)
        
        self.assertEqual(testcyl.get_label(), 0, "incorrect label")
        self.assertEqual(testcyl.get_bc(), 'n', 'incorrect bc')
        self.assertEqual(testcyl.get_pos().tolist(), pos_n.tolist(), 'incorrect position')
        self.assertEqual(testcyl.get_radius(), rad_n, 'incorrect radius')
        self.assertEqual(testcyl.get_sum_index(), None, 'incorrect sum index')
        self.assertEqual(testcyl.get_impedance(), None, 'incorrect impedance')
        
        self.assertEqual(testcyl.label, 0, "incorrect label")
        self.assertEqual(testcyl.bc, 'n', 'incorrect bc')
        self.assertEqual(testcyl.pos.tolist(), pos_n.tolist(), 'incorrect position')
        self.assertEqual(testcyl.radius, rad_n, 'incorrect radius')
        self.assertEqual(testcyl.sum_index, None, 'incorrect sum index')
        self.assertEqual(testcyl.impedance, None, 'incorrect impedance')

    def test_impedance(self):
        pos_i = np.random.random(size=2)
        rad_i = np.random.random() + 0.1
        
        with self.assertRaises(NotImplementedError): 
            cyl('i', pos=pos_i, radius=rad_i)
        
        testcyl=cyl('i', pos = pos_i, radius=rad_i, impedance = 3+1j)
        
        self.assertEqual(testcyl.get_impedance(), 3+1j, 'incorrect impedance')
        self.assertEqual(testcyl.get_label(), 0, "incorrect label")
        self.assertEqual(testcyl.get_bc(), 'i', 'incorrect bc')
        self.assertEqual(testcyl.get_pos().tolist(), pos_i.tolist(), 'incorrect position')
        self.assertEqual(testcyl.get_radius(), rad_i, 'incorrect radius')
        self.assertEqual(testcyl.get_sum_index(), None, 'incorrect sum index')
        
        self.assertEqual(testcyl.impedance, 3+1j, 'incorrect impedance')
        self.assertEqual(testcyl.label, 0, "incorrect label")
        self.assertEqual(testcyl.bc, 'i', 'incorrect bc')
        self.assertEqual(testcyl.pos.tolist(), pos_i.tolist(), 'incorrect position')
        self.assertEqual(testcyl.radius, rad_i, 'incorrect radius')
        self.assertEqual(testcyl.sum_index, None, 'incorrect sum index')

class ScatteringTest(unittest.TestCase):
    def test_attributes(self):
        pos_i = np.random.random(size=2)
        rad_i = np.random.random() + 0.1

        testcyl = [cyl('n', pos = pos_i, radius = rad_i)]
        sc = scattering(testcyl)

        self.assertEqual(sc.cyls[0].label, 0, "incorrect label")
        self.assertEqual(sc.cyls[0].bc, 'n', 'incorrect bc')
        self.assertEqual(sc.cyls[0].pos.tolist(), pos_i.tolist(), 'incorrect position')
        self.assertEqual(sc.cyls[0].radius, rad_i, 'incorrect radius')
        self.assertEqual(sc.cyls[0].sum_index, testcyl[0].sum_index, 'incorrect sum index')

    def test_dirichlet(self):
        rads = np.random.uniform(0.1, 0.4, size=3)
        sc1 = scattering([cyl('d', np.array([0,0]), 0.5)], method='explicit')
        sc2 = scattering([cyl('d', np.array([0,0]), rads[0]), cyl ('d', np.array([0, 1.1]), rads[1]), cyl('d', np.array([0, -1.1]), rads[2])], method='explicit')

        scgmres1 = scattering([cyl('d', np.array([0,0]), 0.5)])
        scgmres2 = scattering([cyl('d', np.array([0,0]), rads[0]), cyl ('d', np.array([0, 1.1]), rads[1]), cyl('d', np.array([0, -1.1]), rads[2])])

        self.assertLess(np.abs(sc1.make_u(1.5,0) - scgmres1.make_u(1.5,0)), 1e-8)
        self.assertLess(np.abs(sc2.make_u(1.5,0) - scgmres2.make_u(1.5,0)), 1e-8)

    def test_neumann(self):
        rads = np.random.uniform(0.1, 0.4, size=3)
        sc1 = scattering([cyl('n', np.array([0,0]), 0.5)], method='explicit')
        sc2 = scattering([cyl('n', np.array([0,0]), rads[0]), cyl ('n', np.array([0, 1.1]), rads[1]), cyl('n', np.array([0, -1.1]), rads[2])], method='explicit')

        scgmres1 = scattering([cyl('n', np.array([0,0]), 0.5)])
        scgmres2 = scattering([cyl('n', np.array([0,0]), rads[0]), cyl ('n', np.array([0, 1.1]), rads[1]), cyl('n', np.array([0, -1.1]), rads[2])])

        self.assertLess(np.abs(sc1.make_u(1,0) - scgmres1.make_u(1,0)), 1e-8)
        self.assertLess(np.abs(sc2.make_u(1,0) - scgmres2.make_u(1,0)), 1e-8)

    
    def test_impedance(self):
        rads = np.random.uniform(0.1, 0.4, size=3)
        imps = np.random.uniform(0.3, 3, size =3) + 1j*np.random.uniform(0.3, 3, size=3)
        
        sc1 = scattering([cyl('i', np.array([0,0]), 0.5, impedance=3+1j)], method='explicit')
        sc2 = scattering([cyl('i', np.array([0,0]), rads[0], impedance=imps[0]), cyl ('i', np.array([0, 1.1]), rads[1], impedance=imps[1]), cyl('i', np.array([0, -1.1]), rads[2], impedance=imps[2])], method='explicit')

        scgmres1 = scattering([cyl('i', np.array([0,0]), 0.5, impedance=3+1j)])
        scgmres2 = scattering([cyl('i', np.array([0,0]), rads[0], impedance=imps[0]), cyl ('i', np.array([0, 1.1]), rads[1], impedance=imps[1]), cyl('i', np.array([0, -1.1]), rads[2], impedance=imps[2])])

        self.assertLess(np.abs(sc1.make_u(1,0) - scgmres1.make_u(1,0)), 1e-8)
        self.assertLess(np.abs(sc2.make_u(1,0) - scgmres2.make_u(1,0)), 1e-8)

    def test_antoine_precond(self):
        rads = np.random.uniform(0.1, 0.4, size=3)
        imps = np.random.uniform(0.3, 3, size =3) + 1j*np.random.uniform(0.3, 3, size=3)
        
        #TODO add other preconditioner tests

        sc1 = scattering([cyl('i', np.array([0,0]), 0.5, impedance=3+1j)], method='explicit')
        sc2 = scattering([cyl('i', np.array([0,0]), rads[0], impedance=imps[0]), cyl('i', np.array([0, 1.1]), rads[1], impedance=imps[1]), cyl('i', np.array([0, -1.1]), rads[2], impedance=imps[2])], method='explicit', precond=preconditioner.antoine, precond_dist = 2)

        scgmres1 = scattering([cyl('i', np.array([0,0]), 0.5, impedance=3+1j)])
        scgmres2 = scattering([cyl('i', np.array([0,0]), rads[0], impedance=imps[0]), cyl('i', np.array([0, 1.1]), rads[1], impedance=imps[1]), cyl('i', np.array([0, -1.1]), rads[2], impedance=imps[2])], precond=preconditioner.antoine, precond_dist=2)

        self.assertLess(np.abs(sc1.make_u(1,0) - scgmres1.make_u(1,0)), 1e-8)
        self.assertLess(np.abs(sc2.make_u(1,0) - scgmres2.make_u(1,0)), 1e-8)

    def test_mul_scatter_block_from_root_vector(self):
        freq = 10

        cyls = [cyl('d', [0, -1], 1), cyl('d', [1, 1], 1.5)]
        sc_r = scattering(cyls, freq=freq)
        sc = scattering(cyls, freq=freq, fast_matrix_build=False)
        A_root = sc.make_scattering_blocks()
        A = sc.make_scattering_blocks()
        self.assertTrue(np.allclose(A_root, A))

        cyls = [cyl('d', [0, -1], 1), cyl('d', [1, 1], 1.5)]
        sc_r = scattering(cyls, freq=freq)
        sc = scattering(cyls, freq=freq, fast_matrix_build=False)
        A_root = sc.make_scattering_blocks()
        A = sc.make_scattering_blocks()
        self.assertTrue(np.allclose(A_root, A))
        
        cyls = [cyl('i', [0, -1], 1, impedance=1+1j), cyl('d', [1, 1], 1.5, impedance=2+0.5*1j)]
        sc_r = scattering(cyls, freq=freq)
        sc = scattering(cyls, freq=freq, fast_matrix_build=False)
        A_root = sc.make_scattering_blocks()
        A = sc.make_scattering_blocks()
        self.assertTrue(np.allclose(A_root, A))

unittest.main()
