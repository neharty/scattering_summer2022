import numpy as np
import unittest
import sys

sys.path.append('../Ncyl/')
from Ncyl_gmres import scattering, cyl

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

class ScatteringTest(unittest.TestCase): 
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


    #def test_rand_types(self):


unittest.main()
