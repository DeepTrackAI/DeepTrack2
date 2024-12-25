import unittest

from deeptrack.backend import polynomials


class TestPolynomials(unittest.TestCase):

    def test_Besselj(self):
        Besselj = polynomials.besselj

        self.assertEqual(Besselj(0, 0), 1)
        self.assertEqual(Besselj(1, 0), 0)

        self.assertEqual(Besselj(1, 5), -Besselj(-1, 5))
        self.assertTrue(Besselj(1, 5) < 0)


    def test_dBesselj(self):
        dBesselj = polynomials.dbesselj

        self.assertEqual(dBesselj(1,0), 0.5)


    def test_Bessely(self):
        Bessely = polynomials.bessely

        self.assertEqual(Bessely(1, 3), -Bessely(-1,3))
        self.assertTrue(Bessely(1, 3) > 0)


    def test_dBessely(self):
        dBessely = polynomials.dbessely

        self.assertEqual(dBessely(1, 3), -dBessely(-1,3))
        self.assertTrue(dBessely(1, 3) > 0)


    def test_RicBesj(self):
        Ricbesj = polynomials.ricbesj

        self.assertEqual(Ricbesj(4, 0), 0)
        self.assertTrue(abs(Ricbesj(1, 8)- 0.2691) < 1e-3)


    def test_dRicBesj(self):
        dRicBesj = polynomials.dricbesj

        self.assertTrue(abs(dRicBesj(1, 1) - 0.5403) < 1e-3)


    def test_RicBesy(self):
        RicBesy = polynomials.ricbesy

        self.assertTrue(abs(RicBesy(2, 3) - 0.8011) < 1e-3)


    def test_dRicBesy(self):
        dRicBesy = polynomials.dricbesy

        self.assertTrue(abs(dRicBesy(1, 1) + 0.8414) < 1e-3)

        
    def test_RicBesh(self):
        RicBesh = polynomials.ricbesh

        self.assertTrue(abs(RicBesh(3, 2) - (0.1214 - 2.968j)) < 1e-3)


    def test_dRicBesh(self):
        dRicBesh = polynomials.dricbesh

        self.assertTrue(abs(dRicBesh(2, 6) - (-0.9321-0.2206j)) < 1e-3)