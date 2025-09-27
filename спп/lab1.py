import unittest
from unittest.mock import MagicMock


def solve_linear(a, b):
    if a == 0:
        if b == 0:
            return "infinity1"
        else:
            return "no solution"
    else:
        x = b / a
        return f"the solution is {x}"

class TestSolveLinear(unittest.TestCase):
    
    def test_positive_coefficients(self):
        self.assertEqual(solve_linear(2, -4), "the solution is 2.0")

    def test_zero_coefficient_a(self):
        self.assertEqual(solve_linear(0, 4), "no solution")

    def test_zero_coefficient_b(self):
        self.assertEqual(solve_linear(2, 0), "the solution is 0.0")

    def test_both_coefficients_zero(self):
        self.assertEqual(solve_linear(0, 0), "infinity")

    def test_negative_coefficients(self):
        self.assertEqual(solve_linear(-1, 1), "the solution is 1.0")

    def test_fractional_coefficients(self):
        self.assertEqual(solve_linear(3, -5), "the solution is 1.6666666666666667")

    def test_float_input(self):
        self.assertEqual(solve_linear(1.5, 3.0), "the solution is -2.0")

    def test_return_type(self):
        """Переконуємось, що функція повертає рядок"""
        self.assertIsInstance(solve_linear(2, 3), str)

    def test_no_solution(self):
        """Перевірка, що при a=0 та b≠0 повертається конкретне значення"""
        self.assertIn("no solution", solve_linear(0, 3))

    def test_infinity_case(self):
        """Перевіряємо, що якщо обидва коефіцієнти 0, повертається 'infinity'"""
        self.assertTrue(solve_linear(0, 0) == "infinity")


def test_magic_mock(self):
        """Використання MagicMock для перевірки викликів"""
        mock = MagicMock()
        mock(5, 10)  # Викликаємо мок-об'єкт з певними аргументами
        mock(0, 0)

        # Перевіряємо, що мок викликався
        mock.assert_called()

        # Перевіряємо, що мок викликався хоча б раз з конкретними аргументами
        mock.assert_any_call(5, 10)
        mock.assert_any_call(0, 0)

        # Перевіряємо кількість викликів моку
        self.assertEqual(mock.call_count, 2)

if __name__ == '__main__':
    unittest.main()
