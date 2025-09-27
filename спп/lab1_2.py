import pytest

def solve_linear(a, b):
    if a == 0:
        if b == 0:
            return "infinity1"
        else:
            return "no solution"
    else:
        x = b / a
        return f"the solution is {x}"

@pytest.mark.parametrize("a,b,expected", [
    (2, -4, "the solution is 2.0"),
    (0, 4, "no solution"),
    (2, 0, "the solution is 0.0"),
    (0, 0, "infinity"),
    (-1, 1, "the solution is 1.0"),
    (3, -5, "the solution is 1.6666666666666667"),
    (1.5, 3.0, "the solution is -2.0"),
])
def test_solve_linear(a, b, expected):
    assert solve_linear(a, b) == expected

def test_return_type():
    """Переконуємось, що функція повертає рядок"""
    assert isinstance(solve_linear(2, 3), str)

def test_no_solution():
    """Перевірка, що при a=0 та b≠0 повертається конкретне значення"""
    assert "no solution" in solve_linear(0, 3)

def test_infinity_case():
    """Перевіряємо, що якщо обидва коефіцієнти 0, повертається 'infinity'"""
    assert solve_linear(0, 0) == "infinity"

def test_mocking_with_pytest_mock(mocker):
    """Використання mock для перевірки викликів"""
    mock = mocker.MagicMock()
    mock(5, 10)  # Викликаємо мок-об'єкт з певними аргументами
    mock(0, 0)

    # Перевіряємо, що мок викликався
    mock.assert_called()

    # Перевіряємо, що мок викликався хоча б раз з конкретними аргументами
    mock.assert_any_call(5, 10)
    mock.assert_any_call(0, 0)

    # Перевіряємо кількість викликів моку
    assert mock.call_count == 2
