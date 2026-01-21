import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)


from lemon.numlib import *

import numpy as np


def test_Integer():
    assert Integer(1) == 1
    assert Integer(1) == Integer("1")
    assert Integer(2) + Integer(2) == Integer(4)
    assert Integer(2) + Integer(2) == 4
    assert Integer(2) + 2 == Integer(4)
    assert Integer(2) + 2 == 4
    assert 2 + Integer(2) == Integer(4)
    assert 2 + Integer(2) == 4
    assert Integer(2) + 2.0 == Real(4.0)
    assert Integer(2) + 2.0 == 4.0
    assert 2.0 + Integer(2) == Real(4.0)
    assert 2.0 + Integer(2) == 4.0
    assert Integer(2) + 2 + 2.0 == Real(6.0)
    assert Integer(2) + 2 + 2.0 == 6.0
    assert 2.0 + 2 + Integer(2) == Real(6.0)
    assert 2.0 + 2 + Integer(2) == 6.0
    assert Integer(2) * Integer(2) == Integer(4)
    assert Integer(2) * Integer(2) == 4
    assert Integer(2) * 2 == Integer(4)
    assert Integer(2) * 2 == 4
    assert 2 * Integer(2) == Integer(4)
    assert 2 * Integer(2) == 4
    assert Integer(2) * 2.0 == Real(4.0)
    assert Integer(2) * 2.0 == 4.0
    assert 2.0 * Integer(2) == Real(4.0)
    assert 2.0 * Integer(2) == 4.0
    assert Integer(2) * 2 * 2.0 == Real(8.0)
    assert Integer(2) * 2 * 2.0 == 8.0
    assert 2.0 * 2 * Integer(2) == Real(8.0)
    assert 2.0 * 2 * Integer(2) == 8.0
    assert Integer(2) - Integer(2) == Integer(0)
    assert Integer(2) - Integer(2) == 0
    assert Integer(2) - 2 == Integer(0)
    assert Integer(2) - 2 == 0
    assert 2 - Integer(2) == Integer(0)
    assert 2 - Integer(2) == 0
    assert Integer(2) - 2.0 == Real(0.0)
    assert Integer(2) - 2.0 == 0.0
    assert 2.0 - Integer(2) == Real(0.0)
    assert 2.0 - Integer(2) == 0.0
    assert Integer(2) - 2 - 2.0 == Real(-2.0)
    assert Integer(2) - 2 - 2.0 == -2.0
    assert 2.0 - 2 - Integer(2) == Real(-2.0)
    assert 2.0 - 2 - Integer(2) == -2.0
    assert Integer(2) / Integer(2) == Real(1.0)
    assert Integer(2) / Integer(2) == 1.0
    assert Integer(2) / 2 == Real(1.0)
    assert Integer(2) / 2 == 1.0
    assert 2 / Integer(2) == Real(1.0)
    assert 2 / Integer(2) == 1.0
    assert Integer(2) / 2.0 == Real(1.0)
    assert Integer(2) / 2.0 == 1.0
    assert 2.0 / Integer(2) == Real(1.0)
    assert 2.0 / Integer(2) == 1.0
    assert Integer(2) / 2 / 2.0 == Real(0.5)
    assert Integer(2) / 2 / 2.0 == 0.5
    assert 2.0 / 2 / Integer(2) == Real(0.5)
    assert 2.0 / 2 / Integer(2) == 0.5
    assert Integer(2) == Integer(2)
    assert Integer(2) == 2
    assert Integer(2) == 2.0
    assert Integer(2) == Real(2.0)
    assert Integer(2) == Complex(2.0)
    assert Integer(1) == Boolean(True)
    assert Integer(0) == Boolean(False)
    assert Integer(1) == True
    assert Integer(0) == False
    assert Integer(2) != Boolean(True)


def test_Real():
    assert Real(1.0) == 1.0
    assert Real(1.0) == Real("1.0")
    assert Real(2.0) + Real(2.0) == Real(4.0)
    assert Real(2.0) + Real(2.0) == 4.0
    assert Real(2.0) + 2 == Real(4.0)
    assert Real(2.0) + 2 == 4.0
    assert 2 + Real(2.0) == Real(4.0)
    assert 2 + Real(2.0) == 4.0
    assert Real(2.0) + 2.0 == Real(4.0)
    assert Real(2.0) + 2.0 == 4.0
    assert 2.0 + Real(2.0) == Real(4.0)
    assert 2.0 + Real(2.0) == 4.0
    assert Real(2.0) + 2 + 2.0 == Real(6.0)
    assert Real(2.0) + 2 + 2.0 == 6.0
    assert 2.0 + 2 + Real(2.0) == Real(6.0)
    assert 2.0 + 2 + Real(2.0) == 6.0
    assert Real(2.0) * Real(2.0) == Real(4.0)
    assert Real(2.0) * Real(2.0) == 4.0
    assert Real(2.0) * 2 == Real(4.0)
    assert Real(2.0) * 2 == 4.0
    assert 2 * Real(2.0) == Real(4.0)
    assert 2 * Real(2.0) == 4.0
    assert Real(2.0) * 2.0 == Real(4.0)
    assert Real(2.0) * 2.0 == 4.0
    assert 2.0 * Real(2.0) == Real(4.0)
    assert 2.0 * Real(2.0) == 4.0
    assert Real(2.0) * 2 * 2.0 == Real(8.0)
    assert Real(2.0) * 2 * 2.0 == 8.0
    assert 2.0 * 2 * Real(2.0) == Real(8.0)
    assert 2.0 * 2 * Real(2.0) == 8.0
    assert Real(2.0) - Real(2.0) == Real(0.0)
    assert Real(2.0) - Real(2.0) == 0.0
    assert Real(2.0) - 2 == Real(0.0)
    assert Real(2.0) - 2 == 0.0
    assert 2 - Real(2.0) == Real(0.0)
    assert 2 - Real(2.0) == 0.0
    assert Real(2.0) - 2.0 == Real(0.0)
    assert Real(2.0) - 2.0 == 0.0
    assert 2.0 - Real(2.0) == Real(0.0)
    assert 2.0 - Real(2.0) == 0.0
    assert Real(2.0) - 2 - 2.0 == Real(-2.0)
    assert Real(2.0) - 2 - 2.0 == -2.0
    assert 2.0 - 2 - Real(2.0) == Real(-2.0)
    assert 2.0 - 2 - Real(2.0) == -2.0
    assert Real(2.0) / Real(2.0) == Real(1.0)
    assert Real(2.0) / Real(2.0) == 1.0
    assert Real(2.0) / 2 == Real(1.0)
    assert Real(2.0) / 2 == 1.0
    assert 2 / Real(2.0) == Real(1.0)
    assert 2 / Real(2.0) == 1.0
    assert Real(2.0) / 2.0 == Real(1.0)
    assert Real(2.0) / 2.0 == 1.0
    assert 2.0 / Real(2.0) == Real(1.0)
    assert 2.0 / Real(2.0) == 1.0
    assert Real(2.0) / 2 / 2.0 == Real(0.5)
    assert Real(2.0) / 2 / 2.0 == 0.5
    assert 2.0 / 2 / Real(2.0) == Real(0.5)
    assert 2.0 / 2 / Real(2.0) == 0.5
    assert Real(2.0) == Real(2.0)
    assert Real(2.0) == 2
    assert Real(2.0) == 2.0
    assert Real(2.0) == Integer(2)
    assert Real(2.0) == Complex(2.0)
    assert Real(1.0) == Boolean(True)
    assert Real(0.0) == Boolean(False)
    assert Real(1.0) == True
    assert Real(0.0) == False
    assert Real(2.0) != Boolean(True)


def test_Complex():
    assert Complex(1 + 1j) == 1 + 1j
    assert Complex(1 + 1j) == Complex("1+1j")
    assert Complex(2 + 2j) + Complex(2 + 2j) == Complex(4 + 4j)
    assert Complex(2 + 2j) + Complex(2 + 2j) == 4 + 4j
    assert Complex(2 + 2j) + 2 == Complex(4 + 2j)
    assert Complex(2 + 2j) + 2 == 4 + 2j
    assert 2 + Complex(2 + 2j) == Complex(2 + (2 + 2j))
    assert 2 + Complex(2 + 2j) == (2 + (2 + 2j))
    assert Complex(2 + 2j) + 2.0 == Complex((2 + 2j) + 2.0)
    assert Complex(2 + 2j) + 2.0 == (2 + 2j) + 2.0
    assert 2.0 + Complex(2 + 2j) == Complex(2.0 + (2 + 2j))
    assert 2.0 + Complex(2 + 2j) == (2.0 + (2 + 2j))
    assert Complex(2 + 2j) + 2 + 2.0 == Complex((2 + 2j) + 2 + 2.0)
    assert Complex(2 + 2j) + 2 + 2.0 == ((2 + 2j) + 2 + 2.0)
    assert 2.0 + 2 + Complex(2 + 2j) == Complex(2.0 + 2 + (2 + 2j))
    assert 2.0 + 2 + Complex(2 + 2j) == (2.0 + 2 + (2 + 2j))
    assert Complex(2 + 2j) * Complex(2 + 2j) == Complex((2 + 2j) * (2 + 2j))
    assert Complex(2 + 2j) * Complex(2 + 2j) == ((2 + 2j) * (2 + 2j))
    assert Complex(2 + 2j) * 2 == Complex((2 + 2j) * 2)
    assert Complex(2 + 2j) * 2 == ((2 + 2j) * 2)
    assert 2 * Complex(2 + 2j) == Complex(2 * (2 + 2j))
    assert 2 * Complex(2 + 2j) == (2 * (2 + 2j))
    assert Complex(2 + 2j) * 2.0 == Complex((2 + 2j) * 2.0)
    assert Complex(2 + 2j) * 2.0 == ((2 + 2j) * 2.0)
    assert 2.0 * Complex(2 + 2j) == Complex(2.0 * (2 + 2j))
    assert 2.0 * Complex(2 + 2j) == (2.0 * (2 + 2j))
    assert Complex(2 + 2j) * 2 * 2.0 == Complex((2 + 2j) * 2 * 2.0)
    assert Complex(2 + 2j) * 2 * 2.0 == ((2 + 2j) * 2 * 2.0)
    assert 2.0 * 2 * Complex(2 + 2j) == Complex(2.0 * 2 * Complex(2 + 2j))
    assert 2.0 * 2 * Complex(2 + 2j) == (2.0 * 2 * Complex(2 + 2j))
    assert Complex(2 + 2j) - Complex(2 + 2j) == Complex((2 + 2j) - (2 + 2j))
    assert Complex(2 + 2j) - Complex(2 + 2j) == ((2 + 2j) - (2 + 2j))
    assert Complex(2 + 2j) - 2 == Complex((2 + 2j) - 2)
    assert Complex(2 + 2j) - 2 == ((2 + 2j) - 2)
    assert 2 - Complex(2 + 2j) == Complex(2 - (2 + 2j))
    assert 2 - Complex(2 + 2j) == (2 - (2 + 2j))
    assert Complex(2 + 2j) - 2.0 == Complex((2 + 2j) - 2.0)
    assert Complex(2 + 2j) - 2.0 == ((2 + 2j) - 2.0)
    assert 2.0 - Complex(2 + 2j) == Complex(2.0 - (2 + 2j))
    assert 2.0 - Complex(2 + 2j) == (2.0 - (2 + 2j))
    assert Complex(2 + 2j) - 2 - 2.0 == Complex((2 + 2j) - 2 - 2.0)
    assert Complex(2 + 2j) - 2 - 2.0 == (2 + 2j) - 2 - 2.0
    assert 2.0 - 2 - Complex(2 + 2j) == Complex(2.0 - 2 - (2 + 2j))
    assert 2.0 - 2 - Complex(2 + 2j) == (2.0 - 2 - (2 + 2j))
    assert Complex(2 + 2j) / Complex(2 + 2j) == Complex((2 + 2j) / (2 + 2j))
    assert Complex(2 + 2j) / Complex(2 + 2j) == (2 + 2j) / (2 + 2j)
    assert Complex(2 + 2j) / 2 == Complex((2 + 2j) / 2)
    assert Complex(2 + 2j) / 2 == ((2 + 2j) / 2)
    assert 2 / Complex(2 + 2j) == Complex(2 / (2 + 2j))
    assert 2 / Complex(2 + 2j) == (2 / (2 + 2j))
    assert Complex(2 + 2j) / 2.0 == Complex((2 + 2j) / 2.0)
    assert Complex(2 + 2j) / 2.0 == (2 + 2j) / 2.0
    assert 2.0 / Complex(2 + 2j) == Complex(2.0 / (2 + 2j))
    assert 2.0 / Complex(2 + 2j) == (2.0 / (2 + 2j))
    assert Complex(2 + 2j) / 2 / 2.0 == Complex((2 + 2j) / 2 / 2.0)
    assert Complex(2 + 2j) / 2 / 2.0 == (2 + 2j) / 2 / 2.0
    assert 2.0 / 2 / Complex(2 + 2j) == Complex(2.0 / 2 / (2 + 2j))
    assert 2.0 / 2 / Complex(2 + 2j) == (2.0 / 2 / (2 + 2j))
    assert Complex(2 + 2j) == Complex(2 + 2j)
    assert Boolean(Complex(1 + 1j)) == Boolean(True)
    assert Boolean(Complex(0 + 0j)) == Boolean(False)
    assert Boolean(Complex(1 + 1j)) == True
    assert Boolean(Complex(0 + 0j)) == False
    assert Boolean(Complex(2 + 2j)) == Boolean(True)


def test_Boolean():
    assert Boolean(True) == Boolean(True)
    assert Boolean(True) == Boolean(1)
    assert Boolean(True) == Boolean(1.0)
    assert Boolean(False) == Boolean(False)
    assert Boolean(False) == Boolean(0)
    assert Boolean(False) == Boolean(0.0)
    assert (Integer(1) > Integer(0)) == Boolean(True)
    assert (Integer(1) < Integer(0)) == Boolean(False)
    assert (Integer(1) == Integer(1)) == Boolean(True)
    assert (Integer(1) == Integer(0)) == Boolean(False)
    if Boolean(True) == True:
        assert True
    else:
        assert False
    if Boolean(False) == False:
        assert True
    assert Boolean(0) == bool(0)
    assert Boolean(1) == bool(1)
    assert Boolean(1.1) == bool(1.1)
    assert Boolean(2 + 5j) == bool(2 + 5j)


def test_Matrix():
    m1 = Matrix([[1, 2], [3, 4]])
    m2 = Matrix([[1, 2], [3, 4]])
    assert (m1 == m2).all()
    assert (m1 == [[1, 2], [3, 4]]).all()
    # +
    assert (m1 + m2 == Matrix([[2, 4], [6, 8]])).all()
    assert (m1 + [[1, 2], [3, 4]] == Matrix([[2, 4], [6, 8]])).all()
    assert ([[1, 2], [3, 4]] + m1 == Matrix([[2, 4], [6, 8]])).all()
    assert (m1 + 1 == Matrix([[2, 3], [4, 5]])).all()
    assert (1 + m1 == Matrix([[2, 3], [4, 5]])).all()
    assert (m1 + 1.0 == Matrix([[2.0, 3.0], [4.0, 5.0]])).all()
    assert (1.0 + m1 == Matrix([[2.0, 3.0], [4.0, 5.0]])).all()
    assert (m1 + 1 + 1.0 == Matrix([[3.0, 4.0], [5.0, 6.0]])).all()
    assert (1.0 + 1 + m1 == Matrix([[3.0, 4.0], [5.0, 6.0]])).all()
    # *
    assert (m1 * m2 == Matrix([[1, 4], [9, 16]])).all()
    assert (m1 * [[1, 2], [3, 4]] == Matrix([[1, 4], [9, 16]])).all()
    assert ([[1, 2], [3, 4]] * m1 == Matrix([[1, 4], [9, 16]])).all()
    assert (m1 * 2 == Matrix([[2, 4], [6, 8]])).all()
    assert (2 * m1 == Matrix([[2, 4], [6, 8]])).all()
    assert (m1 * 2.0 == Matrix([[2.0, 4.0], [6.0, 8.0]])).all()
    assert (2.0 * m1 == Matrix([[2.0, 4.0], [6.0, 8.0]])).all()
    assert (m1 * 2 * 2.0 == Matrix([[4.0, 8.0], [12.0, 16.0]])).all()
    assert (2 * 2.0 * m1 == Matrix([[4.0, 8.0], [12.0, 16.0]])).all()
    # @
    assert (m1 @ m2 == Matrix([[7, 10], [15, 22]])).all()
    assert (m1 @ [[1, 2], [3, 4]] == Matrix([[7, 10], [15, 22]])).all()
    assert ([[1, 2], [3, 4]] @ m1 == Matrix([[7, 10], [15, 22]])).all()
    # assert (m1 @ 2 == Matrix([[2, 4], [6, 8]])).all()
    # assert (2 @ m1 == Matrix([[2, 4], [6, 8]])).all()
    # assert (m1 * 2.0 == Matrix([[2.0, 4.0], [6.0, 8.0]])).all()
    # assert (2.0 * m1 == Matrix([[2.0, 4.0], [6.0, 8.0]])).all()
    # assert (m1 * 2 * 2.0 == Matrix([[4.0, 8.0], [12.0, 16.0]])).all()
    # assert (2 * 2.0 * m1 == Matrix([[4.0, 8.0], [12.0, 16.0]])).all()
    # -
    assert (m1 - m2 == Matrix([[0, 0], [0, 0]])).all()
    assert (m1 - [[1, 2], [3, 4]] == Matrix([[0, 0], [0, 0]])).all()
    assert ([[1, 2], [3, 4]] - m1 == Matrix([[0, 0], [0, 0]])).all()
    assert (m1 - 1 == Matrix([[0, 1], [2, 3]])).all()
    assert (1 - m1 == Matrix([[0, -1], [-2, -3]])).all()
    assert (m1 - 1.0 == Matrix([[0.0, 1.0], [2.0, 3.0]])).all()
    assert (1.0 - m1 == Matrix([[0.0, -1.0], [-2.0, -3.0]])).all()
    # /
    assert (m1 / m2 == Matrix([[1, 1], [1, 1]])).all()
    assert (m1 / [[1, 2], [3, 4]] == Matrix([[1, 1], [1, 1]])).all()
    assert ([[1, 2], [3, 4]] / m1 == Matrix([[1, 1], [1, 1]])).all()
    assert (Matrix([[2, 4], [6, 8]]) / 2 == Matrix([[1.0, 2.0], [3.0, 4.0]])).all()
    assert (
        2 / Matrix([[2, 4], [6, 8]]) == Matrix([[1.0, 0.5], [0.3333333333333333, 0.25]])
    ).all()
    # <
    assert (
        (Matrix([[2, 4], [6, 8]]) < Matrix([[2, 4], [6, 8]]))
        == [[False, False], [False, False]]
    ).all()
    assert (Matrix([[2, 4], [6, 8]]) < Matrix([[2, 4], [6, 8]])).all() == (
        Matrix([[False, False], [False, False]])
    ).all()
    assert (
        (Matrix([[2, 4], [6, 8]]) < [[2, 4], [6, 8]])
        == [[False, False], [False, False]]
    ).all()
    assert (
        ([[2, 4], [6, 8]] < Matrix([[2, 4], [6, 8]]))
        == [[False, False], [False, False]]
    ).all()
    assert ((Matrix([[2, 4], [6, 8]]) < 2) == [[False, False], [False, False]]).all()
    assert ((2 < Matrix([[2, 4], [6, 8]])) == [[False, True], [True, True]]).all()
    assert ((Matrix([[2, 4], [6, 8]]) < 2.0) == [[False, False], [False, False]]).all()
    assert ((2.0 < Matrix([[2, 4], [6, 8]])) == [[False, True], [True, True]]).all()
    # <=
    assert (
        (Matrix([[2, 4], [6, 8]]) <= Matrix([[2, 4], [6, 8]]))
        == [[True, True], [True, True]]
    ).all()
    assert (
        (Matrix([[2, 4], [6, 8]]) <= [[2, 4], [6, 8]]) == [[True, True], [True, True]]
    ).all()
    assert (
        ([[2, 4], [6, 8]] <= Matrix([[2, 4], [6, 8]])) == [[True, True], [True, True]]
    ).all()
    assert ((Matrix([[2, 4], [6, 8]]) <= 2) == [[True, False], [False, False]]).all()
    assert ((2 <= Matrix([[2, 4], [6, 8]])) == [[True, True], [True, True]]).all()
    assert ((Matrix([[2, 4], [6, 8]]) <= 2.0) == [[True, False], [False, False]]).all()
    assert ((2.0 <= Matrix([[2, 4], [6, 8]])) == [[True, True], [True, True]]).all()
    # >
    assert (
        (Matrix([[2, 4], [6, 8]]) > Matrix([[2, 4], [6, 8]]))
        == [[False, False], [False, False]]
    ).all()
    assert (
        (Matrix([[2, 4], [6, 8]]) > [[2, 4], [6, 8]])
        == [[False, False], [False, False]]
    ).all()
    assert (
        ([[2, 4], [6, 8]] > Matrix([[2, 4], [6, 8]]))
        == [[False, False], [False, False]]
    ).all()
    assert ((Matrix([[2, 4], [6, 8]]) > 2) == [[False, True], [True, True]]).all()
    assert ((2 > Matrix([[2, 4], [6, 8]])) == [[False, False], [False, False]]).all()
    assert ((Matrix([[2, 4], [6, 8]]) > 2.0) == [[False, True], [True, True]]).all()
    assert ((2.0 > Matrix([[2, 4], [6, 8]])) == [[False, False], [False, False]]).all()
    # >=
    assert (
        (Matrix([[2, 4], [6, 8]]) >= Matrix([[2, 4], [6, 8]]))
        == [[True, True], [True, True]]
    ).all()
    assert (
        (Matrix([[2, 4], [6, 8]]) >= [[2, 4], [6, 8]]) == [[True, True], [True, True]]
    ).all()
    assert (
        ([[2, 4], [6, 8]] >= Matrix([[2, 4], [6, 8]])) == [[True, True], [True, True]]
    ).all()
    assert ((Matrix([[2, 4], [6, 8]]) >= 2) == [[True, True], [True, True]]).all()
    assert ((2 >= Matrix([[2, 4], [6, 8]])) == [[True, False], [False, False]]).all()
    assert ((Matrix([[2, 4], [6, 8]]) >= 2.0) == [[True, True], [True, True]]).all()
    assert ((2.0 >= Matrix([[2, 4], [6, 8]])) == [[True, False], [False, False]]).all()
    # &
    assert (
        Matrix([[1, 2], [3, 4]]) & Matrix([[1, 2], [3, 4]]) == [[1, 2], [3, 4]]
    ).all()
    assert ((Matrix([[1, 2], [3, 4]]) & [[1, 2], [3, 4]]) == [[1, 2], [3, 4]]).all()
    assert ([[1, 2], [3, 4]] & Matrix([[1, 2], [3, 4]]) == [[1, 2], [3, 4]]).all()
    assert (Matrix([[1, 2], [3, 4]]) & 2 == [[0, 2], [2, 0]]).all()
    assert (2 & Matrix([[1, 2], [3, 4]]) == [[0, 2], [2, 0]]).all()
    # |
    assert (
        Matrix([[1, 2], [3, 4]]) | Matrix([[1, 2], [3, 4]]) == [[1, 2], [3, 4]]
    ).all()
    assert ((Matrix([[1, 2], [3, 4]]) | [[1, 2], [3, 4]]) == [[1, 2], [3, 4]]).all()
    assert ([[1, 2], [3, 4]] | Matrix([[1, 2], [3, 4]]) == [[1, 2], [3, 4]]).all()
    assert (Matrix([[1, 2], [3, 4]]) | 2 == [[3, 2], [3, 6]]).all()
    assert (2 | Matrix([[1, 2], [3, 4]]) == [[3, 2], [3, 6]]).all()
    # ^
    assert (
        Matrix([[1, 2], [3, 4]]) ^ Matrix([[1, 2], [3, 4]]) == [[0, 0], [0, 0]]
    ).all()
    assert ((Matrix([[1, 2], [3, 4]]) ^ [[1, 2], [3, 4]]) == [[0, 0], [0, 0]]).all()
    assert ([[1, 2], [3, 4]] ^ Matrix([[1, 2], [3, 4]]) == [[0, 0], [0, 0]]).all()
    assert (Matrix([[1, 2], [3, 4]]) ^ 2 == [[3, 0], [1, 6]]).all()
    assert (2 ^ Matrix([[1, 2], [3, 4]]) == [[3, 0], [1, 6]]).all()
    # <<
    assert (
        Matrix([[1, 2], [3, 4]]) << Matrix([[1, 2], [3, 4]]) == [[2, 8], [24, 64]]
    ).all()
    assert ((Matrix([[1, 2], [3, 4]]) << [[1, 2], [3, 4]]) == [[2, 8], [24, 64]]).all()
    assert ([[1, 2], [3, 4]] << Matrix([[1, 2], [3, 4]]) == [[2, 8], [24, 64]]).all()
    assert (Matrix([[1, 2], [3, 4]]) << 2 == [[4, 8], [12, 16]]).all()
    assert (2 << Matrix([[1, 2], [3, 4]]) == [[4, 8], [16, 32]]).all()
    # >>
    assert (
        Matrix([[1, 2], [3, 4]]) >> Matrix([[1, 2], [3, 4]]) == [[0, 0], [0, 0]]
    ).all()
    assert ((Matrix([[1, 2], [3, 4]]) >> [[1, 2], [3, 4]]) == [[0, 0], [0, 0]]).all()
    assert (
        [[1, 2], [3, 4]] >> Matrix([[1, 2], [3, 4]]) == Matrix([[0, 0], [0, 0]])
    ).all()
    assert ((Matrix([[1, 2], [3, 4]]) >> 2) == Matrix([[0, 0], [0, 1]])).all()
    assert (Matrix([[1, 2], [3, 4]]) >> 2 == [[0, 0], [0, 1]]).all()
    assert ((2 >> Matrix([[1, 2], [3, 4]])) == Matrix([[1, 0], [0, 0]])).all()
    assert (2 >> Matrix([[1, 2], [3, 4]]) == [[1, 0], [0, 0]]).all()
    # //
    assert (
        (Matrix([[1, 2], [3, 4]]) // Matrix([[1, 2], [3, 4]]))
        == Matrix([[1, 1], [1, 1]])
    ).all()
    assert (
        Matrix([[1, 2], [3, 4]]) // Matrix([[1, 2], [3, 4]]) == [[1, 1], [1, 1]]
    ).all()
    assert (
        (Matrix([[1, 2], [3, 4]]) // [[1, 2], [3, 4]]) == Matrix([[1, 1], [1, 1]])
    ).all()
    assert (Matrix([[1, 2], [3, 4]]) // [[1, 2], [3, 4]] == [[1, 1], [1, 1]]).all()
    assert ([[1, 2], [3, 4]] // Matrix([[1, 2], [3, 4]]) == [[1, 1], [1, 1]]).all()
    assert (Matrix([[1, 2], [3, 4]]) // 2 == [[0, 1], [1, 2]]).all()
    assert (2 // Matrix([[1, 2], [3, 4]]) == [[2, 1], [0, 0]]).all()
    assert (Matrix([[1, 2], [3, 4]]) // 2.0 == [[0.0, 1.0], [1.0, 2.0]]).all()
    assert (2.0 // Matrix([[1, 2], [3, 4]]) == [[2.0, 1.0], [0.0, 0.0]]).all()
    assert (Matrix([[1, 2], [3, 4]]) // 2 // 2.0 == [[0.0, 0.0], [0.0, 1.0]]).all()
    assert (2 // 2.0 // Matrix([[1, 2], [3, 4]]) == [[1.0, 0.0], [0.0, 0.0]]).all()
    # %
    assert (
        Matrix([[1, 2], [3, 4]]) % Matrix([[1, 2], [3, 4]]) == [[0, 0], [0, 0]]
    ).all()
    assert (Matrix([[1, 2], [3, 4]]) % [[1, 2], [3, 4]] == [[0, 0], [0, 0]]).all()
    assert ([[1, 2], [3, 4]] % Matrix([[1, 2], [3, 4]]) == [[0, 0], [0, 0]]).all()
    assert (Matrix([[1, 2], [3, 4]]) % 2 == [[1, 0], [1, 0]]).all()
    assert (2 % Matrix([[1, 2], [3, 4]]) == [[0, 0], [2, 2]]).all()
    assert (Matrix([[1, 2], [3, 4]]) % 2.0 == [[1.0, 0.0], [1.0, 0.0]]).all()
    assert (2.0 % Matrix([[1, 2], [3, 4]]) == [[0.0, 0.0], [2.0, 2.0]]).all()
    assert (Matrix([[1, 2], [3, 4]]) % 2 % 2.0 == [[1.0, 0.0], [1.0, 0.0]]).all()
    # assert (2 % 2.0 % Matrix([[1, 2], [3, 4]]) == [[0.0, 0.0], [0.0, 0.0]]).all()  # nan
    # **
    assert (
        Matrix([[1, 2], [3, 4]]) ** Matrix([[1, 2], [3, 4]]) == [[1, 4], [27, 256]]
    ).all()
    assert (Matrix([[1, 2], [3, 4]]) ** [[1, 2], [3, 4]] == [[1, 4], [27, 256]]).all()
    assert ([[1, 2], [3, 4]] ** Matrix([[1, 2], [3, 4]]) == [[1, 4], [27, 256]]).all()
    assert (Matrix([[1, 2], [3, 4]]) ** 2 == [[1, 4], [9, 16]]).all()
    assert (2 ** Matrix([[1, 2], [3, 4]]) == [[2, 4], [8, 16]]).all()
    # ==
    assert (Matrix([[1, 2], [3, 4]]) == Matrix([[1, 2], [3, 4]])).all()
    assert (Matrix([[1, 2], [3, 4]]) == [[1, 2], [3, 4]]).all()
    assert ([[1, 2], [3, 4]] == Matrix([[1, 2], [3, 4]])).all()
    # !=
    assert (Matrix([[1, 2], [3, 4]]) != Matrix([[1, 2], [3, 4]])).all() == False
    assert (Matrix([[1, 2], [3, 4]]) != [[1, 2], [3, 4]]).all() == False
    assert ([[1, 2], [3, 4]] != Matrix([[1, 2], [3, 4]])).all() == False


# Operators
def test_logic():
    def excute(op, a, b, expected):
        try:
            result = op(a, b)
            assert result == expected, f"Expected {expected} but got {result}"
        except TypeError:
            assert expected == TypeError

    assert Boolean(True) & Boolean(True) == Boolean(True)
    assert Boolean(True) & Boolean(False) == Boolean(False)
    assert Boolean(False) & Boolean(True) == Boolean(False)
    assert Boolean(False) & Boolean(False) == Boolean(False)
    assert Boolean(True) | Boolean(True) == Boolean(True)
    assert Boolean(True) | Boolean(False) == Boolean(True)
    assert Boolean(False) | Boolean(True) == Boolean(True)
    assert Boolean(False) | Boolean(False) == Boolean(False)
    assert Boolean(True) & True == Boolean(True)
    assert Boolean(True) & False == Boolean(False)
    assert Boolean(False) & True == Boolean(False)
    assert Boolean(False) & False == Boolean(False)
    assert True & Boolean(True) == Boolean(True)
    assert True & Boolean(False) == Boolean(False)
    assert False & Boolean(True) == Boolean(False)
    assert False & Boolean(False) == Boolean(False)
    assert Boolean(True) | True == Boolean(True)
    assert Boolean(True) | False == Boolean(True)
    assert Boolean(False) | True == Boolean(True)
    assert Boolean(False) | False == Boolean(False)
    assert True | Boolean(True) == Boolean(True)
    assert True | Boolean(False) == Boolean(True)
    assert False | Boolean(True) == Boolean(True)
    assert False | Boolean(False) == Boolean(False)
    assert Boolean(True) & 1 == Boolean(True)
    assert Boolean(True) & 0 == Boolean(False)
    assert Boolean(False) & 1 == Boolean(False)
    assert Boolean(False) & 0 == Boolean(False)
    assert 1 & Boolean(True) == Boolean(True)
    assert 1 & Boolean(False) == Boolean(False)
    assert 0 & Boolean(True) == Boolean(False)
    assert 0 & Boolean(False) == Boolean(False)
    assert Boolean(True) | 1 == Boolean(True)
    assert Boolean(True) | 0 == Boolean(True)
    assert Boolean(False) | 1 == Boolean(True)
    assert Boolean(False) | 0 == Boolean(False)
    assert 1 | Boolean(True) == Boolean(True)
    assert 1 | Boolean(False) == Boolean(True)
    assert 0 | Boolean(True) == Boolean(True)
    assert 0 | Boolean(False) == Boolean(False)
    assert Integer(5) & Integer(3) == Integer(1)
    assert Integer(5) | Integer(3) == Integer(7)
    try:
        assert Integer(5) & Real(3.0) == Integer(1)
        assert False
    except TypeError:
        assert True
    try:
        assert Integer(5) | Real(3.0) == Integer(7)
        assert False
    except TypeError:
        assert True
    try:
        assert Real(5.0) & Integer(3) == Integer(1)
        assert False
    except TypeError:
        assert True
    try:
        assert Real(5.0) | Integer(3) == Integer(7)
        assert False
    except TypeError:
        assert True
    try:
        assert Real(5.0) & Real(3.0) == Integer(1)
        assert False
    except TypeError:
        assert True
    try:
        assert Real(5.0) | Real(3.0) == Integer(7)
        assert False
    except TypeError:
        assert True
    try:
        assert Integer(5) & Complex(3 + 0j) == Integer(1)
        assert False
    except TypeError:
        assert True
    try:
        assert Integer(5) | Complex(3 + 0j) == Integer(7)
        assert False
    except TypeError:
        assert True
    try:
        assert Complex(5 + 0j) & Integer(3) == Integer(1)
        assert False
    except TypeError:
        assert True
    try:
        assert Complex(5 + 0j) | Integer(3) == Integer(7)
        assert False
    except TypeError:
        assert True
    try:
        assert Complex(5 + 0j) & Complex(3 + 0j) == Integer(1)
        assert False
    except TypeError:
        assert True
    try:
        assert Complex(5 + 0j) | Complex(3 + 0j) == Integer(7)
        assert False
    except TypeError:
        assert True


def test_bitwise():
    def excute(op, a, b, expected):
        try:
            result = op(a, b)
            assert result == expected, f"Expected {expected} but got {result}"
        except TypeError:
            assert expected == TypeError

    a = Integer(10)
    b = Integer(2)
    excute(lambda a, b: a & b, a, b, Integer(2))
    excute(lambda a, b: a | b, a, b, Integer(10))
    excute(lambda a, b: a ^ b, a, b, Integer(8))
    excute(lambda a, b: a << b, a, b, Integer(40))
    excute(lambda a, b: a >> b, a, b, Integer(2))
    excute(lambda a, b: ~a, a, None, Integer(-11))

    a = Integer(10)
    b = 2
    excute(lambda a, b: a & b, a, b, Integer(2))
    excute(lambda a, b: a | b, a, b, Integer(10))
    excute(lambda a, b: a ^ b, a, b, Integer(8))
    excute(lambda a, b: a << b, a, b, Integer(40))
    excute(lambda a, b: a >> b, a, b, Integer(2))
    excute(lambda a, b: ~a, a, None, Integer(-11))

    a = 10
    b = Integer(2)
    excute(lambda a, b: a & b, a, b, Integer(2))
    excute(lambda a, b: a | b, a, b, Integer(10))
    excute(lambda a, b: a ^ b, a, b, Integer(8))
    excute(lambda a, b: a << b, a, b, Integer(40))
    excute(lambda a, b: a >> b, a, b, Integer(2))
    excute(lambda a, b: ~a, a, None, Integer(-11))

    a = Integer(10)
    b = Real(2.0)
    excute(lambda a, b: a & b, a, b, TypeError)
    excute(lambda a, b: a | b, a, b, TypeError)
    excute(lambda a, b: a ^ b, a, b, TypeError)
    excute(lambda a, b: a << b, a, b, TypeError)
    excute(lambda a, b: a >> b, a, b, TypeError)
    excute(lambda a, b: ~a, a, None, TypeError)

    a = Real(10.0)
    b = Integer(2)
    excute(lambda a, b: a & b, a, b, TypeError)
    excute(lambda a, b: a | b, a, b, TypeError)
    excute(lambda a, b: a ^ b, a, b, TypeError)
    excute(lambda a, b: a << b, a, b, TypeError)
    excute(lambda a, b: a >> b, a, b, TypeError)
    excute(lambda a, b: ~a, a, None, TypeError)

    a = Integer(10)
    b = Complex(2 + 0j)
    excute(lambda a, b: a & b, a, b, TypeError)
    excute(lambda a, b: a | b, a, b, TypeError)
    excute(lambda a, b: a ^ b, a, b, TypeError)
    excute(lambda a, b: a << b, a, b, TypeError)
    excute(lambda a, b: a >> b, a, b, TypeError)
    excute(lambda a, b: ~a, a, None, TypeError)

    a = Real(10.0)
    b = 2
    excute(lambda a, b: a & b, a, b, TypeError)
    excute(lambda a, b: a | b, a, b, TypeError)
    excute(lambda a, b: a ^ b, a, b, TypeError)
    excute(lambda a, b: a << b, a, b, TypeError)
    excute(lambda a, b: a >> b, a, b, TypeError)
    excute(lambda a, b: ~a, a, None, TypeError)


def test_cast():
    assert Real(Integer(1)) == Real(1.0)
    assert isinstance(Real(Integer(1.0)), Real)

    assert Integer(Real(1.0)) == Integer(1)
    assert isinstance(Integer(Real(1.0)), Integer)

    assert Complex(Integer(1)) == Complex(1 + 0j)
    assert isinstance(Complex(Integer(1)), Complex)

    try:
        assert Integer(Complex(1 + 0j)) == Integer(1)
        assert False
    except CastError:
        assert True

    try:
        assert Real(Complex(1 + 0j)) == Real(1.0)
        assert False
    except CastError:
        assert True

    assert Complex(Real(1.0)) == Complex(1 + 0j)
    assert isinstance(Complex(Real(1.0)), Complex)

    assert Integer(Real(1.0)) == Integer(1)
    assert isinstance(Integer(Real(1.0)), Integer)

    assert Real(Integer(1)) == Real(1.0)
    assert isinstance(Real(Integer(1)), Real)

    assert Vector(Integer(1)) == Vector([1])
    assert isinstance(Vector(Integer(1)), Vector)
    assert Tensor(Integer(1)) == Tensor([1])
    assert isinstance(Tensor(Integer(1)), Tensor)

    a = int16(256)
    b = int8(a)
    assert b == 0, f"Integer8 cast failed: a: {a}, b: {b}"


def test_inplace():
    # Integer のテスト
    a = Integer(10)
    b = Integer(2)
    a += b
    assert a == Integer(12)
    a = Integer(10)
    a -= b
    assert a == Integer(8)
    a = Integer(10)
    a -= 5
    assert a == Integer(5)

    a = Integer(10)
    a *= b
    assert a == Integer(20)
    a = Integer(10)
    a *= 5
    assert a == Integer(50)

    # Integer の /= は TypeError を発生させる
    a = Integer(10)
    try:
        a /= b
        assert False, "Integer /= should raise TypeError"
    except TypeError:
        pass

    a = Integer(10)
    try:
        a /= 5
        assert False, "Integer /= should raise TypeError"
    except TypeError:
        pass

    # Integer の //= は動作する
    a = Integer(10)
    a //= b
    assert a == Integer(5)
    a = Integer(10)
    a //= 5
    assert a == Integer(2)

    a = Integer(10)
    a %= b
    assert a == Integer(0)
    a = Integer(10)
    a %= 5
    assert a == Integer(0)

    a = Integer(10)
    a **= b
    assert a == Integer(100)
    a = Integer(10)
    a **= 5
    assert a == Integer(100000)

    a = Integer(10)
    a &= b
    assert a == Integer(2)
    a = Integer(10)
    a &= 3
    assert a == Integer(2)

    a = Integer(10)
    a |= b
    assert a == Integer(10)
    a = Integer(10)
    a |= 3
    assert a == Integer(11)

    a = Integer(10)
    a ^= b
    assert a == Integer(8)
    a = Integer(10)
    a ^= 3
    assert a == Integer(9)

    a = Integer(10)
    a <<= b
    assert a == Integer(40)
    a = Integer(10)
    a <<= 3
    assert a == Integer(80)

    a = Integer(10)
    a >>= b
    assert a == Integer(2)
    a = Integer(10)
    a >>= 3
    assert a == Integer(1)

    a = Integer(10)
    a = -a
    assert a == Integer(-10)
    a = Integer(10)
    a = +a
    assert a == Integer(10)
    a = Integer(10)
    a = ~a
    assert a == Integer(-11)

    # Real のテスト
    a = Real(10.0)
    b = Real(2.0)
    a += b
    assert a == Real(12.0)
    a = Real(10.0)
    a -= b
    assert a == Real(8.0)
    a = Real(10.0)
    a -= 5
    assert a == Real(5.0)
    a = Real(10.0)
    a *= b
    assert a == Real(20.0)
    a = Real(10.0)
    a *= 5
    assert a == Real(50.0)

    # Real の /= は動作する
    a = Real(10.0)
    a /= b
    assert a == Real(5.0)
    a = Real(10.0)
    a /= 5
    assert a == Real(2.0)

    a = Real(10.0)
    a //= b
    assert a == Real(5.0)
    a = Real(10.0)
    a //= 5
    assert a == Real(2.0)
    a = Real(10.0)
    a %= b
    assert a == Real(0.0)
    a = Real(10.0)
    a %= 5
    assert a == Real(0.0)
    a = Real(10.0)
    a **= b
    assert a == Real(100.0)
    a = Real(10.0)
    a **= 5
    assert a == Real(100000.0)
    a = Real(10.0)
    a = -a
    assert a == Real(-10.0)
    a = Real(10.0)
    a = +a
    assert a == Real(10.0)

    # Real はビット演算をサポートしない
    try:
        a = Real(10.0)
        a = ~a
        assert False, "Real ~ should raise TypeError"
    except TypeError:
        pass

    try:
        a = Real(10.0)
        a &= b
        assert False, "Real &= should raise TypeError"
    except TypeError:
        pass

    try:
        a = Real(10.0)
        a |= b
        assert False, "Real |= should raise TypeError"
    except TypeError:
        pass

    try:
        a = Real(10.0)
        a ^= b
        assert False, "Real ^= should raise TypeError"
    except TypeError:
        pass

    try:
        a = Real(10.0)
        a <<= b
        assert False, "Real <<= should raise TypeError"
    except TypeError:
        pass

    try:
        a = Real(10.0)
        a >>= b
        assert False, "Real >>= should raise TypeError"
    except TypeError:
        pass


def test():
    a = Integer(1)
    b = Real(2.0)
    c = Complex(1 + 1j)
    # Test __add__
    assert a + 1 == Integer(2), "Integer addition failed"
    assert 1 + a == Integer(2), "Integer reverse addition failed"
    assert b + 1 == Real(3.0), "Real addition failed"
    assert 1 + b == Real(3.0), "Real reverse addition failed"
    assert c + 1 == Complex(2 + 1j), "Complex addition failed"
    assert 1 + c == Complex(2 + 1j), "Complex reverse addition failed"
    # Test __sub__
    assert a - 1 == Integer(0), "Integer subtraction failed"
    assert 2 - a == Integer(1), "Integer reverse subtraction failed"
    assert b - 1 == Real(1.0), "Real subtraction failed"
    assert 3.0 - b == Real(1.0), "Real reverse subtraction failed"
    assert c - 1 == Complex(0 + 1j), "Complex subtraction failed"
    assert 2 - c == Complex(1 - 1j), "Complex reverse subtraction failed"
    # Test __mul__
    assert a * 2 == Integer(2), "Integer multiplication failed"
    assert 2 * a == Integer(2), "Integer reverse multiplication failed"
    assert b * 2 == Real(4.0), "Real multiplication failed"
    assert 2 * b == Real(4.0), "Real reverse multiplication failed"
    assert c * 2 == Complex(2 + 2j), "Complex multiplication failed"
    assert 2 * c == Complex(2 + 2j), "Complex reverse multiplication failed"
    # Test __truediv__
    assert b / 2 == Real(1.0), "Real division failed"
    assert 4.0 / b == Real(2.0), "Real reverse division failed"
    assert c / 2 == Complex(0.5 + 0.5j), "Complex division failed"
    assert (1 + 1j) / c == Complex(1 + 0j), "Complex reverse division failed"
    # Test __eq__
    assert a == 1, "Integer equality failed"
    assert b == 2.0, "Real equality failed"
    assert c == (1 + 1j), "Complex equality failed"
    # Test __lt__ and __gt__
    assert a < 2, "Integer less-than failed"
    assert b > 1.0, "Real greater-than failed"


def test_type():
    assert type(Integer(1)) == Integer, "Integer type failed"
    assert type(Real(1.0)) == Real, "Real type failed"
    assert type(Complex(1 + 1j)) == Complex, "Complex type failed"
    assert type(Boolean(True)) == Boolean, "Boolean type failed"

    # + operator
    val = Integer(1) + 1
    assert type(val) == Integer, "Integer type failed"
    val = 1 + Integer(1)
    assert type(val) == Integer, "Integer type failed"

    val = Real(1.0) + 1
    assert type(val) == Real, "Real type failed"
    val = 1 + Real(1.0)
    assert type(val) == Real, "Real type failed"

    val = Complex(1 + 1j) + 1
    assert type(val) == Complex, "Complex type failed"
    val = 1 + Complex(1 + 1j)
    assert type(val) == Complex, "Complex type failed"

    val = Boolean(True) + 1
    assert val == 2, "Boolean type failed"
    assert type(val) == Integer, "Boolean type failed"
    val = 1 + Boolean(True)
    assert val == 2, "Boolean type failed"
    assert type(val) == Integer, "Boolean type failed"

    b = Boolean(True) == 1
    b = 1 == Boolean(True)
    assert type(b) == bool, "Boolean type failed"
    b = Boolean(True) == True
    assert type(b) == bool, "Boolean type failed"
    b = True == Boolean(True)
    assert type(b) == bool, "Boolean type failed"
    b = Boolean(True) == 1.0
    assert type(b) == bool, "Boolean type failed"
    b = 1.0 == Boolean(True)
    assert type(b) == bool, "Boolean type failed"

    # - operator
    val = Integer(1) - 1
    assert type(val) == Integer, "Integer type failed"
    val = 1 - Integer(1)
    assert type(val) == Integer, "Integer type failed"

    # * operator
    val = Integer(1) * 1
    assert type(val) == Integer, "Integer type failed"
    val = 1 * Integer(1)
    assert type(val) == Integer, "Integer type failed"

    # / operator
    val = Integer(1) / 1
    assert type(val) == Real, "Integer type failed, got: " + str(type(val))
    val = 1 / Integer(1)
    assert type(val) == Real, "Integer type failedgot: " + str(type(val))


if __name__ == "__main__":
    test()
    test_Integer()
    test_Real()
    test_Complex()
    test_Boolean()
    test_Matrix()
    test_type()
    test_logic()
    test_inplace()
    test_bitwise()
    test_cast()
    print("All tests passed.")
