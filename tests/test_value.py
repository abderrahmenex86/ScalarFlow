import math

import pytest

from scalarflow.engine.value import Value


class TestValueCreation:
    def test_init_basic(self):
        v = Value(5.0)
        assert v.data == 5.0
        assert v.grad == 0.0
        assert v._prev == set()
        assert v.op == ""
        assert v.label == ""

    def test_init_with_params(self):
        a = Value(1.0)
        b = Value(2.0)
        c = Value(3.0, (a, b), "+", "c")
        assert c.data == 3.0
        assert c._prev == {a, b}
        assert c.op == "+"
        assert c.label == "c"


class TestAddition:
    def test_add_values(self):
        x = Value(2.0)
        y = Value(3.0)
        result = x + y
        assert result.data == 5.0
        assert result._prev == {x, y}
        assert result.op == "+"

    def test_add_int(self):
        x = Value(2.0)
        result = x + 3
        assert result.data == 5.0

    def test_add_float(self):
        x = Value(2.0)
        result = x + 3.5
        assert result.data == 5.5

    def test_radd_int(self):
        x = Value(2.0)
        result = 3 + x
        assert result.data == 5.0

    def test_radd_float(self):
        x = Value(2.0)
        result = 3.5 + x
        assert result.data == 5.5

    def test_add_backward(self):
        x = Value(2.0)
        y = Value(3.0)
        result = x + y
        result.backward()
        assert x.grad == 1.0
        assert y.grad == 1.0


class TestSubtraction:
    def test_sub_values(self):
        x = Value(5.0)
        y = Value(3.0)
        result = x - y
        assert result.data == 2.0

    def test_sub_int(self):
        x = Value(5.0)
        result = x - 3
        assert result.data == 2.0

    def test_sub_float(self):
        x = Value(5.0)
        result = x - 3.5
        assert result.data == 1.5

    def test_rsub_int(self):
        x = Value(3.0)
        result = 5 - x
        assert result.data == 2.0

    def test_rsub_float(self):
        x = Value(3.0)
        result = 5.5 - x
        assert result.data == 2.5

    def test_sub_backward(self):
        x = Value(5.0)
        y = Value(3.0)
        result = x - y
        result.backward()
        assert x.grad == 1.0
        assert y.grad == -1.0


class TestMultiplication:
    def test_mul_values(self):
        x = Value(2.0)
        y = Value(3.0)
        result = x * y
        assert result.data == 6.0
        assert result._prev == {x, y}
        assert result.op == "*"

    def test_mul_int(self):
        x = Value(2.0)
        result = x * 3
        assert result.data == 6.0

    def test_mul_float(self):
        x = Value(2.0)
        result = x * 3.5
        assert result.data == 7.0

    def test_rmul_int(self):
        x = Value(2.0)
        result = 3 * x
        assert result.data == 6.0

    def test_rmul_float(self):
        x = Value(2.0)
        result = 3.5 * x
        assert result.data == 7.0

    def test_mul_backward(self):
        x = Value(2.0)
        y = Value(3.0)
        result = x * y
        result.backward()
        assert x.grad == 3.0
        assert y.grad == 2.0


class TestDivision:
    def test_div_values(self):
        v1 = Value(6.0)
        v2 = Value(2.0)
        result = v1 / v2
        assert result.data == 3.0
        assert result._prev == {v1, v2}
        assert result.op == "/"

    def test_div_int(self):
        v = Value(6.0)
        result = v / 2
        assert result.data == 3.0

    def test_div_float(self):
        v = Value(6.0)
        result = v / 2.5
        assert result.data == 2.4

    def test_rdiv_int(self):
        v = Value(2.0)
        result = 6 / v
        assert result.data == 3.0

    def test_rdiv_float(self):
        v = Value(2.0)
        result = 6.5 / v
        assert result.data == 3.25

    def test_div_backward(self):
        v1 = Value(6.0)
        v2 = Value(2.0)
        result = v1 / v2
        result.backward()
        assert v1.grad == 0.5  # 1/2
        assert v2.grad == -1.5  # -6/4


class TestPower:
    def test_pow_int(self):
        v = Value(2.0)
        result = v**3
        assert result.data == 8.0
        assert result._prev == {v}
        assert result.op == "**3"

    def test_pow_float(self):
        v = Value(4.0)
        result = v**0.5
        assert result.data == 2.0

    def test_pow_backward(self):
        v = Value(2.0)
        result = v**3
        result.backward()
        assert v.grad == 12.0  # 3 * 2^2


class TestNegation:
    def test_neg(self):
        v = Value(5.0)
        result = -v
        assert result.data == -5.0

    def test_neg_backward(self):
        v = Value(5.0)
        result = -v
        result.backward()
        assert v.grad == -1.0


class TestMathFunctions:
    def test_exp(self):
        v = Value(1.0)
        result = v.exp()
        assert abs(result.data - math.e) < 1e-10
        assert result._prev == {v}
        assert result.op == "exp"

    def test_exp_backward(self):
        v = Value(1.0)
        result = v.exp()
        result.backward()
        assert abs(v.grad - math.e) < 1e-10

    def test_tanh(self):
        v = Value(0.0)
        result = v.tanh()
        assert abs(result.data - 0.0) < 1e-10
        assert result._prev == {v}
        assert result.op == "tanh"

    def test_tanh_backward(self):
        v = Value(0.0)
        result = v.tanh()
        result.backward()
        assert abs(v.grad - 1.0) < 1e-10  # derivative of tanh(0) is 1


class TestBackward:
    def test_backward_simple(self):
        v = Value(2.0)
        v.backward()
        assert v.grad == 1.0

    def test_backward_chain(self):
        v1 = Value(2.0)
        v2 = Value(3.0)
        v3 = v1 * v2
        v4 = v3 + Value(1.0)
        v4.backward()
        assert v1.grad == 3.0
        assert v2.grad == 2.0
        assert v3.grad == 1.0

    def test_backward_complex(self):
        x = Value(2.0)
        y = Value(3.0)
        z = x * y + x**2
        z.backward()
        assert x.grad == 7.0  # y + 2*x = 3 + 4
        assert y.grad == 2.0  # x


class TestTypeAssertions:
    def test_add_invalid_type(self):
        v = Value(1.0)
        with pytest.raises(AssertionError):
            v + "string"

    def test_radd_invalid_type(self):
        v = Value(1.0)
        with pytest.raises(AssertionError):
            "string" + v

    def test_sub_invalid_type(self):
        v = Value(1.0)
        with pytest.raises(AssertionError):
            v - "string"

    def test_rsub_invalid_type(self):
        v = Value(1.0)
        with pytest.raises(AssertionError):
            "string" - v

    def test_mul_invalid_type(self):
        v = Value(1.0)
        with pytest.raises(AssertionError):
            v * "string"

    def test_rmul_invalid_type(self):
        v = Value(1.0)
        with pytest.raises(AssertionError):
            "string" * v

    def test_div_invalid_type(self):
        v = Value(1.0)
        with pytest.raises(AssertionError):
            v / "string"

    def test_rdiv_invalid_type(self):
        v = Value(1.0)
        with pytest.raises(AssertionError):
            "string" / v

    def test_pow_invalid_type(self):
        v = Value(1.0)
        with pytest.raises(AssertionError):
            v ** "string"


class TestEdgeCases:
    def test_zero_operations(self):
        v1 = Value(0.0)
        v2 = Value(5.0)
        assert (v1 + v2).data == 5.0
        assert (v1 * v2).data == 0.0
        assert (v2 / Value(1.0)).data == 5.0

    def test_negative_values(self):
        v = Value(-3.0)
        assert (v + Value(5.0)).data == 2.0
        assert (v * Value(2.0)).data == -6.0
        assert (v**2).data == 9.0

    def test_fractional_power(self):
        v = Value(9.0)
        result = v**0.5
        assert abs(result.data - 3.0) < 1e-10
