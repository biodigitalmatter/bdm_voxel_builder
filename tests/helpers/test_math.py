import pytest

from bdm_voxel_builder.helpers.math import remap


class TestRemap:
    def test_normal(self):
        x = 0.5
        input_domain = (0, 1)
        output_domain = (0, 10)
        expected_result = 5.0
        assert remap(x, output_domain, input_domain) == expected_result

    def test_input_domain_single_point(self):
        x = 0.5
        output_domain = (0, 10)
        input_domain = (1, 1)

        with pytest.raises(ValueError):
            remap(x, output_domain, input_domain)

    def test_output_domain_single_point(self):
        x = 0.5
        output_domain = (5, 5)
        input_domain = (0, 1)

        with pytest.raises(ValueError):
            remap(x, output_domain, input_domain)

    def test_with_zero_div_error(self):
        x = 0.5
        output_domain = (0, 10)
        input_domain = (0, 0)

        with pytest.raises(ValueError):
            remap(x, output_domain, input_domain)
