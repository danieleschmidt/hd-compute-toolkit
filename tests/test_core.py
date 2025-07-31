"""Tests for core HDC functionality."""

import pytest
from hd_compute.core import HDCompute


def test_hdc_initialization():
    """Test HDCompute abstract class cannot be instantiated directly."""
    with pytest.raises(TypeError):
        HDCompute(dim=1000)