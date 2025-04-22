"""Unit tests for the catalytic model parameter transformation functions."""

import pytest
import numpy as np
from scipy.special import logit
from catalytic_model import transform_model_param, ParameterTransformMethod


def test_transform_model_param():
    """
    Test the transform_model_param function with various inputs.
    """
    assert transform_model_param(0, ParameterTransformMethod.INT, (0, 100)) == 50
    assert transform_model_param(0, ParameterTransformMethod.INT, (0,)) == 1
    assert transform_model_param(np.log(15), ParameterTransformMethod.INT, (0,)) == 15
    assert transform_model_param(logit(13/50), ParameterTransformMethod.INT, (0,50)) == 13
    assert transform_model_param(np.log(0.3), ParameterTransformMethod.FLOAT) == 0.3
