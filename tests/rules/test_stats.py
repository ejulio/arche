import numpy as np

from conftest import create_result
from arche.rules.result import Level, Outcome
from arche.rules.stats import compare_field_distribution

np.random.seed(42)


def test_compare_field_distribution_fail(mocker):
    prices_1 = np.random.normal(126.99, 25.3, size=1000)
    prices_2 = np.random.normal(299.56, 78.9, size=2000)
    result = compare_field_distribution(prices_1, prices_2, field="price")

    assert result == create_result(
        "Field distribution",
        {
            Level.WARNING: [
                (Outcome.FAILED, '"price" distribution differs between jobs')
            ]
        },
    )


def test_compare_field_distribution_succeed(mocker):
    prices_1 = np.random.normal(126.99, 25.3, size=790)
    prices_2 = np.random.normal(129.27, 22.78, size=540)
    result = compare_field_distribution(prices_1, prices_2, field="price")

    assert result == create_result("Field distribution", {})
