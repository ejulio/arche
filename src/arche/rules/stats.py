import numpy as np

from arche.rules.result import Result, Outcome


def compare_field_distribution(group1, group2, field):
    groups_diff = np.abs(np.mean(group1) - np.mean(group2))
    p_value = np.mean(
        list(_compare_sample_diffs_to_base_diff(group1, group2, groups_diff))
    )
    result = Result("Field distribution")
    if p_value <= 0.05:
        result.add_warning(
            Outcome.FAILED, detailed=f'"{field}" distribution differs between jobs'
        )

    return result


def _compare_sample_diffs_to_base_diff(group1, group2, base_diff, N=1000):
    m = len(group1)
    pool = np.hstack([group1, group2])
    for _ in range(N):
        np.random.shuffle(pool)
        group1 = pool[:m]
        group2 = pool[m:]
        yield np.abs(np.mean(group1) - np.mean(group2)) > base_diff
