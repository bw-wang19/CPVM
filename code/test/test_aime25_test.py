import asyncio

from CPVM.code.test.aime25_test import (
    distribute_samples,
    get_samples_per_question,
    output_path,
    run_bounded,
    sample_seed_offsets,
)


def test_samples_per_question_supports_the_new_and_legacy_names():
    assert get_samples_per_question({"samples_per_question": 3}) == 3
    assert get_samples_per_question({"repeats": 4}) == 4
    for value in (None, 0, -1, 1.5, "3", True):
        try:
            get_samples_per_question({"samples_per_question": value})
        except ValueError:
            continue
        raise AssertionError(f"accepted invalid sample count: {value!r}")


def test_sample_distribution_preserves_any_positive_count():
    assert distribute_samples(10, 2) == [5, 5]
    assert distribute_samples(5, 2) == [3, 2]
    assert distribute_samples(1, 4) == [1, 0, 0, 0]


def test_sample_seed_offsets_are_contiguous_and_non_overlapping():
    assert sample_seed_offsets([32, 32]) == [0, 32]
    assert sample_seed_offsets([3, 2]) == [0, 3]
    assert sample_seed_offsets([1, 0, 0, 0]) == [0, 1, 1, 1]
    for counts in ([1, -1], [1, 1.5], [True, 1]):
        try:
            sample_seed_offsets(counts)
        except ValueError:
            continue
        raise AssertionError(f"accepted invalid sample counts: {counts!r}")


def test_output_path_records_the_configured_sample_count():
    config = {
        "model_path": "/models/example",
        "output_path": "/results",
        "test_name": "zero-shot",
        "samples_per_question": 3,
        "data_parallel_size": 2,
    }
    assert output_path(config).name == "aime25-example-zero-shot-mean3-dp2.json"
    config["test_name"] = "zero-shot-mean3-dp2"
    assert output_path(config).name == "aime25-example-zero-shot-mean3-dp2.json"


def test_bounded_runner_refills_before_a_slow_request_finishes():
    async def scenario():
        events = []
        active = 0
        maximum_active = 0
        replacement_started = asyncio.Event()
        consumed = []

        async def operation(item):
            nonlocal active, maximum_active
            active += 1
            maximum_active = max(maximum_active, active)
            events.append(("start", item))
            if item == "slow":
                await replacement_started.wait()
            elif item == "fast":
                await asyncio.sleep(0)
            else:
                replacement_started.set()
            events.append(("end", item))
            active -= 1
            return item.upper()

        await asyncio.wait_for(
            run_bounded(
                ["slow", "fast", "replacement"],
                2,
                operation,
                lambda item, result: consumed.append((item, result)),
            ),
            timeout=1,
        )
        return events, maximum_active, consumed

    events, maximum_active, consumed = asyncio.run(scenario())
    assert maximum_active == 2
    assert events.index(("start", "replacement")) < events.index(("end", "slow"))
    assert set(consumed) == {
        ("slow", "SLOW"),
        ("fast", "FAST"),
        ("replacement", "REPLACEMENT"),
    }

