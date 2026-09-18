from CPVM.code.utils.batch_evaluation import _summary_metrics


def test_summary_metrics_preserve_counts_and_scale_rates():
    metrics = _summary_metrics(
        {
            "num_questions": 198,
            "samples_per_question": 3,
            "average_accuracy": 0.25,
            "test_at_n": 0.5,
            "parse_rate": 0.75,
            "average_output_tokens": 1234.5,
            "length_truncated_rate": 0.125,
        }
    )
    assert metrics == {
        "Avg@3 (%)": 25.0,
        "Test@3 (%)": 50.0,
        "Parse rate (%)": 75.0,
        "Samples/question": 3,
        "Avg output tokens": 1234.5,
        "Length-truncated (%)": 12.5,
        "Questions": 198,
    }
