from src.analysis import run_analysis

def test_run_analysis():
    results = run_analysis()
    # Check if the 'mean' key exists in results
    assert 'mean' in results
    # Ensure the 'mean' value is greater than 0
    assert results['mean'] > 0
