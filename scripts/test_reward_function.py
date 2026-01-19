#!/usr/bin/env python3
"""
Test the reward function with real NeoCoder data.

This validates that Phase 3 (reward function) works end-to-end.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from rewards import RewardFunction
from data.neocoder_loader import NeoCoderLoader


def test_reward_with_real_data():
    """Test reward function with actual NeoCoder problems."""
    print("="*80)
    print("TEST: Reward Function with Real NeoCoder Data")
    print("="*80)

    # Load NeoCoder data
    try:
        loader = NeoCoderLoader()
        problems = loader.load()
        print(f"\n✅ Loaded {len(problems)} NeoCoder problems")
    except FileNotFoundError:
        print("\n❌ NeoCoder dataset not found. Run: python scripts/download_neocoder.py")
        return False

    # Initialize simplified reward function
    reward_fn = RewardFunction(
        correctness_weight=1.0,
        denial_penalty_weight=0.5,
    )

    print("\n" + "="*80)
    print("Testing with First Problem")
    print("="*80)

    # Get first problem
    problem = problems[0]
    print(f"\nProblem ID: {problem.problem_id}")
    print(f"Number of test cases: {len(problem.test_cases)}")

    # Get round 0 (original problem, no denial)
    round0 = problem.get_round(0)
    print(f"\nRound 0 (no denial constraints):")
    print(f"  Constraints: {round0['constraints']}")

    # Get round 1 (with denial constraints)
    if len(problem.rounds) > 1:
        round1 = problem.get_round(1)
        print(f"\nRound 1 (with denial constraints):")
        print(f"  Constraints: {round1['constraints']}")
        example_code = round1['example_code']
        denied_techniques = round1['constraints']
    else:
        print("\n⚠️  Problem only has 1 round")
        example_code = round0['example_code']
        denied_techniques = []

    # Test with the example code from NeoCoder
    print("\n" + "-"*80)
    print("Testing Example Code from NeoCoder")
    print("-"*80)
    print(f"\nCode (first 300 chars):\n{example_code[:300]}...")

    # Compute reward
    result = reward_fn.compute_reward(
        generated_code=example_code,
        test_cases=problem.test_cases,
        denied_techniques=denied_techniques,
    )

    print(f"\n📊 Reward Results:")
    print(f"  Total Reward: {result['total_reward']:.3f}")
    print(f"  Correctness: {result['correctness_score']:.3f}")
    print(f"  Denial Penalty: {result['denial_penalty']:.3f}")
    print(f"  Num Violations: {result['num_violations']}")
    print(f"  Success: {result['success']}")

    print(f"\n  Execution:")
    print(f"    Passed: {result['execution_result']['num_passed']}/{result['execution_result']['total']}")
    if result['execution_result']['error']:
        print(f"    Error: {result['execution_result']['error'][:100]}...")
    print(f"    Timeout: {result['execution_result']['timeout']}")

    print(f"\n  Violations:")
    print(f"    Denied: {denied_techniques}")
    print(f"    Detected: {list(result['violation_result']['detected'])[:5]}")
    print(f"    Violations: {result['violation_result']['violations']}")
    print(f"    Compliant: {result['violation_result']['compliant']}")

    return True


def test_reward_scenarios():
    """Test different reward scenarios."""
    print("\n" + "="*80)
    print("TEST: Different Reward Scenarios")
    print("="*80)

    reward_fn = RewardFunction(
        correctness_weight=1.0,
        denial_penalty_weight=0.5,
    )

    # Scenario 1: Perfect solution, no violations
    print("\n--- Scenario 1: Perfect solution, no violations ---")
    code1 = """
def solve(x):
    return x + 1
"""
    test_cases1 = [(5, 6), (0, 1), (-1, 0)]
    denied1 = ['while loop']

    result1 = reward_fn.compute_reward(code1, test_cases1, denied1)
    print(f"Reward: {result1['total_reward']:.3f} (should be > 1.0)")
    print(f"Success: {result1['success']} (should be True)")
    print(f"Violations: {result1['violation_result']['violations']} (should be [])")

    # Scenario 2: Correct but uses denied technique
    print("\n--- Scenario 2: Correct but uses denied technique ---")
    code2 = """
def solve(x):
    result = x
    while result < x + 1:
        result += 1
        break
    return result
"""
    result2 = reward_fn.compute_reward(code2, test_cases1, denied1)
    print(f"Reward: {result2['total_reward']:.3f} (should be < 1.0 due to penalty)")
    print(f"Success: {result2['success']} (should be True)")
    print(f"Violations: {result2['violation_result']['violations']} (should contain 'while loop')")

    # Scenario 3: Incorrect solution
    print("\n--- Scenario 3: Incorrect solution ---")
    code3 = """
def solve(x):
    return x  # Wrong!
"""
    result3 = reward_fn.compute_reward(code3, test_cases1, denied1)
    print(f"Reward: {result3['total_reward']:.3f} (should be ≈ 0)")
    print(f"Success: {result3['success']} (should be False)")

    # Scenario 4: Partial correctness
    print("\n--- Scenario 4: Partial correctness (1/3 tests pass) ---")
    code4 = """
def solve(x):
    if x == 5:
        return 6
    return 0
"""
    result4 = reward_fn.compute_reward(code4, test_cases1, denied1)
    print(f"Reward: {result4['total_reward']:.3f}")
    print(f"Passed: {result4['execution_result']['num_passed']}/3")
    print(f"Correctness score: {result4['correctness_score']:.3f} (should be ≈ 0.33)")

    return True


def main():
    """Run all tests."""
    print("="*80)
    print("PHASE 3: REWARD FUNCTION TESTS")
    print("="*80)

    tests = [
        ("Real NeoCoder Data", test_reward_with_real_data),
        ("Different Scenarios", test_reward_scenarios),
    ]

    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"\n❌ {test_name} crashed: {e}")
            import traceback
            traceback.print_exc()
            results[test_name] = False

    # Print summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for test_name, passed_test in results.items():
        status = "✅ PASS" if passed_test else "❌ FAIL"
        print(f"{status}  {test_name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed! Phase 3 is complete.")
        print("\nNext steps:")
        print("  1. Implement GRPO training loop (Phase 4)")
        print("  2. Create training scripts for NSCC")
        return 0
    else:
        print("\n⚠️  Some tests failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
