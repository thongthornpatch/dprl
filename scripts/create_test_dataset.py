"""
Create minimal test dataset for local pilot run.
"""
import json
from pathlib import Path

# Create data directory
data_dir = Path(__file__).parent.parent / "data" / "neocoder"
data_dir.mkdir(parents=True, exist_ok=True)

# Minimal test dataset with 5 simple problems
test_problems = [
    {
        "problem_id": "test_001",
        "description": "Return the sum of two numbers",
        "test_cases": [[1, 2, 3], [5, 10, 15], [0, 0, 0], [-5, 3, -2]],
        "rounds": [
            {
                "constraints": [],
                "example_code": "def solve(a, b):\n    return a + b"
            },
            {
                "constraints": ["while loop"],
                "example_code": "def solve(a, b):\n    return a + b"
            },
            {
                "constraints": ["while loop", "for loop"],
                "example_code": "def solve(a, b):\n    return a + b"
            }
        ]
    },
    {
        "problem_id": "test_002",
        "description": "Return x multiplied by 2",
        "test_cases": [[5, 10], [0, 0], [3, 6], [-2, -4]],
        "rounds": [
            {
                "constraints": [],
                "example_code": "def solve(x):\n    return x * 2"
            },
            {
                "constraints": ["multiplication operator"],
                "example_code": "def solve(x):\n    return x + x"
            }
        ]
    },
    {
        "problem_id": "test_003",
        "description": "Find maximum of two numbers",
        "test_cases": [[5, 3, 5], [1, 10, 10], [7, 7, 7], [-5, -3, -3]],
        "rounds": [
            {
                "constraints": [],
                "example_code": "def solve(a, b):\n    if a > b:\n        return a\n    return b"
            },
            {
                "constraints": ["if statement"],
                "example_code": "def solve(a, b):\n    return max(a, b)"
            }
        ]
    },
    {
        "problem_id": "test_004",
        "description": "Count from 1 to n",
        "test_cases": [[3, [1,2,3]], [1, [1]], [5, [1,2,3,4,5]]],
        "rounds": [
            {
                "constraints": [],
                "example_code": "def solve(n):\n    result = []\n    for i in range(1, n+1):\n        result.append(i)\n    return result"
            },
            {
                "constraints": ["for loop"],
                "example_code": "def solve(n):\n    return list(range(1, n+1))"
            }
        ]
    },
    {
        "problem_id": "test_005",
        "description": "Check if number is even",
        "test_cases": [[2, True], [3, False], [0, True], [7, False]],
        "rounds": [
            {
                "constraints": [],
                "example_code": "def solve(n):\n    return n % 2 == 0"
            },
            {
                "constraints": ["modulo operator"],
                "example_code": "def solve(n):\n    return n // 2 * 2 == n"
            }
        ]
    }
]

# Save as JSONL
output_file = data_dir / "neocoder.jsonl"
with open(output_file, 'w') as f:
    for problem in test_problems:
        f.write(json.dumps(problem) + '\n')

print(f"✅ Created test dataset: {output_file}")
print(f"   Problems: {len(test_problems)}")
print(f"   Total rounds: {sum(len(p['rounds']) for p in test_problems)}")
