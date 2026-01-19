"""
Technique Detector

Detects programming techniques used in generated code to check
if the model violated denial constraints.

This is a simplified version for the MVP. Full version would use
AST parsing or GPT-4 like NeoCoder does.
"""

import re
import ast
from typing import List, Set, Dict, Any


class TechniqueDetector:
    """
    Detect programming techniques in generated code.

    For MVP, uses pattern matching and AST parsing.
    Can be extended with GPT-4 for more sophisticated detection.
    """

    def __init__(self):
        """Initialize technique detector with pattern definitions."""
        # Pattern-based detection rules
        self.patterns = {
            'for loop': [
                r'\bfor\s+\w+\s+in\s+',
                r'\bfor\s+\w+\s*,\s*\w+\s+in\s+',
            ],
            'while loop': [
                r'\bwhile\s+.+:',
            ],
            'if statement': [
                r'\bif\s+.+:',
            ],
            'list comprehension': [
                r'\[.+\s+for\s+.+\s+in\s+.+\]',
            ],
            'dict comprehension': [
                r'\{.+:.+\s+for\s+.+\s+in\s+.+\}',
            ],
            'set comprehension': [
                r'\{.+\s+for\s+.+\s+in\s+.+\}',
            ],
            'lambda': [
                r'\blambda\s+.+:',
            ],
            'recursion': [
                # Detected via AST (function calling itself)
            ],
            'class': [
                r'\bclass\s+\w+',
            ],
            'try except': [
                r'\btry\s*:',
                r'\bexcept\s+',
            ],
        }

        # Built-in function patterns
        self.builtin_patterns = {
            'sorted() builtin': [r'\bsorted\s*\('],
            'sum() builtin': [r'\bsum\s*\('],
            'map() builtin': [r'\bmap\s*\('],
            'filter() builtin': [r'\bfilter\s*\('],
            'zip() builtin': [r'\bzip\s*\('],
            'enumerate() builtin': [r'\benumerate\s*\('],
            'max() builtin': [r'\bmax\s*\('],
            'min() builtin': [r'\bmin\s*\('],
            'any() builtin': [r'\bany\s*\('],
            'all() builtin': [r'\ball\s*\('],
        }

        self.patterns.update(self.builtin_patterns)

    def detect(self, code: str) -> Set[str]:
        """
        Detect all techniques used in the code.

        Args:
            code: Python code string

        Returns:
            Set of technique names detected
        """
        detected = set()

        # AST-based detection (more robust) - do this FIRST
        try:
            ast_detected = self._detect_with_ast(code)
            detected.update(ast_detected)
        except SyntaxError:
            # If code has syntax errors, fall back to pattern matching
            pass

        # Pattern-based detection (only for things not detected by AST)
        # Skip 'for loop', 'while loop', 'if statement' if AST succeeded
        # since AST is more accurate
        skip_patterns = set()
        if detected:  # If AST worked
            skip_patterns = {'for loop', 'while loop', 'if statement',
                           'list comprehension', 'dict comprehension', 'set comprehension'}

        for technique, patterns in self.patterns.items():
            if technique in skip_patterns:
                continue
            for pattern in patterns:
                if re.search(pattern, code):
                    detected.add(technique)
                    break

        return detected

    def _detect_with_ast(self, code: str) -> Set[str]:
        """
        Use AST parsing for more sophisticated detection.

        Args:
            code: Python code string

        Returns:
            Set of techniques detected via AST
        """
        detected = set()

        try:
            tree = ast.parse(code)
        except SyntaxError:
            return detected

        # Detect recursion
        recursion_detector = RecursionDetector()
        recursion_detector.visit(tree)
        if recursion_detector.has_recursion:
            detected.add('recursion')

        # Detect specific patterns
        for node in ast.walk(tree):
            # Detect for loops (not comprehensions)
            if isinstance(node, ast.For):
                detected.add('for loop')

            # Detect while loops
            if isinstance(node, ast.While):
                detected.add('while loop')

            # Detect if statements
            if isinstance(node, ast.If):
                detected.add('if statement')

            # Detect list comprehensions
            if isinstance(node, ast.ListComp):
                detected.add('list comprehension')

            # Detect dict comprehensions
            if isinstance(node, ast.DictComp):
                detected.add('dict comprehension')

            # Detect set comprehensions
            if isinstance(node, ast.SetComp):
                detected.add('set comprehension')

            # Detect lambda
            if isinstance(node, ast.Lambda):
                detected.add('lambda')

            # Detect class definitions
            if isinstance(node, ast.ClassDef):
                detected.add('class')

            # Detect try/except
            if isinstance(node, ast.Try):
                detected.add('try except')

            # Detect list.sort() method
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute):
                    if node.func.attr == 'sort':
                        detected.add('list.sort() method')

            # Detect string methods
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute):
                    if node.func.attr in ['split', 'join', 'strip', 'replace']:
                        detected.add(f'{node.func.attr}() method')

        return detected

    def check_violations(
        self,
        code: str,
        denied_techniques: List[str]
    ) -> Dict[str, Any]:
        """
        Check if code violates any denial constraints.

        Args:
            code: Generated code string
            denied_techniques: List of techniques that should not be used

        Returns:
            Dictionary with:
            {
                'violations': List[str],      # Techniques used that were denied
                'num_violations': int,        # Number of violations
                'detected': Set[str],         # All techniques detected
                'compliant': bool,            # True if no violations
            }
        """
        detected = self.detect(code)

        # Normalize for comparison (lowercase, strip whitespace)
        detected_normalized = {t.lower().strip() for t in detected}
        denied_normalized = {t.lower().strip() for t in denied_techniques}

        # Find violations
        violations = []
        for denied in denied_techniques:
            denied_norm = denied.lower().strip()
            if denied_norm in detected_normalized:
                violations.append(denied)

        return {
            'violations': violations,
            'num_violations': len(violations),
            'detected': detected,
            'compliant': len(violations) == 0,
        }


class RecursionDetector(ast.NodeVisitor):
    """AST visitor to detect recursion."""

    def __init__(self):
        self.has_recursion = False
        self.function_names = set()
        self.current_function = None

    def visit_FunctionDef(self, node):
        """Visit function definition."""
        # Save current function context
        prev_function = self.current_function
        self.current_function = node.name
        self.function_names.add(node.name)

        # Visit function body
        self.generic_visit(node)

        # Restore context
        self.current_function = prev_function

    def visit_Call(self, node):
        """Visit function call."""
        # Check if calling current function (recursion)
        if isinstance(node.func, ast.Name):
            if node.func.id == self.current_function:
                self.has_recursion = True

        self.generic_visit(node)


if __name__ == "__main__":
    # Test the technique detector
    print("="*80)
    print("Testing Technique Detector")
    print("="*80)

    detector = TechniqueDetector()

    # Test 1: Code with for loop
    print("\n--- Test 1: For loop detection ---")
    code1 = """
def solve(nums):
    result = 0
    for num in nums:
        result += num
    return result
"""
    detected = detector.detect(code1)
    print(f"Detected techniques: {detected}")
    assert 'for loop' in detected

    # Test 2: Code with list comprehension
    print("\n--- Test 2: List comprehension ---")
    code2 = """
def solve(nums):
    return [x * 2 for x in nums]
"""
    detected = detector.detect(code2)
    print(f"Detected techniques: {detected}")
    assert 'list comprehension' in detected

    # Test 3: Code with recursion
    print("\n--- Test 3: Recursion detection ---")
    code3 = """
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)
"""
    detected = detector.detect(code3)
    print(f"Detected techniques: {detected}")
    assert 'recursion' in detected

    # Test 4: Code with built-in functions
    print("\n--- Test 4: Built-in functions ---")
    code4 = """
def solve(nums):
    return sum(sorted(nums))
"""
    detected = detector.detect(code4)
    print(f"Detected techniques: {detected}")
    assert 'sorted() builtin' in detected
    assert 'sum() builtin' in detected

    # Test 5: Check violations
    print("\n--- Test 5: Violation checking ---")
    code5 = """
def solve(nums):
    result = []
    for num in nums:
        result.append(num * 2)
    return result
"""
    denied = ['for loop', 'while loop']
    violations = detector.check_violations(code5, denied)
    print(f"Denied techniques: {denied}")
    print(f"Violations: {violations['violations']}")
    print(f"Compliant: {violations['compliant']}")
    assert not violations['compliant']
    assert 'for loop' in violations['violations']

    # Test 6: Compliant code
    print("\n--- Test 6: Compliant code ---")
    code6 = """
def solve(nums):
    return [num * 2 for num in nums]
"""
    violations = detector.check_violations(code6, denied)
    print(f"Denied techniques: {denied}")
    print(f"Detected: {violations['detected']}")
    print(f"Compliant: {violations['compliant']}")
    assert violations['compliant']

    print("\n" + "="*80)
    print("✅ All tests passed!")
    print("="*80)
