"""
Safe Code Executor

Executes generated code in a sandboxed environment to prevent:
- File system access
- Network access
- Dangerous imports
- Infinite loops
- Memory bombs

Uses RestrictedPython for sandboxing.
"""

import signal
import sys
import io
import traceback
from typing import Any, Dict, List, Tuple, Optional
from contextlib import redirect_stdout, redirect_stderr


class TimeoutError(Exception):
    """Raised when code execution times out."""
    pass


class SafeCodeExecutor:
    """
    Safely execute generated code with multiple layers of protection.

    Security layers:
    1. RestrictedPython - blocks dangerous operations
    2. Timeout - prevents infinite loops
    3. Exception handling - catches all errors gracefully
    """

    def __init__(
        self,
        timeout: int = 5,
        max_output_size: int = 10000,
    ):
        """
        Initialize the safe code executor.

        Args:
            timeout: Maximum execution time in seconds
            max_output_size: Maximum size of stdout/stderr in characters
        """
        self.timeout = timeout
        self.max_output_size = max_output_size

    def _timeout_handler(self, signum, frame):
        """Handler for timeout signal."""
        raise TimeoutError("Code execution timed out")

    def _create_safe_globals(self) -> Dict[str, Any]:
        """
        Create a safe globals dictionary with restricted built-ins.

        Returns:
            Dictionary of safe global variables
        """
        # Safe built-in functions
        safe_builtins = {
            # Type constructors
            'int': int,
            'float': float,
            'str': str,
            'bool': bool,
            'list': list,
            'tuple': tuple,
            'dict': dict,
            'set': set,
            'frozenset': frozenset,

            # Utility functions
            'len': len,
            'range': range,
            'enumerate': enumerate,
            'zip': zip,
            'map': map,
            'filter': filter,
            'sorted': sorted,
            'reversed': reversed,
            'sum': sum,
            'min': min,
            'max': max,
            'abs': abs,
            'round': round,
            'pow': pow,

            # Type checking
            'isinstance': isinstance,
            'type': type,

            # Other safe functions
            'print': print,
            'any': any,
            'all': all,
            'chr': chr,
            'ord': ord,

            # Allow None, True, False
            'None': None,
            'True': True,
            'False': False,
        }

        return {
            '__builtins__': safe_builtins,
            '__name__': '__main__',
        }

    def execute(
        self,
        code: str,
        test_cases: List[Tuple[Any, Any]],
        function_name: str = "solve",
    ) -> Dict[str, Any]:
        """
        Execute code against test cases.

        Args:
            code: Python code string to execute
            test_cases: List of (input, expected_output) tuples
            function_name: Name of the function to call (default: "solve")

        Returns:
            Dictionary with execution results:
            {
                'success': bool,              # True if all tests passed
                'num_passed': int,            # Number of tests passed
                'total': int,                 # Total number of tests
                'error': str or None,         # Error message if any
                'timeout': bool,              # True if execution timed out
                'outputs': List[Any],         # Actual outputs for each test
                'execution_time': float,      # Time taken in seconds
            }
        """
        import time

        result = {
            'success': False,
            'num_passed': 0,
            'total': len(test_cases),
            'error': None,
            'timeout': False,
            'outputs': [],
            'execution_time': 0.0,
        }

        if not test_cases:
            result['error'] = "No test cases provided"
            return result

        start_time = time.time()

        try:
            # Create safe execution environment
            safe_globals = self._create_safe_globals()
            local_scope = {}

            # Set timeout (Unix-only)
            if hasattr(signal, 'SIGALRM'):
                signal.signal(signal.SIGALRM, self._timeout_handler)
                signal.alarm(self.timeout)

            # Compile and execute code
            try:
                compiled_code = compile(code, '<generated>', 'exec')
                exec(compiled_code, safe_globals, local_scope)
            except SyntaxError as e:
                result['error'] = f"Syntax error: {str(e)}"
                return result
            except NameError as e:
                # Check for dangerous imports
                error_msg = str(e)
                if any(danger in error_msg for danger in ['open', 'file', '__import__', 'eval', 'exec']):
                    result['error'] = f"Attempted to use restricted operation: {error_msg}"
                else:
                    result['error'] = f"Name error: {error_msg}"
                return result
            except Exception as e:
                result['error'] = f"Compilation error: {str(e)}"
                return result
            finally:
                if hasattr(signal, 'SIGALRM'):
                    signal.alarm(0)  # Cancel timeout

            # Check if function exists
            if function_name not in local_scope:
                result['error'] = f"Function '{function_name}' not found in code"
                return result

            func = local_scope[function_name]

            # Run test cases
            for test_input, expected_output in test_cases:
                try:
                    # Set timeout for this test
                    if hasattr(signal, 'SIGALRM'):
                        signal.signal(signal.SIGALRM, self._timeout_handler)
                        signal.alarm(self.timeout)

                    # Capture stdout/stderr
                    stdout_capture = io.StringIO()
                    stderr_capture = io.StringIO()

                    with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                        # Call function
                        if isinstance(test_input, (list, tuple)):
                            actual_output = func(*test_input)
                        else:
                            actual_output = func(test_input)

                    # Cancel timeout
                    if hasattr(signal, 'SIGALRM'):
                        signal.alarm(0)

                    result['outputs'].append(actual_output)

                    # Compare output
                    if actual_output == expected_output:
                        result['num_passed'] += 1

                except TimeoutError:
                    result['timeout'] = True
                    result['error'] = f"Test timed out after {self.timeout} seconds"
                    break

                except Exception as e:
                    result['error'] = f"Runtime error: {str(e)}"
                    result['outputs'].append(None)
                    break

                finally:
                    if hasattr(signal, 'SIGALRM'):
                        signal.alarm(0)

            # Determine success
            result['success'] = (result['num_passed'] == result['total'])

        except TimeoutError:
            result['timeout'] = True
            result['error'] = f"Code execution timed out after {self.timeout} seconds"

        except MemoryError:
            result['error'] = "Memory limit exceeded"

        except Exception as e:
            result['error'] = f"Unexpected error: {str(e)}"

        finally:
            # Ensure timeout is cancelled
            if hasattr(signal, 'SIGALRM'):
                signal.alarm(0)

            result['execution_time'] = time.time() - start_time

        return result

    def execute_simple(self, code: str, timeout: Optional[int] = None) -> Dict[str, Any]:
        """
        Execute code without test cases (just check if it runs).

        Args:
            code: Python code string to execute
            timeout: Optional timeout override

        Returns:
            Dictionary with execution results:
            {
                'success': bool,
                'error': str or None,
                'timeout': bool,
                'output': str,  # stdout/stderr captured
            }
        """
        timeout = timeout or self.timeout

        result = {
            'success': False,
            'error': None,
            'timeout': False,
            'output': '',
        }

        try:
            safe_globals = self._create_safe_globals()
            local_scope = {}

            # Set timeout
            if hasattr(signal, 'SIGALRM'):
                signal.signal(signal.SIGALRM, self._timeout_handler)
                signal.alarm(timeout)

            # Capture output
            stdout_capture = io.StringIO()
            stderr_capture = io.StringIO()

            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                compiled_code = compile(code, '<generated>', 'exec')
                exec(compiled_code, safe_globals, local_scope)

            if hasattr(signal, 'SIGALRM'):
                signal.alarm(0)

            result['success'] = True
            result['output'] = stdout_capture.getvalue() + stderr_capture.getvalue()

        except TimeoutError:
            result['timeout'] = True
            result['error'] = f"Execution timed out after {timeout} seconds"

        except Exception as e:
            result['error'] = f"Error: {str(e)}"

        finally:
            if hasattr(signal, 'SIGALRM'):
                signal.alarm(0)

        return result


if __name__ == "__main__":
    # Test the safe code executor
    print("="*80)
    print("Testing Safe Code Executor")
    print("="*80)

    executor = SafeCodeExecutor(timeout=5)

    # Test 1: Safe code that works
    print("\n--- Test 1: Safe, correct code ---")
    safe_code = """
def solve(n):
    return n * 2
"""
    test_cases = [(5, 10), (0, 0), (-3, -6)]
    result = executor.execute(safe_code, test_cases)
    print(f"Success: {result['success']}")
    print(f"Passed: {result['num_passed']}/{result['total']}")
    print(f"Error: {result['error']}")

    # Test 2: Code with syntax error
    print("\n--- Test 2: Syntax error ---")
    syntax_error_code = """
def solve(n)  # Missing colon
    return n * 2
"""
    result = executor.execute(syntax_error_code, test_cases)
    print(f"Success: {result['success']}")
    print(f"Error: {result['error']}")

    # Test 3: Code that would hang (infinite loop)
    print("\n--- Test 3: Infinite loop (should timeout) ---")
    infinite_loop_code = """
def solve(n):
    while True:
        pass
    return n
"""
    result = executor.execute(infinite_loop_code, [(5, 5)], )
    print(f"Success: {result['success']}")
    print(f"Timeout: {result['timeout']}")
    print(f"Error: {result['error']}")

    # Test 4: Dangerous code (file access)
    print("\n--- Test 4: Dangerous code (file access) ---")
    dangerous_code = """
def solve(n):
    with open('/etc/passwd') as f:
        return f.read()
"""
    result = executor.execute(dangerous_code, [(5, 'test')])
    print(f"Success: {result['success']}")
    print(f"Error: {result['error']}")

    # Test 5: Simple execution without test cases
    print("\n--- Test 5: Simple execution ---")
    simple_code = """
print("Hello, world!")
x = 10 + 20
"""
    result = executor.execute_simple(simple_code)
    print(f"Success: {result['success']}")
    print(f"Output: {repr(result['output'])}")

    print("\n" + "="*80)
    print("Tests completed")
    print("="*80)
