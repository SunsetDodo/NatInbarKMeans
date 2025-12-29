import subprocess
import os
import sys

# --- Configuration ---
C_SOURCE = "./kmeans.c"
C_EXE = "./kmeans"
PY_SCRIPT = "./kmeans.py"
TIMEOUT_SEC = 10

# Colors for output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

# --- Error Messages (Must match spec exactly) ---
MSG_ERR_GENERIC = "An Error Has Occurred"
MSG_ERR_K = "Incorrect number of clusters!"
MSG_ERR_ITER = "Incorrect maximum iteration!"

def log(msg, color=Colors.ENDC):
    print(f"{color}{msg}{Colors.ENDC}")

def compile_c():
    log(f"🔨 Compiling {C_SOURCE}...", Colors.OKBLUE)
    # Compilation flags as per standard rigorous requirements
    cmd = ["gcc", "-ansi", "-Wall", "-Wextra", "-Werror", "-pedantic-errors", C_SOURCE, "-o", "kmeans", "-lm"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        log("❌ Compilation Failed!", Colors.FAIL)
        print(result.stderr)
        sys.exit(1)
    log("✅ Compilation Successful.\n", Colors.OKGREEN)

def run_program(command, input_str, test_name):
    """Runs a command with input string and returns (return_code, stdout, stderr)"""
    try:
        result = subprocess.run(
            command,
            input=input_str,
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SEC
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -999, "", "TIMEOUT"
    except Exception as e:
        return -1, "", str(e)

def analyze_result(lang, name, rc, out, err, expected_rc, expected_snippet):
    """Analyzes the result of a single run"""

    # Check Timeout
    if rc == -999:
        log(f"   [{lang}] ❌ FAIL: Timeout ({TIMEOUT_SEC}s) - Infinite Loop?", Colors.FAIL)
        return False

    # Check Return Code
    if expected_rc is not None and rc != expected_rc:
        log(f"   [{lang}] ❌ FAIL: Wrong Return Code. Expected {expected_rc}, Got {rc}", Colors.FAIL)
        if err: print(f"      Stderr: {err.strip()}")
        return False

    # Check Output Snippet (if error expected)
    if expected_snippet:
        # Normalize: remove trailing newlines for loose comparison
        if expected_snippet not in out and expected_snippet not in err:
            log(f"   [{lang}] ❌ FAIL: Missing expected message.", Colors.FAIL)
            log(f"      Expected: '{expected_snippet}'", Colors.WARNING)
            log(f"      Got Out:  '{out.strip()}'", Colors.WARNING)
            if rc != 0: log(f"      Got Err:  '{err.strip()}'", Colors.WARNING)
            return False

    return True

def run_test_case(name, input_data, args, expected_rc, expected_snippet=None, check_py=True):
    log(f"🧪 Test: {name}", Colors.HEADER)

    # 1. Test C
    c_cmd = [C_EXE] + args
    c_rc, c_out, c_err = run_program(c_cmd, input_data, name)
    c_pass = analyze_result("C", name, c_rc, c_out, c_err, expected_rc, expected_snippet)

    # 2. Test Python (Optional)
    py_pass = True
    py_out = ""
    if check_py:
        py_cmd = ["python3", PY_SCRIPT] + args
        py_rc, py_out_res, py_err = run_program(py_cmd, input_data, name)
        py_pass = analyze_result("PY", name, py_rc, py_out_res, py_err, expected_rc, expected_snippet)
        py_out = py_out_res

    # 3. Cross-Check (Only if both succeeded and we expect success)
    if expected_rc == 0 and c_pass and py_pass and check_py:
        # Simple whitespace normalization for comparison
        c_clean = c_out.strip().replace("\r\n", "\n")
        py_clean = py_out.strip().replace("\r\n", "\n")

        if c_clean != py_clean:
            log(f"   ⚠️  WARNING: C and Python outputs differ!", Colors.WARNING)
            log(f"      C output length: {len(c_clean)}", Colors.WARNING)
            log(f"      Py output length: {len(py_clean)}", Colors.WARNING)
            # You can print diffs here if needed
            return False

    if c_pass and py_pass:
        log(f"   ✅ PASS", Colors.OKGREEN)
        return True
    else:
        return False

# --- Test Data Generators ---
def gen_valid_input(n_points=10):
    return "\n".join([f"{float(i)},0.0,0.0" for i in range(n_points)]) + "\n"

# --- Helper to load official files ---
def load_file(filename):
    """Safely reads a file from the current directory."""
    try:
        with open(filename, 'r') as f:
            return f.read()
    except FileNotFoundError:
        print(f"⚠️  WARNING: '{filename}' not found. Test will be skipped or fail.")
        return None


# --- Main Execution ---
if __name__ == "__main__":
    compile_c()

    total_tests = 0
    passed_tests = 0

    # Load the official input/output data
    in1 = load_file("input_1.txt")
    out1 = load_file("output_1.txt")
    in2 = load_file("input_2.txt")
    out2 = load_file("output_2.txt")
    in3 = load_file("input_3.txt")
    out3 = load_file("output_3.txt")

    # Ensure outputs are stripped of trailing whitespace for comparison
    if out1: out1 = out1.strip()
    if out2: out2 = out2.strip()
    if out3: out3 = out3.strip()

    tests = [
        # --- Group A: Basic Argument Validation ---
        {
            "name": "No Arguments",
            "args": [],
            "input": "1,2\n3,4",
            "rc": 1,
            "msg": MSG_ERR_GENERIC
        },
        {
            "name": "K is not a number (string)",
            "args": ["abc"],
            "input": "1,2\n3,4",
            "rc": 1, # Depending on implementation, C might interpret 'abc' as 0
            "msg": MSG_ERR_K # Or generic, but let's see. Usually fails K validity.
        },
        {
            "name": "K is float (2.5)",
            "args": ["2.5"],
            "input": "1,2\n3,4",
            "rc": 1, # C converts to 2, Python fails. This tests robustness.
            # We skip 'msg' check here because C might actually run with K=2!
            # If C runs, RC=0. If Py fails, RC=1.
            # This is a tricky case. Let's assume spec says integers only.
            # We expect failure or strictly integer parsing.
            "msg": None
        },
        {
            "name": "Negative K",
            "args": ["-1"],
            "input": "1,2\n3,4",
            "rc": 1,
            "msg": MSG_ERR_K
        },
        {
            "name": "K=0",
            "args": ["0"],
            "input": "1,2\n3,4",
            "rc": 1,
            "msg": MSG_ERR_K
        },
        {
            "name": "K=1 (Boundary)",
            "args": ["1"],
            "input": "1,2\n3,4",
            "rc": 1,
            "msg": MSG_ERR_K
        },

        # --- Group B: Iteration Validation ---
        {
            "name": "Max Iter is not a number",
            "args": ["2", "abc"],
            "input": "1,2\n3,4\n5,6",
            "rc": 1,
            "msg": MSG_ERR_ITER
        },
        {
            "name": "Max Iter = 0",
            "args": ["2", "0"],
            "input": "1,2\n3,4\n5,6",
            "rc": 1,
            "msg": MSG_ERR_ITER
        },
        {
            "name": "Max Iter Negative",
            "args": ["2", "-50"],
            "input": "1,2\n3,4\n5,6",
            "rc": 1,
            "msg": MSG_ERR_ITER
        },
        {
            "name": "Too Many Arguments",
            "args": ["2", "100", "extra"],
            "input": "1,2\n3,4",
            "rc": 1,
            "msg": MSG_ERR_GENERIC
        },

        # --- Group C: Data Logic Constraints (K vs N) ---
        {
            "name": "K = N (Should Fail)",
            "args": ["3"],
            "input": "1,0\n2,0\n3,0",
            "rc": 1,
            "msg": MSG_ERR_K
        },
        {
            "name": "K > N (Should Fail)",
            "args": ["4"],
            "input": "1,0\n2,0\n3,0",
            "rc": 1,
            "msg": MSG_ERR_K
        },

        # --- Group D: Input Parsing Edge Cases ---
        {
            "name": "Empty Input File",
            "args": ["2"],
            "input": "",
            "rc": 1, # Should fail gracefully
            "msg": MSG_ERR_GENERIC
        },
        {
            "name": "Windows Line Endings (CRLF)",
            "args": ["2"],
            "input": "1.0,0.0\r\n2.0,0.0\r\n3.0,0.0\r\n4.0,0.0",
            "rc": 0,
            "msg": None
        },
        {
            "name": "No Newline at End of File",
            "args": ["2"],
            "input": "1.0,0.0\n2.0,0.0\n3.0,0.0",
            "rc": 0,
            "msg": None
        },
        {
            "name": "Empty Lines in Middle (Should Skip)",
            "args": ["2"],
            "input": "1.0,0.0\n\n\n2.0,0.0\n\n3.0,0.0",
            "rc": 0,
            "msg": None
        },
        {
            "name": "Spaces and Tabs around numbers",
            "args": ["2"],
            "input": " 1.0 , 0.0 \n\t2.0, 0.0\t\n  3.0,0.0",
            "rc": 0,
            "msg": None
        },

        # --- Group E: Happy Path & Verification ---
        {
            "name": "Standard Run (K=2, Default Iter)",
            "args": ["2"],
            "input": "1.0,0.0\n2.0,0.0\n10.0,0.0\n11.0,0.0",
            "rc": 0,
            "msg": None
        },
        {
            "name": "Standard Run (K=2, Iter=50)",
            "args": ["2", "50"],
            "input": "1.0,0.0\n2.0,0.0\n10.0,0.0\n11.0,0.0",
            "rc": 0,
            "msg": None
        },
        {
            "name": "High Dimensionality (d=5)",
            "args": ["2"],
            "input": "1,1,1,1,1\n2,2,2,2,2\n10,10,10,10,10",
            "rc": 0,
            "msg": None
        },

        # --- Group F: Advanced Argument Parsing (strtol checks) ---
        {
            "name": "Strict Integer Check: K=2.5",
            "args": ["2.5"],
            "input": "1,0\n2,0\n3,0",
            "rc": 1,
            "msg": MSG_ERR_K
        },
        {
            "name": "Garbage after K: K=2x",
            "args": ["2x"],
            "input": "1,0\n2,0\n3,0",
            "rc": 1,
            "msg": MSG_ERR_K
        },
        {
            "name": "Strict Integer Check: Iter=100.5",
            "args": ["2", "100.5"],
            "input": "1,0\n2,0\n3,0",
            "rc": 1,
            "msg": MSG_ERR_ITER
        },
        {
            "name": "Garbage after Iter: Iter=100a",
            "args": ["2", "100a"],
            "input": "1,0\n2,0\n3,0",
            "rc": 1,
            "msg": MSG_ERR_ITER
        },

        # --- Group G: Advanced Data Corruption (Content Checks) ---
        {
            "name": "Inconsistent Dimensions (Jagged Array)",
            "args": ["2"],
            "input": "1.0,2.0,3.0\n1.0,2.0\n4.0,5.0,6.0",
            "rc": 1,
            "msg": MSG_ERR_GENERIC
        },
        {
            "name": "Non-numeric data in vector (1.0, 2a, 3.0)",
            "args": ["2"],
            "input": "1.0,2.0,3.0\n1.0,2a,3.0\n4.0,5.0,6.0",
            "rc": 1,
            "msg": MSG_ERR_GENERIC
        },
        {
            "name": "Double Comma (Missing Value)",
            "args": ["2"],
            "input": "1.0,2.0,3.0\n1.0,,3.0\n4.0,5.0,6.0",
            "rc": 1,
            "msg": MSG_ERR_GENERIC
        },
        {
            "name": "Trailing Comma",
            "args": ["2"],
            "input": "1.0,2.0,3.0\n1.0,2.0,3.0,\n4.0,5.0,6.0",
            "rc": 1,
            "msg": MSG_ERR_GENERIC
        },

        # --- Group H: Official Course Inputs/Outputs ---
        {
            "name": "Official Test 1 (K=3, iter=600)",
            "args": ["3", "600"],
            "input": in1,
            "rc": 0,
            "msg": out1
        },
        {
            "name": "Official Test 2 (K=7, default iter)",
            "args": ["7"],
            "input": in2,
            "rc": 0,
            "msg": out2
        },
        {
            "name": "Official Test 3 (K=15, iter=300)",
            "args": ["15", "300"],
            "input": in3,
            "rc": 0,
            "msg": out3
        }
    ]

    log("\n🚀 Starting The ULTIMATE K-means Tester 🚀\n", Colors.BOLD)

    for t in tests:
        total_tests += 1
        success = run_test_case(
            t["name"],
            t["input"],
            t["args"],
            t["rc"],
            t["msg"],
            check_py=True # Set to False if you only want to check C for now
        )
        if success:
            passed_tests += 1

    # --- Summary ---
    log("\n" + "="*40, Colors.BOLD)
    log(f"SUMMARY: {passed_tests}/{total_tests} Tests Passed", Colors.BOLD)

    if passed_tests == total_tests:
        log("🏆 PERFECTION! You are ready to submit.", Colors.OKGREEN)
    else:
        log("⚠️  Some tests failed. Check logs above.", Colors.WARNING)
    log("="*40, Colors.BOLD)