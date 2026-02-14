#!/usr/bin/env python3
"""
m3 Diagnostics — Module Test Harness
=====================================
Runs unit tests for every m3 module and reports each as
"Ok", "BROKEN", or "Skipped".

Usage:
    python m3_diag.py              # Run all tests
    python m3_diag.py -v           # Verbose mode (show individual test names)
    python m3_diag.py --module X   # Run only modules matching X (e.g., --module config)
"""

import sys
import os
import unittest
import io
import argparse
import time
import platform

# Ensure the project root is on sys.path so imports work
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Suppress noisy logs from libraries during testing
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')


# ── Module registry ──────────────────────────────────────────────────────
# Each entry maps a display name to the test module path
TEST_MODULES = [
    ("utils/config",             "tests.test_utils_config"),
    ("utils/device",             "tests.test_device"),
    ("utils/file_handler",       "tests.test_utils_file_handler"),
    ("utils/file_reader",        "tests.test_utils_file_reader"),
    ("core/project_manager",     "tests.test_project_manager"),
    ("core/llm_manager",         "tests.test_llm_manager"),
    ("core/db_manager",          "tests.test_db_manager"),
    ("core/plugin_manager",      "tests.test_plugin_manager"),
    ("core/ingestion_pipeline",  "tests.test_ingestion_pipeline"),
    ("core/vector_manager",      "tests.test_vector_manager"),
    ("core/analyze_manager",     "tests.test_analyze_manager"),
    ("core/session_manager",     "tests.test_session_manager"),
    ("cli/commands",             "tests.test_cli_commands"),
    ("plugins",                  "tests.test_plugins"),
]


# ANSI color codes
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
DIM    = "\033[2m"
BOLD   = "\033[1m"
RESET  = "\033[0m"


def run_module_tests(module_path, verbose=False):
    """
    Run all tests in a single test module.
    Returns (passed, tests_run, failures, errors, skipped, fail_details)
    """
    loader = unittest.TestLoader()

    try:
        suite = loader.loadTestsFromName(module_path)
    except Exception as e:
        return False, 0, 0, 1, 0, [("Import", str(e))]

    # Capture stdout/stderr from test code so the harness output stays clean
    stream = io.StringIO()
    captured_stderr = io.StringIO()
    old_stderr = sys.stderr
    old_stdout = sys.stdout
    if not verbose:
        sys.stderr = captured_stderr
        sys.stdout = io.StringIO()

    runner = unittest.TextTestRunner(
        stream=stream,
        verbosity=2 if verbose else 0,
    )

    try:
        result = runner.run(suite)
    finally:
        if not verbose:
            sys.stderr = old_stderr
            sys.stdout = old_stdout

    # Collect failure details
    fail_details = []
    for test, tb in result.failures + result.errors:
        tb_lines = tb.strip().split('\n')
        # Get the last meaningful lines (skip the boilerplate)
        summary_lines = []
        for line in tb_lines[-3:]:
            stripped = line.strip()
            if stripped:
                summary_lines.append(stripped)
        fail_details.append((str(test), '\n'.join(summary_lines)))

    return (
        result.wasSuccessful(),
        result.testsRun,
        len(result.failures),
        len(result.errors),
        len(result.skipped),
        fail_details,
    )


def main():
    parser = argparse.ArgumentParser(description="m3 Diagnostics — Module Test Harness")
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Show individual test names and results')
    parser.add_argument('--module', type=str, default=None,
                        help='Only run test modules matching this string')
    args = parser.parse_args()

    # ── Header ──
    print()
    print(f"{BOLD}{'=' * 64}")
    print(f"  m3 Diagnostics  —  Validate all m3 modules")
    print(f"{'=' * 64}{RESET}")

    # Platform info
    print(f"  {DIM}Platform: {platform.system()} {platform.machine()}{RESET}")
    try:
        from utils.device import detect_device, get_device_info, DEVICE_LABELS
        # Suppress click.echo output during detection
        old_stderr = sys.stderr
        sys.stderr = io.StringIO()
        device = detect_device()
        sys.stderr = old_stderr
        label = DEVICE_LABELS.get(device, device)
        info = get_device_info()
        torch_ver = info.get('torch_version', 'N/A')
        print(f"  {DIM}Compute:  {label}  (torch {torch_ver}){RESET}")
    except Exception:
        print(f"  {DIM}Compute:  unknown (device detection unavailable){RESET}")

    print()

    # Filter modules if requested
    modules_to_run = TEST_MODULES
    if args.module:
        modules_to_run = [
            (name, path) for name, path in TEST_MODULES
            if args.module.lower() in name.lower() or args.module.lower() in path.lower()
        ]
        if not modules_to_run:
            print(f"  No test modules matching '{args.module}' found.")
            sys.exit(1)

    # ── Run tests ──
    results = []
    total_tests = 0
    total_passed = 0
    total_failures = 0
    total_errors = 0
    total_skipped = 0
    start_time = time.time()

    # Find the longest display name for alignment
    max_name_len = max(len(name) for name, _ in modules_to_run)

    for display_name, module_path in modules_to_run:
        passed, tests_run, failures, errors, skipped, fail_details = run_module_tests(
            module_path, verbose=args.verbose
        )

        total_tests += tests_run
        total_failures += failures
        total_errors += errors
        total_skipped += skipped

        actually_ran = tests_run - skipped

        # Determine status
        if tests_run == 0 and errors > 0:
            # Import error: module couldn't load at all
            status = "BROKEN"
            status_color = RED
        elif skipped == tests_run and tests_run > 0:
            # All tests skipped (missing deps)
            status = "Skipped"
            status_color = YELLOW
        elif passed:
            status = "Ok"
            status_color = GREEN
            total_passed += actually_ran
        else:
            status = "BROKEN"
            status_color = RED

        # Build count info
        if skipped > 0 and skipped < tests_run:
            count_info = f"({actually_ran} ran, {skipped} skip)"
        elif skipped == tests_run and tests_run > 0:
            count_info = f"({skipped} skipped)"
        else:
            count_info = f"({tests_run} tests)"

        print(f"  {display_name:<{max_name_len}}  {count_info:<20} [{status_color}{status}{RESET}]")

        # Show failure details inline
        if fail_details and status == "BROKEN":
            for test_name, detail in fail_details:
                print(f"    {DIM}-> {test_name}{RESET}")
                for line in detail.split('\n'):
                    print(f"       {DIM}{line}{RESET}")

        results.append({
            'name': display_name,
            'passed': passed,
            'status': status,
            'tests_run': tests_run,
            'failures': failures,
            'errors': errors,
            'skipped': skipped,
        })

    elapsed = time.time() - start_time

    # ── Summary ──
    print()
    print(f"{BOLD}{'-' * 64}{RESET}")

    ok_count     = sum(1 for r in results if r['status'] == 'Ok')
    broken_count = sum(1 for r in results if r['status'] == 'BROKEN')
    skip_count   = sum(1 for r in results if r['status'] == 'Skipped')

    print(f"  Modules:  {len(results)} total, "
          f"{GREEN}{ok_count} Ok{RESET}, "
          f"{RED}{broken_count} Broken{RESET}, "
          f"{YELLOW}{skip_count} Skipped{RESET}")
    print(f"  Tests:    {total_tests} total, "
          f"{total_tests - total_failures - total_errors - total_skipped} passed, "
          f"{total_failures + total_errors} failed, "
          f"{total_skipped} skipped")
    print(f"  Time:     {elapsed:.2f}s")
    print()

    if broken_count == 0 and skip_count == 0:
        print(f"  {GREEN}{BOLD}✓ All modules passed.{RESET}")
    elif broken_count == 0:
        print(f"  {GREEN}{BOLD}✓ All runnable modules passed.{RESET}")
        skipped_names = [r['name'] for r in results if r['status'] == 'Skipped']
        print(f"  {YELLOW}⊘ Skipped (missing deps): {', '.join(skipped_names)}{RESET}")
    else:
        print(f"  {RED}{BOLD}✗ Some modules are broken!{RESET}")
        broken_names = [r['name'] for r in results if r['status'] == 'BROKEN']
        print(f"  {RED}  Broken: {', '.join(broken_names)}{RESET}")
        if skip_count > 0:
            skipped_names = [r['name'] for r in results if r['status'] == 'Skipped']
            print(f"  {YELLOW}  Skipped: {', '.join(skipped_names)}{RESET}")

    print()
    print(f"{BOLD}{'=' * 64}{RESET}")

    # Exit with non-zero if any module actually failed (skips are ok)
    sys.exit(1 if broken_count > 0 else 0)


if __name__ == '__main__':
    main()
