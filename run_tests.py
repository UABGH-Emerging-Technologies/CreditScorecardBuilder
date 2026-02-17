#!/usr/bin/env python3
"""
Test runner script for CreditScore project.

This script provides convenient ways to run different categories of tests
and generates coverage reports.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print("=" * 60)

    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"❌ {description} failed with return code {result.returncode}")
        return False
    else:
        print(f"✅ {description} completed successfully")
        return True


def main():
    parser = argparse.ArgumentParser(description="Run CreditScore tests")
    parser.add_argument(
        "--category",
        choices=[
            "all",
            "regression",
            "unit",
            "integration",
            "config",
            "asa",
            "missing_data",
            "variable_selection",
        ],
        default="all",
        help="Category of tests to run",
    )
    parser.add_argument("--coverage", action="store_true", help="Run with coverage reporting")
    parser.add_argument("--slow", action="store_true", help="Include slow tests")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--quiet", "-q", action="store_true", help="Quiet output")
    parser.add_argument("--file", help="Run tests from specific file")
    parser.add_argument("--test", help="Run specific test function")

    args = parser.parse_args()

    # Base command
    cmd = ["python", "-m", "pytest"]

    # Add verbosity
    if args.verbose:
        cmd.append("-vv")
    elif args.quiet:
        cmd.append("-q")

    # Add coverage if requested
    if args.coverage:
        cmd.extend(
            [
                "--cov=CreditScore",
                "--cov=config",
                "--cov-report=html:htmlcov",
                "--cov-report=term-missing",
                "--cov-fail-under=70",
            ]
        )

    # Add specific file or test
    if args.file:
        cmd.append(args.file)
    elif args.test:
        cmd.extend(["-k", args.test])
    else:
        # Add category-specific markers
        if args.category != "all":
            cmd.extend(["-m", args.category])

    # Add slow tests if requested
    if not args.slow:
        if args.category == "all":
            cmd.extend(["-m", "not slow"])
        else:
            cmd.extend(["-m", f"{args.category} and not slow"])

    # Run the tests
    success = run_command(cmd, f"Running {args.category} tests")

    if args.coverage and success:
        print(f"\n📊 Coverage report generated in htmlcov/index.html")

    return 0 if success else 1


def run_regression_tests():
    """Run only the critical regression tests."""
    cmd = ["python", "-m", "pytest", "-m", "regression", "-v", "--tb=short"]
    return run_command(cmd, "Critical regression tests")


def run_quick_tests():
    """Run quick tests (excluding slow ones)."""
    cmd = ["python", "-m", "pytest", "-m", "not slow", "-x", "--tb=line"]  # Stop on first failure
    return run_command(cmd, "Quick test suite")


def run_full_test_suite():
    """Run the complete test suite with coverage."""
    cmd = [
        "python",
        "-m",
        "pytest",
        "--cov=CreditScore",
        "--cov=config",
        "--cov-report=html:htmlcov",
        "--cov-report=term-missing",
        "--cov-fail-under=60",
        "-v",
    ]
    return run_command(cmd, "Full test suite with coverage")


if __name__ == "__main__":
    # Check if specific quick commands are requested
    if len(sys.argv) == 2:
        if sys.argv[1] == "regression":
            sys.exit(0 if run_regression_tests() else 1)
        elif sys.argv[1] == "quick":
            sys.exit(0 if run_quick_tests() else 1)
        elif sys.argv[1] == "full":
            sys.exit(0 if run_full_test_suite() else 1)

    # Otherwise use argument parser
    sys.exit(main())
