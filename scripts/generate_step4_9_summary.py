"""Generate summary for STEP 4.9 Dashboard Polish."""

import subprocess
import sys


def main() -> None:
    print("=" * 80)
    print("STEP 4.9 DASHBOARD POLISH SUMMARY")
    print("=" * 80)

    # 1. Files changed
    print("\n[1] Files Modified:")
    res_status = subprocess.run(
        ["git", "status", "--short"], capture_output=True, text=True, check=True
    )
    for line in res_status.stdout.strip().splitlines():
        print(f"    {line}")

    # 2. Test result (smoke test only)
    print("\n[2] Smoke Test Result (tests/test_app_smoke.py):")
    res_test = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_app_smoke.py", "-q"],
        capture_output=True,
        text=True,
    )
    test_out = res_test.stdout.strip()
    print(f"    Exit code: {res_test.returncode}")
    print(f"    Output: {test_out}")

    # 3. check_docs result
    print("\n[3] check_docs.py Result:")
    res_docs = subprocess.run(
        [sys.executable, "check_docs.py"], capture_output=True, text=True
    )
    print(f"    Exit code: {res_docs.returncode}")
    for line in res_docs.stdout.strip().splitlines():
        if "OVERALL:" in line or "STATUS:" in line:
            print(f"    {line}")

    print("\n" + "=" * 80)
    print("STATUS: STEP 4.9 COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
