# HD-Compute-Toolkit Security Scan Report

**Scan Date**: 2025-08-11T08:35:44.112580

## Executive Summary
- **Total Issues**: 2819
- **High Severity**: 960

⚠️ **Security issues detected.** Review and remediation required.

## Pattern Matches

### 🔴 security_pattern
**File**: `generation2_robustness_test.py`
**Line**: 91
**Category**: command_injection
**Match**: `os.system(`

### 🟡 security_pattern
**File**: `generation2_robustness_test.py`
**Line**: 89
**Category**: hardcoded_secrets
**Match**: `password = "hardcoded_password"`

### 🔴 security_pattern
**File**: `generation2_robustness_test.py`
**Line**: 92
**Category**: unsafe_eval
**Match**: `eval(`

### 🟢 security_pattern
**File**: `generation2_robustness_test.py`
**Line**: 107
**Category**: debug_code
**Match**: `print("✓ Hardcoded secret`

### 🟢 security_pattern
**File**: `production_deployment_test.py`
**Line**: 289
**Category**: debug_code
**Match**: `print("✓ Kubernetes secret`

### 🔴 security_pattern
**File**: `security_scan.py`
**Line**: 350
**Category**: unsafe_eval
**Match**: `__import__(`

### 🔴 security_pattern
**File**: `test_basic_import.py`
**Line**: 51
**Category**: unsafe_eval
**Match**: `__import__(`

### 🔴 security_pattern
**File**: `test_generation2_robustness.py`
**Line**: 146
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `test_generation2_robustness.py`
**Line**: 143
**Category**: unsafe_eval
**Match**: `__import__(`

### 🔴 security_pattern
**File**: `test_global_readiness.py`
**Line**: 403
**Category**: unsafe_eval
**Match**: `__import__(`

### 🔴 security_pattern
**File**: `test_quality_gates.py`
**Line**: 148
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `test_quality_gates.py`
**Line**: 145
**Category**: unsafe_eval
**Match**: `__import__(`

### 🔴 security_pattern
**File**: `test_quality_gates.py`
**Line**: 148
**Category**: unsafe_eval
**Match**: `__import__(`

### 🟡 security_pattern
**File**: `test_runner.py`
**Line**: 128
**Category**: hardcoded_secrets
**Match**: `password="test_password"`

### 🟡 security_pattern
**File**: `tests/test_quantum_task_planning.py`
**Line**: 426
**Category**: hardcoded_secrets
**Match**: `password="test_password"`

### 🟡 security_pattern
**File**: `tests/test_quantum_task_planning.py`
**Line**: 449
**Category**: hardcoded_secrets
**Match**: `password="valid_password"`

### 🟡 security_pattern
**File**: `tests/test_quantum_task_planning.py`
**Line**: 858
**Category**: hardcoded_secrets
**Match**: `password="secure_password"`

### 🟡 security_pattern
**File**: `hd_compute/applications/task_planning.py`
**Line**: 307
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `hd_compute/applications/task_planning.py`
**Line**: 740
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `hd_compute/cache/cache_manager.py`
**Line**: 44
**Category**: weak_crypto
**Match**: `hashlib.md5(`

### 🟡 security_pattern
**File**: `hd_compute/cache/cache_manager.py`
**Line**: 251
**Category**: weak_crypto
**Match**: `hashlib.md5(`

### 🔴 security_pattern
**File**: `hd_compute/cache/cache_manager.py`
**Line**: 69
**Category**: unsafe_pickle
**Match**: `pickle.load(`

### 🟡 security_pattern
**File**: `hd_compute/compliance/data_privacy.py`
**Line**: 121
**Category**: weak_crypto
**Match**: `hashlib.md5(`

### 🟡 security_pattern
**File**: `hd_compute/database/repository.py`
**Line**: 232
**Category**: weak_crypto
**Match**: `hashlib.md5(`

### 🔴 security_pattern
**File**: `hd_compute/database/repository.py`
**Line**: 284
**Category**: unsafe_pickle
**Match**: `pickle.loads(`

### 🟡 security_pattern
**File**: `hd_compute/performance/optimization.py`
**Line**: 116
**Category**: weak_crypto
**Match**: `hashlib.md5(`

### 🟡 security_pattern
**File**: `hd_compute/pure_python/hdc_python.py`
**Line**: 94
**Category**: weak_crypto
**Match**: `random.Random(`

### 🟡 security_pattern
**File**: `hd_compute/research/adaptive_memory.py`
**Line**: 258
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `hd_compute/scalable_backends/scalable_python.py`
**Line**: 85
**Category**: weak_crypto
**Match**: `hashlib.md5(`

### 🟡 security_pattern
**File**: `hd_compute/scalable_backends/scalable_python.py`
**Line**: 321
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `hd_compute/scalable_backends/scalable_python.py`
**Line**: 322
**Category**: weak_crypto
**Match**: `random.random(`

### 🔴 security_pattern
**File**: `hd_compute/security/secure_imports.py`
**Line**: 3
**Category**: unsafe_eval
**Match**: `__import__(`

### 🔴 security_pattern
**File**: `hd_compute/security/secure_imports.py`
**Line**: 23
**Category**: unsafe_eval
**Match**: `__import__(`

### 🔴 security_pattern
**File**: `hd_compute/security/secure_imports.py`
**Line**: 100
**Category**: unsafe_eval
**Match**: `__import__(`

### 🔴 security_pattern
**File**: `hd_compute/security/secure_imports.py`
**Line**: 230
**Category**: unsafe_eval
**Match**: `__import__(`

### 🔴 security_pattern
**File**: `hd_compute/security/secure_serialization.py`
**Line**: 23
**Category**: unsafe_pickle
**Match**: `pickle.load(`

### 🔴 security_pattern
**File**: `hd_compute/security/secure_serialization.py`
**Line**: 184
**Category**: unsafe_pickle
**Match**: `pickle.load(`

### 🟡 security_pattern
**File**: `hd_compute/security/security_config.py`
**Line**: 414
**Category**: weak_crypto
**Match**: `hashlib.md5(`

### 🟡 security_pattern
**File**: `hd_compute/security/security_config.py`
**Line**: 431
**Category**: weak_crypto
**Match**: `hashlib.md5(`

### 🔴 security_pattern
**File**: `hd_compute/security/security_scanner.py`
**Line**: 36
**Category**: command_injection
**Match**: `os.system(`

### 🔴 security_pattern
**File**: `hd_compute/security/security_scanner.py`
**Line**: 37
**Category**: command_injection
**Match**: `os.popen(`

### 🔴 security_pattern
**File**: `hd_compute/security/security_scanner.py`
**Line**: 29
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `hd_compute/security/security_scanner.py`
**Line**: 118
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `hd_compute/security/security_scanner.py`
**Line**: 30
**Category**: unsafe_eval
**Match**: `exec(`

### 🔴 security_pattern
**File**: `hd_compute/security/security_scanner.py`
**Line**: 118
**Category**: unsafe_eval
**Match**: `exec(`

### 🔴 security_pattern
**File**: `hd_compute/security/security_scanner.py`
**Line**: 31
**Category**: unsafe_pickle
**Match**: `pickle.load(`

### 🔴 security_pattern
**File**: `hd_compute/utils/environment.py`
**Line**: 97
**Category**: unsafe_eval
**Match**: `__import__(`

### 🟡 security_pattern
**File**: `hd_compute/validation/quality_assurance.py`
**Line**: 475
**Category**: weak_crypto
**Match**: `hashlib.md5(`

### 🔴 security_pattern
**File**: `hd_compute/validation/quality_assurance.py`
**Line**: 403
**Category**: unsafe_pickle
**Match**: `pickle.load(`

### 🔴 security_pattern
**File**: `tests/security/test_security_fixes.py`
**Line**: 242
**Category**: command_injection
**Match**: `os.system(`

### 🔴 security_pattern
**File**: `tests/security/test_security_fixes.py`
**Line**: 188
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `tests/security/test_security_fixes.py`
**Line**: 236
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `tests/security/test_security_fixes.py`
**Line**: 243
**Category**: unsafe_eval
**Match**: `exec(`

### 🔴 security_pattern
**File**: `tests/security/test_security_fixes.py`
**Line**: 237
**Category**: unsafe_eval
**Match**: `__import__(`

### 🔴 security_pattern
**File**: `tests/security/test_security_fixes.py`
**Line**: 244
**Category**: unsafe_pickle
**Match**: `pickle.loads(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/typing_extensions.py`
**Line**: 1444
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/typing_extensions.py`
**Line**: 3967
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/typing_extensions.py`
**Line**: 3972
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/typing_extensions.py`
**Line**: 4019
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/typing_extensions.py`
**Line**: 4031
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/typing_extensions.py`
**Line**: 4036
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/typing_extensions.py`
**Line**: 4064
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/typing_extensions.py`
**Line**: 1444
**Category**: unsafe_eval
**Match**: `exec(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/mpmath/ctx_fp.py`
**Line**: 239
**Category**: weak_crypto
**Match**: `random.random(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/mpmath/identification.py`
**Line**: 690
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/mpmath/identification.py`
**Line**: 777
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/__init__.py`
**Line**: 47
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/abc.py`
**Line**: 96
**Category**: unsafe_eval
**Match**: `exec(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/_distutils_hack/override.py`
**Line**: 1
**Category**: unsafe_eval
**Match**: `__import__(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/pkg_resources/__init__.py`
**Line**: 1738
**Category**: unsafe_eval
**Match**: `exec(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/pkg_resources/__init__.py`
**Line**: 1749
**Category**: unsafe_eval
**Match**: `exec(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/pkg_resources/__init__.py`
**Line**: 421
**Category**: unsafe_eval
**Match**: `__import__(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/pkg_resources/__init__.py`
**Line**: 2560
**Category**: unsafe_eval
**Match**: `__import__(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/pkg_resources/__init__.py`
**Line**: 2783
**Category**: unsafe_eval
**Match**: `__import__(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/setuptools/build_meta.py`
**Line**: 317
**Category**: unsafe_eval
**Match**: `exec(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/setuptools/extension.py`
**Line**: 23
**Category**: unsafe_eval
**Match**: `__import__(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/setuptools/launch.py`
**Line**: 32
**Category**: unsafe_eval
**Match**: `exec(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/setuptools/namespaces.py`
**Line**: 50
**Category**: unsafe_eval
**Match**: `__import__(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/setuptools/namespaces.py`
**Line**: 51
**Category**: unsafe_eval
**Match**: `__import__(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/setuptools/wheel.py`
**Line**: 190
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/setuptools/wheel.py`
**Line**: 211
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/setuptools/wheel.py`
**Line**: 34
**Category**: unsafe_eval
**Match**: `__import__(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/opt_einsum/testing.py`
**Line**: 75
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/fsspec/spec.py`
**Line**: 1640
**Category**: hardcoded_secrets
**Match**: `password='password'`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/pip/_vendor/__init__.py`
**Line**: 33
**Category**: unsafe_eval
**Match**: `__import__(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/pip/_vendor/six.py`
**Line**: 735
**Category**: unsafe_eval
**Match**: `exec(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/pip/_vendor/six.py`
**Line**: 87
**Category**: unsafe_eval
**Match**: `__import__(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/pip/_vendor/typing_extensions.py`
**Line**: 1251
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/pip/_vendor/typing_extensions.py`
**Line**: 1251
**Category**: unsafe_eval
**Match**: `exec(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/pip/_internal/commands/debug.py`
**Line**: 57
**Category**: unsafe_eval
**Match**: `__import__(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/pip/_internal/locations/_distutils.py`
**Line**: 13
**Category**: unsafe_eval
**Match**: `__import__(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/pip/_internal/utils/setuptools_build.py`
**Line**: 10
**Category**: unsafe_eval
**Match**: `exec(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/pip/_internal/utils/setuptools_build.py`
**Line**: 43
**Category**: unsafe_eval
**Match**: `exec(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/pip/_internal/vcs/git.py`
**Line**: 452
**Category**: path_traversal
**Match**: `os.path.join(git_dir, "..`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/pip/_vendor/distlib/database.py`
**Line**: 1032
**Category**: weak_crypto
**Match**: `hashlib.md5(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/pip/_vendor/distlib/index.py`
**Line**: 269
**Category**: weak_crypto
**Match**: `hashlib.md5(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/pip/_vendor/distlib/resources.py`
**Line**: 323
**Category**: unsafe_eval
**Match**: `__import__(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/pip/_vendor/distlib/util.py`
**Line**: 695
**Category**: unsafe_eval
**Match**: `__import__(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/pip/_vendor/pkg_resources/__init__.py`
**Line**: 1561
**Category**: unsafe_eval
**Match**: `exec(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/pip/_vendor/pkg_resources/__init__.py`
**Line**: 1572
**Category**: unsafe_eval
**Match**: `exec(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/pip/_vendor/pkg_resources/__init__.py`
**Line**: 88
**Category**: unsafe_eval
**Match**: `__import__(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/pip/_vendor/pkg_resources/__init__.py`
**Line**: 89
**Category**: unsafe_eval
**Match**: `__import__(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/pip/_vendor/pkg_resources/__init__.py`
**Line**: 90
**Category**: unsafe_eval
**Match**: `__import__(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/pip/_vendor/pkg_resources/__init__.py`
**Line**: 91
**Category**: unsafe_eval
**Match**: `__import__(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/pip/_vendor/pkg_resources/__init__.py`
**Line**: 92
**Category**: unsafe_eval
**Match**: `__import__(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/pip/_vendor/pkg_resources/__init__.py`
**Line**: 404
**Category**: unsafe_eval
**Match**: `__import__(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/pip/_vendor/pkg_resources/__init__.py`
**Line**: 2352
**Category**: unsafe_eval
**Match**: `__import__(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/pip/_vendor/pkg_resources/__init__.py`
**Line**: 2524
**Category**: unsafe_eval
**Match**: `__import__(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/pip/_vendor/pygments/sphinxext.py`
**Line**: 155
**Category**: unsafe_eval
**Match**: `__import__(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/pip/_vendor/pygments/sphinxext.py`
**Line**: 191
**Category**: unsafe_eval
**Match**: `__import__(`

### 🟢 security_pattern
**File**: `venv/lib/python3.12/site-packages/pip/_vendor/pyparsing/core.py`
**Line**: 898
**Category**: debug_code
**Match**: `print("Matched", self, "->", ret_token`

### 🟢 security_pattern
**File**: `venv/lib/python3.12/site-packages/pip/_vendor/pyparsing/core.py`
**Line**: 1219
**Category**: debug_code
**Match**: `print(' '*start + token`

### 🟢 security_pattern
**File**: `venv/lib/python3.12/site-packages/pip/_vendor/pyparsing/core.py`
**Line**: 1260
**Category**: debug_code
**Match**: `print(
                                {
                                    "tokens": token`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/pip/_vendor/pyparsing/results.py`
**Line**: 57
**Category**: unsafe_eval
**Match**: `eval(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/pip/_vendor/requests/auth.py`
**Line**: 148
**Category**: weak_crypto
**Match**: `hashlib.md5(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/pip/_vendor/requests/auth.py`
**Line**: 156
**Category**: weak_crypto
**Match**: `hashlib.sha1(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/pip/_vendor/requests/auth.py`
**Line**: 205
**Category**: weak_crypto
**Match**: `hashlib.sha1(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/pip/_vendor/requests/packages.py`
**Line**: 8
**Category**: unsafe_eval
**Match**: `__import__(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/pip/_vendor/rich/live.py`
**Line**: 355
**Category**: weak_crypto
**Match**: `random.random(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/pip/_vendor/rich/pager.py`
**Line**: 21
**Category**: unsafe_eval
**Match**: `__import__(`

### 🟢 security_pattern
**File**: `venv/lib/python3.12/site-packages/pip/_vendor/rich/prompt.py`
**Line**: 369
**Category**: debug_code
**Match**: `print("[prompt.invalid]password`

### 🟢 security_pattern
**File**: `venv/lib/python3.12/site-packages/pip/_vendor/rich/prompt.py`
**Line**: 370
**Category**: debug_code
**Match**: `print(f"password={password`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/pip/_vendor/tenacity/wait.py`
**Line**: 72
**Category**: weak_crypto
**Match**: `random.random(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/pip/_vendor/pygments/formatters/__init__.py`
**Line**: 91
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/pip/_vendor/pygments/formatters/__init__.py`
**Line**: 103
**Category**: unsafe_eval
**Match**: `exec(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/pip/_vendor/pygments/formatters/__init__.py`
**Line**: 38
**Category**: unsafe_eval
**Match**: `__import__(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/pip/_vendor/pygments/lexers/__init__.py`
**Line**: 153
**Category**: unsafe_eval
**Match**: `exec(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/pip/_vendor/pygments/lexers/__init__.py`
**Line**: 44
**Category**: unsafe_eval
**Match**: `__import__(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/pip/_vendor/pygments/styles/__init__.py`
**Line**: 89
**Category**: unsafe_eval
**Match**: `__import__(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/pip/_vendor/urllib3/packages/six.py`
**Line**: 787
**Category**: unsafe_eval
**Match**: `exec (`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/pip/_vendor/urllib3/packages/six.py`
**Line**: 87
**Category**: unsafe_eval
**Match**: `__import__(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/mpmath/libmp/backend.py`
**Line**: 47
**Category**: unsafe_eval
**Match**: `exec(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/mpmath/tests/extratest_gamma.py`
**Line**: 56
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/mpmath/tests/extratest_gamma.py`
**Line**: 64
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/mpmath/tests/runtests.py`
**Line**: 109
**Category**: unsafe_eval
**Match**: `__import__(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/mpmath/tests/test_convert.py`
**Line**: 82
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/mpmath/tests/test_convert.py`
**Line**: 88
**Category**: unsafe_eval
**Match**: `eval(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/mpmath/tests/test_convert.py`
**Line**: 87
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/mpmath/tests/test_elliptic.py`
**Line**: 149
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/mpmath/tests/test_elliptic.py`
**Line**: 226
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/mpmath/tests/test_elliptic.py`
**Line**: 229
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/mpmath/tests/test_elliptic.py`
**Line**: 240
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/mpmath/tests/test_elliptic.py`
**Line**: 353
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/mpmath/tests/test_elliptic.py`
**Line**: 353
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/mpmath/tests/test_elliptic.py`
**Line**: 397
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/mpmath/tests/test_elliptic.py`
**Line**: 435
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/mpmath/tests/test_elliptic.py`
**Line**: 472
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/mpmath/tests/test_elliptic.py`
**Line**: 495
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/mpmath/tests/test_elliptic.py`
**Line**: 497
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/mpmath/tests/test_elliptic.py`
**Line**: 510
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/mpmath/tests/test_elliptic.py`
**Line**: 513
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/mpmath/tests/test_elliptic.py`
**Line**: 522
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/mpmath/tests/test_elliptic.py`
**Line**: 525
**Category**: weak_crypto
**Match**: `random.random(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/mpmath/tests/test_fp.py`
**Line**: 19
**Category**: unsafe_eval
**Match**: `eval(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/mpmath/tests/test_functions.py`
**Line**: 562
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/mpmath/tests/test_functions.py`
**Line**: 562
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/mpmath/tests/test_functions.py`
**Line**: 585
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/mpmath/tests/test_functions.py`
**Line**: 585
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/mpmath/tests/test_functions.py`
**Line**: 588
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/mpmath/tests/test_functions.py`
**Line**: 588
**Category**: weak_crypto
**Match**: `random.random(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/mpmath/tests/test_linalg.py`
**Line**: 181
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/mpmath/tests/test_matrices.py`
**Line**: 25
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/mpmath/tests/test_matrices.py`
**Line**: 27
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/mpmath/tests/test_matrices.py`
**Line**: 28
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/mpmath/tests/test_matrices.py`
**Line**: 55
**Category**: unsafe_eval
**Match**: `exec(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/mpmath/tests/test_matrices.py`
**Line**: 57
**Category**: unsafe_eval
**Match**: `exec(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/mpmath/tests/test_matrices.py`
**Line**: 63
**Category**: unsafe_eval
**Match**: `exec(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/mpmath/tests/test_pickle.py`
**Line**: 15
**Category**: unsafe_pickle
**Match**: `pickle.load(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/assumptions/assume.py`
**Line**: 156
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/assumptions/assume.py`
**Line**: 340
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/assumptions/assume.py`
**Line**: 426
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/assumptions/wrapper.py`
**Line**: 90
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/assumptions/wrapper.py`
**Line**: 97
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/benchmarks/bench_meijerint.py`
**Line**: 249
**Category**: unsafe_eval
**Match**: `exec(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/codegen/algorithms.py`
**Line**: 154
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/codegen/algorithms.py`
**Line**: 153
**Category**: unsafe_eval
**Match**: `exec(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/codegen/ast.py`
**Line**: 105
**Category**: unsafe_eval
**Match**: `exec(`

### 🟢 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/codegen/ast.py`
**Line**: 1724
**Category**: debug_code
**Match**: `Print(Token`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/codegen/cfunctions.py`
**Line**: 70
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/codegen/cfunctions.py`
**Line**: 71
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/codegen/cfunctions.py`
**Line**: 138
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/codegen/cfunctions.py`
**Line**: 142
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/codegen/cfunctions.py`
**Line**: 215
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/codegen/cfunctions.py`
**Line**: 264
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/codegen/cfunctions.py`
**Line**: 266
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/codegen/cfunctions.py`
**Line**: 368
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/codegen/cfunctions.py`
**Line**: 370
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/codegen/cfunctions.py`
**Line**: 539
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/codegen/cfunctions.py`
**Line**: 552
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/combinatorics/schur_number.py`
**Line**: 40
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/core/function.py`
**Line**: 309
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/core/function.py`
**Line**: 340
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/core/function.py`
**Line**: 347
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/core/function.py`
**Line**: 352
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/core/function.py`
**Line**: 357
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/core/function.py`
**Line**: 1653
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/core/mod.py`
**Line**: 56
**Category**: unsafe_eval
**Match**: `eval(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/core/random.py`
**Line**: 29
**Category**: weak_crypto
**Match**: `random.Random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/core/random.py`
**Line**: 40
**Category**: weak_crypto
**Match**: `random.Random(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/core/symbol.py`
**Line**: 479
**Category**: unsafe_eval
**Match**: `eval(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/core/symbol.py`
**Line**: 487
**Category**: weak_crypto
**Match**: `random.Random(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/core/sympify.py`
**Line**: 209
**Category**: unsafe_eval
**Match**: `exec(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/core/sympify.py`
**Line**: 219
**Category**: unsafe_eval
**Match**: `exec(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/external/importtools.py`
**Line**: 21
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/external/importtools.py`
**Line**: 87
**Category**: unsafe_eval
**Match**: `__import__(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/external/importtools.py`
**Line**: 88
**Category**: unsafe_eval
**Match**: `__import__(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/external/importtools.py`
**Line**: 90
**Category**: unsafe_eval
**Match**: `__import__(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/external/importtools.py`
**Line**: 116
**Category**: unsafe_eval
**Match**: `__import__(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/external/importtools.py`
**Line**: 145
**Category**: unsafe_eval
**Match**: `__import__(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/external/importtools.py`
**Line**: 154
**Category**: unsafe_eval
**Match**: `__import__(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/geometry/ellipse.py`
**Line**: 1125
**Category**: weak_crypto
**Match**: `random.Random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/geometry/line.py`
**Line**: 1036
**Category**: weak_crypto
**Match**: `random.Random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/geometry/plane.py`
**Line**: 811
**Category**: weak_crypto
**Match**: `random.Random(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/manualintegrate.py`
**Line**: 78
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/manualintegrate.py`
**Line**: 96
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/manualintegrate.py`
**Line**: 107
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/manualintegrate.py`
**Line**: 108
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/manualintegrate.py`
**Line**: 120
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/manualintegrate.py`
**Line**: 133
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/manualintegrate.py`
**Line**: 144
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/manualintegrate.py`
**Line**: 145
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/manualintegrate.py`
**Line**: 158
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/manualintegrate.py`
**Line**: 159
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/manualintegrate.py`
**Line**: 179
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/manualintegrate.py`
**Line**: 181
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/manualintegrate.py`
**Line**: 182
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/manualintegrate.py`
**Line**: 195
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/manualintegrate.py`
**Line**: 199
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/manualintegrate.py`
**Line**: 215
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/manualintegrate.py`
**Line**: 222
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/manualintegrate.py`
**Line**: 229
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/manualintegrate.py`
**Line**: 236
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/manualintegrate.py`
**Line**: 243
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/manualintegrate.py`
**Line**: 250
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/manualintegrate.py`
**Line**: 262
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/manualintegrate.py`
**Line**: 269
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/manualintegrate.py`
**Line**: 279
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/manualintegrate.py`
**Line**: 288
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/manualintegrate.py`
**Line**: 295
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/manualintegrate.py`
**Line**: 302
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/manualintegrate.py`
**Line**: 313
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/manualintegrate.py`
**Line**: 326
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/manualintegrate.py`
**Line**: 349
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/manualintegrate.py`
**Line**: 361
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/manualintegrate.py`
**Line**: 363
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/manualintegrate.py`
**Line**: 371
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/manualintegrate.py`
**Line**: 372
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/manualintegrate.py`
**Line**: 381
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/manualintegrate.py`
**Line**: 391
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/manualintegrate.py`
**Line**: 407
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/manualintegrate.py`
**Line**: 408
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/manualintegrate.py`
**Line**: 424
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/manualintegrate.py`
**Line**: 425
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/manualintegrate.py`
**Line**: 438
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/manualintegrate.py`
**Line**: 443
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/manualintegrate.py`
**Line**: 456
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/manualintegrate.py`
**Line**: 471
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/manualintegrate.py`
**Line**: 507
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/manualintegrate.py`
**Line**: 521
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/manualintegrate.py`
**Line**: 536
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/manualintegrate.py`
**Line**: 548
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/manualintegrate.py`
**Line**: 558
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/manualintegrate.py`
**Line**: 568
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/manualintegrate.py`
**Line**: 577
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/manualintegrate.py`
**Line**: 584
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/manualintegrate.py`
**Line**: 591
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/manualintegrate.py`
**Line**: 600
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/manualintegrate.py`
**Line**: 612
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/manualintegrate.py`
**Line**: 619
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/manualintegrate.py`
**Line**: 626
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/manualintegrate.py`
**Line**: 633
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/manualintegrate.py`
**Line**: 640
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/manualintegrate.py`
**Line**: 647
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/manualintegrate.py`
**Line**: 658
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/manualintegrate.py`
**Line**: 676
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/manualintegrate.py`
**Line**: 689
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/manualintegrate.py`
**Line**: 701
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/manualintegrate.py`
**Line**: 710
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/manualintegrate.py`
**Line**: 720
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/manualintegrate.py`
**Line**: 729
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/manualintegrate.py`
**Line**: 1277
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/manualintegrate.py`
**Line**: 1299
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/manualintegrate.py`
**Line**: 2164
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/meijerint.py`
**Line**: 113
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/prde.py`
**Line**: 138
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/prde.py`
**Line**: 138
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/prde.py`
**Line**: 138
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/rde.py`
**Line**: 233
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/rde.py`
**Line**: 233
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/rde.py`
**Line**: 233
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/rde.py`
**Line**: 245
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/rde.py`
**Line**: 245
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/rde.py`
**Line**: 245
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/rde.py`
**Line**: 246
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/rde.py`
**Line**: 246
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/rde.py`
**Line**: 246
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/risch.py`
**Line**: 630
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/risch.py`
**Line**: 1197
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/interactive/session.py`
**Line**: 105
**Category**: unsafe_eval
**Match**: `exec(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/interactive/session.py`
**Line**: 107
**Category**: unsafe_eval
**Match**: `exec(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/logic/boolalg.py`
**Line**: 909
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/logic/boolalg.py`
**Line**: 1133
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/logic/boolalg.py`
**Line**: 1167
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/logic/boolalg.py`
**Line**: 1198
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/logic/boolalg.py`
**Line**: 1252
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/logic/boolalg.py`
**Line**: 1439
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/logic/boolalg.py`
**Line**: 1445
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/logic/boolalg.py`
**Line**: 1508
**Category**: unsafe_eval
**Match**: `eval(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/matrices/dense.py`
**Line**: 1021
**Category**: weak_crypto
**Match**: `random.Random(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/parsing/ast_parser.py`
**Line**: 79
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/parsing/ast_parser.py`
**Line**: 72
**Category**: unsafe_eval
**Match**: `exec(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/parsing/sympy_parser.py`
**Line**: 905
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/parsing/sympy_parser.py`
**Line**: 1052
**Category**: unsafe_eval
**Match**: `exec(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/physics/secondquant.py`
**Line**: 114
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/physics/secondquant.py`
**Line**: 121
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/physics/secondquant.py`
**Line**: 137
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/physics/secondquant.py`
**Line**: 1440
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/physics/secondquant.py`
**Line**: 1443
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/physics/secondquant.py`
**Line**: 1601
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/physics/secondquant.py`
**Line**: 1699
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/physics/secondquant.py`
**Line**: 1710
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/plotting/experimental_lambdify.py`
**Line**: 268
**Category**: unsafe_eval
**Match**: `exec(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/plotting/experimental_lambdify.py`
**Line**: 249
**Category**: unsafe_eval
**Match**: `__import__(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/plotting/experimental_lambdify.py`
**Line**: 251
**Category**: unsafe_eval
**Match**: `__import__(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/plotting/experimental_lambdify.py`
**Line**: 254
**Category**: unsafe_eval
**Match**: `__import__(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/plotting/experimental_lambdify.py`
**Line**: 259
**Category**: unsafe_eval
**Match**: `__import__(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/plotting/experimental_lambdify.py`
**Line**: 261
**Category**: unsafe_eval
**Match**: `__import__(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/polys/monomials.py`
**Line**: 401
**Category**: unsafe_eval
**Match**: `exec(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/polys/polyclasses.py`
**Line**: 731
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/polys/polyconfig.py`
**Line**: 61
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/polys/polyquinticconst.py`
**Line**: 159
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/polys/polyquinticconst.py`
**Line**: 167
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/polys/polyquinticconst.py`
**Line**: 168
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/polys/polyquinticconst.py`
**Line**: 169
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/polys/polyquinticconst.py`
**Line**: 170
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/polys/polyquinticconst.py`
**Line**: 176
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/polys/polyquinticconst.py`
**Line**: 182
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/polys/polyroots.py`
**Line**: 797
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/polys/polyroots.py`
**Line**: 1037
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/polys/polytools.py`
**Line**: 151
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/polys/polytools.py`
**Line**: 607
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/polys/polytools.py`
**Line**: 2419
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/polys/polytools.py`
**Line**: 2429
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/polys/polytools.py`
**Line**: 2432
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/polys/polytools.py`
**Line**: 2437
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/polys/polytools.py`
**Line**: 2439
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/polys/polytools.py`
**Line**: 2441
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/polys/polytools.py`
**Line**: 2444
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/polys/polytools.py`
**Line**: 2457
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/polys/polytools.py`
**Line**: 2467
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/polys/polytools.py`
**Line**: 2479
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/polys/polytools.py`
**Line**: 2490
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/polys/polytools.py`
**Line**: 2514
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/polys/ring_series.py`
**Line**: 2011
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/polys/rootoftools.py`
**Line**: 310
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/printing/dot.py`
**Line**: 43
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/printing/dot.py`
**Line**: 54
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/printing/python.py`
**Line**: 44
**Category**: unsafe_eval
**Match**: `exec(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/printing/repr.py`
**Line**: 5
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/series/formal.py`
**Line**: 963
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/simplify/gammasimp.py`
**Line**: 465
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/solvers/polysys.py`
**Line**: 457
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/testing/runtests.py`
**Line**: 1243
**Category**: unsafe_eval
**Match**: `exec(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/testing/runtests.py`
**Line**: 1487
**Category**: unsafe_eval
**Match**: `exec(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/testing/runtests.py`
**Line**: 1145
**Category**: weak_crypto
**Match**: `random.random(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/utilities/autowrap.py`
**Line**: 165
**Category**: unsafe_eval
**Match**: `__import__(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/utilities/autowrap.py`
**Line**: 873
**Category**: unsafe_eval
**Match**: `__import__(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/utilities/lambdify.py`
**Line**: 163
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/utilities/lambdify.py`
**Line**: 170
**Category**: unsafe_eval
**Match**: `exec(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/utilities/lambdify.py`
**Line**: 608
**Category**: unsafe_eval
**Match**: `exec(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/utilities/lambdify.py`
**Line**: 620
**Category**: unsafe_eval
**Match**: `exec(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/utilities/lambdify.py`
**Line**: 903
**Category**: unsafe_eval
**Match**: `exec(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/utilities/lambdify.py`
**Line**: 909
**Category**: unsafe_eval
**Match**: `exec(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/utilities/lambdify.py`
**Line**: 920
**Category**: unsafe_eval
**Match**: `exec(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/utilities/matchpy_connector.py`
**Line**: 270
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/utilities/matchpy_connector.py`
**Line**: 269
**Category**: unsafe_eval
**Match**: `exec(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/utilities/source.py`
**Line**: 17
**Category**: unsafe_eval
**Match**: `__import__(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/assumptions/relation/binrel.py`
**Line**: 113
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/assumptions/relation/binrel.py`
**Line**: 200
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/assumptions/relation/binrel.py`
**Line**: 206
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/assumptions/relation/equality.py`
**Line**: 62
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/assumptions/relation/equality.py`
**Line**: 106
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/assumptions/relation/equality.py`
**Line**: 154
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/assumptions/relation/equality.py`
**Line**: 202
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/assumptions/relation/equality.py`
**Line**: 250
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/assumptions/relation/equality.py`
**Line**: 298
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/codegen/tests/test_algorithms.py`
**Line**: 82
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/codegen/tests/test_algorithms.py`
**Line**: 81
**Category**: unsafe_eval
**Match**: `exec(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/codegen/tests/test_algorithms.py`
**Line**: 170
**Category**: unsafe_eval
**Match**: `exec(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/core/tests/test_args.py`
**Line**: 64
**Category**: unsafe_eval
**Match**: `__import__(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/core/tests/test_basic.py`
**Line**: 325
**Category**: unsafe_eval
**Match**: `exec(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/core/tests/test_function.py`
**Line**: 140
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/core/tests/test_function.py`
**Line**: 149
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/core/tests/test_function.py`
**Line**: 158
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/core/tests/test_function.py`
**Line**: 521
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/core/tests/test_function.py`
**Line**: 1431
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/core/tests/test_function.py`
**Line**: 1440
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/core/tests/test_function.py`
**Line**: 1453
**Category**: unsafe_eval
**Match**: `eval(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/core/tests/test_random.py`
**Line**: 10
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/core/tests/test_random.py`
**Line**: 13
**Category**: weak_crypto
**Match**: `random.random(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/core/tests/test_singleton.py`
**Line**: 57
**Category**: unsafe_eval
**Match**: `exec(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/core/tests/test_subs.py`
**Line**: 107
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/core/tests/test_sympify.py`
**Line**: 549
**Category**: unsafe_eval
**Match**: `exec(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/core/tests/test_var.py`
**Line**: 7
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/core/tests/test_var.py`
**Line**: 10
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/core/tests/test_var.py`
**Line**: 17
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/core/tests/test_var.py`
**Line**: 30
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/core/tests/test_var.py`
**Line**: 31
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/core/tests/test_var.py`
**Line**: 39
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/core/tests/test_var.py`
**Line**: 40
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/core/tests/test_var.py`
**Line**: 41
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/core/tests/test_var.py`
**Line**: 49
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/core/tests/test_var.py`
**Line**: 55
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/core/tests/test_var.py`
**Line**: 59
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/external/tests/test_codegen.py`
**Line**: 122
**Category**: command_injection
**Match**: `subprocess.call(command, stdout=null, shell=True`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/external/tests/test_pythonmpq.py`
**Line**: 97
**Category**: unsafe_pickle
**Match**: `pickle.loads(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/functions/combinatorial/factorials.py`
**Line**: 139
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/functions/combinatorial/factorials.py`
**Line**: 341
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/functions/combinatorial/factorials.py`
**Line**: 422
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/functions/combinatorial/factorials.py`
**Line**: 566
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/functions/combinatorial/factorials.py`
**Line**: 728
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/functions/combinatorial/factorials.py`
**Line**: 969
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/functions/combinatorial/numbers.py`
**Line**: 241
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/functions/combinatorial/numbers.py`
**Line**: 310
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/functions/combinatorial/numbers.py`
**Line**: 379
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/functions/combinatorial/numbers.py`
**Line**: 544
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/functions/combinatorial/numbers.py`
**Line**: 724
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/functions/combinatorial/numbers.py`
**Line**: 898
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/functions/combinatorial/numbers.py`
**Line**: 1120
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/functions/combinatorial/numbers.py`
**Line**: 1285
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/functions/combinatorial/numbers.py`
**Line**: 1402
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/functions/combinatorial/numbers.py`
**Line**: 1549
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/functions/combinatorial/numbers.py`
**Line**: 1640
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/functions/combinatorial/numbers.py`
**Line**: 1707
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/functions/combinatorial/numbers.py`
**Line**: 1786
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/functions/combinatorial/numbers.py`
**Line**: 1834
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/functions/combinatorial/numbers.py`
**Line**: 1905
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/functions/combinatorial/numbers.py`
**Line**: 1948
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/functions/combinatorial/numbers.py`
**Line**: 2008
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/functions/combinatorial/numbers.py`
**Line**: 2078
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/functions/combinatorial/numbers.py`
**Line**: 2131
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/functions/combinatorial/numbers.py`
**Line**: 2179
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/functions/combinatorial/numbers.py`
**Line**: 2229
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/functions/combinatorial/numbers.py`
**Line**: 2292
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/functions/combinatorial/numbers.py`
**Line**: 3070
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/functions/elementary/complexes.py`
**Line**: 69
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/functions/elementary/complexes.py`
**Line**: 191
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/functions/elementary/complexes.py`
**Line**: 330
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/functions/elementary/complexes.py`
**Line**: 523
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/functions/elementary/complexes.py`
**Line**: 750
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/functions/elementary/complexes.py`
**Line**: 867
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/functions/elementary/complexes.py`
**Line**: 939
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/functions/elementary/complexes.py`
**Line**: 982
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/functions/elementary/complexes.py`
**Line**: 1062
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/functions/elementary/complexes.py`
**Line**: 1170
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/functions/elementary/complexes.py`
**Line**: 1278
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/functions/elementary/exponential.py`
**Line**: 276
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/functions/elementary/exponential.py`
**Line**: 661
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/functions/elementary/exponential.py`
**Line**: 1146
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/functions/elementary/hyperbolic.py`
**Line**: 189
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/functions/elementary/hyperbolic.py`
**Line**: 392
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/functions/elementary/hyperbolic.py`
**Line**: 644
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/functions/elementary/hyperbolic.py`
**Line**: 857
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/functions/elementary/hyperbolic.py`
**Line**: 1005
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/functions/elementary/hyperbolic.py`
**Line**: 1012
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/functions/elementary/hyperbolic.py`
**Line**: 1235
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/functions/elementary/hyperbolic.py`
**Line**: 1414
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/functions/elementary/hyperbolic.py`
**Line**: 1599
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/functions/elementary/hyperbolic.py`
**Line**: 1764
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/functions/elementary/hyperbolic.py`
**Line**: 1930
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/functions/elementary/hyperbolic.py`
**Line**: 2126
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/functions/elementary/integers.py`
**Line**: 27
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/functions/elementary/integers.py`
**Line**: 546
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/functions/elementary/miscellaneous.py`
**Line**: 901
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/functions/elementary/piecewise.py`
**Line**: 148
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/functions/elementary/piecewise.py`
**Line**: 157
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/functions/elementary/trigonometric.py`
**Line**: 301
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/functions/elementary/trigonometric.py`
**Line**: 607
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/functions/elementary/trigonometric.py`
**Line**: 969
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/functions/elementary/trigonometric.py`
**Line**: 1309
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/functions/elementary/trigonometric.py`
**Line**: 1580
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/functions/elementary/trigonometric.py`
**Line**: 1587
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/functions/elementary/trigonometric.py`
**Line**: 1613
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/functions/elementary/trigonometric.py`
**Line**: 1974
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/functions/elementary/trigonometric.py`
**Line**: 2174
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/functions/elementary/trigonometric.py`
**Line**: 2402
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/functions/elementary/trigonometric.py`
**Line**: 2654
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/functions/elementary/trigonometric.py`
**Line**: 2866
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/functions/elementary/trigonometric.py`
**Line**: 3076
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/functions/elementary/trigonometric.py`
**Line**: 3290
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/functions/elementary/trigonometric.py`
**Line**: 3552
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/functions/special/bessel.py`
**Line**: 63
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/functions/special/bessel.py`
**Line**: 179
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/functions/special/bessel.py`
**Line**: 327
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/functions/special/bessel.py`
**Line**: 482
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/functions/special/bessel.py`
**Line**: 649
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/functions/special/bessel.py`
**Line**: 999
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/functions/special/bessel.py`
**Line**: 1467
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/functions/special/bessel.py`
**Line**: 1641
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/functions/special/bessel.py`
**Line**: 1809
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/functions/special/bessel.py`
**Line**: 1968
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/functions/special/bessel.py`
**Line**: 2094
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/functions/special/beta_functions.py`
**Line**: 121
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/functions/special/delta_functions.py`
**Line**: 152
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/functions/special/delta_functions.py`
**Line**: 160
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/functions/special/delta_functions.py`
**Line**: 163
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/functions/special/delta_functions.py`
**Line**: 496
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/functions/special/delta_functions.py`
**Line**: 504
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/functions/special/delta_functions.py`
**Line**: 507
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/functions/special/elliptic_integrals.py`
**Line**: 59
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/functions/special/elliptic_integrals.py`
**Line**: 146
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/functions/special/elliptic_integrals.py`
**Line**: 237
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/functions/special/elliptic_integrals.py`
**Line**: 356
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/functions/special/error_functions.py`
**Line**: 143
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/functions/special/error_functions.py`
**Line**: 381
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/functions/special/error_functions.py`
**Line**: 565
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/functions/special/error_functions.py`
**Line**: 754
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/functions/special/error_functions.py`
**Line**: 884
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/functions/special/error_functions.py`
**Line**: 971
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/functions/special/error_functions.py`
**Line**: 1057
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/functions/special/error_functions.py`
**Line**: 1181
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/functions/special/error_functions.py`
**Line**: 1380
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/functions/special/error_functions.py`
**Line**: 1389
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/functions/special/error_functions.py`
**Line**: 1603
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/functions/special/error_functions.py`
**Line**: 1735
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/functions/special/error_functions.py`
**Line**: 1770
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/functions/special/error_functions.py`
**Line**: 2333
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/functions/special/error_functions.py`
**Line**: 2717
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/functions/special/gamma_functions.py`
**Line**: 121
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/functions/special/gamma_functions.py`
**Line**: 292
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/functions/special/gamma_functions.py`
**Line**: 485
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/functions/special/gamma_functions.py`
**Line**: 672
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/functions/special/gamma_functions.py`
**Line**: 976
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/functions/special/gamma_functions.py`
**Line**: 1135
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/functions/special/gamma_functions.py`
**Line**: 1229
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/functions/special/gamma_functions.py`
**Line**: 1324
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/functions/special/hyper.py`
**Line**: 226
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/functions/special/hyper.py`
**Line**: 814
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/functions/special/hyper.py`
**Line**: 1165
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/functions/special/mathieu_functions.py`
**Line**: 78
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/functions/special/mathieu_functions.py`
**Line**: 140
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/functions/special/mathieu_functions.py`
**Line**: 202
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/functions/special/mathieu_functions.py`
**Line**: 264
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/functions/special/polynomials.py`
**Line**: 127
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/functions/special/polynomials.py`
**Line**: 360
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/functions/special/polynomials.py`
**Line**: 514
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/functions/special/polynomials.py`
**Line**: 632
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/functions/special/polynomials.py`
**Line**: 721
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/functions/special/polynomials.py`
**Line**: 762
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/functions/special/polynomials.py`
**Line**: 830
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/functions/special/polynomials.py`
**Line**: 954
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/functions/special/polynomials.py`
**Line**: 1063
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/functions/special/polynomials.py`
**Line**: 1169
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/functions/special/polynomials.py`
**Line**: 1270
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/functions/special/polynomials.py`
**Line**: 1392
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/functions/special/singularity_functions.py`
**Line**: 114
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/functions/special/singularity_functions.py`
**Line**: 122
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/functions/special/singularity_functions.py`
**Line**: 125
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/functions/special/spherical_harmonics.py`
**Line**: 138
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/functions/special/spherical_harmonics.py`
**Line**: 326
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/functions/special/tensor_functions.py`
**Line**: 80
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/functions/special/tensor_functions.py`
**Line**: 147
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/functions/special/zeta_functions.py`
**Line**: 294
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/functions/special/zeta_functions.py`
**Line**: 503
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/functions/special/zeta_functions.py`
**Line**: 632
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/functions/special/zeta_functions.py`
**Line**: 692
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/functions/special/zeta_functions.py`
**Line**: 744
**Category**: unsafe_eval
**Match**: `eval(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/functions/combinatorial/tests/test_comb_factorials.py`
**Line**: 110
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/functions/combinatorial/tests/test_comb_factorials.py`
**Line**: 111
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/functions/combinatorial/tests/test_comb_factorials.py`
**Line**: 197
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/functions/combinatorial/tests/test_comb_factorials.py`
**Line**: 198
**Category**: weak_crypto
**Match**: `random.random(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/functions/combinatorial/tests/test_comb_numbers.py`
**Line**: 1183
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/functions/combinatorial/tests/test_comb_numbers.py`
**Line**: 1184
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/functions/elementary/tests/test_interface.py`
**Line**: 26
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/functions/elementary/tests/test_interface.py`
**Line**: 47
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/functions/elementary/tests/test_interface.py`
**Line**: 75
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/matrices/tests/test_commonmatrix.py`
**Line**: 825
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/matrices/tests/test_commonmatrix.py`
**Line**: 864
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/matrices/tests/test_commonmatrix.py`
**Line**: 894
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/matrices/tests/test_commonmatrix.py`
**Line**: 902
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/matrices/tests/test_matrices.py`
**Line**: 209
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/matrices/tests/test_matrices.py`
**Line**: 248
**Category**: unsafe_eval
**Match**: `eval(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/matrices/tests/test_matrices.py`
**Line**: 959
**Category**: weak_crypto
**Match**: `random.Random(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/matrices/tests/test_matrixbase.py`
**Line**: 506
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/matrices/tests/test_matrixbase.py`
**Line**: 514
**Category**: unsafe_eval
**Match**: `eval(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/matrices/tests/test_matrixbase.py`
**Line**: 1695
**Category**: weak_crypto
**Match**: `random.Random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/matrices/tests/test_normalforms.py`
**Line**: 39
**Category**: weak_crypto
**Match**: `random.Random(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/matrices/tests/test_sparse.py`
**Line**: 99
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/parsing/tests/test_autolev.py`
**Line**: 140
**Category**: unsafe_eval
**Match**: `exec(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/parsing/tests/test_sympy_parser.py`
**Line**: 156
**Category**: unsafe_eval
**Match**: `exec(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/physics/biomechanics/curve.py`
**Line**: 31
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/physics/biomechanics/curve.py`
**Line**: 192
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/physics/biomechanics/curve.py`
**Line**: 411
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/physics/biomechanics/curve.py`
**Line**: 633
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/physics/biomechanics/curve.py`
**Line**: 847
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/physics/biomechanics/curve.py`
**Line**: 1103
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/physics/biomechanics/curve.py`
**Line**: 1407
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/physics/biomechanics/curve.py`
**Line**: 1630
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/physics/quantum/anticommutator.py`
**Line**: 92
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/physics/quantum/anticommutator.py`
**Line**: 99
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/physics/quantum/commutator.py`
**Line**: 106
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/physics/quantum/commutator.py`
**Line**: 113
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/physics/quantum/hilbert.py`
**Line**: 146
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/physics/quantum/hilbert.py`
**Line**: 153
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/physics/quantum/hilbert.py`
**Line**: 329
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/physics/quantum/hilbert.py`
**Line**: 336
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/physics/quantum/hilbert.py`
**Line**: 473
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/physics/quantum/hilbert.py`
**Line**: 480
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/physics/quantum/hilbert.py`
**Line**: 591
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/physics/quantum/hilbert.py`
**Line**: 597
**Category**: unsafe_eval
**Match**: `eval(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/physics/quantum/qubit.py`
**Line**: 705
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/physics/quantum/qubit.py`
**Line**: 799
**Category**: weak_crypto
**Match**: `random.random(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/physics/quantum/state.py`
**Line**: 812
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/physics/tests/test_secondquant.py`
**Line**: 273
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/physics/biomechanics/tests/test_curve.py`
**Line**: 78
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/physics/quantum/tests/test_printing.py`
**Line**: 62
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/physics/quantum/tests/test_printing.py`
**Line**: 45
**Category**: unsafe_eval
**Match**: `exec(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/physics/quantum/tests/test_printing.py`
**Line**: 46
**Category**: unsafe_eval
**Match**: `exec(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/physics/quantum/tests/test_printing.py`
**Line**: 47
**Category**: unsafe_eval
**Match**: `exec(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/physics/quantum/tests/test_printing.py`
**Line**: 48
**Category**: unsafe_eval
**Match**: `exec(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/physics/quantum/tests/test_printing.py`
**Line**: 49
**Category**: unsafe_eval
**Match**: `exec(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/physics/quantum/tests/test_printing.py`
**Line**: 50
**Category**: unsafe_eval
**Match**: `exec(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/physics/quantum/tests/test_printing.py`
**Line**: 51
**Category**: unsafe_eval
**Match**: `exec(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/physics/quantum/tests/test_printing.py`
**Line**: 52
**Category**: unsafe_eval
**Match**: `exec(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/physics/quantum/tests/test_printing.py`
**Line**: 53
**Category**: unsafe_eval
**Match**: `exec(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/physics/quantum/tests/test_qubit.py`
**Line**: 112
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/physics/quantum/tests/test_qubit.py`
**Line**: 113
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/physics/quantum/tests/test_qubit.py`
**Line**: 113
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/physics/quantum/tests/test_qubit.py`
**Line**: 114
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/physics/quantum/tests/test_qubit.py`
**Line**: 114
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/physics/quantum/tests/test_qubit.py`
**Line**: 115
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/physics/quantum/tests/test_qubit.py`
**Line**: 117
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/physics/quantum/tests/test_qubit.py`
**Line**: 117
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/physics/quantum/tests/test_qubit.py`
**Line**: 118
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/physics/quantum/tests/test_qubit.py`
**Line**: 118
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/physics/quantum/tests/test_qubit.py`
**Line**: 118
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/physics/quantum/tests/test_qubit.py`
**Line**: 119
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/physics/quantum/tests/test_qubit.py`
**Line**: 120
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/physics/quantum/tests/test_qubit.py`
**Line**: 121
**Category**: weak_crypto
**Match**: `random.random(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/physics/units/tests/test_prefixes.py`
**Line**: 85
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/physics/units/tests/test_prefixes.py`
**Line**: 86
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/physics/vector/tests/test_frame.py`
**Line**: 759
**Category**: unsafe_pickle
**Match**: `pickle.loads(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/plotting/pygletplot/plot_axes.py`
**Line**: 44
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/plotting/pygletplot/plot_interval.py`
**Line**: 27
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/polys/tests/test_injections.py`
**Line**: 9
**Category**: unsafe_eval
**Match**: `exec(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/polys/tests/test_injections.py`
**Line**: 10
**Category**: unsafe_eval
**Match**: `exec(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/polys/tests/test_injections.py`
**Line**: 12
**Category**: unsafe_eval
**Match**: `exec(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/polys/tests/test_injections.py`
**Line**: 13
**Category**: unsafe_eval
**Match**: `exec(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/polys/tests/test_injections.py`
**Line**: 14
**Category**: unsafe_eval
**Match**: `exec(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/polys/tests/test_injections.py`
**Line**: 15
**Category**: unsafe_eval
**Match**: `exec(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/polys/tests/test_injections.py`
**Line**: 16
**Category**: unsafe_eval
**Match**: `exec(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/polys/tests/test_injections.py`
**Line**: 17
**Category**: unsafe_eval
**Match**: `exec(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/polys/tests/test_injections.py`
**Line**: 19
**Category**: unsafe_eval
**Match**: `exec(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/polys/tests/test_injections.py`
**Line**: 20
**Category**: unsafe_eval
**Match**: `exec(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/polys/tests/test_injections.py`
**Line**: 21
**Category**: unsafe_eval
**Match**: `exec(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/polys/tests/test_injections.py`
**Line**: 22
**Category**: unsafe_eval
**Match**: `exec(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/polys/tests/test_injections.py`
**Line**: 26
**Category**: unsafe_eval
**Match**: `exec(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/polys/tests/test_injections.py`
**Line**: 27
**Category**: unsafe_eval
**Match**: `exec(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/polys/tests/test_injections.py`
**Line**: 29
**Category**: unsafe_eval
**Match**: `exec(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/polys/tests/test_injections.py`
**Line**: 30
**Category**: unsafe_eval
**Match**: `exec(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/polys/tests/test_injections.py`
**Line**: 31
**Category**: unsafe_eval
**Match**: `exec(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/polys/tests/test_injections.py`
**Line**: 32
**Category**: unsafe_eval
**Match**: `exec(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/polys/tests/test_injections.py`
**Line**: 33
**Category**: unsafe_eval
**Match**: `exec(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/polys/tests/test_injections.py`
**Line**: 34
**Category**: unsafe_eval
**Match**: `exec(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/polys/tests/test_injections.py`
**Line**: 36
**Category**: unsafe_eval
**Match**: `exec(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/polys/tests/test_injections.py`
**Line**: 37
**Category**: unsafe_eval
**Match**: `exec(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/polys/tests/test_injections.py`
**Line**: 38
**Category**: unsafe_eval
**Match**: `exec(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/polys/tests/test_injections.py`
**Line**: 39
**Category**: unsafe_eval
**Match**: `exec(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/polys/tests/test_polyclasses.py`
**Line**: 218
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/polys/tests/test_polyclasses.py`
**Line**: 219
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/polys/tests/test_polyclasses.py`
**Line**: 221
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/polys/tests/test_polyclasses.py`
**Line**: 221
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/polys/tests/test_polytools.py`
**Line**: 1640
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/polys/tests/test_polytools.py`
**Line**: 1641
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/polys/tests/test_polytools.py`
**Line**: 1642
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/polys/tests/test_polytools.py`
**Line**: 1644
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/polys/tests/test_polytools.py`
**Line**: 1645
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/polys/tests/test_polytools.py`
**Line**: 1646
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/polys/tests/test_polytools.py`
**Line**: 1648
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/polys/tests/test_polytools.py`
**Line**: 1649
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/polys/tests/test_polytools.py`
**Line**: 1650
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/polys/tests/test_polytools.py`
**Line**: 1652
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/polys/tests/test_polytools.py`
**Line**: 1653
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/polys/tests/test_polytools.py`
**Line**: 1654
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/polys/tests/test_polytools.py`
**Line**: 1656
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/polys/tests/test_polytools.py`
**Line**: 1657
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/polys/tests/test_polytools.py`
**Line**: 1658
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/polys/tests/test_polytools.py`
**Line**: 1660
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/polys/tests/test_polytools.py`
**Line**: 1661
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/polys/tests/test_polytools.py`
**Line**: 1662
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/polys/tests/test_polytools.py`
**Line**: 1664
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/polys/tests/test_polytools.py`
**Line**: 1665
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/polys/tests/test_polytools.py`
**Line**: 1666
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/polys/tests/test_polytools.py`
**Line**: 1668
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/polys/tests/test_polytools.py`
**Line**: 1669
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/polys/tests/test_polytools.py`
**Line**: 1670
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/polys/tests/test_polytools.py`
**Line**: 1672
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/polys/tests/test_polytools.py`
**Line**: 1673
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/polys/tests/test_polytools.py`
**Line**: 1675
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/polys/tests/test_polytools.py`
**Line**: 1676
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/polys/tests/test_polytools.py`
**Line**: 1678
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/polys/tests/test_polytools.py`
**Line**: 1679
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/polys/tests/test_polytools.py`
**Line**: 1681
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/polys/tests/test_polytools.py`
**Line**: 1682
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/polys/tests/test_polytools.py`
**Line**: 1684
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/polys/tests/test_polytools.py`
**Line**: 1685
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/polys/tests/test_polytools.py`
**Line**: 1692
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/polys/tests/test_polytools.py`
**Line**: 1695
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/polys/tests/test_polytools.py`
**Line**: 3960
**Category**: unsafe_pickle
**Match**: `pickle.loads(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/polys/matrices/tests/test_domainmatrix.py`
**Line**: 1366
**Category**: unsafe_pickle
**Match**: `pickle.loads(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/polys/matrices/tests/test_domainmatrix.py`
**Line**: 1368
**Category**: unsafe_pickle
**Match**: `pickle.loads(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/printing/tests/test_lambdarepr.py`
**Line**: 36
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/printing/tests/test_lambdarepr.py`
**Line**: 43
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/printing/tests/test_lambdarepr.py`
**Line**: 52
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/printing/tests/test_lambdarepr.py`
**Line**: 60
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/printing/tests/test_lambdarepr.py`
**Line**: 69
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/printing/tests/test_lambdarepr.py`
**Line**: 79
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/printing/tests/test_lambdarepr.py`
**Line**: 89
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/printing/tests/test_lambdarepr.py`
**Line**: 102
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/printing/tests/test_lambdarepr.py`
**Line**: 115
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/printing/tests/test_lambdarepr.py`
**Line**: 128
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/printing/tests/test_lambdarepr.py`
**Line**: 141
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/printing/tests/test_lambdarepr.py`
**Line**: 153
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/printing/tests/test_lambdarepr.py`
**Line**: 158
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/printing/tests/test_latex.py`
**Line**: 3139
**Category**: unsafe_pickle
**Match**: `pickle.loads(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/printing/tests/test_repr.py`
**Line**: 30
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/printing/tests/test_repr.py`
**Line**: 41
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/printing/tests/test_repr.py`
**Line**: 50
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/printing/tests/test_repr.py`
**Line**: 72
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/printing/tests/test_repr.py`
**Line**: 184
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/printing/tests/test_repr.py`
**Line**: 206
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/printing/tests/test_repr.py`
**Line**: 33
**Category**: unsafe_eval
**Match**: `exec(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/printing/tests/test_repr.py`
**Line**: 47
**Category**: unsafe_eval
**Match**: `exec(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/printing/tests/test_tensorflow.py`
**Line**: 55
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/printing/tests/test_tensorflow.py`
**Line**: 87
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/printing/tests/test_tensorflow.py`
**Line**: 115
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/printing/tests/test_tensorflow.py`
**Line**: 132
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/printing/tests/test_tensorflow.py`
**Line**: 148
**Category**: unsafe_eval
**Match**: `eval(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/printing/tests/test_tensorflow.py`
**Line**: 184
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/printing/tests/test_tensorflow.py`
**Line**: 188
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/printing/tests/test_tensorflow.py`
**Line**: 192
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/printing/tests/test_tensorflow.py`
**Line**: 196
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/printing/tests/test_tensorflow.py`
**Line**: 200
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/printing/tests/test_tensorflow.py`
**Line**: 204
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/printing/tests/test_tensorflow.py`
**Line**: 212
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/printing/tests/test_tensorflow.py`
**Line**: 216
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/printing/tests/test_tensorflow.py`
**Line**: 220
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/printing/tests/test_tensorflow.py`
**Line**: 224
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/printing/tests/test_tensorflow.py`
**Line**: 228
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/printing/tests/test_tensorflow.py`
**Line**: 232
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/printing/tests/test_tensorflow.py`
**Line**: 258
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/printing/tests/test_tensorflow.py`
**Line**: 263
**Category**: weak_crypto
**Match**: `random.random(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/printing/tests/test_theanocode.py`
**Line**: 636
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/printing/tests/test_theanocode.py`
**Line**: 639
**Category**: unsafe_eval
**Match**: `eval(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/printing/tests/test_torch.py`
**Line**: 106
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/printing/tests/test_torch.py`
**Line**: 110
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/printing/tests/test_torch.py`
**Line**: 118
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/printing/tests/test_torch.py`
**Line**: 122
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/printing/tests/test_torch.py`
**Line**: 126
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/printing/tests/test_torch.py`
**Line**: 134
**Category**: weak_crypto
**Match**: `random.random(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/series/tests/test_limits.py`
**Line**: 169
**Category**: unsafe_eval
**Match**: `eval(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/tensor/array/tests/test_arrayop.py`
**Line**: 157
**Category**: weak_crypto
**Match**: `random.random(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/testing/tests/diagnose_imports.py`
**Line**: 212
**Category**: unsafe_eval
**Match**: `__import__(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/utilities/tests/test_lambdify.py`
**Line**: 862
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/utilities/tests/test_lambdify.py`
**Line**: 873
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/utilities/tests/test_lambdify.py`
**Line**: 885
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/utilities/tests/test_lambdify.py`
**Line**: 895
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/utilities/tests/test_lambdify.py`
**Line**: 905
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/utilities/tests/test_lambdify.py`
**Line**: 906
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/utilities/tests/test_lambdify.py`
**Line**: 907
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/utilities/tests/test_lambdify.py`
**Line**: 917
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/utilities/tests/test_lambdify.py`
**Line**: 927
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/utilities/tests/test_lambdify.py`
**Line**: 937
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/utilities/tests/test_lambdify.py`
**Line**: 954
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/utilities/tests/test_lambdify.py`
**Line**: 955
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/utilities/tests/test_lambdify.py`
**Line**: 957
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/utilities/tests/test_lambdify.py`
**Line**: 961
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/utilities/tests/test_lambdify.py`
**Line**: 975
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/utilities/tests/test_lambdify.py`
**Line**: 1521
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/utilities/tests/test_lambdify.py`
**Line**: 1523
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/utilities/tests/test_lambdify.py`
**Line**: 1524
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/utilities/tests/test_lambdify.py`
**Line**: 1525
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/utilities/tests/test_lambdify.py`
**Line**: 1527
**Category**: unsafe_eval
**Match**: `eval(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/utilities/tests/test_lambdify.py`
**Line**: 495
**Category**: weak_crypto
**Match**: `random.random(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/utilities/tests/test_matchpy_connector.py`
**Line**: 151
**Category**: unsafe_pickle
**Match**: `pickle.loads(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/utilities/tests/test_matchpy_connector.py`
**Line**: 155
**Category**: unsafe_pickle
**Match**: `pickle.loads(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/utilities/tests/test_matchpy_connector.py`
**Line**: 159
**Category**: unsafe_pickle
**Match**: `pickle.loads(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/utilities/tests/test_matchpy_connector.py`
**Line**: 163
**Category**: unsafe_pickle
**Match**: `pickle.loads(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/utilities/tests/test_pickling.py`
**Line**: 74
**Category**: unsafe_pickle
**Match**: `pickle.loads(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/utilities/tests/test_pickling.py`
**Line**: 197
**Category**: unsafe_pickle
**Match**: `pickle.loads(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/utilities/tests/test_pickling.py`
**Line**: 200
**Category**: unsafe_pickle
**Match**: `pickle.loads(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/utilities/tests/test_pickling.py`
**Line**: 712
**Category**: unsafe_pickle
**Match**: `pickle.loads(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/utilities/tests/test_pickling.py`
**Line**: 723
**Category**: unsafe_pickle
**Match**: `pickle.loads(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/sympy/utilities/_compilation/tests/test_compilation.py`
**Line**: 59
**Category**: weak_crypto
**Match**: `random.random(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/pkg_resources/tests/test_resources.py`
**Line**: 760
**Category**: unsafe_eval
**Match**: `__import__(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/setuptools/_distutils/core.py`
**Line**: 228
**Category**: unsafe_eval
**Match**: `exec(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/setuptools/_distutils/core.py`
**Line**: 268
**Category**: unsafe_eval
**Match**: `exec(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/setuptools/_distutils/dist.py`
**Line**: 845
**Category**: unsafe_eval
**Match**: `__import__(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/setuptools/_distutils/sysconfig.py`
**Line**: 290
**Category**: unsafe_eval
**Match**: `__import__(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/setuptools/_vendor/typing_extensions.py`
**Line**: 1215
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/setuptools/_vendor/typing_extensions.py`
**Line**: 1215
**Category**: unsafe_eval
**Match**: `exec(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/setuptools/command/bdist_wheel.py`
**Line**: 394
**Category**: path_traversal
**Match**: `os.path.join(self.data_dir, "..`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/setuptools/command/build_ext.py`
**Line**: 31
**Category**: unsafe_eval
**Match**: `__import__(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/setuptools/tests/fixtures.py`
**Line**: 242
**Category**: unsafe_eval
**Match**: `__import__(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/setuptools/tests/fixtures.py`
**Line**: 365
**Category**: unsafe_eval
**Match**: `__import__(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/setuptools/tests/namespaces.py`
**Line**: 26
**Category**: unsafe_eval
**Match**: `__import__(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/setuptools/tests/namespaces.py`
**Line**: 29
**Category**: unsafe_eval
**Match**: `__import__(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/setuptools/tests/namespaces.py`
**Line**: 89
**Category**: unsafe_eval
**Match**: `__import__(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/setuptools/tests/test_bdist_wheel.py`
**Line**: 624
**Category**: unsafe_eval
**Match**: `__import__(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/setuptools/tests/test_build_meta.py`
**Line**: 23
**Category**: unsafe_eval
**Match**: `__import__(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/setuptools/tests/test_build_meta.py`
**Line**: 103
**Category**: unsafe_eval
**Match**: `__import__(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/setuptools/tests/test_build_meta.py`
**Line**: 122
**Category**: unsafe_eval
**Match**: `__import__(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/setuptools/tests/test_build_meta.py`
**Line**: 144
**Category**: unsafe_eval
**Match**: `__import__(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/setuptools/tests/test_build_meta.py`
**Line**: 172
**Category**: unsafe_eval
**Match**: `__import__(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/setuptools/tests/test_build_meta.py`
**Line**: 180
**Category**: unsafe_eval
**Match**: `__import__(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/setuptools/tests/test_build_meta.py`
**Line**: 215
**Category**: unsafe_eval
**Match**: `__import__(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/setuptools/tests/test_build_meta.py`
**Line**: 490
**Category**: unsafe_eval
**Match**: `__import__(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/setuptools/tests/test_build_meta.py`
**Line**: 596
**Category**: unsafe_eval
**Match**: `__import__(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/setuptools/tests/test_build_meta.py`
**Line**: 632
**Category**: unsafe_eval
**Match**: `__import__(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/setuptools/tests/test_build_meta.py`
**Line**: 657
**Category**: unsafe_eval
**Match**: `__import__(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/setuptools/tests/test_build_meta.py`
**Line**: 680
**Category**: unsafe_eval
**Match**: `__import__(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/setuptools/tests/test_build_meta.py`
**Line**: 682
**Category**: unsafe_eval
**Match**: `__import__(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/setuptools/tests/test_build_meta.py`
**Line**: 820
**Category**: unsafe_eval
**Match**: `__import__(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/setuptools/tests/test_build_meta.py`
**Line**: 873
**Category**: unsafe_eval
**Match**: `__import__(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/setuptools/tests/test_build_meta.py`
**Line**: 896
**Category**: unsafe_eval
**Match**: `__import__(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/setuptools/tests/test_config_discovery.py`
**Line**: 133
**Category**: unsafe_eval
**Match**: `__import__(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/setuptools/tests/test_dist_info.py`
**Line**: 142
**Category**: unsafe_eval
**Match**: `__import__(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/setuptools/tests/test_distutils_adoption.py`
**Line**: 93
**Category**: unsafe_eval
**Match**: `__import__(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/setuptools/tests/test_editable_install.py`
**Line**: 449
**Category**: unsafe_eval
**Match**: `exec(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/setuptools/tests/test_editable_install.py`
**Line**: 120
**Category**: unsafe_eval
**Match**: `__import__(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/setuptools/tests/test_extern.py`
**Line**: 15
**Category**: unsafe_pickle
**Match**: `pickle.loads(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/setuptools/tests/test_sdist.py`
**Line**: 262
**Category**: path_traversal
**Match**: `os.path.join("sdist_test", "..`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/setuptools/tests/test_sdist.py`
**Line**: 269
**Category**: path_traversal
**Match**: `os.path.join("..`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/setuptools/tests/test_wheel.py`
**Line**: 481
**Category**: unsafe_eval
**Match**: `__import__(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/setuptools/_distutils/command/bdist_rpm.py`
**Line**: 357
**Category**: command_injection
**Match**: `os.popen(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/setuptools/_distutils/compilers/C/base.py`
**Line**: 1120
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/setuptools/_distutils/compilers/C/base.py`
**Line**: 1294
**Category**: unsafe_eval
**Match**: `__import__(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/setuptools/_vendor/backports/__init__.py`
**Line**: 1
**Category**: unsafe_eval
**Match**: `__import__(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/setuptools/_vendor/jaraco/context.py`
**Line**: 345
**Category**: unsafe_eval
**Match**: `__import__(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/setuptools/_vendor/more_itertools/more.py`
**Line**: 3723
**Category**: unsafe_eval
**Match**: `__import__(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/setuptools/_vendor/wheel/_bdist_wheel.py`
**Line**: 411
**Category**: path_traversal
**Match**: `os.path.join(self.data_dir, "..`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/setuptools/_vendor/wheel/_bdist_wheel.py`
**Line**: 39
**Category**: unsafe_eval
**Match**: `__import__(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/setuptools/_vendor/jaraco/collections/__init__.py`
**Line**: 1090
**Category**: weak_crypto
**Match**: `random.random(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/setuptools/_vendor/jaraco/functools/__init__.py`
**Line**: 522
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/setuptools/_vendor/packaging/licenses/__init__.py`
**Line**: 100
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/setuptools/tests/config/test_pyprojecttoml.py`
**Line**: 98
**Category**: unsafe_eval
**Match**: `exec(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/setuptools/tests/config/test_pyprojecttoml.py`
**Line**: 362
**Category**: unsafe_eval
**Match**: `__import__(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/opt_einsum/backends/theano.py`
**Line**: 46
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/_core/arrayprint.py`
**Line**: 1564
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/_core/function_base.py`
**Line**: 533
**Category**: unsafe_eval
**Match**: `__import__(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/_core/records.py`
**Line**: 703
**Category**: unsafe_pickle
**Match**: `pickle.loads(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/core/__init__.py`
**Line**: 19
**Category**: unsafe_eval
**Match**: `__import__(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/f2py/auxfuncs.py`
**Line**: 632
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/f2py/auxfuncs.py`
**Line**: 640
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/f2py/auxfuncs.py`
**Line**: 644
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/f2py/capi_maps.py`
**Line**: 159
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/f2py/capi_maps.py`
**Line**: 296
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/f2py/capi_maps.py`
**Line**: 449
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/f2py/crackfortran.py`
**Line**: 1329
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/f2py/crackfortran.py`
**Line**: 2271
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/f2py/crackfortran.py`
**Line**: 2559
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/f2py/crackfortran.py`
**Line**: 2637
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/f2py/crackfortran.py`
**Line**: 2646
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/f2py/crackfortran.py`
**Line**: 2914
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/f2py/crackfortran.py`
**Line**: 2985
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/f2py/crackfortran.py`
**Line**: 3016
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/f2py/crackfortran.py`
**Line**: 3468
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/lib/_format_impl.py`
**Line**: 838
**Category**: unsafe_pickle
**Match**: `pickle.load(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/lib/_npyio_impl.py`
**Line**: 492
**Category**: unsafe_pickle
**Match**: `pickle.load(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/lib/_npyio_impl.py`
**Line**: 494
**Category**: unsafe_pickle
**Match**: `pickle.load(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/lib/_polynomial_impl.py`
**Line**: 125
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/lib/_polynomial_impl.py`
**Line**: 125
**Category**: weak_crypto
**Match**: `random.random(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/lib/_utils_impl.py`
**Line**: 343
**Category**: unsafe_eval
**Match**: `__import__(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/linalg/_linalg.py`
**Line**: 1244
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/linalg/_linalg.py`
**Line**: 2943
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/linalg/_linalg.py`
**Line**: 2944
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/linalg/_linalg.py`
**Line**: 2945
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/linalg/_linalg.py`
**Line**: 2946
**Category**: weak_crypto
**Match**: `random.random(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/tests/test_public_api.py`
**Line**: 407
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/tests/test_public_api.py`
**Line**: 542
**Category**: unsafe_eval
**Match**: `__import__(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/tests/test_reloading.py`
**Line**: 45
**Category**: unsafe_pickle
**Match**: `pickle.loads(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/_core/tests/test__exceptions.py`
**Line**: 19
**Category**: unsafe_pickle
**Match**: `pickle.loads(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/_core/tests/test__exceptions.py`
**Line**: 84
**Category**: unsafe_pickle
**Match**: `pickle.loads(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/_core/tests/test_arrayprint.py`
**Line**: 340
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/_core/tests/test_arrayprint.py`
**Line**: 341
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/_core/tests/test_custom_dtypes.py`
**Line**: 308
**Category**: unsafe_pickle
**Match**: `pickle.loads(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/_core/tests/test_datetime.py`
**Line**: 851
**Category**: unsafe_pickle
**Match**: `pickle.loads(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/_core/tests/test_datetime.py`
**Line**: 853
**Category**: unsafe_pickle
**Match**: `pickle.loads(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/_core/tests/test_datetime.py`
**Line**: 855
**Category**: unsafe_pickle
**Match**: `pickle.loads(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/_core/tests/test_datetime.py`
**Line**: 858
**Category**: unsafe_pickle
**Match**: `pickle.loads(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/_core/tests/test_datetime.py`
**Line**: 865
**Category**: unsafe_pickle
**Match**: `pickle.loads(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/_core/tests/test_datetime.py`
**Line**: 869
**Category**: unsafe_pickle
**Match**: `pickle.loads(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/_core/tests/test_datetime.py`
**Line**: 873
**Category**: unsafe_pickle
**Match**: `pickle.loads(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/_core/tests/test_dtype.py`
**Line**: 1065
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/_core/tests/test_dtype.py`
**Line**: 1366
**Category**: unsafe_pickle
**Match**: `pickle.loads(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/_core/tests/test_dtype.py`
**Line**: 1428
**Category**: unsafe_pickle
**Match**: `pickle.loads(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/_core/tests/test_dtype.py`
**Line**: 1439
**Category**: unsafe_pickle
**Match**: `pickle.loads(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/_core/tests/test_einsum.py`
**Line**: 85
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/_core/tests/test_einsum.py`
**Line**: 104
**Category**: weak_crypto
**Match**: `random.random(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/_core/tests/test_multiarray.py`
**Line**: 1549
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/_core/tests/test_multiarray.py`
**Line**: 3939
**Category**: unsafe_eval
**Match**: `eval(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/_core/tests/test_multiarray.py`
**Line**: 5160
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/_core/tests/test_multiarray.py`
**Line**: 5198
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/_core/tests/test_multiarray.py`
**Line**: 5267
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/_core/tests/test_multiarray.py`
**Line**: 5346
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/_core/tests/test_multiarray.py`
**Line**: 5357
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/_core/tests/test_multiarray.py`
**Line**: 5364
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/_core/tests/test_multiarray.py`
**Line**: 5370
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/_core/tests/test_multiarray.py`
**Line**: 5915
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/_core/tests/test_multiarray.py`
**Line**: 6279
**Category**: weak_crypto
**Match**: `random.random(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/_core/tests/test_multiarray.py`
**Line**: 189
**Category**: unsafe_pickle
**Match**: `pickle.loads(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/_core/tests/test_multiarray.py`
**Line**: 1701
**Category**: unsafe_pickle
**Match**: `pickle.loads(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/_core/tests/test_multiarray.py`
**Line**: 1855
**Category**: unsafe_pickle
**Match**: `pickle.loads(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/_core/tests/test_multiarray.py`
**Line**: 1862
**Category**: unsafe_pickle
**Match**: `pickle.loads(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/_core/tests/test_multiarray.py`
**Line**: 1871
**Category**: unsafe_pickle
**Match**: `pickle.loads(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/_core/tests/test_multiarray.py`
**Line**: 1882
**Category**: unsafe_pickle
**Match**: `pickle.loads(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/_core/tests/test_multiarray.py`
**Line**: 4404
**Category**: unsafe_pickle
**Match**: `pickle.loads(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/_core/tests/test_multiarray.py`
**Line**: 4406
**Category**: unsafe_pickle
**Match**: `pickle.loads(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/_core/tests/test_multiarray.py`
**Line**: 4427
**Category**: unsafe_pickle
**Match**: `pickle.loads(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/_core/tests/test_multiarray.py`
**Line**: 4446
**Category**: unsafe_pickle
**Match**: `pickle.loads(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/_core/tests/test_multiarray.py`
**Line**: 4459
**Category**: unsafe_pickle
**Match**: `pickle.loads(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/_core/tests/test_multiarray.py`
**Line**: 4461
**Category**: unsafe_pickle
**Match**: `pickle.loads(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/_core/tests/test_multiarray.py`
**Line**: 4463
**Category**: unsafe_pickle
**Match**: `pickle.loads(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/_core/tests/test_multiarray.py`
**Line**: 4465
**Category**: unsafe_pickle
**Match**: `pickle.loads(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/_core/tests/test_multiarray.py`
**Line**: 4476
**Category**: unsafe_pickle
**Match**: `pickle.loads(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/_core/tests/test_multiarray.py`
**Line**: 4496
**Category**: unsafe_pickle
**Match**: `pickle.loads(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/_core/tests/test_multiarray.py`
**Line**: 4505
**Category**: unsafe_pickle
**Match**: `pickle.loads(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/_core/tests/test_multiarray.py`
**Line**: 4559
**Category**: unsafe_pickle
**Match**: `pickle.loads(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/_core/tests/test_multiarray.py`
**Line**: 8312
**Category**: unsafe_pickle
**Match**: `pickle.loads(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/_core/tests/test_multithreading.py`
**Line**: 30
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/_core/tests/test_overrides.py`
**Line**: 718
**Category**: weak_crypto
**Match**: `random.random(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/_core/tests/test_overrides.py`
**Line**: 221
**Category**: unsafe_pickle
**Match**: `pickle.loads(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/_core/tests/test_records.py`
**Line**: 170
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/_core/tests/test_records.py`
**Line**: 171
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/_core/tests/test_records.py`
**Line**: 173
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/_core/tests/test_records.py`
**Line**: 414
**Category**: unsafe_pickle
**Match**: `pickle.loads(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/_core/tests/test_records.py`
**Line**: 415
**Category**: unsafe_pickle
**Match**: `pickle.loads(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/_core/tests/test_records.py`
**Line**: 421
**Category**: unsafe_pickle
**Match**: `pickle.loads(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/_core/tests/test_records.py`
**Line**: 422
**Category**: unsafe_pickle
**Match**: `pickle.loads(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/_core/tests/test_records.py`
**Line**: 429
**Category**: unsafe_pickle
**Match**: `pickle.loads(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/_core/tests/test_records.py`
**Line**: 453
**Category**: unsafe_pickle
**Match**: `pickle.loads(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/_core/tests/test_regression.py`
**Line**: 52
**Category**: unsafe_pickle
**Match**: `pickle.load(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/_core/tests/test_regression.py`
**Line**: 363
**Category**: unsafe_pickle
**Match**: `pickle.load(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/_core/tests/test_regression.py`
**Line**: 489
**Category**: unsafe_pickle
**Match**: `pickle.loads(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/_core/tests/test_regression.py`
**Line**: 833
**Category**: unsafe_pickle
**Match**: `pickle.loads(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/_core/tests/test_regression.py`
**Line**: 1069
**Category**: unsafe_pickle
**Match**: `pickle.loads(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/_core/tests/test_regression.py`
**Line**: 1082
**Category**: unsafe_pickle
**Match**: `pickle.load(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/_core/tests/test_regression.py`
**Line**: 1275
**Category**: unsafe_pickle
**Match**: `pickle.loads(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/_core/tests/test_regression.py`
**Line**: 1277
**Category**: unsafe_pickle
**Match**: `pickle.loads(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/_core/tests/test_regression.py`
**Line**: 1907
**Category**: unsafe_pickle
**Match**: `pickle.loads(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/_core/tests/test_regression.py`
**Line**: 1919
**Category**: unsafe_pickle
**Match**: `pickle.loads(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/_core/tests/test_regression.py`
**Line**: 1931
**Category**: unsafe_pickle
**Match**: `pickle.loads(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/_core/tests/test_regression.py`
**Line**: 1957
**Category**: unsafe_pickle
**Match**: `pickle.loads(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/_core/tests/test_regression.py`
**Line**: 1966
**Category**: unsafe_pickle
**Match**: `pickle.loads(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/_core/tests/test_regression.py`
**Line**: 2212
**Category**: unsafe_pickle
**Match**: `pickle.loads(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/_core/tests/test_regression.py`
**Line**: 2436
**Category**: unsafe_pickle
**Match**: `pickle.loads(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/_core/tests/test_regression.py`
**Line**: 2567
**Category**: unsafe_pickle
**Match**: `pickle.loads(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/_core/tests/test_scalarmath.py`
**Line**: 628
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/_core/tests/test_scalarmath.py`
**Line**: 654
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/_core/tests/test_simd.py`
**Line**: 244
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/_core/tests/test_simd.py`
**Line**: 510
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/_core/tests/test_simd.py`
**Line**: 640
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/_core/tests/test_simd.py`
**Line**: 701
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/_core/tests/test_simd.py`
**Line**: 721
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/_core/tests/test_simd.py`
**Line**: 741
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/_core/tests/test_simd.py`
**Line**: 767
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/_core/tests/test_simd.py`
**Line**: 804
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/_core/tests/test_simd.py`
**Line**: 843
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/_core/tests/test_simd.py`
**Line**: 895
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/_core/tests/test_simd.py`
**Line**: 1102
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/_core/tests/test_stringdtype.py`
**Line**: 366
**Category**: unsafe_pickle
**Match**: `pickle.load(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/_core/tests/test_ufunc.py`
**Line**: 1738
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/_core/tests/test_ufunc.py`
**Line**: 1739
**Category**: weak_crypto
**Match**: `random.random(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/_core/tests/test_ufunc.py`
**Line**: 204
**Category**: unsafe_pickle
**Match**: `pickle.loads(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/_core/tests/test_ufunc.py`
**Line**: 209
**Category**: unsafe_pickle
**Match**: `pickle.loads(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/_core/tests/test_ufunc.py`
**Line**: 216
**Category**: unsafe_pickle
**Match**: `pickle.loads(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/_core/tests/test_ufunc.py`
**Line**: 226
**Category**: unsafe_pickle
**Match**: `pickle.loads(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/_core/tests/test_ufunc.py`
**Line**: 501
**Category**: unsafe_pickle
**Match**: `pickle.loads(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/_core/tests/test_umath.py`
**Line**: 513
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/_core/tests/test_umath.py`
**Line**: 577
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/_core/tests/test_umath_accuracy.py`
**Line**: 71
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/_core/tests/test_umath_accuracy.py`
**Line**: 72
**Category**: unsafe_eval
**Match**: `eval(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/fft/tests/test_helper.py`
**Line**: 25
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/fft/tests/test_helper.py`
**Line**: 161
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/lib/tests/test_function_base.py`
**Line**: 1185
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/lib/tests/test_histograms.py`
**Line**: 777
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/lib/tests/test_index_tricks.py`
**Line**: 531
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/lib/tests/test_io.py`
**Line**: 2689
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/lib/tests/test_io.py`
**Line**: 2726
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/lib/tests/test_nanfunctions.py`
**Line**: 860
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/lib/tests/test_nanfunctions.py`
**Line**: 892
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/lib/tests/test_nanfunctions.py`
**Line**: 1063
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/lib/tests/test_nanfunctions.py`
**Line**: 1096
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/lib/tests/test_nanfunctions.py`
**Line**: 1270
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/lib/tests/test_regression.py`
**Line**: 22
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/linalg/tests/test_linalg.py`
**Line**: 2066
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/linalg/tests/test_linalg.py`
**Line**: 2067
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/linalg/tests/test_linalg.py`
**Line**: 2068
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/linalg/tests/test_linalg.py`
**Line**: 2075
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/linalg/tests/test_linalg.py`
**Line**: 2076
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/linalg/tests/test_linalg.py`
**Line**: 2084
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/linalg/tests/test_linalg.py`
**Line**: 2085
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/linalg/tests/test_linalg.py`
**Line**: 2086
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/linalg/tests/test_linalg.py`
**Line**: 2087
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/linalg/tests/test_linalg.py`
**Line**: 2092
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/linalg/tests/test_linalg.py`
**Line**: 2093
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/linalg/tests/test_linalg.py`
**Line**: 2094
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/linalg/tests/test_linalg.py`
**Line**: 2095
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/linalg/tests/test_linalg.py`
**Line**: 2102
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/linalg/tests/test_linalg.py`
**Line**: 2103
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/linalg/tests/test_linalg.py`
**Line**: 2104
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/linalg/tests/test_linalg.py`
**Line**: 2105
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/linalg/tests/test_linalg.py`
**Line**: 2112
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/linalg/tests/test_linalg.py`
**Line**: 2113
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/linalg/tests/test_linalg.py`
**Line**: 2114
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/linalg/tests/test_linalg.py`
**Line**: 2115
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/linalg/tests/test_linalg.py`
**Line**: 2123
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/linalg/tests/test_linalg.py`
**Line**: 2124
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/linalg/tests/test_linalg.py`
**Line**: 2125
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/linalg/tests/test_linalg.py`
**Line**: 2135
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/linalg/tests/test_linalg.py`
**Line**: 2136
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/linalg/tests/test_linalg.py`
**Line**: 2146
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/linalg/tests/test_linalg.py`
**Line**: 2147
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/linalg/tests/test_linalg.py`
**Line**: 2148
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/linalg/tests/test_linalg.py`
**Line**: 2149
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/linalg/tests/test_linalg.py`
**Line**: 2158
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/linalg/tests/test_linalg.py`
**Line**: 2159
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/linalg/tests/test_linalg.py`
**Line**: 2160
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/linalg/tests/test_linalg.py`
**Line**: 2161
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/linalg/tests/test_linalg.py`
**Line**: 2162
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/linalg/tests/test_linalg.py`
**Line**: 2163
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/linalg/tests/test_linalg.py`
**Line**: 2187
**Category**: weak_crypto
**Match**: `random.random(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/ma/tests/test_core.py`
**Line**: 733
**Category**: unsafe_pickle
**Match**: `pickle.loads(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/ma/tests/test_core.py`
**Line**: 748
**Category**: unsafe_pickle
**Match**: `pickle.loads(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/ma/tests/test_core.py`
**Line**: 757
**Category**: unsafe_pickle
**Match**: `pickle.loads(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/ma/tests/test_core.py`
**Line**: 767
**Category**: unsafe_pickle
**Match**: `pickle.loads(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/ma/tests/test_core.py`
**Line**: 777
**Category**: unsafe_pickle
**Match**: `pickle.loads(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/ma/tests/test_core.py`
**Line**: 5547
**Category**: unsafe_pickle
**Match**: `pickle.load(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/ma/tests/test_extras.py`
**Line**: 1136
**Category**: weak_crypto
**Match**: `random.random(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/ma/tests/test_mrecords.py`
**Line**: 293
**Category**: unsafe_pickle
**Match**: `pickle.loads(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/ma/tests/test_old_ma.py`
**Line**: 621
**Category**: unsafe_pickle
**Match**: `pickle.loads(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/matrixlib/tests/test_masked_matrix.py`
**Line**: 89
**Category**: unsafe_pickle
**Match**: `pickle.loads(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/polynomial/tests/test_chebyshev.py`
**Line**: 135
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/polynomial/tests/test_chebyshev.py`
**Line**: 309
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/polynomial/tests/test_chebyshev.py`
**Line**: 353
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/polynomial/tests/test_chebyshev.py`
**Line**: 366
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/polynomial/tests/test_chebyshev.py`
**Line**: 388
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/polynomial/tests/test_chebyshev.py`
**Line**: 401
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/polynomial/tests/test_hermite.py`
**Line**: 122
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/polynomial/tests/test_hermite.py`
**Line**: 296
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/polynomial/tests/test_hermite.py`
**Line**: 340
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/polynomial/tests/test_hermite.py`
**Line**: 353
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/polynomial/tests/test_hermite.py`
**Line**: 375
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/polynomial/tests/test_hermite.py`
**Line**: 388
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/polynomial/tests/test_hermite_e.py`
**Line**: 122
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/polynomial/tests/test_hermite_e.py`
**Line**: 296
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/polynomial/tests/test_hermite_e.py`
**Line**: 341
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/polynomial/tests/test_hermite_e.py`
**Line**: 354
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/polynomial/tests/test_hermite_e.py`
**Line**: 376
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/polynomial/tests/test_hermite_e.py`
**Line**: 389
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/polynomial/tests/test_laguerre.py`
**Line**: 119
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/polynomial/tests/test_laguerre.py`
**Line**: 293
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/polynomial/tests/test_laguerre.py`
**Line**: 337
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/polynomial/tests/test_laguerre.py`
**Line**: 350
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/polynomial/tests/test_laguerre.py`
**Line**: 372
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/polynomial/tests/test_laguerre.py`
**Line**: 385
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/polynomial/tests/test_legendre.py`
**Line**: 123
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/polynomial/tests/test_legendre.py`
**Line**: 297
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/polynomial/tests/test_legendre.py`
**Line**: 344
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/polynomial/tests/test_legendre.py`
**Line**: 360
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/polynomial/tests/test_legendre.py`
**Line**: 382
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/polynomial/tests/test_legendre.py`
**Line**: 395
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/polynomial/tests/test_polynomial.py`
**Line**: 161
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/polynomial/tests/test_polynomial.py`
**Line**: 412
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/polynomial/tests/test_polynomial.py`
**Line**: 456
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/polynomial/tests/test_polynomial.py`
**Line**: 469
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/polynomial/tests/test_polynomial.py`
**Line**: 491
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/polynomial/tests/test_polynomial.py`
**Line**: 504
**Category**: weak_crypto
**Match**: `random.random(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/polynomial/tests/test_polynomial.py`
**Line**: 62
**Category**: unsafe_pickle
**Match**: `pickle.loads(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/random/tests/test_direct.py`
**Line**: 303
**Category**: unsafe_pickle
**Match**: `pickle.loads(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/random/tests/test_direct.py`
**Line**: 311
**Category**: unsafe_pickle
**Match**: `pickle.loads(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/random/tests/test_direct.py`
**Line**: 321
**Category**: unsafe_pickle
**Match**: `pickle.loads(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/random/tests/test_direct.py`
**Line**: 327
**Category**: unsafe_pickle
**Match**: `pickle.loads(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/random/tests/test_direct.py`
**Line**: 555
**Category**: unsafe_pickle
**Match**: `pickle.load(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/random/tests/test_generator_mt19937.py`
**Line**: 760
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/random/tests/test_generator_mt19937.py`
**Line**: 767
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/random/tests/test_generator_mt19937.py`
**Line**: 772
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/random/tests/test_generator_mt19937.py`
**Line**: 780
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/random/tests/test_generator_mt19937.py`
**Line**: 789
**Category**: weak_crypto
**Match**: `random.random(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/random/tests/test_generator_mt19937.py`
**Line**: 2776
**Category**: unsafe_pickle
**Match**: `pickle.loads(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/random/tests/test_generator_mt19937.py`
**Line**: 2782
**Category**: unsafe_pickle
**Match**: `pickle.loads(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/random/tests/test_generator_mt19937.py`
**Line**: 2798
**Category**: unsafe_pickle
**Match**: `pickle.load(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/random/tests/test_random.py`
**Line**: 374
**Category**: weak_crypto
**Match**: `random.random(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/random/tests/test_randomstate.py`
**Line**: 268
**Category**: unsafe_pickle
**Match**: `pickle.loads(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/random/tests/test_smoke.py`
**Line**: 437
**Category**: unsafe_pickle
**Match**: `pickle.loads(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/random/tests/test_smoke.py`
**Line**: 443
**Category**: unsafe_pickle
**Match**: `pickle.loads(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/testing/_private/utils.py`
**Line**: 1297
**Category**: unsafe_eval
**Match**: `exec(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/numpy/testing/_private/utils.py`
**Line**: 1583
**Category**: unsafe_eval
**Match**: `exec(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/networkx/algorithms/swap.py`
**Line**: 203
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/networkx/tests/test_all_random_functions.py`
**Line**: 17
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/networkx/tests/test_all_random_functions.py`
**Line**: 34
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/networkx/tests/test_all_random_functions.py`
**Line**: 242
**Category**: weak_crypto
**Match**: `random.random(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/networkx/tests/test_lazy_imports.py`
**Line**: 83
**Category**: unsafe_eval
**Match**: `exec(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/networkx/utils/decorators.py`
**Line**: 912
**Category**: unsafe_eval
**Match**: `exec(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/networkx/utils/misc.py`
**Line**: 465
**Category**: weak_crypto
**Match**: `random.Random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/networkx/algorithms/tests/test_cycles.py`
**Line**: 280
**Category**: weak_crypto
**Match**: `random.Random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/networkx/algorithms/tests/test_mis.py`
**Line**: 58
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/networkx/algorithms/approximation/tests/test_maxcut.py`
**Line**: 84
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/networkx/algorithms/connectivity/tests/test_edge_augmentation.py`
**Line**: 45
**Category**: weak_crypto
**Match**: `random.Random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/networkx/algorithms/connectivity/tests/test_edge_augmentation.py`
**Line**: 249
**Category**: weak_crypto
**Match**: `random.Random(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/networkx/algorithms/flow/tests/test_maxflow_large_graph.py`
**Line**: 56
**Category**: unsafe_pickle
**Match**: `pickle.load(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/networkx/algorithms/flow/tests/test_mincost.py`
**Line**: 469
**Category**: unsafe_pickle
**Match**: `pickle.load(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/networkx/algorithms/flow/tests/test_networksimplex.py`
**Line**: 180
**Category**: unsafe_pickle
**Match**: `pickle.load(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/networkx/algorithms/link_analysis/tests/test_pagerank.py`
**Line**: 69
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/networkx/algorithms/link_analysis/tests/test_pagerank.py`
**Line**: 194
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/networkx/algorithms/link_analysis/tests/test_pagerank.py`
**Line**: 197
**Category**: weak_crypto
**Match**: `random.random(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/networkx/algorithms/operators/tests/test_all.py`
**Line**: 28
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/networkx/algorithms/operators/tests/test_binary.py`
**Line**: 23
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/networkx/classes/tests/test_coreviews.py`
**Line**: 16
**Category**: unsafe_pickle
**Match**: `pickle.loads(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/networkx/classes/tests/test_coreviews.py`
**Line**: 19
**Category**: unsafe_pickle
**Match**: `pickle.loads(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/networkx/classes/tests/test_coreviews.py`
**Line**: 73
**Category**: unsafe_pickle
**Match**: `pickle.loads(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/networkx/classes/tests/test_coreviews.py`
**Line**: 153
**Category**: unsafe_pickle
**Match**: `pickle.loads(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/networkx/classes/tests/test_coreviews.py`
**Line**: 211
**Category**: unsafe_pickle
**Match**: `pickle.loads(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/networkx/classes/tests/test_graph.py`
**Line**: 610
**Category**: unsafe_pickle
**Match**: `pickle.loads(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/networkx/classes/tests/test_graph.py`
**Line**: 612
**Category**: unsafe_pickle
**Match**: `pickle.loads(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/networkx/classes/tests/test_graphviews.py`
**Line**: 18
**Category**: unsafe_pickle
**Match**: `pickle.loads(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/networkx/classes/tests/test_graphviews.py`
**Line**: 65
**Category**: unsafe_pickle
**Match**: `pickle.loads(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/networkx/classes/tests/test_graphviews.py`
**Line**: 118
**Category**: unsafe_pickle
**Match**: `pickle.loads(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/networkx/classes/tests/test_graphviews.py`
**Line**: 157
**Category**: unsafe_pickle
**Match**: `pickle.loads(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/networkx/classes/tests/test_graphviews.py`
**Line**: 206
**Category**: unsafe_pickle
**Match**: `pickle.loads(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/networkx/classes/tests/test_reportviews.py`
**Line**: 22
**Category**: unsafe_pickle
**Match**: `pickle.loads(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/networkx/classes/tests/test_reportviews.py`
**Line**: 93
**Category**: unsafe_pickle
**Match**: `pickle.loads(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/networkx/classes/tests/test_reportviews.py`
**Line**: 278
**Category**: unsafe_pickle
**Match**: `pickle.loads(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/networkx/classes/tests/test_reportviews.py`
**Line**: 568
**Category**: unsafe_pickle
**Match**: `pickle.loads(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/networkx/classes/tests/test_reportviews.py`
**Line**: 1023
**Category**: unsafe_pickle
**Match**: `pickle.loads(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/networkx/classes/tests/test_reportviews.py`
**Line**: 1412
**Category**: unsafe_pickle
**Match**: `pickle.loads(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/networkx/generators/tests/test_geometric.py`
**Line**: 124
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/networkx/readwrite/tests/test_text.py`
**Line**: 565
**Category**: weak_crypto
**Match**: `random.Random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/networkx/readwrite/tests/test_text.py`
**Line**: 1678
**Category**: weak_crypto
**Match**: `random.Random(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/networkx/utils/tests/test_backends.py`
**Line**: 28
**Category**: unsafe_pickle
**Match**: `pickle.loads(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/networkx/utils/tests/test_backends.py`
**Line**: 34
**Category**: unsafe_pickle
**Match**: `pickle.loads(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/networkx/utils/tests/test_backends.py`
**Line**: 37
**Category**: unsafe_pickle
**Match**: `pickle.loads(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/networkx/utils/tests/test_config.py`
**Line**: 47
**Category**: unsafe_pickle
**Match**: `pickle.loads(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/networkx/utils/tests/test_config.py`
**Line**: 102
**Category**: unsafe_pickle
**Match**: `pickle.loads(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/networkx/utils/tests/test_decorators.py`
**Line**: 227
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/networkx/utils/tests/test_decorators.py`
**Line**: 232
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/networkx/utils/tests/test_decorators.py`
**Line**: 238
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/networkx/utils/tests/test_decorators.py`
**Line**: 246
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/networkx/utils/tests/test_decorators.py`
**Line**: 248
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/networkx/utils/tests/test_decorators.py`
**Line**: 256
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/networkx/utils/tests/test_decorators.py`
**Line**: 260
**Category**: weak_crypto
**Match**: `random.Random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/networkx/utils/tests/test_decorators.py`
**Line**: 263
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/networkx/utils/tests/test_decorators.py`
**Line**: 267
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/networkx/utils/tests/test_decorators.py`
**Line**: 280
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/networkx/utils/tests/test_decorators.py`
**Line**: 284
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/networkx/utils/tests/test_decorators.py`
**Line**: 297
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/networkx/utils/tests/test_decorators.py`
**Line**: 301
**Category**: weak_crypto
**Match**: `random.Random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/networkx/utils/tests/test_decorators.py`
**Line**: 303
**Category**: weak_crypto
**Match**: `random.Random(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/fsspec/implementations/cache_metadata.py`
**Line**: 64
**Category**: unsafe_pickle
**Match**: `pickle.load(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/fsspec/tests/abstract/mv.py`
**Line**: 36
**Category**: command_injection
**Match**: `os.system(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/fsspec/tests/abstract/mv.py`
**Line**: 55
**Category**: command_injection
**Match**: `os.system(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/triton/language/__init__.py`
**Line**: 306
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/triton/profiler/proton.py`
**Line**: 62
**Category**: unsafe_eval
**Match**: `exec(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/triton/runtime/interpreter.py`
**Line**: 1365
**Category**: unsafe_eval
**Match**: `exec(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/triton/runtime/jit.py`
**Line**: 458
**Category**: unsafe_eval
**Match**: `exec(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/scipy/_lib/_bunch.py`
**Line**: 160
**Category**: unsafe_eval
**Match**: `exec(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/scipy/_lib/decorator.py`
**Line**: 166
**Category**: unsafe_eval
**Match**: `exec(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/scipy/datasets/_fetchers.py`
**Line**: 80
**Category**: unsafe_pickle
**Match**: `pickle.load(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/scipy/optimize/_nonlin.py`
**Line**: 1621
**Category**: unsafe_eval
**Match**: `exec(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/scipy/optimize/_optimize.py`
**Line**: 323
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/scipy/optimize/_optimize.py`
**Line**: 4155
**Category**: unsafe_eval
**Match**: `__import__(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/scipy/stats/_distn_infrastructure.py`
**Line**: 368
**Category**: unsafe_eval
**Match**: `exec(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/scipy/stats/_distn_infrastructure.py`
**Line**: 744
**Category**: unsafe_eval
**Match**: `exec(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/scipy/_lib/tests/test_bunch.py`
**Line**: 59
**Category**: unsafe_pickle
**Match**: `pickle.loads(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/scipy/_lib/array_api_compat/cupy/__init__.py`
**Line**: 10
**Category**: unsafe_eval
**Match**: `__import__(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/scipy/_lib/array_api_compat/cupy/__init__.py`
**Line**: 11
**Category**: unsafe_eval
**Match**: `__import__(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/scipy/_lib/array_api_compat/cupy/fft.py`
**Line**: 6
**Category**: unsafe_eval
**Match**: `exec(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/scipy/_lib/array_api_compat/cupy/linalg.py`
**Line**: 6
**Category**: unsafe_eval
**Match**: `exec(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/scipy/_lib/array_api_compat/numpy/__init__.py`
**Line**: 22
**Category**: unsafe_eval
**Match**: `__import__(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/scipy/_lib/array_api_compat/numpy/__init__.py`
**Line**: 24
**Category**: unsafe_eval
**Match**: `__import__(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/scipy/_lib/array_api_compat/torch/__init__.py`
**Line**: 12
**Category**: unsafe_eval
**Match**: `exec(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/scipy/_lib/array_api_compat/torch/__init__.py`
**Line**: 19
**Category**: unsafe_eval
**Match**: `__import__(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/scipy/_lib/array_api_compat/torch/__init__.py`
**Line**: 20
**Category**: unsafe_eval
**Match**: `__import__(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/scipy/_lib/array_api_compat/dask/array/__init__.py`
**Line**: 11
**Category**: unsafe_eval
**Match**: `__import__(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/scipy/_lib/array_api_compat/dask/array/__init__.py`
**Line**: 12
**Category**: unsafe_eval
**Match**: `__import__(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/scipy/_lib/array_api_compat/dask/array/fft.py`
**Line**: 6
**Category**: unsafe_eval
**Match**: `exec(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/scipy/_lib/array_api_compat/dask/array/linalg.py`
**Line**: 22
**Category**: unsafe_eval
**Match**: `exec(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/scipy/_lib/pyprima/common/linalg.py`
**Line**: 229
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/scipy/fft/tests/mock_backend.py`
**Line**: 20
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/scipy/fft/tests/mock_backend.py`
**Line**: 21
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/scipy/fft/tests/mock_backend.py`
**Line**: 22
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/scipy/fft/tests/mock_backend.py`
**Line**: 24
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/scipy/fft/tests/mock_backend.py`
**Line**: 25
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/scipy/fft/tests/mock_backend.py`
**Line**: 26
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/scipy/fft/tests/mock_backend.py`
**Line**: 28
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/scipy/fft/tests/mock_backend.py`
**Line**: 29
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/scipy/fft/tests/mock_backend.py`
**Line**: 30
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/scipy/fft/tests/mock_backend.py`
**Line**: 32
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/scipy/fft/tests/mock_backend.py`
**Line**: 33
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/scipy/fft/tests/mock_backend.py`
**Line**: 34
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/scipy/fft/tests/mock_backend.py`
**Line**: 36
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/scipy/fft/tests/mock_backend.py`
**Line**: 37
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/scipy/fft/tests/mock_backend.py`
**Line**: 38
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/scipy/fft/tests/mock_backend.py`
**Line**: 40
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/scipy/fft/tests/mock_backend.py`
**Line**: 41
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/scipy/fft/tests/mock_backend.py`
**Line**: 42
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/scipy/fft/tests/mock_backend.py`
**Line**: 44
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/scipy/fft/tests/mock_backend.py`
**Line**: 45
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/scipy/fft/tests/mock_backend.py`
**Line**: 46
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/scipy/fft/tests/mock_backend.py`
**Line**: 47
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/scipy/fft/tests/mock_backend.py`
**Line**: 49
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/scipy/fft/tests/mock_backend.py`
**Line**: 50
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/scipy/fft/tests/mock_backend.py`
**Line**: 51
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/scipy/fft/tests/mock_backend.py`
**Line**: 52
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/scipy/fft/tests/mock_backend.py`
**Line**: 54
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/scipy/fft/tests/mock_backend.py`
**Line**: 55
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/scipy/fft/tests/test_basic.py`
**Line**: 465
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/scipy/fft/tests/test_helper.py`
**Line**: 446
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/scipy/fft/tests/test_real_transforms.py`
**Line**: 90
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/scipy/fft/tests/test_real_transforms.py`
**Line**: 134
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/scipy/fft/_pocketfft/tests/test_basic.py`
**Line**: 705
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/scipy/fft/_pocketfft/tests/test_basic.py`
**Line**: 710
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/scipy/fft/_pocketfft/tests/test_basic.py`
**Line**: 715
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/scipy/fft/_pocketfft/tests/test_basic.py`
**Line**: 738
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/scipy/fft/_pocketfft/tests/test_basic.py`
**Line**: 795
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/scipy/fftpack/tests/test_basic.py`
**Line**: 661
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/scipy/fftpack/tests/test_basic.py`
**Line**: 666
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/scipy/fftpack/tests/test_basic.py`
**Line**: 671
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/scipy/fftpack/tests/test_helper.py`
**Line**: 31
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/scipy/interpolate/tests/test_bsplines.py`
**Line**: 70
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/scipy/interpolate/tests/test_bsplines.py`
**Line**: 1755
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/scipy/interpolate/tests/test_fitpack.py`
**Line**: 429
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/scipy/interpolate/tests/test_fitpack.py`
**Line**: 430
**Category**: weak_crypto
**Match**: `random.random(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/scipy/interpolate/tests/test_interpnd.py`
**Line**: 172
**Category**: unsafe_pickle
**Match**: `pickle.loads(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/scipy/interpolate/tests/test_interpnd.py`
**Line**: 411
**Category**: unsafe_pickle
**Match**: `pickle.loads(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/scipy/interpolate/tests/test_interpolate.py`
**Line**: 1136
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/scipy/interpolate/tests/test_interpolate.py`
**Line**: 1137
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/scipy/interpolate/tests/test_interpolate.py`
**Line**: 1139
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/scipy/interpolate/tests/test_interpolate.py`
**Line**: 1150
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/scipy/interpolate/tests/test_interpolate.py`
**Line**: 1184
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/scipy/interpolate/tests/test_interpolate.py`
**Line**: 1185
**Category**: weak_crypto
**Match**: `random.random(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/scipy/interpolate/tests/test_rbfinterp.py`
**Line**: 418
**Category**: unsafe_pickle
**Match**: `pickle.loads(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/scipy/io/tests/test_mmio.py`
**Line**: 126
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/scipy/io/tests/test_mmio.py`
**Line**: 132
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/scipy/io/tests/test_mmio.py`
**Line**: 261
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/scipy/io/tests/test_mmio.py`
**Line**: 268
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/scipy/linalg/tests/test_decomp_update.py`
**Line**: 99
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/scipy/linalg/tests/test_decomp_update.py`
**Line**: 101
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/scipy/linalg/tests/test_decomp_update.py`
**Line**: 1182
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/scipy/linalg/tests/test_decomp_update.py`
**Line**: 1183
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/scipy/linalg/tests/test_decomp_update.py`
**Line**: 1185
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/scipy/linalg/tests/test_decomp_update.py`
**Line**: 1186
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/scipy/linalg/tests/test_decomp_update.py`
**Line**: 1189
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/scipy/linalg/tests/test_decomp_update.py`
**Line**: 1192
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/scipy/linalg/tests/test_decomp_update.py`
**Line**: 1682
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/scipy/linalg/tests/test_decomp_update.py`
**Line**: 1683
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/scipy/linalg/tests/test_decomp_update.py`
**Line**: 1685
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/scipy/linalg/tests/test_decomp_update.py`
**Line**: 1685
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/scipy/linalg/tests/test_decomp_update.py`
**Line**: 1686
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/scipy/linalg/tests/test_decomp_update.py`
**Line**: 1686
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/scipy/linalg/tests/test_lapack.py`
**Line**: 765
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/scipy/linalg/tests/test_lapack.py`
**Line**: 768
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/scipy/linalg/tests/test_lapack.py`
**Line**: 768
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/scipy/ndimage/tests/test_filters.py`
**Line**: 2699
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/scipy/ndimage/tests/test_filters.py`
**Line**: 2711
**Category**: weak_crypto
**Match**: `random.random(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/scipy/odr/tests/test_odr.py`
**Line**: 576
**Category**: unsafe_pickle
**Match**: `pickle.loads(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/scipy/odr/tests/test_odr.py`
**Line**: 585
**Category**: unsafe_pickle
**Match**: `pickle.loads(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/scipy/odr/tests/test_odr.py`
**Line**: 589
**Category**: unsafe_pickle
**Match**: `pickle.loads(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/scipy/odr/tests/test_odr.py`
**Line**: 598
**Category**: unsafe_pickle
**Match**: `pickle.loads(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/scipy/odr/tests/test_odr.py`
**Line**: 607
**Category**: unsafe_pickle
**Match**: `pickle.loads(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/scipy/optimize/tests/test__numdiff.py`
**Line**: 498
**Category**: weak_crypto
**Match**: `random.random(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/scipy/optimize/tests/test_constraints.py`
**Line**: 183
**Category**: unsafe_eval
**Match**: `eval(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/scipy/optimize/tests/test_slsqp.py`
**Line**: 604
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/scipy/signal/tests/test_signaltools.py`
**Line**: 1205
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/scipy/signal/tests/test_signaltools.py`
**Line**: 1206
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/scipy/signal/tests/test_signaltools.py`
**Line**: 1683
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/scipy/signal/tests/test_signaltools.py`
**Line**: 1684
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/scipy/sparse/tests/test_array_api.py`
**Line**: 109
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/scipy/sparse/tests/test_base.py`
**Line**: 1955
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/scipy/sparse/tests/test_base.py`
**Line**: 5294
**Category**: weak_crypto
**Match**: `random.random(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/scipy/sparse/tests/test_base.py`
**Line**: 2255
**Category**: unsafe_pickle
**Match**: `pickle.loads(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/scipy/sparse/tests/test_csc.py`
**Line**: 11
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/scipy/sparse/tests/test_csc.py`
**Line**: 26
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/scipy/sparse/tests/test_csr.py`
**Line**: 18
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/scipy/sparse/tests/test_csr.py`
**Line**: 35
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/scipy/sparse/tests/test_csr.py`
**Line**: 50
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/scipy/sparse/tests/test_matrix_io.py`
**Line**: 37
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/scipy/sparse/csgraph/tests/test_conversions.py`
**Line**: 9
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/scipy/sparse/csgraph/tests/test_conversions.py`
**Line**: 33
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/scipy/sparse/csgraph/tests/test_conversions.py`
**Line**: 47
**Category**: weak_crypto
**Match**: `random.random(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/scipy/sparse/csgraph/tests/test_graph_laplacian.py`
**Line**: 44
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/scipy/sparse/csgraph/tests/test_graph_laplacian.py`
**Line**: 300
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/scipy/sparse/csgraph/tests/test_graph_laplacian.py`
**Line**: 302
**Category**: unsafe_eval
**Match**: `eval(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/scipy/sparse/csgraph/tests/test_spanning_tree.py`
**Line**: 47
**Category**: weak_crypto
**Match**: `random.random(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/scipy/sparse/linalg/tests/test_interface.py`
**Line**: 397
**Category**: unsafe_pickle
**Match**: `pickle.loads(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/scipy/sparse/linalg/tests/test_matfuncs.py`
**Line**: 63
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/scipy/sparse/linalg/tests/test_matfuncs.py`
**Line**: 72
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/scipy/sparse/linalg/_eigen/arpack/tests/test_arpack.py`
**Line**: 548
**Category**: weak_crypto
**Match**: `random.random(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/scipy/spatial/tests/test_distance.py`
**Line**: 445
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/scipy/spatial/tests/test_distance.py`
**Line**: 452
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/scipy/spatial/tests/test_distance.py`
**Line**: 559
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/scipy/spatial/tests/test_distance.py`
**Line**: 569
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/scipy/spatial/tests/test_distance.py`
**Line**: 754
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/scipy/spatial/tests/test_distance.py`
**Line**: 761
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/scipy/spatial/tests/test_distance.py`
**Line**: 1393
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/scipy/spatial/tests/test_distance.py`
**Line**: 1403
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/scipy/spatial/tests/test_distance.py`
**Line**: 2136
**Category**: unsafe_eval
**Match**: `eval(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/scipy/spatial/tests/test_hausdorff.py`
**Line**: 16
**Category**: weak_crypto
**Match**: `random.random(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/scipy/spatial/tests/test_kdtree.py`
**Line**: 855
**Category**: unsafe_pickle
**Match**: `pickle.loads(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/scipy/spatial/tests/test_kdtree.py`
**Line**: 869
**Category**: unsafe_pickle
**Match**: `pickle.loads(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/scipy/spatial/tests/test_qhull.py`
**Line**: 273
**Category**: weak_crypto
**Match**: `random.random(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/scipy/spatial/transform/tests/test_rotation.py`
**Line**: 2118
**Category**: unsafe_pickle
**Match**: `pickle.loads(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/scipy/special/_precompute/expn_asy.py`
**Line**: 36
**Category**: path_traversal
**Match**: `os.path.join('..`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/scipy/special/_precompute/gammainc_data.py`
**Line**: 116
**Category**: path_traversal
**Match**: `os.path.join(pwd, '..`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/scipy/special/_precompute/wright_bessel.py`
**Line**: 183
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/scipy/special/_precompute/wright_bessel.py`
**Line**: 203
**Category**: unsafe_eval
**Match**: `eval(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/scipy/special/_precompute/wright_bessel_data.py`
**Line**: 144
**Category**: path_traversal
**Match**: `os.path.join(pwd, '..`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/scipy/special/tests/test_basic.py`
**Line**: 3535
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/scipy/special/tests/test_basic.py`
**Line**: 3536
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/scipy/special/tests/test_basic.py`
**Line**: 3972
**Category**: weak_crypto
**Match**: `random.random(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/scipy/special/tests/test_orthogonal.py`
**Line**: 280
**Category**: unsafe_eval
**Match**: `eval(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/scipy/special/tests/test_orthogonal.py`
**Line**: 78
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/scipy/special/tests/test_orthogonal.py`
**Line**: 232
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/scipy/special/tests/test_orthogonal.py`
**Line**: 233
**Category**: weak_crypto
**Match**: `random.random(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/scipy/special/tests/test_support_alternative_backends.py`
**Line**: 194
**Category**: unsafe_pickle
**Match**: `pickle.loads(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/scipy/stats/tests/common_tests.py`
**Line**: 303
**Category**: unsafe_pickle
**Match**: `pickle.loads(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/scipy/stats/tests/common_tests.py`
**Line**: 316
**Category**: unsafe_pickle
**Match**: `pickle.loads(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/scipy/stats/tests/common_tests.py`
**Line**: 326
**Category**: unsafe_pickle
**Match**: `pickle.loads(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/scipy/stats/tests/test_axis_nan_policy.py`
**Line**: 764
**Category**: unsafe_pickle
**Match**: `pickle.loads(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/scipy/stats/tests/test_binned_statistic.py`
**Line**: 164
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/scipy/stats/tests/test_binned_statistic.py`
**Line**: 482
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/scipy/stats/tests/test_binned_statistic.py`
**Line**: 483
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/scipy/stats/tests/test_binned_statistic.py`
**Line**: 497
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/scipy/stats/tests/test_binned_statistic.py`
**Line**: 498
**Category**: weak_crypto
**Match**: `random.random(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/scipy/stats/tests/test_continuous.py`
**Line**: 89
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/scipy/stats/tests/test_continuous.py`
**Line**: 2047
**Category**: unsafe_eval
**Match**: `eval(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/scipy/stats/tests/test_continuous.py`
**Line**: 974
**Category**: unsafe_pickle
**Match**: `pickle.loads(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/scipy/stats/tests/test_distributions.py`
**Line**: 6876
**Category**: unsafe_pickle
**Match**: `pickle.loads(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/scipy/stats/tests/test_mstats_basic.py`
**Line**: 939
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/scipy/stats/tests/test_mstats_basic.py`
**Line**: 1834
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/scipy/stats/tests/test_multivariate.py`
**Line**: 2806
**Category**: weak_crypto
**Match**: `random.random(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/scipy/stats/tests/test_multivariate.py`
**Line**: 3723
**Category**: unsafe_pickle
**Match**: `pickle.loads(`

### 🔴 security_pattern
**File**: `venv/lib/python3.12/site-packages/scipy/stats/tests/test_sampling.py`
**Line**: 258
**Category**: unsafe_pickle
**Match**: `pickle.loads(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/scipy/stats/tests/test_stats.py`
**Line**: 2399
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/scipy/stats/tests/test_stats.py`
**Line**: 5331
**Category**: weak_crypto
**Match**: `random.random(`

### 🟡 security_pattern
**File**: `venv/lib/python3.12/site-packages/scipy/stats/tests/test_stats.py`
**Line**: 5332
**Category**: weak_crypto
**Match**: `random.random(`

## Sensitive Files

### 🟡 sensitive_file
**File**: `.env.example`

## Code Quality

### 🟢 code_quality
**File**: `security_scan.py`
**Line**: 306
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/typing_extensions.py`
**Line**: 713
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/typing_extensions.py`
**Line**: 1885
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/typing_extensions.py`
**Line**: 1928
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/typing_extensions.py`
**Line**: 2061
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/typing_extensions.py`
**Line**: 3267
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/mpmath/ctx_mp.py`
**Line**: 306
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/mpmath/math2.py`
**Line**: 207
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/mpmath/math2.py`
**Line**: 221
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/mpmath/math2.py`
**Line**: 359
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/mpmath/math2.py`
**Line**: 589
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/pkg_resources/__init__.py`
**Line**: 118
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/pkg_resources/__init__.py`
**Line**: 2059
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/pkg_resources/__init__.py`
**Line**: 3227
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/pkg_resources/__init__.py`
**Line**: 3334
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/pkg_resources/__init__.py`
**Line**: 3635
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/setuptools/_core_metadata.py`
**Line**: 122
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/setuptools/_core_metadata.py`
**Line**: 150
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/setuptools/_normalization.py`
**Line**: 148
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/setuptools/_static.py`
**Line**: 24
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/setuptools/_static.py`
**Line**: 41
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/setuptools/dist.py`
**Line**: 129
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/setuptools/dist.py`
**Line**: 174
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/setuptools/dist.py`
**Line**: 443
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/setuptools/dist.py`
**Line**: 545
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/setuptools/dist.py`
**Line**: 682
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/setuptools/installer.py`
**Line**: 70
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/setuptools/msvc.py`
**Line**: 611
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/setuptools/unicode_utils.py`
**Line**: 99
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/opt_einsum/paths.py`
**Line**: 926
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/numpy/__init__.py`
**Line**: 914
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/numpy/conftest.py`
**Line**: 105
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/networkx/conftest.py`
**Line**: 102
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/fsspec/asyn.py`
**Line**: 334
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/fsspec/asyn.py`
**Line**: 498
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/fsspec/asyn.py`
**Line**: 985
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/fsspec/caching.py`
**Line**: 93
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/fsspec/caching.py`
**Line**: 501
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/fsspec/compression.py`
**Line**: 13
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/fsspec/generic.py`
**Line**: 326
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/fsspec/generic.py`
**Line**: 327
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/fsspec/spec.py`
**Line**: 493
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/fsspec/utils.py`
**Line**: 300
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/triton/knobs.py`
**Line**: 355
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/triton/knobs.py`
**Line**: 376
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/conftest.py`
**Line**: 504
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/pip/_internal/build_env.py`
**Line**: 201
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/pip/_internal/cache.py`
**Line**: 279
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/pip/_vendor/typing_extensions.py`
**Line**: 700
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/pip/_vendor/typing_extensions.py`
**Line**: 1733
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/pip/_vendor/typing_extensions.py`
**Line**: 1762
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/pip/_internal/cli/base_command.py`
**Line**: 145
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/pip/_internal/commands/inspect.py`
**Line**: 60
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/pip/_internal/index/collector.py`
**Line**: 356
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/pip/_internal/locations/base.py`
**Line**: 15
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/pip/_internal/locations/base.py`
**Line**: 59
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/pip/_internal/metadata/base.py`
**Line**: 37
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/pip/_internal/metadata/base.py`
**Line**: 174
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/pip/_internal/metadata/base.py`
**Line**: 184
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/pip/_internal/models/installation_report.py`
**Line**: 50
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/pip/_internal/network/lazy_wheel.py`
**Line**: 174
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/pip/_internal/operations/prepare.py`
**Line**: 550
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/pip/_internal/operations/prepare.py`
**Line**: 624
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/pip/_internal/req/constructors.py`
**Line**: 290
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/pip/_internal/req/req_file.py`
**Line**: 247
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/pip/_internal/req/req_file.py`
**Line**: 491
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/pip/_internal/req/req_install.py`
**Line**: 287
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/pip/_internal/req/req_install.py`
**Line**: 375
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/pip/_internal/req/req_uninstall.py`
**Line**: 490
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/pip/_internal/utils/subprocess.py`
**Line**: 26
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/pip/_internal/utils/unpacking.py`
**Line**: 248
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/pip/_internal/utils/unpacking.py`
**Line**: 249
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/pip/_internal/vcs/subversion.py`
**Line**: 59
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/pip/_internal/vcs/versioncontrol.py`
**Line**: 45
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/pip/_internal/resolution/resolvelib/candidates.py`
**Line**: 346
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/pip/_internal/resolution/resolvelib/candidates.py`
**Line**: 442
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/pip/_internal/resolution/resolvelib/factory.py`
**Line**: 196
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/pip/_internal/resolution/resolvelib/factory.py`
**Line**: 611
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/pip/_internal/resolution/resolvelib/found_candidates.py`
**Line**: 33
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/pip/_internal/resolution/resolvelib/provider.py`
**Line**: 67
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/pip/_vendor/cachecontrol/controller.py`
**Line**: 220
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/pip/_vendor/cachecontrol/filewrapper.py`
**Line**: 67
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/pip/_vendor/chardet/codingstatemachinedict.py`
**Line**: 6
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/pip/_vendor/chardet/resultdict.py`
**Line**: 6
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/pip/_vendor/chardet/sbcharsetprober.py`
**Line**: 95
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/pip/_vendor/chardet/sbcsgroupprober.py`
**Line**: 57
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/pip/_vendor/chardet/sbcsgroupprober.py`
**Line**: 63
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/pip/_vendor/chardet/sbcsgroupprober.py`
**Line**: 78
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/pip/_vendor/chardet/universaldetector.py`
**Line**: 194
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/pip/_vendor/chardet/universaldetector.py`
**Line**: 202
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/pip/_vendor/distlib/database.py`
**Line**: 955
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/pip/_vendor/distlib/locators.py`
**Line**: 931
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/pip/_vendor/distlib/metadata.py`
**Line**: 255
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/pip/_vendor/distlib/metadata.py`
**Line**: 464
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/pip/_vendor/distlib/metadata.py`
**Line**: 585
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/pip/_vendor/distlib/metadata.py`
**Line**: 1018
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/pip/_vendor/distlib/util.py`
**Line**: 413
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/pip/_vendor/distlib/version.py`
**Line**: 516
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/pip/_vendor/distlib/wheel.py`
**Line**: 852
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/pip/_vendor/msgpack/fallback.py`
**Line**: 554
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/pip/_vendor/msgpack/fallback.py`
**Line**: 558
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/pip/_vendor/msgpack/fallback.py`
**Line**: 566
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/pip/_vendor/msgpack/fallback.py`
**Line**: 571
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/pip/_vendor/packaging/requirements.py`
**Line**: 95
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/pip/_vendor/packaging/requirements.py`
**Line**: 98
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/pip/_vendor/packaging/tags.py`
**Line**: 326
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/pip/_vendor/pkg_resources/__init__.py`
**Line**: 1866
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/pip/_vendor/pkg_resources/__init__.py`
**Line**: 2949
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/pip/_vendor/pkg_resources/__init__.py`
**Line**: 3051
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/pip/_vendor/pygments/cmdline.py`
**Line**: 209
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/pip/_vendor/pygments/regexopt.py`
**Line**: 29
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/pip/_vendor/pygments/regexopt.py`
**Line**: 31
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/pip/_vendor/pygments/regexopt.py`
**Line**: 35
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/pip/_vendor/pygments/regexopt.py`
**Line**: 38
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/pip/_vendor/pygments/regexopt.py`
**Line**: 52
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/pip/_vendor/pygments/regexopt.py`
**Line**: 55
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/pip/_vendor/pygments/regexopt.py`
**Line**: 61
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/pip/_vendor/pygments/regexopt.py`
**Line**: 70
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/pip/_vendor/pygments/regexopt.py`
**Line**: 75
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/pip/_vendor/pygments/unistring.py`
**Line**: 125
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/pip/_vendor/pyparsing/core.py`
**Line**: 812
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/pip/_vendor/pyparsing/core.py`
**Line**: 829
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/pip/_vendor/pyparsing/core.py`
**Line**: 875
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/pip/_vendor/pyparsing/core.py`
**Line**: 898
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/pip/_vendor/pyparsing/core.py`
**Line**: 3145
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/pip/_vendor/pyparsing/core.py`
**Line**: 4693
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/pip/_vendor/pyparsing/core.py`
**Line**: 4847
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/pip/_vendor/pyparsing/core.py`
**Line**: 5754
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/pip/_vendor/pyparsing/helpers.py`
**Line**: 56
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/pip/_vendor/requests/adapters.py`
**Line**: 505
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/pip/_vendor/requests/hooks.py`
**Line**: 19
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/pip/_vendor/rich/cells.py`
**Line**: 122
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/pip/_vendor/rich/cells.py`
**Line**: 123
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/pip/_vendor/rich/pretty.py`
**Line**: 988
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/pip/_vendor/rich/text.py`
**Line**: 542
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/pip/_vendor/truststore/_macos.py`
**Line**: 482
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/pip/_vendor/urllib3/connection.py`
**Line**: 199
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/pip/_vendor/urllib3/connectionpool.py`
**Line**: 520
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/pip/_vendor/urllib3/exceptions.py`
**Line**: 289
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/pip/_vendor/urllib3/response.py`
**Line**: 441
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/pip/_vendor/urllib3/response.py`
**Line**: 446
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/pip/_vendor/urllib3/response.py`
**Line**: 798
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/pip/_vendor/chardet/metadata/languages.py`
**Line**: 11
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/pip/_vendor/pygments/formatters/img.py`
**Line**: 511
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/pip/_vendor/pygments/formatters/img.py`
**Line**: 516
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/pip/_vendor/pygments/formatters/latex.py`
**Line**: 337
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/pip/_vendor/pygments/formatters/terminal256.py`
**Line**: 17
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/pip/_vendor/pygments/lexers/__init__.py`
**Line**: 208
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/pip/_vendor/urllib3/contrib/pyopenssl.py`
**Line**: 371
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/pip/_vendor/urllib3/contrib/securetransport.py`
**Line**: 660
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/pip/_vendor/urllib3/contrib/securetransport.py`
**Line**: 820
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/pip/_vendor/urllib3/contrib/securetransport.py`
**Line**: 830
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/pip/_vendor/urllib3/util/response.py`
**Line**: 103
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/pip/_vendor/urllib3/util/retry.py`
**Line**: 31
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/pip/_vendor/urllib3/util/retry.py`
**Line**: 261
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/pip/_vendor/urllib3/util/retry.py`
**Line**: 323
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/pip/_vendor/urllib3/util/retry.py`
**Line**: 454
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/pip/_vendor/urllib3/util/retry.py`
**Line**: 608
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/pip/_vendor/urllib3/util/url.py`
**Line**: 402
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/mpmath/calculus/extrapolation.py`
**Line**: 1779
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/mpmath/calculus/extrapolation.py`
**Line**: 1809
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/mpmath/calculus/extrapolation.py`
**Line**: 1833
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/mpmath/calculus/extrapolation.py`
**Line**: 1969
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/mpmath/calculus/optimization.py`
**Line**: 264
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/mpmath/calculus/optimization.py`
**Line**: 289
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/mpmath/calculus/optimization.py`
**Line**: 418
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/mpmath/calculus/optimization.py`
**Line**: 457
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/mpmath/calculus/optimization.py`
**Line**: 503
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/mpmath/calculus/optimization.py`
**Line**: 560
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/mpmath/calculus/optimization.py`
**Line**: 573
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/mpmath/calculus/optimization.py`
**Line**: 601
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/mpmath/calculus/optimization.py`
**Line**: 984
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/mpmath/functions/__init__.py`
**Line**: 2
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/mpmath/functions/bessel.py`
**Line**: 28
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/mpmath/functions/bessel.py`
**Line**: 147
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/mpmath/functions/bessel.py`
**Line**: 384
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/mpmath/functions/bessel.py`
**Line**: 423
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/mpmath/functions/bessel.py`
**Line**: 468
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/mpmath/functions/bessel.py`
**Line**: 521
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/mpmath/functions/bessel.py`
**Line**: 562
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/mpmath/functions/bessel.py`
**Line**: 718
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/mpmath/functions/bessel.py`
**Line**: 757
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/mpmath/functions/bessel.py`
**Line**: 878
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/mpmath/functions/expintegrals.py`
**Line**: 265
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/mpmath/functions/factorials.py`
**Line**: 112
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/mpmath/functions/factorials.py`
**Line**: 128
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/mpmath/functions/factorials.py`
**Line**: 133
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/mpmath/functions/functions.py`
**Line**: 174
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/mpmath/functions/functions.py`
**Line**: 182
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/mpmath/functions/functions.py`
**Line**: 212
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/mpmath/functions/functions.py`
**Line**: 257
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/mpmath/functions/functions.py`
**Line**: 358
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/mpmath/functions/functions.py`
**Line**: 562
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/mpmath/functions/functions.py`
**Line**: 604
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/mpmath/functions/hypergeometric.py`
**Line**: 284
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/mpmath/functions/hypergeometric.py`
**Line**: 285
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/mpmath/functions/hypergeometric.py`
**Line**: 404
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/mpmath/functions/hypergeometric.py`
**Line**: 409
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/mpmath/functions/hypergeometric.py`
**Line**: 524
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/mpmath/functions/hypergeometric.py`
**Line**: 760
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/mpmath/functions/hypergeometric.py`
**Line**: 763
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/mpmath/functions/hypergeometric.py`
**Line**: 786
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/mpmath/functions/hypergeometric.py`
**Line**: 790
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/mpmath/functions/hypergeometric.py`
**Line**: 832
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/mpmath/functions/hypergeometric.py`
**Line**: 835
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/mpmath/functions/hypergeometric.py`
**Line**: 864
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/mpmath/functions/hypergeometric.py`
**Line**: 868
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/mpmath/functions/hypergeometric.py`
**Line**: 890
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/mpmath/functions/hypergeometric.py`
**Line**: 914
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/mpmath/functions/hypergeometric.py`
**Line**: 1077
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/mpmath/functions/hypergeometric.py`
**Line**: 1084
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/mpmath/functions/hypergeometric.py`
**Line**: 1091
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/mpmath/functions/hypergeometric.py`
**Line**: 1107
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/mpmath/functions/orthogonal.py`
**Line**: 18
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/mpmath/functions/orthogonal.py`
**Line**: 313
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/mpmath/functions/orthogonal.py`
**Line**: 373
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/mpmath/functions/rszeta.py`
**Line**: 833
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/mpmath/functions/theta.py`
**Line**: 926
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/mpmath/functions/zeta.py`
**Line**: 238
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/mpmath/functions/zeta.py`
**Line**: 282
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/mpmath/functions/zeta.py`
**Line**: 287
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/mpmath/functions/zeta.py`
**Line**: 385
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/mpmath/functions/zeta.py`
**Line**: 548
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/mpmath/functions/zeta.py`
**Line**: 552
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/mpmath/functions/zeta.py`
**Line**: 620
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/mpmath/functions/zeta.py`
**Line**: 697
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/mpmath/functions/zeta.py`
**Line**: 737
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/mpmath/functions/zetazeros.py`
**Line**: 163
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/mpmath/functions/zetazeros.py`
**Line**: 167
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/mpmath/functions/zetazeros.py`
**Line**: 169
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/mpmath/libmp/gammazeta.py`
**Line**: 127
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/mpmath/libmp/gammazeta.py`
**Line**: 185
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/mpmath/libmp/gammazeta.py`
**Line**: 211
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/mpmath/libmp/gammazeta.py`
**Line**: 332
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/mpmath/libmp/gammazeta.py`
**Line**: 442
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/mpmath/libmp/gammazeta.py`
**Line**: 830
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/mpmath/libmp/gammazeta.py`
**Line**: 941
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/mpmath/libmp/gammazeta.py`
**Line**: 1160
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/mpmath/libmp/libelefun.py`
**Line**: 342
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/mpmath/libmp/libelefun.py`
**Line**: 711
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/mpmath/libmp/libelefun.py`
**Line**: 876
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/mpmath/libmp/libelefun.py`
**Line**: 1158
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/mpmath/libmp/libelefun.py`
**Line**: 1249
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/mpmath/libmp/libhyper.py`
**Line**: 74
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/mpmath/libmp/libhyper.py`
**Line**: 199
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/mpmath/libmp/libhyper.py`
**Line**: 306
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/mpmath/libmp/libhyper.py`
**Line**: 332
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/mpmath/libmp/libhyper.py`
**Line**: 355
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/mpmath/libmp/libhyper.py`
**Line**: 417
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/mpmath/libmp/libhyper.py`
**Line**: 475
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/mpmath/libmp/libhyper.py`
**Line**: 617
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/mpmath/libmp/libhyper.py`
**Line**: 911
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/mpmath/libmp/libhyper.py`
**Line**: 914
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/mpmath/libmp/libhyper.py`
**Line**: 1015
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/mpmath/libmp/libhyper.py`
**Line**: 1029
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/mpmath/libmp/libhyper.py`
**Line**: 1081
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/mpmath/libmp/libintmath.py`
**Line**: 129
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/mpmath/libmp/libmpc.py`
**Line**: 498
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/mpmath/libmp/libmpc.py`
**Line**: 584
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/mpmath/libmp/libmpf.py`
**Line**: 1175
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/mpmath/libmp/libmpi.py`
**Line**: 124
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/mpmath/libmp/libmpi.py`
**Line**: 367
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/mpmath/libmp/libmpi.py`
**Line**: 631
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/mpmath/libmp/libmpi.py`
**Line**: 643
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/mpmath/libmp/libmpi.py`
**Line**: 670
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/mpmath/libmp/libmpi.py`
**Line**: 759
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/mpmath/libmp/libmpi.py`
**Line**: 848
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/mpmath/libmp/libmpi.py`
**Line**: 885
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/mpmath/matrices/calculus.py`
**Line**: 3
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/mpmath/matrices/eigen.py`
**Line**: 8
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/mpmath/matrices/eigen_symmetric.py`
**Line**: 8
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/mpmath/matrices/linalg.py`
**Line**: 99
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/mpmath/matrices/linalg.py`
**Line**: 136
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/mpmath/matrices/linalg.py`
**Line**: 218
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/mpmath/matrices/linalg.py`
**Line**: 239
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/mpmath/matrices/linalg.py`
**Line**: 372
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/mpmath/matrices/matrices.py`
**Line**: 4
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/mpmath/tests/runtests.py`
**Line**: 57
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/mpmath/tests/test_functions2.py`
**Line**: 1215
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/mpmath/tests/test_gammazeta.py`
**Line**: 599
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/mpmath/tests/test_interval.py`
**Line**: 273
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/mpmath/tests/test_interval.py`
**Line**: 378
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/mpmath/tests/test_interval.py`
**Line**: 404
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/mpmath/tests/test_linalg.py`
**Line**: 1
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/mpmath/tests/test_linalg.py`
**Line**: 259
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/mpmath/tests/test_linalg.py`
**Line**: 306
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/mpmath/tests/test_linalg.py`
**Line**: 307
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/mpmath/tests/test_linalg.py`
**Line**: 308
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/mpmath/tests/test_linalg.py`
**Line**: 309
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/mpmath/tests/test_linalg.py`
**Line**: 313
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/mpmath/tests/test_linalg.py`
**Line**: 318
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/mpmath/tests/test_linalg.py`
**Line**: 322
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/mpmath/tests/test_linalg.py`
**Line**: 327
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/mpmath/tests/test_linalg.py`
**Line**: 331
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/mpmath/tests/test_matrices.py`
**Line**: 57
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/assumptions/refine.py`
**Line**: 53
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/assumptions/satask.py`
**Line**: 103
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/calculus/accumulationbounds.py`
**Line**: 688
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/calculus/util.py`
**Line**: 327
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/calculus/util.py`
**Line**: 328
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/calculus/util.py`
**Line**: 329
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/codegen/ast.py`
**Line**: 1713
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/codegen/rewriting.py`
**Line**: 330
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/combinatorics/coset_table.py`
**Line**: 985
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/combinatorics/fp_groups.py`
**Line**: 353
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/combinatorics/fp_groups.py`
**Line**: 870
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/combinatorics/fp_groups.py`
**Line**: 903
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/concrete/summations.py`
**Line**: 607
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/core/add.py`
**Line**: 292
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/core/expr.py`
**Line**: 3692
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/core/exprtools.py`
**Line**: 252
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/core/facts.py`
**Line**: 310
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/core/facts.py`
**Line**: 398
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/core/function.py`
**Line**: 212
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/core/function.py`
**Line**: 1416
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/core/function.py`
**Line**: 1673
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/core/function.py`
**Line**: 1675
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/core/mul.py`
**Line**: 435
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/core/mul.py`
**Line**: 1049
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/core/mul.py`
**Line**: 1190
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/core/mul.py`
**Line**: 1617
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/core/numbers.py`
**Line**: 165
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/core/numbers.py`
**Line**: 180
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/core/numbers.py`
**Line**: 278
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/core/numbers.py`
**Line**: 1456
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/core/numbers.py`
**Line**: 1885
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/core/symbol.py`
**Line**: 637
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/diffgeom/diffgeom.py`
**Line**: 26
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/diffgeom/diffgeom.py`
**Line**: 27
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/diffgeom/diffgeom.py`
**Line**: 28
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/diffgeom/diffgeom.py`
**Line**: 1101
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/diffgeom/diffgeom.py`
**Line**: 1309
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/diffgeom/diffgeom.py`
**Line**: 1438
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/diffgeom/diffgeom.py`
**Line**: 1439
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/diffgeom/diffgeom.py`
**Line**: 1594
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/diffgeom/diffgeom.py`
**Line**: 1892
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/diffgeom/diffgeom.py`
**Line**: 1919
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/diffgeom/diffgeom.py`
**Line**: 1920
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/diffgeom/diffgeom.py`
**Line**: 1965
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/diffgeom/diffgeom.py`
**Line**: 1966
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/geometry/ellipse.py`
**Line**: 666
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/geometry/ellipse.py`
**Line**: 919
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/geometry/ellipse.py`
**Line**: 927
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/geometry/ellipse.py`
**Line**: 1290
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/geometry/plane.py`
**Line**: 412
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/holonomic/holonomic.py`
**Line**: 732
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/holonomic/holonomic.py`
**Line**: 868
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/deltafunctions.py`
**Line**: 144
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/heurisch.py`
**Line**: 474
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/heurisch.py`
**Line**: 495
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/heurisch.py`
**Line**: 700
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/heurisch.py`
**Line**: 726
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/integrals.py`
**Line**: 437
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/integrals.py`
**Line**: 452
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/integrals.py`
**Line**: 462
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/intpoly.py`
**Line**: 977
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/intpoly.py`
**Line**: 978
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/laplace.py`
**Line**: 400
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/laplace.py`
**Line**: 466
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/laplace.py`
**Line**: 875
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/manualintegrate.py`
**Line**: 1587
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/manualintegrate.py`
**Line**: 2040
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/meijerint.py`
**Line**: 119
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/meijerint.py`
**Line**: 164
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/meijerint.py`
**Line**: 170
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/meijerint.py`
**Line**: 187
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/meijerint.py`
**Line**: 189
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/meijerint.py`
**Line**: 202
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/meijerint.py`
**Line**: 204
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/meijerint.py`
**Line**: 207
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/meijerint.py`
**Line**: 230
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/meijerint.py`
**Line**: 256
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/meijerint.py`
**Line**: 282
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/meijerint.py`
**Line**: 286
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/meijerint.py`
**Line**: 524
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/meijerint.py`
**Line**: 852
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/meijerint.py`
**Line**: 876
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/meijerint.py`
**Line**: 972
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/meijerint.py`
**Line**: 1445
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/prde.py`
**Line**: 114
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/prde.py`
**Line**: 158
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/prde.py`
**Line**: 822
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/prde.py`
**Line**: 878
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/prde.py`
**Line**: 921
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/prde.py`
**Line**: 952
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/prde.py`
**Line**: 957
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/prde.py`
**Line**: 1035
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/prde.py`
**Line**: 1160
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/prde.py`
**Line**: 1185
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/prde.py`
**Line**: 1266
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/prde.py`
**Line**: 1301
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/rde.py`
**Line**: 38
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/rde.py`
**Line**: 205
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/rde.py`
**Line**: 286
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/rde.py`
**Line**: 510
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/rde.py`
**Line**: 653
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/rde.py`
**Line**: 713
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/risch.py`
**Line**: 333
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/risch.py`
**Line**: 366
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/risch.py`
**Line**: 395
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/risch.py`
**Line**: 497
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/risch.py`
**Line**: 530
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/risch.py`
**Line**: 779
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/risch.py`
**Line**: 781
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/risch.py`
**Line**: 798
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/risch.py`
**Line**: 855
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/risch.py`
**Line**: 1006
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/risch.py`
**Line**: 1007
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/risch.py`
**Line**: 1250
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/risch.py`
**Line**: 1280
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/risch.py`
**Line**: 1350
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/risch.py`
**Line**: 1365
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/risch.py`
**Line**: 1458
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/risch.py`
**Line**: 1621
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/risch.py`
**Line**: 1622
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/risch.py`
**Line**: 1691
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/transforms.py`
**Line**: 201
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/transforms.py`
**Line**: 745
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/trigonometry.py`
**Line**: 5
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/ntheory/residue_ntheory.py`
**Line**: 1539
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/physics/paulialgebra.py`
**Line**: 144
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/physics/secondquant.py`
**Line**: 2758
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/plotting/experimental_lambdify.py`
**Line**: 78
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/plotting/experimental_lambdify.py`
**Line**: 403
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/plotting/plot.py`
**Line**: 128
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/plotting/plot.py`
**Line**: 204
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/plotting/plot.py`
**Line**: 205
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/plotting/plot.py`
**Line**: 206
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/plotting/series.py`
**Line**: 381
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/plotting/series.py`
**Line**: 1146
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/plotting/series.py`
**Line**: 1156
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/plotting/series.py`
**Line**: 1774
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/plotting/series.py`
**Line**: 1790
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/plotting/series.py`
**Line**: 1871
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/plotting/series.py`
**Line**: 1972
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/plotting/series.py`
**Line**: 2094
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/plotting/utils.py`
**Line**: 159
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/polys/distributedmodules.py`
**Line**: 515
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/polys/distributedmodules.py`
**Line**: 675
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/polys/distributedmodules.py`
**Line**: 683
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/polys/groebnertools.py`
**Line**: 688
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/polys/heuristicgcd.py`
**Line**: 124
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/polys/modulargcd.py`
**Line**: 796
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/polys/modulargcd.py`
**Line**: 2129
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/polys/partfrac.py`
**Line**: 487
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/polys/polyroots.py`
**Line**: 761
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/polys/polyutils.py`
**Line**: 378
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/polys/ring_series.py`
**Line**: 1977
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/polys/rings.py`
**Line**: 163
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/polys/rings.py`
**Line**: 467
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/polys/rings.py`
**Line**: 1251
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/polys/rings.py`
**Line**: 2275
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/polys/rings.py`
**Line**: 3044
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/printing/codeprinter.py`
**Line**: 389
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/printing/codeprinter.py`
**Line**: 399
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/printing/glsl.py`
**Line**: 162
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/printing/julia.py`
**Line**: 118
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/printing/julia.py`
**Line**: 283
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/printing/latex.py`
**Line**: 238
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/printing/latex.py`
**Line**: 1176
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/printing/latex.py`
**Line**: 1981
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/printing/latex.py`
**Line**: 2585
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/printing/latex.py`
**Line**: 2777
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/printing/latex.py`
**Line**: 2785
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/printing/latex.py`
**Line**: 2802
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/printing/latex.py`
**Line**: 2850
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/printing/llvmjitcode.py`
**Line**: 95
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/printing/mathml.py`
**Line**: 946
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/printing/mathml.py`
**Line**: 949
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/printing/octave.py`
**Line**: 136
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/printing/octave.py`
**Line**: 258
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/printing/octave.py`
**Line**: 284
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/printing/printer.py`
**Line**: 121
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/printing/rcode.py`
**Line**: 194
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/printing/rcode.py`
**Line**: 204
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/printing/repr.py`
**Line**: 228
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/printing/rust.py`
**Line**: 558
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/printing/smtlib.py`
**Line**: 189
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/printing/str.py`
**Line**: 483
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/printing/str.py`
**Line**: 962
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/printing/tensorflow.py`
**Line**: 107
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/printing/tensorflow.py`
**Line**: 202
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/series/formal.py`
**Line**: 149
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/series/gruntz.py`
**Line**: 633
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/sets/fancysets.py`
**Line**: 1424
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/sets/setexpr.py`
**Line**: 84
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/sets/sets.py`
**Line**: 2555
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/sets/sets.py`
**Line**: 2558
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/simplify/gammasimp.py`
**Line**: 393
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/simplify/hyperexpand.py`
**Line**: 50
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/simplify/hyperexpand.py`
**Line**: 86
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/simplify/hyperexpand.py`
**Line**: 253
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/simplify/hyperexpand.py`
**Line**: 736
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/simplify/hyperexpand.py`
**Line**: 1947
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/simplify/hyperexpand.py`
**Line**: 1991
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/simplify/hyperexpand.py`
**Line**: 2116
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/simplify/hyperexpand.py`
**Line**: 2245
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/simplify/hyperexpand.py`
**Line**: 2250
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/simplify/hyperexpand.py`
**Line**: 2443
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/simplify/simplify.py`
**Line**: 646
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/simplify/simplify.py`
**Line**: 1087
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/simplify/simplify.py`
**Line**: 1247
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/simplify/trigsimp.py`
**Line**: 121
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/solvers/bivariate.py`
**Line**: 34
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/solvers/pde.py`
**Line**: 175
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/solvers/pde.py`
**Line**: 284
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/solvers/pde.py`
**Line**: 521
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/solvers/pde.py`
**Line**: 611
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/solvers/pde.py`
**Line**: 937
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/solvers/recurr.py`
**Line**: 564
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/solvers/solvers.py`
**Line**: 341
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/solvers/solvers.py`
**Line**: 359
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/solvers/solvers.py`
**Line**: 2902
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/solvers/solveset.py`
**Line**: 694
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/solvers/solveset.py`
**Line**: 966
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/solvers/solveset.py`
**Line**: 972
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/solvers/solveset.py`
**Line**: 1662
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/stats/drv.py`
**Line**: 152
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/stats/frv.py`
**Line**: 460
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/stats/joint_rv.py`
**Line**: 415
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/stats/joint_rv_types.py`
**Line**: 134
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/stats/matrix_distributions.py`
**Line**: 114
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/stats/random_matrix_models.py`
**Line**: 249
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/stats/rv.py`
**Line**: 1323
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/stats/rv.py`
**Line**: 1340
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/stats/rv.py`
**Line**: 1366
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/stats/rv.py`
**Line**: 1383
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/stats/rv.py`
**Line**: 1610
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/stats/rv.py`
**Line**: 1611
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/tensor/index_methods.py`
**Line**: 150
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/tensor/index_methods.py`
**Line**: 196
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/tensor/index_methods.py`
**Line**: 279
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/tensor/index_methods.py`
**Line**: 449
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/tensor/indexed.py`
**Line**: 89
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/tensor/tensor.py`
**Line**: 2208
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/tensor/tensor.py`
**Line**: 2590
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/tensor/tensor.py`
**Line**: 3064
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/tensor/tensor.py`
**Line**: 3203
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/tensor/tensor.py`
**Line**: 3209
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/tensor/tensor.py`
**Line**: 3225
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/tensor/tensor.py`
**Line**: 3268
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/tensor/tensor.py`
**Line**: 3344
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/tensor/tensor.py`
**Line**: 3359
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/tensor/tensor.py`
**Line**: 3661
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/tensor/tensor.py`
**Line**: 3662
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/tensor/tensor.py`
**Line**: 4540
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/tensor/tensor.py`
**Line**: 4563
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/tensor/tensor.py`
**Line**: 5136
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/testing/runtests.py`
**Line**: 64
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/testing/runtests.py`
**Line**: 684
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/testing/runtests.py`
**Line**: 1936
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/testing/runtests.py`
**Line**: 2023
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/utilities/codegen.py`
**Line**: 249
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/utilities/codegen.py`
**Line**: 1530
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/utilities/enumerative.py`
**Line**: 447
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/utilities/misc.py`
**Line**: 220
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/vector/coordsysrect.py`
**Line**: 702
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/vector/functions.py`
**Line**: 158
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/vector/functions.py`
**Line**: 503
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/vector/operators.py`
**Line**: 214
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/assumptions/handlers/matrices.py`
**Line**: 47
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/assumptions/handlers/matrices.py`
**Line**: 77
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/assumptions/handlers/matrices.py`
**Line**: 94
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/assumptions/handlers/order.py`
**Line**: 218
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/assumptions/predicates/calculus.py`
**Line**: 57
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/assumptions/predicates/common.py`
**Line**: 17
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/assumptions/predicates/matrices.py`
**Line**: 70
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/assumptions/predicates/sets.py`
**Line**: 238
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/assumptions/predicates/sets.py`
**Line**: 337
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/assumptions/predicates/sets.py`
**Line**: 394
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/codegen/tests/test_rewriting.py`
**Line**: 442
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/concrete/tests/test_sums_products.py`
**Line**: 1043
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/core/tests/test_arit.py`
**Line**: 1182
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/core/tests/test_arit.py`
**Line**: 1198
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/core/tests/test_arit.py`
**Line**: 2171
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/core/tests/test_assumptions.py`
**Line**: 410
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/core/tests/test_basic.py`
**Line**: 208
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/core/tests/test_diff.py`
**Line**: 138
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/core/tests/test_expr.py`
**Line**: 917
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/core/tests/test_facts.py`
**Line**: 70
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/core/tests/test_function.py`
**Line**: 1447
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/core/tests/test_relational.py`
**Line**: 617
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/core/tests/test_relational.py`
**Line**: 643
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/core/tests/test_relational.py`
**Line**: 925
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/diffgeom/tests/test_class_structure.py`
**Line**: 19
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/diffgeom/tests/test_class_structure.py`
**Line**: 20
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/diffgeom/tests/test_diffgeom.py`
**Line**: 103
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/diffgeom/tests/test_diffgeom.py`
**Line**: 117
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/diffgeom/tests/test_diffgeom.py`
**Line**: 120
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/diffgeom/tests/test_diffgeom.py`
**Line**: 123
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/diffgeom/tests/test_diffgeom.py`
**Line**: 128
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/diffgeom/tests/test_diffgeom.py`
**Line**: 131
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/diffgeom/tests/test_diffgeom.py`
**Line**: 134
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/diffgeom/tests/test_hyperbolic_space.py`
**Line**: 86
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/functions/combinatorial/factorials.py`
**Line**: 423
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/functions/combinatorial/numbers.py`
**Line**: 2758
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/functions/elementary/exponential.py`
**Line**: 985
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/functions/elementary/hyperbolic.py`
**Line**: 283
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/functions/elementary/hyperbolic.py`
**Line**: 480
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/functions/elementary/piecewise.py`
**Line**: 456
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/functions/elementary/piecewise.py`
**Line**: 578
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/functions/elementary/trigonometric.py`
**Line**: 498
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/functions/elementary/trigonometric.py`
**Line**: 499
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/functions/elementary/trigonometric.py`
**Line**: 867
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/functions/elementary/trigonometric.py`
**Line**: 1226
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/functions/elementary/trigonometric.py`
**Line**: 1579
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/functions/special/bessel.py`
**Line**: 25
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/functions/special/delta_functions.py`
**Line**: 656
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/functions/special/error_functions.py`
**Line**: 24
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/functions/special/error_functions.py`
**Line**: 25
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/functions/special/error_functions.py`
**Line**: 1220
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/functions/special/error_functions.py`
**Line**: 2738
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/functions/special/gamma_functions.py`
**Line**: 687
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/functions/special/hyper.py`
**Line**: 47
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/functions/special/hyper.py`
**Line**: 48
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/functions/special/hyper.py`
**Line**: 210
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/functions/special/hyper.py`
**Line**: 543
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/functions/special/hyper.py`
**Line**: 980
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/functions/special/spherical_harmonics.py`
**Line**: 150
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/functions/special/spherical_harmonics.py`
**Line**: 179
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/functions/special/spherical_harmonics.py`
**Line**: 180
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/functions/special/spherical_harmonics.py`
**Line**: 189
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/functions/special/spherical_harmonics.py`
**Line**: 190
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/functions/special/spherical_harmonics.py`
**Line**: 197
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/functions/special/spherical_harmonics.py`
**Line**: 202
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/functions/special/zeta_functions.py`
**Line**: 150
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/functions/special/zeta_functions.py`
**Line**: 179
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/functions/special/zeta_functions.py`
**Line**: 181
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/functions/elementary/tests/test_complexes.py`
**Line**: 938
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/functions/elementary/tests/test_piecewise.py`
**Line**: 1223
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/functions/special/tests/test_bessel.py`
**Line**: 514
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/functions/special/tests/test_delta_functions.py`
**Line**: 37
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/tests/test_heurisch.py`
**Line**: 221
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/tests/test_heurisch.py`
**Line**: 254
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/tests/test_heurisch.py`
**Line**: 343
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/tests/test_integrals.py`
**Line**: 328
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/tests/test_integrals.py`
**Line**: 1140
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/tests/test_integrals.py`
**Line**: 1329
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/tests/test_laplace.py`
**Line**: 110
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/tests/test_laplace.py`
**Line**: 698
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/tests/test_laplace.py`
**Line**: 699
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/tests/test_laplace.py`
**Line**: 714
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/tests/test_laplace.py`
**Line**: 756
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/tests/test_manual.py`
**Line**: 113
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/tests/test_meijerint.py`
**Line**: 149
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/tests/test_meijerint.py`
**Line**: 165
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/tests/test_meijerint.py`
**Line**: 245
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/tests/test_meijerint.py`
**Line**: 257
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/tests/test_meijerint.py`
**Line**: 262
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/tests/test_meijerint.py`
**Line**: 273
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/tests/test_meijerint.py`
**Line**: 274
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/tests/test_meijerint.py`
**Line**: 275
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/tests/test_meijerint.py`
**Line**: 276
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/tests/test_meijerint.py`
**Line**: 277
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/tests/test_meijerint.py`
**Line**: 278
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/tests/test_meijerint.py`
**Line**: 374
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/tests/test_meijerint.py`
**Line**: 517
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/tests/test_meijerint.py`
**Line**: 523
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/tests/test_meijerint.py`
**Line**: 544
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/tests/test_meijerint.py`
**Line**: 579
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/tests/test_meijerint.py`
**Line**: 658
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/tests/test_meijerint.py`
**Line**: 668
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/tests/test_prde.py`
**Line**: 105
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/tests/test_prde.py`
**Line**: 147
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/tests/test_prde.py`
**Line**: 262
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/tests/test_prde.py`
**Line**: 280
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/tests/test_rde.py`
**Line**: 71
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/tests/test_rde.py`
**Line**: 114
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/tests/test_rde.py`
**Line**: 115
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/tests/test_rde.py`
**Line**: 118
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/tests/test_rde.py`
**Line**: 119
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/tests/test_rde.py`
**Line**: 179
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/tests/test_rde.py`
**Line**: 193
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/tests/test_rde.py`
**Line**: 197
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/tests/test_risch.py`
**Line**: 235
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/tests/test_risch.py`
**Line**: 250
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/tests/test_risch.py`
**Line**: 374
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/tests/test_transforms.py`
**Line**: 80
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/tests/test_transforms.py`
**Line**: 164
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/tests/test_transforms.py`
**Line**: 244
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/tests/test_transforms.py`
**Line**: 253
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/tests/test_transforms.py`
**Line**: 264
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/tests/test_transforms.py`
**Line**: 271
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/tests/test_transforms.py`
**Line**: 272
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/tests/test_transforms.py`
**Line**: 386
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/tests/test_transforms.py`
**Line**: 420
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/tests/test_transforms.py`
**Line**: 436
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/tests/test_transforms.py`
**Line**: 444
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/tests/test_transforms.py`
**Line**: 466
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/tests/test_transforms.py`
**Line**: 479
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/tests/test_transforms.py`
**Line**: 481
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/tests/test_transforms.py`
**Line**: 492
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/tests/test_transforms.py`
**Line**: 502
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/tests/test_transforms.py`
**Line**: 504
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/tests/test_transforms.py`
**Line**: 506
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/integrals/tests/test_trigonometry.py`
**Line**: 32
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/interactive/tests/test_ipython.py`
**Line**: 10
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/interactive/tests/test_ipython.py`
**Line**: 74
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/interactive/tests/test_ipython.py`
**Line**: 155
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/matrices/expressions/hadamard.py`
**Line**: 165
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/matrices/tests/test_commonmatrix.py`
**Line**: 1167
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/matrices/expressions/tests/test_derivatives.py`
**Line**: 52
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/matrices/expressions/tests/test_derivatives.py`
**Line**: 105
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/matrices/expressions/tests/test_derivatives.py`
**Line**: 225
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/matrices/expressions/tests/test_derivatives.py`
**Line**: 292
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/matrices/expressions/tests/test_derivatives.py`
**Line**: 407
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/matrices/expressions/tests/test_derivatives.py`
**Line**: 416
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/matrices/expressions/tests/test_derivatives.py`
**Line**: 435
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/matrices/expressions/tests/test_derivatives.py`
**Line**: 444
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/matrices/expressions/tests/test_derivatives.py`
**Line**: 448
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/parsing/autolev/_listener_autolev_antlr.py`
**Line**: 361
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/parsing/autolev/_listener_autolev_antlr.py`
**Line**: 362
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/parsing/autolev/_listener_autolev_antlr.py`
**Line**: 796
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/parsing/autolev/_listener_autolev_antlr.py`
**Line**: 1269
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/parsing/autolev/_listener_autolev_antlr.py`
**Line**: 1636
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/parsing/c/c_parser.py`
**Line**: 520
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/parsing/fortran/fortran_parser.py`
**Line**: 102
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/parsing/fortran/fortran_parser.py`
**Line**: 151
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/parsing/fortran/fortran_parser.py`
**Line**: 239
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/parsing/fortran/fortran_parser.py`
**Line**: 257
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/parsing/latex/lark/latex_parser.py`
**Line**: 84
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/parsing/latex/lark/latex_parser.py`
**Line**: 86
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/parsing/latex/lark/transformer.py`
**Line**: 444
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/parsing/latex/lark/transformer.py`
**Line**: 450
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/parsing/latex/lark/transformer.py`
**Line**: 660
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/physics/hep/gamma_matrices.py`
**Line**: 315
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/physics/mechanics/kane.py`
**Line**: 630
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/physics/optics/gaussopt.py`
**Line**: 526
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/physics/optics/gaussopt.py`
**Line**: 885
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/physics/optics/gaussopt.py`
**Line**: 906
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/physics/optics/gaussopt.py`
**Line**: 911
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/physics/quantum/cg.py`
**Line**: 1
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/physics/quantum/cg.py`
**Line**: 486
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/physics/quantum/cg.py`
**Line**: 675
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/physics/quantum/gate.py`
**Line**: 242
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/physics/quantum/matrixcache.py`
**Line**: 78
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/physics/quantum/matrixutils.py`
**Line**: 141
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/physics/quantum/operator.py`
**Line**: 428
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/physics/quantum/operator.py`
**Line**: 491
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/physics/quantum/qapply.py`
**Line**: 104
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/physics/quantum/qapply.py`
**Line**: 251
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/physics/quantum/represent.py`
**Line**: 417
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/physics/quantum/spin.py`
**Line**: 145
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/physics/quantum/spin.py`
**Line**: 157
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/physics/quantum/spin.py`
**Line**: 165
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/physics/quantum/spin.py`
**Line**: 616
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/physics/quantum/spin.py`
**Line**: 1017
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/physics/quantum/spin.py`
**Line**: 1487
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/physics/quantum/tensorproduct.py`
**Line**: 151
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/physics/quantum/trace.py`
**Line**: 94
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/physics/quantum/trace.py`
**Line**: 172
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/physics/quantum/trace.py`
**Line**: 189
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/physics/quantum/trace.py`
**Line**: 192
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/physics/units/dimensions.py`
**Line**: 351
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/physics/units/dimensions.py`
**Line**: 522
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/physics/units/quantities.py`
**Line**: 38
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/physics/vector/vector.py`
**Line**: 735
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/physics/mechanics/tests/test_particle.py`
**Line**: 55
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/physics/quantum/tests/test_cartesian.py`
**Line**: 113
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/physics/quantum/tests/test_density.py`
**Line**: 269
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/physics/quantum/tests/test_printing.py`
**Line**: 678
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/physics/quantum/tests/test_trace.py`
**Line**: 82
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/physics/units/tests/test_quantities.py`
**Line**: 206
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/physics/units/tests/test_quantities.py`
**Line**: 227
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/physics/units/tests/test_quantities.py`
**Line**: 244
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/physics/units/tests/test_quantities.py`
**Line**: 249
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/physics/units/tests/test_quantities.py`
**Line**: 253
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/physics/vector/tests/test_functions.py`
**Line**: 302
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/physics/vector/tests/test_printing.py`
**Line**: 47
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/physics/vector/tests/test_printing.py`
**Line**: 48
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/physics/vector/tests/test_printing.py`
**Line**: 50
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/plotting/backends/matplotlibbackend/matplotlib.py`
**Line**: 240
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/plotting/backends/matplotlibbackend/matplotlib.py`
**Line**: 304
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/polys/agca/ideals.py`
**Line**: 102
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/polys/agca/modules.py`
**Line**: 32
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/polys/agca/modules.py`
**Line**: 1281
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/polys/domains/domain.py`
**Line**: 450
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/polys/domains/domain.py`
**Line**: 459
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/polys/domains/fractionfield.py`
**Line**: 34
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/polys/domains/polynomialring.py`
**Line**: 40
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/polys/domains/quotientring.py`
**Line**: 9
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/polys/domains/quotientring.py`
**Line**: 142
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/polys/matrices/_dfm.py`
**Line**: 23
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/polys/matrices/_dfm.py`
**Line**: 660
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/polys/matrices/dense.py`
**Line**: 173
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/polys/matrices/dense.py`
**Line**: 321
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/polys/matrices/normalforms.py`
**Line**: 11
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/polys/matrices/rref.py`
**Line**: 256
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/polys/numberfields/basis.py`
**Line**: 216
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/polys/numberfields/modules.py`
**Line**: 1839
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/polys/numberfields/primes.py`
**Line**: 678
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/polys/tests/test_distributedmodules.py`
**Line**: 50
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/polys/tests/test_heuristicgcd.py`
**Line**: 53
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/printing/pretty/pretty.py`
**Line**: 155
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/printing/pretty/pretty.py`
**Line**: 1375
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/printing/pretty/pretty.py`
**Line**: 1442
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/printing/pretty/pretty.py`
**Line**: 1490
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/printing/pretty/pretty.py`
**Line**: 1574
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/printing/pretty/pretty.py`
**Line**: 1910
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/printing/pretty/pretty.py`
**Line**: 2239
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/printing/pretty/pretty.py`
**Line**: 2602
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/printing/pretty/pretty.py`
**Line**: 2671
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/printing/pretty/pretty.py`
**Line**: 2808
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/printing/pretty/pretty_symbology.py`
**Line**: 48
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/printing/pretty/pretty_symbology.py`
**Line**: 200
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/printing/pretty/pretty_symbology.py`
**Line**: 333
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/printing/tests/test_aesaracode.py`
**Line**: 182
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/printing/tests/test_aesaracode.py`
**Line**: 236
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/printing/tests/test_aesaracode.py`
**Line**: 573
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/printing/tests/test_aesaracode.py`
**Line**: 611
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/printing/tests/test_julia.py`
**Line**: 184
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/printing/tests/test_julia.py`
**Line**: 293
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/printing/tests/test_latex.py`
**Line**: 2079
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/printing/tests/test_octave.py`
**Line**: 243
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/printing/tests/test_octave.py`
**Line**: 358
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/printing/tests/test_repr.py`
**Line**: 93
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/printing/tests/test_repr.py`
**Line**: 339
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/printing/tests/test_smtlib.py`
**Line**: 216
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/printing/tests/test_smtlib.py`
**Line**: 457
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/printing/tests/test_smtlib.py`
**Line**: 509
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/printing/tests/test_theanocode.py`
**Line**: 172
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/printing/tests/test_theanocode.py`
**Line**: 226
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/printing/tests/test_theanocode.py`
**Line**: 561
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/printing/tests/test_theanocode.py`
**Line**: 599
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/printing/pretty/tests/test_pretty.py`
**Line**: 4575
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/printing/pretty/tests/test_pretty.py`
**Line**: 7256
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/printing/pretty/tests/test_pretty.py`
**Line**: 7620
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/series/tests/test_formal.py`
**Line**: 441
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/series/tests/test_gruntz.py`
**Line**: 148
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/series/tests/test_gruntz.py`
**Line**: 152
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/series/tests/test_order.py`
**Line**: 162
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/sets/handlers/functions.py`
**Line**: 38
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/sets/handlers/functions.py`
**Line**: 39
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/sets/handlers/intersection.py`
**Line**: 371
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/sets/handlers/mul.py`
**Line**: 34
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/sets/handlers/mul.py`
**Line**: 41
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/sets/handlers/power.py`
**Line**: 46
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/sets/handlers/power.py`
**Line**: 49
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/sets/handlers/power.py`
**Line**: 85
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/sets/tests/test_fancysets.py`
**Line**: 152
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/sets/tests/test_setexpr.py`
**Line**: 29
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/sets/tests/test_setexpr.py`
**Line**: 206
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/simplify/tests/test_hyperexpand.py`
**Line**: 116
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/simplify/tests/test_hyperexpand.py`
**Line**: 130
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/simplify/tests/test_hyperexpand.py`
**Line**: 131
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/simplify/tests/test_hyperexpand.py`
**Line**: 135
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/simplify/tests/test_hyperexpand.py`
**Line**: 949
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/simplify/tests/test_hyperexpand.py`
**Line**: 1011
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/simplify/tests/test_hyperexpand.py`
**Line**: 1039
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/solvers/diophantine/diophantine.py`
**Line**: 2566
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/solvers/diophantine/diophantine.py`
**Line**: 3893
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/solvers/ode/ode.py`
**Line**: 798
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/solvers/ode/ode.py`
**Line**: 1072
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/solvers/ode/ode.py`
**Line**: 1783
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/solvers/ode/single.py`
**Line**: 204
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/solvers/tests/test_polysys.py`
**Line**: 185
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/solvers/tests/test_solvers.py`
**Line**: 1727
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/solvers/tests/test_solveset.py`
**Line**: 323
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/solvers/tests/test_solveset.py`
**Line**: 1862
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/solvers/ode/tests/test_ode.py`
**Line**: 911
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/solvers/ode/tests/test_ode.py`
**Line**: 929
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/solvers/ode/tests/test_ode.py`
**Line**: 948
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/solvers/ode/tests/test_ode.py`
**Line**: 956
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/solvers/ode/tests/test_single.py`
**Line**: 313
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/solvers/ode/tests/test_systems.py`
**Line**: 2480
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/solvers/ode/tests/test_systems.py`
**Line**: 2485
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/solvers/ode/tests/test_systems.py`
**Line**: 2516
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/solvers/ode/tests/test_systems.py`
**Line**: 2524
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/stats/tests/test_compound_rv.py`
**Line**: 90
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/stats/tests/test_continuous_rv.py`
**Line**: 677
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/stats/tests/test_continuous_rv.py`
**Line**: 1290
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/stats/tests/test_continuous_rv.py`
**Line**: 1348
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/stats/tests/test_continuous_rv.py`
**Line**: 1358
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/stats/tests/test_finite_rv.py`
**Line**: 68
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/stats/tests/test_mix.py`
**Line**: 80
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/stats/tests/test_stochastic_process.py`
**Line**: 77
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/stats/tests/test_stochastic_process.py`
**Line**: 109
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/stats/tests/test_stochastic_process.py`
**Line**: 408
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/tensor/array/array_derivatives.py`
**Line**: 91
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/tensor/array/ndim_array.py`
**Line**: 567
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/tensor/tests/test_tensor.py`
**Line**: 497
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/tensor/array/expressions/array_expressions.py`
**Line**: 605
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/tensor/array/expressions/array_expressions.py`
**Line**: 606
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/tensor/array/expressions/array_expressions.py`
**Line**: 641
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/tensor/array/expressions/array_expressions.py`
**Line**: 842
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/tensor/array/expressions/array_expressions.py`
**Line**: 1263
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/tensor/array/expressions/array_expressions.py`
**Line**: 1390
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/tensor/array/expressions/from_array_to_matrix.py`
**Line**: 124
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/tensor/array/expressions/from_array_to_matrix.py`
**Line**: 297
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/tensor/array/expressions/from_array_to_matrix.py`
**Line**: 448
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/tensor/array/expressions/from_array_to_matrix.py`
**Line**: 542
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/tensor/array/expressions/from_indexed_to_array.py`
**Line**: 113
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/tensor/array/expressions/tests/test_array_expressions.py`
**Line**: 50
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/tensor/array/expressions/tests/test_array_expressions.py`
**Line**: 54
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/tensor/array/expressions/tests/test_array_expressions.py`
**Line**: 445
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/tensor/array/expressions/tests/test_convert_array_to_matrix.py`
**Line**: 189
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/utilities/tests/test_codegen.py`
**Line**: 16
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/utilities/tests/test_codegen_julia.py`
**Line**: 85
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/utilities/tests/test_codegen_julia.py`
**Line**: 235
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/utilities/tests/test_codegen_octave.py`
**Line**: 83
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/utilities/tests/test_codegen_octave.py`
**Line**: 225
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/utilities/tests/test_codegen_rust.py`
**Line**: 89
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/utilities/tests/test_codegen_rust.py`
**Line**: 248
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/utilities/tests/test_misc.py`
**Line**: 64
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/utilities/tests/test_pickling.py`
**Line**: 382
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/utilities/tests/test_pickling.py`
**Line**: 409
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/utilities/tests/test_pickling.py`
**Line**: 420
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/utilities/tests/test_pickling.py`
**Line**: 424
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/utilities/tests/test_pickling.py`
**Line**: 438
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/utilities/tests/test_pickling.py`
**Line**: 444
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/utilities/tests/test_pickling.py`
**Line**: 448
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/utilities/tests/test_pickling.py`
**Line**: 457
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/utilities/tests/test_pickling.py`
**Line**: 472
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/utilities/tests/test_pickling.py`
**Line**: 489
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/utilities/tests/test_pickling.py`
**Line**: 493
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/utilities/tests/test_pickling.py`
**Line**: 500
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/utilities/tests/test_pickling.py`
**Line**: 504
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/utilities/tests/test_pickling.py`
**Line**: 526
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/utilities/tests/test_pickling.py`
**Line**: 559
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/utilities/tests/test_pickling.py`
**Line**: 563
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/utilities/tests/test_pickling.py`
**Line**: 612
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/utilities/tests/test_pickling.py`
**Line**: 622
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/utilities/tests/test_pickling.py`
**Line**: 635
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/utilities/tests/test_pickling.py`
**Line**: 639
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/utilities/tests/test_wester.py`
**Line**: 920
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/utilities/tests/test_wester.py`
**Line**: 956
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/utilities/tests/test_wester.py`
**Line**: 966
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/utilities/tests/test_wester.py`
**Line**: 967
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/utilities/tests/test_wester.py`
**Line**: 1014
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/utilities/tests/test_wester.py`
**Line**: 1032
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/utilities/tests/test_wester.py`
**Line**: 1038
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/utilities/tests/test_wester.py`
**Line**: 1047
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/utilities/tests/test_wester.py`
**Line**: 1052
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/utilities/tests/test_wester.py`
**Line**: 1059
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/utilities/tests/test_wester.py`
**Line**: 1077
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/utilities/tests/test_wester.py`
**Line**: 1084
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/utilities/tests/test_wester.py`
**Line**: 1092
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/utilities/tests/test_wester.py`
**Line**: 1099
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/utilities/tests/test_wester.py`
**Line**: 1118
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/utilities/tests/test_wester.py`
**Line**: 1194
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/utilities/tests/test_wester.py`
**Line**: 2252
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/sympy/utilities/tests/test_wester.py`
**Line**: 3082
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/setuptools/_distutils/core.py`
**Line**: 285
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/setuptools/_vendor/typing_extensions.py`
**Line**: 659
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/setuptools/_vendor/typing_extensions.py`
**Line**: 948
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/setuptools/_vendor/typing_extensions.py`
**Line**: 1756
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/setuptools/_vendor/typing_extensions.py`
**Line**: 1785
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/setuptools/_vendor/typing_extensions.py`
**Line**: 3123
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/setuptools/command/bdist_egg.py`
**Line**: 127
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/setuptools/command/bdist_egg.py`
**Line**: 164
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/setuptools/command/bdist_wheel.py`
**Line**: 72
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/setuptools/command/bdist_wheel.py`
**Line**: 327
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/setuptools/command/build_ext.py`
**Line**: 247
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/setuptools/command/dist_info.py`
**Line**: 101
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/setuptools/command/editable_wheel.py`
**Line**: 62
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/setuptools/command/editable_wheel.py`
**Line**: 83
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/setuptools/command/editable_wheel.py`
**Line**: 290
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/setuptools/command/editable_wheel.py`
**Line**: 323
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/setuptools/command/editable_wheel.py`
**Line**: 584
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/setuptools/command/egg_info.py`
**Line**: 38
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/setuptools/command/install_lib.py`
**Line**: 62
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/setuptools/config/_apply_pyprojecttoml.py`
**Line**: 344
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/setuptools/config/_apply_pyprojecttoml.py`
**Line**: 520
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/setuptools/config/setupcfg.py`
**Line**: 104
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/setuptools/config/setupcfg.py`
**Line**: 651
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/setuptools/config/setupcfg.py`
**Line**: 770
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/setuptools/tests/fixtures.py`
**Line**: 137
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/setuptools/tests/test_bdist_wheel.py`
**Line**: 668
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/setuptools/tests/test_build_py.py`
**Line**: 169
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/setuptools/tests/test_build_py.py`
**Line**: 199
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/setuptools/tests/test_core_metadata.py`
**Line**: 596
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/setuptools/tests/test_editable_install.py`
**Line**: 1039
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/setuptools/tests/test_editable_install.py`
**Line**: 1099
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/setuptools/tests/test_egg_info.py`
**Line**: 420
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/setuptools/tests/test_setuptools.py`
**Line**: 119
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/setuptools/_distutils/tests/test_build_ext.py`
**Line**: 87
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/setuptools/_distutils/compilers/C/msvc.py`
**Line**: 10
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/setuptools/_vendor/autocommand/autoparse.py`
**Line**: 139
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/setuptools/_vendor/autocommand/autoparse.py`
**Line**: 152
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/setuptools/_vendor/autocommand/autoparse.py`
**Line**: 303
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/setuptools/_vendor/inflect/__init__.py`
**Line**: 1998
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/setuptools/_vendor/inflect/__init__.py`
**Line**: 2790
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/setuptools/_vendor/inflect/__init__.py`
**Line**: 3282
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/setuptools/_vendor/packaging/metadata.py`
**Line**: 204
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/setuptools/_vendor/packaging/metadata.py`
**Line**: 806
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/setuptools/_vendor/packaging/requirements.py`
**Line**: 29
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/setuptools/_vendor/packaging/requirements.py`
**Line**: 32
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/setuptools/_vendor/packaging/tags.py`
**Line**: 378
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/setuptools/_vendor/typeguard/_checkers.py`
**Line**: 725
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/setuptools/_vendor/typeguard/_checkers.py`
**Line**: 736
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/setuptools/_vendor/wheel/_bdist_wheel.py`
**Line**: 92
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/setuptools/_vendor/wheel/_bdist_wheel.py`
**Line**: 344
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/setuptools/_vendor/wheel/vendored/packaging/requirements.py`
**Line**: 28
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/setuptools/_vendor/wheel/vendored/packaging/requirements.py`
**Line**: 31
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/setuptools/_vendor/wheel/vendored/packaging/tags.py`
**Line**: 382
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/setuptools/config/_validate_pyproject/extra_validations.py`
**Line**: 78
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/opt_einsum/tests/test_paths.py`
**Line**: 189
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/numpy/_core/_add_newdocs_scalars.py`
**Line**: 129
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/numpy/_core/_add_newdocs_scalars.py`
**Line**: 338
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/numpy/_core/_dtype.py`
**Line**: 166
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/numpy/_core/_dtype.py`
**Line**: 174
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/numpy/_core/_dtype.py`
**Line**: 183
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/numpy/_core/_methods.py`
**Line**: 88
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/numpy/_core/arrayprint.py`
**Line**: 1567
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/numpy/_core/numeric.py`
**Line**: 545
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/numpy/_typing/_array_like.py`
**Line**: 48
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/numpy/_typing/_char_codes.py`
**Line**: 211
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/numpy/_typing/_dtype_like.py`
**Line**: 31
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/numpy/f2py/_isocbind.py`
**Line**: 55
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/numpy/f2py/capi_maps.py`
**Line**: 249
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/numpy/f2py/capi_maps.py`
**Line**: 501
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/numpy/f2py/cfuncs.py`
**Line**: 852
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/numpy/f2py/crackfortran.py`
**Line**: 2469
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/numpy/f2py/crackfortran.py`
**Line**: 2558
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/numpy/f2py/f2py2e.py`
**Line**: 461
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/numpy/f2py/f2py2e.py`
**Line**: 653
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/numpy/f2py/symbolic.py`
**Line**: 23
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/numpy/f2py/symbolic.py`
**Line**: 24
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/numpy/f2py/symbolic.py`
**Line**: 25
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/numpy/f2py/symbolic.py`
**Line**: 519
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/numpy/f2py/symbolic.py`
**Line**: 569
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/numpy/f2py/symbolic.py`
**Line**: 810
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/numpy/f2py/symbolic.py`
**Line**: 845
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/numpy/f2py/symbolic.py`
**Line**: 895
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/numpy/f2py/symbolic.py`
**Line**: 1107
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/numpy/fft/__init__.py`
**Line**: 203
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/numpy/lib/_datasource.py`
**Line**: 71
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/numpy/lib/_datasource.py`
**Line**: 331
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/numpy/lib/_datasource.py`
**Line**: 397
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/numpy/lib/_datasource.py`
**Line**: 511
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/numpy/lib/_datasource.py`
**Line**: 514
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py`
**Line**: 866
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/numpy/lib/_nanfunctions_impl.py`
**Line**: 1689
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/numpy/lib/_npyio_impl.py`
**Line**: 249
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/numpy/lib/_npyio_impl.py`
**Line**: 2253
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/numpy/lib/mixins.py`
**Line**: 166
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/numpy/ma/core.py`
**Line**: 237
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/numpy/ma/core.py`
**Line**: 480
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/numpy/ma/core.py`
**Line**: 2915
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/numpy/ma/core.py`
**Line**: 4750
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/numpy/polynomial/_polybase.py`
**Line**: 107
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/numpy/polynomial/_polybase.py`
**Line**: 432
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/numpy/polynomial/chebyshev.py`
**Line**: 798
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/numpy/polynomial/polynomial.py`
**Line**: 405
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/numpy/polynomial/polyutils.py`
**Line**: 538
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/numpy/_core/tests/test_array_coercion.py`
**Line**: 452
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/numpy/_core/tests/test_array_coercion.py`
**Line**: 905
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/numpy/_core/tests/test_arrayprint.py`
**Line**: 154
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/numpy/_core/tests/test_casting_unittests.py`
**Line**: 781
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/numpy/_core/tests/test_datetime.py`
**Line**: 1585
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/numpy/_core/tests/test_datetime.py`
**Line**: 2566
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/numpy/_core/tests/test_machar.py`
**Line**: 19
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/numpy/_core/tests/test_multiarray.py`
**Line**: 6367
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/numpy/_core/tests/test_multiarray.py`
**Line**: 7665
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/numpy/_core/tests/test_print.py`
**Line**: 100
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/numpy/_core/tests/test_scalarmath.py`
**Line**: 103
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/numpy/_core/tests/test_scalarmath.py`
**Line**: 1143
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/numpy/_core/tests/test_stringdtype.py`
**Line**: 1551
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/numpy/_core/tests/test_umath.py`
**Line**: 1152
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/numpy/_core/tests/test_umath.py`
**Line**: 1879
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/numpy/_core/tests/test_umath.py`
**Line**: 2941
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/numpy/_core/tests/test_umath_complex.py`
**Line**: 17
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/numpy/_core/tests/test_umath_complex.py`
**Line**: 18
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/numpy/_core/tests/test_umath_complex.py`
**Line**: 19
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/numpy/_core/tests/test_umath_complex.py`
**Line**: 24
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/numpy/_core/tests/test_umath_complex.py`
**Line**: 28
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/numpy/_core/tests/test_umath_complex.py`
**Line**: 31
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/numpy/_core/tests/test_umath_complex.py`
**Line**: 126
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/numpy/_core/tests/test_umath_complex.py`
**Line**: 341
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/numpy/_core/tests/test_umath_complex.py`
**Line**: 483
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/numpy/f2py/_backends/_meson.py`
**Line**: 229
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/numpy/f2py/tests/test_docs.py`
**Line**: 64
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/numpy/f2py/tests/test_f2py2e.py`
**Line**: 415
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/numpy/f2py/tests/test_f2py2e.py`
**Line**: 682
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/numpy/f2py/tests/test_f2py2e.py`
**Line**: 691
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/numpy/f2py/tests/test_f2py2e.py`
**Line**: 747
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/numpy/f2py/tests/test_f2py2e.py`
**Line**: 821
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/numpy/f2py/tests/test_f2py2e.py`
**Line**: 827
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/numpy/f2py/tests/test_f2py2e.py`
**Line**: 835
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/numpy/f2py/tests/test_f2py2e.py`
**Line**: 843
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/numpy/f2py/tests/test_f2py2e.py`
**Line**: 851
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/numpy/f2py/tests/test_f2py2e.py`
**Line**: 859
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/numpy/f2py/tests/test_f2py2e.py`
**Line**: 867
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/numpy/f2py/tests/test_f2py2e.py`
**Line**: 875
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/numpy/f2py/tests/test_f2py2e.py`
**Line**: 883
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/numpy/f2py/tests/test_f2py2e.py`
**Line**: 891
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/numpy/f2py/tests/test_f2py2e.py`
**Line**: 899
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/numpy/f2py/tests/test_f2py2e.py`
**Line**: 907
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/numpy/f2py/tests/test_f2py2e.py`
**Line**: 915
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/numpy/f2py/tests/test_f2py2e.py`
**Line**: 923
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/numpy/f2py/tests/test_f2py2e.py`
**Line**: 931
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/numpy/f2py/tests/test_f2py2e.py`
**Line**: 939
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/numpy/f2py/tests/test_f2py2e.py`
**Line**: 947
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/numpy/f2py/tests/test_f2py2e.py`
**Line**: 955
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/numpy/f2py/tests/test_f2py2e.py`
**Line**: 963
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/numpy/lib/tests/test_function_base.py`
**Line**: 3816
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/numpy/lib/tests/test_function_base.py`
**Line**: 4532
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/numpy/lib/tests/test_io.py`
**Line**: 340
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/numpy/lib/tests/test_recfunctions.py`
**Line**: 560
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/numpy/lib/tests/test_recfunctions.py`
**Line**: 829
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/numpy/lib/tests/test_type_check.py`
**Line**: 279
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/numpy/lib/tests/test_type_check.py`
**Line**: 310
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/numpy/lib/tests/test_type_check.py`
**Line**: 445
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/numpy/linalg/tests/test_linalg.py`
**Line**: 1059
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/numpy/ma/tests/test_core.py`
**Line**: 5617
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/numpy/ma/tests/test_old_ma.py`
**Line**: 720
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/numpy/ma/tests/test_old_ma.py`
**Line**: 890
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/numpy/random/tests/test_random.py`
**Line**: 1069
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/numpy/testing/_private/utils.py`
**Line**: 2281
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/networkx/algorithms/clique.py`
**Line**: 300
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/networkx/algorithms/cuts.py`
**Line**: 19
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/networkx/algorithms/cuts.py`
**Line**: 322
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/networkx/algorithms/cuts.py`
**Line**: 362
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/networkx/algorithms/cycles.py`
**Line**: 841
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/networkx/algorithms/dag.py`
**Line**: 1175
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/networkx/algorithms/distance_measures.py`
**Line**: 201
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/networkx/algorithms/distance_measures.py`
**Line**: 202
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/networkx/algorithms/distance_measures.py`
**Line**: 205
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/networkx/algorithms/distance_measures.py`
**Line**: 261
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/networkx/algorithms/distance_measures.py`
**Line**: 263
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/networkx/algorithms/distance_measures.py`
**Line**: 265
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/networkx/algorithms/distance_measures.py`
**Line**: 266
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/networkx/algorithms/distance_measures.py`
**Line**: 267
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/networkx/algorithms/distance_measures.py`
**Line**: 292
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/networkx/algorithms/distance_measures.py`
**Line**: 293
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/networkx/algorithms/distance_measures.py`
**Line**: 296
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/networkx/algorithms/distance_measures.py`
**Line**: 297
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/networkx/algorithms/distance_measures.py`
**Line**: 298
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/networkx/algorithms/distance_regular.py`
**Line**: 184
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/networkx/algorithms/efficiency_measures.py`
**Line**: 118
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/networkx/algorithms/hybrid.py`
**Line**: 68
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/networkx/algorithms/node_classification.py`
**Line**: 94
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/networkx/algorithms/node_classification.py`
**Line**: 173
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/networkx/algorithms/similarity.py`
**Line**: 686
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/networkx/algorithms/similarity.py`
**Line**: 1190
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/networkx/algorithms/threshold.py`
**Line**: 1021
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/networkx/algorithms/threshold.py`
**Line**: 1033
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/networkx/algorithms/walks.py`
**Line**: 72
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/networkx/drawing/layout.py`
**Line**: 792
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/networkx/drawing/layout.py`
**Line**: 1152
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/networkx/drawing/nx_latex.py`
**Line**: 214
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/networkx/drawing/nx_latex.py`
**Line**: 277
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/networkx/drawing/nx_pylab.py`
**Line**: 2499
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/networkx/generators/community.py`
**Line**: 1034
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/networkx/generators/degree_seq.py`
**Line**: 671
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/networkx/generators/geometric.py`
**Line**: 190
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/networkx/linalg/algebraicconnectivity.py`
**Line**: 192
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/networkx/linalg/algebraicconnectivity.py`
**Line**: 279
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/networkx/linalg/algebraicconnectivity.py`
**Line**: 296
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/networkx/linalg/bethehessianmatrix.py`
**Line**: 75
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/networkx/linalg/bethehessianmatrix.py`
**Line**: 77
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/networkx/linalg/laplacianmatrix.py`
**Line**: 128
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/networkx/linalg/laplacianmatrix.py`
**Line**: 238
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/networkx/linalg/laplacianmatrix.py`
**Line**: 244
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/networkx/linalg/laplacianmatrix.py`
**Line**: 341
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/networkx/linalg/laplacianmatrix.py`
**Line**: 344
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/networkx/linalg/laplacianmatrix.py`
**Line**: 436
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/networkx/linalg/laplacianmatrix.py`
**Line**: 498
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/networkx/linalg/laplacianmatrix.py`
**Line**: 503
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/networkx/readwrite/gexf.py`
**Line**: 761
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/networkx/tests/test_all_random_functions.py`
**Line**: 30
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/networkx/tests/test_all_random_functions.py`
**Line**: 36
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/networkx/tests/test_all_random_functions.py`
**Line**: 56
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/networkx/tests/test_all_random_functions.py`
**Line**: 63
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/networkx/tests/test_all_random_functions.py`
**Line**: 65
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/networkx/tests/test_all_random_functions.py`
**Line**: 67
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/networkx/tests/test_all_random_functions.py`
**Line**: 73
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/networkx/tests/test_all_random_functions.py`
**Line**: 94
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/networkx/tests/test_all_random_functions.py`
**Line**: 98
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/networkx/tests/test_all_random_functions.py`
**Line**: 157
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/networkx/tests/test_all_random_functions.py`
**Line**: 216
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/networkx/tests/test_all_random_functions.py`
**Line**: 226
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/networkx/tests/test_all_random_functions.py`
**Line**: 240
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/networkx/tests/test_all_random_functions.py`
**Line**: 244
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/networkx/tests/test_all_random_functions.py`
**Line**: 248
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/networkx/algorithms/approximation/dominating_set.py`
**Line**: 22
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/networkx/algorithms/approximation/traveling_salesman.py`
**Line**: 805
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/networkx/algorithms/bipartite/matching.py`
**Line**: 276
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/networkx/algorithms/bipartite/matching.py`
**Line**: 282
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/networkx/algorithms/bipartite/matching.py`
**Line**: 287
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/networkx/algorithms/bipartite/redundancy.py`
**Line**: 93
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/networkx/algorithms/centrality/reaching.py`
**Line**: 111
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/networkx/algorithms/centrality/reaching.py`
**Line**: 206
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/networkx/algorithms/coloring/equitable_coloring.py`
**Line**: 163
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/networkx/algorithms/connectivity/edge_kcomponents.py`
**Line**: 97
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/networkx/algorithms/connectivity/edge_kcomponents.py`
**Line**: 314
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/networkx/algorithms/isomorphism/ismags.py`
**Line**: 292
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/networkx/algorithms/isomorphism/ismags.py`
**Line**: 293
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/networkx/algorithms/isomorphism/ismags.py`
**Line**: 713
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/networkx/algorithms/isomorphism/ismags.py`
**Line**: 809
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/networkx/algorithms/isomorphism/ismags.py`
**Line**: 859
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/networkx/algorithms/isomorphism/isomorphvf2.py`
**Line**: 217
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/networkx/algorithms/isomorphism/vf2pp.py`
**Line**: 353
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/networkx/algorithms/isomorphism/vf2pp.py`
**Line**: 450
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/networkx/algorithms/link_analysis/pagerank_alg.py`
**Line**: 463
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/networkx/algorithms/shortest_paths/unweighted.py`
**Line**: 181
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/networkx/algorithms/shortest_paths/unweighted.py`
**Line**: 475
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/networkx/algorithms/shortest_paths/weighted.py`
**Line**: 1135
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/networkx/algorithms/tests/test_swap.py`
**Line**: 40
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/networkx/algorithms/traversal/beamsearch.py`
**Line**: 77
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/networkx/algorithms/tree/branchings.py`
**Line**: 11
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/networkx/algorithms/tree/mst.py`
**Line**: 124
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/networkx/algorithms/tree/mst.py`
**Line**: 132
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/networkx/algorithms/assortativity/tests/test_connectivity.py`
**Line**: 138
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/networkx/algorithms/bipartite/tests/test_matching.py`
**Line**: 110
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/networkx/algorithms/shortest_paths/tests/test_weighted.py`
**Line**: 890
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/networkx/generators/tests/test_expanders.py`
**Line**: 37
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/networkx/readwrite/json_graph/node_link.py`
**Line**: 129
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/networkx/readwrite/json_graph/node_link.py`
**Line**: 274
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/networkx/readwrite/json_graph/tests/test_node_link.py`
**Line**: 27
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/fsspec/implementations/cache_metadata.py`
**Line**: 159
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/fsspec/implementations/cached.py`
**Line**: 323
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/fsspec/implementations/http.py`
**Line**: 104
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/fsspec/implementations/http_sync.py`
**Line**: 105
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/fsspec/implementations/http_sync.py`
**Line**: 883
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/fsspec/implementations/http_sync.py`
**Line**: 896
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/fsspec/implementations/local.py`
**Line**: 356
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/fsspec/implementations/local.py`
**Line**: 397
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/fsspec/implementations/reference.py`
**Line**: 147
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/fsspec/implementations/reference.py`
**Line**: 473
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/fsspec/implementations/reference.py`
**Line**: 546
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/fsspec/implementations/reference.py`
**Line**: 748
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/fsspec/implementations/reference.py`
**Line**: 884
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/fsspec/implementations/reference.py`
**Line**: 985
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/fsspec/implementations/smb.py`
**Line**: 394
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/fsspec/implementations/tar.py`
**Line**: 78
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/fsspec/implementations/tar.py`
**Line**: 92
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/fsspec/implementations/tar.py`
**Line**: 101
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/triton/backends/driver.py`
**Line**: 40
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/triton/backends/driver.py`
**Line**: 51
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/triton/compiler/code_generator.py`
**Line**: 198
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/triton/compiler/code_generator.py`
**Line**: 323
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/triton/compiler/code_generator.py`
**Line**: 634
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/triton/compiler/code_generator.py`
**Line**: 704
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/triton/compiler/code_generator.py`
**Line**: 758
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/triton/compiler/code_generator.py`
**Line**: 828
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/triton/compiler/compiler.py`
**Line**: 41
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/triton/compiler/compiler.py`
**Line**: 102
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/triton/compiler/compiler.py`
**Line**: 393
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/triton/compiler/compiler.py`
**Line**: 489
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/triton/language/__init__.py`
**Line**: 300
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/triton/language/core.py`
**Line**: 1297
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/triton/language/random.py`
**Line**: 131
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/triton/language/semantic.py`
**Line**: 1573
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/triton/language/standard.py`
**Line**: 332
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/triton/language/standard.py`
**Line**: 355
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/triton/profiler/specs.py`
**Line**: 6
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/triton/profiler/viewer.py`
**Line**: 269
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/triton/runtime/interpreter.py`
**Line**: 1125
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/triton/runtime/interpreter.py`
**Line**: 1181
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/triton/runtime/jit.py`
**Line**: 208
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/triton/runtime/jit.py`
**Line**: 671
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/triton/runtime/jit.py`
**Line**: 691
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/triton/backends/nvidia/compiler.py`
**Line**: 94
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/triton/backends/nvidia/compiler.py`
**Line**: 247
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/triton/backends/nvidia/driver.py`
**Line**: 715
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/triton/experimental/gluon/language/_core.py`
**Line**: 17
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/triton/experimental/gluon/language/_core.py`
**Line**: 108
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/cluster/hierarchy.py`
**Line**: 1385
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/differentiate/_differentiate.py`
**Line**: 372
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/differentiate/_differentiate.py`
**Line**: 1089
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/integrate/_ode.py`
**Line**: 392
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/integrate/_tanhsinh.py`
**Line**: 14
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/interpolate/_bsplines.py`
**Line**: 2369
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/io/_netcdf.py`
**Line**: 20
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/linalg/_decomp.py`
**Line**: 799
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/linalg/_decomp.py`
**Line**: 800
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/linalg/_decomp.py`
**Line**: 805
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/linalg/_matfuncs.py`
**Line**: 221
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/linalg/_matfuncs_inv_ssq.py`
**Line**: 38
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/linalg/_matfuncs_inv_ssq.py`
**Line**: 73
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/optimize/_basinhopping.py`
**Line**: 170
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/optimize/_bracket.py`
**Line**: 168
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/optimize/_chandrupatla.py`
**Line**: 7
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/optimize/_constraints.py`
**Line**: 492
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/optimize/_direct_py.py`
**Line**: 256
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/optimize/_linprog_ip.py`
**Line**: 92
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/optimize/_linprog_rs.py`
**Line**: 72
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/optimize/_linprog_rs.py`
**Line**: 73
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/optimize/_linprog_rs.py`
**Line**: 376
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/optimize/_linprog_util.py`
**Line**: 862
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/optimize/_linprog_util.py`
**Line**: 876
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/optimize/_optimize.py`
**Line**: 2050
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/optimize/_remove_redundancy.py`
**Line**: 423
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/optimize/_shgo.py`
**Line**: 477
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/optimize/_shgo.py`
**Line**: 716
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/optimize/_shgo.py`
**Line**: 1176
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/optimize/_shgo.py`
**Line**: 1502
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/optimize/_trustregion.py`
**Line**: 289
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/signal/_filter_design.py`
**Line**: 528
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/signal/_filter_design.py`
**Line**: 1655
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/signal/_filter_design.py`
**Line**: 4898
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/signal/_ltisys.py`
**Line**: 1996
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/signal/_signaltools.py`
**Line**: 4339
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/sparse/_base.py`
**Line**: 659
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/sparse/_bsr.py`
**Line**: 134
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/sparse/_bsr.py`
**Line**: 346
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/sparse/_compressed.py`
**Line**: 227
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/sparse/_compressed.py`
**Line**: 700
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/sparse/_compressed.py`
**Line**: 967
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/sparse/_construct.py`
**Line**: 412
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/sparse/_construct.py`
**Line**: 541
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/sparse/_construct.py`
**Line**: 627
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/sparse/_construct.py`
**Line**: 684
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/sparse/_csr.py`
**Line**: 229
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/sparse/_data.py`
**Line**: 18
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/sparse/_dok.py`
**Line**: 474
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/sparse/_index.py`
**Line**: 238
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/sparse/_index.py`
**Line**: 303
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/spatial/_kdtree.py`
**Line**: 913
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/special/_lambertw.py`
**Line**: 146
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/special/_support_alternative_backends.py`
**Line**: 125
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/stats/_axis_nan_policy.py`
**Line**: 167
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/stats/_continued_fraction.py`
**Line**: 10
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/stats/_correlation.py`
**Line**: 10
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/stats/_discrete_distns.py`
**Line**: 901
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/stats/_discrete_distns.py`
**Line**: 1294
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/stats/_distribution_infrastructure.py`
**Line**: 44
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/stats/_distribution_infrastructure.py`
**Line**: 1262
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/stats/_distribution_infrastructure.py`
**Line**: 1263
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/stats/_distribution_infrastructure.py`
**Line**: 1264
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/stats/_distribution_infrastructure.py`
**Line**: 3015
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/stats/_distribution_infrastructure.py`
**Line**: 3453
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/stats/_distribution_infrastructure.py`
**Line**: 4442
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/stats/_distribution_infrastructure.py`
**Line**: 5124
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/stats/_morestats.py`
**Line**: 2627
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/stats/_mstats_basic.py`
**Line**: 2406
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/stats/_page_trend_test.py`
**Line**: 322
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/stats/_qmc.py`
**Line**: 456
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/stats/_tukeylambda_stats.py`
**Line**: 26
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/stats/_tukeylambda_stats.py`
**Line**: 27
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/stats/_tukeylambda_stats.py`
**Line**: 131
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/stats/_tukeylambda_stats.py`
**Line**: 132
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/_lib/array_api_extra/testing.py`
**Line**: 23
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/_lib/array_api_compat/common/_aliases.py`
**Line**: 16
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/_lib/array_api_compat/common/_aliases.py`
**Line**: 309
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/_lib/array_api_compat/common/_aliases.py`
**Line**: 381
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/_lib/array_api_compat/common/_helpers.py`
**Line**: 42
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/_lib/array_api_compat/common/_helpers.py`
**Line**: 119
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/_lib/array_api_compat/common/_helpers.py`
**Line**: 264
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/_lib/array_api_compat/common/_helpers.py`
**Line**: 293
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/_lib/array_api_compat/common/_helpers.py`
**Line**: 300
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/_lib/array_api_compat/common/_helpers.py`
**Line**: 642
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/_lib/array_api_compat/common/_helpers.py`
**Line**: 740
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/_lib/array_api_compat/common/_helpers.py`
**Line**: 876
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/_lib/array_api_compat/cupy/_info.py`
**Line**: 181
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/_lib/array_api_compat/cupy/_info.py`
**Line**: 243
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/_lib/array_api_compat/torch/_aliases.py`
**Line**: 833
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/_lib/array_api_compat/dask/array/_aliases.py`
**Line**: 65
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/_lib/array_api_compat/dask/array/_aliases.py`
**Line**: 96
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/_lib/array_api_compat/dask/array/_aliases.py`
**Line**: 163
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/_lib/array_api_compat/dask/array/_aliases.py`
**Line**: 224
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/_lib/array_api_compat/dask/array/linalg.py`
**Line**: 32
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/_lib/array_api_compat/dask/array/linalg.py`
**Line**: 59
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/_lib/array_api_extra/_lib/_at.py`
**Line**: 23
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/_lib/array_api_extra/_lib/_utils/_helpers.py`
**Line**: 36
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/_lib/array_api_extra/_lib/_utils/_helpers.py`
**Line**: 325
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/_lib/pyprima/cobyla/cobylb.py`
**Line**: 119
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/differentiate/tests/test_differentiate.py`
**Line**: 585
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/fft/_pocketfft/basic.py`
**Line**: 82
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/fft/_pocketfft/basic.py`
**Line**: 194
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/fft/tests/test_real_transforms.py`
**Line**: 110
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/fft/_pocketfft/tests/test_basic.py`
**Line**: 865
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/integrate/_rules/_gauss_kronrod.py`
**Line**: 83
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/integrate/_rules/_gauss_legendre.py`
**Line**: 56
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/integrate/_rules/_genz_malik.py`
**Line**: 78
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/integrate/_rules/_genz_malik.py`
**Line**: 134
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/io/_harwell_boeing/hb.py`
**Line**: 12
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/io/arff/_arffread.py`
**Line**: 21
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/io/arff/_arffread.py`
**Line**: 842
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/linalg/tests/test_blas.py`
**Line**: 902
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/linalg/tests/test_blas.py`
**Line**: 903
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/linalg/tests/test_blas.py`
**Line**: 904
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/linalg/tests/test_blas.py`
**Line**: 942
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/linalg/tests/test_lapack.py`
**Line**: 100
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/linalg/tests/test_lapack.py`
**Line**: 101
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/linalg/tests/test_lapack.py`
**Line**: 1941
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/optimize/_shgo_lib/_complex.py`
**Line**: 1179
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/optimize/_shgo_lib/_complex.py`
**Line**: 1180
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/optimize/_shgo_lib/_complex.py`
**Line**: 1187
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/optimize/_shgo_lib/_complex.py`
**Line**: 1191
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/optimize/_shgo_lib/_complex.py`
**Line**: 1217
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/optimize/_shgo_lib/_complex.py`
**Line**: 1220
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/optimize/_shgo_lib/_complex.py`
**Line**: 1222
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/optimize/_trustregion_constr/projections.py`
**Line**: 61
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/optimize/_trustregion_constr/projections.py`
**Line**: 101
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/optimize/_trustregion_constr/qp_subproblem.py`
**Line**: 54
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/optimize/_trustregion_constr/tr_interior_point.py`
**Line**: 349
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/optimize/tests/test__dual_annealing.py`
**Line**: 69
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/optimize/tests/test__remove_redundancy.py`
**Line**: 5
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/optimize/tests/test__shgo.py`
**Line**: 579
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/optimize/tests/test__shgo.py`
**Line**: 715
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/optimize/tests/test_chandrupatla.py`
**Line**: 970
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/optimize/tests/test_optimize.py`
**Line**: 3063
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/signal/tests/test_bsplines.py`
**Line**: 110
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/signal/tests/test_filter_design.py`
**Line**: 2584
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/signal/tests/test_filter_design.py`
**Line**: 2589
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/signal/tests/test_filter_design.py`
**Line**: 2599
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/signal/tests/test_filter_design.py`
**Line**: 2604
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/signal/tests/test_ltisys.py`
**Line**: 604
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/signal/tests/test_ltisys.py`
**Line**: 676
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/sparse/tests/test_base.py`
**Line**: 297
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/sparse/tests/test_base.py`
**Line**: 298
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/sparse/tests/test_base.py`
**Line**: 2259
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/sparse/tests/test_base.py`
**Line**: 4736
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/sparse/tests/test_base.py`
**Line**: 5472
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/sparse/tests/test_base.py`
**Line**: 5612
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/sparse/tests/test_construct.py`
**Line**: 23
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/sparse/tests/test_spfuncs.py`
**Line**: 16
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/sparse/linalg/_isolve/lsmr.py`
**Line**: 213
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/sparse/linalg/_isolve/lsmr.py`
**Line**: 214
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/sparse/linalg/_isolve/minres.py`
**Line**: 357
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/sparse/linalg/tests/test_onenormest.py`
**Line**: 145
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/sparse/linalg/tests/test_onenormest.py`
**Line**: 146
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/sparse/linalg/tests/test_onenormest.py`
**Line**: 147
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/sparse/linalg/_eigen/tests/test_svds.py`
**Line**: 626
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/sparse/linalg/_isolve/tests/test_iterative.py`
**Line**: 20
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/sparse/linalg/_isolve/tests/test_iterative.py`
**Line**: 21
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/sparse/linalg/_isolve/tests/test_iterative.py`
**Line**: 366
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/spatial/transform/tests/test_rotation.py`
**Line**: 885
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/spatial/transform/tests/test_rotation.py`
**Line**: 920
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/spatial/transform/tests/test_rotation.py`
**Line**: 2083
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/special/tests/test_basic.py`
**Line**: 1438
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/special/tests/test_basic.py`
**Line**: 2448
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/special/tests/test_cdflib.py`
**Line**: 451
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/special/tests/test_erfinv.py`
**Line**: 21
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/special/tests/test_exponential_integrals.py`
**Line**: 48
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/special/tests/test_exponential_integrals.py`
**Line**: 50
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/special/tests/test_logit.py`
**Line**: 23
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/special/tests/test_logit.py`
**Line**: 24
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/special/tests/test_orthogonal.py`
**Line**: 234
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/special/tests/test_sf_error.py`
**Line**: 34
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/stats/_levy_stable/__init__.py`
**Line**: 193
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/stats/_levy_stable/__init__.py`
**Line**: 321
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/stats/tests/test_contingency.py`
**Line**: 97
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/stats/tests/test_contingency.py`
**Line**: 147
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/stats/tests/test_continuous.py`
**Line**: 890
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/stats/tests/test_continuous_basic.py`
**Line**: 136
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/stats/tests/test_continuous_basic.py`
**Line**: 424
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/stats/tests/test_discrete_distns.py`
**Line**: 237
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/stats/tests/test_discrete_distns.py`
**Line**: 306
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/stats/tests/test_discrete_distns.py`
**Line**: 307
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/stats/tests/test_distributions.py`
**Line**: 2035
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/stats/tests/test_distributions.py`
**Line**: 3291
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/stats/tests/test_distributions.py`
**Line**: 3324
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/stats/tests/test_distributions.py`
**Line**: 3671
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/stats/tests/test_distributions.py`
**Line**: 4656
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/stats/tests/test_distributions.py`
**Line**: 4657
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/stats/tests/test_distributions.py`
**Line**: 4847
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/stats/tests/test_distributions.py`
**Line**: 4858
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/stats/tests/test_distributions.py`
**Line**: 5081
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/stats/tests/test_distributions.py`
**Line**: 5083
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/stats/tests/test_distributions.py`
**Line**: 5343
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/stats/tests/test_distributions.py`
**Line**: 5344
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/stats/tests/test_distributions.py`
**Line**: 7260
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/stats/tests/test_distributions.py`
**Line**: 7274
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/stats/tests/test_distributions.py`
**Line**: 7285
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/stats/tests/test_distributions.py`
**Line**: 7438
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/stats/tests/test_distributions.py`
**Line**: 7440
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/stats/tests/test_distributions.py`
**Line**: 7469
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/stats/tests/test_distributions.py`
**Line**: 7471
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/stats/tests/test_distributions.py`
**Line**: 7514
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/stats/tests/test_distributions.py`
**Line**: 9016
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/stats/tests/test_distributions.py`
**Line**: 9862
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/stats/tests/test_fast_gen_inversion.py`
**Line**: 145
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/stats/tests/test_morestats.py`
**Line**: 2367
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/stats/tests/test_morestats.py`
**Line**: 2387
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/stats/tests/test_mstats_basic.py`
**Line**: 1260
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/stats/tests/test_mstats_extras.py`
**Line**: 43
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/stats/tests/test_resampling.py`
**Line**: 218
**Category**: commented_debug

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/stats/tests/test_stats.py`
**Line**: 79
**Category**: todo_in_code

### 🟢 code_quality
**File**: `venv/lib/python3.12/site-packages/scipy/stats/tests/test_stats.py`
**Line**: 2245
**Category**: commented_debug

## Recommendations

1. **Priority**: Address all high-severity issues immediately
2. **Code Review**: Implement mandatory security code reviews
3. **Static Analysis**: Integrate automated security scanning in CI/CD