# Copyright 2026 The xLLM Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/xLLM-AI/xllm/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Exercise native host location and ELF loading without requiring an NPU."""

import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_VENDOR_LIB = Path("opp/vendors/custom_xllm_math/op_api/lib")


@unittest.skipUnless(
    sys.platform.startswith("linux") and all(shutil.which(tool) for tool in ("cmake", "c++", "readelf")),
    "Linux ELF loader, CMake and a C++ compiler are required",
)
class NpuNativeHostTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        temporary = tempfile.TemporaryDirectory(prefix="xllm native build ")
        cls.addClassCleanup(temporary.cleanup)
        cls.build_root = Path(temporary.name).resolve()
        source = cls.build_root / "source"
        source.mkdir()
        (source / "host.cpp").write_text(
            textwrap.dedent(
                """\
                #include <cstdint>
                #include <filesystem>
                #include <iostream>
                #include <string>
                #include "core/platform/npu/native_host_path.h"

                extern "C" const char* host_path() {
                  static const int32_t kAnchor = 0;
                  std::error_code error;
                  static const std::string path =
                      xllm::native_host_path(&kAnchor, error).string();
                  return path.c_str();
                }

                #ifndef SHARED_HOST
                extern "C" int operator_revision();
                int main(int argc, char** argv) {
                  if (argc > 1 && std::string(argv[1]) == "--invalid-anchor") {
                    std::error_code error;
                    const auto path = xllm::native_host_path(nullptr, error);
                    return path.empty() && error ? 0 : 1;
                  }
                  if (argc > 1) {
                    std::filesystem::current_path(argv[1]);
                  }
                  std::cout << operator_revision() << std::endl
                            << host_path() << std::endl;
                }
                #endif
                """
            ),
            encoding="utf-8",
        )
        (source / "vendor.cpp").write_text(
            'extern "C" int operator_revision() { return REVISION; }\n', encoding="utf-8"
        )
        (source / "CMakeLists.txt").write_text(
            textwrap.dedent(
                f"""\
                cmake_minimum_required(VERSION 3.26)
                project(native_host_fixture LANGUAGES CXX)
                set(CMAKE_CXX_STANDARD 17)
                # Reproduce modern toolchains even when the host ld defaults
                # to old dtags. Only the executable should override this.
                add_link_options("LINKER:--enable-new-dtags")
                include("{_ROOT / "cmake/npu_runtime.cmake"}")
                include_directories("{_ROOT / "xllm"}")
                add_library(cust_opapi SHARED vendor.cpp)
                target_compile_definitions(cust_opapi PRIVATE REVISION=7)
                add_library(stale_opapi SHARED vendor.cpp)
                target_compile_definitions(stale_opapi PRIVATE REVISION=99)
                set_target_properties(stale_opapi PROPERTIES OUTPUT_NAME cust_opapi
                  LIBRARY_OUTPUT_DIRECTORY "${{CMAKE_BINARY_DIR}}/external")
                add_executable(native_host host.cpp)
                set_target_properties(native_host PROPERTIES OUTPUT_NAME xllm)
                target_link_libraries(native_host PRIVATE cust_opapi ${{CMAKE_DL_LIBS}})
                # Include an existing stale RPATH to verify the bundle is first.
                set_target_properties(native_host PROPERTIES
                  BUILD_RPATH "${{CMAKE_BINARY_DIR}}/external"
                  INSTALL_RPATH "${{CMAKE_BINARY_DIR}}/external")
                xllm_configure_npu_runtime(native_host)
                add_library(shared_host SHARED host.cpp)
                target_compile_definitions(shared_host PRIVATE SHARED_HOST)
                target_link_libraries(shared_host PRIVATE ${{CMAKE_DL_LIBS}})
                xllm_configure_npu_runtime(shared_host)
                install(TARGETS native_host shared_host DESTINATION xllm)
                install(TARGETS cust_opapi DESTINATION xllm/{_VENDOR_LIB})
                """
            ),
            encoding="utf-8",
        )
        build = cls.build_root / "build"
        cls.staged = cls.build_root / "staged"
        for command in (
            ["cmake", "-S", str(source), "-B", str(build)],
            ["cmake", "--build", str(build), "--parallel", "2"],
            ["cmake", "--install", str(build), "--prefix", str(cls.staged)],
        ):
            result = subprocess.run(command, capture_output=True, text=True, timeout=120)
            if result.returncode:
                raise AssertionError(result.stdout + result.stderr)
        cls.external = build / "external"
        cls.native_build = build / "xllm"

    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory(prefix="xllm relocated wheel ")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        shutil.copytree(self.staged, self.root / "installed")
        self.package = self.root / "installed/xllm"
        self.work = self.root / "work"
        self.work.mkdir()
        self.environment = dict(
            os.environ,
            PATH=f"{self.package}:{os.environ.get('PATH', os.defpath)}",
            LD_LIBRARY_PATH=str(self.external),
        )
        self.environment.pop("LD_PRELOAD", None)

    def _run(self, command: list[str]) -> subprocess.CompletedProcess[str]:
        result = subprocess.run(
            command, cwd=self.work, env=self.environment, capture_output=True, text=True, timeout=30
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def _assert_bundled_host(self, command: list[str]) -> None:
        result = self._run(command)
        self.assertEqual(result.stdout.splitlines(), ["7", str(self.package / "xllm")])

    def test_path_launch_selects_relocated_bundle_over_stale_environment(self) -> None:
        # A same-name file in cwd must not be mistaken for the actual host.
        (self.work / "xllm").write_bytes(b"unrelated file")
        self._assert_bundled_host(["xllm"])

    def test_relative_launch_resolves_executable_after_changing_directory(self) -> None:
        elsewhere = self.root / "elsewhere"
        elsewhere.mkdir()
        self._assert_bundled_host(["../installed/xllm/xllm", str(elsewhere)])

    def test_symlink_launch_uses_bundle_beside_real_executable(self) -> None:
        link = self.work / "alias"
        link.symlink_to(self.package / "xllm")
        self._assert_bundled_host([str(link)])

    def test_missing_bundle_preserves_external_operator_loading(self) -> None:
        shutil.rmtree(self.package / "opp")
        result = self._run(["xllm"])
        self.assertEqual(result.stdout.splitlines(), ["99", str(self.package / "xllm")])

    def test_invalid_anchor_returns_error(self) -> None:
        self._run(["xllm", "--invalid-anchor"])

    def test_shared_host_resolves_library_instead_of_python_executable(self) -> None:
        library = self.package / "libshared_host.so"
        result = self._run(
            [
                sys.executable,
                "-c",
                "import ctypes; "
                f"library = ctypes.CDLL({str(library)!r}); "
                "library.host_path.restype = ctypes.c_char_p; "
                "print(library.host_path().decode())",
            ]
        )
        self.assertEqual(result.stdout.strip(), str(library))

    def test_only_executables_use_rpath_in_build_and_install(self) -> None:
        for binary in (self.native_build, self.package / "xllm"):
            with self.subTest(binary=binary):
                tags = self._run(["readelf", "-d", str(binary)]).stdout
                self.assertIn("(RPATH)", tags)
                self.assertNotIn("(RUNPATH)", tags)
                self.assertIn(f"[\u0024ORIGIN/{_VENDOR_LIB}:", tags)
        tags = self._run(["readelf", "-d", str(self.package / "libshared_host.so")]).stdout
        self.assertIn("(RUNPATH)", tags)
        self.assertNotIn("(RPATH)", tags)


if __name__ == "__main__":
    unittest.main()
