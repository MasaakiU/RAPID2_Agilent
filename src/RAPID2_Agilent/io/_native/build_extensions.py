"""Build native modules in place, resolving their C++ sources locally."""

import argparse
import os
from pathlib import Path

from Cython.Build import cythonize
from setuptools import Extension, setup


def main():
    native_dir = Path(__file__).resolve().parent
    available = sorted(path.stem for path in native_dir.glob("*.pyx"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("modules", nargs="*", help=f"Modules to build (default: {', '.join(available)})")
    args = parser.parse_args()
    names = args.modules or available
    for name in names:
        if name not in available:
            parser.error(f"Unknown native module: {name}")
    # cythonize -i can switch to a parent package directory before compiling,
    # making the relative sources in the .pyx distutils directives unavailable.
    os.chdir(native_dir)
    extensions = [Extension(name, [f"{name}.pyx"]) for name in names]
    setup(
        name="rapid2-agilent-native",
        packages=[],
        ext_modules=cythonize(extensions, language_level=3, force=True),
        script_args=["build_ext", "--inplace"],
    )


if __name__ == "__main__":
    main()
