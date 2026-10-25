"""Build the Cython/C++ extensions in RAPID2_Agilent.io._native.

`pip install -e .` builds them in place. After editing the native sources, rebuild with
    python setup.py build_ext --inplace --force
"""

from pathlib import Path

from Cython.Build import cythonize
from setuptools import Extension, setup


NATIVE_DIR = Path("src/RAPID2_Agilent/io/_native")


def native_extensions() -> list[Extension]:
    extensions = cythonize(
        [
            Extension(f"RAPID2_Agilent.io._native.{pyx.stem}", [pyx.as_posix()])
            for pyx in sorted(NATIVE_DIR.glob("*.pyx"))
        ],
        language_level=3,
    )
    # The "# distutils: sources" directives name the C++ sources relative to the .pyx
    # file, but Cython resolves them against the working directory.
    for extension in extensions:
        extension.sources = [
            source if Path(source).exists() else (NATIVE_DIR / source).as_posix()
            for source in extension.sources
        ]
    return extensions


setup(ext_modules=native_extensions())
