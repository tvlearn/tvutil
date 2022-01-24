from setuptools import setup, find_packages

setup(
    name="tvutil",
    packages=find_packages(exclude=("test",)),
    zip_safe=False,
)
