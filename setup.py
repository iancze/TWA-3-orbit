import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="twa",
    version="0.0.1",
    author="Ian Czekala",
    author_email="iancze@gmail.ccom",
    description="Fitting TWA 3 with exoplanet",
    include_package_data=True,  # copy the things in Manifest.in
    long_description=long_description,
    long_description_content_type="text/markdown",
    package_dir={"": "src"},
    url="https://github.com/pypa/TWA3Orbits",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
