from setuptools import setup, find_packages

setup(
<<<<<<< HEAD
    name="energystats",
    version="0.5",
    author="Omar Mohamed Ghanem, Faidulla Mahmoud Ryad, Eiad Samih, Mohamed Samir Elsayed",
    description="A python statistical library for energy statistics methods.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/E-Stats/estats",
    packages=find_packages(exclude=["Tests*", "R_Tests*"]),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=[
    "numpy",
    "pandas",
    "scipy",
    "numba"
    ]
=======
    name="estats",
    version="0.1",
    packages=find_packages(),  
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "scikit-learn>=0.24.0",
        "pandas>=1.3.0",
        "matplotlib>=3.4.0"
    ],
>>>>>>> db3eeb34c890029aa5dd994bf5b8f46f2a4213d9
)
