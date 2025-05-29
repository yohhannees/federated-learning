from setuptools import setup, find_packages

setup(
    name="federated-learning",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "tensorflow>=2.15.0",
        "numpy>=1.19.2",
        "matplotlib>=3.3.2",
        "scikit-learn>=0.23.2",
    ],
    python_requires=">=3.8",
)
