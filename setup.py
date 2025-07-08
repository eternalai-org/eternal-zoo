from setuptools import setup, find_packages
import os

# Read version from _version.py
version = {}
with open(os.path.join(os.path.dirname(__file__), 'crypto_models', '_version.py')) as f:
    exec(f.read(), version)

setup(
    name="CryptoModels",
    version=version['__version__'],
    packages=find_packages(),
    package_data={
        "crypto_models": [
            "examples/templates/*.jinja",
            "examples/best_practices/*.json",
        ],
    },
    include_package_data=True,
    install_requires=[
        "rich==14.0.0",
        "requests==2.32.4",
        "tqdm==4.67.1",
        "loguru==0.7.3",
        "psutil==7.0.0",
        "httpx==0.28.1",
        "lighthouseweb3==0.1.5",
        "python-dotenv==1.1.1",
        "fastapi==0.115.14",
        "uvicorn==0.35.0",
        "aiohttp==3.12.13",
        "setuptools==80.9.0",
        "pydantic==2.11.7",
        "pillow==11.3.0",
        "asyncio==3.4.3",
        "json_repair==0.47.6",
        "msgpack==1.1.1"
    ],
    entry_points={
        "console_scripts": [
            "eai = crypto_models.cli:main",
        ],
    },
    author="EternalAI",
    description="A library to manage local language models",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.11",
)