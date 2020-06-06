import io
import re

from setuptools import find_packages, setup

with io.open("int_rew/__init__.py", "rt", encoding="utf8") as f:
    version = re.search(r"__version__ = \"(.*?)\"", f.read()).group(1)


setup(
    name="intrinsic_rewards",
    version=version,
    url="https://github.com/kngwyu/intrinsic_rewards",
    project_urls={
        "Code": "https://github.com/kngwyu/intrinsic_rewards",
        "Issue tracker": "https://github.com/kngwyu/intrinsic_rewards/issues",
    },
    author="Yuji Kanagawa",
    author_email="yuji.kngw.80s.revive@gmail.com",
    description="A collection of DRL algorithms with intrinsic rewards",
    packages=find_packages(),
    python_requires=">=3.6",
    classifiers=[
        "License :: OSI Approved :: Apache Software License",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
)
