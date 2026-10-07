from setuptools import find_packages, setup
from pathlib import Path

setup(
    name="equiewald",
    version="1.0.0",
    description="Ewald-based long-range message passing for equivariant GNNs",
    packages=find_packages(include=[
        "ocpmodels", "ocpmodels.*",
        "fairchem", "fairchem.*",
        "datasets", "datasets.*",
    ]),
    include_package_data=True,
    package_data={
        "fairchem": ["core/models/uma/Jd.pt"],
    },
    python_requires=">=3.10",
    install_requires=[
        line.strip()
        for line in Path(__file__).with_name('requirements.txt').read_text().splitlines()
        if line.strip() and not line.lstrip().startswith('#')
    ],
)
