from setuptools import find_packages, setup

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
    python_requires=">=3.9",
)
