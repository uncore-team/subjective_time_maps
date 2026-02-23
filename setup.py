from setuptools import setup, find_packages

setup(
    name="rl_coppelia",
    version="2.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},  # src is the root of the package
    entry_points={
        "console_scripts": [
            "uncore_rl=rl_coppelia.cli:main",
            "uncore_rl_gui=gui.main:main"
        ]
    },
    include_package_data=True,
    package_data={
        "rl_coppelia": ["assets/*.png"],
    },

)