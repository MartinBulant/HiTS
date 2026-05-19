from setuptools import setup, find_packages

setup(
    name="hac_envs",
    packages=find_packages(),
    version="1.0",
    install_requires=["numpy", "gymnasium", "mujoco"],
    include_package_data=True,
)
