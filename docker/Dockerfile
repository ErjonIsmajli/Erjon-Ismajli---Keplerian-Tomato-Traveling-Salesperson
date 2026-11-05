# Environment with pykep 2.6.4 (the library behind ESA's official evaluator).
# pykep 2.x has no macOS / recent-Python wheels on PyPI, so it is installed from conda-forge
# inside a Linux container and used only to cross-check the pure-NumPy implementation.
FROM condaforge/miniforge3:26.9.4-0
RUN mamba create -y -n kttsp -c conda-forge python=3.12 pykep=2.6.4 numpy && mamba clean -afy
ENV PATH=/opt/conda/envs/kttsp/bin:$PATH
WORKDIR /work
CMD ["python", "scripts/crosscheck_pykep.py"]
