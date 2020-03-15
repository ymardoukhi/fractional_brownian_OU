from distutils.core import setup
from Cython.Build import cythonize

setup(ext_modules = cythonize('fou_trajectory.pyx'))
setup(ext_modules = cythonize('msd_sim.pyx'))
setup(ext_modules = cythonize('tamsd_sim.pyx'))
setup(ext_modules = cythonize('tamsd_sim_gen.pyx'))
