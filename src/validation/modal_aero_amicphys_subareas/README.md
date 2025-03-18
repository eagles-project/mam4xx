**NOTE:** The tests in this directory all depend on a non-conventional indexing scheme that is used by arrays with dimension `microphysics::maxsubarea() = 3`.
As used in MAM4xx, when these tests were written, the values of the variables `jclea` and `jcldy` serve a dual purpose as flags and indices.
See `set_subarea_gases_and_aerosols()` for an example of this, found near line 742.
This is a rather brittle practice and should be changed, and when that occurs, all of these tests will need to be modified correspondingly.
