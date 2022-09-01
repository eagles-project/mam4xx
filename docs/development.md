# Developing MAM4xx

Welcome to the MAM4xx development team! MAM4xx is a function-by-function
port of MAM4 (the 4-mode Modal Aerosol Model package) from Fortran to
"performance portable" C++ code that can run on GPU accelerators as well as
traditional CPUs.

This C++ port of MAM4 provides a much-needed prognostic aerosol capability to
the [Energy Exascale Earth System Model (E3SM)](https://e3sm.org), which is
designed to run on the Department of Energy's
[leadership-class computing facilities](https://www.doeleadershipcomputing.org).

## The Big Picture

### SCREAM and its atmosphere driver
### Atmosphere processes
### HAERO and aerosol processes

## MAM4xx Code Structure

### Aerosol processes and parameterizations
### Aerosol process structure

## C++ Guidelines

### Style
### Best practices
### Common data types

## Kokkos and Intranodal Parallelism

### Views: C++ multidimensional arrays
### Parallel dispatch
### "Host" vs "device"
### Tips and gotchas

## Packs and Vectorization

### Why Packs?
### Masks and predicates
### Tips and gotchas

## How to Get Help

As you've likely noticed, there's a lot of technical stuff involved in
developing MAM4xx. If you get stuck on anything, please don't spend too much
time suffering without assistance from others!

If you're a core member of the MAM4xx development team on the EAGLES Project,
the easiest way for you to get help is to post a message to the
`#eagles-mam-cpp` channel in the ESMD-BER Slack workspace. If you don't have
access to this workspace, talk to any one of your team members and get them to
invite you to it. The `#eagles-mam-cpp` channel is devoted to answering
questions and troubleshooting problems encountered in the day-to-day work of
porting MAM4 from Fortran to C++.

You can also create issues in MAM4xx's [GitHub repository](https://github.com/eagles-project/mam4xx).
