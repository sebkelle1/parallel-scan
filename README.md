
# Experimental parallel prefix sum implementation with OpenMP

With serial and C++17 parallel STL reference (requires libtbb-dev)

### compile and run

```
make
OMP_NUM_THREADS=N ./scan <vector length>
./scan_tbb <vector length>
```
