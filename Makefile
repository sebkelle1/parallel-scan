
all: scan scan_tbb

scan: scan_v1.hpp scan_v2.hpp test.hpp main.cpp
	g++ -std=c++17 -O3 -fopenmp main.cpp -o scan

scan_tbb: scan_v1.hpp scan_v2.hpp test.hpp main_tbb.cpp
	g++ -std=c++17 -O3 -fopenmp main_tbb.cpp -o scan_tbb -ltbb

clean:
	rm scan scan_tbb
