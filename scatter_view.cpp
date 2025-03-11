#include<Kokkos_Core.hpp>
#include<Kokkos_ScatterView.hpp>
#include <Kokkos_StdAlgorithms.hpp>

#include <iostream> // for std::cout

// Scatter Add algorithm using atomics
double scatter_view_loop(Kokkos::View<int**> v, 
		 Kokkos::View<int*> r) {
  Kokkos::Experimental::ScatterView<int*> results(r);
  Kokkos::Timer timer;

  results.reset();
  // Run Atomic Loop not r is already using atomics by default
  Kokkos::parallel_for("Atomic Loop", v.extent(0), 
    KOKKOS_LAMBDA(const int i) {
    auto access = results.access();
    for(int j=0; j<v.extent(1); j++)
      access(v(i,j))+=1;
  });
  Kokkos::Experimental::contribute(r,results);
  // Wait for Kernel to finish before timing
  Kokkos::fence();
  double time = timer.seconds();
  return time;
}

// Scatter Add algorithm using atomics
double atomic_loop(Kokkos::View<int**> v, 
  Kokkos::View<int*,Kokkos::MemoryTraits<Kokkos::Atomic>> r) {
Kokkos::Timer timer;
// Run Atomic Loop not r is already using atomics by default
Kokkos::parallel_for("Atomic Loop", v.extent(0), 
 KOKKOS_LAMBDA(const int i) {
 for(int j=0; j<v.extent(1); j++)
   r(v(i,j))++;
});
// Wait for Kernel to finish before timing
Kokkos::fence();
return timer.seconds();
}

// Scatter Add algorithm using atomic add
double atomic_addif_loop(Kokkos::View<int**> v, 
  Kokkos::View<int*,Kokkos::MemoryTraits<Kokkos::Atomic>> r) {
Kokkos::Timer timer;
Kokkos::View<long int*> counter("Counter",1);
Kokkos::parallel_for("Atomic Loop", 100000, 
 KOKKOS_LAMBDA(const long int i) {
      const auto idx = Kokkos::atomic_fetch_add(&counter(0),1);
});
// Wait for Kernel to finish before timing
Kokkos::fence();
return timer.seconds();
}

// Scatter Add algorithm using atomic add
double atomic_add_loop(Kokkos::View<int**> v, 
  Kokkos::View<int*,Kokkos::MemoryTraits<Kokkos::Atomic>> r) {
Kokkos::Timer timer;
Kokkos::View<int*> counter("Counter",1);
Kokkos::parallel_for("Atomic Loop", 100000, 
 KOKKOS_LAMBDA(const int i) {
      const auto idx = Kokkos::atomic_fetch_add(&counter(0),1);
});
// Wait for Kernel to finish before timing
Kokkos::fence();
return timer.seconds();
}

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc,argv);
  {

    int N = argc > 1?atoi(argv[1]):10000;
    int M = argc > 2?atoi(argv[2]):10000;

    Kokkos::View<int**> values("V",N,M);
    Kokkos::View<int*> results("R",N);
    auto values_h = Kokkos::create_mirror_view(values);

    for(int i=0; i<N; i++)
      for(int j=0; j<M; j++)
	values_h(i,j) = rand()%N;

    Kokkos::deep_copy(values,values_h);

    // double time_atomic = atomic_loop(values,results);
    // std::cout << "Time Atomic: " << N << " " << M << " " << time_atomic << std::endl;

    // double time_scatter_view = scatter_view_loop(values,results);
    // std::cout << "Time ScatterView: " << N << " " << M << " " << time_scatter_view << std::endl;

    double time_atomic_add_loop = atomic_add_loop(values,results);
    std::cout << "Time AtomicAddView: " << N << " " << M << " " << time_atomic_add_loop << std::endl;

    double time_atomic_addif_loop = atomic_addif_loop(values,results);
    std::cout << "Time AtomicAddView: " << N << " " << M << " " << time_atomic_addif_loop << std::endl;

  }
  Kokkos::finalize();
}