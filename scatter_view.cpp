#include<Kokkos_Core.hpp>
#include<Kokkos_ScatterView.hpp>
#include <Kokkos_StdAlgorithms.hpp>

#include <iostream> // for std::cout

// Scatter Add algorithm using atomic add
double atomic_double_loop() {
Kokkos::Timer timer;
Kokkos::View<double*> counter("Counter",1);
Kokkos::parallel_for("Atomic Loop", 1000000, 
 KOKKOS_LAMBDA(const int i) {
      const auto idx = Kokkos::atomic_fetch_add(&counter(0),1);
});
// Wait for Kernel to finish before timing
Kokkos::fence();
return timer.seconds();
}

// Scatter Add algorithm using atomic add
double atomic_longint_loop() {
Kokkos::Timer timer;
Kokkos::View<long int*> counter("Counter",1);
Kokkos::parallel_for("Atomic Loop", 1000000, 
 KOKKOS_LAMBDA(const int i) {
      const auto idx = Kokkos::atomic_fetch_add(&counter(0),1);
});
// Wait for Kernel to finish before timing
Kokkos::fence();
return timer.seconds();
}

// Scatter Add algorithm using atomic add
double atomic_longlongint_loop() {
Kokkos::Timer timer;
Kokkos::View<long long int*> counter("Counter",1);
Kokkos::parallel_for("Atomic Loop", 1000000, 
 KOKKOS_LAMBDA(const int i) {
      const auto idx = Kokkos::atomic_fetch_add(&counter(0),1);
});
// Wait for Kernel to finish before timing
Kokkos::fence();
return timer.seconds();
}

// Scatter Add algorithm using atomic add
double atomic_int_loop() {
Kokkos::Timer timer;
Kokkos::View<int*> counter("Counter",1);
Kokkos::parallel_for("Atomic Loop", 1000000, 
 KOKKOS_LAMBDA(const int i) {
      const auto idx = Kokkos::atomic_fetch_add(&counter(0),1);
});
// Wait for Kernel to finish before timing
Kokkos::fence();
return timer.seconds();
}

// Scatter Add algorithm using atomic add
double atomic_int32_loop() {
Kokkos::Timer timer;
Kokkos::View<std::int32_t*> counter("Counter",1);
Kokkos::parallel_for("Atomic Loop", 1000000, 
 KOKKOS_LAMBDA(const int i) {
      const auto idx = Kokkos::atomic_fetch_add(&counter(0),1);
});
// Wait for Kernel to finish before timing
Kokkos::fence();
return timer.seconds();
}

// Scatter Add algorithm using atomic add
double atomic_int64_loop() {
Kokkos::Timer timer;
Kokkos::View<std::int64_t*> counter("Counter",1);
Kokkos::parallel_for("Atomic Loop", 1000000, 
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


    double time_loop = atomic_int_loop();
    std::cout << "Time AtomicInt: " << time_loop << std::endl;

    time_loop = atomic_int32_loop();
    std::cout << "Time AtomicInt32: " << time_loop << std::endl;

    time_loop = atomic_int64_loop();
    std::cout << "Time AtomicInt64: " << time_loop << std::endl;

    time_loop = atomic_longint_loop();
    std::cout << "Time AtomicLongInt: " << time_loop << std::endl;

    time_loop = atomic_longlongint_loop();
    std::cout << "Time AtomicLongLongInt: " << time_loop << std::endl;

    time_loop = atomic_double_loop();
    std::cout << "Time AtomicDouble: " << time_loop << std::endl;


  }
  Kokkos::finalize();
}