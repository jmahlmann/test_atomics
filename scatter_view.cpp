#include<Kokkos_Core.hpp>
#include<Kokkos_ScatterView.hpp>
#include <Kokkos_StdAlgorithms.hpp>
#include <chrono>     

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
Kokkos::View<unsigned long long int*> counter("Counter",1);
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

      std::chrono::time_point<std::chrono::system_clock> start, end;
 
   
      // std::chrono::duration<double> elapsed_seconds = end - start;
      // std::time_t end_time = std::chrono::system_clock::to_time_t(end);
   
      // std::cout << "finished computation at " << std::ctime(&end_time)
      //           << "elapsed time: " << elapsed_seconds.count() << "s\n";

    start = std::chrono::system_clock::now();
    double time_loop = atomic_int_loop();
    std::cout << "Time AtomicInt: " << time_loop << std::endl;
    end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::time_t end_time = std::chrono::system_clock::to_time_t(end);
    std::cout << "finished computation at " << std::ctime(&end_time)
                << "elapsed time: " << elapsed_seconds.count() << "s\n";

    start = std::chrono::system_clock::now();
    time_loop = atomic_int32_loop();
    std::cout << "Time AtomicInt32: " << time_loop << std::endl;
    end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::time_t end_time = std::chrono::system_clock::to_time_t(end);
    std::cout << "finished computation at " << std::ctime(&end_time)
                << "elapsed time: " << elapsed_seconds.count() << "s\n";

    start = std::chrono::system_clock::now();
    time_loop = atomic_int64_loop();
    std::cout << "Time AtomicInt64: " << time_loop << std::endl;
    end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::time_t end_time = std::chrono::system_clock::to_time_t(end);
    std::cout << "finished computation at " << std::ctime(&end_time)
                << "elapsed time: " << elapsed_seconds.count() << "s\n";

    start = std::chrono::system_clock::now();
    time_loop = atomic_longint_loop();
    std::cout << "Time AtomicLongInt: " << time_loop << std::endl;
    end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::time_t end_time = std::chrono::system_clock::to_time_t(end);
    std::cout << "finished computation at " << std::ctime(&end_time)
                << "elapsed time: " << elapsed_seconds.count() << "s\n";

    start = std::chrono::system_clock::now();
    time_loop = atomic_longlongint_loop();
    std::cout << "Time AtomicLongLongInt: " << time_loop << std::endl;
    end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::time_t end_time = std::chrono::system_clock::to_time_t(end);
    std::cout << "finished computation at " << std::ctime(&end_time)
                << "elapsed time: " << elapsed_seconds.count() << "s\n";

    start = std::chrono::system_clock::now();
    time_loop = atomic_double_loop();
    std::cout << "Time AtomicDouble: " << time_loop << std::endl;
    end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::time_t end_time = std::chrono::system_clock::to_time_t(end);
    std::cout << "finished computation at " << std::ctime(&end_time)
                << "elapsed time: " << elapsed_seconds.count() << "s\n";

  }
  Kokkos::finalize();
}