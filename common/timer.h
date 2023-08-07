#pragma once

#include <ratio>
#include <vector>

// #include <ctime>
// #include <sys/time.h>
#include <chrono>

class StopWatchTimer {
public:
  StopWatchTimer()
      : running_(false), clock_sessions_(0), diff_time_(0), total_time_(0) {}
  virtual ~StopWatchTimer() {}

public:
  // Start time measurement
  void Start() {
    start_time_ = std::chrono::high_resolution_clock::now();
    running_ = true;
  }

  // Stop time measurement
  void Stop() {
    diff_time_ = GetDiffTime();
    std::cout << "diff time is " << diff_time_ << std::endl;
    total_time_ += diff_time_;
    running_ = false;
    ++clock_sessions_;
  }

  // Reset time counters to zero. Does not change the timer running state but
  // does recapture this point in time as the current start time if it is
  // running.
  void Reset() {
    diff_time_ = 0;
    total_time_ = 0;
    clock_sessions_ = 0;

    if (running_) {
      start_time_ = std::chrono::high_resolution_clock::now();
    }
  }

  // Time in msec. After start if the stop watch is still running (i.e. there
  // was no call to stop()) then the elapsed time is returned, otherwise the
  // time between the last start() and stop call is returned.
  double GetTime() {
    double retval = total_time_;

    if (running_) {
      retval += GetDiffTime();
    }

    return retval;
  }

  // Mean time to date based on the number of times the stopwatch has been
  // stopped and the current total time
  double GetAverageTime() {
    return (clock_sessions_ > 0) ? (total_time_ / clock_sessions_) : 0.0;
  }

private:
  inline double GetDiffTime() {
    auto end_time = std::chrono::high_resolution_clock::now();
    // return std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time_).count();
    return std::chrono::duration<double, std::milli>(end_time - start_time_).count();
  }

private:
  bool running_;

  int clock_sessions_;

  std::chrono::time_point<std::chrono::high_resolution_clock> start_time_;

  double diff_time_;

  double total_time_;
};
