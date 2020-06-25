#ifndef CMM_QUEUE_HH_
#define CMM_QUEUE_HH_

#include <chrono>
#include <condition_variable>
#include <deque>
#include <mutex>

namespace cmm { namespace bit {
template <typename _Tp, typename Mutex_=std::mutex>
class FullyBlockingQueue {
 public:
  using Mutex = Mutex_;
  using Type  = _Tp;

  FullyBlockingQueue(std::size_t high_water)
    : high_water(high_water),
      count(0),
      closed(false)
  {}

  ~FullyBlockingQueue()
  {
    Close();
  }

  bool
  Push(const _Tp& element, double timeout=-1)
  {
    std::unique_lock<Mutex> lock(mutex);

    auto condition = [&]() { return closed || count <= high_water; };
    if (timeout < 0) {
      while (!condition())
        condition_pub.wait(lock, condition);
    } else {
      using namespace std::literals::chrono_literals;
      std::chrono::duration<double> duration = 1s * timeout;
      while (!condition()) {
        auto rc = condition_pub.wait_for(
                    lock,
                    duration,
                    condition);
        if (!rc)
          return false;
      }
    }

    if (closed)
      return false;

    buffer.emplace_back(element);
    count++;
    condition_sub.notify_one();

    return true;
  }

  bool
  Push(_Tp&& element, double timeout=-1)
  {
    std::unique_lock<Mutex> lock(mutex);

    auto condition = [&]() { return closed || count <= high_water; };

    if (timeout < 0) {
      while (!condition())
        condition_pub.wait(lock, condition);
    } else {
      using namespace std::literals::chrono_literals;
      std::chrono::duration<double> duration = 1s * timeout;
      while (!condition()) {
        auto rc = condition_pub.wait_for(
                    lock,
                    duration,
                    condition);
        if (!rc)
          return false;
      }
    }

    if (closed)
      return false;

    buffer.emplace_back(std::move(element));
    count++;
    condition_sub.notify_one();

    return true;
  }

  bool
  Pop(_Tp& value, double timeout=-1)
  {
    if (closed)
      return false;

    std::unique_lock<Mutex> lock(mutex);
    auto condition = [&]() { return closed || count > 0; };

    if (timeout < 0) {
      while (!condition())
        condition_sub.wait(lock, condition);
    } else {
      using namespace std::literals::chrono_literals;
      std::chrono::duration<double> duration = 1s * timeout;
      while (!condition()) {
        auto rc = condition_sub.wait_for(
                    lock,
                    duration,
                    condition);
        if (!rc)
          return false;
      }
    }

    if (closed)
      return false;

    --count;

    value = std::move(buffer.front());
    buffer.pop_front();

    condition_pub.notify_one();

    return true;
  }

  std::size_t
  Size(bool lock = false) const
  {
    if (lock) {
      std::lock_guard<Mutex> guard(mutex);
      return buffer.size();
    } else {
      return buffer.size();
    }
  }

  void
  Close()
  {
    closed = true;
    count = 0;
    condition_pub.notify_all();
    condition_sub.notify_all();
  }

  bool
  Closed() const
  {
    return closed;
  }

  void
  Clear()
  {
    std::lock_guard<Mutex> guard(mutex);
    buffer.clear();
    count = 0;
    condition_pub.notify_all();
    condition_sub.notify_all();
  }

 private:
  std::size_t high_water;
  std::deque<_Tp> buffer;
  mutable Mutex mutex;
  std::atomic<std::size_t> count;
  std::atomic<bool> closed;
  std::condition_variable condition_pub;
  std::condition_variable condition_sub;
};

template <typename _Tp, typename Mutex_=std::mutex>
class RecvBlockingQueue {
 public:
  using Mutex = Mutex_;
  using Type  = _Tp;

  RecvBlockingQueue()
    : count(0),
      closed(false)
  {}

  ~RecvBlockingQueue()
  {
    Close();
  }

  bool
  Push(const _Tp& element)
  {
    std::lock_guard<Mutex> lock(mutex);
    if (closed)
      return false;

    buffer.emplace_back(element);
    count++;
    condition_sub.notify_one();

    return true;
  }

  bool
  Push(_Tp&& element)
  {
    std::lock_guard<Mutex> lock(mutex);
    if (closed)
      return false;

    buffer.emplace_back(std::move(element));
    count++;
    condition_sub.notify_one();

    return true;
  }

  bool
  Pop(_Tp& value, double timeout=-1)
  {
    if (closed)
      return false;

    std::unique_lock<Mutex> lock(mutex);
    auto condition = [&]() { return closed || count > 0; };

    if (timeout < 0) {
      while (!condition())
        condition_sub.wait(lock, condition);
    } else {
      using namespace std::literals::chrono_literals;
      std::chrono::duration<double> duration = 1s * timeout;
      while (!condition()) {
        auto rc = condition_sub.wait_for(
                    lock,
                    duration,
                    condition);
        if (!rc)
          return false;
      }
    }

    if (closed)
      return false;

    --count;

    value = std::move(buffer.front());
    buffer.pop_front();

    condition_pub.notify_one();

    return true;
  }

  std::size_t
  Size(bool lock = false) const
  {
    if (lock) {
      std::lock_guard<Mutex> guard(mutex);
      return buffer.size();
    } else {
      return buffer.size();
    }
  }

  void
  Close()
  {
    closed = true;
    count = 0;
    condition_pub.notify_all();
    condition_sub.notify_all();
  }

  bool
  Closed() const
  {
    return closed;
  }

  void
  Clear()
  {
    std::lock_guard<Mutex> guard(mutex);
    buffer.clear();
    count = 0;
    condition_sub.notify_all();
  }

 private:
  std::deque<_Tp> buffer;
  mutable Mutex mutex;
  std::atomic<std::size_t> count;
  std::atomic<bool> closed;
  std::condition_variable condition_sub;
};
} // ns bits
} // ns shannon

#endif // CMM_QUEUE_HH_
