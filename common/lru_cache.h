#pragma once

#include <list>
#include <stdexcept>
#include <unordered_map>
#include <utility>

// hash_combine reference:
// https://stackoverflow.com/questions/2590677/how-do-i-combine-hash-values-in-c0x
inline void hash_combine(const size_t seed) {}
template <typename T, typename... Rest>
void hash_combine(size_t &seed, const T &v, Rest... rest) {
  std::hash<T> hasher;
  seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  hash_combine(seed, rest...);
}

// References:
// https://github.com/lamerman/cpp-lru-cache/tree/master
// https://github.com/nitnelave/lru_cache/tree/master
// https://github.com/mohaps/lrucache11/tree/master

// TODO(wilber): ThreadSafe.
template <typename key_t, typename value_t> class LRUCache {
public:
  using key_value_pair_t = std::pair<key_t, value_t>;
  using list_iterator_t = typename std::list<key_value_pair_t>::iterator;

  explicit LRUCache(size_t max_size) : max_size_(max_size) {}

  void Put(const key_t &key, const value_t &value) {
    auto it = cache_items_map_.find(key);
    if (it != cache_items_map_.end()) {
      cache_items_list_.erase(it->second);
      cache_items_map_.erase(it);
    }
    cache_items_list_.push_front(key_value_pair_t(key, value));
    cache_items_map_[key] = cache_items_list_.begin();

    if (cache_items_map_.size() > max_size_) {
      auto last = cache_items_list_.end();
      last--;
      cache_items_map_.erase(last->first);
      cache_items_list_.pop_back();
    }
  }

  // const value_t &Get(const key_t &key) {
  //   auto it = cache_items_map_.find(key);
  //   if (it == cache_items_map_.end()) {
  //     throw std::range_error("There is no such key in cache");
  //   } else {
  //     cache_items_list_.splice(cache_items_list_.begin(), cache_items_list_,
  //                              it->second);
  //     return it->second->second;
  //   }
  // }

  value_t &Get(const key_t &key) {
    auto it = cache_items_map_.find(key);
    if (it == cache_items_map_.end()) {
      throw std::range_error("There is no such key in cache");
    } else {
      cache_items_list_.splice(cache_items_list_.begin(), cache_items_list_,
                               it->second);
      return it->second->second;
    }
  }

  bool Exists(const key_t &key) {
    return cache_items_map_.find(key) != cache_items_map_.end();
  }

  size_t Size() { return cache_items_map_.size(); }

private:
  std::unordered_map<key_t, list_iterator_t> cache_items_map_;
  std::list<key_value_pair_t> cache_items_list_;
  size_t max_size_;
};