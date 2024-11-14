#include "common/lru_cache.h"
#include <gtest/gtest.h>

TEST(LRUCacheTest, SimplePut) {
  LRUCache<int, int> lru(2);
  lru.Put(1, 111);
  EXPECT_TRUE(lru.Exists(1));
  EXPECT_EQ(111, lru.Get(1));
  EXPECT_EQ(1, lru.Size());
}

TEST(LRUCacheTest, MissingValue) {
  LRUCache<int, int> lru(2);
  EXPECT_THROW(lru.Get(1), std::range_error);
}

TEST(LRUCacheTest, KeepsAllValuesWithinCapacity) {
  LRUCache<int, int> cache_lru(50);

  for (int i = 0; i < 100; ++i) {
    cache_lru.Put(i, i);
  }

  for (int i = 0; i < 50; ++i) {
    EXPECT_FALSE(cache_lru.Exists(i));
  }

  for (int i = 50; i < 100; ++i) {
    EXPECT_TRUE(cache_lru.Exists(i));
    EXPECT_EQ(i, cache_lru.Get(i));
  }

  size_t size = cache_lru.Size();
  EXPECT_EQ(50, size);
}