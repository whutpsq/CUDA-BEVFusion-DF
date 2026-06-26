#pragma once

#include <cstdint>
#include <deque>
#include <map>
#include <string>
#include <vector>

#include "rscl_adapter/types.hpp"

namespace rscl_adapter {

class FrameSynchronizer {
 public:
  FrameSynchronizer(const std::vector<std::string>& camera_topics, const std::vector<std::string>& camera_order,
                    const std::string& lidar_topic, float tolerance_ms, size_t max_queue_size = 30,
                    bool debug = false, int debug_limit = 100,
                    const std::vector<float>& camera_time_offsets_ms = std::vector<float>(),
                    float lidar_time_offset_ms = 0.0f);

  bool add_camera(const CameraPacket& packet, SyncedFrame* frame);
  bool add_lidar(const LidarPacket& packet, SyncedFrame* frame);

 private:
  struct CameraQueueItem {
    int64_t timestamp_us = 0;
    Image image;
  };

  bool try_sync(SyncedFrame* frame);
  void drop_older_than(int64_t timestamp_us);
  void debug_status(const char* reason, int64_t lidar_timestamp_us, const std::vector<int64_t>& best_diffs_us,
                    const std::vector<int64_t>& signed_diffs_us);

  std::vector<std::string> camera_topics_;
  std::vector<std::string> camera_order_;
  std::map<std::string, size_t> topic_to_index_;
  std::string lidar_topic_;
  int64_t tolerance_us_ = 0;
  std::vector<int64_t> camera_time_offsets_us_;
  int64_t lidar_time_offset_us_ = 0;
  size_t max_queue_size_ = 30;
  std::vector<std::deque<CameraQueueItem> > camera_queues_;
  std::deque<LidarPacket> lidar_queue_;
  int64_t last_emitted_timestamp_us_ = -1;
  bool debug_ = false;
  int debug_limit_ = 100;
  int debug_count_ = 0;
};

}  // namespace rscl_adapter
