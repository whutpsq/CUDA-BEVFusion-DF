#include "rscl_adapter/sync.hpp"

#include <cstdlib>
#include <iostream>
#include <stdexcept>

namespace rscl_adapter {

FrameSynchronizer::FrameSynchronizer(const std::vector<std::string>& camera_topics,
                                     const std::vector<std::string>& camera_order, const std::string& lidar_topic,
                                     float tolerance_ms, size_t max_queue_size, bool debug, int debug_limit,
                                     const std::vector<float>& camera_time_offsets_ms, float lidar_time_offset_ms)
    : camera_topics_(camera_topics),
      camera_order_(camera_order),
      lidar_topic_(lidar_topic),
      tolerance_us_(static_cast<int64_t>(tolerance_ms * 1000.0f)),
      lidar_time_offset_us_(static_cast<int64_t>(lidar_time_offset_ms * 1000.0f)),
      max_queue_size_(max_queue_size),
      camera_queues_(camera_order.size()),
      debug_(debug),
      debug_limit_(debug_limit) {
  if (camera_topics_.size() != camera_order_.size()) {
    throw std::runtime_error("camera_topics and camera_order must have the same length");
  }
  if (!camera_time_offsets_ms.empty() && camera_time_offsets_ms.size() != camera_topics_.size()) {
    throw std::runtime_error("camera_time_offsets_ms must be empty or have the same length as camera_topics");
  }
  camera_time_offsets_us_.assign(camera_topics_.size(), 0);
  for (size_t i = 0; i < camera_time_offsets_ms.size(); ++i) {
    camera_time_offsets_us_[i] = static_cast<int64_t>(camera_time_offsets_ms[i] * 1000.0f);
  }
  for (size_t i = 0; i < camera_topics_.size(); ++i) topic_to_index_[camera_topics_[i]] = i;
}

bool FrameSynchronizer::add_camera(const CameraPacket& packet, SyncedFrame* frame) {
  std::map<std::string, size_t>::const_iterator it = topic_to_index_.find(packet.topic);
  if (it == topic_to_index_.end()) throw std::runtime_error("Unknown camera topic: " + packet.topic);
  std::deque<CameraQueueItem>& queue = camera_queues_[it->second];
  if (queue.size() >= max_queue_size_) queue.pop_front();
  CameraQueueItem item;
  item.timestamp_us = packet.timestamp_us + camera_time_offsets_us_[it->second];
  item.image = packet.image;
  queue.push_back(item);

  return try_sync(frame);
}

bool FrameSynchronizer::add_lidar(const LidarPacket& packet, SyncedFrame* frame) {
  if (packet.topic.empty() || packet.topic == lidar_topic_) {
    if (lidar_queue_.size() >= max_queue_size_) lidar_queue_.pop_front();
    LidarPacket adjusted = packet;
    adjusted.timestamp_us += lidar_time_offset_us_;
    lidar_queue_.push_back(adjusted);
  }

  return try_sync(frame);
}

bool FrameSynchronizer::try_sync(SyncedFrame* frame) {
  if (lidar_queue_.empty()) return false;
  const LidarPacket& lidar = lidar_queue_.back();
  if (last_emitted_timestamp_us_ == lidar.timestamp_us) return false;

  for (size_t i = 0; i < camera_queues_.size(); ++i) {
    if (camera_queues_[i].empty()) {
      debug_status("missing_camera", lidar.timestamp_us, std::vector<int64_t>(), std::vector<int64_t>());
      return false;
    }
  }

  std::vector<Image> cameras(camera_queues_.size());
  std::vector<int64_t> best_diffs_us(camera_queues_.size(), 0);
  std::vector<int64_t> signed_diffs_us(camera_queues_.size(), 0);
  bool outside_tolerance = false;
  for (size_t i = 0; i < camera_queues_.size(); ++i) {
    const std::deque<CameraQueueItem>& queue = camera_queues_[i];
    const CameraQueueItem* nearest = &queue.front();
    int64_t signed_best = nearest->timestamp_us - lidar.timestamp_us;
    int64_t best = std::llabs(signed_best);
    for (size_t j = 1; j < queue.size(); ++j) {
      int64_t signed_diff = queue[j].timestamp_us - lidar.timestamp_us;
      int64_t diff = std::llabs(signed_diff);
      if (diff < best) {
        best = diff;
        signed_best = signed_diff;
        nearest = &queue[j];
      }
    }
    best_diffs_us[i] = best;
    signed_diffs_us[i] = signed_best;
    if (best > tolerance_us_) outside_tolerance = true;
    cameras[i] = nearest->image;
  }
  if (outside_tolerance) {
    debug_status("outside_tolerance", lidar.timestamp_us, best_diffs_us, signed_diffs_us);
    return false;
  }

  debug_status("synced", lidar.timestamp_us, best_diffs_us, signed_diffs_us);
  last_emitted_timestamp_us_ = lidar.timestamp_us;
  drop_older_than(lidar.timestamp_us - tolerance_us_);

  if (frame) {
    frame->timestamp_us = lidar.timestamp_us;
    frame->cameras = cameras;
    frame->lidar = lidar.points;
    frame->lidar_point_dim = lidar.point_dim;
  }
  return true;
}

void FrameSynchronizer::drop_older_than(int64_t timestamp_us) {
  while (!lidar_queue_.empty() && lidar_queue_.front().timestamp_us < timestamp_us) lidar_queue_.pop_front();
  for (size_t i = 0; i < camera_queues_.size(); ++i) {
    while (!camera_queues_[i].empty() && camera_queues_[i].front().timestamp_us < timestamp_us) {
      camera_queues_[i].pop_front();
    }
  }
}

void FrameSynchronizer::debug_status(const char* reason, int64_t lidar_timestamp_us,
                                     const std::vector<int64_t>& best_diffs_us,
                                     const std::vector<int64_t>& signed_diffs_us) {
  if (!debug_ || debug_count_ >= debug_limit_) return;
  ++debug_count_;
  std::cerr << "sync_debug reason=" << reason << " lidar_us=" << lidar_timestamp_us
            << " tolerance_ms=" << static_cast<double>(tolerance_us_) / 1000.0
            << " lidar_offset_ms=" << static_cast<double>(lidar_time_offset_us_) / 1000.0
            << " lidar_queue=" << lidar_queue_.size();
  for (size_t i = 0; i < camera_queues_.size(); ++i) {
    std::cerr << " cam[" << i << "]=" << camera_topics_[i] << " q=" << camera_queues_[i].size()
              << " offset_ms=" << static_cast<double>(camera_time_offsets_us_[i]) / 1000.0;
    if (i < best_diffs_us.size()) {
      std::cerr << " nearest_diff_ms=" << static_cast<double>(best_diffs_us[i]) / 1000.0;
      std::cerr << " signed_diff_ms=" << static_cast<double>(signed_diffs_us[i]) / 1000.0;
      std::cerr << " suggested_offset_ms="
                << static_cast<double>(camera_time_offsets_us_[i] - signed_diffs_us[i]) / 1000.0;
    }
  }
  std::cerr << std::endl;
}

}  // namespace rscl_adapter
