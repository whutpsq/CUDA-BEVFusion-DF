/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 */

#ifndef __HEAD_MAP_HPP__
#define __HEAD_MAP_HPP__

#include <memory>
#include <string>
#include <vector>

#include "common/dtype.hpp"

namespace bevfusion {
namespace head {
namespace map {

struct MapParameter {
  std::string model;
  std::string input = "middle";
  std::string output = "map";
};

struct MapOutput {
  std::vector<int> shape;
  std::vector<float> data;
};

class MapSeg {
 public:
  virtual ~MapSeg() = default;
  virtual MapOutput forward(const nvtype::half* transfusion_feature, void* stream) = 0;
  virtual void print() = 0;
};

std::shared_ptr<MapSeg> create_mapseg(const MapParameter& param);

};  // namespace map
};  // namespace head
};  // namespace bevfusion

#endif  // __HEAD_MAP_HPP__
