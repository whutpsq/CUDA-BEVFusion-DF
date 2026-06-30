#include "rscl_adapter/online_node.hpp"

#include <stdexcept>

namespace rscl_adapter {

std::unique_ptr<OnlineNode> create_rscl_online_node(const AdapterConfig&) {
  throw std::runtime_error(
      "rscl_bevfusion_online_node was built without an RSCL C++ online backend. "
      "Implement create_rscl_online_node() with the vehicle RSCL SDK, or build with "
      "-DRSCL_ONLINE_BACKEND_SOURCE=/path/to/vehicle_rscl_online_node.cpp.");
}

}  // namespace rscl_adapter
