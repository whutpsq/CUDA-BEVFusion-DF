#include "rscl_adapter/bag_reader.hpp"

#include <stdexcept>

namespace rscl_adapter {

std::unique_ptr<BagReader> create_rscl_bag_reader(const std::string&,
                                                  const std::vector<std::string>&) {
  throw std::runtime_error(
      "rscl_bevfusion_bag_runner was built without an RSCL C++ bag backend. "
      "Implement create_rscl_bag_reader() with the vehicle RSCL SDK, or build with the SDK backend source.");
}

}  // namespace rscl_adapter
