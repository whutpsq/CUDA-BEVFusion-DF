#!/usr/bin/env bash
set -euo pipefail

real_nvcc="${REAL_NVCC:-/usr/local/cuda/bin/nvcc}"
if [ ! -x "$real_nvcc" ]; then
  echo "Real NVCC is not executable: $real_nvcc" >&2
  exit 1
fi

arguments=()
for argument in "$@"; do
  # FindCUDA also embeds -std=c++14 inside its comma-delimited -Xcompiler
  # argument.  Replace every occurrence so NVCC's device frontend and the
  # host compiler parse each .cu translation unit with the same dialect.
  argument="${argument//--std=c++14/--std=c++17}"
  argument="${argument//-std=c++14/-std=c++17}"
  arguments+=("$argument")
done

exec "$real_nvcc" "${arguments[@]}"
