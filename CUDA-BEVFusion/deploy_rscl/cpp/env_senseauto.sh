#!/usr/bin/env bash

# Source this file before configuring or running the C++ rsclbag runner:
#   source deploy_rscl/cpp/env_senseauto.sh [/opt/senseauto_active]

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "This script must be sourced, not executed:" >&2
  echo "  source deploy_rscl/cpp/env_senseauto.sh [/opt/senseauto_active]" >&2
  exit 1
fi

_senseauto_add_path() {
  local var_name="$1"
  local path_value="$2"
  [[ -d "$path_value" ]] || return 0
  case ":${!var_name:-}:" in
    *":$path_value:"*) ;;
    *) export "$var_name=$path_value${!var_name:+:${!var_name}}" ;;
  esac
}

_senseauto_find_module() {
  local root="$1"
  local module="$2"
  local candidate

  for candidate in \
    "$root/tmp/$module" \
    "$root/$module" \
    "$root"/*/tmp/"$module" \
    "$root"/*/opt/senseauto/tmp/"$module" \
    "$root"/*/*/opt/senseauto/tmp/"$module"; do
    if [[ -d "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done

  return 1
}

_senseauto_root="${1:-${SENSEAUTO_INSTALL_ROOT:-}}"
if [[ -z "$_senseauto_root" ]]; then
  for _candidate_root in /opt/senseauto_active /opt/senseauto /opt/senseauto_install_path; do
    if [[ -d "$_candidate_root" ]]; then
      _senseauto_root="$_candidate_root"
      break
    fi
  done
fi

if [[ -z "$_senseauto_root" || ! -d "$_senseauto_root" ]]; then
  echo "Cannot find SenseAuto install root. Pass it explicitly:" >&2
  echo "  source deploy_rscl/cpp/env_senseauto.sh /opt/senseauto_active" >&2
  return 1
fi

export SENSEAUTO_INSTALL_ROOT="$_senseauto_root"

if _rscl_root="$(_senseauto_find_module "$_senseauto_root" senseauto-rscl)"; then
  export RSCL_SDK_ROOT="$_rscl_root"
  _senseauto_add_path LD_LIBRARY_PATH "$_rscl_root/lib"
  _senseauto_add_path PATH "$_rscl_root/bin"
fi

if _msgs_root="$(_senseauto_find_module "$_senseauto_root" senseauto-msgs)"; then
  export RSCL_MSGS_ROOT="$_msgs_root"
  _senseauto_add_path LD_LIBRARY_PATH "$_msgs_root/lib"
fi

if _thirdparty_root="$(_senseauto_find_module "$_senseauto_root" senseauto-3rdparty)"; then
  export RSCL_THIRDPARTY_ROOT="$_thirdparty_root"
  _senseauto_add_path LD_LIBRARY_PATH "$_thirdparty_root/3rdparty/lib"
fi

echo "SENSEAUTO_INSTALL_ROOT=$SENSEAUTO_INSTALL_ROOT"
echo "RSCL_SDK_ROOT=${RSCL_SDK_ROOT:-<not found>}"
echo "RSCL_MSGS_ROOT=${RSCL_MSGS_ROOT:-<not found>}"
echo "RSCL_THIRDPARTY_ROOT=${RSCL_THIRDPARTY_ROOT:-<not found>}"
