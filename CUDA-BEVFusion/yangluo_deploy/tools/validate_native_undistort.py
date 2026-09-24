#!/usr/bin/env python3
"""Compare the Yangluo native remap contract with OpenCV CV_16SC2."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


INTER_BITS = 5
INTER_SIZE = 1 << INTER_BITS
INTER_AREA = INTER_SIZE * INTER_SIZE


def native_map(camera_matrix, distortion, width, height):
    inverse = np.linalg.inv(camera_matrix)
    x0 = np.empty((height, width), dtype=np.int32)
    y0 = np.empty((height, width), dtype=np.int32)
    fractions = np.empty((height, width), dtype=np.uint16)
    k1, k2, p1, p2, k3, k4, k5, k6 = distortion
    columns = np.arange(width, dtype=np.float64)[None, :]

    for row_start in range(0, height, 64):
        row_end = min(height, row_start + 64)
        rows = np.arange(row_start, row_end, dtype=np.float64)[:, None]
        homogeneous_x = inverse[0, 0] * columns + inverse[0, 1] * rows + inverse[0, 2]
        homogeneous_y = inverse[1, 0] * columns + inverse[1, 1] * rows + inverse[1, 2]
        homogeneous_w = inverse[2, 0] * columns + inverse[2, 1] * rows + inverse[2, 2]
        x = homogeneous_x / homogeneous_w
        y = homogeneous_y / homogeneous_w
        r2 = x * x + y * y
        r4 = r2 * r2
        r6 = r4 * r2
        radial = (1.0 + k1 * r2 + k2 * r4 + k3 * r6) / (
            1.0 + k4 * r2 + k5 * r4 + k6 * r6
        )
        distorted_x = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
        distorted_y = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
        projected_x = (
            camera_matrix[0, 0] * distorted_x
            + camera_matrix[0, 1] * distorted_y
            + camera_matrix[0, 2]
        )
        projected_y = (
            camera_matrix[1, 0] * distorted_x
            + camera_matrix[1, 1] * distorted_y
            + camera_matrix[1, 2]
        )
        projected_w = (
            camera_matrix[2, 0] * distorted_x
            + camera_matrix[2, 1] * distorted_y
            + camera_matrix[2, 2]
        )
        quantized_x = np.rint(projected_x / projected_w * INTER_SIZE).astype(np.int32)
        quantized_y = np.rint(projected_y / projected_w * INTER_SIZE).astype(np.int32)
        base_x = np.floor_divide(quantized_x, INTER_SIZE)
        base_y = np.floor_divide(quantized_y, INTER_SIZE)
        fraction_x = quantized_x - base_x * INTER_SIZE
        fraction_y = quantized_y - base_y * INTER_SIZE
        x0[row_start:row_end] = base_x
        y0[row_start:row_end] = base_y
        fractions[row_start:row_end] = (fraction_y << INTER_BITS) | fraction_x
    return x0, y0, fractions


def native_remap(image, x0, y0, fractions):
    height, width = x0.shape
    output = np.empty_like(image)
    for row_start in range(0, height, 32):
        row_end = min(height, row_start + 32)
        bx = x0[row_start:row_end]
        by = y0[row_start:row_end]
        fx = (fractions[row_start:row_end] & (INTER_SIZE - 1)).astype(np.int32)
        fy = (fractions[row_start:row_end] >> INTER_BITS).astype(np.int32)
        weights = (
            (INTER_SIZE - fx) * (INTER_SIZE - fy),
            fx * (INTER_SIZE - fy),
            (INTER_SIZE - fx) * fy,
            fx * fy,
        )
        accumulator = np.zeros((row_end - row_start, width, 3), dtype=np.int32)
        for dx, dy, weight in ((0, 0, weights[0]), (1, 0, weights[1]),
                               (0, 1, weights[2]), (1, 1, weights[3])):
            sample_x = bx + dx
            sample_y = by + dy
            valid = ((sample_x >= 0) & (sample_x < width) &
                     (sample_y >= 0) & (sample_y < height))
            clipped_x = np.clip(sample_x, 0, width - 1)
            clipped_y = np.clip(sample_y, 0, height - 1)
            pixels = image[clipped_y, clipped_x].astype(np.int32)
            accumulator += pixels * (weight * valid).astype(np.int32)[..., None]
        output[row_start:row_end] = ((accumulator + INTER_AREA // 2) // INTER_AREA).astype(np.uint8)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration", required=True)
    args = parser.parse_args()
    calibration = json.loads(Path(args.calibration).read_text(encoding="utf-8"))
    failures = 0
    for name, camera in calibration["cameras"].items():
        matrix = np.asarray(camera["camera_intrinsics"], dtype=np.float64)[:3, :3]
        distortion = np.asarray(camera["distortion"], dtype=np.float64)
        width = int(camera["width"])
        height = int(camera["height"])
        reference_map1, reference_map2 = cv2.initUndistortRectifyMap(
            matrix, distortion, None, matrix, (width, height), cv2.CV_16SC2
        )
        x0, y0, fractions = native_map(matrix, distortion, width, height)
        map_mismatches = int(np.count_nonzero(reference_map1[..., 0].astype(np.int32) != x0))
        map_mismatches += int(np.count_nonzero(reference_map1[..., 1].astype(np.int32) != y0))
        map_mismatches += int(np.count_nonzero(reference_map2.astype(np.uint16) != fractions))

        rows, columns = np.indices((height, width), dtype=np.uint32)
        image = np.stack(((columns * 17 + rows * 3) & 255,
                          (columns * 5 + rows * 11) & 255,
                          (columns * 7 + rows * 13) & 255), axis=-1).astype(np.uint8)
        reference = cv2.remap(image, reference_map1, reference_map2, cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT)
        candidate = native_remap(image, x0, y0, fractions)
        difference = np.abs(reference.astype(np.int16) - candidate.astype(np.int16))
        differing_values = int(np.count_nonzero(difference))
        maximum_difference = int(difference.max())
        print(f"camera={name} map_mismatches={map_mismatches} "
              f"pixel_value_mismatches={differing_values} max_abs_diff={maximum_difference}")
        if map_mismatches or maximum_difference > 1:
            failures += 1
    if failures:
        raise SystemExit(f"NATIVE_UNDISTORT_VALIDATION_FAILED cameras={failures}")
    print("NATIVE_UNDISTORT_MATCHES_OPENCV")


if __name__ == "__main__":
    main()
