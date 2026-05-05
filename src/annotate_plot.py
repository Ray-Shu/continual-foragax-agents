import logging
from typing import Dict, Optional, Any

import numpy as np
from matplotlib.axes import Axes

from plotting_utils import FONTSIZE, get_mapped_label


def annotate_plot(
    ax: Axes,
    label_map: Optional[Dict[str, str]] = None,
    color_map: Optional[Dict[str, Any]] = None,
    flip: bool = False,
    disable_fov: bool = False,
):
    """Annotates lines on a plot and removes the legend."""
    logger = logging.getLogger(__name__)

    legend = ax.get_legend()
    if legend:
        _, labels = ax.get_legend_handles_labels()
        logger.info(f"Legend labels: {labels}")
        lines_with_data = [
            line
            for line in ax.get_lines()
            if np.asarray(line.get_xydata()).shape[0] > 0
            and line.get_label().startswith("_")  # type: ignore
        ]
        logger.info(f"Lines with data (startswith _): {len(lines_with_data)}")
        if len(lines_with_data) == len(labels):
            for line, label in zip(lines_with_data, labels, strict=True):
                mapped_label = get_mapped_label(label, label_map, disable_fov=disable_fov)
                line.set_label(mapped_label)
        legend.remove()

    lines = ax.get_lines()
    lines_to_label = [
        line
        for line in lines
        if (
            np.asarray(line.get_xydata()).shape[0] > 0
            and isinstance(line.get_label(), str)
            and not line.get_label().startswith("_")  # type: ignore
        )
    ]
    logger.info(
        f"Lines to label: {len(lines_to_label)}, labels: {[line.get_label() for line in lines_to_label]}"
    )

    if lines_to_label:
        # Calculate intelligent offsets based on line data and confidence intervals
        num_lines = len(lines_to_label)
        final_ys = np.array(
            [np.asarray(line.get_ydata())[-1] for line in lines_to_label]
        )

        # Get CI bounds for each line (x-dependent)
        ci_lows_arrays = []
        ci_highs_arrays = []
        ci_xdata_arrays = []
        all_lines = ax.get_lines()
        collections = ax.collections
        for line in lines_to_label:
            try:
                idx = all_lines.index(line)
                if idx < len(collections):
                    collection = collections[idx]
                    if hasattr(collection, "get_paths") and collection.get_paths():
                        path = collection.get_paths()[0]
                        vertices = np.asarray(path.vertices)
                        n = len(vertices) // 2
                        ci_x = vertices[:n, 0]
                        low_ci = vertices[:n, 1]
                        high_ci = vertices[n:, 1][::-1]
                        # Ensure same length
                        min_len = min(len(ci_x), len(low_ci), len(high_ci))
                        ci_lows_arrays.append(low_ci[:min_len])
                        ci_highs_arrays.append(high_ci[:min_len])
                        ci_xdata_arrays.append(ci_x[:min_len])
                    else:
                        xdata = np.asarray(line.get_xdata())
                        ydata = np.asarray(line.get_ydata())
                        ci_lows_arrays.append(ydata - 0.05)
                        ci_highs_arrays.append(ydata + 0.05)
                        ci_xdata_arrays.append(xdata)
                else:
                    xdata = np.asarray(line.get_xdata())
                    ydata = np.asarray(line.get_ydata())
                    ci_lows_arrays.append(ydata - 0.05)
                    ci_highs_arrays.append(ydata + 0.05)
                    ci_xdata_arrays.append(xdata)
            except ValueError:
                xdata = np.asarray(line.get_xdata())
                ydata = np.asarray(line.get_ydata())
                ci_lows_arrays.append(ydata - 0.05)
                ci_highs_arrays.append(ydata + 0.05)
                ci_xdata_arrays.append(xdata)

        # Calculate text dimensions
        ylim = ax.get_ylim()
        y_range = ylim[1] - ylim[0]
        xlim = ax.get_xlim()
        fig_size_inches = ax.figure.get_size_inches()
        dpi = ax.figure.dpi
        fig_height_points = fig_size_inches[1] * dpi
        fig_width_points = fig_size_inches[0] * dpi

        y_data_per_point = y_range / fig_height_points
        x_data_per_point = (xlim[1] - xlim[0]) / fig_width_points

        text_height = FONTSIZE * y_data_per_point
        margin = text_height * 1.0

        # X offsets and Y offsets
        xoffsets = np.zeros(num_lines)
        yoffsets = np.zeros(num_lines)

        # Sort lines by final y position (highest to lowest)
        sorted_indices = np.argsort(final_ys)[::-1]

        # Track placed labels: list of (x_min, x_max, y_min, y_max)
        placed_labels = []

        def get_label_bbox(x_label, y_label, label_text):
            """Calculate bounding box for label.

            Note: labelLines centers text at the given position, so we need to
            account for text width extending both left and right of x_label.
            """
            lines = label_text.split("\n")
            max_line_length = max(len(line) for line in lines)
            num_lines = len(lines)
            width = max_line_length * FONTSIZE * 0.8 * x_data_per_point
            height = num_lines * text_height
            return (
                x_label - width / 2,  # Center the text horizontally
                x_label + width / 2,
                y_label - height / 2,
                y_label + height / 2,
            )

        def check_overlap_with_line(x_min, x_max, y_min, y_max, line_xdata, line_ydata):
            """Check if bbox overlaps with line in x-y space."""
            # Sample line at multiple points within bbox x range
            if x_max < line_xdata[0] or x_min > line_xdata[-1]:
                return False

            x_sample = np.linspace(
                max(x_min, line_xdata[0]), min(x_max, line_xdata[-1]), 50
            )
            y_line_samples = np.interp(x_sample, line_xdata, line_ydata)

            # Check if any point on line falls within bbox (with margin)
            for y_line in y_line_samples:
                if y_min - margin <= y_line <= y_max + margin:
                    return True
            return False

        def check_overlap_with_ci(
            x_min, x_max, y_min, y_max, ci_xdata, ci_low, ci_high
        ):
            """Check if bbox overlaps with CI band in x-y space."""
            if x_max < ci_xdata[0] or x_min > ci_xdata[-1]:
                return False

            x_sample = np.linspace(
                max(x_min, ci_xdata[0]), min(x_max, ci_xdata[-1]), 50
            )
            ci_low_samples = np.interp(x_sample, ci_xdata, ci_low)
            ci_high_samples = np.interp(x_sample, ci_xdata, ci_high)

            # Check if bbox overlaps with CI band at any x (with margin)
            for ci_l, ci_h in zip(ci_low_samples, ci_high_samples, strict=True):
                if not (y_max + margin < ci_l or y_min - margin > ci_h):
                    return True
            return False

        def check_overlap_with_bbox(x_min, x_max, y_min, y_max, other_bbox):
            """Check if two bboxes overlap."""
            ox_min, ox_max, oy_min, oy_max = other_bbox
            x_overlap = not (x_max + margin < ox_min or x_min - margin > ox_max)
            y_overlap = not (y_max + margin < oy_min or y_min - margin > oy_max)
            return x_overlap and y_overlap

        def distance_to_ci(y_label, idx, x_label):
            """Calculate minimum distance from y_label to CI at x_label."""
            ci_xdata = ci_xdata_arrays[idx]
            ci_low = ci_lows_arrays[idx]
            ci_high = ci_highs_arrays[idx]

            if x_label < ci_xdata[0] or x_label > ci_xdata[-1]:
                return float("inf")

            ci_l = np.interp(x_label, ci_xdata, ci_low)
            ci_h = np.interp(x_label, ci_xdata, ci_high)

            return min(abs(y_label - ci_l), abs(y_label - ci_h))

        # Store label positions for manual placement
        label_positions = []

        fallback_count = 0

        for idx in sorted_indices:
            xdata = np.asarray(lines_to_label[idx].get_xdata())
            ydata = np.asarray(lines_to_label[idx].get_ydata())
            label_text = str(lines_to_label[idx].get_label())

            logger.info(f"Placing label for line {idx} ({label_text})")
            logger.info(
                f"  Line data range: x=[{xdata[0]:.1f}, {xdata[-1]:.1f}], y=[{ydata.min():.3f}, {ydata.max():.3f}]"
            )
            logger.info(
                f"  Line y at end: {ydata[-1]:.3f}, CI: [{ci_lows_arrays[idx][-1]:.3f}, {ci_highs_arrays[idx][-1]:.3f}]"
            )

            # Allow searching across the entire plot for better label placement
            x_min_search = xlim[0]
            x_max_search = xlim[1]

            # Calculate label width to adjust search bounds
            # Since labels are centered, we need half-width margin on each side
            lines = label_text.split("\n")
            max_line_length = max(len(line) for line in lines)
            num_lines = len(lines)
            label_width = max_line_length * FONTSIZE * 0.6 * x_data_per_point
            label_height = num_lines * text_height
            half_width = label_width / 2
            half_height = label_height / 2

            # Adjust search space to ensure label bbox stays within bounds
            # Label center must be at least half_width/half_height away from edges
            x_min_search_adjusted = max(
                x_min_search, xlim[0] + half_width + 0.05 * (xlim[1] - xlim[0])
            )
            x_max_search_adjusted = min(x_max_search, xlim[1] - half_width)
            y_min_search_adjusted = ylim[0] + half_height
            y_max_search_adjusted = ylim[1] - half_height

            # If there's no valid search space, use the original bounds
            if x_min_search_adjusted >= x_max_search_adjusted:
                x_min_search_adjusted = x_min_search
                x_max_search_adjusted = x_max_search
                logger.warning(
                    f"  Label too wide ({label_width:.1f}) for plot width, "
                    f"using unadjusted search space"
                )

            # Try candidate positions across plot
            x_candidates = np.linspace(x_min_search_adjusted, x_max_search_adjusted, 30)
            y_candidates = np.linspace(y_min_search_adjusted, y_max_search_adjusted, 50)

            logger.info(
                f"  Search space: x=[{x_min_search_adjusted:.1f}, {x_max_search_adjusted:.1f}], "
                f"y=[{y_min_search_adjusted:.3f}, {y_max_search_adjusted:.3f}] "
                f"(label_width={label_width:.1f}, label_height={label_height:.3f})"
            )
            logger.info(
                f"  X candidates: {len(x_candidates)}, Y candidates: {len(y_candidates)}"
            )

            best_position = None
            best_score = float("inf")
            candidates_tried = 0
            candidates_rejected = {
                "bounds": 0,
                "own_line": 0,
                "own_ci": 0,
                "other_labels": 0,
                "other_lines": 0,
            }

            # Track valid positions near the target line
            valid_near_line = []

            for x_label in x_candidates:
                for y_label in y_candidates:
                    candidates_tried += 1
                    bbox = get_label_bbox(x_label, y_label, label_text)
                    x_min, x_max, y_min, y_max = bbox

                    # Priority 1: Stay within bounds
                    if (
                        x_min < xlim[0]
                        or x_max > xlim[1]
                        or y_min < ylim[0]
                        or y_max > ylim[1]
                    ):
                        candidates_rejected["bounds"] += 1
                        continue

                    # Priority 2: No overlap with own line
                    if check_overlap_with_line(
                        x_min, x_max, y_min, y_max, xdata, ydata
                    ):
                        candidates_rejected["own_line"] += 1
                        continue

                    # Priority 3: No overlap with own CI
                    if check_overlap_with_ci(
                        x_min,
                        x_max,
                        y_min,
                        y_max,
                        ci_xdata_arrays[idx],
                        ci_lows_arrays[idx],
                        ci_highs_arrays[idx],
                    ):
                        candidates_rejected["own_ci"] += 1
                        continue

                    # Priority 4: No overlap with other labels
                    if any(
                        check_overlap_with_bbox(x_min, x_max, y_min, y_max, other)
                        for other in placed_labels
                    ):
                        candidates_rejected["other_labels"] += 1
                        continue

                    # Priority 5: No overlap with other lines
                    overlap_other_line = False
                    for j in range(num_lines):
                        if j != idx:
                            other_xdata = np.asarray(lines_to_label[j].get_xdata())
                            other_ydata = np.asarray(lines_to_label[j].get_ydata())
                            if check_overlap_with_line(
                                x_min, x_max, y_min, y_max, other_xdata, other_ydata
                            ):
                                overlap_other_line = True
                                break
                    if overlap_other_line:
                        candidates_rejected["other_lines"] += 1
                        continue

                    # Priority 6: Minimize distance to own CI
                    dist = distance_to_ci(y_label, idx, x_label)

                    # Priority 7: Penalize overlap with other CIs (but allow it)
                    # Use small penalty to prefer whitespace but still allow CI overlap
                    penalty = 0
                    for j in range(num_lines):
                        if j != idx:
                            if check_overlap_with_ci(
                                x_min,
                                x_max,
                                y_min,
                                y_max,
                                ci_xdata_arrays[j],
                                ci_lows_arrays[j],
                                ci_highs_arrays[j],
                            ):
                                penalty += 0.1  # Small penalty, much less than distance to own CI

                    score = dist + penalty

                    # Track valid positions near the line
                    y_line_at_x = np.interp(x_label, xdata, ydata)
                    if abs(y_label - y_line_at_x) < 0.2:  # Within 0.2 of line
                        valid_near_line.append((x_label, y_label, score))

                    if score < best_score:
                        best_score = score
                        best_position = (x_label, y_label, bbox)

            if best_position:
                x_label, y_label, bbox = best_position

                # Calculate the line's y-value at the label's x position
                y_line_at_label = np.interp(x_label, xdata, ydata)

                xoffsets[idx] = x_label - xdata[-1]
                yoffsets[idx] = y_label - y_line_at_label
                placed_labels.append(bbox)

                # Store absolute position for manual label placement
                label_color = lines_to_label[idx].get_color()
                mapped_label = get_mapped_label(label_text, label_map, disable_fov=disable_fov)
                label_positions.append(
                    {
                        "x": x_label,
                        "y": y_label,
                        "text": mapped_label,
                        "color": label_color,
                    }
                )

                # Validate placement
                x_min, x_max, y_min, y_max = bbox
                has_overlap_line = check_overlap_with_line(
                    x_min, x_max, y_min, y_max, xdata, ydata
                )
                has_overlap_ci = check_overlap_with_ci(
                    x_min,
                    x_max,
                    y_min,
                    y_max,
                    ci_xdata_arrays[idx],
                    ci_lows_arrays[idx],
                    ci_highs_arrays[idx],
                )

                # Calculate actual bbox for logging
                final_bbox = get_label_bbox(x_label, y_label, label_text)
                bbox_x_min, bbox_x_max, bbox_y_min, bbox_y_max = final_bbox
                bbox_width = bbox_x_max - bbox_x_min

                logger.info(
                    f"Placed line {idx} at x={x_label:.3f}, y={y_label:.3f}, "
                    f"xoff={xoffsets[idx]:.3f}, yoff={yoffsets[idx]:.3f}, "
                    f"y_line_at_label={y_line_at_label:.3f}, "
                    f"score={best_score:.3f}, overlap_line={has_overlap_line}, overlap_ci={has_overlap_ci}"
                )
                logger.info(
                    f"  Label position: x={x_label:.1f} (in [{xdata[0]:.1f}, {xdata[-1]:.1f}]), y={y_label:.3f} (in [{ylim[0]:.3f}, {ylim[1]:.3f}])"
                )
                logger.info(
                    f"  Label bbox: x=[{bbox_x_min:.1f}, {bbox_x_max:.1f}] (width={bbox_width:.1f}), y=[{bbox_y_min:.3f}, {bbox_y_max:.3f}]"
                )
                logger.info(
                    f"  Bbox in bounds: x=[{bbox_x_min >= xlim[0]}, {bbox_x_max <= xlim[1]}], y=[{bbox_y_min >= ylim[0]}, {bbox_y_max <= ylim[1]}]"
                )
                logger.info(
                    f"  Candidates tried: {candidates_tried}, rejected: {candidates_rejected}"
                )
                logger.info(
                    f"  Valid positions near line (within 0.2): {len(valid_near_line)}"
                )
                if len(valid_near_line) > 0:
                    # Sort by score
                    valid_near_line_sorted = sorted(valid_near_line, key=lambda x: x[2])
                    for vx, vy, vscore in valid_near_line_sorted[:10]:
                        logger.info(f"    x={vx:.1f}, y={vy:.3f}, score={vscore:.3f}")
            else:
                # Fallback: place at end with offset to avoid overlap
                fallback_count += 1
                xoffsets[idx] = (fallback_count % 3 - 1) * 0.05 * (xlim[1] - xlim[0])
                yoffsets[idx] = (
                    ((fallback_count // 3) % 3 - 1) * 0.05 * (ylim[1] - ylim[0])
                )
                logger.warning(
                    f"Could not find valid position for line {idx}, using fallback with offset"
                )
                logger.warning(
                    f"  Candidates tried: {candidates_tried}, rejected: {candidates_rejected}"
                )
                logger.warning(
                    f"  Valid positions near line (within 0.2): {len(valid_near_line)}"
                )

        # Manually place labels using ax.text() with precise data coordinates
        logger.info(f"\nPlacing {len(label_positions)} labels manually:")
        for label_info in label_positions:
            ax.text(
                label_info["x"],
                label_info["y"],
                label_info["text"],
                color=label_info["color"],
                fontsize=FONTSIZE,
                ha="center",  # Horizontal alignment: center
                va="center",  # Vertical alignment: center
                clip_on=False,  # Allow labels to extend beyond plot bounds if needed
                zorder=100,  # Place labels on top of other elements
            )
            logger.info(
                f"  Placed '{label_info['text']}' at x={label_info['x']:.1f}, y={label_info['y']:.3f}"
            )

        logger.info(f"Successfully placed {len(label_positions)} labels")
