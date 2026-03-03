import logging

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import polars as pl
import seaborn as sns
import tol_colors as tc

from annotate_plot import annotate_plot
from plotting_utils import (
    COLOR_MAP,
    LABEL_MAP,
    PlottingArgumentParser,
    despine,
    filter_by_alg_aperture,
    format_metric_name,
    get_mapped_label,
    get_ylabel_mapping,
    load_data,
    parse_plotting_args,
    save_plot,
)


def main():
    parser = PlottingArgumentParser(description="Plot learning curves.")
    parser.add_argument(
        "--metric",
        type=str,
        default=None,
        help="Metric to plot on the y-axis (deprecated: use --metrics).",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=None,
        help="Metrics to plot on the y-axis. Multiple metrics will be plotted as subplots.",
    )
    parser.add_argument(
        "--sample-type",
        type=str,
        default="every",
        help="Sample type to filter from the data.",
    )
    parser.add_argument(
        "--ylim",
        type=float,
        nargs=2,
        default=None,
        help="Y-axis limits for the plot.",
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help="Minimum frame to include in the plot",
    )
    parser.add_argument(
        "--end-frame",
        type=int,
        default=None,
        help="Maximum frame to include in the plot",
    )
    parser.add_argument(
        "--legend",
        action="store_true",
        help="Use legend instead of auto-labeling",
    )
    parser.add_argument(
        "--normalize",
        type=str,
        default=None,
        help="Algorithm to normalize against.",
    )
    parser.add_argument(
        "--plot-all-seeds",
        action="store_true",
        help="Plot all individual seeds as separate lines instead of confidence intervals.",
    )
    parser.add_argument(
        "--subplot-by-seed",
        action="store_true",
        help="Create a grid of subplots, one per seed.",
    )
    parser.add_argument(
        "--vertical-lines",
        type=int,
        nargs="+",
        default=None,
        help="Frame numbers to draw vertical lines at.",
    )
    parser.add_argument(
        "--horizontal-lines",
        type=str,
        nargs="+",
        default=None,
        help="Horizontal lines to draw (e.g. upper bounds). "
             "Format: 'value:label' or just 'value'. "
             "Example: --horizontal-lines 0.95:Upperbound 0.5:Random",
    )
    parser.add_argument(
        "--plot-avg",
        action="store_true",
        help="Plot the Average Value alongside the learning curve.",
    )
    parser.add_argument(
        "--grid",
        type=str,
        nargs="+",
        default=None,
        help="Grid layout specification. Format: 'nrows,ncols cell1 cell2 ...'. "
             "Each cell can contain multiple algorithms separated by '+'. "
             "Example: '2,2 DQN EQRC DQN+EQRC PPO+A2C' creates a 2x2 grid where "
             "the first two cells show single algorithms, and the last two cells "
             "show comparisons of multiple algorithms.",
    )
    parser.add_argument(
        "--color-algs",
        type=str,
        nargs="+",
        default=None,
        help="List of algorithms to determine the color of each algorithm for consistent coloring across plots.",
    )

    parser.add_argument(
        "--disable-fov",
        action="store_true",
        help="Disable inserting the FOV (aperture) in the legend name.",
    )

    args = parse_plotting_args(parser)

    # Parse horizontal lines specification
    horizontal_lines = []
    if args.horizontal_lines:
        for spec in args.horizontal_lines:
            if ":" in spec:
                value_str, label = spec.split(":", 1)
            else:
                value_str, label = spec, None
            horizontal_lines.append((float(value_str), label))

    # Handle backward compatibility: --metric takes precedence over --metrics default
    if args.metric:
        args.metrics = [args.metric]
    elif args.metrics is None:
        args.metrics = ["ewm_reward"]

    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

    # Load and filter data
    df = load_data(args.experiment_path)
    logger.info(f"After load_data: {df.select('alg').unique().to_pandas()}")
    logger.info(f"DataFrame columns: {df.columns}")

    df = filter_by_alg_aperture(df, args.filter_alg_apertures)
    logger.info(
        f"After filter_by_alg_aperture: {df.select('alg').unique().to_pandas()}"
    )

    logger.info(
        f"Sample types present: {df.select('sample_type').unique().to_pandas()}"
    )
    df = df.filter(pl.col("sample_type") == args.sample_type)
    logger.info(f"After sample_type filter: {df.select('alg').unique().to_pandas()}")

    if args.filter_algs:
        df = df.filter(pl.col("alg").is_in(args.filter_algs))

    hue_col = "alg"
    if args.filter_alg_apertures:
        df = df.with_columns(
            pl.when(pl.col("aperture").is_not_null())
            .then(pl.col("alg") + ":" + pl.col("aperture").cast(pl.Utf8))
            .otherwise(pl.col("alg"))
            .alias("alg_ap")
        )
        hue_col = "alg_ap"
        logger.info(
            f"After creating alg_ap: {df.select('alg_ap').unique().to_pandas()}"
        )

    if args.filter_seeds:
        df = df.filter(pl.col("seed").is_in(args.filter_seeds))

    df = df.filter(pl.col("frame") >= args.start_frame)
    if args.end_frame is not None:
        df = df.filter(pl.col("frame") <= args.end_frame)

    # # Patch temperature data for Weather environments (after filtering to reduce memory usage)
    env = df["env"][0]
    # if "Weather" in env:
    #     df = patch_temperature_data(df)
    #     logger.info("Patched temperature data into dataframe")

    logger.info(f"Final df shape: {df.shape}")
    logger.info(f"Final unique {hue_col}: {df.select(hue_col).unique().to_pandas()}")

    # Sort by order specified in arguments
    if args.filter_alg_apertures:
        alg_ap_order = {ap: i for i, ap in enumerate(args.filter_alg_apertures)}
        df = df.with_columns(pl.col("alg_ap").replace(alg_ap_order).alias("order_col"))
        df = df.sort("order_col")
    elif args.filter_algs:
        alg_order = {alg: i for i, alg in enumerate(args.filter_algs)}
        df = df.with_columns(pl.col("alg").replace(alg_order).alias("order_col"))
        df = df.sort("order_col")

    # Normalization
    if args.normalize:
        for metric in args.metrics:
            norm_df = df.filter(pl.col("alg") == args.normalize)
            norm_df = norm_df.select(
                [pl.col("frame"), pl.col("seed"), pl.col(metric).alias("norm_val")]
            )

            df = df.join(norm_df, on=["frame", "seed"], how="left")
            df = df.with_columns((pl.col(metric) / pl.col("norm_val")).alias(metric))
            df = df.drop("norm_val")

    # Plotting
    num_metrics = len(args.metrics)

    # Parse grid specification if provided
    grid_cells = None  # List of lists, each inner list contains algorithms for that cell
    grid_nrows = None
    grid_ncols = None
    if args.grid:
        # Handle both formats:
        # 1. Separate args: --grid 2,2 DQN EQRC PPO A2C
        # 2. Single quoted string: --grid "2,2 DQN EQRC PPO A2C"
        grid_args = args.grid
        # If first element contains spaces, it was passed as a single quoted string
        if len(grid_args) == 1 and " " in grid_args[0]:
            grid_args = grid_args[0].split()

        # First element is 'nrows,ncols', rest are cell specifications
        grid_dims = grid_args[0].split(",")
        grid_nrows = int(grid_dims[0])
        grid_ncols = int(grid_dims[1])
        # Each cell spec can have multiple algorithms separated by '+'
        grid_cells = [cell.split("+") for cell in grid_args[1:]]

        if len(grid_cells) > grid_nrows * grid_ncols:
            raise ValueError(
                f"Too many cells ({len(grid_cells)}) for grid size {grid_nrows}x{grid_ncols}"
            )

    # Determine subplot layout
    if args.grid:
        assert grid_nrows is not None and grid_ncols is not None and grid_cells is not None
        fig, axes = plt.subplots(
            grid_nrows, grid_ncols, layout="constrained", figsize=(6 * grid_ncols, 4 * grid_nrows)
        )
        # Flatten axes array for easier indexing
        if grid_nrows == 1 and grid_ncols == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
    elif args.subplot_by_seed:
        unique_seeds = df.select("seed").unique().sort("seed").to_series().to_list()
        num_seeds = len(unique_seeds)

        # Calculate grid dimensions
        import math

        ncols = math.ceil(math.sqrt(num_seeds))
        nrows = math.ceil(num_seeds / ncols)

        fig, axes = plt.subplots(
            nrows, ncols, layout="constrained", figsize=(6 * ncols, 4 * nrows)
        )
        # Flatten axes array for easier indexing
        if num_seeds == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if nrows > 1 or ncols > 1 else [axes]
    elif num_metrics == 1:
        if args.plot_avg:
            fig, axes = plt.subplots(1, 2, layout="constrained", gridspec_kw={'width_ratios': [3, 1]}, figsize=(10, 6))
            axes = axes.reshape(1, 2)
        else:
            fig, axes = plt.subplots(1, 1, layout="constrained")
            axes = [axes]  # Make it a list for consistent handling
    else:
        if args.plot_avg:
            fig, axes = plt.subplots(
                num_metrics, 2, layout="constrained", figsize=(10, 6 * num_metrics), gridspec_kw={'width_ratios': [3, 1]}
            )
        else:
            fig, axes = plt.subplots(
                num_metrics, 1, layout="constrained", figsize=(8, 6 * num_metrics)
            )

    present_hues = df.select(hue_col).unique().to_series().to_list()
    if args.color_algs:
        hue_order = [h for h in args.color_algs if h in present_hues]
        hue_order += [h for h in present_hues if h not in hue_order]
    elif args.filter_alg_apertures:
        hue_order = [h for h in args.filter_alg_apertures if h in present_hues]
        hue_order += [h for h in present_hues if h not in hue_order]
    elif args.filter_algs:
        hue_order = [h for h in args.filter_algs if h in present_hues]
        hue_order += [h for h in present_hues if h not in hue_order]
    else:
        hue_order = df.select(hue_col).unique(maintain_order=True).to_series().to_list()

    # Create color palette matching the order in filter_alg_apertures
    import seaborn as sns
    vibrant_colors = list(tc.colorsets["vibrant"])
    
    # If we need more than 7 colors, use seaborn's default categorical palette or colorblind which has 10
    total_algs_to_color = len(args.color_algs) if args.color_algs else (len(args.filter_alg_apertures) if args.filter_alg_apertures else len(hue_order))
    if total_algs_to_color > len(vibrant_colors):
        vibrant_colors = sns.color_palette("tab20", n_colors=20) # 20 colors should be enough

    if args.color_algs:
        # Map colors to algorithms in the order specified
        palette = {
            alg: vibrant_colors[i % len(vibrant_colors)]
            for i, alg in enumerate(args.color_algs)
        }
        # Ensure all items in hue_order have a color
        used_colors = len(args.color_algs)
        for h in hue_order:
            if h not in palette:
                palette[h] = vibrant_colors[used_colors % len(vibrant_colors)]
                used_colors += 1
    elif args.filter_alg_apertures:
        # Map colors to alg-aperture combinations in the order specified
        palette = {
            alg_ap: vibrant_colors[i % len(vibrant_colors)]
            for i, alg_ap in enumerate(args.filter_alg_apertures)
        }
    elif args.filter_algs:
        # Map colors: use COLOR_MAP if available, else fall back to cycling
        palette = {}
        fallback_idx = [0]
        for alg in args.filter_algs:
            palette[alg] = lookup_color(alg, fallback_idx)
    else:
        # Use default palette ordering
        palette = None

    if args.grid:
        assert grid_cells is not None and grid_nrows is not None and grid_ncols is not None
        # Grid plot with specified algorithms per cell
        metric = args.metrics[0]  # Use the first metric for all grid cells

        # Determine if we're using alg_ap based on cell specifications
        # If any cell contains ':', we're using alg_ap format (e.g., "DQN:5")
        use_alg_ap = any(":" in alg for cell in grid_cells for alg in cell)
        grid_hue_col = "alg_ap" if use_alg_ap else "alg"

        # Create alg_ap column if needed and it doesn't exist
        if use_alg_ap and "alg_ap" not in df.columns:
            df = df.with_columns(
                (pl.col("alg") + ":" + pl.col("aperture").cast(pl.Utf8)).alias("alg_ap")
            )

        for i, cell_algs in enumerate(grid_cells):
            ax = axes[i]

            # Build cell-specific palette for algorithms in this cell
            cell_palette = {
                alg: vibrant_colors[j % len(vibrant_colors)]
                for j, alg in enumerate(cell_algs)
            }

            # Filter data for algorithms in this cell
            cell_df_list = []
            missing_algs = []
            for alg in cell_algs:
                if use_alg_ap:
                    # Exact match for alg_ap (e.g., "DQN:5")
                    alg_df = df.filter(pl.col("alg_ap") == alg)
                else:
                    alg_df = df.filter(pl.col("alg") == alg)

                if alg_df.is_empty():
                    missing_algs.append(alg)
                else:
                    cell_df_list.append(alg_df)

            if not cell_df_list:
                logger.warning(f"No algorithms found for cell {i}, skipping")
                ax.text(0.5, 0.5, f"No data found", ha='center', va='center', transform=ax.transAxes)
                title = " + ".join(get_mapped_label(a, LABEL_MAP, disable_fov=args.disable_fov) for a in cell_algs)
                ax.set_title(title)
                despine(ax)
                continue

            if missing_algs:
                logger.warning(f"Algorithms not found in cell {i}: {missing_algs}")

            cell_df = pl.concat(cell_df_list)

            # Configure lineplot based on whether to show all seeds or confidence intervals
            lineplot_kwargs = {
                "data": cell_df.to_pandas(),
                "x": "frame",
                "y": metric,
                "hue": grid_hue_col,
                "hue_order": [a for a in cell_algs if a not in missing_algs],
                "palette": cell_palette,
                "ax": ax,
                "legend": "full" if len(cell_algs) > 1 else False,
            }

            if args.plot_all_seeds:
                lineplot_kwargs["units"] = "seed"
                lineplot_kwargs["estimator"] = None
                lineplot_kwargs["alpha"] = 0.05
            else:
                lineplot_kwargs["errorbar"] = ("ci", 95)

            sns.lineplot(**lineplot_kwargs)

            # Formatting
            formatted_metric = format_metric_name(metric)
            ylabel_map = get_ylabel_mapping(env)
            ylabel = ylabel_map.get(formatted_metric, formatted_metric)

            if args.normalize:
                ylabel = f"Normalized {ylabel}"

            # Set y-label only on leftmost column
            col_idx = i % grid_ncols
            if col_idx == 0:
                ax.set_ylabel(ylabel)
            else:
                ax.set_ylabel("")

            # Set x-label only on bottom row
            row_idx = i // grid_ncols
            if row_idx == grid_nrows - 1:
                ax.set_xlabel(r"Time steps $(\times 10^6)$")
            else:
                ax.set_xlabel("")

            # Title: show algorithm names (mapped)
            title = " + ".join(get_mapped_label(a, LABEL_MAP, disable_fov=args.disable_fov) for a in cell_algs if a not in missing_algs)
            ax.set_title(title)
            ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
            ax.xaxis.set_major_formatter(
                ticker.FuncFormatter(lambda x, _: f"{x / 1000000:g}")
            )
            ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
            despine(ax)

            # Handle legend for multi-algorithm cells
            if len(cell_algs) > 1:
                handles, labels = ax.get_legend_handles_labels()
                mapped_labels = [get_mapped_label(label, LABEL_MAP, disable_fov=args.disable_fov) for label in labels]
                ax.legend(handles, mapped_labels, title=None, frameon=False, loc='best')

            if args.vertical_lines:
                for x in args.vertical_lines:
                    ax.axvline(x=x, color="grey", linestyle=":", alpha=0.5)

            for hline_val, hline_label in horizontal_lines:
                ax.axhline(y=hline_val, color="grey", linestyle="--", alpha=0.7,
                           label=hline_label)
                if hline_label:
                    ax.annotate(
                        hline_label,
                        xy=(1, hline_val),
                        xycoords=("axes fraction", "data"),
                        ha="right", va="bottom", fontsize=8, color="grey",
                    )

            if args.ylim:
                ax.set_ylim(args.ylim)

        # Share y-axis limits across all grid cells
        if not args.ylim:
            # Collect all y-limits from non-empty cells
            y_mins = []
            y_maxs = []
            for j in range(len(grid_cells)):
                ax = axes[j]
                if ax.has_data():
                    ylim = ax.get_ylim()
                    y_mins.append(ylim[0])
                    y_maxs.append(ylim[1])

            if y_mins and y_maxs:
                global_ylim = (min(y_mins), max(y_maxs))
                for j in range(len(grid_cells)):
                    axes[j].set_ylim(global_ylim)

        # Hide extra subplots if grid is not fully filled
        for j in range(len(grid_cells), grid_nrows * grid_ncols):
            axes[j].axis("off")
    elif args.subplot_by_seed:
        # Plot each seed in its own subplot
        for i, seed in enumerate(unique_seeds):
            ax = axes[i]
            seed_df = df.filter(pl.col("seed") == seed)

            for metric in args.metrics:
                sns.lineplot(
                    data=seed_df.to_pandas(),
                    x="frame",
                    y=metric,
                    hue=hue_col,
                    hue_order=hue_order,
                    palette=palette,
                    ax=ax,
                    legend="full" if i == 0 else False,
                )

            # Formatting
            if len(args.metrics) == 1:
                formatted_metric = format_metric_name(args.metrics[0])
                ylabel_map = get_ylabel_mapping(env)
                ylabel = ylabel_map.get(formatted_metric, formatted_metric)
                if args.normalize:
                    ylabel = f"Normalized {ylabel}"
            else:
                ylabel = "Value"

            ax.set_ylabel(ylabel)
            ax.set_xlabel(r"Time steps $(\times 10^6)$")
            ax.set_title(f"Seed {seed}")
            ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
            ax.xaxis.set_major_formatter(
                ticker.FuncFormatter(lambda x, _: f"{x / 1000000:g}")
            )
            ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
            despine(ax)

            if args.vertical_lines:
                for x in args.vertical_lines:
                    ax.axvline(x=x, color="grey", linestyle=":", alpha=0.5)

            for hline_val, hline_label in horizontal_lines:
                ax.axhline(y=hline_val, color="grey", linestyle="--", alpha=0.7,
                           label=hline_label)
                if hline_label:
                    ax.annotate(
                        hline_label,
                        xy=(1, hline_val),
                        xycoords=("axes fraction", "data"),
                        ha="right", va="bottom", fontsize=8, color="grey",
                    )

            if args.ylim:
                ax.set_ylim(args.ylim)

        # Hide extra subplots if grid is not fully filled
        for j in range(num_seeds, len(axes)):
            axes[j].axis("off")

        # Handle legend
        if not args.legend:
            annotate_plot(axes[0], label_map=LABEL_MAP, disable_fov=args.disable_fov)
        else:
            handles, labels = axes[0].get_legend_handles_labels()
            mapped_labels = [get_mapped_label(label, LABEL_MAP, disable_fov=args.disable_fov) for label in labels]
            axes[0].legend(handles, mapped_labels, title=None, frameon=False)
    else:
        # Original plotting logic for metrics
        for i, metric in enumerate(args.metrics):
            if args.plot_avg:
                ax = axes[i][0]
                ax_auc = axes[i][1]
            else:
                ax = axes[i] if num_metrics > 1 else axes[0]

            # Configure lineplot based on whether to show all seeds or confidence intervals
            lineplot_kwargs = {
                "data": df.to_pandas(),
                "x": "frame",
                "y": metric,
                "hue": hue_col,
                "hue_order": hue_order,
                "palette": palette,
                "ax": ax,
                "legend": False,
            }

            if args.plot_all_seeds:
                # Plot each seed as a separate line
                lineplot_kwargs["units"] = "seed"
                lineplot_kwargs["estimator"] = None
                lineplot_kwargs["alpha"] = 0.05  # Make individual lines semi-transparent
            else:
                # Plot mean with confidence intervals
                lineplot_kwargs["errorbar"] = ("ci", 95)

            sns.lineplot(**lineplot_kwargs)

            # Formatting
            formatted_metric = format_metric_name(metric)
            ylabel_map = get_ylabel_mapping(env)
            ylabel = ylabel_map.get(formatted_metric, formatted_metric)

            if args.normalize:
                ylabel = f"Normalized {ylabel}"
            ax.set_ylabel(ylabel)
            if i == num_metrics - 1:  # Only set x-label on the last subplot
                ax.set_xlabel(r"Time steps $(\times 10^6)$")
            else:
                ax.set_xlabel("")  # Remove x-label for non-last subplots
            ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
            ax.xaxis.set_major_formatter(
                ticker.FuncFormatter(lambda x, _: f"{x / 1000000:g}")
            )
            ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4))
            despine(ax)

            if args.vertical_lines:
                for x in args.vertical_lines:
                    ax.axvline(x=x, color="grey", linestyle=":", alpha=0.5)

            for hline_val, hline_label in horizontal_lines:
                ax.axhline(y=hline_val, color="grey", linestyle="--", alpha=0.7,
                           label=hline_label)
                if hline_label:
                    ax.annotate(
                        hline_label,
                        xy=(1, hline_val),
                        xycoords=("axes fraction", "data"),
                        ha="right", va="bottom", fontsize=8, color="grey",
                    )

            if args.ylim:
                ax.set_ylim(args.ylim)

            if args.plot_avg:
                # Calculate the average over frames for each seed
                avg_df = df.group_by([hue_col, "seed"]).agg(pl.col(metric).mean().alias("avg"))
                sns.barplot(
                    data=avg_df.to_pandas(),
                    x=hue_col,
                    y="avg",
                    hue=hue_col,
                    palette=palette,
                    order=hue_order,
                    ax=ax_auc,
                    capsize=.1,
                    errorbar=("ci", 95),
                    n_boot=1000,
                    legend=False
                )
                ax_auc.set_ylabel("")
                ax_auc.set_xlabel("")
                ax_auc.set_xticklabels([])
                ax_auc.set_xticks([])
                despine(ax_auc)

        # Handle legend
        ax_for_legend = axes[0][0] if args.plot_avg else axes[0]
        if not args.legend:
            annotate_plot(ax_for_legend, label_map=LABEL_MAP, disable_fov=args.disable_fov)
        else:
            handles, labels = ax_for_legend.get_legend_handles_labels()
            mapped_labels = [get_mapped_label(label, LABEL_MAP, disable_fov=args.disable_fov) for label in labels]
            legend_obj = ax_for_legend.legend(handles, mapped_labels, title=None, frameon=True, loc='best')
            legend_obj.get_frame().set_alpha(0.9)
            legend_obj.get_frame().set_facecolor('white')

    # Save plot
    if args.grid:
        # Flatten grid_cells to get all algorithm names for the filename
        all_algs = [alg for cell in grid_cells for alg in cell] if grid_cells else []
        algs_str = "_".join(all_algs) if all_algs else "grid"
        plot_name = args.plot_name or f"{env}_{algs_str}_grid"
    elif args.subplot_by_seed:
        plot_name = args.plot_name or f"{env}_by_seed"
    elif len(args.metrics) == 1:
        plot_name = args.plot_name or f"{env}_{args.metrics[0]}_curve"
    else:
        metrics_str = "_".join(args.metrics)
        plot_name = args.plot_name or f"{env}_{metrics_str}_curves"
    save_plot(fig, args.experiment_path, plot_name, args.save_type)


if __name__ == "__main__":
    main()
