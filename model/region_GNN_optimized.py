"""Compatibility entry point for the optimized annotated-region trainer."""

try:
    from .region_GNN_optimized_none_negative import main
except ImportError:
    from region_GNN_optimized_none_negative import main


if __name__ == "__main__":
    main()
