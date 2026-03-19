import numpy as np

def main():
    '''This is a template to create a minimal Astroth Python interface. The purpose of
    it is to do base essential interface between Astaroth and Python, to help
    create initial building blocks for simulations and multi-GPU implementations. 
    '''
    # ------------------------------------------------------------------
    # 1. Load config
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # 2. Create device
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # 3. Compute mesh dimensions
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # 4. Initialize: Set a initial field, apply BCs
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # 5. Kernel loop
    # ------------------------------------------------------------------

    steps = 10 #Dummy value
    for step in range(steps):
        # (a) Launch the kernel

        # (b) Swap buffers -- output becomes input

        # (c) Apply periodic boundary conditions -- fill ghost zones

        # (d) Diagnostics every N steps
        field_max = 0.0 #Dummy value
        field_min = 1.0 #Dummy value
        if np.isnan(field_max) or np.isnan(field_min):
            print("NaN detected, stopping.")
            break

    # ------------------------------------------------------------------
    # 7. Cleanup
    # ------------------------------------------------------------------

    return 0


if __name__ == "__main__":
    sys.exit(main())
