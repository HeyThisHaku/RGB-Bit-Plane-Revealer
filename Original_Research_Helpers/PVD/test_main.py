"""
@file   test_main.py
@brief  PVD library CLI usage example
@author Tony Josi (https://tony-josi.github.io)
@copyright Copyright (C) 2022 Tony Josi
"""

import sys
from pvd_lib import pvd_lib

if __name__ == "__main__":
    pvd_obj = pvd_lib()
    mode = sys.argv[1].lower()
    if mode == "e":
        pvd_obj.pvd_embed(sys.argv[2], sys.argv[3], sys.argv[4])
    elif mode == "d":
        pvd_obj.pvd_extract(sys.argv[2], sys.argv[3], sys.argv[4])
