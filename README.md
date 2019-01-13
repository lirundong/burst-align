# Quick Alignment for Burst Photography

This repository contains a multi-threading implementation of L2/L1 mixed alignment
algorithm described in [Hasinoff et. al, HDR+, SIGGRAPH'16](http://www.hdrplusdata.org/).

> Author: Rundong Li \<lird@shanghaitech.edu.cn\><br/>

## Get Started

Note this implementation gets **NO WARRANTY** of running on Windows platform.
MacOS with `LLVM >= 6.0` or Ubuntu with `GCC >= 4.9` is required.

0. If you're using MacOS:

   Install `LLVM` with:
   ```bash
   brew install llvm
   ```
1. Install required Python extensions by `pip` or `anaconda`, note that
   `Python >= 3.6` is required.
   ```bash
   conda install -r requirements.txt
   ```
   
   Then you may build C- extensions by
   ```bash
   ./build.sh
   ```

2. Create a folder `data` and copy your burst into it, execute
   ```bash
   python faster_align_cpu.py
   ```
   then you should find the results plotted in `data/demo.png`.
   
   *Note*: the `demo.py` is just for illustrating the algorithms we used, itself
   can work too, but **very slow**.

3. To run tests for C- extensions, execute
   ```bash
   pytest test
   ```
   under project folder.

## License

This work is licensed under GNU GPLv3, see [LICENSE](LICENSE) for details.
