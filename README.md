# Motion-Capture-Aided Multirotor Drag Parameter Estimation

Simple batch (least-squares, using OSQP) optimizer to estimate a multirotor's rotor blade drag coefficient ($\mu$) given motion capture data for attitude and linear/angular velocities over a flight trajectory without excessive acceleration.

## Installation

```bash
git clone --recurse-submodules https://github.com/goromal/multirotor-drag-sysid.git
cd multirotor-drag-sysid.git
mkdir build && cd build
cmake ..
make
```

## Usage


