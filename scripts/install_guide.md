## You may use a conda environment:
```bash
conda create -n rangedik python=3.8
conda activate rangedik
```

## First install Python dependencies:

```bash
pip install numpy pyaml urdf_parser_py
conda install python-orocos-kdl -c conda-forge
```

## Before proceeding, Make sure both `relaxed_ik_ros1` and relaxed_ik_core are on the `ranged-ik` branch!
Otherwise, run `git checkout ranged-ik` in both the base and `relaxed_ik_core` directory.

## Compile the rust library and install its wrapper via pip(relaxed_ik_core):
First, install rust if you don't have it yet;
then use `cargo` to build it.
```bash
cd relaxed_ik_core
cargo build
pip install -e .
cd ..
```

## Finally, you can run the demo:
```bash
python scripts/relaxed_ik_rust_demo.py
```

To change to other robots, you need to put the URDF in `relaxed_ik_core/configs/urdfs`, and edit the `relaxed_ik_core/configs/settings.yaml` file.

In this example, there are two chains (right and left arm, which shares one linear joint). You should change the `urdf`, `base_links`, `ee_links` to match the URDF.
For the starting config, once you run the demo, it should print something like this:
```
Robot Articulated Joint names: ['linear_joint', 'right_shoulder_pan_joint', 'right_shoulder_lift_joint', 'right_arm_half_joint', 'right_elbow_joint', 'right_wrist_spherical_1_joint', 'right_wrist_spherical_2_joint', 'right_wrist_3_joint', 'left_shoulder_pan_joint', 'left_shoulder_lift_joint', 'left_arm_half_joint', 'left_elbow_joint', 'left_wrist_spherical_1_joint', 'left_wrist_spherical_2_joint', 'left_wrist_3_joint']
```
and you can adjust the initial values of the joints accordingly.
You also need to edit the `chains_def` field. This marks the indices of every articulated joint of each chain. *This helps to handle cases where arms/fingers share some joints.*

```yaml
urdf: simplified_movo.urdf
link_radius: 0.05 
base_links:
  - base_link
  - base_link
ee_links:
  - right_ee_link
  - left_ee_link

starting_config:   [0.35, -1.5, -0.2, -0.0, -2.0, 2.0, -1.2354, -1.1, 1.5, 0.2, 0.0, 2.0, -2.0, 1.2354, 1.1]

chains_def: [[0,1,2,3,4,5,6,7],[0,8,9,10,11,12,13,14]]
```