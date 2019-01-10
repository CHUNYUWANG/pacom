Multiview 2D + 3D Human pose estimation using pytorch

# Quick start
## Installation
1. Clone this repo, and we'll call the directory that you cloned pose.pytorch as ${POSE_ROOT}
2. Install dependencies.
3. Download pytorch imagenet pretrained models. Please download them under ${POSE_ROOT}/models, and make them look like this:

   ```
   ${POSE_ROOT}/models
   └── pytorch
       └── imagenet
           ├── resnet101-5d3b4d8f.pth
           ├── resnet152-b121ed2d.pth
           ├── resnet18-5c106cde.pth
           ├── resnet34-333f7ec4.pth
           └── resnet50-19c8e357.pth
   ```

4. Init output(training model output directory) and log(tensorboard log directory) directory.

   ```
   mkdir ouput 
   mkdir log
   ```

   and your directory tree should like this

   ```
   ${POSE_ROOT}
   ├── data
   ├── experiments
   ├── lib
   ├── log
   ├── models
   ├── output
   ├── pose_estimation
   ├── README.md
   ├── requirements.txt
   ```

## Data preparation
**For MPII data**, please download from [MPII Human Pose Dataset](http://human-pose.mpi-inf.mpg.de/), the original annotation files are matlab's format. We have converted to json format, you also need download them from [OneDrive](https://1drv.ms/f/s!AhIXJn_J-blWzyhpADBbpJRusuT0).
Extract them under {POSE_ROOT}/data, and make them look like this:

```
${POSE_ROOT}
|-- data
|-- |-- MPII
    |-- |-- annot
        |   |-- gt_valid.mat
        |   |-- test.json
        |   |-- train.json
        |   |-- trainval.json
        |   |-- valid.json
        |-- images
            |-- 000001163.jpg
            |-- 000003072.jpg
```

If you zip the image files into a single zip file, you should organize the data like this:

```
${POSE_ROOT}
|-- data
`-- |-- MPII
    `-- |-- annot
        |   |-- gt_valid.mat
        |   |-- test.json
        |   |-- train.json
        |   |-- trainval.json
        |   `-- valid.json
        `-- images.zip
            |-- 000001163.jpg
            |-- 000003072.jpg
```



**For Human36M data**, please download them from [OneDrive](https://1drv.ms/f/s!AhIXJn_J-blWzyhpADBbpJRusuT0)
Extract them under {POSE_ROOT}/data, and make them look like this:

```
${POSE_ROOT}
|-- data
|-- |-- h36m
    |-- |-- annot
        |   |-- h36m_train.pkl
        |   |-- h36m_validation.pkl
        |-- images
            |-- s_01_act_02_subact_01_ca_01 
            |-- s_01_act_02_subact_01_ca_02
```

If you zip the image files into a single zip file, you should organize the data like this:
```
${POSE_ROOT}
|-- data
`-- |-- h36m
    `-- |-- annot
        |   |-- gt_valid.mat
        |   |-- test.json
        |   |-- train.json
        |   |-- trainval.json
        |   `-- valid.json
        `-- images.zip
            |-- s_01_act_02_subact_01_ca_01
            |-- s_01_act_02_subact_01_ca_02
```




## Training
**Multiview Training on Mixed Dataset (MPII+H36M) and test on H36M**

```
python run/pose2d/train.py --cfg experiments/mixed/resnet50/256_nofusion.yaml
python run/pose2d/train.py --cfg experiments/mixed/resnet50/256_fusion.yaml
```

you should get the following results respectively

root-| rhip- | rkne- | rank- | lhip- | lkne- | lank- | belly-| neck- | nose- | head- | lsho- | lelb- | lwri- | rsho- | relb- | rwri- |

1.000| 0.978 | 0.903 | 0.853 | 0.980 | 0.889 | 0.799 | 0.969 | 0.982 | 0.984 | 0.990 | 0.924 | 0.869 | 0.788 | 0.928 | 0.861 | 0.784 |

root-| rhip- | rkne- | rank- | lhip- | lkne- | lank- | belly-| neck- | nose- | head- | lsho- | lelb- | lwri- | rsho- | relb- | rwri- |

1.000 | 0.993 | 0.947 | 0.891 | 0.994 | 0.936 | 0.846 | 0.980 | 0.990 | 0.990 | 0.990 | 0.954 | 0.933 | 0.903 | 0.944 | 0.913 | 0.895 |


