#pragma once

//Setting of the target images
#define SZ 512
#define SZCENTER 256
#define INIT_LSZ 1024

//Setting of AKAZE
#define INIT_AKAZE_DESCRIPTOR_SIZE 0
#define INIT_AKAZE_DESCRIPTOR_CHANNELS 3
#define INIT_AKAZE_THRESHOLD 0.000001
//-0.000005 0.000001
#define INIT_AKAZE_NOCTAVES 4
#define INIT_AKAZE_NOCTARVELAYERS 4

//Setting of KAZE
#define INIT_KAZE_THRESHOLD 0.001
#define INIT_KAZE_NOCTAVES 4
#define INIT_KAZE_NOCTARVELAYERS 4

//Setting of SIFT
#define INIT_SIFT_NFEATURES 0
#define INIT_SIFT_NOCTAVELAYERS 3
#define INIT_SIFT_CONTRASTTH 0.04
#define INIT_SIFT_EDGETH 10
#define INIT_SIFT_SIGMA 1.6

//Setting of SURF
#define INIT_SURF_THRESHOLD 0.00001  //75

//Setting of BRISK
#define INIT_BRISK_OCTAVES 3
#define INIT_BRISK_THRESHOLD 30

//Setting of ORB
#define INIT_ORB_MAXFEATURES 4000
#define INIT_ORB_SCALEFACTOR 1.2
#define INIT_ORB_THRESHOLD 31
#define INIT_ORB_FASTTHRESHOLD 20

//Setting of Basic RANSAC
#define INIT_MAXITERATION 5000 //5000
#define INIT_CONFIDENCE  99.99 //99.99
#define INIT_MAXDISTANCE 5.0 //5.0

//Setting of k-NN Method
#define INIT_KNNK 2
#define INIT_NNMATCHRATIO 0.90 //0.93 　　　0.84 407用0.90
#define INIT_DISTTH 3.0
#define INIT_MAXNUM 200
#define INIT_SAMEPOINT 0.5
#define INIT_KNNSORT true

//Setting of PROSAC
#define INIT_PROSAC_TN 20000000.0

//Max. Distance of feature point that is decided as the same point
//#define MAXTDIS 0.5

//Filter size of Image by GNC data
//#define INIT_SEARCHAREASIZE 1024

//Setting of vBayes estimation
//Transformation matrix estimation
#define INIT_VBMX_MAXITERATION 50
#define INIT_VBMX_DISTERROR 1.0e-5
//Corresponding point estimation
#define INIT_VBCP_MAXITERATION 25
#define INIT_VBCP_DISTERROR 1.0e-15
#define INIT_VBCP_ETA 0.0
#define INIT_VBCP_BETA 10.0e-3
#define INIT_VBCP_ALPHA 0.1

//Setting of Kernel Density Estimation
#define INIT_KDEh 5.0
#define INIT_KDEITERATION 5000
#define INIT_SETNUM 1
#define INIT_DISTERROR 0.75

//Iterative Number of RANSAC when sRANSAC3OR is used
#define INIT_NUMRANSAC 3

//Filters by Standerd Deviation when sDTDDEV1,2 are used
#define INIT_SD11 1.5
#define INIT_SD12 1.0
#define INIT_SD22 2.0

//Filters by Hample Identifier when sHAMPLE1,2 are used
#define INIT_HI11 1.0
#define INIT_HI12 1.0
#define INIT_HI22 1.0

//Setting of Gauss-Newton method
#define INIT_GN_EPS 1.0e-20
#define INIT_GN_ALPHA 1.0
#define INIT_GN_BETA 0.1
#define INIT_GN_MAXITR 10

//Setting of Reinforcement learning
#define INIT_RL_ALPHA 0.1

//Setting of SVD_Eigen
#define EG_MAXITR 1000
#define EG_CNV1 1.0e-16
#define EG_CNV2 1.0e-36
#define EG_CNV3 1.0e-50

