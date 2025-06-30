#pragma once
#include "classes.h"
#include "akaze.h"
#include "kaze.h"
#include "surf.h"
#include "sift.h"
#include "orb.h"
#include "brisk.h"
#include "knn.h"
#include "gaussnewton.h"
#include "sac.h"
#include "prosac.h"
#include "nransac.h"
#include "rfilters.h"
#include "vbayes_tt.h"
#include "vbayes_cp.h"

class allanalysis {
public:
	methods mms;
//	featureType ft;		//特徴点の検出法：選択肢：fAKAZE, fKAZE, fSURF, fSIFT, fBRISK, fORB
//	knnType kt;			//kNN+Matchratioの形式：kNORMAL, kDELETETHESAME, kNN200DTS, kNN1000DTS
//	matchingType mt;	//変換行列の形式：選択肢：mSIMILARITY, mAFFINE, mPROJECTIVE, mPROJECTIVE3
//	matrixcalType ct;	//変換行列の推定方法：選択肢：cSVD, cGAUSSNEWTON
//	sacType rt;			//ransac類の形式：選択肢：rRANSAC, rRANSACWITHNORM, rREINFORCEMENT, rREINFORCEMENTWITHNORM, rKERNELDE, rKERNELDEWITHNORM
//	sacAppType st;		//ransac類の応用方法：選択肢：sNORMAL, sRANSAC3OR, sSTDDEV1, sSTDDEV2
//	kernelFType nt;		//カーネル密度推定時に利用するカーネル形状

	// featureType { fAKAZE, fKAZE, fSURF, fSIFT, fBRISK, fORB };<=ms.ft;
	akaze fdt_akaze;
	kaze fdt_kaze;
	surf fdt_surf;
	sift fdt_sift;
	orb fdt_orb;
	brisk fdt_brisk;

	// knnType { kNORMAL, kDELETETHESAME, kNNFIXNDTS };<=ms.kt;
	knn knnt;

	prosac psac_org;
	kerneldensityestimation kde_org;
	gaussnewtonmethod gnm_org;
	reinforcementlearning rfl_org;
	vbayes_tt vb_org;
	vbayes_cp vbc_org;

	// sacType { sNORMAL, sRANSAC3OR, sSTDDEV1, sSTDDEV2 };<=ms.st;
	sac sct_sac;				//class sac<=ms.rt,ms.mt,ms.ct;
	nransac sct_nransac;		//class nransac : public ransac<=ms.rt,ms.mt,ms.ct;
	filter_sd sct_sd;			//class filter_sd : public ransac<=ms.rt,ms.mt,ms.ct;
	filter_sdsd sct_sdsd;		//class filter_sdsd : public ransac<=ms.rt,ms.mt,ms.ct;
	filter_hi sct_hi;			//class filter_hi : public ransac<=ms.rt,ms.mt,ms.ct;
	filter_hihi sct_hihi;		//class filter_hihi : public ransac<=ms.rt,ms.mt,ms.ct;
	filter_sdhi sct_sdhi;		//class filter_sdhi : public ransac<=ms.rt,ms.mt,ms.ct;
	filter_hisd sct_hisd;		//class filter_hisd : public ransac<=ms.rt,ms.mt,ms.ct;

	void featurematching_main(map_data& map, target_data& tgt, analysis_results& rst);
	void featurepoints_detection_main(clipedmap_data& clpd, target_data& tgt, analysis_results& rst);
	bool msac_main(clipedmap_data& cm, target_data& td, cv::Mat& tform);
	void positionestimation_main(clipedmap_data& cm, target_data& td, analysis_results& rs);
	void positionestimation_vb(methods mms, clipedmap_data& cm, target_data& td, analysis_results& rs);
};
