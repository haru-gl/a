#include "classes.h"
#include "enclasses.h"
#include "akaze.h"
#include "kaze.h"
#include "surf.h"
#include "sift.h"
#include "brisk.h"
#include "orb.h"

void featurepointsdetection(const featureType& mt, clipedmap_data& clpd, target_data& tgt, analysis_results& rst)
{
	switch (mt) {
	case featureType::fAKAZE:
	{
		akaze fdt_akaze;//Default Setting
		fdt_akaze.featuredetection(clpd, tgt);
		rst.map_ptsNum = (int)clpd.oPts.size();
		rst.target_ptsNum = (int)tgt.oPts.size();
	}
	break;
	case featureType::fKAZE:
	{
		kaze fdt_kaze;//Default Setting
		fdt_kaze.featuredetection(clpd, tgt);
		rst.map_ptsNum = (int)clpd.oPts.size();
		rst.target_ptsNum = (int)tgt.oPts.size();
	}
	break;
	case featureType::fSURF:
	{
		surf fdt_surf;//Default Setting
		fdt_surf.featuredetection(clpd, tgt);
		rst.map_ptsNum = (int)clpd.oPts.size();
		rst.target_ptsNum = (int)tgt.oPts.size();
	}
	break;
	case featureType::fSIFT:
	{
		sift fdt_sift;//Default Setting
		fdt_sift.featuredetection(clpd, tgt);
		rst.map_ptsNum = (int)clpd.oPts.size();
		rst.target_ptsNum = (int)tgt.oPts.size();
	}
	break;
	case featureType::fBRISK:
	{
		brisk fdt_brisk;//Default Setting
		fdt_brisk.featuredetection(clpd, tgt);
		rst.map_ptsNum = (int)clpd.oPts.size();
		rst.target_ptsNum = (int)tgt.oPts.size();
	}
	break;
	case featureType::fORB:
	{
		orb fdt_orb;//Default Setting
		fdt_orb.featuredetection(clpd, tgt);
		rst.map_ptsNum = (int)clpd.oPts.size();
		rst.target_ptsNum = (int)tgt.oPts.size();
	}
	break;
	}
}
