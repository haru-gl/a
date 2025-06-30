#include <iostream>
#include "enclasses.h"

featureType begin(featureType)
{
    return featureType::fAKAZE;
}
featureType end(featureType)
{
    return featureType::fDAMMY;
}
featureType operator*(featureType ft)
{
    return ft;
}
featureType operator++(featureType& ft)
{
    return ft = featureType(std::underlying_type<featureType>::type(ft) + 1);
}
std::ostream& operator<<(std::ostream& os, featureType ft) 
{
    switch (ft) {
    case featureType::fAKAZE:
        return os << "fAKAZE";
    case featureType::fKAZE:
        return os << "fKAZE";
    case featureType::fSURF:
        return os << "fSURF";
    case featureType::fSIFT:
        return os << "fSIFT";
    case featureType::fBRISK:
        return os << "fBRISK";
    case featureType::fORB:
        return os << "fORB";
    default: return os;
    }
}

knnType begin(knnType) {
    return knnType::kNORMAL;
}
knnType end(knnType) {
    // ★ kDAMMY が最後になるようにする
    return knnType::kDAMMY;
}
knnType operator*(knnType kt) { // 引数名変更
    return kt;
}
knnType operator++(knnType& kt) { // 引数名変更
    return kt = knnType(std::underlying_type<knnType>::type(kt) + 1);
}
std::ostream& operator<<(std::ostream& os, knnType kt) { // 引数名変更
    switch (kt) {
    case knnType::kNORMAL:
        return os << "kNORMAL";
    case knnType::kDTSFP:
        return os << "kDTSFP"; // 元のコードで "kDTSF" だったので合わせるか確認
    case knnType::kNNFIXN:
        return os << "kNNFIXN";
    case knnType::kNNFIXNDTSFP:
        return os << "kNNFIXNDTSFP"; // 元のコードで "kNNFIXNDTSF" だったので合わせるか確認
    case knnType::kCUSTOM_LSH:      // ★ 追加
        return os << "kCUSTOM_LSH";
    case knnType::kFLANN_LSH:       // ★ 追加
        return os << "kFLANN_LSH";
    default: return os << "UnknownKnnType";
    }
}

matchingType begin(matchingType)
{
    return matchingType::mSIMILARITY;
}
matchingType end(matchingType)
{
    return matchingType::mDAMMY;
}
matchingType operator*(matchingType ft)
{
    return ft;
}
matchingType operator++(matchingType& ft)
{
    return ft = matchingType(std::underlying_type<matchingType>::type(ft) + 1);

}
std::ostream& operator<<(std::ostream& os, matchingType ft)
{
    switch (ft) {
    case matchingType::mSIMILARITY:
        return os << "mSIMILARITY";
    case matchingType::mAFFINE:
        return os << "mAFFINE";
    case matchingType::mPROJECTIVE:
        return os << "mPROJECTIVE";
    case matchingType::mPROJECTIVE3:
        return os << "mPROJECTIVE3";
    case matchingType::mPROJECTIVE_EV:
        return os << "mPROJECTIVE_EV";
    default: return os;
    }
}

matrixcalType begin(matrixcalType)
{
    return matrixcalType::cSVD;
}
matrixcalType end(matrixcalType)
{
    return matrixcalType::cDAMMY;
}
matrixcalType operator*(matrixcalType ft)
{
    return ft;
}
matrixcalType operator++(matrixcalType& ft)
{
    return ft = matrixcalType(std::underlying_type<matrixcalType>::type(ft) + 1);
}
std::ostream& operator<<(std::ostream& os, matrixcalType ft)
{
    switch (ft) {
    case matrixcalType::cSVD:
        return os << "cSVD";
    case matrixcalType::cGAUSSNEWTON:
        return os << "cGAUSSNEWTON";
    case matrixcalType::cSVD_EIGEN:
        return os << "cSVD_EIGEN";
    case matrixcalType::cGAUSSNEWTON_EIGEN:
        return os << "cGAUSSNEWTON_EIGEN";
    case matrixcalType::cTAUBIN:
        return os << "cTAUBIN";
    case matrixcalType::cVBAYES:
        return os << "cVBAYES";
    default: return os;
    }
}

sacAppType begin(sacAppType)
{
    return sacAppType::sNORMAL;
}
sacAppType end(sacAppType)
{
    return sacAppType::sDAMMY;
}
sacAppType operator*(sacAppType ft)
{
    return ft;
}
sacAppType operator++(sacAppType& ft)
{
    return ft = sacAppType(std::underlying_type<sacAppType>::type(ft) + 1);
}
std::ostream& operator<<(std::ostream& os, sacAppType ft)
{
    switch (ft) {
    case sacAppType::sNORMAL:
        return os << "sNORMAL";
    case sacAppType::sRANSACNOR:
        return os << "sRANSACNOR";
    case sacAppType::sFILTERS1:
        return os << "sFILTERS1";
    case sacAppType::sFILTERH1:
        return os << "sFILTERH1";
    case sacAppType::sFILTERS2:
        return os << "sFILTERS2";
    case sacAppType::sFILTERH2:
        return os << "sFILTERH2";
    case sacAppType::sFILTERSH:
        return os << "sFILTERSH";
    case sacAppType::sFILTERHS:
        return os << "sFILTERHS";
    case sacAppType::sVBAYESONLY:
        return os << "sVBAYESONLY";
    case sacAppType::sVBAYESWITHKNN:
        return os << "sVBAYESWITHKNN";
    default: return os;
    }
}

sacType begin(sacType)
{
    return sacType::rRANSAC;
}
sacType end(sacType)
{
    return sacType::rDAMMY;
}
sacType operator*(sacType ft)
{
    return ft;
}
sacType operator++(sacType& ft)
{
    return ft = sacType(std::underlying_type<sacType>::type(ft) + 1);

}
std::ostream& operator<<(std::ostream& os, sacType ft)
{
    switch (ft) {
    case sacType::rRANSAC:
        return os << "rRANSAC";
    case sacType::rRANSACWITHNORM:
        return os << "rRANSACWITHNORM";
    case sacType::rREINFORCEMENT:
        return os << "rREINFORCEMENT";
    case sacType::rREINFORCEMENTWITHNORM:
        return os << "rREINFORCEMENTWITHNORM";
    case sacType::rKERNELDE:
        return os << "rKERNELDE";
    case sacType::rKERNELDEWITHNORM:
        return os << "rKERNELDEWITHNORM";
    case sacType::rPROSAC:
        return os << "rPROSAC";
    case sacType::rPROSACWITHNORM:
        return os << "rPROSACWITHNORM";
    case sacType::rVBAYESONLY:
        return os << "rVBAYESONLY";
    case sacType::rVBAYESWITHKNN:
        return os << "rVBAYESWITHKNN";
    default: return os;
    }
}

ransacMode begin(ransacMode)
{
    return ransacMode::dNORMAL;
}
ransacMode end(ransacMode)
{
    return ransacMode::dDAMMY;
}
ransacMode operator*(ransacMode ft)
{
    return ft;
}
ransacMode operator++(ransacMode& ft)
{
    return ft = ransacMode(std::underlying_type<ransacMode>::type(ft) + 1);

}
std::ostream& operator<<(std::ostream& os, ransacMode ft)
{
    switch (ft) {
    case ransacMode::dNORMAL:
        return os << "dNORMAL";
    case ransacMode::dSTDDEV:
        return os << "dSTDDEV";
    case ransacMode::dHAMPLEI:
        return os << "dHAMPLEI";
    default: return os;
    }
}

kernelFType begin(kernelFType)
{
    return kernelFType::nGAUSS;
}
kernelFType end(kernelFType)
{
    return kernelFType::nDAMMY;
}
kernelFType operator*(kernelFType ft)
{
    return ft;
}
kernelFType operator++(kernelFType& ft)
{
    return ft = kernelFType(std::underlying_type<kernelFType>::type(ft) + 1);

}
std::ostream& operator<<(std::ostream& os, kernelFType ft)
{
    switch (ft) {
    case kernelFType::nGAUSS:
        return os << "nGAUSS";
    case kernelFType::nRECTANGLE:
        return os << "nRECTANGLE";
    case kernelFType::nTRIANGLE:
        return os << "nTRIANGLE";
    case kernelFType::nEPANECHNIKOV:
        return os << "nEPANECHNIKOV";
    default: return os;
    }

}
/*
vbType begin(vbType)
{
    return vbType::vBAYESONLY;
}
vbType end(vbType)
{
    return vbType::vDAMMY;
}
vbType operator*(vbType ft)
{
    return ft;
}
vbType operator++(vbType& ft)
{
    return ft = vbType(std::underlying_type<vbType>::type(ft) + 1);

}
std::ostream& operator<<(std::ostream& os, vbType ft)
{
    switch (ft) {
    case vbType::vBAYESONLY:
        return os << "vBAYESONLY";
    case vbType::vWITHKNN:
        return os << "vWITHKNN";;
    default: return os;
    }
}
*/

