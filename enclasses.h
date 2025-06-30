#pragma once
#include <fstream>

//Type of Feature Detection Method
enum class featureType { fAKAZE, fKAZE, fSURF, fSIFT, fBRISK, fORB, fDAMMY };
featureType begin(featureType);
featureType end(featureType);
featureType operator*(featureType ft);
featureType operator++(featureType& ft);
std::ostream& operator<<(std::ostream& os, featureType ft);

//Type of kNN Method (LSH‚à‚±‚±‚ÉŠÜ‚ß‚é)
enum class knnType {
    kNORMAL,        // Šù‘¶‚Ìk-NN
    kDTSFP,         // Šù‘¶‚Ìk-NN
    kNNFIXN,        // Šù‘¶‚Ìk-NN
    kNNFIXNDTSFP,   // Šù‘¶‚Ìk-NN
    kCUSTOM_LSH,    // š Ž©ìLSH‚ð’Ç‰Á
    kFLANN_LSH,     // š FLANN LSH‚ð’Ç‰Á
    kDAMMY          // Šù‘¶‚Ìƒ_ƒ~[
};;
knnType begin(knnType);
knnType end(knnType);
knnType operator*(knnType ft);
knnType operator++(knnType& ft);
std::ostream& operator<<(std::ostream& os, knnType ft);

//Type of Transformation Matrix
enum class matchingType { mSIMILARITY, mAFFINE, mPROJECTIVE, mPROJECTIVE3, mPROJECTIVE_EV, mDAMMY };
matchingType begin(matchingType);
matchingType end(matchingType);
matchingType operator*(matchingType ft);
matchingType operator++(matchingType& ft);
std::ostream& operator<<(std::ostream& os, matchingType ft);

//Type of Estimation Method for Transformation Matrix
enum class matrixcalType { cSVD, cGAUSSNEWTON, cSVD_EIGEN, cGAUSSNEWTON_EIGEN, cTAUBIN, cVBAYES, cDAMMY };
matrixcalType begin(matrixcalType);
matrixcalType end(matrixcalType);
matrixcalType operator*(matrixcalType ft);
matrixcalType operator++(matrixcalType& ft);
std::ostream& operator<<(std::ostream& os, matrixcalType ft);

//Type of Application using RANSAC
enum class sacAppType { sNORMAL,sRANSACNOR, sFILTERS1, sFILTERH1, sFILTERS2, sFILTERH2, sFILTERSH, sFILTERHS, sVBAYESONLY, sVBAYESWITHKNN, sDAMMY };
sacAppType begin(sacAppType);
sacAppType end(sacAppType);
sacAppType operator*(sacAppType ft);
sacAppType operator++(sacAppType& ft);
std::ostream& operator<<(std::ostream& os, sacAppType ft);

//Type of RANSAC
enum class sacType { rRANSAC, rRANSACWITHNORM, rREINFORCEMENT, rREINFORCEMENTWITHNORM, rKERNELDE, rKERNELDEWITHNORM, rPROSAC, rPROSACWITHNORM, rVBAYESONLY, rVBAYESWITHKNN, rDAMMY };
sacType begin(sacType);
sacType end(sacType);
sacType operator*(sacType ft);
sacType operator++(sacType& ft);
std::ostream& operator<<(std::ostream& os, sacType ft);

//Action Mode of RANSAC
enum class ransacMode { dNORMAL, dSTDDEV, dHAMPLEI, dDAMMY };
// sacType=sNORMAL:    ransacMode=dNORMAL
// sacType=sRANSACNOR: ransacMode=dNORMAL
// sacType=sFILTERS1:  ransacMode=dSTDDEV
// sacType=sFILTERH1:  ransacMode=dHAMPLEI
// sacType=sFILTERS2:  ransacMode=dSTDDEV
// sacType=sFILTERS2:  ransacMode=dHAMPLEI
// sacType=sFILTERSH:  ransacMode=dSTDDEV and dHAMPLEI
// sacType=sFILTERHS:  ransacMode=dHAMPLEI and dSTDDEV
ransacMode begin(ransacMode);
ransacMode end(ransacMode);
ransacMode operator*(ransacMode ft);
ransacMode operator++(ransacMode& ft);
std::ostream& operator<<(std::ostream& os, ransacMode ft);

//Type of Kernel when Kernel Density Estimation is used
enum class kernelFType { nGAUSS, nRECTANGLE, nTRIANGLE, nEPANECHNIKOV, nDAMMY };
kernelFType begin(kernelFType);
kernelFType end(kernelFType);
kernelFType operator*(kernelFType ft);
kernelFType operator++(kernelFType& ft);
std::ostream& operator<<(std::ostream& os, kernelFType ft);

//Variation Bayes Analysis
// This setting is valid only when sacAppType is set to sVBAYES.
//enum class vbType { vBAYESONLY, vWITHKNN, vDAMMY };
// vBAYESONLY: VB is applied to both correspondence point search and transformation matrix estimation
// vWITHKNN  : k-NN method is applied as a pre-processing of correspondence point search
// The shape of transformation matrix is set by matchingType
//vbType begin(vbType);
//vbType end(vbType);
//vbType operator*(vbType ft);
//vbType operator++(vbType& ft);
//std::ostream& operator<<(std::ostream& os, vbType ft);

class posestType {
public:
	matchingType mt;
	ransacMode rm;
	bool ndon;
	matrixcalType ct;
};
