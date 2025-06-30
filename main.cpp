// featurematching_org.cpp : このファイルには 'main' 関数が含まれています。プログラム実行の開始と終了がそこで行われます。
//
//#define _CRT_SECURE_NO_WARNINGS
//Visual Studioでscanf、fopen、sprintfを利用する場合に必要となる宣言
//Mac,Linuxでは不要
//この宣言はヘッダーファイルのインクルードの前に書く
#include <stdlib.h>

#include <iostream>
#include <string>
#include <sstream>
#include <fstream>
#include <vector>
#include <opencv2/opencv.hpp>
#include "classes.h"

//Path of Files of the Estimated Data
#ifdef __APPLE__
#include <unistd.h>
const char* Rootfolder= "/Users/kamatahiroyuki/Pictures/JAXA_database/";
const char* Fprintfolder="/Users/kamatahiroyuki/Documents/Cpp/";
//const char* Rootfolder= "/Users/hiroyukikamata/Pictures/JAXA_database/";
//const char* Fprintfolder="/Users/hiroyukikamata/Documents/Cpp/";
#else
const char* Rootfolder = "C:/JAXA_database/";
const char* Fprintfolder="./";
#endif

const char* Sizefolder = "512/";
const char* Imgfolder = "jpg8k/";//else、"jpg8k/","bmp/"
const char* Mapimg = "mapimg/CST1/";
const char* Wdtfile = "TCO_CST1_TM_SIM_a7351_i3438_h36000_lanczos3_v2.txt";
int ImgNum, DatasetNum, * DatasetIdx;

std::vector<std::vector<double>> csvread(std::string fnm);
bool get_mapImage(map_data& map);
bool get_targetImages(int dsn, target_data tgtd[], int imgNum, simulation_data simd[]);
void evaluation(analysis_results& rst, simulation_data& simd);
void spresults(std::ofstream& ofs, const simulation_data& sd, const analysis_results& md);

void featurematching_main(map_data& map, target_data& tgt, analysis_results& rst);

int main(void)
{
	map_data mapd;
	get_mapImage(mapd);

//Get the Current Directory
//    char *pwd=getcwd(NULL,0);
//    printf("Pwd=%s¥n",pwd);
    
	double iratio = 1000.0 / (double)mapd.oImage.rows;
	cv::Mat cpmap; cv::resize(mapd.oImage, cpmap, cv::Size(), iratio, iratio, cv::INTER_NEAREST);
	cv::imshow("MapImage", cpmap);

    cv::waitKey(1);
	cv::Mat cpmapo = cpmap.clone();
	std::cout << mapd.averageDistance << " " << mapd.averageHeight << " " << mapd.lsz << std::endl;
	std::cout << ImgNum << " " << DatasetNum << std::endl;
	for (int i = 0; i < DatasetNum; i++)
		std::cout << DatasetIdx[i] << std::endl;

	target_data *tgtd;
	tgtd = new target_data[ImgNum];

	simulation_data *simd;
	simd = new simulation_data[ImgNum];
	analysis_results *rst;
	rst = new analysis_results[ImgNum];

    char fnm[512];
    snprintf(fnm,std::size(fnm),"%sResults.txt",Fprintfolder);
	std::ofstream rfs; rfs.open(fnm, std::ios::app);
    if(!rfs){
        std::cout << "File[" << fnm << "] cannot be created" <<std::endl;
        exit(0);
    }
    rfs.close();

	for (int dsn = 0; dsn < DatasetNum; dsn++) {
		std::cout << "Dataset Num=" << DatasetIdx[dsn] << std::endl;
		get_targetImages(dsn, tgtd, ImgNum, simd);
		rfs.open(fnm, std::ios::app); rfs << "Dataset Num = " << DatasetIdx[dsn] << std::endl;
        rfs.close();
		statistics sts; sts.clear();
		char fnmp[512];
        snprintf(fnmp, std::size(fnmp), "%sfm%0d.csv", Fprintfolder, DatasetIdx[dsn]);
		std::ofstream ofnmp(fnmp, std::ios::app);
        if(!ofnmp){
            std::cout << "File[" << fnmp << "] cannot be created" << std::endl;
            exit(0);
        }
		
		for (int k = 0; k < ImgNum; k++) {
			std::cout << "Image No.=" << k << std::endl;
			cv::imshow("target", tgtd[k].oImage);
			//int iw1 =
            cv::waitKey(1);
			std::cout << tgtd[k] << std::endl;

			featurematching_main(mapd, tgtd[k], rst[k]);



			std::cout << rst[k] << std::endl;
			evaluation(rst[k], simd[k]);
			std::cout << simd[k] << std::endl;
			sts.add_stat(rst[k], simd[k]);
			ofnmp << k << ","; spresults(ofnmp, simd[k], rst[k]);
			cpmap = cpmapo.clone();
			cv::line(cpmap, iratio * rst[k].c00, iratio * rst[k].c01, cv::Scalar(250));
			cv::line(cpmap, iratio * rst[k].c01, iratio * rst[k].c11, cv::Scalar(250));
			cv::line(cpmap, iratio * rst[k].c11, iratio * rst[k].c10, cv::Scalar(250));
			cv::line(cpmap, iratio * rst[k].c10, iratio * rst[k].c00, cv::Scalar(250));
			cv::imshow("MapImage", cpmap);
			//int iw2 =
            cv::waitKey(1);
		}
		
		ofnmp.close();
		sts.cal_stat();
		std::cout << sts << std::endl;
		rfs.open(fnm, std::ios::app); rfs << sts << std::endl; rfs.close();

		for (int k = 0; k < ImgNum; k++)
			tgtd[k].dd_clear();
	}
    delete[] rst;
    delete[] simd;
    delete[] tgtd;
	return 0;
}

std::vector<std::vector<double>> csvread(std::string fnm)
{
	std::ifstream ifs;
	std::string line;
	int rows, cols;

	ifs.open(fnm);
	if (!ifs) {
		std::cout << "Could not read true value file\n";
		exit(1);
	}
	else {
		std::cout << fnm.c_str() << " is opened\n";
	}
	rows = cols = 0;

	while (std::getline(ifs, line, '\n')) {
		if (cols == 0) {
			char delimiter = ',';
			std::istringstream iss(line);
			std::string field;
			while (std::getline(iss, field, delimiter)) {
				cols++;
			}
		}
		rows++;
	}
	ifs.clear();
	ifs.seekg(0, std::ios_base::beg);
	std::vector<std::vector<double>> params(rows, std::vector<double>(cols));
	int i, j;
	i = 0; j = 0;
	while (std::getline(ifs, line, '\n')) {
		char delimiter = ',';
		std::istringstream iss(line);
		std::string field;
		j = 0;
		while (std::getline(iss, field, delimiter)) {
			std::istringstream(field) >> params[i][j];
			j++;
		}
		i++;
	}
	return params;
}


void evaluation(analysis_results& rst, simulation_data& simd)
{
	simd.errDx = rst.estimatedCenter2dx - simd.x_true;
	simd.errDy = rst.estimatedCenter2dy - simd.y_true;
	simd.errD = sqrt(simd.errDx * simd.errDx + simd.errDy * simd.errDy);
	simd.errHeightP = (rst.estimatedHeight - simd.h_true) / simd.h_true * 100.0;
	if ((simd.errD > simd.distTh) && (rst.status == 0)) rst.status = 1;
}

void spresults(std::ofstream& ofs, const simulation_data& sd, const analysis_results& md)
{
	ofs << sd.x_true << "," << sd.y_true << "," << sd.h_true << "," << md.estimatedCenter2dx << "," << md.estimatedCenter2dy << "," << md.estimatedHeight << ",";
	ofs << sd.errDx << "," << sd.errDy << "," << sd.errD << "," << sd.errHeightP << "," << md.elapsedTime << ",";
	ofs << md.map_ptsNum << "," << md.target_ptsNum << "," << md.goodPairsNum << "," << md.status << std::endl;
}

bool get_mapImage(map_data& map)
{
	char wimgfile[1024];
	std::string mapdatafile = Rootfolder;
	mapdatafile.append(Mapimg);mapdatafile.append(Wdtfile);
	std::ifstream idfs;
	idfs.open(mapdatafile.c_str());
	if (!idfs) {
		std::cout << "Could not open map image data file in function(get_mapImages)" << std::endl;
		return 0;
	}
	idfs >> map.averageDistance;
	idfs >> map.averageHeight;
	idfs >> wimgfile;

	std::string mapimgfile = Rootfolder;
	mapimgfile.append(Mapimg); mapimgfile.append(wimgfile);
	map.oImage = cv::imread(mapimgfile.c_str(), cv::IMREAD_GRAYSCALE);

    map.lsz = INIT_LSZ;//INIT_SEARCHAREASIZE;

	idfs >> ImgNum;
	idfs >> DatasetNum;
	DatasetIdx = new int[DatasetNum];
	for (int i = 0; i < DatasetNum; i++)
		idfs >> DatasetIdx[i];
	idfs.close();

	return true;
}

bool get_targetImages(int dsn, target_data tgtd[], int imgNum, simulation_data simd[])
{
	std::string fname = Rootfolder;
	fname.append(Sizefolder); fname.append("commondata/");
    fname.append(std::to_string(DatasetIdx[dsn]));
    fname.append("/true_point.csv");

	std::vector<std::vector<double>> params;
	params = csvread(fname);

	std::string imfname = Rootfolder;
	imfname.append(Sizefolder); imfname.append(Imgfolder); imfname.append(std::to_string(DatasetIdx[dsn])); imfname.append("/imgfile.txt");

	std::ifstream ifs;
	ifs.open(imfname.c_str());
	if (!ifs) {
		std::cout << "Could not open list file in function(get_targetImages)" << std::endl;
		return false;
	}
	std::cout << imfname << " opened" << std::endl;
	for (int k = 0; k < imgNum; k++) {	//imgNum-Loop
		char imgfileName[256];
		ifs >> imgfileName;
		std::string fnamea = Rootfolder;
		fnamea.append(Sizefolder);
        fnamea.append(Imgfolder);
        fnamea.append(std::to_string(DatasetIdx[dsn]));
        fnamea.append("/");
        fnamea.append(imgfileName);
		tgtd[k].oImage = cv::imread(fnamea.c_str(), cv::IMREAD_GRAYSCALE);
		if (tgtd[k].oImage.empty()) {
			std::cout << "The image file did not load in function(get_targetImages)" << std::endl;
			return false;
		}
		tgtd[k].x_gnc = params[k][6];
		tgtd[k].y_gnc = params[k][7];
		tgtd[k].sz = tgtd[k].oImage.rows;
		tgtd[k].szcenter = tgtd[k].sz / 2;
		simd[k].x_true = params[k][0];
		simd[k].y_true = params[k][1];
		simd[k].h_true = params[k][2];
	}
	ifs.close();
	params.clear();
	params.shrink_to_fit();
	return true;
}

