#include "classes.h"
#include <fstream>

std::ostream& operator<<(std::ostream& os, const map_data& md)
{
	os << "Avarage Distance=" << md.averageDistance << " Avarage Height=" << md.averageHeight << " Search Area=" << md.lsz << "[pixel] ";
	return os;
}

std::ostream& operator<<(std::ostream& os, const clipedmap_data& md)
{
	os << "Left-Upper x=" << md.lux << " y=" << md.luy;
	return os;
}

std::ostream& operator<<(std::ostream& os, const target_data& md)
{
	os << "GNC x=" << md.x_gnc << " y=" << md.y_gnc << " Size=" << md.sz << " Center=" << md.szcenter;
	return os;
}

std::ostream& operator<<(std::ostream& os, const analysis_results& md)
{
	os << "Estimated x=" << md.estimatedCenter2dx << " y=" << md.estimatedCenter2dy << " h=" << md.estimatedHeight;
	return os;
}

std::ostream& operator<<(std::ostream& os, const simulation_data& md)
{
	os << "True x=" << md.x_true << " y=" << md.y_true << " h=" << md.h_true << "\nError x = " << md.errDx << " y=" << md.errDy << " r=" << md.errD << " h=" << md.errHeightP << "[%]";
	return os;
}

std::ostream& operator<<(std::ostream& os, const statistics& md)
{
	os << "count:Success(0)=" << md.count0 << ", Error(1) =" << md.count1 << ", Matching not completed(2) =" << md.count2 << ", Insufficient inlayer(3) =" << md.count3 << "\n";
	os << "statistics(Success0)\n";
	os << "d[px]:mean=" << md.sc0dmean << ", max =" << md.sc0dmax << ", min =" << md.sc0dmin << ", 3Sigma=" << md.sc0d3sg << "\n";
	os << "dx[px]:mean=" << md.sc0dxmean << ", max =" << md.sc0dxmax << ", min =" << md.sc0dxmin << ", 3Sigma=" << md.sc0dx3sg << "\n";
	os << "dy[px]:mean=" << md.sc0dymean << ", max =" << md.sc0dymax << ", min =" << md.sc0dymin << ", 3Sigma=" << md.sc0dy3sg << "\n";
	os << "height[%]:mean=" << md.sc0hgmean << ", max =" << md.sc0hgmax << ", min =" << md.sc0hgmin << ", 3Sigma=" << md.sc0hg3sg << "\n";
	os << "sc[ms]:mean=" << md.sc0tmean << ", max =" << md.sc0tmax << ", min =" << md.sc0tmin << ", 3Sigma=" << md.sc0t3sg << "\n";
	os << "statistics(Success0+Error1)\n";
	os << "d[px]:mean=" << md.sc01dmean << ", max =" << md.sc01dmax << ", min = " << md.sc01dmin << ", 3Sigma= " << md.sc01d3sg << "\n";
	os << "dx[px]:mean=" << md.sc01dxmean << ", max =" << md.sc01dxmax << ", min =" << md.sc01dxmin << ", 3Sigma=" << md.sc01dx3sg << "\n";
	os << "dy[px]:mean=" << md.sc01dymean << ", max =" << md.sc01dymax << ", min =" << md.sc01dymin << ", 3Sigma=" << md.sc01dy3sg << "\n";
	os << "height[%]:mean=" << md.sc01hgmean << ", max =" << md.sc01hgmax << ", min =" << md.sc01hgmin << ", 3Sigma=" << md.sc01hg3sg << "\n";
	os << "sc[ms]:mean=" << md.sc0tmean << ", max =" << md.sc0tmax << ", min =" << md.sc0tmin << ", 3Sigma=" << md.sc0t3sg << "\n";
	os << "statistics(Overall)\n";
	os << "all[ms]:mean=" << md.altmean << ", max =" << md.altmax << ", min =" << md.altmin << ", 3Sigma=" << md.alt3sg << "\n";
	return os;
}

std::ofstream& operator<<(std::ofstream& os, const statistics& md)
{
	os << "count:Success(0)=" << md.count0 << ", Error(1) =" << md.count1 << ", Matching not completed(2) =" << md.count2 << ", Insufficient inlayer(3) =" << md.count3 << "\n";
	os << "statistics(Success0)\n";
	os << "d[px]:mean=" << md.sc0dmean << ", max =" << md.sc0dmax << ", min =" << md.sc0dmin << ", 3Sigma=" << md.sc0d3sg << "\n";
	os << "dx[px]:mean=" << md.sc0dxmean << ", max =" << md.sc0dxmax << ", min =" << md.sc0dxmin << ", 3Sigma=" << md.sc0dx3sg << "\n";
	os << "dy[px]:mean=" << md.sc0dymean << ", max =" << md.sc0dymax << ", min =" << md.sc0dymin << ", 3Sigma=" << md.sc0dy3sg << "\n";
	os << "height[%]:mean=" << md.sc0hgmean << ", max =" << md.sc0hgmax << ", min =" << md.sc0hgmin << ", 3Sigma=" << md.sc0hg3sg << "\n";
	os << "sc[ms]:mean=" << md.sc0tmean << ", max =" << md.sc0tmax << ", min =" << md.sc0tmin << ", 3Sigma=" << md.sc0t3sg << "\n";
	os << "statistics(Success0+Error1)\n";
	os << "d[px]:mean=" << md.sc01dmean << ", max =" << md.sc01dmax << ", min = " << md.sc01dmin << ", 3Sigma= " << md.sc01d3sg << "\n";
	os << "dx[px]:mean=" << md.sc01dxmean << ", max =" << md.sc01dxmax << ", min =" << md.sc01dxmin << ", 3Sigma=" << md.sc01dx3sg << "\n";
	os << "dy[px]:mean=" << md.sc01dymean << ", max =" << md.sc01dymax << ", min =" << md.sc01dymin << ", 3Sigma=" << md.sc01dy3sg << "\n";
	os << "height[%]:mean=" << md.sc01hgmean << ", max =" << md.sc01hgmax << ", min =" << md.sc01hgmin << ", 3Sigma=" << md.sc01hg3sg << "\n";
	os << "sc[ms]:mean=" << md.sc0tmean << ", max =" << md.sc0tmax << ", min =" << md.sc0tmin << ", 3Sigma=" << md.sc0t3sg << "\n";
	os << "statistics(Overall)\n";
	os << "all[ms]:mean=" << md.altmean << ", max =" << md.altmax << ", min =" << md.altmin << ", 3Sigma=" << md.alt3sg << "\n";
	return os;
}

