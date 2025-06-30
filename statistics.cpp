#include "classes.h"

void statistics::add_stat(const analysis_results& md, const simulation_data& si)
{
	switch (md.status) {
	case 0:count0++; break;
	case 1:count1++; break;
	case 2:count2++; break;
	default:count3++; break;
	}
	if (md.status == 0) {
		sc0dmean += si.errD;
		if (sc0dmax < si.errD) sc0dmax = si.errD;
		if (sc0dmin > si.errD) sc0dmin = si.errD;
		sc0d3sg += si.errD * si.errD;

		sc0dxmean += si.errDx;
		if (sc0dxmax < si.errDx) sc0dxmax = si.errDx;
		if (sc0dxmin > si.errDx) sc0dxmin = si.errDx;
		sc0dx3sg += si.errDx * si.errDx;

		sc0dymean += si.errDy;
		if (sc0dymax < si.errDy) sc0dymax = si.errDy;
		if (sc0dymin > si.errDy) sc0dymin = si.errDy;
		sc0dy3sg += si.errDy * si.errDy;

		sc0hgmean += si.errHeightP;
		if (sc0hgmax < si.errHeightP) sc0hgmax = si.errHeightP;
		if (sc0hgmin > si.errHeightP) sc0hgmin = si.errHeightP;
		sc0hg3sg += si.errHeightP * si.errHeightP;

		sc0tmean += (double)md.elapsedTime;
		if (sc0tmax < (double)md.elapsedTime) sc0tmax = (double)md.elapsedTime;
		if (sc0tmin > (double)md.elapsedTime) sc0tmin = (double)md.elapsedTime;
		sc0t3sg += (double)md.elapsedTime * (double)md.elapsedTime;
	}
	if (md.status == 0 || md.status == 1) {
		sc01dmean += si.errD;
		if (sc01dmax < si.errD) sc01dmax = si.errD;
		if (sc01dmin > si.errD) sc01dmin = si.errD;
		sc01d3sg += si.errD * si.errD;

		sc01dxmean += si.errDx;
		if (sc01dxmax < si.errDx) sc01dxmax = si.errDx;
		if (sc01dxmin > si.errDx) sc01dxmin = si.errDx;
		sc01dx3sg += si.errDx * si.errDx;

		sc01dymean += si.errDy;
		if (sc01dymax < si.errDy) sc01dymax = si.errDy;
		if (sc01dymin > si.errDy) sc01dymin = si.errDy;
		sc01dy3sg += si.errDy * si.errDy;

		sc01hgmean += si.errHeightP;
		if (sc01hgmax < si.errHeightP) sc01hgmax = si.errHeightP;
		if (sc01hgmin > si.errHeightP) sc01hgmin = si.errHeightP;
		sc01hg3sg += si.errHeightP * si.errHeightP;

		sc01tmean += (double)md.elapsedTime;
		if (sc01tmax < (double)md.elapsedTime) sc01tmax = (double)md.elapsedTime;
		if (sc01tmin > (double)md.elapsedTime) sc01tmin = (double)md.elapsedTime;
		sc01t3sg += (double)md.elapsedTime * (double)md.elapsedTime;
	}

	altmean += (double)md.elapsedTime;
	if (altmax < (double)md.elapsedTime) altmax = (double)md.elapsedTime;
	if (altmin > (double)md.elapsedTime) altmin = (double)md.elapsedTime;
	alt3sg += (double)md.elapsedTime * (double)md.elapsedTime;
}

void statistics::cal_stat(void)
{
	double c0 = (double)count0, c01 = (double)(count0 + count1), ca = (double)(count0 + count1 + count2 + count3);
	sc0dmean /= c0;  sc0d3sg = 3.0 * sqrt(sc0d3sg / c0 - sc0dmean * sc0dmean);
	sc0dxmean /= c0; sc0dx3sg = 3.0 * sqrt(sc0dx3sg / c0 - sc0dxmean * sc0dxmean);
	sc0dymean /= c0; sc0dy3sg = 3.0 * sqrt(sc0dy3sg / c0 - sc0dymean * sc0dymean);
	sc0hgmean /= c0; sc0hg3sg = 3.0 * sqrt(sc0hg3sg / c0 - sc0hgmean * sc0hgmean);
	sc0tmean /= c0;  sc0t3sg = 3.0 * sqrt(sc0t3sg / c0 - sc0tmean * sc0tmean);

	sc01dmean /= c01;  sc01d3sg = 3.0 * sqrt(sc01d3sg / c01 - sc01dmean * sc01dmean);
	sc01dxmean /= c01; sc01dx3sg = 3.0 * sqrt(sc01dx3sg / c01 - sc01dxmean * sc01dxmean);
	sc01dymean /= c01; sc01dy3sg = 3.0 * sqrt(sc01dy3sg / c01 - sc01dymean * sc01dymean);
	sc01hgmean /= c01; sc01hg3sg = 3.0 * sqrt(sc01hg3sg / c01 - sc01hgmean * sc01hgmean);
	sc01tmean /= c01;  sc01t3sg = 3.0 * sqrt(sc01t3sg / c01 - sc01tmean * sc01tmean);

	altmean /= ca; alt3sg = 3.0 * sqrt(alt3sg / ca - altmean * altmean);
}

