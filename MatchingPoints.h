/*
 *	MatchingPoints.h
 *
 *	Description:
 *		Records of matching points
 *
 *
 *
 * 	History:
 *	 	Author			Date			Modify Reason
 *		----------------------------------------------------------------
 *		Chi-Yi Tsai		2012/09/15		File Creation
 *
 *
 */

// Matching points (X,Y) <-> (x,y)
#ifndef __HOMOGRAPHY_INCLUDED__
#define __HOMOGRAPHY_INCLUDED__

#include "fMatrix.h"

extern Float g_Ref_X[];
extern Float g_x1[];
extern Float g_x2[];
extern Float g_x3[];
extern Float g_x4[];
extern Float g_x5[];
extern int g_nNumPoints;

fMatrix findHomography(Float *reference, Float *destination, int num_points);

#endif  //__HOMOGRAPHY_INCLUDED__