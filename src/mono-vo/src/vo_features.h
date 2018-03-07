#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/core/version.hpp"
#include "opencv2/viz.hpp"

#include <iostream>
#include <list>
#include <ctype.h>
#include <algorithm> // for copy
#include <iterator> // for ostream_iterator
#include <vector>
#include <ctime>
#include <sstream>
#include <fstream>
#include <string>

using namespace cv;
using namespace std;
using namespace cv::xfeatures2d;

const double nn_match_ratio = 0.8f;

void featureDetection(Mat img_1, vector<KeyPoint>& points1)
{ 
  // int fast_threshold = 20;
  // bool nonmaxSuppression = true;
  Ptr<AKAZE> detector = AKAZE::create();
  detector->detect(img_1, points1, 3e-4); 
}

void descriptorDetection(Mat img_1, Mat& descriptors1, vector<KeyPoint>& points1)
{
    Ptr<AKAZE> extractor = AKAZE::create();
    extractor->compute(img_1, points1, descriptors1);
}


bool compare_dist(DMatch first, DMatch second)
{
  if (first.distance < second.distance) return true;
  else return false;
}

void featureTrackingFlann(Mat img_1, Mat img_2, vector<KeyPoint>& points1, vector<KeyPoint>& points2, Mat& descriptors1, Mat& descriptors2)
{

    std::vector< vector<DMatch> > matches; // change to <DMatch> when not using knn
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
    //matcher->match(descriptors1, descriptors2, matches);
    
    matcher->knnMatch(descriptors1, descriptors2, matches, 2); // 2 nearest neighbor. Use above line when not knn
    vector<KeyPoint> tempPoints1; vector<KeyPoint> tempPoints2;
    
    vector<DMatch> good_matches;

    for(unsigned i = 0; i < matches.size(); i++) 
    {
        if(matches[i][0].distance < nn_match_ratio * matches[i][1].distance) 
        {
            tempPoints1.push_back(points1[matches[i][0].queryIdx]);
            tempPoints2.push_back(points2[matches[i][0].trainIdx]);
            good_matches.push_back(matches[i][0]);
        }
    }

    // If 50 best points needed hook or crook. use DMatch instead of vector<DMatch>
    // sort(matches.begin(), matches.end(), compare_dist);

    // for (int i=0; i<50 && i<matches.size(); i++)
    // {
    //   good_matches.push_back(matches[i]);
    //   tempPoints1.push_back(points1[matches[i].queryIdx]);
    //   tempPoints2.push_back(points2[matches[i].trainIdx]);
    // }

    Mat img_matches;
    drawMatches(img_1, points1, img_2, points2,
               good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
               vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    imshow("Good matches", img_matches);
    cout << "Total features: " << matches.size() << "  ";
    cout << "First image: " << tempPoints1.size() << "  "; 
    cout << "Second image: " << tempPoints2.size() << endl;

    points1 = tempPoints1; points2 = tempPoints2;

}