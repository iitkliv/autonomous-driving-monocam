#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"


#include <iostream>
#include <ctype.h>
#include <algorithm> // for copy
#include <iterator> // for ostream_iterator
#include <vector>
#include <ctime>
#include <sstream>
#include <fstream>
#include <string>
#include <math.h>

#include "ros/ros.h"
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include "std_msgs/String.h"
#include "geometry_msgs/TwistStamped.h"
#include "sensor_msgs/CameraInfo.h"
#include "sensor_msgs/Image.h"
#include "geometry_msgs/Pose.h"
#include "geometry_msgs/PoseWithCovariance.h"
#include "tf2_msgs/TFMessage.h"
#include <tf/tf.h>

using namespace cv;
using namespace std;

#define MAX_FRAME 10000
#define MIN_NUM_FEAT 2000

Mat img, img_prev;
Mat R_f = (Mat_<double>(3, 3) << 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);

Mat t_f ;//= cv::Mat::zeros(3, 0, CV_64F); //= (Mat_<double>(1, 3) << 0.0, 0.0, 0.0);

vector<Point2f> prevFeatures;
double gt_pose[12];

ofstream myfile("/home/asimo/BTP/data/kitti_dataset/00_nn.txt");

ifstream speedfile("/home/asimo/BTP/data/kitti_dataset/00_speed_nn.txt");
ifstream gtfile("/home/asimo/BTP/data/kitti_dataset/00.txt");



void getQuaternion(Mat R, double Q[])
{
    double trace = R.at<double>(0,0) + R.at<double>(1,1) + R.at<double>(2,2);
 
    if (trace > 0.0) 
    {
        double s = sqrt(trace + 1.0);
        Q[3] = (s * 0.5);
        s = 0.5 / s;
        Q[0] = ((R.at<double>(2,1) - R.at<double>(1,2)) * s);
        Q[1] = ((R.at<double>(0,2) - R.at<double>(2,0)) * s);
        Q[2] = ((R.at<double>(1,0) - R.at<double>(0,1)) * s);
    } 
    
    else 
    {
        int i = R.at<double>(0,0) < R.at<double>(1,1) ? (R.at<double>(1,1) < R.at<double>(2,2) ? 2 : 1) : (R.at<double>(0,0) < R.at<double>(2,2) ? 2 : 0); 
        int j = (i + 1) % 3;  
        int k = (i + 2) % 3;

        double s = sqrt(R.at<double>(i, i) - R.at<double>(j,j) - R.at<double>(k,k) + 1.0);
        Q[i] = s * 0.5;
        s = 0.5 / s;

        Q[3] = (R.at<double>(k,j) - R.at<double>(j,k)) * s;
        Q[j] = (R.at<double>(j,i) + R.at<double>(i,j)) * s;
        Q[k] = (R.at<double>(k,i) + R.at<double>(i,k)) * s;
    }
}


geometry_msgs::PoseWithCovariance createPoseMsg(Mat R, Mat t)
{
    double Q[4];
    getQuaternion(R, Q);

    geometry_msgs::PoseWithCovariance msg;
    
    msg.pose.position.x = t.at<double>(0);
    msg.pose.position.y = t.at<double>(1);
    msg.pose.position.z = t.at<double>(2);    

    msg.pose.orientation.x = Q[0];
    msg.pose.orientation.y = Q[1];
    msg.pose.orientation.z = Q[2];
    msg.pose.orientation.w = Q[3];

    msg.covariance[0] = 1e-9;
    msg.covariance[7] = 1e-9;
    msg.covariance[14] = 1e-9;
    msg.covariance[21] = 1e-9;
    msg.covariance[28] = 1e-9;
    msg.covariance[35] = 1e-9;

    return msg;

}


geometry_msgs::PoseWithCovariance createPoseMsgArray(double pose[])
{
    double Q[4];
    Mat R = (Mat_<double>(3, 3)<< pose[0], pose[1], pose[2], pose[4], pose[5], pose[6], pose[8], pose[9], pose[10]);
    getQuaternion(R, Q);

    geometry_msgs::PoseWithCovariance msg;
    
    msg.pose.position.x = pose[3];
    msg.pose.position.y = pose[7];
    msg.pose.position.z = pose[11];    

    msg.pose.orientation.x = Q[0];
    msg.pose.orientation.y = Q[1];
    msg.pose.orientation.z = Q[2];
    msg.pose.orientation.w = Q[3];

    msg.covariance[0] = 1e-9;
    msg.covariance[7] = 1e-9;
    msg.covariance[14] = 1e-9;
    msg.covariance[21] = 1e-9;
    msg.covariance[28] = 1e-9;
    msg.covariance[35] = 1e-9;

    return msg;

}


void featureTracking(Mat img_1, Mat img_2, vector<Point2f>& points1, vector<Point2f>& points2, vector<uchar>& status)
{ 

//this function automatically gets rid of points for which tracking fails

    vector<float> err;                    
    Size winSize=Size(21,21);                                                                                             
    TermCriteria termcrit=TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 30, 0.01);

    calcOpticalFlowPyrLK(img_1, img_2, points1, points2, status, err, winSize, 3, termcrit, 0, 0.001);

    //getting rid of points for which the KLT tracking failed or those who have gone outside the frame
    int indexCorrection = 0;
    for(int i=0; i<status.size(); i++)
        {  

        Point2f pt = points2.at(i-indexCorrection);
        if ((status.at(i) == 0)||(pt.x<0)||(pt.y<0))
        {
            if((pt.x<0)||(pt.y<0))
            {
                status.at(i) = 0;
            }
        points1.erase (points1.begin() + (i - indexCorrection));
        points2.erase (points2.begin() + (i - indexCorrection));
        indexCorrection++;
        }

     }

}


void featureDetection(Mat img_1, vector<Point2f>& points1)
{   //uses FAST as of now, modify parameters as necessary
    vector<KeyPoint> keypoints_1;
    int fast_threshold = 20;
    bool nonmaxSuppression = true;
    FAST(img_1, keypoints_1, fast_threshold, nonmaxSuppression);
    KeyPoint::convert(keypoints_1, points1, vector<int>());
}


// Currently uses KLT for tracking instead of finding new feature points in new frame with flann tracker.
vector<Point2f> outputPose(Mat currImage, Mat prevImage, vector<Point2f> prevFeatures)
{

    Mat E, R, t, mask;
    vector<Point2f> currFeatures;

    float speed, dt = 0.1;

    // Use CameraInfo manager to get 
    double focal = 718.8560;
    cv::Point2d pp(607.1928, 185.2157);

    // cvtColor(currImage, currImage, COLOR_BGR2GRAY);
    
    vector<uchar> status;
    featureTracking(prevImage, currImage, prevFeatures, currFeatures, status);

    E = findEssentialMat(currFeatures, prevFeatures, focal, pp, RANSAC, 0.999, 1.0, mask);
    recoverPose(E, currFeatures, prevFeatures, R, t, focal, pp, mask);

    Mat prevPts(2,prevFeatures.size(), CV_64F), currPts(2,currFeatures.size(), CV_64F);


    for(int i=0;i<prevFeatures.size();i++)	
    {   
        prevPts.at<double>(0,i) = prevFeatures.at(i).x;
        prevPts.at<double>(1,i) = prevFeatures.at(i).y;

        currPts.at<double>(0,i) = currFeatures.at(i).x;
        currPts.at<double>(1,i) = currFeatures.at(i).y;
    }

    speedfile >> speed;
    double scale_nn = fabs(speed) * dt;
	//double scale = getAbsoluteScale(numFrame, 0, t.at<double>(2));
    //cout << "Scale is " << scale_nn << " , " << scale << endl;

    // resolve at low scale value. Heuristic
    t_f = t_f + scale_nn * (R_f * t);
    R_f = R * R_f;


    myfile<<R_f.at<double>(0, 0)<<" "<<R_f.at<double>(0, 1)<<" "<<R_f.at<double>(0, 2)<<" "<<t_f.at<double>(0)<<" ";
    myfile<<R_f.at<double>(1, 0)<<" "<<R_f.at<double>(1, 1)<<" "<<R_f.at<double>(1, 2)<<" "<<t_f.at<double>(1)<<" ";
    myfile<<R_f.at<double>(2, 0)<<" "<<R_f.at<double>(2, 1)<<" "<<R_f.at<double>(2, 2)<<" "<<t_f.at<double>(2)<<endl;

	if (prevFeatures.size() < MIN_NUM_FEAT)	
    {
  		featureDetection(prevImage, prevFeatures);
  		featureTracking(prevImage, currImage, prevFeatures, currFeatures, status);
	}

    //prevImage = currImage.clone();
    prevFeatures = currFeatures;
    
    // ground truth
    for (int i=0; i<12; ++i)
         gtfile >> gt_pose[i];

    //getQuaternion(R_f, Q_f);

    return prevFeatures;
         
}


// void imageCallback(const sensor_msgs::ImageConstPtr& msg)
// {
//   try
//   {
//     img = cv_bridge::toCvShare(msg, "bgr8")->image.clone();

//     cvtColor(img, img, COLOR_BGR2GRAY);
    
//     if (!img_prev.empty())
//     {
//         flag = 1;
//         //prevFeatures = outputPose(img, img_prev, prevFeatures);
//     }
//     else
//     {
//         featureDetection(img, prevFeatures);
//         myfile<<"1.0"<<" "<<"0.0"<<" "<<"0.0"<<" "<<"0.0"<<" "<<"0.0"<<" "<<"1.0"<<" "<<"0.0"<<" "<<"0.0"<<" "<<"0.0"<<" "<<"0.0"<<" "<<"1.0"<<" "<<"0.0"<<endl;
//         img_prev = img.clone();
//     }

//   }

//   catch (cv_bridge::Exception& e)
//   {
//     ROS_ERROR("Could not convert from '%s' to 'bgr8'.", msg->encoding.c_str());
//   }
// }


class SubscribeAndPublish
{

    private:
        ros::NodeHandle n;
        ros::Publisher pose_pub;
        ros::Publisher gt_pub;
        ros::Subscriber sub;
        geometry_msgs::PoseWithCovariance r_msg, gt_msg;

    public:
        SubscribeAndPublish()
        {
            //Topic you want to publish
            pose_pub = n.advertise<geometry_msgs::PoseWithCovariance>("kitti/vo_odometry", 100);
            gt_pub = n.advertise<geometry_msgs::PoseWithCovariance>("kitti/gt_odometry", 100);        
            sub = n.subscribe("/kitti/camera_color_left/image_raw", 1000, &SubscribeAndPublish::callback, this);
        }

        void callback(const sensor_msgs::ImageConstPtr& msg)
        {
            //PUBLISHED_MESSAGE_TYPE output;

            try
            {
                img = cv_bridge::toCvShare(msg, "bgr8")->image.clone();

                cvtColor(img, img, COLOR_BGR2GRAY);

                if (!img_prev.empty())
                {
                    prevFeatures = outputPose(img, img_prev, prevFeatures);
                    r_msg = createPoseMsg(R_f, t_f);
                    gt_msg = createPoseMsgArray(gt_pose);
                    pose_pub.publish(r_msg);
                    gt_pub.publish(gt_msg);

                }
                else
                {
                    featureDetection(img, prevFeatures);
                    myfile<<"1.0"<<" "<<"0.0"<<" "<<"0.0"<<" "<<"0.0"<<" "<<"0.0"<<" "<<"1.0"<<" "<<"0.0"<<" "<<"0.0"<<" "<<"0.0"<<" "<<"0.0"<<" "<<"1.0"<<" "<<"0.0"<<endl;
                    
                    for (int i=0; i<12; ++i)
                        gtfile >> gt_pose[i];

                }

                img_prev = img.clone();

            }

            catch (cv_bridge::Exception& e)
            {
                ROS_ERROR("Could not convert from '%s' to 'bgr8'.", msg->encoding.c_str());
            }
        }

};//End of class SubscribeAndPublish



int main( int argc, char** argv )
{

    // Initialize with identity

    // geometry_msgs::PoseWithCovariance r_msg, gt_msg;

    ros::init(argc, argv, "mono_vo");
    // ros::NodeHandle n;
    // ros::Publisher pose_pub = n.advertise<geometry_msgs::PoseWithCovariance>("kitti/vo_odometry", 100);
    // ros::Publisher gt_pub = n.advertise<geometry_msgs::PoseWithCovariance>("kitti/gt_odometry", 100);
    
    // ros::Subscriber sub = n.subscribe("/kitti/camera_color_left/image_raw", 1000, imageCallback);

    // ros::Rate loop_rate(20);

    // while(ros::ok())
    // {

    //     if (flag == 1)
    //     {
    //         prevFeatures = outputPose(img, img_prev, prevFeatures);
    //         img_prev = img.clone();

    //         r_msg = createPoseMsg(R_f, t_f);
    //         gt_msg = createPoseMsgArray(gt_pose);
    //         pose_pub.publish(r_msg);
    //         gt_pub.publish(gt_msg);
            
    //         //uncomment when to synchronize. Increase loop rate also.
    //         //flag = 0;
    //     }

    //     ros::spinOnce();
    //     loop_rate.sleep();
    // }

    SubscribeAndPublish MonoVo;
    ros::spin();

    return 0;
}



