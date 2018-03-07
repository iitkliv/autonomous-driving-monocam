
#include "vo_features.h"
#include <math.h>

using namespace cv;
using namespace std;

#define MAX_FRAME 6000

const double dt = 0.090;

// Intrinsic parameters
const double focal = 718.8560;
const cv::Point2d pp(607.1928, 185.2157);
// IMP: Change the file directories (4 places) according to where your dataset is saved before running!
const Matx33d K = Matx33d( focal, 0, pp.x,
                           0, focal, pp.y,
                           0, 0, 1);



/* Keyboard callback to control 3D visualization
 */
bool camera_pov = false;
void keyboard_callback(const viz::KeyboardEvent &event, void* cookie)
{
  if (event.action == 0 &&!event.symbol.compare("s") )
    camera_pov = !camera_pov;
}


// Absolute scale finder
double getAbsoluteScale(int frame_id, int sequence_id, double z_cal)
{
  
    string line;
    int i = 0;
    ifstream myfile ("/home/asimo/BTP/data/kitti_dataset/00.txt");
    double x =0, y=0, z = 0;
    double x_prev, y_prev, z_prev;
    if (myfile.is_open())
    {
        while(( getline (myfile,line) ) && (i<=frame_id))
        {
            z_prev = z;
            x_prev = x;
            y_prev = y;
            std::istringstream in(line);
            //cout << line << '\n';
            for (int j=0; j<12; j++)
            {
                in >> z ;
                if (j==7) y=z;
                if (j==3)  x=z;
            }

            i++;
        }
        
        myfile.close();
    }

    else
    {
        cout << "Unable to open file";
        return 0;
    }

    return sqrt((x-x_prev)*(x-x_prev) + (y-y_prev)*(y-y_prev) + (z-z_prev)*(z-z_prev)) ;

}




int main( int argc, char** argv )
{
    // declare matrices for cv code
    Mat prevImage, currImage;
    Mat R_f = (Mat_<double>(3, 3) <<1.0, 0.0, 0.0,
                                    0.0, 1.0, 0.0,
                                    0.0, 0.0, 1.0);
    
    Mat t_f = (Mat_<double>(3, 1) << 0.0, 0.0, 0.0);
    Mat R_gt, t_gt;
    Mat E, R, t, mask;


    // some variables for iterative storage
    float speed;
    float gt_pose[12];
    double scale = 1.00;
    char filename[100];


    // feature detection, tracking datatypes
    vector<KeyPoint> prevFeatures, currFeatures;        //vectors to store the coordinates of the feature points
    vector<Point2f> prevPoints, currPoints;
    Mat prevDescriptors, currDescriptors;

    // pose output file
    ofstream myfile;
    myfile.open("/home/asimo/BTP/data/kitti_dataset/00_nn.txt");

    // time stamp file
    ifstream timefile("/home/asimo/BTP/data/kitti_dataset/00/times.txt");
    // neural network speed output as input to code
    ifstream speedfile("/home/asimo/BTP/data/kitti_dataset/00_speed_nn.txt");
    // ground truth file
    ifstream gtfile("/home/asimo/BTP/data/kitti_dataset/00.txt");


    // define visualizer window and other basic stuff
    viz::Viz3d window_est("Estimation Coordinate Frame");
    window_est.setBackgroundColor(); // black by default
    window_est.registerKeyboardCallback(&keyboard_callback);
    
    // path storing variable
    vector<Affine3d> path_est_gt;
    vector<Affine3d> path_est_nn;
    
    // ground truth frustum and pose axes configuration
    viz::WCameraPosition cpw_gt(1.0); // Coordinate axes
    viz::WCameraPosition cpw_frustum_gt(K, 1.0, viz::Color::yellow());

    // result frustum and pose axes configuration
    viz::WCameraPosition cpw_nn(1.0); // Coordinate axes
    viz::WCameraPosition cpw_frustum_nn(K, 1.0, viz::Color::yellow());

    // world origin axes configuration
    viz::WCameraPosition world_ax(1.0); // Coordinate axes world
    Affine3d world_frame = Affine3d(R_f, t_f);

    Affine3d cam_pose_gt, cam_pose_nn;
    int idx = 0, forw = -1;


    // Add first frame pose
    myfile<<"1.0"<<" "<<"0.0"<<" "<<"0.0"<<" "<<"0.0"<<" ";
    myfile<<"0.0"<<" "<<"1.0"<<" "<<"0.0"<<" "<<"0.0"<<" ";
    myfile<<"0.0"<<" "<<"0.0"<<" "<<"1.0"<<" "<<"0.0"<<endl;

    // read first frame gt
    for(int i=0; i<12; ++i)
        gtfile >> gt_pose[i];

    R_gt = (Mat_<double>(3, 3)<< gt_pose[0], gt_pose[1], gt_pose[2],
                                 gt_pose[4], gt_pose[5], gt_pose[6], 
                                 gt_pose[8], gt_pose[9], gt_pose[10]);
    t_gt = (Mat_<double>(3, 1)<< gt_pose[3], 0.0, gt_pose[11]);

    // push the poses to their respective path storing vector
    path_est_gt.push_back(Affine3d(R_gt, t_gt));
    path_est_nn.push_back(Affine3d(R_f, t_f));

    // forward the index
    forw *= (idx==0) ? -1: 1; idx += forw;



    // first iter
    sprintf(filename, "/home/asimo/BTP/data/kitti_dataset/00/image_2/%06d.png", 0);
    //read the first two frames from the dataset
    prevImage = imread(filename, 0);
    //currImage = imread(filename2, 0);
    if ( !prevImage.data)
    { 
        std::cout<< " --(!) Error reading images " << std::endl; return -1;
    }



    for(int numFrame=1; numFrame < MAX_FRAME; numFrame++)
    {
        // read image
        sprintf(filename, "/home/asimo/BTP/data/kitti_dataset/00/image_2/%06d.png", numFrame);
        currImage = imread(filename, 0);
        
        // detect features and descriptors from prev image
        featureDetection(prevImage, prevFeatures);
        descriptorDetection(prevImage, prevDescriptors, prevFeatures);
        
        // detect features and descriptors from current image 
        featureDetection(currImage, currFeatures);
        descriptorDetection(currImage, currDescriptors, currFeatures);
        
        // track features       
        featureTrackingFlann(prevImage, currImage, prevFeatures, currFeatures, prevDescriptors, currDescriptors);

        // convert keypoints to point
        KeyPoint::convert(prevFeatures, prevPoints, vector<int>());
        KeyPoint::convert(currFeatures, currPoints, vector<int>());

        // Find essential mat
        E = findEssentialMat(currPoints, prevPoints, focal, pp, RANSAC, 0.999, 1.0, mask);
        // Recover R, t
        recoverPose(E, currPoints, prevPoints, R, t, focal, pp, mask);

        // generate scale
        speedfile >> speed;
        float scale_nn = fabs(speed) * dt;
    	scale = getAbsoluteScale(numFrame, 0, t.at<double>(2));

        cout << "Scale is " << scale_nn << " , " << scale << endl;

        // Forward kinematics
        if ((scale_nn> 0.10)&&(t.at<double>(2) > t.at<double>(0)) && (t.at<double>(2) > t.at<double>(1)))
        //if (scale>0.01)
        {

          t_f = t_f + scale_nn*(R_f*t);
          R_f = R*R_f;

        }

        // update image
        prevImage = currImage.clone(); //prevFeatures = currFeatures; prevDescriptors = currDescriptors.clone();


        // update result pose file
        myfile<<R_f.at<double>(0, 0)<<" "<<R_f.at<double>(0, 1)<<" "<<R_f.at<double>(0, 2)<<" "<<t_f.at<double>(0, 0)<<" ";
        myfile<<R_f.at<double>(1, 0)<<" "<<R_f.at<double>(1, 1)<<" "<<R_f.at<double>(1, 2)<<" "<<t_f.at<double>(1, 0)<<" ";
        myfile<<R_f.at<double>(2, 0)<<" "<<R_f.at<double>(2, 1)<<" "<<R_f.at<double>(2, 2)<<" "<<t_f.at<double>(2, 0)<<endl;

        cout<<R.at<double>(0, 0)<<" "<<R.at<double>(0, 1)<<" "<<R.at<double>(0, 2)<<" "<<t.at<double>(0, 0)<<endl;
        cout<<R.at<double>(1, 0)<<" "<<R.at<double>(1, 1)<<" "<<R.at<double>(1, 2)<<" "<<t.at<double>(1, 0)<<endl;
        cout<<R.at<double>(2, 0)<<" "<<R.at<double>(2, 1)<<" "<<R.at<double>(2, 2)<<" "<<t.at<double>(2, 0)<<endl;
        cout << endl;

        // constraint to 2d plane:: x-z plane
        Mat temp_tf = (Mat_<double>(3, 1) << t_f.at<double>(0), 0.0, t_f.at<double>(2));
        path_est_nn.push_back(Affine3d(R_f, temp_tf));


        for (int i=0; i<12; ++i)
            gtfile >> gt_pose[i];

        R_gt = (Mat_<double>(3, 3)<< gt_pose[0], gt_pose[1], gt_pose[2],
                                     gt_pose[4], gt_pose[5], gt_pose[6], 
                                     gt_pose[8], gt_pose[9], gt_pose[10]);
        t_gt = (Mat_<double>(3, 1)<< gt_pose[3], 0.0, gt_pose[11]);

        path_est_gt.push_back(Affine3d(R_gt, t_gt));

        // get current respective camera poses
        cam_pose_nn = path_est_nn[idx]; cam_pose_gt = path_est_gt[idx]; 

        if (camera_pov)
            window_est.setViewerPose(cam_pose_nn);
        else
        {
            // world axes
            window_est.showWidget("world_axes", world_ax, world_frame);

            // ground truth
            window_est.showWidget("cameras_frames_and_lines_gt", 
                                  viz::WTrajectory(path_est_gt, viz::WTrajectory::PATH,
                                  3.0, viz::Color::green()));
            window_est.showWidget("cpw_gt", cpw_gt, cam_pose_gt);
            window_est.showWidget("CPW_FRUSTUM_gt", cpw_frustum_gt, cam_pose_gt);

            // nn output
            window_est.showWidget("cameras_frames_and_lines_nn", 
                                  viz::WTrajectory(path_est_nn, viz::WTrajectory::PATH,
                                  3.0, viz::Color::red()));
            window_est.showWidget("CPW_nn", cpw_nn, cam_pose_nn);
            window_est.showWidget("CPW_FRUSTUM_nn", cpw_frustum_nn, cam_pose_nn);
        }
        
        forw *= (idx==0) ? -1: 1; idx += forw;

        window_est.spinOnce(1, true);
        window_est.removeAllWidgets();
         

        waitKey(1);
    }

    return 0;
}
