#include <3DHoPD/3DHoPD.h>

int main()
{

    pcl::PointCloud<pcl::PointXYZ> cloud2, cloud1;

    pcl::io::loadPCDFile<pcl::PointXYZ>("../sample_data/scene005_0.pcd", cloud2);          // Scene 1

    //pcl::io::loadPCDFile<pcl::PointXYZ>("../sample_data/PeterRabbit001_0.pcd", cloud1);  // Model 1 for Scene 1
    //pcl::io::loadPCDFile<pcl::PointXYZ>("../sample_data/Doll018_0.pcd", cloud1);         // Model 2 for Scene 1
    //pcl::io::loadPCDFile<pcl::PointXYZ>("../sample_data/mario000_0.pcd", cloud1);        // Model 3 for Scene 1


    //pcl::io::loadPCDFile<pcl::PointXYZ>("../sample_data/Scene0_0.1.pcd", cloud2);        // Scene2
    //pcl::io::loadPCDFile<pcl::PointXYZ>("../sample_data/bun_zipper_0.pcd", cloud1);      // Model for Scene 2

    pcl::io::loadPCDFile<pcl::PointXYZ>("../sample_data/Scene4_0.1.pcd", cloud2);          // Scene 3
    pcl::io::loadPCDFile<pcl::PointXYZ>("../sample_data/happy_vrip_res3_0.pcd", cloud1);   // Model for Scene 3

    //pcl::io::loadPCDFile<pcl::PointXYZ>("../sample_data/Scene6_0.1.pcd", cloud2);        // Scene 4
    //pcl::io::loadPCDFile<pcl::PointXYZ>("../sample_data/bun_zipper_0.pcd", cloud1);      // Model for Scene 4




// For testing robustness to Rotations
    Eigen::Matrix4f A = Eigen::Matrix4f::Identity();

    // Define a rotation matrix (see https://en.wikipedia.org/wiki/Rotation_matrix)
    float theta = M_PI/2; // The angle of rotation in radians
//    cout << theta << endl;
    A (0,0) = cos (theta);
    A (0,1) = -sin(theta);
    A (1,0) = sin (theta);
    A (1,1) = cos (theta);

    transformPointCloud(cloud1,cloud1, A); // This is to apply a rotation on the input point cloud. Can be Commented if not needed


    threeDHoPD RP1, RP2;


    // Using Simple Uniform Keypoint Detection

    RP1.cloud = cloud1;
    RP1.detect_uniform_keypoints_on_cloud(0.01);
    cout << "Keypoints on Model: " << RP1.cloud_keypoints.size() << endl;

    RP2.cloud = cloud2;
    RP2.detect_uniform_keypoints_on_cloud(0.01);
    cout << "Keypoints on Scene: " << RP2.cloud_keypoints.size() << endl;



    clock_t start1, end1;
    double cpu_time_used1;
    start1 = clock();

    // setup
    RP1.kdtree.setInputCloud(RP1.cloud.makeShared());// This is required for SUPER FAST
    RP2.kdtree.setInputCloud(RP2.cloud.makeShared());// THIS IS REQUIRED FOR SUPER FAST

    RP1.JUST_REFERENCE_FRAME_descriptors(0.06);
    RP2.JUST_REFERENCE_FRAME_descriptors(0.06);

    end1 = clock();
    cpu_time_used1 = ((double) (end1 - start1)) / CLOCKS_PER_SEC;
    cout <<  "Time taken for Feature Descriptor Extraction: " << (double)cpu_time_used1 << "\n";


    pcl::Correspondences corrs;

    clock_t start_shot2, end_shot2;
    double cpu_time_used_shot2;
    start_shot2 = clock();

    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree_LRF;
    pcl::PointCloud<pcl::PointXYZ> pcd_LRF;
    for (int i = 0; i < RP2.cloud_LRF_descriptors.size(); i++)
    {
        pcl::PointXYZ point;
        point.x = RP2.cloud_LRF_descriptors[i].vector[0];
        point.y = RP2.cloud_LRF_descriptors[i].vector[1];
        point.z = RP2.cloud_LRF_descriptors[i].vector[2];

        pcd_LRF.push_back(point);
    }



    kdtree_LRF.setInputCloud(pcd_LRF.makeShared());
    for (int i = 0; i < RP1.cloud_LRF_descriptors.size(); i++)
    {
        pcl::PointXYZ searchPoint;
        searchPoint.x = RP1.cloud_LRF_descriptors[i].vector[0];
        searchPoint.y = RP1.cloud_LRF_descriptors[i].vector[1];
        searchPoint.z = RP1.cloud_LRF_descriptors[i].vector[2];

        std::vector<int> nn_indices;
        std::vector<float> nn_sqr_distances;

        std::vector<double> angles_vector;

        if (kdtree_LRF.radiusSearch(searchPoint,0.01,nn_indices,nn_sqr_distances) > 0)// IMPORTANT PARAMETER 0.2m or 0.1m ...
        {
            for (int j = 0; j < nn_indices.size(); j++)
            {
                //if (angle < 0.5)// Important Threshold!!!
                {
                    Eigen::VectorXf vec1, vec2;
                    vec1.resize(15); vec2.resize(15);

                    for (int k = 0; k < 15; k++)
                    {
                        vec1[k] = RP1.cloud_distance_histogram_descriptors[i].vector[k];
                        vec2[k] = RP2.cloud_distance_histogram_descriptors[nn_indices[j]].vector[k];

                    }

                    double dist = (vec1-vec2).norm();
                    angles_vector.push_back(dist);

                }
            }

            //cout << "Second THreshold Potential Matches: " << good_angles_accumulate_indices.size()<< endl;

            std::vector<double>::iterator result;
            result = std::min_element(angles_vector.begin(), angles_vector.end());
            //std::cout << "Max element at: " << std::distance(match_distance.begin(), result) << '\n';
            //std::cout << "Max element is: " << match_distance[std::distance(match_distance.begin(), result)] << '\n';
            int min_element_index = std::distance(angles_vector.begin(), result);

            pcl::Correspondence corr;
            corr.index_query = RP1.patch_descriptor_indices[i];// vulnerable
            corr.index_match = RP2.patch_descriptor_indices[nn_indices[min_element_index]];// vulnerable

            corrs.push_back(corr);


            /*
            std::vector<float> row;
            for (int l = 0; l < angles_vector.size(); l++)
            {
                row.push_back(angles_vector[l]);
            }

            std::sort(row.begin(), row.end());
            float smallest = row[0];
            float smaller = row[1];
            float compare_threshold = (float)smallest/(float)smaller;

            //if ( compare_threshold <= THRESHOLD ) // Threshold
            {

                //cout << "Second THreshold Potential Matches: " << good_angles_accumulate_indices.size()<< endl;

                std::vector<double>::iterator result;
                result = std::min_element(angles_vector.begin(), angles_vector.end());
                //std::cout << "Max element at: " << std::distance(match_distance.begin(), result) << '\n';
                //std::cout << "Max element is: " << match_distance[std::distance(match_distance.begin(), result)] << '\n';
                int min_element_index = std::distance(angles_vector.begin(), result);

                pcl::Correspondence corr;
                corr.index_query = RP1.patch_descriptor_indices[i];// vulnerable
                corr.index_match = RP2.patch_descriptor_indices[nn_indices[min_element_index]];// vulnerable

                corrs.push_back(corr);

            }
        }

        else
        {
            pcl::Correspondence corr;
            corr.index_match = corr.index_query = 0;
            corrs.push_back(corr);
        }

        */

        }
    }


    end_shot2 = clock();
    cpu_time_used_shot2 = ((double) (end_shot2 - start_shot2)) / CLOCKS_PER_SEC;
    //cout <<  "Time taken for Feature Descriptor Matching : " << (double)cpu_time_used_shot2 << "\n";

    //////////////////////////////////////////////////////////////////////////////////////
    /// \brief kdtree_LRF----------JUST REFERENCE FRAME descriptors
    //////////////////////////////////////////////////////////////////////////////////////

    cout << "No. of Reciprocal Correspondences : " << corrs.size() << endl;


    // RANSAC based false matches removal
    pcl::CorrespondencesConstPtr corrs_const_ptr = boost::make_shared< pcl::Correspondences >(corrs);

    pcl::Correspondences corr_shot;
    pcl::registration::CorrespondenceRejectorSampleConsensus< pcl::PointXYZ > Ransac_based_Rejection_shot;
    Ransac_based_Rejection_shot.setInputSource(RP1.cloud_keypoints.makeShared());
    Ransac_based_Rejection_shot.setInputTarget(RP2.cloud_keypoints.makeShared());
    Ransac_based_Rejection_shot.setInlierThreshold(0.01);
    Ransac_based_Rejection_shot.setInputCorrespondences(corrs_const_ptr);
    Ransac_based_Rejection_shot.getCorrespondences(corr_shot);


    cout << "Transformation Matrix : \n" << Ransac_based_Rejection_shot.getBestTransformation()<< endl;

    cout << "True correspondences after RANSAC : " << corr_shot.size() << endl;


    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
    viewer->setBackgroundColor (255, 255, 255);



    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> single_color1(RP1.cloud_keypoints.makeShared(), 255, 0, 0);
    viewer->addPointCloud<pcl::PointXYZ> (RP1.cloud_keypoints.makeShared(), single_color1, "sample cloud1");
    viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "sample cloud1");
    //viewer->addCoordinateSystem (1.0);
    viewer->initCameraParameters ();

    Eigen::Matrix4f t;
    t<<1,0,0,0.4,
            0,1,0,0,
            0,0,1,0,
            0,0,0,1;

    //cloudNext is my target cloud
    pcl::transformPointCloud(RP2.cloud_keypoints,RP2.cloud_keypoints,t);

    //int v2(1);
    //viewer->createViewPort (0.5,0.0,0.1,1.0,1);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> single_color2(RP2.cloud_keypoints.makeShared(), 0, 0, 255);
    viewer->addPointCloud<pcl::PointXYZ> (RP2.cloud_keypoints.makeShared(), single_color2, "sample cloud2");
    viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "sample cloud2");



    //viewer->addCorrespondences<pcl::PointXYZ>(RP1.cloud_keypoints.makeShared(), RP2.cloud_keypoints.makeShared(), corrs /*corr_shot*/, "correspondences"/*,v1*/);
    viewer->addCorrespondences<pcl::PointXYZ>(RP1.cloud_keypoints.makeShared(), RP2.cloud_keypoints.makeShared(), /*corrs*/ corr_shot, "correspondences"/*,v1*/);




    while (!viewer->wasStopped ())
    {
        viewer->spinOnce (100);
        boost::this_thread::sleep (boost::posix_time::microseconds (100000));
    }




    return 0;
}









