#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <math.h>
#include <random>
#include <vector>
#include "ros/ros.h"
#include "ros/package.h"
#include "rosbag/bag.h"
#include "rosbag/view.h"
#include "sensor_msgs/LaserScan.h"
#include "../perception_tools/perception_2d.h"
#include <CImg.h>
#include "shared_structs.h"
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>


using std::string;
using std::vector;

vector<vector<float>> all_templates_;
vector<float> all_orientations_;
vector<float> all_depths_;

//TODO: max laser range needs to be set in a single place and used with a single
//name. Currently, there are two ways of setting / using max range

unsigned int clip_l = 0;
unsigned int clip_u = 0;
//unsigned int clip_l = 28;
//unsigned int clip_u = 29;
float min_range = 0.0;
float max_range = 10.0;
string bag_name;

void initTemplates() {
// 1. Human (Dirk)
/*
2.746 2.722 2.695 1.998 1.846 1.779 1.716 1.69 1.66 1.655 1.66 1.655 1.661
1.66 1.664 1.666 1.667 1.699 1.697 1.719 1.788 1.878 2.019 2.161 2.547 2.582
2.581 2.577 2.578 2.58 2.562 2.552 2.547 2.546 2.499 2.203 1.86 1.774 1.74
1.701 1.69 1.684 1.681 1.68 1.684 1.678 1.67 1.679 1.679 1.661 1.681 1.687 1.758
1.839 1.939 2.132 2.236 2.427
*/

// 2. Curvy Couch
/*
6.266 6.278 10 1.964 1.941 1.898 1.86 1.826 1.794 1.762 1.713 1.696 1.649
1.624 1.604 1.571 1.547 1.522 1.516 1.464 1.44 1.425 1.405 1.392 1.366 1.352
1.338 1.33 1.328 1.324 1.327 1.328 1.325 1.326 1.327 1.327 1.338 1.34 1.338
1.338 1.338 1.341 1.341 1.346 1.359 1.347 1.34 1.353 1.357 1.361 1.361 1.361
1.364 1.379 1.389 1.389 1.382 1.382 1.382 1.383 1.382 1.383 1.391 1.4 1.403
1.403 1.402 1.4 1.398 1.405 1.406 1.41 1.414 1.42 1.423 1.42 1.425 1.427
1.427 1.43 1.432 1.437 1.441 1.444 1.445 1.446 1.447 1.452 1.452 1.459 1.466
1.468 1.466 1.465 1.464 1.465 1.468 1.473 1.478 1.482 1.482 1.49 1.489 1.493
1.496 1.492 1.498 1.501 1.506 1.509 1.509 1.509 1.509 1.519 1.52 1.519 1.522
1.529 1.53 1.531 1.53 1.531 1.537 1.53 1.537 1.545 1.543 1.553 1.565 1.568 1.568
1.576 1.571 1.571 1.574 1.583 1.587 1.588 1.594 1.597 1.597 1.598 1.6 1.603
1.602 1.609 1.609 1.609 1.613 1.603 1.611 1.613 1.622 1.632 1.631 1.637
1.638 1.647 1.636 1.644 1.644 1.649 1.654 1.661 1.663 1.663 1.662 1.662 1.674
1.68 1.682 1.678 1.675 1.678 1.686 1.693 1.694 1.691 1.689 1.701 1.707 1.708
1.708 1.707 1.713 1.713 1.716 1.73 1.735 1.735 1.735 1.737 1.734 1.742 1.742
1.75 1.754 1.754 1.754 1.758 1.764 1.77 1.772 1.772 1.782 1.791 1.791 1.799
1.799 1.797 1.799 1.803 1.808 1.809 1.809 1.81 1.81 1.81 1.816 1.817 1.819
1.826 1.829 1.831 1.838 1.843 1.845 1.845 1.853 1.852 1.858 1.861 1.866 1.87
1.874 1.873 1.869 1.871 1.869 1.869 1.881 1.889 1.9 1.9 1.893 1.895 1.899 1.902
1.903 1.909 1.911 1.917 1.926 1.923 1.923 1.924 1.931 1.936 1.94 1.94 1.946
1.95 1.953 1.955 1.956 1.96 1.961 1.968 1.972 1.98 1.982 1.985 1.998 2.052
*/

// 3. Left Turn
/*
3.944 3.957 3.946 3.928 3.933 3.942 3.943 3.955 3.957 3.96 3.972 3.978 3.979
3.978 3.974 3.981 3.992 3.998 4 4.006 4.019 4.02 4.024 4.029 4.038 4.046
4.053 4.057 4.075 4.076 4.077 4.081 4.108 4.111 4.12 4.12 4.126 4.137 4.146
4.156 4.162 4.177 4.177 4.183 4.198 4.205 4.212 4.229 4.232 4.232 4.245 4.251
4.264 4.281 4.279 4.28 4.304 4.313 4.33 4.337 4.357 4.359 4.363 4.365 4.387
4.394 4.409 4.416 4.425 4.44 4.463 4.463 4.486 4.495 4.512 4.524 4.537 4.538
4.548 4.578 4.594 4.591 4.6 4.618 4.645 4.653 4.662 4.678 4.692 4.709 4.722 4.75
4.756 4.77 4.77 4.794 4.826 4.824 4.835 4.863 4.87 4.891 4.906 4.921 4.94
4.977 4.988 5 5.007 5.032 5.059 5.078 5.107 5.121 5.118 5.143 5.204 5.439
5.481 5.526 5.525 5.567 5.597 5.614 5.624 5.674 5.679 5.723 5.734 5.75 5.8 5.812
5.835 5.853 5.883 10 1.946 1.946 1.952 1.936 1.926 1.916 1.909 1.904 1.9
1.893 1.888 1.884 1.874 1.868 1.858 1.853 1.848 1.837 1.832 1.83 1.83 1.813
1.81 1.812 1.806 1.794 1.788 1.778 1.776 1.772 1.762 1.761 1.761 1.757 1.752
1.744 1.736 1.733 1.717 1.715 1.716 1.715 1.712 1.705 1.691 1.688 1.689
1.693 1.69 1.678 1.671 1.665 1.664 1.662 1.658 1.651 1.643 1.646 1.649 1.641
1.633 1.625 1.622 1.625 1.626 1.622 1.618 1.618 1.608 1.6 1.6 1.602 1.595 1.591
1.588 1.587 1.585 1.583 1.584 1.579 1.575 1.572 1.564 1.56 1.558 1.559 1.556
1.551 1.544 1.552 1.552 1.541 1.54 1.533 1.53 1.53 1.53 1.524 1.52 1.52
1.521 1.524 1.521 1.514 1.514 1.513 1.513 1.505 1.5 1.499 1.494 1.499 1.5 1.493
1.492 1.489 1.487 1.484 1.481 1.482 1.484 1.485 1.484 1.484 1.482 1.477
1.481 1.482 1.481 1.483 1.482 1.476 1.469 1.469 1.465 1.462 1.459 1.456 1.46
1.462 1.465 1.469 1.468 1.466 1.464 1.463 1.463 1.462 1.461 1.462 1.462 1.464
1.462 1.461 1.461 1.452 1.455 1.447 1.449 1.446 1.446 1.445 1.454 1.451
1.452 1.45 1.455 1.454 1.45 1.445 1.448 1.449 1.449 1.447 1.446 1.448 1.452
1.462 1.456 1.455 1.449 1.449 1.448 1.451 1.454 1.459 1.458 1.457 1.457 1.458
1.459 1.453 1.453 1.457 1.461 1.464 1.464 1.467 1.467 1.465 1.464 1.464
1.465 1.471 1.476 1.47 1.469 1.466 1.467 1.472 1.475 1.475 1.482 1.489 1.49
1.479 1.481 1.487 1.487 1.484 1.485 1.485 1.485 1.482 1.485 1.487 1.492 1.494
1.496 1.496 1.503 1.51 1.514 1.514 1.516 1.515 1.515 1.519 1.523 1.523 1.523
1.524 1.533 1.533 1.525 1.527 1.531 1.531 1.537 1.542 1.553 1.562 1.562
1.562
*/

// 4. Right Turn
/*
1.187 1.183 1.183 1.189 1.188 1.188 1.199 1.202 1.203 1.198 1.198 1.2 1.205
1.209 1.208 1.205 1.205 1.21 1.211 1.21 1.21 1.21 1.214 1.221 1.228 1.224
1.224 1.237 1.239 1.239 1.235 1.235 1.238 1.238 1.243 1.247 1.257 1.257 1.257
1.254 1.258 1.265 1.269 1.272 1.272 1.274 1.281 1.281 1.286 1.288 1.291 1.29
1.296 1.298 1.298 1.299 1.301 1.311 1.319 1.32 1.321 1.321 1.324 1.33 1.332
1.341 1.351 1.351 1.351 1.364 1.368 1.364 1.371 1.374 1.381 1.381 1.383
1.387 1.397 1.401 1.402 1.403 1.404 1.408 1.411 1.416 1.414 1.417 1.435 1.439
1.443 1.443 1.455 1.462 1.463 1.468 1.473 1.476 1.478 1.49 1.493 1.504 1.509
1.516 1.532 1.532 1.528 1.534 1.551 1.555 1.557 1.564 1.582 1.585 1.596
1.598 1.599 1.61 1.616 1.626 1.631 1.634 1.642 1.652 1.664 1.673 1.67 1.684
1.686 1.696 1.704 1.707 1.725 1.733 1.734 1.755 1.761 1.776 1.787 1.788
1.792 1.815 1.812 1.813 1.832 1.845 1.854 1.877 1.877 1.881 1.889 1.909 1.92
2 1.928 1.937 1.942 1.957 1.967 1.974 1.989 2.008 2.016 2.038 2.063 2.068 2.067
2.087 2.106 2.12 2.132 2.144 2.159 2.173 2.187 2.192 2.211 2.231 2.262 2.27
2.281 2.299 2.324 2.341 2.354 2.381 2.4 2.416 2.434 2.45 2.469 2.482 2.501
2.527 2.555 2.578 2.586 2.614 2.648 2.668 2.695 2.723 2.748 2.764 2.805 2.824
2.844 2.881 2.913 2.943 2.98 2.993 3.038 3.066 3.089 3.135 3.174 3.198 3.252
3.272 3.315 3.374 3.391 3.421 3.473 3.508 3.569 3.6 3.649 3.743 3.722 3.764
10 6.298 6.298 6.292 6.283 6.277 6.266 6.261 6.245 6.239 6.236 6.232 6.222 6.215
6.203 6.198 6.193 6.184 6.171 6.169 6.167 6.166 6.157 6.15 6.154 6.134 6.129
6.129 6.132 6.132 6.127 6.119 6.117 6.103 6.1 6.093 6.093 6.097 6.089 6.079
6.078 6.081 6.081 6.074 6.074 6.061 6.055 6.055 6.055 6.054 6.04 6.054 6.056
6.043 6.04 6.034 6.03 6.029 6.029 6.034 6.04 6.04 6.035 6.037 6.031 6.033
6.038 6.038
*/

// 5. Elevator
/*
2.407 2.389 2.386 2.386 2.384 2.38 2.378 2.372 2.366 2.366 2.37 2.367 2.354
2.352 2.349 2.339 2.337 2.336 2.324 2.326 2.325 2.322 2.327 2.337 2.429 2.49
2.556 2.61 2.624 2.639 2.638 2.631 2.625 2.625 2.625 2.625 2.619 2.616 2.616
2.615 2.609 2.607 2.607 2.608 2.608 2.602 2.594 2.594 2.591 2.589 2.587 2.587
2.586 2.59 2.586 2.583 2.585 2.585 2.586 2.581 2.577 2.576 2.569 2.565 2.566
2.564 2.565 2.569 2.567 2.558 2.557 2.557 2.557 2.554 2.556 2.553 2.555
2.559 2.559 2.559 2.546 2.547 2.547 2.549 2.55 2.548 2.548 2.549 2.552 2.561
2.557 2.559 2.559 2.56 2.561 2.559 2.559 2.555 2.553 2.553 2.555 2.559 2.561
2.56 2.562 2.559 2.557 2.559 2.563 2.564 2.566 2.578 2.571 2.573 2.575 2.576
2.577 2.581 2.582 2.58 2.58 2.582 2.582 2.582 2.596 2.584 2.584 2.587 2.599 2.59
2.579 2.552 2.479 2.395 2.299 2.285 2.273 2.273 2.275 2.287 2.279 2.284
2.289 2.289 2.288 2.288 2.3 2.3 2.299
*/

// 6. Straight Hall
/*
1.243 1.255 1.257 1.261 1.267 1.265 1.271 1.275 1.275 1.278 1.286 1.288 1.293
1.299 1.307 1.317 1.31 1.316 1.322 1.322 1.329 1.338 1.338 1.344 1.349 1.355
1.358 1.367 1.381 1.382 1.382 1.383 1.386 1.393 1.404 1.423 1.435 1.436
1.437 1.447 1.451 1.458 1.462 1.468 1.472 1.482 1.5 1.509 1.514 1.523 1.533 1.54
1.548 1.561 1.569 1.571 1.582 1.59 1.598 1.607 1.617 1.62 1.626 1.634 1.644
1.667 1.675 1.677 1.679 1.696 1.71 1.715 1.727 1.741 1.751 1.761 1.772 1.784
1.791 1.803 1.816 1.832 1.841 1.845 1.865 1.87 1.889 1.895 1.911 1.924 1.933
1.951 1.972 1.987 2.01 2.019 2.035 2.047 2.06 2.091 2.101 2.108 2.132 2.153
2.178 2.196 2.203 2.239 2.254 2.268 2.284 2.298 2.331 2.343 2.359 2.391
2.417 2.43 2.461 2.467 2.49 2.539 2.554 2.596 2.619 2.635 2.672 2.7 2.727 2.76
2.78 2.813 2.866 2.893 2.928 2.962 2.996 3.028 3.072 3.112 3.149 3.192 3.24
3.277 3.33 3.358 3.408 3.457 3.518 3.564 3.611 3.667 3.732 3.794 3.839 3.916
3.973 4.065 4.117 4.205 4.271 4.353 4.43 4.515 4.608 4.697 4.784 4.889 5.003
5.103 5.202 5.342 5.455 5.542 5.537 5.543 5.547 5.551 5.55 5.573 5.642 5.875
6.088 6.313 10 7.831 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10
10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10
10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10
10 10 10 10 10 10 10 10 10 6.53 6.393 6.25 6.141 6.023 5.948 5.843 5.748
5.647 5.556 5.478 5.376 5.294 5.225 5.145 5.099 10 10 10 10 10 10 10 10 10 10 10
10 10 10 104.142 4.057 4.003 3.973 3.933 3.884 3.845 3.8 3.764 3.723 3.686
3.655 3.618 3.578 3.54 3.5 3.478 3.447 3.408 3.376 3.355 3.315 3.291 3.257
3.226 3.201 3.182 3.146 3.121 3.092 3.07 3.044 3.017 2.997 2.98 2.958 2.937
2.908 2.888 2.868 2.851 2.827 2.802 2.791 2.756 2.745 2.735 2.717 2.694
2.672 2.65 2.644 2.632 2.608 2.6 2.558 2.549 2.529 2.511 2.496 2.491 2.468
2.455 2.446 2.415 2.41 2.403 2.393 2.369 2.368 2.353 2.338 2.318 2.304 2.288
2.276 2.272 2.259 2.251 2.24 2.24 2.228 2.212 2.203 2.19 2.181 2.154 2.155
2.154 2.153 2.131 2.128 2.115 2.106 2.091 2.068 2.065 2.061 2.051 2.045
2.029 2.027 2.018 2.017 2.011 2 1.982 1.974 1.958 1.951 1.949 1.951 1.939 1.93
1.922 1.92 1.907 1.895 1.885 1.884 1.884 1.876 1.873 1.862 1.855 1.847 1.837
1.831 1.829 1.827 1.821 1.812 1.8 1.798 1.797 1.787 1.786 1.784 1.782 1.768
1.761 1.752 1.745 1.745 1.742 1.738 1.728 1.728 1.726 1.724 1.722 1.705 1.696
1.694 1.692 1.691 1.678 1.674 1.68 1.678 1.671 1.655 1.649 1.649 1.651 1.645
*/

// 7. Two Doors
/*
2.289 2.288 2.292 2.285 2.295 2.296 2.296 2.297 2.3 2.292 2.294 2.388 2.394 2.4
2.403 2.402 2.406 2.405 2.403 2.408 2.418 2.418 2.421 2.428 2.426 2.428
2.435 2.438 2.445 2.446 2.454 2.454 2.454 2.462 2.468 2.459 2.467 2.477 2.487 
2.491 2.491 2.491 2.495 2.504 2.499 2.503 2.519 2.535 2.535 2.532 2.532
2.541 2.544 2.54 2.55 2.57 2.575 2.575 2.578 2.581 2.583 2.601 2.598 2.615
2.615 2.61 2.619 2.618 2.635 2.65 2.651 2.651 2.64 2.664 2.672 2.683 2.69 2.692 
2.698 2.718 2.721 2.728 2.73 2.746 2.748 2.746 2.743 2.739 2.738 2.712
2.705 10 2.717 2.716 2.718 2.732 2.74 2.751 2.752 2.756 2.764 2.781 2.782
2.801 2.847 10 2.934 2.948 2.956 2.972 2.979 2.984 2.994 3.004 3.017 3.029
3.038 3.036 3.065 3.076 3.09 3.1 3.116 3.113 3.144 3.152 3.171 3.178 3.192 3.211
3.221 3.229 3.239 3.258 3.27 3.289 3.307 3.323 3.34 3.347 3.368 3.38 3.393
3.404 3.417 3.422 3.466 3.474 3.484 3.483 3.469 3.488 3.47 3.5 10 3.479
3.513 3.513 3.531 3.545 3.567 3.59 3.608 3.636 3.648 3.665 3.699 3.713 3.736
3.752 3.776 3.797
*/

// 8. Fire Door
/*
3.744 3.791 3.841 3.891 3.948 4.002 4.022 4.021 4.023 4.016 4.012 4.012 4.013
4.007 3.998 3.986 3.986 3.983 3.984 3.974 3.974 3.972 3.972 3.945 3.948
3.952 3.969 4.02 4.052 4.071 4.099 4.121 4.123 4.121 10 9.899 9.898 9.891 9.88 
9.865 9.857 9.851 9.849 9.842 9.838 9.877 9.933 10 10 10 10 10 10 10 10 10 10
10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10
10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 9.358 10
10 10 8.191 8.038 10 4.081 4.076 4.066 4.06 4.046 3.969 3.945 3.907 3.918 3.915
3.912 3.912 3.93 3.938 3.937 3.936 3.936 3.945 3.949 3.948 3.948 3.948 3.955
3.955 3.963 3.964 3.972 3.958 3.912 3.834 3.768 3.725 3.672
*/

// 9. Large Convex Corner
/*
5.679 5.65 5.587 5.537 5.497 5.452 5.385 5.345 5.297 5.255 5.206 5.165 5.132
5.078 5.036 5.009 4.957 4.924 4.877 4.849 4.805 4.776 4.736 4.702 4.669
4.631 4.597 4.563 4.522 4.493 4.475 4.452 4.413 4.383 4.352 4.322 4.295 4.273
4.246 4.22 4.192 4.163 4.148 4.116 4.081 4.062 4.038 4.015 3.995 3.983 3.964
3.947 3.918 3.904 3.861 3.839 3.823 3.793 3.781 3.764 3.735 3.715 3.699 3.69
3.665 3.644 3.627 3.606 3.599 3.57 3.566 3.539 3.529 3.503 3.493 3.477 3.459
3.445 3.432 3.409 3.393 3.388 3.374 3.366 3.348 3.329 3.317 3.302 3.285 3.269
3.257 3.238 3.23 3.216 3.2 3.188 3.184 3.177 3.164 3.154 3.139 3.133 3.113
3.106 3.091 3.085 3.067 3.051 3.05 3.042 3.029 3.018 3.016 3.008 2.979 2.975
2.978 2.961 2.947 2.941 2.927 2.927 2.909 2.904 2.897 2.895 2.877 2.875 2.87
2.854 2.852 2.845 2.833 2.821 2.818 2.81 2.807 2.797 2.794 2.789 2.778 2.775
2.772 2.772 2.758 2.746 2.74 2.734 2.719 2.723 2.719 2.712 2.712 2.708 2.692
2.689 2.691 2.706 2.747 2.751 2.776 2.805 2.828 2.86 2.887 2.909 2.943 2.968
2.998 3.04 3.073 3.099 3.141 3.161 3.195 3.222 3.262 3.306 3.345 3.38 3.427
3.467 3.503 3.537 3.587 3.633 3.69 3.74 3.787 3.843 3.895 3.954 3.995 4.059
4.102 4.17 4.237 4.308 4.374 4.446 4.517 4.603 4.667 4.755 4.82 4.916 5.005
5.111 5.206 5.295
*/

// 10. Large Concave Corner
/*
3.608 3.61 3.62 3.62 3.625 3.631 3.645 3.653 3.645 3.649 3.646 3.652 3.647 3.654
3.661 3.66 3.656 3.674 3.682 3.681 3.681 3.68 3.675 3.688 3.706 3.711 3.712
3.713 3.72 3.715 3.716 3.717 3.715 3.74 3.74 3.746 3.746 3.744 3.753 3.759
3.775 3.781 3.781 3.788 3.799 3.802 3.813 3.814 3.812 3.814 3.82 3.836 3.845
3.851 3.854 3.869 3.871 3.875 3.878 3.89 3.902 3.901 3.915 3.913 3.932 3.927
3.94 3.951 3.964 3.965 3.967 3.972 3.979 3.994 4.007 4.016 4.022 4.025 4.043
4.05 4.06 4.079 4.081 4.094 4.098 4.111 4.126 4.134 4.133 4.15 4.154 4.16 4.176
4.184 4.193 4.203 4.221 4.235 4.244 4.25 4.251 4.285 4.296 4.316 4.319 4.335
4.353 4.361 4.37 4.377 4.367 4.34 4.313 4.284 4.262 4.254 4.215 4.196 4.179
4.152 4.129 4.1 4.076 4.057 4.033 4.016 3.982 3.972 3.952 3.929 3.92 3.894 3.874
3.834 3.794 3.758 3.734 3.723 3.713 3.714 3.686 3.669 3.653 3.638 3.618
3.603 3.586 3.567 3.556 3.544 3.531 3.519 3.504 3.496 3.472 3.464 3.44 3.425
3.419 3.401 3.386 3.374 3.365 3.354 3.344 3.337 3.323 3.303 3.3 3.282 3.274
3.256 3.239 3.24 3.23 3.224 3.216 3.198 3.185 3.177 3.168 3.161 3.142 3.136
3.128 3.125 3.119 3.09 3.089 3.081 3.077 3.064 3.053 3.043 3.039 3.03 3.025
3.009 3.006 2.999 2.99 2.981 2.976 2.965 2.962 2.953 2.939 2.924 2.92 2.915
2.898 2.896 2.887 2.887
*/

// 11. 
/*

*/

// 12. 
/*

*/

// 13. 
/*

*/

// 14. 
/*

*/

// 15. 
/*

*/

// 16. 
/*

*/

// 17. 
/*

*/

// 18. 
/*

*/

// 19. 
/*

*/

// 20. 
/*

*/

  std::string line;
  std::ifstream infile("../../../../src/bag_to_image/templates.txt");

  while (std::getline(infile, line)) {
    std::istringstream iss(line);
    float value;
    vector<float> single_template;
    while (iss >> value) {
      single_template.push_back(value);
    }
    std::cout << "push back" << single_template.size() << std::endl;
    all_templates_.push_back(single_template);
  }
}

void initDepthsAndOrientations() {
  all_depths_.push_back(1.0);
  all_depths_.push_back(5.0);
  all_depths_.push_back(9.0);
  all_orientations_.push_back(0.0);
  all_orientations_.push_back(90.0);
  all_orientations_.push_back(-90.0);
}


bool epsilonClose(const float eps, const float n1, const float n2) {
  if (fabs(n1 - n2) <= eps) {
    return true;
  }
  return false;
}

float medianFilter(const vector<float> to_be_filtered) {
  if (to_be_filtered.size() % 2) {
    return to_be_filtered[to_be_filtered.size() / 2];
  }
  else {
    return (to_be_filtered[(to_be_filtered.size() / 2)] +
            to_be_filtered[(to_be_filtered.size() / 2) - 1]) / 2.0;
  }
}

float solveQuadratic(float a, float b, float c) {
  float det = b * b - 4 * a * c;
  if (det < 0.0) {
    return 0.0;
  }
  float t1 = ((-1.0) * b + sqrt(det)) / (2.0 * a);
  float t2 = ((-1.0) * b - sqrt(det)) / (2.0 * a);
  t1 = std::max(float(0.0), t1);
  t2 = std::max(float(0.0), t2);
  return std::min(t1, t2);
}

float magDot2D(float v1, float v2, float w1, float w2) {
  return fabs(v1*w1 + v2*w2);
}

void getScansFromBag(const float laser_max_range, vector<vector<float>>* all_scans) {
  rosbag::Bag bag;
  printf("Opening bag file %s...", bag_name.c_str()); fflush(stdout);
  bag.open(bag_name, rosbag::bagmode::Read);
  printf(" Done.\n"); fflush(stdout);
  float eps = 0.02;

  vector<string> topics;
  topics.push_back("/Cobot/Laser");
  topics.push_back("Cobot/Laser");
  topics.push_back("/laser");
  topics.push_back("laser");
  
  printf("Reading bag file..."); fflush(stdout);
  rosbag::View view(bag, rosbag::TopicQuery(topics));
  for (rosbag::View::iterator it = view.begin(); it != view.end(); ++it) {
    const rosbag::MessageInstance &message = *it;
    sensor_msgs::LaserScanPtr laser_message = message.instantiate<sensor_msgs::LaserScan>();
    //const bool is_cobot_laser = message.getTopic() == kCobotLaserTopic;
    //if (laser_message != NULL && is_cobot_laser) {
    if (laser_message != NULL) {
      vector<float> single_scan;
      for (unsigned int i = clip_l; i < laser_message->ranges.size() - clip_u; ++i) {
        float range = laser_message->ranges[i];
        range = std::max(range, min_range);
        range = std::min(range, laser_max_range);
        if (epsilonClose(eps, range, laser_max_range)) {
          //range = 0.0; NOTE: I think this breaks some nice embedding space
          //properties... should test / compare!
          range = max_range;
        }
        single_scan.push_back(range);
      }
      all_scans->push_back(single_scan);
    }
  }
  //printf(" Done.\n"); fflush(stdout);
  //printf("%d scans loaded.\n", static_cast<int>(all_scans->size()));
}

vector<ScanFeatureMetaData> getFeaturesFromText(string filename) {
  std::string line;
  std::ifstream infile(filename);

  vector<ScanFeatureMetaData> all_features;

  while (std::getline(infile, line)) {
    std::istringstream iss(line);
    ScanFeatureMetaData scan_feature;

    //bag name, scan number, type, start angle, end angle, range[i], ..., range[j]

    
    string bag_name;
    iss >> bag_name;
    int scan_number;
    iss >> scan_number;
    int feature_type;
    iss >> feature_type;
    float start_angle;
    iss >> start_angle;
    float end_angle;
    iss >> end_angle;

    scan_feature.start_angle = start_angle;
    scan_feature.end_angle = end_angle;
    scan_feature.type = feature_type;

    float value;
    vector<float> ranges;
    while (iss >> value) {
      ranges.push_back(value);
    }

    scan_feature.ranges = ranges;

    std::cout << bag_name << " " << scan_number << " " << feature_type << " " 
              << start_angle << " " << end_angle << std::endl;
    // have scan feature, now generate many

    all_features.push_back(scan_feature);
  }
  return all_features;
}


/*
void convertScansToImages(const vector<vector<float>>& all_scans) {
  std::stringstream ss;
  ss.str(bag_name);
  string item;
  vector<string> elems;
  while (getline(ss, item, '/')) {
    elems.push_back(item);
  }
  std::stringstream ss2;
  ss2.str(elems.back());
  elems.clear();
  string item2;
  while (getline(ss2, item2, '.')) {
    elems.push_back(item2);
  }
  string bag_date = elems[0];
  std::cout << "bag_date: " << bag_date << std::endl;

  size_t downsampling_rate = 4;

  size_t width = 256;
  size_t height = 256;
  cimg_library::CImg<uint8_t> scan_image(width, height);
  size_t scan_number = 1;

  printf("Downsampling scans..."); fflush(stdout);
  for (size_t i = 0; i < all_scans.size(); ++i) {
    vector<float> single_scan = all_scans[i];
    vector<float> downsampled_scan;
    for (size_t j = 0; j < single_scan.size(); j += downsampling_rate) {
      vector<float> scan_seg;
      for (size_t k = j; k < j + downsampling_rate; ++k) {
        scan_seg.push_back(single_scan[k]);
      }
      std::sort(scan_seg.begin(), scan_seg.end());
      float median = (scan_seg[1] + scan_seg[2]) / 2.0;
      downsampled_scan.push_back(median);
    }

    for (size_t y = 0; y < height; ++y) {
      for (size_t x = 0; x < downsampled_scan.size(); ++x) {
        scan_image((x + y) % height, y) = size_t(255 - (downsampled_scan[x]/max_range)*255);
      }
    }
    string scan_image_file = bag_date + "_" + std::to_string(scan_number) + ".png";
    scan_image.save_png(scan_image_file.c_str());
    scan_number++;

  }
  printf(" Done.\n"); fflush(stdout);
}
*/

void applyTruncate(const float new_max_range, vector<vector<float>>* all_scans) {
  for (size_t i = 0; i < all_scans->size(); ++i) {
    for (size_t j = 0; j < all_scans[0][i].size(); ++j) {
      if ((*all_scans)[i][j] >= new_max_range) { // truncate ==> set max range+ to 0
        (*all_scans)[i][j] = 0.0;
      }
    }
  }
}

void applySeasoning(const float amount, vector<vector<float>>* all_scans) {
  for (size_t i = 0; i < all_scans->size(); ++i) {
    for (size_t j = 0; j < all_scans[0][i].size(); ++j) {
      float r = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
      if (r <= amount) {
        const float seasoned_obs = static_cast <float> (rand()) / (static_cast <float> (RAND_MAX/max_range));
        all_scans[0][i][j] = seasoned_obs;
      }
    }
  }
}

void applyBlur(const float haze, vector<vector<float>>* all_scans) {
  std::default_random_engine generator;
  std::normal_distribution<float> distribution(0.0, haze);
  for (size_t i = 0; i < all_scans->size(); ++i) {
    for (size_t j = 0; j < all_scans[0][i].size(); ++j) {
      float r = distribution(generator);
      float blurred_obs = all_scans[0][i][j] + r;
      blurred_obs = fmin(blurred_obs, max_range);
      blurred_obs = fmax(blurred_obs, min_range);
      all_scans[0][i][j] = blurred_obs;
    }
  }
}

void applyHumans(const int FOV, vector<vector<float>>* all_scans) {
  const float min_human_dist = 1.0;
  const float FOV_rad = (float(FOV)/180.0) * 2 * M_PI;
  const int num_obs = (*all_scans)[0].size();
  std::default_random_engine generator;
  for (size_t i = 0; i < all_scans->size(); ++i) {
    vector<float>* single_scan = &(*all_scans)[i];
    //const float d = ((static_cast <float> (rand()) / static_cast <float> (RAND_MAX)) / 10.0) + 0.05; // 0.05 - 0.15 meters
    const float d = ((static_cast <float> (rand()) / static_cast <float> (RAND_MAX)) / 10.0) + 0.15; // 0.15 - 0.30 meters
    const float leg_radius = d / 2.0;
    //const float s = ((static_cast <float> (rand()) / static_cast <float> (RAND_MAX)) / 4.0) + 0.05; // 0.05 - 0.30 meters
    const float s = ((static_cast <float> (rand()) / static_cast <float> (RAND_MAX)) / 5.0) + 0.10; // 0.10 - 0.30 meters
    const float conservative_theta_T = 2*atan2(d, min_human_dist) + atan2(s, min_human_dist);
    const float phi = (-0.5*FOV_rad + conservative_theta_T)
                    + (static_cast <float> (rand()) / static_cast <float>(RAND_MAX)) *
                      (FOV_rad - 2*conservative_theta_T);
    const int r_ub_idx = int(((phi + 0.5*FOV_rad) / FOV_rad) * float(num_obs));
    const float r_ub = std::max(min_human_dist, (*all_scans)[i][r_ub_idx]);
    float r = 1.0;
    if (r_ub > r) {
      r = ((static_cast <float> (rand()) / static_cast <float> (RAND_MAX)) *
          (r_ub - min_human_dist))
        + min_human_dist;
    }
    const float theta_d = atan2(d, r);
    const float theta_s = atan2(s, r);
    const float theta_T = 2 * atan2(d, r) + atan2(s, r);
    const int to_replace = (num_obs / FOV_rad) * theta_d;
    const int start_idx1 = num_obs/2.0 + (phi - (theta_T / 2.0)) * (num_obs / FOV_rad);
    const int start_idx2 = num_obs/2.0 + (phi + (theta_s / 2.0)) * (num_obs / FOV_rad);
    const int stop_idx1 = start_idx1 + to_replace;
    const int stop_idx2 = start_idx2 + to_replace;
    const float L1x = r * cos(phi - (theta_d/2.0) - (theta_s/2.0));
    const float L1y = r * sin(phi - (theta_d/2.0) - (theta_s/2.0));
    const float L2x = r * cos(phi + (theta_d/2.0) + (theta_s/2.0));
    const float L2y = r * sin(phi + (theta_d/2.0) + (theta_s/2.0));
    for (int j = start_idx1; j <= stop_idx1; ++j) { // leg 1
      float theta_r = -0.5*FOV_rad + FOV_rad * (float(j) / float(num_obs));
      float a = 1.0; // a = sin^2 + cos^2 = 1
      float b = -2.0 * ((cos(theta_r) * L1x) + (sin(theta_r) * L1y));
      float c = L1x*L1x + L1y*L1y - leg_radius*leg_radius;
      float new_range_gt = solveQuadratic(a, b, c);
      if (new_range_gt < min_human_dist) {
        (*single_scan)[j] = new_range_gt;
      }
      else {
        float v1 = cos(theta_r);
        float v2 = sin(theta_r);
        float w1 = r*cos(theta_r) - L1x;
        float w2 = r*sin(theta_r) - L1y;
        float dot = magDot2D(v1, v2, w1, w2);
        //float mean = 0.05 / pow((dot + 1), 10);
        float mean = 0.0;
        float variance = 0.01 / pow((dot + 1), 12);
        std::normal_distribution<float> distribution(mean, variance);
        float noise = distribution(generator);
        float noised_range = new_range_gt + noise;
        (*single_scan)[j] = std::min(max_range, noised_range);
      }
    }
    for (int j = start_idx2; j <= stop_idx2; ++j) { // leg 2
      float theta_r = -0.5*FOV_rad + FOV_rad * (float(j) / float(num_obs));
      float a = 1.0; // a = sin^2 + cos^2 = 1
      float b = -2.0 * ((cos(theta_r) * L2x) + (sin(theta_r) * L2y));
      float c = L2x*L2x + L2y*L2y - leg_radius*leg_radius;
      float new_range_gt = solveQuadratic(a, b, c);
      if (new_range_gt < min_human_dist) {
        (*single_scan)[j] = new_range_gt;
      }
      else {
        float v1 = cos(theta_r);
        float v2 = sin(theta_r);
        float w1 = r*cos(theta_r) - L2x;
        float w2 = r*sin(theta_r) - L2y;
        float dot = magDot2D(v1, v2, w1, w2);
        //float mean = 0.05 / pow((dot + 1), 10);
        float mean = 0.0;
        float variance = 0.01 / pow((dot + 1), 12);
        std::normal_distribution<float> distribution(mean, variance);
        float noise = distribution(generator);
        float noised_range = new_range_gt + noise;
        (*single_scan)[j] = std::min(max_range, noised_range);
      }
    }
  }
}

void writeScans(vector<vector<float>>* all_scans) {
  std::ofstream outfile;
  //outfile.open("FeatureScans.txt", std::ios_base::app);
  //outfile.open("CorruptionQuality.txt", std::ios_base::app);
  //outfile.open("RawScans.txt", std::ios_base::app);
  outfile.open("RawSynthScans.txt", std::ios_base::app);
  for (size_t i = 0; i < all_scans->size(); ++i) {
    //std::cout << "scan size: " << all_scans[0][i].size() << std::endl;
    for (size_t j = 0; j < all_scans[0][i].size(); ++j) {
      outfile << all_scans[0][i][j] << " ";
    }
    outfile << "\n";
  }
}

void applyCorruption(const float FOV,
                     const bool truncate,
                     const bool seasoning,
                     const bool blur,
                     const bool humans,
                     vector<vector<float>>* all_scans) {
  //writeScans(all_scans);
  if (humans) {
    applyHumans(FOV, all_scans);
  }
  if (blur) {
    float haze = 0.1; // meters
    applyBlur(haze, all_scans);
  }
  if (seasoning) {
    float amount = 0.05;
    applySeasoning(amount, all_scans);
  }
  if (truncate) {
    float new_max_range = 4.0;
    applyTruncate(new_max_range, all_scans);
  }
  //writeScans(all_scans);
}





void insertSyntheticTemplate(const float FOV,
                             const int depth_idx,
                             const int orientation_idx,
                             const int template_type_idx,
                             const vector<vector<float>> orig_scans,
                             vector<vector<float>>* all_scans) {

  //TODO: possibly pseudo random depths / orientations

  const float depth = all_depths_[depth_idx];
  const float orientation = all_orientations_[orientation_idx];
  vector<float> current_template = all_templates_[template_type_idx];

  // Compute indices
  const int insert_size = current_template.size();
  const float scan_start_angle = -FOV / 2.0;
  const size_t numrays = orig_scans[0].size();
  const float angular_res = FOV / float(numrays);

  int insert_center_index = (orientation - scan_start_angle) / angular_res;
  int insert_start_index = insert_center_index - (insert_size / 2);
  int insert_end_index = insert_start_index + insert_size;

  if (insert_start_index < 0) {
    insert_start_index = 0;
    insert_end_index = insert_size - 1;
  }
  else if (insert_end_index > (int(numrays) - 1)) {
    insert_start_index = numrays - insert_size - 1;
    insert_end_index = numrays - 1;
  }

  // Compute depth
  float min_insert_depth = max_range;
  float max_insert_depth = min_range;
  for (size_t i = 0; i < current_template.size(); ++i) {
    min_insert_depth = std::fmin(current_template[i], min_insert_depth);
    if (current_template[i] < 0.98 * max_range) {
      max_insert_depth = std::fmax(current_template[i], max_insert_depth);
    }
  }
  const float insert_mid = (max_insert_depth + min_insert_depth) / 2.0;
  //const float insert_range = max_insert_depth - min_insert_depth;
  float offset = depth - insert_mid;
  if (max_insert_depth + offset > max_range) {
    offset = max_range - max_insert_depth;
  }
  else if (min_insert_depth + offset < min_range) {
    offset = min_insert_depth - min_range;
  }

  //TODO: haven't figured out depth scaling yet (in absense of geometry for ray casting)
  //for (size_t i = 0; i < current_template.size(); ++i) {
  //  current_template[i] = current_template[i] + offset;
  //}

  // Insert template into scan
  for (size_t i = 0; i < orig_scans.size(); ++i) {
    vector<float> synth_scan = orig_scans[i];
    for (int j = 0; j < int(numrays); ++j) { // each ray
      if (j >= insert_start_index && j <= insert_end_index) {
        synth_scan[j] = current_template[j - insert_start_index];
      }
    }
    all_scans->push_back(synth_scan);
  }
}


//TODO: procedures for adding noise
void convertScansToHighResImages(const vector<vector<float>>& all_scans, const int FOV, const float max_laser_range) {
  std::stringstream ss;
  ss.str(bag_name);
  string item;
  vector<string> elems;
  while (getline(ss, item, '/')) {
    elems.push_back(item);
  }
  std::stringstream ss2;
  ss2.str(elems.back());
  elems.clear();
  string item2;
  while (getline(ss2, item2, '.')) {
    elems.push_back(item2);
  }
  string bag_date = elems[0];
  if (bag_date == "txt") {
    //bag_date = "FeatureOnly";
    bag_date = "MCFeature";
  }
  //std::cout << "bag_date: " << bag_date << std::endl;
  std::cout << bag_date << ", " << all_scans.size() << std::endl;

  int num_obs = all_scans[0].size(); //how many points each scan has
  float angular_res = ((M_PI/180.0) * float(FOV)) / float(num_obs);
  float start_angle = (-3.0 * M_PI / 4.0);

  //size_t width = 2048;
  //size_t height = 2048;
  //size_t width = 1024;
  //size_t height = 1024;
  //size_t width = 512;
  //size_t height = 512;
  size_t width = 256;
  size_t height = 256;
//  size_t origin_x = 1023;
//  size_t origin_y = 1023;
//  float pixel_res = 0.01; //1 cm
//  size_t origin_x = 511;
//  size_t origin_y = 511;
//  float pixel_res = 0.02; //2 cm
//  size_t origin_x = 255;
//  size_t origin_y = 255;
//  float pixel_res = 0.04; //4 cm
  size_t origin_x = 127;
  size_t origin_y = 127;
  float pixel_res = 0.08; //8 cm
  cimg_library::CImg<uint8_t> scan_image(width, height);

//  std::ofstream outfile;


  size_t scan_number = 1;
  for (size_t k = 0; k < all_scans.size(); ++k) {
    for (size_t x = 0; x < width; ++x) {
      for (size_t y = 0; y < height; ++y) {
        scan_image(x, y) = 0;
      }
    }
    for (size_t i = 0; i < size_t(num_obs); ++i) {
      if (all_scans[k][i] < 0.98 * max_laser_range) {
        float current_angle = start_angle + i * angular_res;
        Eigen::Vector2f ray_dir(cos(current_angle), sin(current_angle));
        Eigen::Vector2f obs_loc = all_scans[k][i] * ray_dir;
        scan_image(int(obs_loc.x() / pixel_res) + origin_x, int(obs_loc.y() / pixel_res) + origin_y) = 255;
      }
    }
    string scan_image_file = bag_date + "_" + std::to_string(scan_number) + ".png";
    scan_image.save_png(scan_image_file.c_str());
    scan_number++;
  }
  //printf(" Done.\n"); fflush(stdout);
}




void convertScansToImagesPadAndFixedBounds(const vector<vector<float>>& all_scans, const int FOV) {
  std::stringstream ss;
  ss.str(bag_name);
  string item;
  vector<string> elems;
  while (getline(ss, item, '/')) {
    elems.push_back(item);
  }
  std::stringstream ss2;
  ss2.str(elems.back());
  elems.clear();
  string item2;
  while (getline(ss2, item2, '.')) {
    elems.push_back(item2);
  }
  string bag_date = elems[0];
  if (bag_date == "txt") {
    //bag_date = "FeatureOnly";
    bag_date = "MCFeature";
  }
  //std::cout << "bag_date: " << bag_date << std::endl;
  std::cout << bag_date << ", " << all_scans.size() << std::endl;

  int num_obs = all_scans[0].size(); //how many points each scan has
  int padding = int((float(num_obs) / float(FOV)) * (360 - FOV));
  if (padding % 2) {
    padding--;
  }
  int length = num_obs + padding;
  int extra = length % 256;
  int deletion_interval = length / (extra + 1);

  //printf("Padding scans..."); fflush(stdout);
  vector<vector<float>> all_padded_scans;
  for (size_t i = 0; i < all_scans.size(); ++i) {
    vector<float> single_scan = all_scans[i];
    vector<float> padded_scan;
    for (int j = 0; j < (padding / 2); ++j) {
      padded_scan.push_back(0.0);
    }
    for (size_t j = 0; j < single_scan.size(); ++j) {
      padded_scan.push_back(single_scan[j]);
    }
    for (int j = 0; j < (padding / 2); ++j) {
      padded_scan.push_back(0.0);
    }
    // delete entries to make length k * 256 s.t. k is an int
    for (int j = extra; j > 0; j--) {
      padded_scan.erase(padded_scan.begin() + (j * deletion_interval));
    }
    all_padded_scans.push_back(padded_scan);
  }

  //printf("mod 256... should be 0: %d", int(all_padded_scans[0].size() % 256));
  //printf("size of scan: %d", int(all_padded_scans[0].size()));
  size_t downsampling_rate = all_padded_scans[0].size() / 256;
  //printf("downsample rate: %d", int(downsampling_rate));

  size_t width = 256;
  size_t height = 256;
  cimg_library::CImg<uint8_t> scan_image_rot(width, height);
  cimg_library::CImg<uint8_t> scan_image_norm(width, height);
  size_t scan_number = 1;

  std::ofstream outfile;
  //outfile.open("downsampledscans.txt", std::ios_base::app);
  //printf("Downsampling scans..."); fflush(stdout);
  for (size_t i = 0; i < all_padded_scans.size(); ++i) {
    vector<float> padded_scan = all_padded_scans[i];
    vector<float> downsampled_scan;
    for (size_t j = 0; j < padded_scan.size(); j += downsampling_rate) {
      vector<float> scan_seg;
      for (size_t k = j; k < j + downsampling_rate; ++k) {
        scan_seg.push_back(padded_scan[k]);
      }
      std::sort(scan_seg.begin(), scan_seg.end());
      float median = medianFilter(scan_seg);
      downsampled_scan.push_back(median);
    }

    for (size_t y = 0; y < height; ++y) {
      //outfile << size_t(255 - (downsampled_scan[y]/max_range)*255);
      for (size_t x = 0; x < downsampled_scan.size(); ++x) {
        scan_image_rot((x + y) % width, y) = size_t(255 - (downsampled_scan[x]/max_range)*255);
        scan_image_norm(x, y) = size_t(255 - (downsampled_scan[x]/max_range)*255);
      }
    }
    //outfile << "\n";
    string rot_scan_image_file = bag_date + "_" + std::to_string(scan_number) + "_rot.png";
    string norm_scan_image_file = bag_date + "_" + std::to_string(scan_number) + "_norm.png";
    scan_image_rot.save_png(rot_scan_image_file.c_str());
    scan_image_norm.save_png(norm_scan_image_file.c_str());
    scan_number++;
  }
  //printf(" Done.\n"); fflush(stdout);
}

int main(int argc, char* argv[]) {
  if (argc < 4) {
    printf("Usage: ./exec <bag file> <laser FoV (deg)> <max laser range (m)> (opt)<feature file>");
    fflush(stdout);
    exit(1);
  }
  bag_name = argv[1];
  int FOV = atoi(argv[2]); // Field of View of the laser
  float laser_max_range = atof(argv[3]);
  vector<vector<float>> all_scans;
  if (bag_name == "synth") {
    bag_name = argv[4];
    getScansFromBag(laser_max_range, &all_scans);
    initTemplates();
    initDepthsAndOrientations();
    vector<vector<float>> orig_scans = all_scans;
    all_scans.clear();

    //TODO: need to factor in scaling of template sizes for changing depths
    //for (size_t depth = 0; depth < all_depths_.size(); ++depth) {
      for (size_t orientation = 0; orientation < all_orientations_.size(); ++orientation) {
        for (size_t template_type = 0; template_type < all_templates_.size(); ++template_type) {
          insertSyntheticTemplate(FOV, 0, orientation, template_type, orig_scans, &all_scans);
//          insertSyntheticTemplate(FOV, 0, 0, 0, orig_scans, &all_scans);
        }
      }
    //}
    writeScans(&all_scans);
  }
  else {
    bool truncate = false;
    bool seasoning = false; // Add salt-and-pepper-type noise
    bool blur = false;
    bool humans = false;
    //printf(argv[1]); printf("\n"); fflush(stdout);
    //printf("%d \n", FOV); fflush(stdout);
    //printf("%f \n", laser_max_range); fflush(stdout);

    getScansFromBag(laser_max_range, &all_scans);
    //NOTE: for baseline testing
    ////writeScans(&all_scans);

    srand(static_cast <unsigned> (time(0)));
    applyCorruption(FOV, truncate, seasoning, blur, humans, &all_scans);


    ////convertScansToImages(all_scans);
    ////writeScans(&all_scans);
  }
  std::cout << "size: " << all_scans.size() << std::endl;
  //writeScans(&all_scans);
  //convertScansToImagesPadAndFixedBounds(all_scans, FOV);
  convertScansToHighResImages(all_scans, FOV, laser_max_range);
  return 0;
}
