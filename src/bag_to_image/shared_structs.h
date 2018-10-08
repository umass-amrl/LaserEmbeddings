#ifndef COMMON_EMBEDDING_DATA_H
#define COMMON_EMBEDDING_DATA_H

//TODO: enum?
struct ScanFeatureMetaData {
  std::vector<float> ranges;
  float start_angle;
  float end_angle;
  int type; // human = 0, door = 1, corner = 2
};

#endif
