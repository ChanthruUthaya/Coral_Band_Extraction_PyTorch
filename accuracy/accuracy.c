#include <math.h>
#include <stdlib.h>

double find_distance(int x1, int y1, int x2, int y2) {
  return sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2));
}

int euclidean_single(int* whites, int length, int* whites2, int length2, int thresh) {
  length  /= 2;
  length2 /= 2;
  double distances[length];

  int counts = 0;

  for (int i = 0; i < length; i++) {
    distances[i] = 1000.f;
    for (int j = 0; j < length2; j++) {
      double distance = find_distance(whites[i * 2], whites[i * 2 + 1], 
                                     whites2[j * 2], whites2[j * 2 + 1]);
      if (distance < distances[i]) distances[i] = distance;
    }
  }
  for(int i = 0; i < length; i++){
    
    if(distances[i] < thresh){
      counts += 1;
    }

  }

  return counts;
}


double one_value_euclidean(int x, int y, int*label_boundaries, int length){

  double max_distance = 1000.f;
  
  for(int i = 0; i < length/2; i++){

    double distance = find_distance(x, y, label_boundaries[i*2], label_boundaries[i*2+1]);

    if(distance < max_distance) max_distance = distance;

  }

  return max_distance;

}

double euclidean(int* whites, int length, int* whites2, int length2, int thresh) {
  int one = euclidean_single(whites, length, whites2, length2, thresh);
  int two = euclidean_single(whites2, length2, whites, length, thresh);


  double acc = (double)(one+two)/(double)((length/2)+(length2/2));
  return acc;
}

int main() {
  return 0;
}