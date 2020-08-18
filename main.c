/*Q1B*/
#include <stdio.h>
#include <stdlib.h>

int main()
{
double dist=0.0;
double avdist=0.0;
int count=0;
int i, j, k, l;

for (i=0;i<6;i++){
    for (j=0;j<6;j++){
        for (k=0;k<6;k++){
            for (l=0;l<6;l++){
                dist=dist+abs(k-i)+abs(l-j);
                count+=1;}}}}

printf ("total distance %lf\n",dist);
printf ("total counting %d\n",count);

avdist=dist/count;
printf ("average distance between two points in 6 by 6 2d grid is %lf", avdist);

    return 0;
}
/* total distance 5040.000000
   total counting 1296
   average distance between two points in 6 by 6 2d grid is 3.888889 */
