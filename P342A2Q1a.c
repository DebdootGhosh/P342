#include <stdio.h>
#include <stdlib.h>

int main()
{
 double dist=0.0;
 double avdist=0.0;
 int count=0;
 int i;
 int j;

for(i=0;i<6;i++){
    for(j=0;j<6;j++){
        dist=dist+abs(j-i);
        count=count+1;} }

printf ("total distance %lf \n",dist);
printf ("total counting %d\n" ,count);

avdist=dist/count;
printf ("average distance between two points %lf",avdist);
return 0;
}
/*total distance 70.000000
total counting 36
average distance between two points 1.944444*/
