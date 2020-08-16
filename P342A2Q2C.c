/*Q2*/
#include <stdio.h>
#include <stdlib.h>
void main() {
 int i;
 double r;
 double add[3];
 double A[3]= {1.9, 2.8, -3}, B[3]= {-4, 5.5, -6.7};;

 for (i = 0, r = 0.0; i < 3; i ++) {
  r += A[i] * B[i];
 }
printf("dot product is: %lf\n", r);
for (i = 0; i < 3; i++)
    add[i] = A[i] + B[i];

  /* print addition vector C */
  printf ("Addition vector:");

  for (i = 0; i < 3; i++)
    printf ("%lf ", add[i]);
}
/* dot product is: 27.900000*/
/*Addition vector: -2.100000 8.3000000 -9.700000*/
