#include <stdio.h>
#include <stdlib.h>
int main()
{

FILE *ar1,*ar2;
 int i,j,k;
 double N[3][3],M[3][3],res[3][3],ress[3][3];
 double A[]={1.0,1.0,1.0};
 ar1 =fopen("M11.txt","r");
 ar2 =fopen("matriA.txt","r");
 //reading the matrix N from the file
 if(ar1==NULL){
     printf("file not found");
     return 0 ;
 }
 if(ar2==NULL){
     printf("file not found");
     return 0 ;
 }

 for( i=0 ;i<3; i++)
 {
 for( j=0; j<3; j++)
 {
 fscanf(ar1, "%lf",&N[i][j]);

 }
 }
 //reading the matrix M from the file
 for( i=0 ;i<3; i++)
 {
 for( j=0; j<3; j++)
 {
 fscanf(ar2, "%lf",&M[i][j]);

 }
 }
 for(int i=0 ;i<3; i++)
 {printf("\n");
 for(int j=0; j<3; j++)
 {
 printf("%lf", M[i][j]);

 }
 }
 printf("\n");
 for(int i=0 ;i<3; i++)
 {printf("\n");
 for(int j=0; j<3; j++)
 {
 printf("%lf", N[i][j]);

 }
 }
 //matrix multiplication M X N
 for( i=0;i<3;i++)
 {
 for( j=0;j<3;j++)
 {
 res[i][j]=0;
 for( k=0;k<3;k++)
 {
 res[i][j]+=M[i][k]*N[k][j];
 }
 }
 }
 printf("\n");
 //print result
 printf("Matrix multiplication M X N is = \n");
 for( i=0 ;i<3; i++)
 {
 for( j=0; j<3; j++)
 {
 printf("%lf ",res[i][j]);

 }
 printf("\n");
 }
 //matrix multiplication M X A
 for( i=0;i<3;i++)
 {
 for( j=0;j<1;j++)
 {
 ress[i][j]=0;
 for( k=0;k<3;k++)
 {
 ress[i][j]+=M[i][k]*A[k];
 }
 }
 }
 //print result
 printf("Matrix multiplication with vector A, M X A is = \n");
 for( i=0 ;i<3; i++){
    for(j=0;j<1;j++) {
 printf("%lf ",ress[i][j]);

 printf("\n");}}

return 0;}
/* 2.000000 -6.000000 7.000000
   9.000000 -7.000000 0.000000
   0.000000 -4.000000 6.000000

   9.000000 7.000000 8.000000
   0.000000-8.000000 9.000000
   -7.000000 1.000000 0.000000
   Matrix multiplication M X N is =
   31.000000 69.000000  -38.000000
   81.000000 119.000000  9.000000
   -42.000000 38.000000 -36.000000
   Matrix multiplication with vector A, M X A is =
   3.000000
   2.000000
   2.000000*/
