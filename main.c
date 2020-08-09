/*Q1*/
#include <stdio.h>
#include <stdlib.h>

int main()

{ int i,n,sum;

    printf("enter the value of n: ");
    scanf("%d",&n);
    sum=0;

 if(n<=0)
    printf("enter a positive whole number");
 else{

    for(i=1;i<=n;i++)
        sum= sum+ i;


    printf("Sum of the series is: %d\n",sum);}

    return 0;
}

