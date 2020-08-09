/*Q2*/
#include <stdio.h>
#include <stdlib.h>

int main()
{   int i,n;
    unsigned long long int fact=1;
    printf("enter a value of n: ");
    scanf("%d",&n);
    if(n<0)
        printf("factorial of negative number does not exist.");
    else if (n==0)
        printf("factorial of 0 is 1.");
    else{
        for(i=1;i<=n;i++)
            fact=fact*i;
        printf("Factorial of %d = %llu ", n,fact);
    }
    return 0;
}
