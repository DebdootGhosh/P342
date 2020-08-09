/*Q3*/
#include <stdio.h>
#include <stdlib.h>

int main()
{   double sumprev=0;
    double sumnext=1;
    double n=2;

    while(sumnext-sumprev>0.001){
        sumprev=sumnext;
        sumnext=sumnext+1/n;
        n=n+1;
    }
    printf("The sum of the series is %lf\n",sumnext);
    return 0;
}
