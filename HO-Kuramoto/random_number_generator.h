#include <stdlib.h>

//Random number generator (by Giorgio Parisi and Federico Rapuano)
//Parisi, Giorgio, and Federico Rapuano. "Effects of the Random Number Generator on Computer Simulations." Physics Letters B, vol. 157, no. 4, 1985, pp. 301-302, https://doi.org/10.1016/0370-2693(85)90670-7.
#define NormRANu (2.3283063671E-10F)
unsigned int irr[256];
unsigned int ir1;
unsigned char ind_ran,ig1,ig2,ig3;
float Random(void)
{
    float r;
    ig1=ind_ran-24;
    ig2=ind_ran-55;
    ig3=ind_ran-61;
    irr[ind_ran]=irr[ig1]+irr[ig2];
    ir1=(irr[ind_ran]^irr[ig3]);
    ind_ran++;
    r=ir1*NormRANu;
    return r;
}
void ini_ran(int SEMILLA)
{
    int INI,FACTOR,SUM,i;
    srand(SEMILLA);
    INI=SEMILLA;
    FACTOR=67397;
    SUM=7364893;
    for(i=0;i<256;i++){
        INI=(INI*FACTOR+SUM);
        irr[i]=INI;
    }
    ind_ran=ig1=ig2=ig3=0;
}