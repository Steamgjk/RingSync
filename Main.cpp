#include "MyRing.h"
int main(int argc, char* argv[])
{

	printf("Main Start\n");

	printf("%d %d\n", atoi(argv[1]), atoi(argv[2]) );
	MyRing* mring = new MyRing(atoi(argv[1]), atoi(argv[2]));
	mring->InitBGThread();
	printf("End\n");
}