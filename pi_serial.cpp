#include <cstdio>
#include <cmath>

using namespace std;

int main()
{
   int const num_steps = 100000;
   double const step = 1.0 / num_steps;

   double sum = 0;

   for (int i = 0; i < num_steps; ++i)
   {
      double xi = (i+0.5)*step;
      double f = 1.0 / (xi*xi+1.0);
      sum += f;
   }

   double result = step * sum;

   printf("integral is %.15g\n", result);
   printf("error is %.10e\n", abs(result - M_PI/4));
}
