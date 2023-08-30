Dependencies required: numpy, pandas, matplotlib libraries

Answers to the questions are given below, also provided in the notebook.

**answers to the questions in part 1**
 
a) the relationship is NOT linear
 
b) Yes I do need feature engineering to add non-linearity. For this, I noticed that the plot increases and decreases monotonically as x varies. Also, the curve intersects the Y=0 line five times. This motivates me to model a 5-th order polynomial for this given data, a polynomial that has five real roots (intersecting y=0 line five times) and goes from negative infinity to positive infinity, like any other dominant odd-power graph.

**answers to questions in part 2**
 
1. the average least squares error is the j value we computed in the end, which is 1.022
 
2. From the list of theta parameters, we see that largest three values are 25.78, 25.23, and 20.80, that correspond to ['Bathrooms'] ['Living area'] ['Local Price'] If we only consider ['Bathrooms'], cost functional value is 365.51. We can see the blue lines on the graph for this contribution. It is not sufficient alone.
 
3. ['# Rooms'] has the least contribution with theta = 1.02. We can see that black and red lines overlap on the figure. Without this term, there is almost no change in the prediction.


**answers to questions in part 3**
1. we didn't need any basis function, since weight was applied on linear relationship between y and x.
2. This implementation did have a large error compared to the solution I
provided for the first problem
