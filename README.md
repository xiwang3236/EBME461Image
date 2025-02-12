# EBME461Image

Questions
1. You will recall from class 3 primary similarity metrics: (1) Normalized Cross-Correlation, (2) 
Mutual Information, and (3) Sum of Squared Error (also known as sum of squared differ￾ences). Here you will investigate another, Sum of Absolute Difference. Write a function 
“mySumAbs.m” which computes normalized cross correlation between two input images. 
You will call the function as follows:
R = mySumAbs(image1, image2)
Here R is a measure of how much similarity there is between the two images (R=0 à per￾fectly similar). 
Paste the code for this function in your submission.
2. Contrast2_new.tif is a translated version of Contrast1-new.tif. It is translated by 1.8 pixels
and 2.1 pixels in the x and y directions, respectively. When you do this operation, there will 
be an “overlap region.” Note: in general, for registration, when an image is transformed, it 
will not longer exactly overlap. This will be an issue below. It is not necessary for you to 
compute a similarity metric to the ends of an image. Often one might use a sub-region for 
the calculation. 
a. Calculate mySumAbs with the two input images. 
b. Translate Contrast2-new.tif by the required amounts so that the images are regis￾tered. (This will require image interpolation,) Calculate mySumAbs. Report the val￾ue(s) obtained. This is a QUANTITATIVE method of evaluating registration. Of 
course, to do this, you will need to use an image interpolation approach such as bi￾linear. NOTE: You should use matrix operations versus looping to do this problem. It 
will come in handy for the subsequent problems.
c. Create a subtracted image to ensure that the images are registered. Display your fi￾nal subtracted image. This is a QUALITATIVE way of evaluating registration. 
3. Now, register the same image pair from problem 2 using the “Nelder-Mead simplex minimi￾zation algorithm”, fminsearch. 
a. Use your function, mySumAbs.
b. Use x and y translation only (no rotation, no scaling, no shearing). You will need to 
evaluate a similarity cost function (that is, SSE) and iteratively minimize the cost 
function. Start from a reasonable initialization, say (0,0).
c. The minimization algorithm should iteratively compute the optimum x and y transla￾tion parameters and evaluate the cost function. Display Contrast1_new side by-by￾side with translated Contrast2_new and report the “optimal” translation in x and y
yielded by the algorithm.
d. At each step in the optimization algorithm (when each new Tx and Ty is computed), 
please using MATLAB’s getframe() to iteratively capture the difference image be￾tween the fixed image and the translated image. The first frame should be the differ￾ence image before any translation and the last frame should be the difference image 
after registration has been optimized. You can visualize your frames as a movie us￾ing movie(); but please also export this movie using movie2avi() for ease in grading. 
e. For any evaluation of registration, you should include 2 original images, the sub￾tracted image without registration, and the subtracted image with registration. 
4. Adjust fminsearch parameters. Modify the algorithm in Problem 3 as follows:
a. Use different stopping criteria (the tolerances). Sample a range of values.
b. In addition, implement the scaling parameter described later which solves the prob￾lem of making step sizes in the optimization too small. Sample a range of values.
c. Describe your experiences. I suggest making a small table with the results of regis￾tration as you modified parameters. I’d like to know how these parameters affected 
(1) the number of iterations until completion and (2) whether or not there was a successful registration.
5. Validation of your algorithm with digital subtraction angiography (DSA) images having an 
unknown displacement. Use live-new.tiff (containing contrast) and mask-new.tiff (no contrast). 
a. Use your program to register the images. 
b. As the contrast image contains the vessels of interest, it should not be distorted. It 
will be the reference image. Mask will be the floating image. For display, always subtract the mask_new from live_new, yielding dark arteries, the convention in DSA.
c. For any evaluation of DSA registration, you must include: 2 original images, the subtracted image without registration, and the subtracted image with registration. 
d. What are the optimal translation parameters?
6. The displacement in the DSA image pair (live-new.tif, mask-new.tif) is not entirely captured 
by simple translation, suggesting the need for non-rigid registration. We will Apply the 
MATLAB function imregdemons, which uses the famous demons non-rigid registration algorithm (see references below). Use suggestions above for reference and floating. Present 
your image results. You should use similar window and level for any subtracted images. 
You should be able to get a result similar to that below obtained by the TA. 
Here are some optimizations to try.
1- Iteration array=[5000 400 200], Accumulated Field Smoothing (AFS)= 30
2- Iteration array=[5000 400 200], Accumulated Field Smoothing (AFS)= 3
3- Iteration array=[50 20 5], Accumulated Field Smoothing (AFS)= 30
4- Iteration array=[50 20 5], Accumulated Field Smoothing (AFS)= 3

![image](https://github.com/user-attachments/assets/6caef69f-d94b-4fb9-bccd-3b46a709e5e9)
